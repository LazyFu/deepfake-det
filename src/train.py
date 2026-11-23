import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DeepfakeDataset
from src.engine import evaluate
from src.metrics import compute_metrics, threshold_at_best_f1, threshold_at_fpr
from src.model import DualStream


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def class_weights_from_csv(path):
    """按类别频率的倒数给权重，缓解真假 1:2 不平衡导致的偏置。"""
    labels = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    w = len(labels) / (2.0 * np.maximum(counts, 1.0))
    print(f"类别权重: real={w[0]:.3f} fake={w[1]:.3f} (n={len(labels)})")
    return torch.tensor(w, dtype=torch.float32)


def build_loaders(cfg, train_csv, val_csv):
    size = cfg["data"]["image_size"]
    workers = cfg["data"]["num_workers"]
    bs = cfg["train"]["batch_size"]
    degrade_prob = float(cfg["train"].get("degrade_prob", 0.5))

    train_ds = DeepfakeDataset(
        train_csv, size=size, train=True, degrade_prob=degrade_prob
    )
    val_ds = DeepfakeDataset(val_csv, size=size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--name", default="run")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rgb-only", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--consistency", type=float, default=None)
    ap.add_argument("--class-weight", action="store_true", help="启用类别加权（cfg 里也开了）")
    ap.add_argument("--train-csv", default="data/splits/train.csv")
    ap.add_argument("--val-csv", default="data/splits/val.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.consistency is not None:
        cfg["train"]["consistency"] = args.consistency
    if args.rgb_only:
        cfg["model"]["use_freq"] = False
        cfg["model"]["use_gate"] = False
    if args.no_gate:
        cfg["model"]["use_gate"] = False

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(cfg["train"]["out_dir"]) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    train_loader, val_loader = build_loaders(cfg, args.train_csv, args.val_csv)
    print(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    model = DualStream(
        pretrained=cfg["model"]["pretrained"],
        use_freq=cfg["model"]["use_freq"],
        use_gate=cfg["model"]["use_gate"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    consistency = float(cfg["train"]["consistency"])
    patience = int(cfg["train"]["patience"])

    use_cw = bool(cfg["train"].get("class_weight", False)) or args.class_weight
    cls_w = class_weights_from_csv(args.train_csv).to(device) if use_cw else None

    # 按 AUC 选模型，避免阈值依赖
    best_score, best_epoch, bad = -1.0, -1, 0
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            ctx = torch.autocast("cuda") if use_amp else nullcontext()
            with ctx:
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=cls_w)
                if consistency > 0:
                    x2 = x + 0.05 * torch.randn_like(x)
                    logits2 = model(x2)
                    loss = loss + consistency * F.kl_div(
                        F.log_softmax(logits2, dim=1),
                        F.softmax(logits.detach(), dim=1),
                        reduction="batchmean",
                    )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            total += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{total / max(n, 1):.4f}")

        sched.step()

        labels, probs = evaluate(model, val_loader, device)
        thr = threshold_at_fpr(labels, probs, 0.01)
        m = compute_metrics(labels, probs, thr)
        m["epoch"] = epoch
        m["train_loss"] = total / max(n, 1)
        m["seconds"] = round(time.time() - t0, 1)
        history.append(m)

        print(
            f"[val] epoch {epoch} auc={m.get('auc', 0):.4f} f1@fpr1={m['f1']:.4f} "
            f"acc={m['acc']:.4f} fpr={m['fpr']:.4f} thr={thr:.4f}"
        )

        score = float(m.get("auc", m["f1"]))
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                 "val_auc": m.get("auc"), "val_f1": m["f1"], "threshold": thr},
                out_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}")
                break

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    labels, probs = evaluate(model, val_loader, device)
    thr = threshold_at_fpr(labels, probs, 0.01)
    thr_f1 = threshold_at_best_f1(labels, probs)
    ckpt["threshold"] = thr          # 满足 FPR<=1% 的阈值（部署默认）
    ckpt["threshold_f1"] = thr_f1    # 验证集 F1 最大的阈值（报告对照）
    torch.save(ckpt, out_dir / "best.pt")

    result = compute_metrics(labels, probs, thr)
    result["best_epoch"] = best_epoch
    result["best_val_auc"] = best_score
    result["threshold_f1"] = thr_f1
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
