import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DeepfakeDataset
from src.engine import evaluate
from src.metrics import compute_metrics
from src.model import DualStream


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def print_metrics(tag, m):
    print(
        f"[{tag:<10s}] n={m['n']:<5d} thr={m['threshold']:.4f} f1={m['f1']:.4f} "
        f"acc={m['acc']:.4f} auc={m.get('auc', 0):.4f} "
        f"fpr={m['fpr']:.4f} recall={m['recall']:.4f} prec={m['precision']:.4f} "
        f"tpr@fpr1={m.get('tpr_at_fpr1', 0):.4f} tpr@fpr10={m.get('tpr_at_fpr10', 0):.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/splits/test.csv")
    ap.add_argument("--degrade", default="none",
                    help="jpeg:30 / noise:5 / blur:5 / downscale:0.5 / crop:0.1")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--by-method", action="store_true")
    ap.add_argument("--video-level", action="store_true", help="同一视频的帧取平均分再统计")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    model = DualStream(
        pretrained=False,
        use_freq=cfg["model"]["use_freq"],
        use_gate=cfg["model"]["use_gate"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ds = DeepfakeDataset(
        args.manifest, size=cfg["data"]["image_size"], train=False, degrade=args.degrade
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    labels, probs = evaluate(model, loader, device)
    rows = read_rows(args.manifest)
    items = [
        (r["method"], r["video"], int(y), float(p))
        for r, y, p in zip(rows, labels, probs)
    ]

    if args.video_level:
        agg = defaultdict(list)
        for method, video, y, p in items:
            agg[(method, video)].append((y, p))
        items = [
            (method, video, ys[0][0], float(np.mean([p for _, p in ys])))
            for (method, video), ys in agg.items()
        ]
        print(f"视频级聚合: {len(items)} 条视频")

    y = np.array([it[2] for it in items])
    p = np.array([it[3] for it in items])

    thr_main = args.threshold if args.threshold is not None else float(ckpt.get("threshold", 0.5))
    thr_f1 = ckpt.get("threshold_f1")

    results = {}
    results["fpr1"] = compute_metrics(y, p, thr_main)
    print_metrics("fpr1", results["fpr1"])

    results["half"] = compute_metrics(y, p, 0.5)
    print_metrics("half", results["half"])

    if thr_f1 is not None and abs(thr_f1 - thr_main) > 1e-9:
        results["best_f1"] = compute_metrics(y, p, float(thr_f1))
        print_metrics("best_f1_ckpt", results["best_f1"])

    if args.by_method:
        buckets = defaultdict(lambda: ([], []))
        for method, video, yy, pp in items:
            buckets[method][0].append(yy)
            buckets[method][1].append(pp)

        print("\n按方法（阈值 = 默认阈值）：")
        by = {}
        for method, (ys, ps) in sorted(buckets.items()):
            ys = np.array(ys)
            pred = np.array(ps) >= thr_main
            if ys.min() == ys.max():
                if ys[0] == 0:
                    by[method] = {"n": int(len(ys)), "fpr": float(pred.mean())}
                    print(f"  {method:<20s} n={len(ys):>5d} 全部为真  FPR={pred.mean():.4f}")
                else:
                    by[method] = {"n": int(len(ys)), "recall": float(pred.mean())}
                    print(f"  {method:<20s} n={len(ys):>5d} 全部为假  召回={pred.mean():.4f}")
            else:
                mm = compute_metrics(ys, ps, thr_main)
                by[method] = mm
                print(f"  {method:<20s} n={len(ys):>5d} f1={mm['f1']:.4f} "
                      f"acc={mm['acc']:.4f} auc={mm.get('auc', 0):.4f}")
        results["by_method"] = by

    results["degrade"] = args.degrade
    results["manifest"] = args.manifest
    results["ckpt"] = str(args.ckpt)
    results["video_level"] = bool(args.video_level)

    tag = args.degrade.replace(":", "_") if args.degrade != "none" else "clean"
    if args.video_level:
        tag += "_vid"
    # 文件名带上 manifest，避免同库/跨库两次评测互相覆盖
    manifest_tag = Path(args.manifest).stem
    out = args.out or str(Path(args.ckpt).parent / f"eval_{manifest_tag}_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
