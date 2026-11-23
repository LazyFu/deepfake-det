#!/usr/bin/env python3
"""在一个 checkpoint 上跑完整的退化矩阵，输出 Markdown 表和 json。

F1/召回/FPR 使用冻结阈值；TPR@1%/@10% 由该退化数据自身的 ROC 计算。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 允许 `python scripts/06_...py` 直接运行（把仓库根目录加进 import 路径）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import DeepfakeDataset
from src.engine import evaluate
from src.metrics import compute_metrics
from src.model import DualStream

DEGRADATIONS = [
    ("clean", "none"),
    ("jpeg70", "jpeg:70"),
    ("jpeg50", "jpeg:50"),
    ("jpeg30", "jpeg:30"),
    ("noise5", "noise:5"),
    ("blur5", "blur:5"),
    ("down0.5", "downscale:0.5"),
    ("crop10", "crop:0.1"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/splits/test.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    thr = float(ckpt.get("threshold", 0.5))

    model = DualStream(
        pretrained=False,
        use_freq=cfg["model"]["use_freq"],
        use_gate=cfg["model"]["use_gate"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    for name, spec in DEGRADATIONS:
        ds = DeepfakeDataset(
            args.manifest, size=cfg["data"]["image_size"], train=False, degrade=spec
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        labels, probs = evaluate(model, loader, device)
        m = compute_metrics(labels, probs, thr)
        m["degrade"] = spec
        m["name"] = name
        rows.append(m)
        print(
            f"{name:<8s} auc={m.get('auc', 0):.4f} f1={m['f1']:.4f} "
            f"recall={m['recall']:.4f} fpr={m['fpr']:.4f} "
            f"tpr@1%={m.get('tpr_at_fpr1', 0):.4f} tpr@10%={m.get('tpr_at_fpr10', 0):.4f}"
        )

    base = rows[0]
    lines = [
        "# 鲁棒性矩阵",
        "",
        f"- checkpoint: `{args.ckpt}`",
        f"- manifest: `{args.manifest}`",
        f"- 阈值（验证集冻结）: {thr:.4f}",
        "",
        "| 退化 | AUC | F1 | 召回 | FPR | TPR@1% | TPR@10% | F1 相对干净 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        delta = m["f1"] - base["f1"]
        lines.append(
            f"| {m['name']} | {m.get('auc', 0):.4f} | {m['f1']:.4f} | "
            f"{m['recall']:.4f} | {m['fpr']:.4f} | {m.get('tpr_at_fpr1', 0):.4f} | "
            f"{m.get('tpr_at_fpr10', 0):.4f} | {delta:+.4f} |"
        )

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "robustness.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"threshold": thr, "ckpt": args.ckpt, "rows": rows}, f, indent=2)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
