#!/usr/bin/env python3
"""批量检测目录下的图片，输出 CSV。"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.detector import Detector  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="docs/experiments/batch.csv")
    ap.add_argument("--ckpt", default="runs/rgb_700/best.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="f1", choices=["f1", "strict"])
    args = ap.parse_args()

    det = Detector(args.ckpt)
    paths = sorted(p for p in Path(args.input).rglob("*") if p.suffix.lower() in EXTS)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"{args.input} 下没有图片")

    thr = det.threshold if args.mode == "strict" else det.threshold_f1
    rows, scores, lat = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out = det.predict(img, want_cam=False)
        verdict = "fake" if out["score"] >= thr else "real"
        rows.append([str(p), f"{out['score']:.6f}", verdict, f"{out['latency_ms']:.1f}"])
        scores.append(out["score"])
        lat.append(out["latency_ms"])
        if i % 20 == 0:
            print(f"  {i}/{len(paths)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "score", "verdict", "latency_ms"])
        w.writerows(rows)

    n_fake = sum(1 for r in rows if r[2] == "fake")
    print(f"\n共 {len(rows)} 张，判为伪造 {n_fake} 张（{n_fake/len(rows):.1%}）")
    print(f"平均耗时 {sum(lat)/len(lat):.1f} ms，总耗时 {time.time()-t0:.1f} s")
    print(f"保存 {out_path}")


if __name__ == "__main__":
    main()
