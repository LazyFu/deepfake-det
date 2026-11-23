#!/usr/bin/env python3
"""把 Celeb-DF 图片整理成和 FF++ 一样的 manifest 格式。

Celeb-DF 的目录是 Celeb_V2/Test/{real,fake}/*.jpg。
本项目的标签约定：0 = 真，1 = 假。
注意 Celeb-DF 原标签是 0=Fake、1=Real，这里做了翻转。

用法：
    python scripts/04_make_celeb_manifest.py
产物：
    data/celebdf_test_full.csv   全部 1 万张
    data/celebdf_test_2k.csv     平衡采样 1000 真 + 1000 假（跨库评测用）
"""
import argparse
import csv
import random
from pathlib import Path

ROOT = Path("data/raw/celebdf/Celeb_V2/Test")


def collect():
    rows = []
    for sub, label in (("real", 0), ("fake", 1)):
        for p in sorted((ROOT / sub).glob("*.jpg")):
            rows.append([str(p), label, "CelebDF", p.stem, 0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not ROOT.exists():
        raise SystemExit(f"找不到 {ROOT}，先解压 celeb-df.zip 的 Test 目录")

    rows = collect()
    n_real = sum(1 for r in rows if r[1] == 0)
    n_fake = sum(1 for r in rows if r[1] == 1)
    print(f"共 {len(rows)} 张：真 {n_real} / 假 {n_fake}")

    Path("data").mkdir(exist_ok=True)
    with open("data/celebdf_test_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "method", "video", "frame"])
        w.writerows(rows)
    print("写出 data/celebdf_test_full.csv")

    rng = random.Random(args.seed)
    real = [r for r in rows if r[1] == 0]
    fake = [r for r in rows if r[1] == 1]
    rng.shuffle(real)
    rng.shuffle(fake)
    picked = real[: args.per_class] + fake[: args.per_class]
    rng.shuffle(picked)
    with open("data/celebdf_test_2k.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "method", "video", "frame"])
        w.writerows(picked)
    print(f"写出 data/celebdf_test_2k.csv（真 {args.per_class} / 假 {args.per_class}）")


if __name__ == "__main__":
    main()
