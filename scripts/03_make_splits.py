#!/usr/bin/env python3
"""按源视频 id 划分 train/val/test，保证同一身份的真/假视频不跨 split。

FF++ 真视频 000.mp4 与伪造 000_003.mp4 共享源 id 000，所以按 id 分组。
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


def read_manifest(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_split(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "method", "video", "frame"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name:<24s} {len(rows):>6d} 帧")


def summarize(rows):
    stat = defaultdict(int)
    for r in rows:
        stat[(r["method"], int(r["label"]))] += 1
    for (method, label), n in sorted(stat.items()):
        print(f"    {method:<20s} label={label} {n:>5d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leave-out-method", default=None,
                    help="跨方法实验：该方法的伪造样本只出现在 test")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    rows = read_manifest(Path(args.manifest))
    if not rows:
        sys.exit("manifest 为空")

    groups = defaultdict(list)
    for r in rows:
        # 用 video 列（如 000_003.mp4）取源视频 id，保证同一身份不跨 split
        vid = Path(r["video"]).stem.split("_")[0]
        groups[vid].append(r)

    ids = sorted(groups)
    random.Random(args.seed).shuffle(ids)
    n = len(ids)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    if n >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    if n_test + n_val >= n:
        n_test = max(1, n // 4)
        n_val = max(1, n // 4)
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    train_ids = set(ids[n_test + n_val:])

    print(f"身份数: {n}  train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    train, val, test = [], [], []
    for vid, rs in groups.items():
        if vid in train_ids:
            bucket = train
        elif vid in val_ids:
            bucket = val
        else:
            bucket = test
        for r in rs:
            if args.leave_out_method:
                if r["method"] == args.leave_out_method:
                    # 留出方法只出现在 test 身份上，避免身份泄漏
                    if bucket is test:
                        test.append(r)
                    continue
                if bucket is test and r["method"] != "Real":
                    continue
            bucket.append(r)

    out_dir = Path(args.out_dir)
    print("写出：")
    write_split(out_dir / f"train{args.suffix}.csv", train)
    write_split(out_dir / f"val{args.suffix}.csv", val)
    write_split(out_dir / f"test{args.suffix}.csv", test)

    print("\ntrain:")
    summarize(train)
    print("val:")
    summarize(val)
    print("test:")
    summarize(test)


if __name__ == "__main__":
    main()
