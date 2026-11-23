#!/usr/bin/env python3
"""方案 A：测试/验证集不变，只把新身份追加进训练集。

产物（默认）：
    data/splits/train_700.csv   旧 train + 新身份
    data/splits/val_700.csv     与旧 val 完全相同
    data/splits/test_700.csv    与旧 test 完全相同

用法：
    python scripts/09_expand_train.py \
        --extra-manifest data/manifest_extra.csv --suffix _700
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

FIELDS = ["path", "label", "method", "video", "frame"]


def read(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  写出 {path.name:<22s} {len(rows):>6d} 帧")


def ids_of(rows):
    return {Path(r["video"]).stem.split("_")[0] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train.csv")
    ap.add_argument("--val", default="data/splits/val.csv")
    ap.add_argument("--test", default="data/splits/test.csv")
    ap.add_argument("--extra-manifest", default="data/manifest_extra.csv")
    ap.add_argument("--suffix", default="_700")
    args = ap.parse_args()

    train = read(Path(args.train))
    val = read(Path(args.val))
    test = read(Path(args.test))
    extra = read(Path(args.extra_manifest))

    old_ids = ids_of(train) | ids_of(val) | ids_of(test)
    new_ids = ids_of(extra)
    overlap = new_ids & old_ids
    if overlap:
        raise SystemExit(f"新身份与旧身份重叠 {len(overlap)} 个，停止：{sorted(overlap)[:5]}")

    train_new = train + extra
    out = Path("data/splits")
    print("统计：")
    write(out / f"train{args.suffix}.csv", train_new)
    write(out / f"val{args.suffix}.csv", val)
    write(out / f"test{args.suffix}.csv", test)

    tr_ids, va_ids, te_ids = ids_of(train_new), ids_of(val), ids_of(test)
    print("\n自检：")
    print(f"  身份数 train/val/test = {len(tr_ids)}/{len(va_ids)}/{len(te_ids)}")
    print(f"  训练与测试身份重叠 = {len(tr_ids & te_ids)}（必须为 0）")
    print(f"  训练集标签分布 = {dict(Counter(r['label'] for r in train_new))}")
    print(f"  测试集标签分布 = {dict(Counter(r['label'] for r in test))}")


if __name__ == "__main__":
    main()
