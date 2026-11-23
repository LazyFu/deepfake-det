#!/usr/bin/env python3
"""从 FF++ C23 zip 按身份抽取视频子集。

zip 结构：FaceForensics++_C23/{real,fake/<method>}/*.mp4。
每个身份取 1 条真视频 + K 条不同方法的假视频，避免身份泄漏。

用法：
    python scripts/01_extract_subset.py --zip <zip> --out data/raw/ffpp --ids 400
"""
import argparse
import csv
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ALIASES = {
    "real": "Real",
    "original": "Real",
    "deepfakedetection": "DeepFakeDetection",
    "deepfakes": "Deepfakes",
    "faceshifter": "FaceShifter",
    "faceswap": "FaceSwap",
    "face2face": "Face2Face",
    "neuraltextures": "NeuralTextures",
}
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def classify(name: str):
    """从 zip 内路径认出 (类别, 类别目录之后的相对路径)。"""
    parts = [p for p in name.split("/") if p]
    for i, part in enumerate(parts[:-1]):
        key = part.lower().replace("-", "").replace("_", "").replace(" ", "")
        if key in ALIASES:
            return ALIASES[key], "/".join(parts[i + 1:])
    return None, None


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def scan(zf: zipfile.ZipFile):
    """返回 {类别: [(ZipInfo, 相对路径, 身份id)]} 和 {身份id: {类别: [...]}}。"""
    by_class = defaultdict(list)
    by_id = defaultdict(lambda: defaultdict(list))
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(VIDEO_EXT):
            continue
        cls, rel = classify(info.filename)
        if cls is None:
            continue
        vid = Path(rel).stem.split("_")[0]
        by_class[cls].append((info, rel, vid))
        by_id[vid][cls].append((info, rel, vid))
    return by_class, by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data/raw/ffpp")
    ap.add_argument("--ids", type=int, default=400, help="抽多少个身份")
    ap.add_argument("--fakes-per-id", type=int, default=2, help="每个身份取几条假视频")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-csv", default="",
                    help="逗号分隔的 split csv；这些文件里出现过的身份会被排除，用于扩数据")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--delete-zip", action="store_true")
    args = ap.parse_args()

    zip_path = Path(args.zip).expanduser()
    if not zip_path.exists():
        sys.exit(f"找不到 zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        by_class, by_id = scan(zf)
        if not by_class:
            sys.exit("zip 里没找到已知类别，先 unzip -l 看一下结构")

        methods = sorted(c for c in by_class if c != "Real")
        print("zip 内视频数量：")
        for cls in sorted(by_class):
            print(f"  {cls:<20s} {len(by_class[cls]):>5d}")
        print(f"  身份数: {len(by_id)}")

        if args.list_only:
            return

        # 只保留“有真视频”的身份，保证每个身份真假配对
        real_ids = [vid for vid in by_id if "Real" in by_id[vid]]
        excluded = set()
        for path in [p for p in args.exclude_csv.split(",") if p.strip()]:
            for row in csv.DictReader(open(path)):
                excluded.add(Path(row["video"]).stem.split("_")[0])
        if excluded:
            before = len(real_ids)
            real_ids = [vid for vid in real_ids if vid not in excluded]
            print(f"\n有真视频的身份: {before}，排除已用 {len(excluded)} 个后剩 {len(real_ids)}")
        else:
            print(f"\n有真视频的身份: {len(real_ids)}")
        rng = random.Random(args.seed)
        rng.shuffle(real_ids)
        chosen = sorted(real_ids[: min(args.ids, len(real_ids))])

        # 按方法轮转，保证各方法数量尽量均衡
        usage = Counter()
        picks = []  # (cls, rel, info, vid)
        for vid in chosen:
            entries = by_id[vid]
            if "Real" in entries:
                info, rel, _ = entries["Real"][0]
                picks.append(("Real", rel, info, vid))

            avail = [m for m in methods if m in entries]
            used_here = set()
            for _ in range(args.fakes_per_id):
                cand = [m for m in avail if m not in used_here]
                if not cand:
                    break
                m = min(cand, key=lambda x: (usage[x], x))
                info, rel, _ = entries[m][0]
                picks.append((m, rel, info, vid))
                used_here.add(m)
                usage[m] += 1

        print(f"\n选中 {len(chosen)} 个身份，共 {len(picks)} 条视频")
        print("  其中按方法：")
        for m, n in sorted(Counter(p[0] for p in picks).items()):
            print(f"    {m:<20s} {n:>4d}")

        out_root = Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
        total = 0
        for cls, rel, info, _ in picks:
            dest = out_root / cls / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            total += 1

    size = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    print(f"\n完成：新写入 {total} 个文件，目录总计 {human(size)} -> {out_root}")

    if args.delete_zip:
        zip_path.unlink()
        print(f"已删除 {zip_path}")


if __name__ == "__main__":
    main()
