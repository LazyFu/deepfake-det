#!/usr/bin/env python3
"""抽帧 + YuNet 人脸裁剪，输出 256x256 人脸帧和 manifest.csv。

输入目录结构：<src>/<方法>/<...>/xxx.mp4
输出结构：    <out>/<方法>/<视频名>/<帧号>.jpg
"""
import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_model(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载 YuNet: {path}")
        try:
            urllib.request.urlretrieve(YUNET_URL, path)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"YuNet 下载失败（{e}）。\n"
                f"请手动下载这个文件，放到 {path}：\n  {YUNET_URL}\n"
                f"如果 GitHub 打不开，用镜像：\n"
                f"  https://ghfast.top/https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
                f"models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            ) from e
    return path


def crop_largest_face(frame, detector, size: int, margin: float = 1.4):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None

    areas = faces[:, 2] * faces[:, 3]
    x, y, bw, bh = faces[int(np.argmax(areas)), :4]
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * margin

    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(w, cx + side / 2))
    y1 = int(min(h, cy + side / 2))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None

    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def iter_videos(src: Path):
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            rel = p.relative_to(src)
            method = rel.parts[0]
            yield p, method, p.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/ffpp")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=float, default=2.0, help="每秒抽几帧（FF++ 300 帧约 10 秒）")
    ap.add_argument("--max-frames", type=int, default=20, help="每条视频最多存几帧")
    ap.add_argument("--limit-per-method", type=int, default=0, help="调试用，每个方法只处理 N 条")
    ap.add_argument("--model", default="models/face_detection_yunet_2023mar.onnx")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"找不到输入目录: {src}")

    model_path = ensure_model(Path(args.model))
    detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.6, 0.3, 5000)

    out_root = Path(args.out)
    videos = list(iter_videos(src))
    if args.limit_per_method > 0:
        kept, seen = [], {}
        for v in videos:
            m = v[1]
            seen[m] = seen.get(m, 0) + 1
            if seen[m] <= args.limit_per_method:
                kept.append(v)
        videos = kept
    print(f"待处理视频: {len(videos)}")

    rows = []
    failed = []
    for video, method, stem in tqdm(videos, desc="extract"):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            failed.append(str(video))
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / max(args.fps, 0.01))))
        out_dir = out_root / method / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        label = 0 if method.lower() == "real" else 1
        idx = saved = 0
        while saved < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                face = crop_largest_face(frame, detector, args.size)
                if face is not None:
                    path = out_dir / f"{idx:05d}.jpg"
                    cv2.imwrite(str(path), face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    rows.append([str(path), label, method, stem, idx])
                    saved += 1
            idx += 1

        cap.release()
        if saved == 0:
            failed.append(str(video))

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "method", "video", "frame"])
        writer.writerows(rows)

    print(f"\n保存 {len(rows)} 帧 -> {manifest}")
    if failed:
        print(f"{len(failed)} 条视频没抽到人脸，前 5 条：")
        for v in failed[:5]:
            print("  ", v)


if __name__ == "__main__":
    main()
