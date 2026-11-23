#!/usr/bin/env bash
# 下载 FF++ C23 镜像（16.7 GiB，32 线程分块 + 断点续传）。
# huggingface.co 直连不通，这里走 hf-mirror.com。
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"

python3 scripts/00_download.py --out "$DEST" --workers "${WORKERS:-32}"

echo "完成:"
ls -lh "$DEST"/FaceForensics++_C23.zip
