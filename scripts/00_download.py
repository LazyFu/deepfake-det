#!/usr/bin/env python3
"""多线程分块下载 FF++ C23 镜像（支持断点续传）。

背景：huggingface.co 直连不通，hf-mirror.com 可用但单连接只有 ~80 KB/s。
      分成 32 MB 块、开 32 个连接并行拉，速度大约能到 2-4 MB/s。

用法：
    python scripts/00_download.py --out data/raw --workers 32
    # 中断后重跑同一条命令即可续传；全部完成会自动拼成 zip 并删掉分块。

产物：
    data/raw/FaceForensics++_C23.zip
    data/raw/.chunks/          （下载中的分块，完成后自动删除）
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://hf-mirror.com/datasets/bitmind/FaceForensicsC23/resolve/main/"
    "FaceForensics%2B%2B_C23.zip"
)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
CHUNK = 32 * 1024 * 1024


def resolve(url: str, timeout: int = 60) -> tuple[str, int]:
    """跟随重定向拿到最终 CDN 地址和文件大小。"""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        size = int(r.headers.get("Content-Length", 0))
        final = r.geturl()
    if size <= 0:
        raise RuntimeError("拿不到 Content-Length，检查 URL 或网络")
    return final, size


def fetch_range(url: str, start: int, end: int, dest: Path, timeout: int = 120) -> None:
    """下载 [start, end] 到 dest；失败抛异常，由调用方重试。"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise RuntimeError(f"服务器没按 Range 返回（HTTP {r.status}）")
        with open(dest, "wb") as f:
            while True:
                buf = r.read(1024 * 512)
                if not buf:
                    break
                f.write(buf)


class Progress:
    def __init__(self, total: int, parts: Path):
        self.total = total
        self.parts = parts
        self.lock = threading.Lock()
        self.stop = False
        self.t0 = time.time()
        self.done_bytes = 0

    def scan(self) -> int:
        return sum(p.stat().st_size for p in self.parts.glob("chunk_*"))

    def run(self) -> None:
        last = time.time()
        while not self.stop:
            time.sleep(5)
            now = time.time()
            cur = self.scan()
            speed = (cur - self.done_bytes) / max(now - last, 1e-6) if self.done_bytes else cur / max(now - self.t0, 1e-6)
            self.done_bytes, last = cur, now
            pct = 100.0 * cur / self.total
            eta = (self.total - cur) / speed / 60 if speed > 0 else float("inf")
            print(
                f"[{time.strftime('%H:%M:%S')}] {cur/2**30:.2f}/{self.total/2**30:.2f} GiB "
                f"({pct:.1f}%) {speed/2**20:.2f} MiB/s ETA {eta:.0f} min",
                flush=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("FFPP_URL", DEFAULT_URL))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final_zip = out / "FaceForensics++_C23.zip"
    if final_zip.exists() and final_zip.stat().st_size > 0:
        print(f"已存在 {final_zip}（{final_zip.stat().st_size/2**30:.2f} GiB），跳过下载")
        return 0

    parts = out / ".chunks"
    parts.mkdir(exist_ok=True)
    meta_file = parts / "meta.json"

    print(f"解析地址: {args.url}", flush=True)
    cdn_url, size = resolve(args.url)
    print(f"文件大小: {size/2**30:.2f} GiB", flush=True)

    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("size") != size:
            print("检测到旧的、大小不一致的分块目录，清空重下", flush=True)
            shutil.rmtree(parts)
            parts.mkdir()
        else:
            cdn_url = meta.get("cdn", cdn_url)
            print("发现未完成的分块，继续续传", flush=True)
    meta_file.write_text(json.dumps({"size": size, "cdn": cdn_url}))

    n_chunks = (size + CHUNK - 1) // CHUNK
    todo: list[tuple[int, int, int]] = []
    for i in range(n_chunks):
        start = i * CHUNK
        end = min(start + CHUNK, size) - 1
        p = parts / f"chunk_{i:05d}"
        if p.exists() and p.stat().st_size == end - start + 1:
            continue
        todo.append((i, start, end))
    print(f"共 {n_chunks} 块，待下载 {len(todo)} 块，并发 {args.workers}", flush=True)

    q: queue.Queue = queue.Queue()
    for item in todo:
        q.put(item)
    errors: list[str] = []
    err_lock = threading.Lock()
    prog = Progress(size, parts)
    th = threading.Thread(target=prog.run, daemon=True)
    th.start()

    def worker(idx: int) -> None:
        while True:
            try:
                i, start, end = q.get_nowait()
            except queue.Empty:
                return
            dest = parts / f"chunk_{i:05d}"
            for attempt in range(1, args.retries + 1):
                try:
                    fetch_range(cdn_url, start, end, dest, args.timeout)
                    if dest.stat().st_size != end - start + 1:
                        raise RuntimeError("长度不对")
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == args.retries:
                        with err_lock:
                            errors.append(f"chunk {i}: {e}")
                        return
                    time.sleep(min(2 ** attempt, 30))

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    prog.stop = True
    th.join(timeout=1)
    if errors:
        print("有分块下载失败（重跑本脚本会只补失败块）：", flush=True)
        for e in errors[:10]:
            print("  ", e, flush=True)
        return 1

    print("拼接分块 → zip ...", flush=True)
    with open(final_zip, "wb") as fo:
        for i in range(n_chunks):
            with open(parts / f"chunk_{i:05d}", "rb") as fi:
                shutil.copyfileobj(fi, fo, length=8 * 1024 * 1024)
    got = final_zip.stat().st_size
    if got != size:
        print(f"拼接后大小不符: {got} != {size}", flush=True)
        return 1
    shutil.rmtree(parts)
    print(f"完成: {final_zip} ({got/2**30:.2f} GiB, 用时 {(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
