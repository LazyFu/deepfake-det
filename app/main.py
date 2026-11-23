"""FastAPI 服务：上传图片 → 伪造分数 + 热图 + 频谱图 + PDF 报告 + 审计日志。"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.audit import AuditLog
from app.detector import Detector
from app.report import render_report

CKPT = os.getenv("APP_CKPT", "runs/rgb_700/best.pt")
ROOT = Path("data/app")
UPLOADS = ROOT / "uploads"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"
for d in (UPLOADS, REPORTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Deepfake Detection Demo", version="1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/artifacts", StaticFiles(directory=str(REPORTS)), name="artifacts")
templates = Jinja2Templates(directory="app/templates")

detector = Detector(CKPT)
audit = AuditLog()


def _save_png(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


@app.get("/")
def index(request: Request):
    # Starlette 1.x 的新签名：TemplateResponse(request, name, context)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"ckpt": CKPT,
         "threshold_strict": detector.threshold,
         "threshold_f1": detector.threshold_f1},
    )


@app.get("/healthz")
def healthz():
    return {"ok": True, "ckpt": CKPT, "device": str(detector.device),
            "threshold_strict": detector.threshold,
            "threshold_f1": detector.threshold_f1}


@app.post("/v1/detect/image")
async def detect_image(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空文件")
    img = Detector.decode(raw)
    if img is None:
        raise HTTPException(400, "无法解码图片")

    rid = uuid.uuid4().hex[:12]
    ts = datetime.now().isoformat(timespec="seconds")
    sha = hashlib.sha256(raw).hexdigest()

    out = detector.predict(img)
    crop = out["crop"]
    cam = out["cam"]

    rdir = REPORTS / rid
    rdir.mkdir(parents=True, exist_ok=True)
    input_path = rdir / "input.jpg"
    cv2.imwrite(str(input_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    heat_path = rdir / "heatmap.png"
    _save_png(heat_path, detector._overlay(crop, cam) if cam is not None else crop)
    spec_path = rdir / "spectrum.png"
    _save_png(spec_path, out["spectrum"])

    thr = detector.threshold_f1
    verdict = "fake" if out["score"] >= thr else "real"
    label = 1 if verdict == "fake" else 0

    result = {
        "id": rid,
        "ts": ts,
        "filename": file.filename or "upload.jpg",
        "sha256": sha,
        "score": round(out["score"], 6),
        "threshold": thr,
        "threshold_strict": detector.threshold,
        "threshold_f1": detector.threshold_f1,
        "verdict": verdict,
        "label": label,
        "face_box": out["face_box"],
        "latency_ms": out["latency_ms"],
        "input_size": out["image_size"],
        "ckpt": CKPT,
        "artifacts": {
            "input": f"/artifacts/{rid}/input.jpg",
            "heatmap": f"/artifacts/{rid}/heatmap.png",
            "spectrum": f"/artifacts/{rid}/spectrum.png",
            "pdf": f"/v1/report/{rid}.pdf",
        },
    }
    rec = audit.append({"id": rid, "ts": ts, "sha256": sha, "filename": result["filename"],
                        "score": result["score"], "verdict": verdict, "ckpt": CKPT})
    result["audit_hash"] = rec["hash"]
    # 先写审计哈希再落盘，PDF 报告需要它
    (RESULTS / f"{rid}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return JSONResponse(result)


@app.get("/v1/report/{rid}.pdf")
def report_pdf(rid: str):
    rpath = RESULTS / f"{rid}.json"
    if not rpath.exists():
        raise HTTPException(404, "记录不存在")
    r = json.loads(rpath.read_text())
    rdir = REPORTS / rid
    ctx = {
        **r,
        "input_abs": str((rdir / "input.jpg").resolve()),
        "heat_abs": str((rdir / "heatmap.png").resolve()),
        "spec_abs": str((rdir / "spectrum.png").resolve()),
        "verdict_cn": "伪造" if r["verdict"] == "fake" else "真实",
    }
    pdf = rdir / "report.pdf"
    render_report(ctx, pdf)
    return FileResponse(str(pdf), media_type="application/pdf",
                        filename=f"deepfake_report_{rid}.pdf")


@app.get("/v1/audit/verify")
def audit_verify():
    return audit.verify()
