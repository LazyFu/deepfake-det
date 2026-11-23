"""人脸裁剪、伪造打分、Grad-CAM 与频谱图。"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.model import DualStream

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Detector:
    def __init__(self, ckpt_path: str = "runs/rgb_700/best.pt",
                 device: str | None = None,
                 yunet_path: str = "models/face_detection_yunet_2023mar.onnx"):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt["cfg"]
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.model = DualStream(
            pretrained=False,
            use_freq=cfg["model"]["use_freq"],
            use_gate=cfg["model"]["use_gate"],
            dropout=cfg["model"]["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.threshold = float(ckpt.get("threshold", 0.5))          # 严格：FPR<=1%
        self.threshold_f1 = float(ckpt.get("threshold_f1", self.threshold))
        self.image_size = int(cfg["data"]["image_size"])

        self._acts = None
        self._grads = None
        self._register_hooks()
        self.face_det = self._build_face_detector(Path(yunet_path))

    def _register_hooks(self):
        layer = self.model.rgb.layer4[-1].conv2

        def fwd(m, i, o):
            self._acts = o.detach()

        def bwd(m, gi, go):
            self._grads = go[0].detach()

        layer.register_forward_hook(fwd)
        layer.register_full_backward_hook(bwd)

    def _build_face_detector(self, path: Path):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(YUNET_URL, path)
            except Exception:
                return None
        return cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.6, 0.3, 5000)

    def _crop_face(self, img: np.ndarray, margin: float = 1.4):
        if self.face_det is None:
            return img, None
        h, w = img.shape[:2]
        self.face_det.setInputSize((w, h))
        _, faces = self.face_det.detect(img)
        if faces is None or len(faces) == 0:
            return img, None
        areas = faces[:, 2] * faces[:, 3]
        x, y, bw, bh = faces[int(np.argmax(areas)), :4]
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * margin
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return img, None
        return img[y0:y1, x0:x1], [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

    def _to_tensor(self, crop: np.ndarray):
        img = cv2.resize(crop, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return x.to(self.device)

    def _gradcam(self, x) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        logits[0, 1].backward()
        act, grad = self._acts[0], self._grads[0]          # (C,H,W)
        w = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * act).sum(0)).detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    @staticmethod
    def _spectrum_img(gray: np.ndarray) -> np.ndarray:
        spec = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        amp = np.log1p(np.abs(spec))
        amp = (amp - amp.mean()) / (amp.std() + 1e-8)
        amp = np.clip((amp + 3) / 6, 0, 1)
        vis = (amp * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)

    @staticmethod
    def _overlay(crop: np.ndarray, cam: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (256, 256))
        heat = cv2.resize(cam, (256, 256))
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 0.55, heat, 0.45, 0)

    def predict(self, img_bgr: np.ndarray, want_cam: bool = True) -> dict:
        t0 = time.time()
        crop, box = self._crop_face(img_bgr)
        x = self._to_tensor(crop)

        with torch.no_grad():
            logits = self.model(x)
            prob = float(F.softmax(logits.float(), dim=1)[0, 1])

        cam = self._gradcam(x) if want_cam else None
        gray = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2GRAY)

        out = {
            "score": prob,
            "label": int(prob >= self.threshold_f1),
            "verdict": "fake" if prob >= self.threshold_f1 else "real",
            "threshold_strict": self.threshold,
            "threshold_f1": self.threshold_f1,
            "face_box": box,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "image_size": list(img_bgr.shape[:2]),
            "crop": crop,
            "cam": cam,
            "spectrum": self._spectrum_img(gray),
        }
        return out

    @staticmethod
    def decode(img_bytes: bytes):
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
