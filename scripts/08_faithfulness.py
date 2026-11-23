#!/usr/bin/env python3
"""Grad-CAM 的 deletion/insertion 定量验证，与随机遮挡对比。"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DualStream  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STEPS = np.linspace(0.0, 1.0, 11)


def load_images(manifest: str, n: int, size: int = 224):
    import cv2

    rows = [r for r in csv.DictReader(open(manifest)) if int(r["label"]) == 1]
    out = []
    for r in rows:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        out.append(torch.from_numpy(img.transpose(2, 0, 1)).float())
        if len(out) >= n:
            break
    return torch.stack(out)


def gradcam_batch(model, x):
    """对 batch 里每张图分别算 CAM，返回 (N,H,W)，CPU float。"""
    layer = model.rgb.layer4[-1].conv2
    acts, grads = {}, {}

    def fwd(m, i, o):
        acts["a"] = o

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    cams = []
    for i in range(x.size(0)):
        model.zero_grad(set_to_none=True)
        xi = x[i : i + 1].clone().requires_grad_(True)
        model(xi)[0, 1].backward()
        w = grads["g"].detach()[0].mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * acts["a"].detach()[0]).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cams.append(cam.cpu())
    h1.remove()
    h2.remove()
    return torch.stack(cams)


@torch.no_grad()
def prob_fake_batch(model, x):
    return torch.softmax(model(x).float(), dim=1)[:, 1]


def curve(model, x, cams, mode: str, order: torch.Tensor):
    """order: (N,H*W) 像素访问顺序索引。返回每个遮挡比例下的平均伪造概率。"""
    n, _, H, W = x.shape
    flat = x.reshape(n, 3, H * W)
    # 用强模糊图做“填充背景”，避免灰色块本身变成 OOD 伪影
    blur = F.avg_pool2d(x, kernel_size=31, stride=1, padding=15).reshape(n, 3, H * W)
    probs = []
    for f in STEPS:
        k = int(round(f * H * W))
        mask = torch.zeros(n, H * W, dtype=torch.bool, device=x.device)
        if k > 0:
            idx = order[:, :k]
            mask.scatter_(1, idx, True)
        mask3 = mask.unsqueeze(1).expand(-1, 3, -1)
        if mode == "deletion":
            xin = torch.where(mask3, blur, flat)
        else:
            xin = torch.where(mask3, flat, blur)
        probs.append(prob_fake_batch(model, xin.reshape(n, 3, H, W)).cpu().numpy())
    return np.array(probs)  # (steps, N)


def auc_of(curve_arr, baseline=0.5):
    """对概率曲线做归一化后算面积，0.5 为无变化。"""
    p = curve_arr.mean(axis=1) / max(baseline, 1e-6)
    return float(np.trapezoid(np.clip(p, 0, 2), STEPS) / (STEPS[-1] - STEPS[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/baseline_rgb_v2/best.pt")
    ap.add_argument("--manifest", default="data/splits/test.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="docs/experiments/faithfulness.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DualStream(pretrained=False, use_freq=cfg["model"]["use_freq"],
                       use_gate=cfg["model"]["use_gate"],
                       dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = load_images(args.manifest, args.n).to(device)
    cams = gradcam_batch(model, x)
    n, _, H, W = x.shape
    # CAM 是 7x7（layer4 输出），必须上采样到输入分辨率，否则像素顺序全错
    cams = F.interpolate(cams.unsqueeze(1), size=(H, W), mode="bilinear",
                         align_corners=False).squeeze(1)

    order_cam = cams.reshape(n, -1).argsort(dim=1, descending=True).to(device)
    # 3 组随机像素顺序作为基线，抵消单次随机排列的方差
    orders_rand = [
        torch.stack([torch.randperm(H * W) for _ in range(n)]).to(device)
        for _ in range(3)
    ]

    base = float(prob_fake_batch(model, x).mean())
    res = {"ckpt": args.ckpt, "n": n, "base_score": base}

    with torch.no_grad():
        for mode in ("deletion", "insertion"):
            c_cam = curve(model, x, cams, mode, order_cam)
            c_rand = np.mean([curve(model, x, cams, mode, o) for o in orders_rand], axis=0)
            res[f"{mode}_cam"] = auc_of(c_cam, base)
            res[f"{mode}_random"] = auc_of(c_rand, base)
            print(f"{mode:9s} CAM AUC={res[f'{mode}_cam']:.4f}  "
                  f"随机 AUC={res[f'{mode}_random']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"保存 {args.out}")


if __name__ == "__main__":
    main()
