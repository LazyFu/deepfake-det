"""退化/增强操作。训练时随机组合，评测时按固定参数施加。"""
import cv2
import numpy as np


def jpeg(img, quality):
    q = int(np.clip(quality, 1, 100))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def gaussian_noise(img, sigma, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gaussian_blur(img, ksize):
    k = int(ksize) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def downscale(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def crop_resize(img, ratio, rng=None):
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    ch, cw = max(1, int(h * (1 - ratio))), max(1, int(w * (1 - ratio)))
    y = int(rng.integers(0, max(1, h - ch + 1)))
    x = int(rng.integers(0, max(1, w - cw + 1)))
    crop = img[y:y + ch, x:x + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def random_degradation(img, rng=None):
    """训练用：随机挑 1-2 种退化，模拟线上压缩/噪声/低分辨率。"""
    rng = rng or np.random.default_rng()
    ops = []
    if rng.random() < 0.70:
        q = int(rng.integers(30, 96))
        ops.append(lambda im, q=q: jpeg(im, q))
    if rng.random() < 0.30:
        s = float(rng.uniform(1, 8))
        ops.append(lambda im, s=s: gaussian_noise(im, s, rng))
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5]))
        ops.append(lambda im, k=k: gaussian_blur(im, k))
    if rng.random() < 0.25:
        sc = float(rng.choice([0.5, 0.25]))
        ops.append(lambda im, sc=sc: downscale(im, sc))
    if rng.random() < 0.20:
        r = float(rng.choice([0.1, 0.2]))
        ops.append(lambda im, r=r: crop_resize(im, r, rng))

    rng.shuffle(ops)
    for op in ops[:2]:
        img = op(img)
    return img


def apply_degradation(img, spec, rng=None):
    """固定退化，spec 形如 jpeg:30 / noise:5 / blur:5 / downscale:0.5 / crop:0.1。"""
    if not spec or spec == "none":
        return img
    name, _, val = spec.partition(":")
    if name == "jpeg":
        return jpeg(img, float(val))
    if name == "noise":
        return gaussian_noise(img, float(val), rng or np.random.default_rng())
    if name == "blur":
        return gaussian_blur(img, float(val))
    if name == "downscale":
        return downscale(img, float(val))
    if name == "crop":
        return crop_resize(img, float(val), rng or np.random.default_rng())
    raise ValueError(f"未知退化: {spec}")
