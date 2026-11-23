"""生成报告所需图表（英文标注，避免字体问题）。"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc as sk_auc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.dataset import DeepfakeDataset  # noqa: E402
from src.model import DualStream  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(".")
IMG = ROOT / "report" / "images"
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = DualStream(pretrained=False, use_freq=cfg["model"]["use_freq"],
                   use_gate=cfg["model"]["use_gate"],
                   dropout=cfg["model"]["dropout"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck, cfg


@torch.no_grad()
def probs_of(m, manifest, size=224, device="cuda"):
    ds = DeepfakeDataset(manifest, size=size, train=False)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    ps, ys = [], []
    for x, y in dl:
        p = torch.softmax(m(x.to(device)).float(), 1)[:, 1]
        ps.append(p.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def fig_training_curves():
    runs = [("baseline_rgb_v2", "RGB-280"), ("rgb_700", "RGB-580"),
            ("dualstream", "Dual+gate"), ("dualstream_nogate", "Dual")]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, label in runs:
        p = ROOT / "runs" / name / "history.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        ep = [int(r["epoch"]) for r in rows]
        aucs = [float(r["auc"]) for r in rows]
        ax.plot(ep, aucs, marker="o", ms=3, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC during training")
    ax.legend()
    fig.savefig(IMG / "fig_training_curve.png"); plt.close(fig)


def fig_robustness():
    files = [("baseline_rgb_v2", "RGB-280"), ("rgb_700", "RGB-580")]
    names = ["clean", "jpeg70", "jpeg50", "jpeg30", "noise5", "blur5", "down0.5", "crop10"]
    labels = ["clean", "JPEG70", "JPEG50", "JPEG30", "Noise", "Blur", "Down0.5", "Crop10"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for run, label in files:
        p = ROOT / "reports" / f"robustness_{run}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        lut = {r["name"]: r["f1"] for r in data["rows"]}
        ys = [lut.get(n, np.nan) for n in names]
        ax.plot(labels, ys, marker="o", ms=4, label=label)
    ax.set_ylabel("F1 (frozen threshold)")
    ax.set_title("Robustness under degradations")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.savefig(IMG / "fig_robustness.png"); plt.close(fig)


def fig_roc_and_scores(device):
    m, ck, cfg = load_model(ROOT / "runs/rgb_700/best.pt", device)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    panels = [("data/splits/test_700.csv", "FF++ test"),
              ("data/celebdf_test_2k.csv", "Celeb-DF")]
    scores = {}
    for manifest, label in panels:
        y, p = probs_of(m, manifest, cfg["data"]["image_size"], device)
        fpr, tpr, _ = roc_curve(y, p)
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={a:.3f})")
        scores[label] = (y, p)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (RGB-580)")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(IMG / "fig_roc.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (label, (y, p)) in zip(axes, scores.items()):
        ax.hist(p[y == 0], bins=40, alpha=0.55, label="real", density=True)
        ax.hist(p[y == 1], bins=40, alpha=0.55, label="fake", density=True)
        ax.set_title(label); ax.set_xlabel("p(fake)"); ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.savefig(IMG / "fig_score_dist.png"); plt.close(fig)


def fig_gate(device):
    m, ck, cfg = load_model(ROOT / "runs/dualstream/best.pt", device)
    ds = DeepfakeDataset("data/splits/test.csv", size=cfg["data"]["image_size"], train=False)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    gates, hfs, ys = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            logits, parts = m(x.to(device), return_parts=True)
            gray = x.to(device).mean(1, keepdim=True)
            hf = (gray - F.avg_pool2d(gray, 3, 1, 1)).abs().mean(dim=(1, 2, 3))
            gates.append(parts["gate"].squeeze(1).cpu().numpy())
            hfs.append(hf.cpu().numpy()); ys.append(y.numpy())
            if i >= 10:
                break
    g = np.concatenate(gates); hf = np.concatenate(hfs); y = np.concatenate(ys)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].scatter(hf * 1000, g, c=y, s=6, cmap="coolwarm", alpha=0.6)
    axes[0].set_xlabel("High-frequency energy (x1e-3)"); axes[0].set_ylabel("Gate value")
    axes[0].set_title("Gate is nearly constant")
    axes[1].hist(g, bins=40)
    axes[1].set_xlabel("Gate value"); axes[1].set_ylabel("count")
    axes[1].set_title(f"mean={g.mean():.4f}, std={g.std():.4f}")
    fig.savefig(IMG / "fig_gate.png"); plt.close(fig)


def fig_samples():
    rows = list(csv.DictReader(open("data/splits/test_700.csv")))
    real = [r["path"] for r in rows if r["label"] == "0"][:4]
    fake = [r["path"] for r in rows if r["label"] == "1"][:4]
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
    for j, p in enumerate(real):
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        axes[0, j].imshow(im); axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("Real", fontsize=11)
    for j, p in enumerate(fake):
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        axes[1, j].imshow(im); axes[1, j].axis("off")
        if j == 0:
            axes[1, j].set_ylabel("Fake", fontsize=11)
    fig.suptitle("Sample faces from FF++ C23")
    fig.savefig(IMG / "fig_samples.png"); plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_training_curves()
    fig_robustness()
    fig_roc_and_scores(device)
    fig_gate(device)
    fig_samples()
    for name in ["spectrum_diff.png", "spectrum_real.png", "spectrum_fake.png",
                 "heatmap_example.png", "spectrum_example.png"]:
        src = ROOT / "reports" / "figs" / name
        if src.exists():
            shutil.copy(src, IMG / name)
    for p in sorted(IMG.glob("*.png")):
        print(p.name, f"{p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
