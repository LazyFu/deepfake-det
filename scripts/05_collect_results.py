#!/usr/bin/env python3
"""扫描 runs/ 下的评测结果，生成 Markdown 汇总表。

用法：
    python3 scripts/05_collect_results.py
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

# 顺序很重要：先匹配更具体的关键词
MANIFEST_LABEL = [
    ("celebdf_test_2k", "Celeb-DF 跨库"),
    ("celebdf_test_full", "Celeb-DF 全量"),
    ("test_unseen_faceshifter", "FF++ 留出 FaceShifter"),
    ("test", "FF++ 同库"),
]


def manifest_label(stem: str) -> str:
    for key, label in MANIFEST_LABEL:
        if key in stem:
            return label
    return "?"


def load(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def norm(d):
    """统一成 {fpr1, half, best_f1} 三个阈值的字典。"""
    if not d:
        return {}
    if isinstance(d.get("fpr1"), dict):
        return {"fpr1": d.get("fpr1"), "half": d.get("half"), "best_f1": d.get("best_f1")}
    return {"fpr1": d, "half": None, "best_f1": None}


def fmt(m, key, nd=4):
    if not m:
        return "-"
    v = m.get(key)
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/experiments/results.md")
    args = ap.parse_args()

    runs = sorted(p for p in Path(args.runs).glob("*") if p.is_dir())
    lines = [
        "# 结果汇总（自动生成）",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "> 由 `python3 scripts/05_collect_results.py` 生成，不要手改。",
        "",
        "## 1. 验证集（模型选择依据）",
        "",
        "| 实验 | 最佳 epoch | AUC | F1@FPR1% | 阈值(FPR1) | 阈值(F1最优) |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        vm = load(run / "val_metrics.json")
        if not vm:
            continue
        lines.append(
            f"| {run.name} | {vm.get('best_epoch', '-')} | {fmt(vm, 'auc')} | "
            f"{fmt(vm, 'f1')} | {fmt(vm, 'threshold')} | {fmt(vm, 'threshold_f1')} |"
        )

    lines += [
        "",
        "## 2. 测试集（默认阈值 = 验证集 FPR1% 阈值）",
        "",
        "| 实验 | 数据 | 退化 | 级别 | 阈值 | AUC | F1 | 召回 | FPR | n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for run in runs:
        for f in sorted(run.glob("eval_*.json")):
            raw = load(f)
            if not raw:
                continue
            parts = f.stem.split("_")
            # eval_<manifest>_<tag>.json
            level = "视频级" if f.stem.endswith("_vid") else "帧级"
            degrade = "none"
            for d in ("jpeg", "noise", "blur", "downscale", "crop"):
                if d in f.stem:
                    degrade = d
            manifest = manifest_label(f.stem)
            m = norm(raw).get("fpr1")
            lines.append(
                f"| {run.name} | {manifest} | {degrade} | {level} | {fmt(m, 'threshold')} | "
                f"{fmt(m, 'auc')} | {fmt(m, 'f1')} | {fmt(m, 'recall')} | {fmt(m, 'fpr')} | "
                f"{fmt(m, 'n', 0)} |"
            )

    lines += [
        "",
        "## 3. 同一实验的三档阈值对照",
        "",
        "| 实验 | 数据 | 阈值档 | 阈值 | F1 | 召回 | FPR |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        for f in sorted(run.glob("eval_*.json")):
            raw = load(f)
            if not raw:
                continue
            d = norm(raw)
            manifest = manifest_label(f.stem)
            for tag, m in (("FPR1%", d.get("fpr1")), ("0.5", d.get("half")), ("F1最优", d.get("best_f1"))):
                if not m:
                    continue
                lines.append(
                    f"| {run.name} | {manifest} | {tag} | {fmt(m, 'threshold')} | "
                    f"{fmt(m, 'f1')} | {fmt(m, 'recall')} | {fmt(m, 'fpr')} |"
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"已生成 {out}（{len(runs)} 个实验目录）")


if __name__ == "__main__":
    main()
