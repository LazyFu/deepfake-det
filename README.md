# 深度伪造检测课程设计

基于空域特征的深度伪造图像检测与可解释分析系统。
主模型为 ResNet18（ImageNet 预训练）RGB 基线，训练使用 580 个身份（`runs/rgb_700`）；
频域双流与质量感知门控作为消融与负结果记录在实验日志中。

主结果（3,600 帧 / 180 条视频测试集）：

| 指标 | 数值 |
|---|---|
| 帧级 AUC | 0.9328 |
| 帧级 TPR@FPR=1% | 0.596 |
| 视频级 AUC | 0.9499 |
| 视频级 TPR@FPR=1% | 0.700 |
| 跨库 Celeb-DF AUC | 0.7163 |
| 未见方法 FaceShifter AUC | 0.598 |

## 目录

```
app/                  FastAPI 服务（detector / main / report / audit / templates）
src/                  模型、数据、训练、评测
scripts/              数据处理、评测、批量、鲁棒性、可解释性脚本
configs/base.yaml     训练配置
deploy/               Dockerfile、systemd、nginx 示例
report/               LaTeX 报告与使用手册（main.pdf / manual.pdf）及图片
docs/
  guide.md            新手导览（术语、流程、代码对照）
  experiments/        实验日志、结果汇总、鲁棒性矩阵、图
data/                 数据集与运行时数据（不入库）
runs/                 模型权重与评测 json（不入库）
models/               YuNet 人脸检测模型
```

## 环境

Python 3.12+（实测 3.14），PyTorch 2.15。

```bash
python3 -m venv .venv
source .venv/bin/activate
# GPU（按 CUDA 版本调整 cu134/cu124）；纯 CPU 用 https://download.pytorch.org/whl/cpu
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 训练与评测

```bash
# 主模型：RGB 基线（580 训练身份）
python -m src.train --config configs/base.yaml --name rgb_700 --rgb-only --epochs 15 \
    --train-csv data/splits/train_700.csv --val-csv data/splits/val_700.csv

# 双流 / 无门控消融
python -m src.train --config configs/base.yaml --name dualstream --epochs 15
python -m src.train --config configs/base.yaml --name dualstream_nogate --no-gate --epochs 15

# 同库（帧级 / 视频级）
python -m src.eval --ckpt runs/rgb_700/best.pt --manifest data/splits/test_700.csv --by-method
python -m src.eval --ckpt runs/rgb_700/best.pt --manifest data/splits/test_700.csv --video-level

# 跨库
python -m src.eval --ckpt runs/rgb_700/best.pt --manifest data/celebdf_test_2k.csv

# 鲁棒性矩阵
python scripts/06_robustness_matrix.py --ckpt runs/rgb_700/best.pt \
    --out docs/experiments/robustness_rgb_700.md

# 汇总
python scripts/05_collect_results.py
```

**口径约定**：阈值只在验证集上选、在测试集冻结。
每个结果同时报“部署口径”（冻结阈值下的 F1/召回/FPR）与“排序口径”（TPR@FPR=1%）。

## 演示系统

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000
```

- `POST /v1/detect/image`：分数、判定、热图、频谱、审计哈希
- `GET /v1/report/{id}.pdf`：单页 PDF 报告
- `GET /v1/audit/verify`：审计日志哈希链校验
- `GET /healthz`

批量检测与部署方式见 [deploy/README.md](deploy/README.md)。

## 文档

- 使用手册：[report/manual.pdf](report/manual.pdf)
- 课程报告：[report/main.pdf](report/main.pdf)
- 实验日志与负结果：[docs/experiments/experiment_log.md](docs/experiments/experiment_log.md)
- 结果汇总：`python scripts/05_collect_results.py` → [docs/experiments/results.md](docs/experiments/results.md)
- 新手导览：[docs/guide.md](docs/guide.md)

## 备注

- FF++ 为第三方镜像，仅学术使用，不再分发。
- `DeepFakeDetection` 的视频 id 与真人视频对不上，未纳入子集，因此是 5 种伪造方法。
- 跨库阈值不重调；跨库指标下降主要来自分数分布偏移。
