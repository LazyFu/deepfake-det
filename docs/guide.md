# 新手导览：这个项目到底在做什么

> 注：本文写于项目早期，用于解释数据与术语；最终主模型为 `runs/rgb_700`（580 个训练身份），
> 双流与门控作为消融保留。命令与指标以 README 和 `docs/experiments/experiment_log.md` 为准。

> 给第一次做深度伪造检测的人看的。目标：读完能对着代码讲清楚每个名词，
> 知道下一步敲什么命令，答辩被问到时知道怎么答。

---

## 0. 一句话说清项目

给系统一张人脸图（或一段视频），它输出一个 0~1 的分数：
**这个分数是"AI 伪造"的概率**；同时给一张热图说明"我主要看哪里判的"，
再生成一份 PDF 报告存档。

我们不追求"什么伪造都能检测"，只追求四件事：

1. 同一个数据集内（同库）测得准；
2. 换一个数据集（跨库）别崩；
3. 图片被压缩、加噪、裁剪后别崩；
4. 结果能解释、系统能演示。

这四件事对应报告里的四张核心表，也是答辩的主线。

---

## 1. 全局地图：一条流水线

```
原始视频 ──抽帧+裁脸──> 人脸图片 ──训练──> 模型权重 ──评测──> 指标表
                                    │
                                    └──部署──> FastAPI 系统 ──> 网页/PDF 报告
```

对应目录：

| 阶段 | 目录/文件 | 产物 |
|---|---|---|
| 数据 | `scripts/00~03`、`data/` | `data/frames/`、`data/manifest.csv`、`data/splits/*.csv` |
| 模型 | `src/model.py` | 网络结构 |
| 训练 | `src/train.py` | `runs/<实验名>/best.pt` |
| 评测 | `src/eval.py`、`src/metrics.py` | `runs/<实验名>/eval_*.json` |
| 系统 | 待写 `app/` | 网页 + 接口 + PDF |

**先有数据，再有模型，再有指标，最后才是系统。** 顺序不能反。

---

## 2. 数据：FF++ 到底是什么

### 2.1 数据集本身

FaceForensics++（简称 FF++）是深度伪造检测最常用的公开数据集：

- 从 YouTube 抓了 **1000 段真人视频**（1000 个不同的人，这是关键）；
- 用多种换脸/重演算法，基于这些真人视频生成伪造视频，每人若干条；
- 常见伪造方法 6 种：`Deepfakes`、`Face2Face`、`FaceSwap`、`NeuralTextures`、
  `FaceShifter`、`DeepFakeDetection`（最后一个是 Google 的，单独一批）；
- 所以总量约 **1000 条真 + 6000 条假**，且真假视频共享同一个"人"。

### 2.2 c23 / c40 / raw 是什么

FF++ 官方提供三种压缩版本，因为**视频在网络上会被重新压缩，检测器必须扛得住**：

| 版本 | 含义 | 类比 |
|---|---|---|
| raw | 几乎无损 | 刚拍出来的原片 |
| c23 | H.264 压缩，CRF=23（中等） | 微信发一遍的画质 |
| c40 | H.264 压缩，CRF=40（很重） | 压了三四遍的网图 |

CRF 是 x264 的质量参数，**数字越大压缩越狠、画质越差**。c23 是主实验常用档，
c40 用来测鲁棒性。你的镜像只有 c23，所以 c40 这一档我们**自己用 OpenCV 重新编码**生成
（`src/degradations.py` 里的 `jpeg` 只是单帧近似；真正的二次编码后面要补一个 H.264 版本）。

### 2.3 你看到的那个 17 GB 文件是什么

`huggingface.co/datasets/bitmind/FaceForensicsC23` 是**第三方转存的 FF++ C23 压缩包**
（16.66 GiB 的 zip），不是官方发布渠道。内容就是上面说的 1000 真 + 6000 假、6 种方法。

注意三点：

1. **只有 C23**，没有 raw/c40；
2. 是**别人上传的镜像**，学术使用没问题，但报告里要引用 FF++ 原论文，
   并且**不要把数据再分发**（官方要求签协议）；
3. 下载走 `hf-mirror.com`（huggingface.co 在国内直连不通，实测单连接只有 ~80 KB/s，
   所以脚本改成了 32 线程分块下载）。

### 2.4 为什么只抽 1200 条视频

7,000 条视频全用：抽帧后几十万张，4060 8G 要跑很久，而且课程项目不需要。

`scripts/01_extract_subset.py` 默认抽 **600 真 + 6 种假各 100 = 1200 条**，
每条视频抽 ~20 帧 → 约 2.4 万张人脸图。这个量级：

- 一个 epoch 几分钟；
- 真假大致平衡；
- 足够跑出可信的对比实验。

### 2.5 抽帧、裁脸、身份划分（三个必须懂的概念）

**抽帧**：视频是 30 帧/秒，相邻帧几乎一样。每秒取 1 帧（`--fps 1`），
一条 10 秒视频取 10 张，既降数据量又不丢信息。

**裁脸**：检测器只看脸。用 OpenCV 的 YuNet 人脸检测器找到最大的一张脸，
外扩 1.4 倍后裁成 256×256。这样输入统一，模型不用学"人脸在哪"。

**身份划分（identity split）**：这是最容易出错、也最容易被老师问的地方。

> FF++ 里 `000.mp4`（真人）和 `000_003.mp4`（用 000 生成的假视频）是**同一个人**。

如果随机按帧划分，同一个人的脸会同时出现在训练集和测试集里，模型只要"记住这张脸"
就能拿高分——这叫**身份泄漏（identity leakage）**，指标虚高，跨库立刻崩。

正确做法：**按源视频 id（`000`）分组**，整组一起进 train / val / test。
`scripts/03_make_splits.py` 就是干这个的（`video` 列取 `_` 前的 id）。

**帧级 vs 视频级**：

- 帧级：每张图算一个分，指标按图统计；
- 视频级：一条视频抽 16 帧，取平均/最大分，再判这条视频真假。

论文两个都报。视频级通常更高，因为多帧投票平滑了噪声。

---

## 3. 术语表（按"你会用到的顺序"排）

### 3.1 任务类

| 术语 | 大白话 | 在本项目里 |
|---|---|---|
| 二分类 | 只分"真/假"两类 | `label=0` 真，`label=1` 假 |
| 同库测试 | 训练和测试用同一个数据集的不同人 | FF++ 内部 train/test |
| 跨库测试 | 训练用 FF++，测试用 Celeb-DF | 检验泛化，最重要的表 |
| 零样本 | 测试时的伪造方法训练时**完全没见过** | 我们用"留出一种方法"近似（leave-one-method-out），成本低 |
| 开集识别 | 不仅判真假，还要判"这是不是我没见过的伪造类型" | 本项目**不做**，报告里提一句即可 |
| 域偏移 | 两个数据集的压缩、分辨率、人脸尺寸分布不一样 | 跨库掉点的主因 |

### 3.2 特征类（空域 / 频域 / 噪声）

**空域**：就是像素本身。RGB 三个通道，模型直接看颜色和纹理。

**频域**：把图像做二维傅里叶变换（FFT），从"每个像素多亮"变成
"每个频率成分多强"。低频=大块颜色和轮廓，高频=边缘、纹理、噪声。

> 类比：一首歌，空域是波形，频域是频谱。人说话的低音是低频，齿音是高频。

**为什么伪造有频域痕迹**：换脸要做**缩放、旋转、上采样（插值）**，
生成模型（GAN/扩散）还会留下**棋盘格（checkerboard）**和固定频率的噪声。
这些操作会在频谱上留下规则的高频亮点，真人照片没有。这是本项目的核心假设。

**log|FFT|**：频谱的动态范围极大（低频比高频强几个数量级），
先取绝对值再取 log，才能看清高频的细节。代码：`src/model.py:90-94`。

**SRM（Steganalysis Rich Model）**：一组固定的高通滤波器。
高通=只让"变化剧烈"的地方通过，把平滑的内容滤掉，剩下的就是**噪声残差**。

> 类比：把照片减去它的模糊版，剩下的就是"细节和噪声"。

伪造的拼接边界、生成器噪声都藏在残差里。代码：`SRMFilter`，`src/model.py:7-35`。
**滤波器是固定的、不参与训练**，所以参数量几乎不增加。

**高频能量**：图像里"变化有多剧烈"的简单度量。
`gray - 3×3 均值滤波(gray)` 的绝对值平均。压缩越狠，高频越少。
代码：`src/model.py:101-104`，被用作"质量代理"。

### 3.3 模型类

| 术语 | 解释 | 代码位置 |
|---|---|---|
| CNN | 卷积神经网络，用滑动的卷积核提特征 | 整个 `src/model.py` |
| ResNet18 | 18 层的残差网络，ImageNet 预训练，轻量 | `src/model.py:57` |
| 迁移学习/预训练 | 先在大规模图像上学通用特征，再在小型任务上微调 | `weights=ResNet18_Weights.IMAGENET1K_V1` |
| 双流/双分支 | 两条独立的特征提取路：RGB 一路、频域+残差一路 | `self.rgb` + `self.freq` |
| 特征拼接 concat | 两路各输出一个向量，首尾接起来 | `torch.cat([r, f], dim=1)`，`model.py:108` |
| 门控 / FiLM | 用一个小网络算一个权重，去缩放另一路的特征 | `self.gate`，`model.py:67-69, 99-106` |
| GAP 全局平均池化 | 把 H×W 的特征图压成一个数，省参数 | `AdaptiveAvgPool2d(1)` |
| logits | 网络最后一层输出的原始分数（未归一化） | `self.head` 输出 2 维 |
| softmax | 把 logits 变成概率，两类加起来=1 | `src/engine.py:13` |
| 阈值 threshold | 概率超过多少判为假 | 验证集上按 FPR=1% 选 |

**门控到底在干什么**（这是答辩最容易被追问的点）：

图像高频能量高（清晰）→ 频域特征可信 → 门控值接近 1，频域分支全开；
图像被压得很糊（高频低）→ 频域特征不可信 → 门控值接近 0，自动降权。

这样模型在强压缩下不会"硬信"一个已经坏掉的线索。产物是一张
**"门控值 vs JPEG 质量"曲线**，能直接画进报告。

### 3.4 训练类

| 术语 | 解释 | 代码 |
|---|---|---|
| 交叉熵 CE | 分类任务的标准损失，预测越偏、罚得越重 | `F.cross_entropy`，`train.py:124` |
| 一致性正则 / KL | 同一张图的两个版本，预测应该一致；用 KL 散度衡量差异 | `train.py:125-132` |
| 数据增强 | 随机翻转、加退化，让模型见到更多变化 | `dataset.py:33-39`、`degradations.py` |
| AdamW | 常用优化器，带权重衰减 | `train.py:98` |
| cosine 学习率 | 学习率按余弦曲线从大到小，末期更稳 | `train.py:101` |
| AMP 混合精度 | 用 fp16 算，省显存、提速 | `train.py:103-104` |
| early stopping | 验证集连续 N 轮不涨就停，防过拟合 | `train.py:168-172` |
| batch size | 一次喂多少张图 | 8G 显存用 32，OOM 就 16 |
| 随机种子 | 固定后结果可复现 | `train.py:22-26` |

**注意**：现在代码里的"一致性"是给图加高斯噪声（`x + 0.05*randn`），
比"干净图 vs 退化图"弱。报告里如果要讲"多视图一致性"，建议改成
对同一张图施加两次不同的随机退化，再算 KL——改动只有几行，效果更好讲。

### 3.5 评测类（报告里最容易被追问）

先把四个格子记住（混淆矩阵）：

```
              预测真    预测假
真实是真        TN        FP      ← FP = 误报（把真图当假）
真实是假        FN        TP      ← FN = 漏报（把假图当真）
```

| 指标 | 公式/含义 | 一句话 |
|---|---|---|
| ACC 准确率 | (TP+TN)/全部 | 整体对多少 |
| Precision 精确率 | TP/(TP+FP) | 说它是假的时候，有多准 |
| Recall 召回率 | TP/(TP+FN) | 假的里面，抓出来多少 |
| F1 | 2PR/(P+R) | 精确率和召回率的调和平均，类别不平衡时比 ACC 可信 |
| AUC | ROC 曲线下面积 | 与阈值无关的排序能力，0.5=瞎猜，1=完美 |
| EER | 误报率=漏报率时的错误率 | 越低越好，跨库常用 |
| TPR@FPR=1% | 误报控制在 1% 时能抓多少假 | 业务最关心：不能冤枉好人 |
| FPR | FP/(FP+TN) | 误报率 |

**阈值校准**：模型输出的是概率，"多少分算假"这个线要在**验证集**上定。
本项目按 `FPR ≤ 1%` 选阈值（`metrics.py:19-28`），然后**把这个阈值写进 checkpoint**，
跨库测试时**绝不重新调阈值**——一调就说明你在偷看测试集，老师一问就穿帮。

**为什么 F1 和 FPR 会打架**：阈值调低，召回高但误报多；调高反过来。
课程项目里两个都报，并说明你选的折中点。

**消融实验（ablation）**：每次只加一个模块，看指标涨多少，证明"这个模块有用"。
我们的 6 组：

```
A0 RGB-only
A1 + 频域(FFT)
A2 + SRM 残差
A3 + 质量门控
A4 + 一致性损失
A5 + 退化增广（完整版）
```

**鲁棒性矩阵**：行=训练条件（干净/加退化），列=测试退化
（clean / JPEG 70/50/30 / noise / blur / downscale / crop），格子里填 F1。
一眼看出"训练时加退化，测试时掉得少"。

### 3.6 可解释类

**Grad-CAM**：把模型最后一层卷积的梯度反传回去，得到一张热图，
标出"哪些像素对判假贡献大"。

> 类比：模型判"假"，Grad-CAM 告诉你它盯着眼睛边缘还是嘴角。

**deletion / insertion AUC**（定量验证热图，别人一般不做）：

- deletion：按热图从高到低遮住像素，看假概率掉多快；掉得越快，热图越准；
- insertion：反过来，只露出高分区域，看概率升多快；
- 和"随机遮挡"对比，画两条曲线算 AUC。**有一个数字，比只放图强很多。**

**频谱异常图**：算一批真人图的平均 log 频谱，测试图减去它，
正值（红色）就是"这张图多出来的高频成分"。直观展示频域伪影。

### 3.7 系统类

| 术语 | 解释 |
|---|---|
| FastAPI | Python 的 Web 框架，几行代码就能起 HTTP 接口，自带 Swagger 文档页 |
| REST 接口 | 用 URL + HTTP 方法表示操作，如 `POST /v1/detect/image` |
| 异步任务 | 视频检测慢，先返回 job_id，前端轮询结果，避免请求超时 |
| PDF 报告 | HTML 模板 + 数据 → 渲染成 PDF（weasyprint，纯 Python） |
| 审计日志 | 每次调用记一行 JSON（时间、输入哈希、结果、耗时），可追溯 |
| 哈希链 | 每条日志带前一条的 SHA256，改一条就对不上，证明日志没被篡改 |
| p50/p95 时延 | 一半请求快于 p50，95% 快于 p95；比"平均值"更能反映体验 |
| 吞吐 | 每秒处理多少帧/多少请求 |

---

## 4. 术语 ↔ 代码对照（答辩前对着看）

| 你要讲的东西 | 打开哪个文件 | 看哪几行 |
|---|---|---|
| 双流结构 | `src/model.py` | 53-111 |
| FFT 幅度谱 | `src/model.py` | 90-94 |
| SRM 残差 | `src/model.py` | 7-35, 96-97 |
| 质量门控 | `src/model.py` | 99-106 |
| 损失函数 | `src/train.py` | 121-140 |
| 退化生成 | `src/degradations.py` | 全文件 |
| 阈值选择 | `src/metrics.py` | 19-28 |
| 指标计算 | `src/metrics.py` | 31-56 |
| 身份划分 | `scripts/03_make_splits.py` | 54-72 |
| 抽帧裁脸 | `scripts/02_extract_faces.py` | 31-51, 95-123 |
| 单库/跨库评测 | `src/eval.py` | 全文件 |

---

## 5. 8 周排期与三人分工

| 周 | 做什么 | 完成标志 |
|---|---|---|
| W0 | 环境 + 下载 + 跑通 forward | `python -c "import torch"` 成功，数据在下 |
| W1 | 抽子集/抽帧/划分 + RGB 基线 | 同库 c23 F1 ≥ 85 |
| W2 | 频域 + SRM + 门控 + 消融 v1 | 比基线 +2pt |
| W3 | 退化增广 + 压缩矩阵 | c40 相对 c23 掉 ≤10pt |
| W4 | Celeb-DF 跨库 + 阈值校准 | 跨库 AUC ≥ 80 |
| W5 | Grad-CAM + deletion/insertion + badcase | 有图有数字 |
| W6 | FastAPI + 网页 + PDF + 日志 | 上传→出报告全通 |
| W7 | 补实验 + 时延 + 报告初稿 | 报告 80% |
| W8 | PPT + 演示视频 + 终稿 | 能讲 15 分钟 |

分工口径（写进报告"分工"一节，三人等量）：

- **A 数据与鲁棒性**：数据获取清洗、退化基准、压缩矩阵、跨方法/跨库实验；
- **B 模型与可解释**：双流网络、门控、损失、消融、Grad-CAM；
- **C 系统与主实验**：主结果、阈值校准、FastAPI/网页/PDF、时延、统稿。

实际只有你一个人做，就按 A→B→C 串行，每个阶段把图表单独存到 `docs/A|B|C/`，
最后按三个包拼报告。答辩前让 B、C 各讲一遍自己那 5 分钟。

---

## 6. 现在按这个顺序敲命令

### Step 0：环境（一次性）

```bash
# Python 3.12 + PyTorch cu124（4060 用这个）
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.x.x True
```

如果 `cuda.is_available()` 是 False：先在你自己的终端跑 `nvidia-smi`，
能列出 4060 就说明驱动没问题，是当前 shell 环境没挂到 GPU；不能列出就装驱动
（Arch：`sudo pacman -S nvidia-open nvidia-utils`，重启）。

### Step 1：下载数据（16.7 GiB，约 2-4 小时）

```bash
bash scripts/00_download.sh data/raw
# 中断了就重跑同一条命令，会续传
```

### Step 2：抽子集（先看结构，再抽，抽完删 zip）

```bash
python scripts/01_extract_subset.py \
  --zip data/raw/FaceForensics++_C23.zip --list-only

python scripts/01_extract_subset.py \
  --zip data/raw/FaceForensics++_C23.zip \
  --out data/raw/ffpp --delete-zip
```

### Step 3：抽帧 + 裁脸（约 20-40 分钟）

```bash
python scripts/02_extract_faces.py \
  --src data/raw/ffpp --out data/frames \
  --manifest data/manifest.csv --fps 1 --max-frames 20
```

### Step 4：划分

```bash
python scripts/03_make_splits.py --manifest data/manifest.csv
# 跨方法实验（留出 FaceShifter 只在 test 出现）
python scripts/03_make_splits.py --manifest data/manifest.csv \
  --leave-out-method FaceShifter --suffix _unseen_faceshifter
```

### Step 5：训练基线

```bash
python -m src.train --config configs/base.yaml \
  --name baseline_rgb --rgb-only --epochs 8
```

### Step 6：训练双流 + 门控

```bash
python -m src.train --config configs/base.yaml --name dualstream --epochs 15
```

### Step 7：评测（干净 / 退化 / 按方法）

```bash
python -m src.eval --ckpt runs/dualstream/best.pt \
  --manifest data/splits/test.csv

python -m src.eval --ckpt runs/dualstream/best.pt \
  --manifest data/splits/test.csv --degrade jpeg:30

python -m src.eval --ckpt runs/dualstream/best.pt \
  --manifest data/splits/test.csv --by-method
```

### Step 8：系统（W6 再做）

`app/` 目录：FastAPI + 单页 HTML + PDF 报告 + 审计日志。

---

## 7. 常见坑（按踩坑概率排序）

1. **huggingface.co 直连超时**：用 `hf-mirror.com`，脚本已改好。
2. **单连接下载只有 80 KB/s**：必须多线程分块，脚本已改好（32 线程）。
3. **磁盘不够**：16.7G zip + 子集 3G + 帧 2G，抽完立刻 `--delete-zip`。
4. **身份泄漏**：按 `video` 列分组，不是按帧文件名（这个 bug 已修）。
5. **同库 99% 但跨库 60%**：正常。报告里要主动分析，别藏。
6. **FPR=1% 阈值下 F1 很低**：说明模型排序能力不够，先看 AUC；
   可以同时报"F1@0.5"和"F1@FPR1%"，并解释取舍。
7. **OOM**：batch 降到 16，或把 `num_workers` 降到 2。
8. **数据许可**：报告引用 FF++ 原论文，不二次分发数据。

---

## 8. 答辩 15 分钟怎么讲

1. **问题（1 分钟）**：深度伪造泛滥；检测器在同库很准，一压缩、一换库就崩。
2. **观察（2 分钟）**：伪造要缩放/插值/生成，会在频域和噪声残差里留下痕迹；
   但压缩会破坏这些痕迹——所以不能盲信频域。
3. **方法（4 分钟）**：双流（RGB + FFT/SRM）+ 质量门控（按高频能量调频域权重）
   + 退化增广 + 一致性约束。画一张结构图。
4. **实验（4 分钟）**：同库表、跨库表、压缩矩阵、消融表、热图 + deletion AUC。
5. **系统（2 分钟）**：现场演示上传 → 分数 → 热图 → PDF。
6. **局限（1 分钟）**：没做零样本/开集；数据只有 C23；视频级聚合还很简单。
7. **Q&A（1 分钟起）**：数据/鲁棒问 A，方法/消融问 B，系统/指标问 C。

主动说局限，比被问出来强。
