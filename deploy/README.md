# 部署说明

三种方式，按需要选：

## 1. 本机运行（开发/演示）

```bash
cd /home/lazyfu/work/deepfake-det
source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

## 2. 局域网访问

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

放行防火墙端口（Arch 用 ufw/firewalld 自行调整），同事访问 `http://<你的IP>:8000`。

注意：**workers 必须为 1**。模型常驻显存，多 worker 会复制多份模型，8 GB 卡容易 OOM。

## 3. systemd 常驻

```bash
sudo cp deploy/deepfake.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deepfake
systemctl status deepfake
journalctl -u deepfake -f
```

`deploy/deepfake.service` 里的 `User`、`WorkingDirectory`、`ExecStart` 按实际路径改。

## 4. Docker

在仓库根目录构建（需要把模型权重一起打进去）：

```bash
docker build -f deploy/Dockerfile -t deepfake-det:1.0 .
docker run --rm -p 8000:8000 --name deepfake deepfake-det:1.0
```

镜像基于 `python:3.12-slim`，CPU 推理；GPU 部署需要 `nvidia-container-toolkit` 并把基础镜像换成 CUDA 版本。

## 5. Nginx 反向代理（可选）

```bash
sudo cp deploy/nginx.conf.example /etc/nginx/conf.d/deepfake.conf
sudo nginx -t && sudo systemctl reload nginx
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `APP_CKPT` | `runs/rgb_700/best.pt` | 模型权重路径 |

## 数据目录

运行时自动创建：

```
data/app/uploads/    上传原图
data/app/reports/    热图、频谱、PDF
data/app/results/    每次检测的 JSON
data/app/audit.jsonl 审计日志（哈希链）
```

定期清理旧文件即可；审计日志建议保留。

## 自检

```bash
curl http://127.0.0.1:8000/healthz
# {"ok":true,"ckpt":"runs/rgb_700/best.pt","device":"cuda",...}
```
