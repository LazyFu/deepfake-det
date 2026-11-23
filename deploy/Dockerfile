FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_CKPT=runs/rgb_700/best.pt

# WeasyPrint（PDF）依赖 pango/cairo；PDF 中文需要 Noto CJK 字体
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
        libglib2.0-0 fonts-noto-cjk curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        fastapi "uvicorn[standard]" jinja2 python-multipart pillow \
        weasyprint numpy opencv-python-headless scikit-learn pyyaml tqdm

COPY app ./app
COPY src ./src
COPY configs ./configs
COPY models ./models
COPY runs/rgb_700/best.pt ./runs/rgb_700/best.pt

RUN mkdir -p data/app/uploads data/app/reports data/app/results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
