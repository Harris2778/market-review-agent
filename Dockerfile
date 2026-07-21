# market-review-agent 生产镜像（Fly.io / 任意容器平台通用；Render 也可选 Docker runtime）
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖再吃镜像缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 非 root 运行；/data 为持久卷挂载点（问责存档/自选股/SVG 图表）
RUN useradd -m agent && mkdir -p /data && chown -R agent:agent /app /data
USER agent

ENV PORT=8000 \
    DATA_DIR=/data
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/',timeout=4)"

# main.py 读 $PORT，平台注入多少就监听多少
CMD ["python", "main.py"]
