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

# 主进程仍以非 root 的 agent 用户运行；/data 为持久卷挂载点（问责存档/自选股/SVG 图表/研报库）。
# 注意：不能以 USER agent 直接启动——平台挂载的卷默认 root:root，会遮住构建期 chown 的 /data，
# 必须由入口脚本在运行期修正属主后再降权（见 scripts/docker-entrypoint.sh）。
RUN useradd -m agent && mkdir -p /data && chown -R agent:agent /app /data \
    && chmod +x scripts/docker-entrypoint.sh

ENV PORT=8000 \
    DATA_DIR=/data
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/',timeout=4)"

# main.py 读 $PORT，平台注入多少就监听多少
CMD ["scripts/docker-entrypoint.sh"]
