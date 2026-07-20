FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖 + 中文字体（matplotlib 图表用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ fonts-wqy-microhei && \
    rm -rf /var/lib/apt/lists/* && \
    fc-cache -fv

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "main.py"]
