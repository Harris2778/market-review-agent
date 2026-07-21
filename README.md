# 市场复盘智能体

OpenAI 兼容的 A 股市场每日复盘智能体。接入清华清小搭智能体大赛平台。

## 功能

- **全市场每日复盘**：31 个申万一级行业全覆盖 + S/A/B/C 四级新闻分级 + 资金面分析
- **单板块深度聚焦**：指定行业 7 维度深度分析 + 产业链联动 + 机构观点
- **通用金融对话**：回答市场、行业、宏观相关问题

## 触发词

| 输入 | 功能 |
|---|---|
| 今日复盘 / 今天市场 / 今天A股 / 大盘分析 | 全市场复盘 |
| 聚焦半导体 / 电子板块怎么样 / 深入看新能源 | 单板块聚焦 |

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key 及其他数据源 Key
```

必填：
- `DEEPSEEK_API_KEY` — DeepSeek API Key

推荐：
- `TUSHARE_TOKEN` — A 股行情/行业/资金（200 元/年）
- `FINNHUB_API_KEY` — 全球指数（免费）
- `FRED_API_KEY` — 美国宏观数据（免费）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000
```

服务启动后访问 `http://localhost:8000/health` 查看状态。

### 4. 测试接口

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer market-review-agent-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "market-review-agent",
    "messages": [{"role": "user", "content": "今日复盘"}],
    "stream": false
  }'
```

## 部署到 Railway

1. 把代码推到 GitHub 仓库
2. 在 [Railway](https://railway.app) 连接 GitHub 仓库
3. 在 Railway 环境变量中设置所需 API Key
4. Railway 自动构建和部署，分配公网域名 `https://xxx.up.railway.app`
5. 将公网地址填入清小搭平台的"标准协议接入"

## API 端点

| 端点 | 说明 |
|---|---|
| `GET /` | 服务信息 |
| `GET /health` | 健康检查 + API 配置状态 |
| `GET /v1/models` | 列出可用模型 |
| `POST /v1/chat/completions` | OpenAI 兼容对话接口 |

### 鉴权方式

请求头中携带 `Authorization: Bearer <AGENT_API_KEY>`

## 架构

```
清小搭平台
  ↓ POST /v1/chat/completions (Authorization: Bearer xxx)
FastAPI Server
  ├─ 意图识别：全市场复盘 / 单板块聚焦 / 通用对话
  ├─ 数据采集：Tushare(行情) + Finnhub(全球) + FRED(宏观) + 财联社(新闻)
  ├─ 系统提示词：S/A/B/C 新闻分级 + 反幻觉规则 + 输出模板
  └─ DeepSeek API：生成结构化复盘报告
```

## 技术栈

- FastAPI + uvicorn
- DeepSeek API (deepseek-chat)
- Tushare Pro / Finnhub / FRED（可选数据源）
- SSE 流式输出支持
- Docker 容器化部署

## 运行测试

```bash
# 首次运行前安装测试依赖
/usr/local/bin/pip3 install -r requirements-dev.txt

# 运行全部测试
/usr/local/bin/python3 -m pytest tests/ -v
```

测试通过 `tests/conftest.py` 自动注入假的 API Key 环境变量，不会发起真实网络请求、不消耗真实 API 配额。

## 许可

MIT
