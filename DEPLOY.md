# 部署与数据持久化指南

> **2026-07-22 更新**：Railway Hobby/Free 计划高峰期会暂停构建（部署卡 QUEUED 不构建，
> 官方政策，Pro 可绕过）。如遇此问题，推荐迁移到 **Render**（仓库已内置 `render.yaml`
> 一键 Blueprint）或 **Fly.io**（仓库已内置 `Dockerfile`）。见下文「平台迁移」。

## 平台迁移：Render（推荐，最平滑）

仓库根目录的 `render.yaml` 是 Render Blueprint，包含服务定义 + 1GB 持久卷 +
全部环境变量骨架。**迁移步骤（约 10 分钟，仅需网页操作）**：

1. 注册/登录 [Render](https://render.com)（GitHub OAuth）。
2. Dashboard → **New → Blueprint** → 选 `Harris2778/market-review-agent` 仓库 → **Apply**。
3. 按提示填写 `sync: false` 的密钥变量（`DEEPSEEK_API_KEY` / `AGENT_API_KEY` /
   `TUSHARE_TOKEN` / `FINNHUB_API_KEY` / `FRED_API_KEY` / `SINA_MCP_TOKEN`，
   与 Railway 上的值相同，从 Railway Variables 页抄过来）。
4. 等首次构建完成（region 已选 Singapore，plan=starter $7/月常驻不睡，
   disk 1GB 自动挂到 `/data`，`DATA_DIR=/data` 已在 Blueprint 里配好）。
5. 验证：`curl https://<render 域名>/` 应返回 `"version":"1.3.0"`。
6. **切换清小搭**：把接入方的 API 地址从 Railway 域名改成 Render 域名（`/v1` 路径不变，
   `AGENT_API_KEY` 不变）。
7. 确认稳定后，回 Railway 把旧服务关停（Settings → Delete Service 或暂停），
   避免双份计费。

迁移后自动部署行为与 Railway 相同：push 到 main 即触发构建部署。

## 平台迁移：Fly.io（备选，最便宜约 $3.4/月）

仓库已内置 `Dockerfile`（python:3.13-slim，非 root，自带 HEALTHCHECK）。
需要本机装 `flyctl`：

```bash
brew install flyctl && fly auth login
cd market-review-agent
fly launch --region sin --vm-size shared-cpu-1x --vm-memory 512   # 约 $3.19/月
fly volumes create agent_data --size 1 --region sin               # $0.15/GB/月
# fly.toml 里加：mounts = { source = "agent_data", destination = "/data" }
fly secrets set DEEPSEEK_API_KEY=... AGENT_API_KEY=... TUSHARE_TOKEN=... \
  FINNHUB_API_KEY=... FRED_API_KEY=... SINA_MCP_TOKEN=... DATA_DIR=/data
fly deploy
```

持续自动部署：在 GitHub 仓库 Secrets 加 `FLY_API_TOKEN`（`fly tokens create deploy`），
再加一个调用 `superfly/flyctl-actions` 的 workflow（如需要可让工程师补）。

---

# 附录：Railway 部署与数据持久化指南（保留备查）

## 背景：为什么必须挂卷

Railway 容器的文件系统是**临时的（ephemeral）**：每次重部署、重启、崩溃恢复后，
容器内文件全部重置。本服务有两类运行时产出默认写在容器本地：

| 产出 | 默认目录 | 丢失后果 |
| --- | --- | --- |
| 问责存档（自我问责系统的分析 JSONL + 打分写回） | `data/archive/` | 全部历史分析与 hit/miss 记录清零 |
| SVG 图表（复盘配图，`/charts/<日期>/<文件>` URL 的来源） | `data/charts/` | 历史图表 URL 全部 404 |

解决方案：挂载 Railway Volume 到 `/data`，并把数据根目录指到卷上。

## 目录解析约定（本服务四模块行为一致）

- `ARCHIVE_DIR` 显式设置时优先；缺省推导为 `${DATA_DIR:-data}/archive`
- `CHART_DIR` 显式设置时优先；缺省推导为 `${DATA_DIR:-data}/charts`

即只需设置一个 `DATA_DIR=/data`，存档与图表即全部落卷，无需逐项配置。
模块运行时若检测到 `RAILWAY_ENVIRONMENT` 存在（Railway 自动注入）但最终目录
不在挂载卷路径下（不以 `/data` 开头），会在日志中警告一次：

```
运行在 Railway 但未挂卷，重启后数据将丢失（当前目录=...）
```

部署后请留意启动日志——出现这条警告即说明挂卷/变量未生效。

## 挂卷步骤（Railway Dashboard）

1. 打开 [Railway Dashboard](https://railway.com/dashboard) → 选择本项目服务（Service）。
2. 进入 **Volumes** 标签页 → 点击 **Add Volume**（或 **+ New Volume**）。
3. **Mount Path** 填写 `/data`（必须与下文 `DATA_DIR` 一致），保存。
4. 进入 **Variables** 标签页，添加：

   | 变量 | 值 | 说明 |
   | --- | --- | --- |
   | `DATA_DIR` | `/data` | 数据根目录；存档/图表缺省全部推导到卷上 |
   | `CHART_DIR` | `/data/charts` | **建议显式设置**：`/charts` 静态挂载目录（main.py 启动时解析）与图表生成目录保持一致，确保图表 URL 可访问 |

5. 保存变量后 Railway 自动触发重部署（或手动 **Deploy → Redeploy**）。

> 说明：`ARCHIVE_DIR` 一般无需显式设置，`DATA_DIR=/data` 推导即为 `/data/archive`。
> 仅当需要把存档放到非标准位置时才显式覆盖（显式值优先于 `DATA_DIR` 推导）。

## 部署后验证

1. **服务存活与版本**：

   ```bash
   curl https://<你的服务域名>/
   ```

   应返回 JSON，含 `"status": "running"` 与 `version` 字段。

2. **存档持久化**：通过平台问一个板块问题（例如「分析一下银行板块」），
   触发一次 `sector_deep_dive` 分析并落档。然后在 Railway 服务 Shell
   （Dashboard → 服务 → **Shell**，或 `railway shell`）检查：

   ```bash
   ls -la /data/archive/
   # 应看到 archive_YYYYMMDD.jsonl（当天日期）
   ls -la /data/charts/
   ```

3. **重启不丢**：手动 **Restart** 服务，再次 `ls /data/archive/`，
   文件应仍然存在；日志中不应再出现『未挂卷』警告。

4. **图表 URL**：若当日推送/复盘生成了图表，访问
   `https://<你的服务域名>/charts/<YYYYMMDD>/indices.svg` 应返回 SVG。

## 环境变量全表

### 必填

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API Key（LLM 驱动） |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址，默认 `https://api.deepseek.com/v1` |
| `AGENT_API_KEY` | 智能体调用密钥（平台 → 本服务的鉴权） |

### 推荐（金融/新闻数据源）

| 变量 | 说明 |
| --- | --- |
| `TUSHARE_TOKEN` | Tushare Pro Token（A 股行情/行业/资金流向） |
| `FINNHUB_API_KEY` | Finnhub API Key（全球指数） |
| `FRED_API_KEY` | FRED API Key（美国宏观数据） |
| `BRAVE_SEARCH_API_KEY` | Brave Search API Key（新闻搜索增强） |
| `SINA_MCP_TOKEN` | 新浪 MCP Token（新浪数据源，可选） |

### 持久化（本指南重点）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `data` | 数据根目录；云上必须设为 `/data` |
| `ARCHIVE_DIR` | `${DATA_DIR}/archive` | 问责存档目录；显式设置时优先于推导 |
| `CHART_DIR` | `${DATA_DIR}/charts` | 图表目录；显式设置时优先于推导 |
| `WATCHLIST_PATH` | `${DATA_DIR}/watchlist.json` | 自选股存储文件；显式设置时优先于推导 |

### 定时推送（第五波）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PUSH_WEBHOOK_URL` | 空 | 推送目标 Webhook；未配置则只生成内容记日志不发送 |
| `PUSH_TIME` | `15:40` | 推送触发时间，上海时区 `HH:MM`，仅工作日 |

### 限额（第六波已引入）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RATE_LIMIT_PER_MIN` | `30` | `/v1/chat/completions` 每分钟限流，超限 429 |
| `QUOTA_DAILY` | `500` | 每日配额（上海时区自然日），超限 429；`GET /v1/usage` 查用量 |

> 内存计数，仅单 worker 有效；多实例需外置共享计数。

### 平台自动注入（无需配置，仅供排查）

| 变量 | 说明 |
| --- | --- |
| `PORT` | Railway 注入的监听端口 |
| `RAILWAY_ENVIRONMENT` | Railway 注入；本服务用它判定是否运行在 Railway 以发出未挂卷警告 |
