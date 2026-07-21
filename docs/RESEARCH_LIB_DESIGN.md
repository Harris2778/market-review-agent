# 研报库功能设计文档（v1 施工图纸）

> 本文档是完整实施规格，已含实测验证的接口细节。实施前请先 `git pull`。
> 制定日期：2026-07-22。状态：待实施。

## 目标

用户询问行业/个股时，智能体能基于券商研报库回答，差异化能力：
1. **观点聚合**：「近30天 8 篇研报覆盖茅台：6 买入 2 增持，目标价区间 X-Y」
2. **共识与分歧**：不同券商评级/目标价差异及原因
3. **引用溯源**：每个观点标注「券商名+日期+评级」，训练知识不算数

## 数据源（已实测验证 2026-07-22）

**东方财富研报中心公开接口**（免费、结构化、无需登录）：

```
GET https://reportapi.eastmoney.com/report/list
参数: industryCode=*  pageSize=50  pageNo=1  qType=0
      beginTime=YYYY-MM-DD  endTime=YYYY-MM-DD
      code=股票代码(查个股)  industryCode=行业代码(查行业)  orgCode=机构(查券商)
实测: 2 天 36 篇；返回 hits 总数 + data 列表
```

关键返回字段（实测确认）：
- `title` 标题、`orgSName` 券商简称、`publishDate` 日期、`author` 分析师
- `emRatingName` 评级（买入/增持/中性…）、`ratingChange` 评级变动
- `stockName`/`stockCode` 个股、`indvInduName` 行业
- `predictThisYearEps`/`predictNextYearEps` 盈利预测、`indvAimPriceT/L` 目标价上下限
- `encodeUrl` PDF 附件标识（v2 全文用）、`infoCode` 研报唯一 ID

补充源：新浪智研 MCP `cnStockRatingHistory`（个股评级历史，已有 token）。

## 覆盖度实测结论（2026-07-22，实施时必读）

东财接口近 30 天实测：行业研报 1322 篇、策略报告 902 篇、个股研报 398 篇（月合计 2600+）。
- **行业/策略研报非常丰富**，与板块分析主场景高度匹配，v1 够用
- **个股研报偏中小盘/新股覆盖**：大白马稀疏（宁德时代 90 天仅 2 篇，茅台 16 篇），
  头部券商（中信/中金/华泰）在免费渠道占比低，腰部券商为主
- 个股场景的短板用 `cnStockRatingHistory`（新浪 MCP 评级历史）+ 已有业绩预告数据互补
- v2 补源优先级：慧博投研 > 发现报告 > 新浪研报频道 > 韭研公社；
  爬虫层预留多源接口（ReportSource 抽象），东财只是第一个实现

## 慧博投研调研结论（2026-07-22 实地探测）

**可爬性**：
- 列表页（hibor.com.cn/report.html、/anreport_*.html、/freport_*.html）
  **服务端渲染**，requests + BeautifulSoup 即可解析，无需 Selenium；
  标题自带完整元数据：「东吴证券-璞泰来-603659-2026H1业绩预告点评：…-260720」
  （券商+个股+代码+标题+日期一条全含）
- 分类齐全：宏观经济/投资策略/行业分析/公司调研/债券/晨会/新股/并购/港美/金工
- 详情页（/data/{md5}.html）内容薄、疑似 JS 加载；
  **全文 PDF 基本需要登录/VIP**（慧博商业模式就是卖终端）；
  GitHub 现有慧博爬虫依赖 undetected-chromedriver，说明深层内容有反爬

**对比东财的优势（用户判断正确）**：
- 慧博是研报聚合平台，头部券商（中信/中金/华泰）收录远比东财免费 feed 全
- 个股覆盖（公司调研分类）比东财个股频道均衡，大白马不再稀疏
- 首页即展示数百篇近期研报，量级大于东财

**风险与对策**：
- 反爬：控制频率（≤1 req/s + 随机抖动）、缓存增量、UA 池；列表页压力小
- 条款风险：抓取列表元数据（标题/券商/日期/评级）风险较低；
  批量下载全文 PDF 违反其服务条款，v1 不碰全文
- 页面结构变动：解析器写容错 + 每日抓取任务失败告警

**结论**：v1 双源并行——东财 API（结构化评级/EPS/目标价）
+ 慧博列表页（补齐头部券商与大白马覆盖）；
全文 RAG（v2）需要用户提供慧博账号或转向其他免费全文源

## 架构

```
scripts/report_crawler.py     每日增量抓取 → 入 SQLite（也可手动跑）
agent/report_library.py       存储+检索层：init_db / upsert_reports /
                              search_reports(关键词/股票/行业/天数) /
                              rating_summary(评级聚合统计)
agent/tools.py                注册 2 个新工具挂进 TOOL_REGISTRY
agent/orchestrator.py         无需改路由（Agent 循环自动会用新工具）；
                              仅在 system prompt 注入引用规则
agent/system_prompts.py       AGENT_QUERY_PROMPT 追加研报引用规范
data/reports.db               SQLite 库（加 .gitignore，不入库）
```

## 数据库 Schema

```sql
CREATE TABLE reports (
  info_code TEXT PRIMARY KEY,   -- 东财唯一 ID
  title TEXT, org TEXT, author TEXT, publish_date TEXT,
  stock_code TEXT, stock_name TEXT, industry TEXT,
  rating TEXT, rating_change TEXT,
  eps_this_year REAL, eps_next_year REAL,
  target_price_high REAL, target_price_low REAL,
  encode_url TEXT,               -- v2 全文下载用
  created_at TEXT
);
CREATE INDEX idx_reports_stock ON reports(stock_code, publish_date);
CREATE INDEX idx_reports_industry ON reports(industry, publish_date);
```

## 新工具契约（挂进 agent/tools.py）

```python
search_research_reports(query="", stock_code="", industry="", days=30, limit=10)
  → {"ok": True, "data": {"total": N, "reports": [{title, org, date, rating,
     target_price, eps_forecast}]}}
  # 按日期倒序，limit 默认 10

get_rating_summary(stock_code="", industry="", days=30)
  → {"ok": True, "data": {"total": 8, "rating_dist": {"买入": 6, "增持": 2},
     "target_price_range": [1800, 2100], "avg_eps_forecast": 68.5,
     "latest_reports": [前 3 篇标题+机构+日期]}}
```

## Prompt 引用规范（追加进 AGENT_QUERY_PROMPT）

- 引用研报观点必须标注来源：「中金公司 7 月 20 日研报（买入评级）认为…」
- 评级/目标价/盈利预测只能用 search_research_reports / get_rating_summary
  返回的数据，训练知识里的评级信息一律无效
- 多家券商观点并存时，先给共识（评级分布、目标价区间），再给分歧点
- 工具未返回研报的标的，明说「研报库暂未覆盖」，不得编造券商观点

## 实施分期

- **v1（本规格）**：爬虫 + SQLite + 2 个工具 + 引用规范 + 测试
  预估：新文件 3 个（crawler/report_library/test），改文件 2 个（tools/prompts）
- **v2 全文 RAG**：PDF 下载（encode_url）→ pymupdf 解析 → 分块 →
  本地 bge 中文向量模型 → 向量检索工具 search_report_content
- **v3 自动化**：每日定时增量抓取（Railway cron / GitHub Actions 定时 +
  数据库文件作为 artifact 或挂 volume）

## 验收标准（v1）

1. 问「券商最近怎么看贵州茅台」→ 输出评级分布 + 目标价区间 + 各篇来源标注
2. 问「比较两家券商对半导体的观点」→ Agent 循环自主调用 search 工具对比
3. 新增测试 ≥10 个（schema/检索/聚合/工具接线/prompt 断言），全量 pytest 全绿
4. 真实 E2E：研报库 ≥500 篇存量（回填 90 天），查询响应 <3s
