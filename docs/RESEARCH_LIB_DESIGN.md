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
- 补源规划（2026-07-22 六路实测后修订，详见文末「数据源全景」）：
  全文层 = 东财 PDF + 新浪研报网页全文（均免登录）；发现层 = 慧博列表 +
  洞见研报 API（补大白马）+ 证券之星（结构化评级/目标价）；
  v2 回填 = 未来智库（中小盘）+ 三个皮匠（北交所/小券商）+ 同花顺 F10；
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

**结论**（2026-07-22 全景调研后修订）：v1 以「东财 API + 慧博列表」为骨架，
全文层由东财 PDF + 新浪研报网页全文双源组成（均免登录，见文末「数据源全景」），
慧博 VIP 账号降级为可选项——仅在需要头部券商 PDF 原版式时启用慢速回填

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


---

## 数据源全景调研（2026-07-22，六路并行实测）

> 本节所有结论均来自当日实际抓取验证，附实测 URL 与返回形态。
> 实施采集器时以本节为准；上文物源结论如与本节冲突，以本节为准。

### 采纳源 A：新浪财经研报中心（全文层第二源，强烈推荐）

- 列表页（SSR，GB2312 编码，裸 curl 可通，无验证码/签名/登录墙）：
  ```
  https://vip.stock.finance.sina.com.cn/q/go.php/vReport_List/kind/lastest/index.phtml?p=N
  ```
  `kind` 支持 lastest/industry 等分类；存量约 203 页 × 40 条 ≈ 8000 篇，含当日报告。
- 详情页 `vReport_Show/kind/lastest/rptid/{rptid}/index.phtml`：
  **网页全文免费**（分节正文 + 盈利预测 + 目标价 + 风险提示），实测拿到光大证券
  新易盛报告完整正文。缺点：无 PDF 版式、无图表、无结构化 EPS/目标价字段。
- 按个股查询：`https://biz.finance.sina.com.cn/qmx/stockreports.php?symbol=股票代码`
- 角色：**全文层与东财 PDF 互补**——东财有 PDF 无正文 API，新浪有正文无 PDF，
  两者都免登录，全文层零成本双保险。限速建议 ≤1 req/s。

### 采纳源 B：洞见研报 api.djyanbao.com（个股元数据索引，推荐）

- **免登录公开 JSON API**（纯 Vite SPA，数据全走此接口，无频控无验证码）：
  ```
  GET https://api.djyanbao.com/api/report/?page=1&limit=20&q=贵州茅台
  ```
  返回字段：`id, title, orgName, authors, publishAt, stockName（个股结构化！）,
  fileUrl, fileSize, pageTotal, typeId, labelIds`
- 限制：每个查询匿名只放前 ~250 条（limit=20 时 page=10 通、page=15 拒）；
  日期过滤参数 `publishAtBegin` 匿名直接 401；PDF 为私有桶匿名 403（VIP 下载）。
- 角色：**补东财大白马覆盖稀疏的发现问题**——按 stockName 建个股索引，
  元数据入库即可，全文靠东财/新浪补。茅台实测命中 ≥10000 条（被截断）。

### 采纳源 C：证券之星研报频道（增量监控 + 结构化评级，推荐）

- 五栏目 SSR 列表（GBK，裸 curl 可通，时间戳精确到秒，当晚报告当晚可见）：
  ```
  https://stock.stockstar.com/report_list/report1.htm   公司研究
  report2 行业研究 / report3 策略趋势 / report4 券商晨会 / report5 宏观研究
  分页: /list/{栏目id}_{N}.shtml
  ```
- 详情页为「研报附件原文摘录」（核心观点/投资要点，非全文，无 PDF）。
- **结构化栏目（对评级聚合极有价值）**：
  `data_all.htm` 指标速递（评级/目标价/EPS/预期涨幅）、
  `data_ih.htm` 评级调高、`data_fn.htm` 首次关注。
- 角色：每日增量监控入口 + 评级/目标价结构化数据补充；全文深度靠 A/东财。

### 采纳源 D：未来智库 vzkoo.com（v2 中小盘回填，推荐但需限速）

- 证券研报频道列表（SSR 可直接解析）：
  ```
  https://www.vzkoo.com/document/list-0-0?char=2&page=N
  ```
  标题规范含代码（如「万华化学-600309-首次覆盖报告…」），**覆盖中小盘 + 港股**
  （实测三友化工、兰花科创、南旋控股 1982.HK 等），正是东财薄弱区。
- UGC 上传模式：单篇覆盖不稳定（某只股票不保证有）；全站百万级文档。
- 全文条件：登录 + 积分兑换（3–6 积分/篇，部分免费）；详情页**免费 AI 摘要
  （研究目的/核心内容/核心数据）+ 分页在线预览**，不下载 PDF 摘要也可入库。
- 反爬：三家中最强——裸 curl/FetchURL 被拦（IP+行为风控页），
  完整浏览器 UA + Referer + Accept-Language 可过；**必须限速（≤0.5 req/s）**。

### 采纳源 E：三个皮匠 sgpjbg.com（v2 元数据/去重索引，备选）

- ⚠️ 原域名 3mbang.com 已失效（DNS 级），**代码中禁止硬编码该域名**；
  现域名 sgpjbg.com（另有 .com.cn 镜像）。
- 券商研报频道（ASP.NET SSR，裸 curl 全 200）：
  ```
  https://www.sgpjbg.com/search.html?type=3&cd=1
  ```
  日更量自称 1000+，实测首页 50 条全是当日/前日研报；标题极规范：
  「【华金证券】维琪科技（920176）-新股覆盖研究…-260720（14页）.pdf」
  覆盖含**北交所新股、中小券商、外资/港股机构**（一页即 17+ 家）。
- 全文：硬 VIP 付费墙（「下载积分：VIP专享」），免费仅摘要 + 有限预览。
- 角色：只抓列表元数据做覆盖度索引与去重，不碰全文。

### 采纳源 F：同花顺 F10 研报页（按股票池回填兜底，备选）

- ⚠️ 旧研报中心 `data.10jqka.com.cn/report/` 已 404 下线，勿用。
- F10 个股研报页（GBK，裸 curl 可通，JS 渲染但**数据内嵌在 HTML 隐藏 JSON 中**）：
  ```
  https://basic.10jqka.com.cn/{股票代码}/report.html
  解析 <div id="report_list_contents"> 内嵌 JSON
  ```
  字段：`thspj`(评级)/`title`/`source`(券商)/`researcher`/`date`/`content`(摘要)；
  实测单股 362 条历史记录。
- 无全站增量列表，需自维护股票池轮询；问财 iwencai 有 hexin-v JS 签名，不建议碰。
- 角色：重点股票池（如沪深 300）的慢速回填兜底源。

### 采纳源 G：韭研公社（v3 情绪/题材语料，备选）

- 首页 + 详情页均为 Nuxt SSR（正文在 `window.__NUXT__` 载荷），免登录免费，
  时效极好（当天帖子分钟级更新）。
- 内容为用户转载/脱水/纪要/原创，**非券商研报原文**；无评级/EPS/目标价结构。
- app API（app.jiuyangongshe.com）有 MD5 签名校验，历史回溯需逆向，不碰。
- 角色：不进研报库主表；若做「市场热点/题材线索」功能可低成本每日抓首页。

### 现成库封装：akshare（无增量，可用作便捷封装）

- `ak.stock_research_report_em(symbol)`：底层就是东财 reportapi（qType=0），
  实测茅台返回 760 条（2017 至今），含 PDF 直链（pdf.dfcfw.com 免登录 200）。
  **与东财源零增量**，但按股拉取很方便，可作回填封装。
- akshare 其余相关接口（`stock_analyst_rank_em`、`stock_institute_recommend`、
  `stock_jgdy_tj_em`）均为衍生数据，非研报本身。
- baostock、efinance 经核实**均无研报接口**。
- Tushare Pro `research_report`（含摘要 abstr + 下载链接，形态最贴合 RAG），
  但为**单独 ¥500/年**付费项，200 元档不含；预算允许时优先接入。

### 排除源（实测否决，勿再评估）

| 源 | 否决原因（实测证据） |
|---|---|
| 雪球 xueqiu.com | 无券商研报库（只有公告/新闻/帖子）；阿里云 WAF JS 挑战 + API 需登录 token，成本最高价值为零 |
| 萝卜投研 robo.datayes.com | 全部接口实测 `-403 Need login`（2024 年后免登录路径已封死）；datayes 官方 API 为付费 B 端 |
| 和讯研报 | report.hexun.com 已 NXDOMAIN；现存页面仅 AI 摘要快讯，频道实质关停 |
| 199IT 199it.com | 国际机构报告编译博客，零券商研报；PDF 走付费知识星球 |
| 迈博汇金 microbell.com | 慧博同库镜像，边际增量小；IP 限流狠（302 到解锁页）。唯一价值：逐页页图 GIF 免登录（huibobjb.hibor.com.cn），可作慧博全文的 OCR 旁路备选 |
| 发现报告 fxbaogao.com | SSG 分类页免登录可翻约 8 万条元数据，但全文 PDF 签名防盗链 + 阅读接口强制登录；且混入大量非券商内容需按 orgName 过滤。可作元数据补充，优先级低于洞见 |

### v1 源架构定稿（5 源互补，全部免费或已有凭据）

| 层 | 源 | 提供 |
|---|---|---|
| 结构化元数据 + PDF 全文 | 东方财富 reportapi（主） | 评级/EPS/目标价/PDF |
| 网页全文 | 新浪研报中心（主） | 分节正文全文，免登录 |
| 发现层 | 慧博列表页（主，列表即可） | 头部券商 + 大白马覆盖 |
| 个股索引 | 洞见研报 API（辅） | stockName 结构化个股元数据 |
| 增量监控 | 证券之星（辅） | 评级变动/首次关注/目标价提醒 |

慧博 VIP 账号的紧迫性因此下调：列表层免费可爬，全文层已有东财+新浪双源；
若后续仍要头部券商 PDF 原文，再启用 VIP 登录慢速回填方案（见上文慧博章节约束：
≤1 req/s 随机抖动、白天运行、慢回填 10–15 天、每 PDF 只抓一次永久缓存、
本地 Mac 运行、凭据走 .env 的 HIBOR_USER/HIBOR_PASS）。

### 通用采集纪律（所有源共用）

- 编码注意：新浪 GB2312、证券之星/同花顺 GBK，解析前显式指定。
- 限速：默认 ≤1 req/s + 随机抖动；未来智库 ≤0.5 req/s + 完整浏览器头。
- 所有源按 `info_code`/标题+机构+日期 去重；源字段冲突时以结构化程度高的为准
  （东财 > 证券之星 > 洞见 > 新浪 > 慧博）。
- 每日增量任务需有失败告警；页面结构变动时解析器容错降级而非崩溃。
