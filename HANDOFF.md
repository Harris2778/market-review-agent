# 项目交接文档（2026-07-22 更新，第六/七波 + 新闻模式三问题修复 + 输出卫生/MCP兜底修复 + 研报库v1/v2全文RAG/每日自动化 + 开源借鉴移植四模块 + 社媒舆情爬取v1 + 东财股吧落地，路线图全部完成）

> 本文档记录当前开发状态，新会话/新协作者从这里开始读。

## 项目简介

A 股金融分析智能体。FastAPI 提供 OpenAI 兼容接口（`/v1/chat/completions`），
DeepSeek（deepseek-chat）生成分析，数据源：Tushare Pro（200元/年）+ 新浪智研 MCP（75 工具）
+ yfinance + FRED + Finnhub。部署：Railway（从 GitHub main 分支自动部署；
2026-07-22 起其 Hobby 计划构建队列频繁卡死，仓库已内置 render.yaml/Dockerfile 可迁 Render/Fly，见 DEPLOY.md）。

- 仓库：https://github.com/Harris2778/market-review-agent
- 生产：https://market-review-agent-production.up.railway.app
- 接入方：清小搭（API 地址 /v1，鉴权 AGENT_API_KEY）

## 本地开发环境

- 依赖安装在 `/usr/local/bin/python3`（Python 3.14），**不要用系统其他 python**
- 运行测试：`cd ~/market-review-agent && /usr/local/bin/python3 -m pytest tests/ -q`
- 本地起服务：`cd ~/market-review-agent && python main.py`（读 .env，端口 8000）
- CI：GitHub Actions（.github/workflows/test.yml），push/PR 自动跑全量测试

## 架构

```
main.py                    FastAPI 层（鉴权/流式/错误兜底/消息历史提取）
                           + lifespan 定时推送循环(60s tick) + /charts 静态挂载
                           + 限流/每日配额(内存计数,429) + GET /v1/usage，版本 1.3.0
agent/orchestrator.py      编排层：detect_intent 规则路由 + 上下文追问继承
                           + Agent 工具循环(_agent_query) + 多pass审查(_critique_and_revise)
                           + 数字校验接入（非流式注入critique，流式log-only）
                           + 问责存档接入（非流式必档，流式结束后best-effort）
                           + 自选股意图与handler(_watchlist) + 知识库/以史为鉴注入
agent/validators.py        数字溯源校验（确定性非LLM）：extract_numbers /
                           find_unsourced_numbers / format_violations_for_critique
agent/archive.py           问责存档：JSONL(ARCHIVE_DIR缺省${DATA_DIR:-data}/archive)
                           save_analysis/load_records/update_record，原子回写+全局锁
agent/scorer.py            事后打分：方向提取(综合判断优先) + ±1% hit/miss/neutral
                           + apply_scores 写回存档（与 archive 同格式自实现读写）
agent/charts.py            零依赖 SVG 图表：generate_daily_charts(snapshot) 契约函数
                           → ${DATA_DIR:-data}/charts/YYYYMMDD/{indices,sectors,breadth}.svg
agent/push.py              定时推送：should_fire(上海时区/工作日/防重复) +
                           build_push_payload(文字+图表URL) + send_push(httpx webhook)
agent/watchlist.py         自选股：WATCHLIST_PATH(缺省${DATA_DIR:-data}/watchlist.json)
                           add/remove/list/format_watchlist_block，resolver可注入
agent/industry_kb.py       行业知识库：31申万行业档案(chain/drivers/indicators/leaders)
                           + industry_kb_data.json，format_kb_block 注入块(≤400字)
agent/history_lens.py      以史为鉴：get_history_note(sector,mode) 历史判断回顾注入块
                           + get_accuracy_summary 命中率汇总（读问责存档JSONL）
agent/tools.py             30 个 OpenAI function calling 工具（完整 JSON Schema）
agent/social_weibo.py      微博热搜直连（桌面端 ajax/side/hotSearch）
agent/social_douyin.py     抖音热榜直连（无签名公开端点，脆弱红利内置降级）
agent/social_bilibili.py   B站三件套：热搜/搜索(video·article)/评论(412 buvid3热身重试)
agent/social_zhihu.py      知乎热榜直连（topstory/hot-list）
agent/social_aggregator.py newsnow 聚合兜底（四源，仅直连失败降级用）
agent/social_store.py      社媒帖子 SQLite 持久化（social.db，hit_count 幂等累计）
agent/social_media.py      社媒门面：热榜聚合/搜索分发/股票关联提取/情感聚合
agent/social_guba.py       东财股吧：个股吧帖子列表+HTML详情正文点赞回填(个股舆情专用)
agent/sentiment.py         社交情绪层（BettaFish灵感）：东财人气榜+人气历史+涨跌停池
                           + 词典情感打分 + 情绪温度0-100 + 进程内当日缓存
agent/technical.py         确定性技术分析（daily_stock_analysis灵感）：MA七态/MACD/RSI/
                           量能五态/支撑压力/0-100评分 + 数据质量护栏 + 交易日历
agent/agent_audit.py       Agent工程三件套（Dexter灵感）：Scratchpad JSONL审计 +
                           ToolCallGuard软护栏 + microcompact上下文管理
agent/personas.py          投资人格框架库（Fincept灵感）：persona_defs.json四人格
                           （价值/成长/趋势/逆向）+ 框架渲染 + LLM产出归一化校验
agent/data_fetcher.py      数据采集：30+ 函数（Tushare/新浪MCP/yfinance/FRED）
                           + 板块extras当日缓存（进程内dict+锁）+ 景气度报告期直查
                           + 新闻注入净化 _sanitize_news_text（所有新闻源+pool双保险）
                           + 标题标点边界截断 _truncate_at_boundary + 源HTTP状态码日志
                           + Tushare无权限进程级短路 + pool保留content/summary/brief
agent/system_prompts.py    提示词：v6.0 合规 + 五维板块框架 + Agent/审查/新闻分析
                           + watchlist 自选股 + 知识库/以史为鉴指引 + 注入防护行
scripts/score_accountability.py  打分CLI：--days 5（唯一允许触网路径，lazy Tushare）
DEPLOY.md                  Railway 挂卷部署手册（Volume /data + DATA_DIR 环境变量）
eval/                      离线评估集：12 cases + rubric.py（复用validators）+ run_eval.py
tests/                     1718 个测试，全 mock 零网络（ARCHIVE_DIR/CHART_DIR 隔离到 /tmp）
```

## 核心能力（按开发顺序）

1. 市场复盘：27+ 路数据并行采集 → 复盘报告（当日缓存）
2. 板块五维深挖：趋势/估值水位(加权PE/PB+近一年分位)/资金博弈/景气度(业绩预告聚合)/催化风险 + 综合判断
3. 多轮对话：20 条历史 + 追问意图继承（"那半导体呢"→电子）
4. Agent 工具循环：复杂跨实体问题（"比较白酒和半导体"）模型自主调工具，≤8 轮，降级 _chat
5. 多 pass 生成：草稿→CRITIQUE 审查（数字出处/禁用词/越界/AI腔，含确定性数字校验注入）→修正（≥500字启用）
6. 新闻系统：五源聚合（去重后约 176 条/48h）+ 重要性评分截断 + 新闻分析模式（"分析新闻影响"）
   + 板块查询默认在确定性清单后追加 LLM 解读 + 头部覆盖描述按实际数据生成
7. 数字校验层：validators 确定性溯源（容差±0.05/相对0.5%，亿/万归一，日期/小整数豁免）
8. eval 评估集：12 case 正反例，`/usr/local/bin/python3 eval/run_eval.py` 可独立运行
9. 问责系统：三路径产出落 JSONL 存档 → CLI 按方向判断 vs 后市实际涨跌打 hit/miss/neutral
10. 可视化+推送：SVG 图表（/charts 静态服务）+ 工作日 PUSH_TIME(默认15:40上海) webhook 推送
11. 个性化自选股：加/删/列表/自选股复盘（"加自选茅台"/"自选股复盘"），上限 50 只
12. 行业知识库+以史为鉴：深挖注入行业档案（背景知识）+ 历史判断回顾（自省克制）
13. 工程化：新闻注入净化（〔已过滤〕）+ 限流30/分+日配额500（/v1/usage）+ 裸except日志清零

## 数据纪律（血泪教训）

- daily/index_daily 的 amount 单位是**千元，÷1e5=亿**（曾错用 1e7 差 100 倍）
- moneyflow 金额单位万元，÷10000=亿；daily_basic 的 total_mv 单位万元；rzye 单位元 ÷1e8=亿
- Tushare 按分钟限频，单次市场复盘 62 次调用；板块 extras 已当日缓存（同板块重复问 0 增量），
  景气度改为报告期直查（1~2 次，失败降级按周 17 次采样）
- prompt 红线：每个数字必须有数据块出处，缺的写"数据未覆盖"，训练知识不算数
- 语言红线：禁用词清单（护城河/飞轮/赋能/格局/综上所述等）+ 禁排比升华 + 长短句交错

## 路线图进度

```
✅ 第一波: 多轮对话 + CI
✅ 第二波: 真 Agent 架构 + 多 pass 生成
✅ 新闻系统扩容 + 分析层（计划外插入，已完成）
✅ 第三波: eval 评估集 + 输出后数字校验层（含 Tushare 优化 + 新闻空括号修复）
✅ 第四波: 自我问责系统（分析存档+事后打分）
✅ 第五波: 可视化 + 主动推送（定时复盘）
✅ 第六波: 工程化（新闻注入防护/配额/日志覆盖）
✅ 第七波: 个性化(自选股) + 行业知识库 + 以史为鉴
🔧 新闻模式三问题修复（计划外插入，已完成）
🔧 输出卫生 + MCP 兜底修复（计划外插入，已完成）
```

路线图七波全部完成。研报库 v1 + v2 全文 RAG + 每日自动化均已落地（见下文专项记录）。后续方向：全量全文回填 / 生产挂卷后问责数据积累 / LLM judge 实现。

**新闻模式三问题修复（2026-07-22，后续记录）**

- 现象：① 条目截断——新闻标题被拦腰切断，出现半句话；② 五源出货率低——
  部分查询只有 1~2 个源出条目；③ 板块新闻查询只回确定性清单、无解读
  （此前仅全市场查询含触发词时才走 LLM 分析）。
- 根因（逐源实测结论）：
  - 新浪财经：接口 date 参数已失效（传不同日期返回同一当前滚动列表）；且条目
    time 此前直接用查询日期填充——当天新闻被错标到回溯的历史日期（时间错标）；
  - 智研 MCP：大量条目只返回 content 不带 title，旧逻辑 content[:80] 硬切当标题；
    且接口单页上限 20 条，旧代码未翻页；
  - Tushare：token 无 news 接口权限（积分不足），新闻池逐日回溯每次白耗配额；
  - 东方财富/财联社：本地正常、Railway 生产空返回，疑似海外 IP 受限（未坐实）；
  - 修复已给各源加 HTTP 状态码 warning 日志，生产出货情况可在 Railway View logs 查证。
- 修复方案要点：
  - 抓取层（data_fetcher）：_truncate_at_boundary 标点边界截断替代硬切——东财/新浪/
    智研/财联社标题完整保留，仅超长时在句末/分句标点处截断加『…』；新浪 time 一律
    取真实 ctime（date_str 仅作兜底，接口恢复按日查询则回溯自动生效）；智研自动翻页
    （页间限速）+ 跨页标题去重 + content 字段保留；Tushare 权限错误置进程级 denied
    标记，后续调用（含逐日回溯）整体短路不再耗配额；东财/财联社 HTTP 非 200 记
    warning 并安全降级；pool 统一条目保留 content/summary/brief 正文字段（供下游行业
    关键词匹配）；板块查询加深抓取（东财加翻第 2 页、财联社拉满约 100 条、智研翻 3 页）。
  - 编排层（orchestrator）：头部诚实化——覆盖描述按实际数据生成（当日/实际日期/
    起止日期），不再硬写「48小时覆盖」，来源统计只列实际有贡献的源；板块查询默认在
    确定性清单后追加 LLM 解读段（清单本体绝不经 LLM 改写，解读失败降级只返清单）；
    防御性展示修复——title 是 content/summary/brief 裸前缀时，改用摘要按句子边界
    截断（≤200 字）展示完整句。
- 测试：793 passed 全绿（tests/test_news.py +18 用例；新增 tests/test_news_fetch_layer.py
  34 用例；全部 mock 零网络）。

**输出卫生 + MCP 兜底修复（2026-07-22，后续记录）**

- 现象：① 新闻条目【来源】标签冗余——头部已有「来源：xxxN条」统计行，
  每条目再标【新浪】【东方财富】浪费字符且视觉嘈杂；② Markdown 符号泄漏——
  解读/报告中偶发 # 标题与 * 加粗符号直接到达用户（流式路径此前完全未清洗）；
  ③ 「查询鲟龙科技近期的财务指标」类 MCP 查询跑满工具轮次后，把原始 JSON
  dump 直接甩给用户。
- 根因：
  - _clean_markdown 只挂在 3 个用户可见出口（流式路径、个股/期货/基金/
    自选股等确定性拼装出口均未清洗），且旧实现不清行内 *斜体* 与残留孤立符号；
  - _generic_mcp 循环跑满 max_rounds 后直接拼接 all_results 原始 JSON 返回，
    无任何综合；模型对 {"data":[],"status":{"code":11,"msg":"Input error"}}
    或全 "--" 占位数据识别不出失败，反复重试同一工具直至跑满轮次；
  - MCP 错误与空占位无识别：_mcp_call HTTP 非 200 时大段原始错误页直接进
    上下文，无 {"error":...} 归一化；status.code 非成功、全 "--" 占位均无判定；
  - main.py 流式加载提示语「正在采集…约需N秒」无差别发送，普通快查询
    （闲聊/新闻/个股）也弹，误导等待预期。
- 修复方案要点：
  - 抓取层（data_fetcher，工程师B）：新增 compact_mcp_result / is_mcp_error /
    mcp_error_brief 三助手（签名固定）——递归剔除占位字段（0/False 不误判）、
    句子边界截断、HTTP 错误与 status.code 非成功统一识别、单行中文错误摘要；
    _mcp_call 的 initialize 与 tools/call 两跳 HTTP 非 200 归一化为
    {"error":"HTTP_<code>"}（不抛异常、不回传原始错误页；mock 兼容——
    status_code 非真实 int 时不做非 200 判定，不误伤既有测试）。
  - 编排层（orchestrator，工程师A）：_clean_markdown 强化（行内 *斜体*、
    1-6 个 # 标题、末尾 _strip_md_symbols 符号级兜底，保证 * 与 # 零到达用户），
    挂到所有用户可见出口——含流式逐 chunk 符号级清洗（无状态、跨 chunk 安全，
    干净 chunk 原样透传）与个股/期货/基金/自选股等确定性拼装出口；新闻条目
    去【来源】标签改「[时间] 标题」，来源归属由头部统计行统一承载；_generic_mcp
    跑满轮次后改走 _mcp_final_synthesis——把所有已收集工具结果拼入上下文做
    一次无 tools 最终综合调用，让 LLM 用人话总结，综合失败/为空返回优雅降级
    提示，任何路径绝不 dump 原始 JSON；工具结果喂回模型前经
    _mcp_result_for_prompt 压缩（错误返回压成单行摘要，正常返回经
    compact_mcp_result 压缩防上下文挤爆）；三助手防御式导入，未就绪退回
    现状行为，绝不硬依赖。
  - 提示词（system_prompts）：六个提示词统一追加「输出必须是纯文本：禁止
    使用任何Markdown语法——不用 # 和 * 号，用【】标记小节标题」。
  - 接入层（main.py）：加载提示语分级——仅 detect_intent 判定为
    market_review / sector_deep_dive（重数据采集意图）才发「正在采集…约需N秒」，
    其他意图直接出答案；意图判定异常按不发提示处理（fail-safe）。
- 测试：895 passed 全绿（tests/test_news.py 断言同步；新增
  tests/test_mcp_result_tools.py 41 用例、tests/test_output_hygiene.py
  61 用例；全部 mock 零网络）。

### 2026-07-22 追加：_generic_mcp 财报参数枚举丢失根因修复

- 根因（实测确认）：function calling 链路丢弃 MCP 工具 inputSchema 的
  参数文档。data_fetcher.get_mcp_tools 只留 name/desc(截200)/params(纯参数名)，
  orchestrator._generic_mcp 构建 ds_tools 时参数描述=参数名本身；模型调
  cnFinanceReportsFull 不知道 source 只能填 lrb/fzb/llb/gjzb/zxzb，瞎填
  "1"/"sina" → code=11 Input error → 用户看到「财务指标暂未获取」全占位符。
  同因曾致 hkFinanceReportsByIndex 报错。工具本身无恙（正确参数实测返回
  21KB 完整财报）。
- 方案：
  - get_mcp_tools 缓存条目新增 schema 键（properties 保留 type/
    description≤300字/enum + required 列表，剔除不在 properties 中的
    required 项）；name/desc/params 旧键不变兼容其他调用方。
  - _generic_mcp ds_tools 改用真实 schema 构建 parameters（含 required）；
    无 schema 键或 properties 为空的旧缓存条目防御式降级回原 params 逻辑。
  - _GENERIC_MCP_SYSTEM_PROMPT 补一句：财报类工具严格按枚举填参，
    A股财务指标推荐链路 cnFinanceReportDateList 拿报告期 →
    cnFinanceReportsFull(source=gjzb 等枚举) 取数。
- 实测：真实链路跑「查询西麦食品近期的财务指标」——模型调用序列
  globalStockSearchSymbols → cnFinanceReportDateList →
  cnFinanceReportsFull(source="gjzb")×多报告期（+利润表 lrb），返回含
  营业总收入 18.96亿(+20.16%)、归母净利 1.33亿(+15.36%) 等真实数据。
- 测试：902 passed 全绿（新增 tests/test_mcp_tool_schema.py 7 用例：
  schema 保留/300字截断/无 inputSchema 兜底/真实 schema 构建/
  旧格式降级×2/端到端枚举填参；mock tools/list 零网络）。

### 2026-07-22 追加：报告期选择纠偏（日期注入 + 诚实约束）

- 现象：问「近期财务指标」，模型只取 2024年报+2025一季报等早期报告期，
  并谎称 2025年报/2026一季报「返回占位符无数据」（实测智研侧数据完整）。
- 根因：模型训练记忆的时间感滞后，把已披露的最新年报当未来数据跳过；
  且对未查询过的报告期随口声称无数据。
- 方案：_GENERIC_MCP_SYSTEM_PROMPT 新增报告期选择规则（必须覆盖最新年报+
  最新季报、已取各期都要呈现）与诚实约束（未实际查询的报告期不得声称无数据）；
  新增 _generic_mcp_system_prompt() 助手在 system 提示词注入当天日期，
  _generic_mcp 与 _mcp_final_synthesis 两处统一使用。
- 实测（真实 MCP+DeepSeek）：取数报告期变为 2025年报/2026一季报/2025三季报，
  回答突出最新期且无虚报。测试 902 passed 全绿。

### 2026-07-22 追加：港股财报链路与数字保真修复

- 现象：问鲟龙科技（港股 06715）财务，先说「字段灰底=数据缺失」投降；
  链路修复后回答的营收/EPS 数字仍全系编造（与智研真实值无一吻合）。
- 根因（三层）：① 财报数据 item_display="灰底" 是前端行样式标记，模型误读为
  占位缺失；② 模型用 A 股工具查港股、瞎猜 hkFinanceReportsByIndex 的 field；
  ③ 工具结果喂回模型截断 3000 字符，34KB 财报载荷只剩零头，模型看不到
  真实数值遂编造。
- 方案：compact_mcp_result 剔除纯样式键（item_display 等）+ 指标条目瘦身
  （只留 item_title/item_value/item_tongbi/item_field，19.5KB→9.7KB）；
  喂回上限 3000→12000（_MCP_TOOL_FEED_MAX_CHARS），综合存档 500→4000；
  提示词新增港股链路指引（hk_finance_all 参数用法、勿用 A 股工具查港股）
  与数字保真约束（逐项抄 item_value，宁少勿假）。
- 实测：回答数字与智研真实值逐项吻合（2024 营收 7.119亿+12.15%、
  2025 EPS 4.484 等）。测试 902 passed 全绿。
**研报库 v1（2026-07-22，新工作线落地）**

能力：用户问行业/个股时，Agent 循环自主调用研报工具回答券商观点——观点聚合
（评级分布+目标价区间+EPS 均值）、共识与分歧、逐篇「券商+日期+评级」溯源。
施工图纸：docs/RESEARCH_LIB_DESIGN.md（含全部数据源实测规格，v2/v3 分期见原文）。

- scripts/report_crawler.py 多源爬虫：ReportSource 抽象 + 东财 reportapi(主)/
  慧博列表/证券之星/洞见 四源；限速 ≤1req/s+随机抖动（可注入）、GBK/GB2312
  显式编码、单源失败记 warning 不拖垮整体；http_get/upsert/sleep 均可注入；
  CLI：--days N（默认 1）/--sources/--rate/--db-path/--verbose
- agent/report_library.py 存储+检索层：init_db/upsert_reports/search_reports/
  rating_summary；SQLite schema=图纸+source 列（老库自动 ALTER 补列）；
  upsert 按 info_code 去重（东财用 infoCode 原值，他源 sha1(title+org+date)[:16]），
  冲突合并=空缺回填+源优先级（eastmoney>stockstar>djyanbao>sina>hibor）；
  空库/无库返回 total=0 合法结构绝不抛出
- agent/tools.py 新增 search_research_reports / get_rating_summary：经
  _REPORT_IMPL 映射 + _get_report_library() 惰性解析接入 execute_tool，
  既有 21 工具零改动；stock_code 兼容 sh600519/600519（去前缀归一）
- agent/system_prompts.py AGENT_QUERY_PROMPT 插入「## 研报引用规范」一节
  （标注券商+日期+评级 / 评级目标价EPS只用工具返回 / 先共识后分歧 /
  未覆盖明说「研报库暂未覆盖」不得编造）
- 测试：新增 67 用例（library 21 + crawler 25 + tools 21），全 mock 零网络，
  全量 969 passed 全绿
- 回填实测（2026-07-22，90 天）：总量 9961 篇——东财 9182 / 慧博 946(去重后 237)/
  证券之星 364 / 洞见 198；评级非空 5401、个股研报 3055、目标价非空 236
- E2E 实测：工具层查询约 3ms（≪3s）；真实 LLM 问「券商最近怎么看贵州茅台」
  30s 出稿——评级分布+目标价+逐篇溯源+局限说明齐备
- v2 已落地（见下节）；encode_url 仍入库备用

**研报库 v2 全文 RAG + 每日自动化（2026-07-22，同日落地）**

能力：search_report_content(query, stock_code, industry, days, top_k) 语义检索
研报正文段落（bge-small-zh-v1.5 中文向量，512 维本地模型），回答可引用原文观点。

- scripts/report_fulltext.py 全文层：东财 PDF（infoCode 直链
  H3_{infoCode}_1.pdf——⚠️ 不是 encodeUrl：encodeUrl 含 / 时原样拼 404、
  quote 后 Tomcat 400 双死路，2026-07-22 实测定案）+ 新浪网页全文
  （vReport_Show 详情页分节正文，GB2312；标题匹配含短标题三约束）；
  EO_Bot 挑战求解留作兜底；表 report_fulltext(info_code PK, source,
  fulltext, sections_json, fetched_at)；PDF 内存解析用完即弃；
  CLI --days/--limit/--db-path/--verbose
- agent/report_vectors.py：chunk_report(≤500字/块重叠50) + Embedder 协议
  （FakeEmbedder 测试零网络 / BgeEmbedder 构造时惰性导入 sentence_transformers）
  + numpy 暴力余弦（表 report_chunks/report_embeddings/vector_meta，与
  reports 同库）+ build_index（幂等跳过/force 重建）+ search_vectors
  （JOIN reports 过滤，异常降级返 note 绝不抛）
- 模型获取：huggingface.co 与 hf-mirror 本机实测均不可达 → 走 ModelScope：
  snapshot_download('BAAI/bge-small-zh-v1.5')，REPORT_EMBED_MODEL 环境变量
  指向本地路径（BgeEmbedder 解析序：显式参数 > 该 env > HF 默认 ID）；
  依赖安装：pip install -i https://pypi.org/simple -r requirements-rag.txt
  （默认镜像源实测缺 sentence-transformers；含 torch 约 2GB + modelscope）
- agent/tools.py：search_report_content 经 _REPORT_VEC_IMPL 接入（查表序
  _REPORT_VEC_IMPL → _REPORT_IMPL → _IMPL）；AGENT_QUERY_PROMPT 研报引用
  规范追加第 5/6 条（正文引用标注券商+日期+标题；只用工具返回段落）
- scripts/daily_report_update.sh 每日增量统一入口（PYTHON_BIN/天数参数/
  REPORTS_DB_PATH/WITH_FULLTEXT=1 可选全文链；macOS bash 3.2 兼容）
- .github/workflows/report_crawler.yml：cron "23 13 * * *"（北京 21:23）+
  workflow_dispatch + concurrency 防重叠；年周 cache 滚动 reports.db +
  artifact 保留 7 天 + schedule 失败自动建 issue（label report-crawler）；
  零 secret（数据源全免费）
- 测试：+89 用例（fulltext 38 + vectors 21 + content_tool 20 + workflow 10，
  含 test_report_tools 连锁改 23→24），全量 1058 passed 全绿
- 真实 E2E（2026-07-22）：全文抓取 8 候选 7 入库（东财 PDF 7 篇，1 篇新浪
  未收录正确跳过）；bge 建索引 7 篇 160 块；工具层语义检索「AI 算力产业链
  观点」命中算力过剩/资本开支相关段落（score 0.62+）排序合理

**开源借鉴移植（2026-07-22 晚，新工作线落地）**

能力：借鉴四个开源项目的精华设计（BettaFish 社交情绪 / FinceptTerminal 投资人格 /
daily_stock_analysis 技术仪表盘 / Dexter Agent 工程），全部自写代码（License 红线：
BettaFish=GPL、Fincept=AGPL+商业双许可，只移植思路不复制）。工具 24→28。

- agent/sentiment.py（BettaFish 灵感，合规数据源替代其社媒爬虫）：东财人气榜
  （getAllCurrentList，POST，data 直为 list、无 name——名称经 push2 ulist 批量回填，
  失败留空串降级）+ 个股人气历史（getHisList，⚠️ 参数是 srcSecurityCode 且必须带
  SH/SZ/BJ 市场前缀，stockCode/entityId 均报 -1，2026-07-22 实测定案）+ 涨跌停池
  （push2ex 涨停/跌停/炸板三池）+ 中文金融利好/利空词典确定性打分（scorer 可注入）
  + 情绪温度 0-100（涨停/跌停/炸板率/最高连板加权公式，模块常量带注释）；进程内当日缓存
- agent/technical.py（daily_stock_analysis 灵感）：纯函数确定性技术分析——
  MA 七态/乖离率/量能五态/MACD(12,26,9)/RSI6·12/支撑压力/0-100 评分（权重常量可覆盖），
  + verdict_from_score 评分带→结论 + 数据质量护栏（insufficient→cap0.3/stale→0.5/
  no_volume→0.7 取最严）+ is_trade_day（calendar 注入优先，否则周一~五启发式）
- agent/agent_audit.py（Dexter 灵感）：Scratchpad JSONL 审计（每次 Agent 查询一文件，
  ${DATA_DIR:-data}/scratchpad，best-effort 吞错）+ ToolCallGuard 软护栏（单工具
  调用>3 次或参数相似度≥0.7 时生成中文警告注入工具结果尾部，绝不阻断）
  + microcompact（tool 消息>8 条或总字符>80k 时最旧的替换为占位符，保留最近 4 条，
  assistant tool_calls 配对完好）
- agent/personas.py + persona_defs.json（Fincept 灵感）：配置驱动的四个 A 股方法论
  框架——value_cn/growth_cn/trend_cn/contrarian_cn（instructions/权重/阈值/规则/输出
  schema 全原创中文）；validate_persona_output 归一化（signal 枚举外→观望，
  confidence 钳≤0.9，violations 记录）
- tools.py 新增 4 工具：get_market_sentiment（市场情绪快照）/ get_stock_sentiment
  （个股人气+新闻情感）/ get_technical_analysis（技术仪表盘，Tushare daily 取数+
  trade_cal 判 stale）/ analyze_with_persona（人格框架+指引）；走 _ANALYSIS_IMPL
  第四查表 + _lazy_import_module 惰性解析，绝不抛异常
- orchestrator._agent_query 接入审计三件套（钩子经 _audit_safe 包裹，异常零影响主循环）
- system_prompts AGENT_QUERY_PROMPT 追加三节：情绪引用规范 / 技术分析纪律
  （置信度不得突破 confidence_cap，guardrail_reason 原文透传）/ 投资人格框架用法
- 测试：+319 用例（sentiment 92/technical 77/agent_audit 52/personas 55/集成 43），
  全量 1377 passed 全绿（test_report_tools 等 3 处工具总数断言按惯例 24→28）
- 真实 E2E（2026-07-22）：涨跌停池 47 涨停/36 炸板/最高连板 5；人气榜 top10 含名称
  （紫光股份 1/德明利 2）；茅台人气 44（30 日均 26.9，趋势下降）；情绪温度 68.6 活跃；
  技术分析茅台 as_of 2026-07-22 收 1305.0 弱多头 MACD 多头 score 60.25→偏多，
  confidence_cap 1.0 无护栏触发

**社媒舆情爬取 v1（2026-07-22 晚，微舆 BettaFish 式能力落地）**

能力：微博/知乎/抖音/B站 四平台热榜直连 + B站搜索/评论 + 情感聚合 + 股票关联提取。
BettaFish 的 MediaCrawler（Playwright+登录态）未采用——v1 全部走**无登录公开端点**
（端点 2026-07-22 实测定案，规格见 workspace/research/social_endpoints_recon.md）。

- agent/social_weibo.py：桌面热搜 weibo.com/ajax/side/hotSearch（data.realtime[]，
  url 拼 s.weibo.com 搜索页；⚠️ 移动端全线 432 Sina Visitor 判死未实现）
- agent/social_douyin.py：热榜 aweme/v1/web/hot/search/list/（无签名直连红利，
  脆弱——结构漂移即降级空列表+warning；搜索/评论需 X-Bogus 判死）
- agent/social_bilibili.py（主力源，三件套全通）：热搜 search/square（+可选 popular
  六维指标）+ 搜索 search/type（video/article，<em> 标签清洗）+ 评论 x/v2/reply
  plain 版（wbi 变体 -403 判死）；⚠️ 412 风控——_warmup 拿 buvid3 后自动重试恰好一次
- agent/social_zhihu.py：热榜 api.zhihu.com/topstory/hot-list（billboard 403 /
  search_v3 400 判死，知乎搜索缺席）
- agent/social_aggregator.py：newsnow 聚合兜底（weibo/zhihu/douyin/bilibili 四源，
  有缓存无 SLA，仅直连失败降级用；小红书源实测 500 不存在）
- agent/social_store.py：SQLite 持久化（${DATA_DIR:-data}/social.db，SOCIAL_DB_PATH
  可覆盖；social_posts 表 platform+post_id 主键幂等 upsert，hit_count 累计）
- agent/social_media.py 门面：惰性 importlib+getattr 能力探测（不 import 平台函数），
  get_hot_all（直连→聚合器兜底→去重→落盘）/ search_all（仅有搜索能力平台分发，
  缺席进 notes）/ extract_stock_mentions（6位代码正则+价格语境排除+自选股名称匹配）
  / aggregate_buzz（复用 sentiment 词典打分）
- tools.py 28→30：get_social_hot（全平台热榜+buzz+股票关联，posts 瘦身封顶）/
  search_social_media（B站搜索，with_comments=true 拉前 3 条视频评论合并打分）；
  prompts 追加「## 社媒舆情引用规范」（标注平台+日期/情绪仅辅助/能力边界诚实）
- 小红书 v1 整体缺席（无可用无登录端点，x-s 签名超出合规边界），工具层中文 note 说明
- 测试：+232 用例（weibo 25/douyin 27/bilibili 39/zhihu 20/aggregator 20/store 20/
  门面 47/集成 34），全量 1636 passed 全绿（工具总数断言按惯例 28→30 共 3 处）
- 真实 E2E（2026-07-22）：四平台直连各 5 条热榜（微博 top「别再给AI乱传文件了」
  热度 254 万）；搜「A股」B站 5 帖+9 评论（播放/评论/点赞真实指标）；情感分布与
  降级路径（小红书/空关键词/搜索缺席平台 notes）全部按设计工作

**东财股吧接入（2026-07-22 晚，社媒舆情 v1.5）**

能力：个股股吧帖子列表+正文+点赞（散户情绪第一现场）。端点 2026-07-22 实测定案
（workspace/research/guba_endpoints_recon.md）。

- agent/social_guba.py：帖子列表 POST gbapi Articlelist（⚠️ 魔法参数
  deviceid=Wap10.0.0.1+version=200 必带，缺了 rc=0 空数据伪装繁忙）+ 详情走 HTML
  SSR 内嵌 var post_article={...} 花括号配平提取（点赞数唯一来源；详情 API 被 WAF
  403 判死）+ enrich_posts 按评论数 top_n 回填正文/点赞；评论/全站热榜(AES 密文)/
  吧内搜索全部判死未实现
- social_media.get_guba_buzz：个股舆情专用通道（不进 PLATFORM_MODULES），
  列表→富化→aggregate_buzz 情感打分；tools.py get_stock_sentiment 追加 guba 块
  （posts+buzz，股吧失败绝不影响主返回）；prompts 社媒节补股吧引用规范
- 测试：+82 用例（模块 52+集成 30），全量 1718 passed 全绿
- 真实 E2E（2026-07-22）：茅台吧 10 帖（阅读/评论/转发/点赞真实，朱少醒调出茅台帖
  阅读 1.1 万/评论 66/点赞 28 含正文），情感 利好2/利空1/中性7

## 已知问题

- 社媒端点全部为无登录公开接口，平台风控/结构变动即降级：抖音「无签名直连」是脆弱
  红利随时可能加签；B站 412 靠 buvid3 热身自愈（双 412 放弃）；微博移动端/知乎搜索/
  小红书已判死，v2 需 Cookie 池或 headless 方案并单独评估合规
- newsnow 聚合器为第三方公共服务，有缓存延迟、无 SLA，仅作兜底
- 社媒情感为词典弱信号；extract_stock_mentions 代码正则不校验真实性（价格语境已排除）
- social.db 随抓取增长无清理策略；query_posts 的 LIKE 未转义 %/_（调用方受信）
- 股吧列表无点赞字段（仅详情页有，enrich 只回填 top_n 条）；评论数仅为计数（评论
  内容端点全灭）；详情页点赞为抓取时点快照
- 人气榜接口不含股票名称：名称经 push2 ulist 批量回填，本机系统代理异常时名称留空串
  优雅降级（排名/代码不受影响）；生产无代理环境实测正常
- 个股人气历史必须传 srcSecurityCode+市场前缀；BJ（北交所）前缀规则按 4/8/920 推导，
  未经实网验证（东财 sc 格式以实测为准）
- 情绪温度/词典情感为确定性弱信号，反讽/新词无能为力；scorer 注入点已预留（可换 LLM 打分）
- 技术分析 RSI 用简单平均法（Cutler's），与 Wilder 平滑口径数值有差异（趋势方向一致）；
  无交易日历时 stale 判定退化为周一~五启发式，调休工作日可能误判
- ToolCallGuard 相似度为字符级 SequenceMatcher，语义相近字面不同的重复查询不敏感（低成本有意设计）
- Scratchpad 每会话一文件无轮转清理，长期运行需运维侧定期归档
- 东财 PDF 的 EO_Bot 挑战兜底求解未端到端复验（infoCode 直链实测直 200 未触发）；
  若未来直链也触发挑战且 cookie 重载失效，东财全文通道退化（warning 可观测，新浪兜底）
- pdf.dfcfw.com 调试期间对该 IP 触发过 400 级限流（约 10 分钟自愈）；每日低频无碍
- 部分 PDF 分节退化为单节「正文」（券商模板节名未覆盖），只影响分块粒度不影响检索
- 向量检索为 numpy 暴力余弦，万块级毫秒；10 万+ 块再考虑 sqlite-vec/分桶
- 全量 90 天全文回填未跑（限速下载约 1~3 小时）；当前索引为近 3 天小批量验证
- GH Actions 工作流未经真实 GitHub 环境验证（YAML 结构已断言）；首次触发后核对
  issue 告警链路与 cache 命中日志；年周缓存键存在周内回滚窗口（upsert 幂等可控）
- 研报跨源同报告可能各存一条（东财 infoCode vs 他源 sha1 合成码，标题实测一致）；
  v2 可加标题+日期指纹辅助索引做跨源合并
- 洞见 API 匿名单查询上限约 250 条且列表非日期序，增量出货率低（定位=个股索引补源）
- 东财 qType=3（宏观）实测近 30 天恒 hits=0 疑似下线，保留遍历每轮多 1 次请求
- 慧博/证券之星列表记录 industry 留空（页面无结构化行业字段），避免污染行业检索
- Tushare news 接口无权限（积分不足），新闻池 tushare 源恒为空，已安全降级
  （进程级 denied 标记短路：首次命中权限错误后本进程不再调用，不白耗配额）
- 新浪智研 3 个接口权限不足 + swSymbolList 服务端 bug（新浪侧，不可修）
- 板块 extras 已当日缓存（进程内）；重启进程后同板块首问仍需 1 轮采集（已大幅降耗）
- 流式路径跳过多 pass 审查（保延迟），非流式才有；流式数字校验为 log-only
- 成分股行业分类可能过期（Tushare 数据源问题）
- Railway 挂卷需手动操作（DEPLOY.md 三步：挂 /data → 设 DATA_DIR=/data → 重部署验证）；
  未挂卷时四模块启动会 warning『运行在 Railway 但未挂卷』
- 限流/配额为内存计数，仅单 worker 有效；多实例需外置共享计数（main.py 注释已注明）
- 流式存档为 best-effort：客户端中途断开则该次不落档
- 自选股缺省 resolver 取搜索首条，重名股票可能解析偏差（handler 已回显 code+name 供确认）
- eval 的 LLM judge 仅接口预留（EVAL_LLM=1），未实现
- 推送仅接通用 JSON webhook（PUSH_WEBHOOK_URL 未配置时只生成 payload 记 log）
- MARKET_REVIEW_PROMPT 输出模板的 ``` 代码块历史遗留未闭合（不影响使用，未动）

## 环境变量全表

必填：`DEEPSEEK_API_KEY`、`AGENT_API_KEY`
推荐：`TUSHARE_TOKEN`、`FINNHUB_API_KEY`、`FRED_API_KEY`、`SINA_MCP_TOKEN`
持久化：`DATA_DIR`（根目录，缺省 data；Railway 挂卷设 /data）、
  `ARCHIVE_DIR`/`CHART_DIR`/`WATCHLIST_PATH`/`REPORTS_DB_PATH`（显式设置优先于
  DATA_DIR 推导；研报库缺省 ${DATA_DIR:-data}/reports.db）
研报库v2：`REPORT_EMBED_MODEL`（bge 模型本地路径或 HF ID，缺省 HF 默认；
  国内 huggingface.co 不可达时用 ModelScope 下载后设本地路径）
开源移植线（均可选）：`SCRATCHPAD_DIR`（Agent 审计 JSONL 目录，缺省
  ${DATA_DIR:-data}/scratchpad）、`PERSONA_DEFS_PATH`（人格定义文件覆盖路径，
  缺省 agent/persona_defs.json）、`SOCIAL_DB_PATH`（社媒帖子库，缺省
  ${DATA_DIR:-data}/social.db）；情绪/社媒端点免登录零新增 Key
推送：`PUSH_WEBHOOK_URL`（未配置只生成不发送）、`PUSH_TIME`（默认 15:40 上海，工作日）
配额：`RATE_LIMIT_PER_MIN`（默认 30）、`QUOTA_DAILY`（默认 500，上海时区自然日）
