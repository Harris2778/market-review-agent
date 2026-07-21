# 项目交接文档（2026-07-22 更新，第六/七波 + 新闻模式三问题修复 + 输出卫生/MCP兜底修复，路线图全部完成）

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
agent/tools.py             21 个 OpenAI function calling 工具（完整 JSON Schema）
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
tests/                     895 个测试，全 mock 零网络（ARCHIVE_DIR/CHART_DIR 隔离到 /tmp）
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

路线图七波全部完成。后续方向：研报库工作线（另线进行）/ 生产挂卷后问责数据积累 / LLM judge 实现。

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

## 已知问题

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
  `ARCHIVE_DIR`/`CHART_DIR`/`WATCHLIST_PATH`（显式设置优先于 DATA_DIR 推导）
推送：`PUSH_WEBHOOK_URL`（未配置只生成不发送）、`PUSH_TIME`（默认 15:40 上海，工作日）
配额：`RATE_LIMIT_PER_MIN`（默认 30）、`QUOTA_DAILY`（默认 500，上海时区自然日）
