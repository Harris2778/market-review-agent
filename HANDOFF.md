# 项目交接文档（2026-07-22 更新，第四/五波更新）

> 本文档记录当前开发状态，新会话/新协作者从这里开始读。

## 项目简介

A 股金融分析智能体。FastAPI 提供 OpenAI 兼容接口（`/v1/chat/completions`），
DeepSeek（deepseek-chat）生成分析，数据源：Tushare Pro（200元/年）+ 新浪智研 MCP（75 工具）
+ yfinance + FRED + Finnhub。部署：Railway（从 GitHub main 分支自动部署）。

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
agent/orchestrator.py      编排层：detect_intent 规则路由 + 上下文追问继承
                           + Agent 工具循环(_agent_query) + 多pass审查(_critique_and_revise)
                           + 数字校验接入（非流式注入critique，流式log-only）
                           + 问责存档接入（非流式必档，流式结束后best-effort）
agent/validators.py        数字溯源校验（确定性非LLM）：extract_numbers /
                           find_unsourced_numbers / format_violations_for_critique
agent/archive.py           问责存档：JSONL(ARCHIVE_DIR默认data/archive/) 按天分文件
                           save_analysis/load_records/update_record，原子回写+全局锁
agent/scorer.py            事后打分：方向提取(综合判断优先) + ±1% hit/miss/neutral
                           + apply_scores 写回存档（与 archive 同格式自实现读写）
agent/charts.py            零依赖 SVG 图表：generate_daily_charts(snapshot) 契约函数
                           → charts/YYYYMMDD/{indices,sectors,breadth}.svg 正红负绿
agent/push.py              定时推送：should_fire(上海时区/工作日/防重复) +
                           build_push_payload(文字+图表URL) + send_push(httpx webhook)
agent/tools.py             21 个 OpenAI function calling 工具（完整 JSON Schema）
agent/data_fetcher.py      数据采集：30+ 函数（Tushare/新浪MCP/yfinance/FRED）
                           + 板块extras当日缓存（进程内dict+锁）+ 景气度报告期直查
agent/system_prompts.py    提示词：v6.0 合规 + 五维板块框架 + Agent/审查/新闻分析 prompt
scripts/score_accountability.py  打分CLI：--days 5（唯一允许触网路径，lazy Tushare）
eval/                      离线评估集：12 cases + rubric.py（复用validators）+ run_eval.py
tests/                     399 个测试，全 mock 零网络（ARCHIVE_DIR/CHART_DIR 隔离到 /tmp）
```

## 核心能力（按开发顺序）

1. 市场复盘：27+ 路数据并行采集 → 复盘报告（当日缓存）
2. 板块五维深挖：趋势/估值水位(加权PE/PB+近一年分位)/资金博弈/景气度(业绩预告聚合)/催化风险 + 综合判断
3. 多轮对话：20 条历史 + 追问意图继承（"那半导体呢"→电子）
4. Agent 工具循环：复杂跨实体问题（"比较白酒和半导体"）模型自主调工具，≤8 轮，降级 _chat
5. 多 pass 生成：草稿→CRITIQUE 审查（数字出处/禁用词/越界/AI腔，含确定性数字校验注入）→修正（≥500字启用）
6. 新闻系统：五源聚合（去重后约 176 条/48h）+ 重要性评分截断 + 新闻分析模式（"分析新闻影响"）
7. 数字校验层：validators 确定性溯源（容差±0.05/相对0.5%，亿/万归一，日期/小整数豁免）
8. eval 评估集：12 case 正反例，`/usr/local/bin/python3 eval/run_eval.py` 可独立运行
9. 问责系统：三路径产出落 JSONL 存档 → CLI 按方向判断 vs 后市实际涨跌打 hit/miss/neutral
10. 可视化+推送：SVG 图表（/charts 静态服务）+ 工作日 PUSH_TIME(默认15:40上海) webhook 推送

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
⬜ 第六波: 工程化（新闻注入防护/配额/日志覆盖）   ← 下一个
⬜ 第七波: 个性化(自选股) + 行业知识库 + 以史为鉴
```

## 已知问题

- Tushare news 接口无权限（积分不足），新闻池 tushare 源恒为空，已安全降级
- 新浪智研 3 个接口权限不足 + swSymbolList 服务端 bug（新浪侧，不可修）
- 板块 extras 已当日缓存（进程内）；重启进程后同板块首问仍需 1 轮采集（已大幅降耗）
- 约 30 处非关键路径裸 except 未换日志
- 流式路径跳过多 pass 审查（保延迟），非流式才有；流式数字校验为 log-only
- 成分股行业分类可能过期（Tushare 数据源问题）
- Railway 文件系统临时性：data/archive（问责存档）与 charts/（SVG）重部署即丢，
  长期问责需挂 volume 或外置存储（ARCHIVE_DIR/CHART_DIR 可配）
- 流式存档为 best-effort：客户端中途断开则该次不落档
- eval 的 LLM judge 仅接口预留（EVAL_LLM=1），未实现
- 推送仅接通用 JSON webhook（PUSH_WEBHOOK_URL 未配置时只生成 payload 记 log）

## 新增环境变量（第五波）

- `PUSH_WEBHOOK_URL` 推送目标 webhook（未配置则只生成不发送）
- `PUSH_TIME` 推送时间，默认 `15:40`（Asia/Shanghai，仅工作日）
- `CHART_DIR` 图表目录，默认 `charts/`（挂载于 /charts）
- `ARCHIVE_DIR` 问责存档目录，默认 `data/archive/`
