# eval/ — 离线评估集（零网络、零 LLM、可 CI）

第三波「eval 评估集」。对 A 股分析智能体三种输出模式
（`market_review` / `sector_deep_dive` / `news_only`）的质量红线做
**确定性**校验：不发起任何网络请求、不调用任何 LLM，可直接在 CI 运行。

## 质量红线（rubric 与之一一对应）

| rubric | 红线 | 实现 |
|---|---|---|
| `number_sourcing` | 每个数字必须有数据块出处 | 复用 `agent.validators.find_unsourced_numbers`，不重造 |
| `banned_words` | 禁用词清单（护城河/飞轮/赋能/格局/综上所述/深度 等） | 清单镜像 `agent/system_prompts.py`，见 `rubric.BANNED_WORDS` |
| `markdown_table` | `_clean_markdown` 后不应有管道表格残留 | 检测 ≥2 管道符的行与 `|---|` 分隔行 |
| `risk_disclaimer` | 非透传类输出必须含「不构成任何投资建议」 | 子串匹配 |
| `sector_structure` | `sector_deep_dive` 必须具备五维结构标记（趋势/估值/资金/景气度/催化） | 仅对 `sector_deep_dive` 生效，其他 mode 自动通过 |

每个 rubric 返回 `{"name": str, "passed": bool, "detail": str}`。

## 目录结构

```
eval/
  cases/*.json   评估用例（正例 + 埋雷反例）
  rubric.py      确定性检查器（5 个 rubric + run_all）
  run_eval.py    运行器：加载 case → 跑 rubric → 对比 expect → 退出码
  README.md      本文件
```

## case 格式

每个 case 是一个 JSON object：

```json
{
  "id": "sector_deep_dive_basic_ok",
  "mode": "sector_deep_dive",
  "description": "可选，一句话说明这个 case 在测什么",
  "fixture_context": "模拟注入 prompt 的数据块文本……",
  "output": "待评估的模型输出……",
  "expect": {
    "number_sourcing": true,
    "banned_words": true,
    "markdown_table": true,
    "risk_disclaimer": true,
    "sector_structure": true
  }
}
```

字段规则：

- `id`：必填，全局唯一，文件名建议与 id 一致（`<id>.json`）。
- `mode`：必填，`market_review` / `sector_deep_dive` / `news_only` 三选一。
- `fixture_context`：必填字符串，模拟注入 prompt 的数据块文本，
  是 `number_sourcing` 的出处比对基准。
- `output`：必填字符串，待评估的模型输出（非透传类应自带风险提示语）。
- `description`：可选，运行器会打印。
- `expect`：必填，**必须覆盖全部 5 个 rubric**，值为期望的 `passed` 布尔。
  - 正例：全部 `true`。
  - 埋雷反例：期望被抓出的 rubric 标 `false`，其余保持 `true`
    （用于验证埋雷是「精准命中」而非「整体崩坏」）。

## 运行

```bash
# 全部 case（退出码 0 = 正例全过且埋雷全被抓出）
/usr/local/bin/python3 eval/run_eval.py

# 只跑一个 case
/usr/local/bin/python3 eval/run_eval.py --case sector_deep_dive_basic_ok

# 严格质量模式：忽略 expect，任一 rubric 实际失败即非零
# （用于「这批输出必须绝对干净」的场景，如上线前抽检真实输出）
/usr/local/bin/python3 eval/run_eval.py --strict
```

退出码：

- `0`：全部 case 的实际结果与 `expect` 一致；
- `1`：任一不一致、case 文件非法，或 `--strict` 下任一 rubric 实际失败；
- `2`：运行器参数错误。

## 新增 case

1. 在 `eval/cases/` 新建 `<id>.json`，按上面的格式写齐字段，
   `expect` 覆盖全部 5 个 rubric。
2. 写 `output` 时注意**数字纪律**：凡是带单位（% / 亿 / 万 / 点 / 倍 / 家 等）
   的数字，正例必须能在 `fixture_context` 中找到出处（容差 ±0.05 或 0.5%，
   万→亿 自动 ×1e-4 归一）；日期、无单位小整数（<10）、近 N 日时间窗口、
   括号内 6 位证券代码按 `agent.validators` 规则豁免。
3. `/usr/local/bin/python3 eval/run_eval.py` 本地验证。
4. `/usr/local/bin/python3 -m pytest tests/test_eval_offline.py -q`
   会自动把新 case 纳入守护（正例必须全过、埋雷必须被抓出）。

数据单位纪律（与项目约定一致）：daily/index_daily 的 amount 单位千元；
moneyflow 金额万元（÷10000=亿）；daily_basic 的 total_mv 万元；rzye 元。

## LLM judge 钩子（预留，默认关闭）

设置环境变量 `EVAL_LLM=1` 后，运行器会逐 case 在确定性 rubric 之外追加调用
`run_eval.llm_judge(case, rubric_results)`。接口约定：

- 返回 `{"name": "llm_judge", "passed": bool, "detail": str}`，
  结构与确定性 rubric 一致，并入该 case 判定；
- 返回 `None` 表示跳过，不影响判定。

当前为占位实现（恒返回 `None`），**不发起任何网络/LLM 调用**。
后续接入时在 `run_eval.llm_judge` 内实现或替换为外部模块；
启用方需自行保证凭据与网络可用，CI 默认不开。

## harness 自守测试

`tests/test_eval_offline.py` 守护评估集自身：

- 全部 case 可解析、字段齐全、`expect` 覆盖全部 rubric、三种 mode 均有覆盖、
  每类埋雷（无出处数字/禁用词/表格残留/缺风险提示/五维缺失）至少一个 case；
- 正例 case 跑全部 rubric 必须通过；
- 埋雷 case 的实际结果必须与 `expect` 完全一致（该抓的抓到、不该抓的不误抓）；
- `run_eval.main`：完整评估集退出码为 0；被误标为正例的埋雷 case 退出码非零；
  `--strict` 下埋雷 case 退出码非零；
- `rubric.BANNED_WORDS` 每个词必须仍出现在 `agent/system_prompts.py`
  文本中（防止 prompt 侧清单更新后两侧漂移）。
