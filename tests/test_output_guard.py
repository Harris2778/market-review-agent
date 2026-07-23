"""tests/test_output_guard.py — agent/output_guard.py 的单元测试。

覆盖 QA 实锤的三类「数据未覆盖却仍下结论」案例：
① 同句自相矛盾（RSI6 技术分析路径）
② 跨句主题矛盾（北向资金大盘复盘路径）
③ 头部比较级结论（五粮液估值人格对比路径）
以及幂等性、正常文本零改动、豁免句、保守边界与异常输入 fail-safe。
"""

from agent.output_guard import fix_uncovered_conclusions


# ---------- QA 实锤案例 ①：同句自相矛盾 ----------

def test_case1_same_sentence_rsi6():
    text = "RSI6数据未覆盖，进入超买区域，短期有回调压力。"
    assert fix_uncovered_conclusions(text) == "RSI6数据未覆盖。"


def test_case1_keeps_neutral_clauses():
    # 中性分句保留，只删结论分句
    text = "数据暂缺，今日市场交投活跃，存在回调压力。"
    assert fix_uncovered_conclusions(text) == "数据暂缺，今日市场交投活跃。"


# ---------- QA 实锤案例 ②：跨句主题矛盾（北向资金） ----------

def test_case2_cross_sentence_northbound():
    text = (
        "【沪深港通】\n"
        "北向资金数据未覆盖。\n"
        "【资金流向】\n"
        "北向资金整体呈净流入态势，市场情绪偏暖。"
    )
    result = fix_uncovered_conclusions(text)
    assert "净流入" not in result
    assert "北向资金数据未覆盖" in result
    # 矛盾句删除后留下的空小节标题一并清掉
    assert "【资金流向】" not in result
    assert "【沪深港通】" in result


# ---------- QA 实锤案例 ③：头部比较级结论（五粮液估值） ----------

def test_case3_head_compare_wuliangye():
    text = "五粮液更便宜，配置价值凸显。\n估值数据未覆盖，仅整理已披露信息。"
    result = fix_uncovered_conclusions(text)
    assert result == "估值数据未覆盖，无法判断。\n估值数据未覆盖，仅整理已披露信息。"
    assert "更便宜" not in result


# ---------- 幂等性 ----------

def test_idempotent_all_qa_cases():
    texts = [
        "RSI6数据未覆盖，进入超买区域，短期有回调压力。",
        "【沪深港通】\n北向资金数据未覆盖。\n北向资金整体呈净流入态势。",
        "五粮液更便宜。\n估值数据未覆盖。",
    ]
    for text in texts:
        once = fix_uncovered_conclusions(text)
        twice = fix_uncovered_conclusions(once)
        assert twice == once


# ---------- 无矛盾正常文本零改动 ----------

def test_normal_text_untouched():
    text = "今日沪指收涨1.23%，北向资金净流入58.20亿元。\nRSI6进入超买区域，短期或有波动。"
    assert fix_uncovered_conclusions(text) == text


def test_uncovered_only_statement_untouched():
    # 只陈述未覆盖、不下结论的句子不动
    text = "RSI数据未覆盖。\nMACD运行于零轴上方。"
    assert fix_uncovered_conclusions(text) == text


# ---------- 豁免句 ----------

def test_exemption_no_conclusion_words():
    text = "估值数据未覆盖，无法判断是否低估。"
    assert fix_uncovered_conclusions(text) == text


def test_exemption_cross_sentence():
    text = "北向资金数据未覆盖。\n从现有信息难以判断北向资金净流入趋势。"
    assert fix_uncovered_conclusions(text) == text


# ---------- 保守边界：分句内矛盾混合，整句不改 ----------

def test_mixed_clause_conservative():
    text = "数据未覆盖但估值仍处低估区域。"
    assert fix_uncovered_conclusions(text) == text


# ---------- 异常输入 fail-safe ----------

def test_abnormal_inputs_no_crash():
    assert fix_uncovered_conclusions(None) is None
    assert fix_uncovered_conclusions("") == ""
    assert fix_uncovered_conclusions("   ") == "   "
    assert fix_uncovered_conclusions(123) == 123
    assert fix_uncovered_conclusions(["未覆盖"]) == ["未覆盖"]
