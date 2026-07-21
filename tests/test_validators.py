"""数字溯源校验层测试（tests/test_validators.py）。

覆盖 agent/validators.py 的全部公开函数：
- extract_numbers：形态识别（%/亿/万/点/倍、符号、小数）
- find_unsourced_numbers：有出处通过、编造数字抓出、±0.05 容差、
  亿/万单位归一、日期豁免、小整数豁免、缺失标记豁免、空输入
- format_violations_for_critique：输出含违规数字与 snippet

全部纯文本构造，零网络、零外部依赖。
"""

import sys
from pathlib import Path

# 保证无论 conftest.py 是否就绪，都能从项目根导入 agent 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.validators import (  # noqa: E402
    extract_numbers,
    find_unsourced_numbers,
    format_violations_for_critique,
)

CONTEXT = (
    "【一、行情】上证指数 3891.25 点，涨跌幅 +1.25%；深证成指涨跌幅 -0.83%。"
    "成交额 1234.56 亿。上涨 3241 家，下跌 1523 家。"
    "【二、估值】加权PE 18.52 倍，近一年分位 72.30%。"
    "【三、资金】主力资金净流入 35000 万元，北向资金净流入 -12.40 亿。"
    "板块区间涨跌幅 2.346%。"
)


# ─────────────────────────────────────────────
# extract_numbers：形态识别
# ─────────────────────────────────────────────

class TestExtractNumbers:
    def test_basic_shapes(self):
        text = "涨幅12.3% 另一日+2.35% 成交1234.56亿 换手3.2万 收4500.12点 PE为1.2倍"
        toks = extract_numbers(text)
        raws = [t['raw'] for t in toks]
        assert '12.3%' in raws
        assert '+2.35%' in raws
        assert '1234.56亿' in raws
        assert '3.2万' in raws
        assert '4500.12点' in raws
        assert '1.2倍' in raws

    def test_values_and_units(self):
        toks = extract_numbers("+2.35% 1234.56亿 3.2万")
        by_raw = {t['raw']: t for t in toks}
        assert by_raw['+2.35%']['value'] == 2.35
        assert by_raw['+2.35%']['unit'] == '%'
        assert by_raw['1234.56亿']['value'] == 1234.56
        assert by_raw['1234.56亿']['normalized'] == 1234.56
        # 万单位归一：×1e-4 转亿
        assert abs(by_raw['3.2万']['normalized'] - 3.2e-4) < 1e-12

    def test_token_fields_present(self):
        toks = extract_numbers("收于 4500.12点。")
        assert len(toks) == 1
        t = toks[0]
        for key in ('value', 'normalized', 'raw', 'unit', 'start', 'end', 'snippet'):
            assert key in t
        assert t['start'] >= 0 and t['end'] > t['start']
        assert '4500.12点' in t['snippet']

    def test_negative_and_thousands(self):
        toks = extract_numbers("净流出 -12.40亿，成交 1,234.56亿")
        values = [t['value'] for t in toks]
        assert -12.40 in values
        assert 1234.56 in values

    def test_empty_input(self):
        assert extract_numbers(None) == []
        assert extract_numbers("") == []
        assert extract_numbers("没有数字的文本") == []

    def test_non_string_input_fail_safe(self):
        assert extract_numbers(123) != [] or extract_numbers(123) == []  # 不抛出即可
        assert extract_numbers(['a', 1]) == [] or True


# ─────────────────────────────────────────────
# find_unsourced_numbers：有出处 / 编造 / 容差
# ─────────────────────────────────────────────

class TestFindUnsourced:
    def test_sourced_numbers_pass(self):
        report = (
            "上证指数 3891.25点，涨跌幅 +1.25%；成交 1234.56亿。"
            "加权PE 18.52倍，分位 72.30%。主力净流入 35000万元。"
        )
        assert find_unsourced_numbers(report, CONTEXT) == []

    def test_fabricated_number_caught(self):
        report = "上证指数 3891.25点，涨跌幅 +9.87%。"
        violations = find_unsourced_numbers(report, CONTEXT)
        assert len(violations) == 1
        v = violations[0]
        assert v['value'] == 9.87
        assert v['raw'] == '+9.87%'
        assert '9.87' in v['snippet']
        assert '找不到匹配数字' in v['reason']

    def test_absolute_tolerance_005(self):
        # 上下文 2.346，报告写 2.35（四舍五入，差 0.004 < 0.05）→ 通过
        report = "板块区间涨跌幅 2.35%。"
        assert find_unsourced_numbers(report, CONTEXT) == []
        # 报告写 2.41（差 0.064 > max(0.05, 0.5%*2.346)）→ 抓出
        report_bad = "板块区间涨跌幅 2.41%。"
        violations = find_unsourced_numbers(report_bad, CONTEXT)
        assert len(violations) == 1
        assert violations[0]['value'] == 2.41

    def test_relative_tolerance(self):
        # 上下文 3891.25，相对容差 0.5% ≈ 19.46 > 0.05
        report = "上证指数 3899.0点。"   # 差 7.75 < 19.46 → 通过
        assert find_unsourced_numbers(report, CONTEXT) == []
        report_bad = "上证指数 3920.0点。"  # 差 28.75 > 19.46 → 抓出
        assert len(find_unsourced_numbers(report_bad, CONTEXT)) == 1

    def test_unit_normalization_wan_to_yi(self):
        # 上下文是 35000 万元，报告写 3.5亿 → 万×1e-4 归一后命中
        report = "主力资金净流入 3.5亿。"
        assert find_unsourced_numbers(report, CONTEXT) == []
        # 上下文有 -12.40亿，报告写 -124000万 → 同样归一命中
        report2 = "北向资金净流入 -124000万。"
        assert find_unsourced_numbers(report2, CONTEXT) == []
        # 报告写 4.5亿（归一后仍无出处）→ 抓出
        report_bad = "主力资金净流入 4.5亿。"
        assert len(find_unsourced_numbers(report_bad, CONTEXT)) == 1

    def test_percent_sign_ignored(self):
        # 上下文 72.30% 与报告 72.3% 视为同值
        report = "估值分位 72.3%。"
        assert find_unsourced_numbers(report, CONTEXT) == []


# ─────────────────────────────────────────────
# 豁免规则
# ─────────────────────────────────────────────

class TestExemptions:
    def test_date_exemption(self):
        report = (
            "【2026-07-22】A股市场复盘。2026年07月22日 收盘，"
            "较 07-19 以来走强。上证指数 3891.25点。"
        )
        # 2026 / 07 / 22 / 19 都是日期成分，不应被抓
        assert find_unsourced_numbers(report, CONTEXT) == []

    def test_small_int_exemption(self):
        # Top5、前3 名这类无单位小整数豁免
        report = "领涨板块 Top5 中，前3 名领涨股走强。成交 1234.56亿。"
        assert find_unsourced_numbers(report, CONTEXT) == []
        # 但带 % 的 5% 不豁免（上下文无 5）
        report_bad = "板块涨幅 5%。"
        assert len(find_unsourced_numbers(report_bad, CONTEXT)) == 1
        # 带「家」单位的 8 不豁免（上下文无 8 家）
        report_bad2 = "涨停 8家。"
        assert len(find_unsourced_numbers(report_bad2, CONTEXT)) == 1

    def test_time_window_exemption(self):
        report = "近20日 板块走强，近3日 缩量。成交 1234.56亿。"
        assert find_unsourced_numbers(report, CONTEXT) == []

    def test_missing_marker_exemption(self):
        report = "业绩预告数据未覆盖，涉及约 42 家公司。成交 1234.56亿。"
        assert find_unsourced_numbers(report, CONTEXT) == []

    def test_stock_code_exemption(self):
        report = "贵州茅台（600519）报收，000001.SZ 成交额 1234.56亿。"
        assert find_unsourced_numbers(report, CONTEXT) == []

    def test_false_date_range_not_exempted(self):
        # 「20-30倍」是区间不是日期（20 不是合法月份），其中数字仍需出处
        report = "PE 区间 20-30倍。"
        violations = find_unsourced_numbers(report, CONTEXT)
        assert len(violations) >= 1


# ─────────────────────────────────────────────
# 空输入与 fail-safe
# ─────────────────────────────────────────────

class TestFailSafe:
    def test_empty_inputs(self):
        assert find_unsourced_numbers(None, None) == []
        assert find_unsourced_numbers("", "") == []
        assert find_unsourced_numbers(None, CONTEXT) == []

    def test_empty_context_flags_all_non_exempt(self):
        # 上下文为空 → 数据集合为空 → 非豁免数字全部无出处
        report = "上证指数 3891.25点，涨跌幅 +1.25%。"
        violations = find_unsourced_numbers(report, "")
        assert len(violations) == 2

    def test_long_text_no_crash(self):
        long_report = ("上证指数 3891.25点。" * 2000) + "编造数字 77.77亿。"
        violations = find_unsourced_numbers(long_report, CONTEXT)
        assert any(v['value'] == 77.77 for v in violations)

    def test_weird_types(self):
        assert find_unsourced_numbers(12345, CONTEXT) == [] or True  # 不抛出
        assert find_unsourced_numbers("涨幅 1.25%", None) is not None
        assert format_violations_for_critique(None) != ""
        assert format_violations_for_critique("not a list") != ""


# ─────────────────────────────────────────────
# format_violations_for_critique
# ─────────────────────────────────────────────

class TestFormat:
    def test_format_contains_raw_and_snippet(self):
        report = "上证指数 3891.25点，涨跌幅 +9.87%，成交额 8888.88亿。"
        violations = find_unsourced_numbers(report, CONTEXT)
        assert len(violations) == 2
        out = format_violations_for_critique(violations)
        assert '+9.87%' in out
        assert '8888.88亿' in out
        # snippet 内容（数字前后片段）应出现在输出里
        assert violations[0]['snippet'] in out
        assert violations[1]['snippet'] in out
        assert '数字溯源校验' in out
        assert '2 个数字' in out

    def test_format_empty(self):
        out = format_violations_for_critique([])
        assert '未发现' in out

    def test_format_roundtrip_pipeline(self):
        # 完整管线：合规报告 → 无违规 → 「未发现」
        good = "成交 1234.56亿，PE 18.52倍。"
        out = format_violations_for_critique(
            find_unsourced_numbers(good, CONTEXT))
        assert '未发现' in out
