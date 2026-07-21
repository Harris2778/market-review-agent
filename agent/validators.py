"""输出后数字溯源校验层（agent/validators.py）。

确定性（非 LLM）数字校验：系统 prompt 红线要求报告里每一个数字都必须
能在注入的数据上下文（数据块）里找到出处。本模块在报告产出后做纯
stdlib 的正则级核对，把「在数据上下文中找不到出处的数字」挑出来，
供下游 critique 审查流程使用。

三个公开函数：
- extract_numbers(text)        从文本抽取数字 token（带符号/小数/单位上下文）
- find_unsourced_numbers(report, context)  找出无出处数字
- format_violations_for_critique(violations)  格式化为可追加给 LLM 审查 prompt 的中文段落

匹配容差与单位归一：
- 容差：abs(a-b) <= max(绝对 0.05, 相对 0.5% * abs(上下文值))，取大者。
  0.05 的绝对容差覆盖「2.346% 四舍五入写成 2.35%」这类合法舍入。
- 单位归一：报告侧带『万』/『万元』的数字 ×1e-4 转亿后再比；
  带『亿』与裸数值直接按数值比；百分号忽略（2.35% 与 2.35 视为同值）。
  上下文侧同时收录原始值与（万单位时的）×1e-4 归一值，双向都能命中。

豁免启发式（避免误报，按优先级逐条判定，命中即豁免）：
1. 日期成分：YYYY-MM-DD / YYYY/MM/DD / YYYY.M.D / YYYY年MM月DD日 /
   MM-DD / MM月DD日 等形态（月 1-12、日 1-31、年 1900-2100 校验，
   防止把「20-30倍」这种区间误判成日期）；以及无单位的裸四位年份
   （1900-2100）。落在日期区间内的数字一律豁免。
2. 上下文小整数：无单位且 |值| < 10 的整数豁免（如「Top5」「近3年」
   之外的「5」「3」这类量词性整数）；一旦带 % / 亿 / 万 / 点 / 倍 /
   家 等任意单位即不再豁免（「8家」「5%」都是必须找出处的真数据）。
3. 时间窗口整数：带 年/月/日 单位的整数（如「近20日」「近3日」「5日」），
   描述的是回看窗口而非数据值，豁免。
4. 证券代码形态：无单位无小数的 6 位整数，且被 （）/() 包裹或后随
   .SZ/.SH/.BJ（如「贵州茅台（600519）」「000001.SZ」），属于标的
   标识符而非数据，豁免。
5. 数据缺失标记邻近：数字前后 ±15 字符内出现「数据未覆盖 / 数据暂缺 /
   数据缺失」等标记时豁免（该数字所在句已明确标注数据缺口）。

已知边界（留给后续迭代）：「31个行业」「20条新闻」这类 ≥10 的计数型
整数仍按「需出处」处理——在复盘模板里家数/条数本身是数据块应提供的
真实数据，宁可多报一条让 critique 去判，也不静默放过。

所有函数 fail-safe：None、空串、非字符串、超长文本等异常输入一律
返回空结果，绝不抛出。
"""

from __future__ import annotations

import re

# ── 容差常量 ──
_ABS_TOL = 0.05        # 绝对容差
_REL_TOL = 0.005       # 相对容差（0.5%）

# ── 摘录片段半径（字符）──
_SNIPPET_RADIUS = 15

# ── 数据缺失标记：出现在数字邻近时豁免 ──
_MISSING_MARKERS = ("数据未覆盖", "数据暂缺", "数据缺失", "未覆盖", "暂缺")

# ── 数字本体：支持千分位、小数、正负号 ──
_NUM_CORE = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

# ── 单位（长单位优先，避免「亿元」被截成「亿」）──
_UNITS = (
    '亿元', '万元', '亿', '万',
    '%', '％',
    '点', '倍', '元',
    '家', '只', '条', '手', '股',
    '年', '月', '日',
)
_NUMBER_RE = re.compile(_NUM_CORE + r'\s*(?:' + '|'.join(_UNITS) + r')?')
_NUM_CORE_RE = re.compile(_NUM_CORE)

# ── 日期候选形态（后续再做月/日/年范围校验）──
_DATE_CANDIDATE_RE = re.compile(
    r'(?P<y1>\d{4})\s*[-/.年]\s*(?P<m1>\d{1,2})\s*[-/.月]\s*(?P<d1>\d{1,2})\s*日?'
    r'|(?P<m2>\d{1,2})\s*[-/月]\s*(?P<d2>\d{1,2})\s*日'
    r'|(?P<m3>\d{1,2})\s*-\s*(?P<d3>\d{1,2})'
)

_STOCK_CODE_SUFFIX_RE = re.compile(r'\.(?:SZ|SH|BJ)\b', re.IGNORECASE)

# 时间窗口单位（整数 + 年/月/日 → 豁免）
_TIME_WINDOW_UNITS = {'年', '月', '日'}


def _as_text(x) -> str:
    """任意输入安全转字符串；失败返回空串。"""
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        return str(x)
    except Exception:
        return ""


def _valid_date(y, m, d) -> bool:
    """校验日期候选的年/月/日范围，过滤「20-30倍」这类伪日期。"""
    try:
        if y is not None and not (1900 <= int(y) <= 2100):
            return False
        if m is not None and not (1 <= int(m) <= 12):
            return False
        if d is not None and not (1 <= int(d) <= 31):
            return False
        return True
    except Exception:
        return False


def _date_spans(text: str) -> list:
    """返回文本中所有合法日期形态的 (start, end) 区间。"""
    spans = []
    try:
        for m in _DATE_CANDIDATE_RE.finditer(text):
            g = m.groupdict()
            if g.get('y1') is not None:
                ok = _valid_date(g['y1'], g['m1'], g['d1'])
            elif g.get('m2') is not None:
                ok = _valid_date(None, g['m2'], g['d2'])
            else:
                ok = _valid_date(None, g['m3'], g['d3'])
            if ok:
                spans.append((m.start(), m.end()))
    except Exception:
        pass
    return spans


def _in_spans(start: int, end: int, spans: list) -> bool:
    for s, e in spans:
        if start < e and s < end:
            return True
    return False


def _snippet(text: str, start: int, end: int, radius: int = _SNIPPET_RADIUS) -> str:
    """取数字前后各 radius 字符的上下文片段，压平换行。"""
    try:
        frag = text[max(0, start - radius):min(len(text), end + radius)]
        return re.sub(r'\s+', ' ', frag).strip()
    except Exception:
        return ""


def extract_numbers(text) -> list:
    """从文本抽取数字 token。

    返回 list[dict]，每项字段：
      value      float 数值（含符号、去千分位）
      normalized float 归一值：万/万元 ×1e-4 转亿，其余同 value
      raw        str   原文匹配（含单位）
      unit       str   单位（无单位为空串；% 含全角％）
      start/end  int   在原文中的字符区间
      snippet    str   前后各 15 字符的上下文片段

    可识别形态：12.3%、+2.35%、1234.56亿、3.2万、4500.12点、1.2倍、
    35000万元、1,234.56 等。任何异常输入返回 []。
    """
    results = []
    try:
        text = _as_text(text)
        if not text:
            return results
        for m in _NUMBER_RE.finditer(text):
            raw = m.group(0)
            core = _NUM_CORE_RE.match(raw)
            if not core:
                continue
            num_str = core.group(0)
            unit = raw[len(num_str):].strip()
            try:
                value = float(num_str.replace(',', ''))
            except (ValueError, TypeError):
                continue
            normalized = value * 1e-4 if unit in ('万', '万元') else value
            results.append({
                'value': value,
                'normalized': normalized,
                'raw': raw.strip(),
                'unit': unit,
                'start': m.start(),
                'end': m.end(),
                'snippet': _snippet(text, m.start(), m.end()),
            })
    except Exception:
        return []
    return results


def _is_bare_year(tok) -> bool:
    """无单位裸四位年份（1900-2100）。"""
    try:
        return (tok['unit'] == ''
                and float(tok['value']).is_integer()
                and 1900 <= abs(tok['value']) <= 2100)
    except Exception:
        return False


def _is_small_int(tok) -> bool:
    """无单位且 |值| < 10 的整数（量词性小整数）。"""
    try:
        return (tok['unit'] == ''
                and float(tok['value']).is_integer()
                and abs(tok['value']) < 10)
    except Exception:
        return False


def _is_time_window(tok) -> bool:
    """带 年/月/日 单位的整数（回看窗口，如 近20日、5日）。"""
    try:
        return (tok['unit'] in _TIME_WINDOW_UNITS
                and float(tok['value']).is_integer())
    except Exception:
        return False


def _is_stock_code(text: str, tok) -> bool:
    """6 位整数且被括号包裹或后随 .SZ/.SH/.BJ 的证券代码形态。"""
    try:
        if tok['unit'] != '' or not float(tok['value']).is_integer():
            return False
        digits = tok['raw']
        if len(digits) != 6 or not digits.isdigit():
            return False
        s, e = tok['start'], tok['end']
        # 后随 .SZ/.SH/.BJ
        if _STOCK_CODE_SUFFIX_RE.match(text[e:e + 4]):
            return True
        # 被 （）或 () 包裹
        if s > 0 and e < len(text):
            if text[s - 1] in '（(' and text[e] in '）)':
                return True
        return False
    except Exception:
        return False


def _near_missing_marker(text: str, tok, radius: int = 15) -> bool:
    """数字邻近出现「数据未覆盖/数据暂缺」等缺失标记。"""
    try:
        window = text[max(0, tok['start'] - radius):
                      min(len(text), tok['end'] + radius)]
        return any(marker in window for marker in _MISSING_MARKERS)
    except Exception:
        return False


def _context_values(context: str) -> list:
    """上下文数字集合：原始值 + （万单位时的）×1e-4 归一值，双倍收录。"""
    values = []
    try:
        for tok in extract_numbers(context):
            values.append(tok['value'])
            if tok['unit'] in ('万', '万元'):
                values.append(tok['value'] * 1e-4)
    except Exception:
        pass
    return values


def _matches(value: float, context_values: list) -> bool:
    """容差匹配：abs 差 <= max(0.05, 0.5% * abs(上下文值))。"""
    try:
        for c in context_values:
            try:
                tol = max(_ABS_TOL, _REL_TOL * abs(c))
                if abs(value - c) <= tol:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def find_unsourced_numbers(report, context) -> list:
    """找出报告中在数据上下文里找不到出处的数字。

    参数：
      report  报告文本
      context 注入 prompt 的数据上下文文本（数据块）

    返回 list[dict]，每项字段：
      value    float 违规数值
      raw      str   原文数字（含单位）
      snippet  str   原文上下文片段
      reason   str   判定原因（中文，可直接给 LLM 看）

    豁免规则见模块 docstring。任何异常输入返回 []。
    """
    violations = []
    try:
        report = _as_text(report)
        context = _as_text(context)
        if not report.strip():
            return violations

        tokens = extract_numbers(report)
        spans = _date_spans(report)
        ctx_vals = _context_values(context)

        for tok in tokens:
            try:
                # ── 豁免链（命中即跳过）──
                if _in_spans(tok['start'], tok['end'], spans):
                    continue                                  # 1. 日期成分
                if _is_bare_year(tok):
                    continue                                  # 1b. 裸四位年份
                if _is_small_int(tok):
                    continue                                  # 2. 无单位小整数
                if _is_time_window(tok):
                    continue                                  # 3. 时间窗口
                if _is_stock_code(report, tok):
                    continue                                  # 4. 证券代码
                if _near_missing_marker(report, tok):
                    continue                                  # 5. 缺失标记邻近

                # ── 容差匹配（万单位先归一成亿）──
                if _matches(tok['normalized'], ctx_vals):
                    continue

                reason = '在数据上下文中找不到匹配数字（容差 ±0.05 或 0.5%）'
                if tok['unit'] in ('万', '万元'):
                    reason += '，已按 万→亿 ×1e-4 归一后比对'
                violations.append({
                    'value': tok['value'],
                    'raw': tok['raw'],
                    'snippet': tok['snippet'],
                    'reason': reason,
                })
            except Exception:
                continue
    except Exception:
        return []
    return violations


def format_violations_for_critique(violations) -> str:
    """把违规清单格式化为可追加给 LLM 审查 prompt 的中文段落。

    空清单返回「未发现」一句话；任何异常输入返回兜底说明，绝不抛出。
    """
    try:
        if not violations:
            return '【数字溯源校验】未发现无出处数字。'
        items = list(violations)
        lines = [
            '【数字溯源校验】以下 %d 个数字在数据上下文中找不到出处，请逐项核对：'
            % len(items),
        ]
        for i, v in enumerate(items, 1):
            try:
                raw = _as_text(v.get('raw')) if isinstance(v, dict) else _as_text(v)
                snippet = _as_text(v.get('snippet')) if isinstance(v, dict) else ''
                reason = _as_text(v.get('reason')) if isinstance(v, dict) else ''
                value = v.get('value') if isinstance(v, dict) else None
                try:
                    value_str = '%g' % float(value)
                except (TypeError, ValueError):
                    value_str = _as_text(value)
                lines.append(
                    '%d. 数字「%s」（数值 %s）｜原文片段：…%s…｜原因：%s'
                    % (i, raw, value_str, snippet, reason or '未注明')
                )
            except Exception:
                continue
        lines.append(
            '处理建议：删除该数字、改写为「数据未覆盖」，'
            '或回到数据块指认正确出处后修正数值。'
        )
        return '\n'.join(lines)
    except Exception:
        return '【数字溯源校验】格式化失败，请人工核对报告数字出处。'
