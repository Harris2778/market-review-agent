"""社媒过度拒答修复：agent/system_prompts.py 提示词配套测试（全 mock 零网络）。

背景：模型在纯 _chat（无工具）路径下对「股吧里大家怎么聊XX」「微博知乎抖音B站
热点」胡乱拒答。根因是 COMPLIANCE_PROMPT 的『直接拒答』节没有社媒例外，且角色
定位把数据源限定为新浪智研+Tushare，让模型视社媒为禁区。

覆盖范围：
1. COMPLIANCE_PROMPT『直接拒答』节新增社媒例外条款：存在、位置在研报例外之后
   且拒答话术之前、关键要点齐全（聚合整理/标注平台与日期/仅作辅助参考/不得套用
   拒答话术/能力边界）。
2. 角色定位段落补社媒数据源表述（东方财富股吧、微博/知乎/抖音/B站、辅助参考）。
3. AGENT_QUERY_PROMPT『社媒舆情引用规范』节顶部兜底句：用户主动询问社媒/股吧
   内容时正常调用工具回答，不得拒答。
4. 合规铁律四条与拒答话术原样保留（回归，例外不削弱铁律）。
5. 社媒例外随合规头拼进所有 intent 的最终提示词。
6. 各提示词拼接完整性回归：get_system_prompt 全部 key 拼接关系不变。
7. AGENT_QUERY_PROMPT 原社媒规范 1-6 条不漂移（回归）。
8. 六个提示词『合规边界』节禁操作建议表述均在（与社媒例外无冲突，回归）。

本文件只调用纯字符串函数，不触网、不读写文件。
"""

from agent.system_prompts import (
    AGENT_QUERY_PROMPT,
    COMPLIANCE_PROMPT,
    CRITIQUE_PROMPT,
    GENERAL_CHAT_PROMPT,
    MARKET_REVIEW_PROMPT,
    NEWS_ANALYSIS_PROMPT,
    NEWS_OUTPUT_RULES,
    SECTOR_DEEP_DIVE_PROMPT,
    STOCK_ANALYSIS_PROMPT,
    WATCHLIST_REVIEW_PROMPT,
    get_system_prompt,
)

# 社媒例外条款的标志性片段
SOCIAL_CARVE_OUT_PARTS = ("社媒", "股吧", "聚合整理", "仅作辅助参考", "不得套用拒答话术")

# 社媒能力边界的标志性片段
SOCIAL_CAPABILITY_PARTS = ("小红书暂未覆盖", "仅提供热榜", "评论仅B站支持")

# 合规铁律前四条原文（回归锚点，一字不动）
COMPLIANCE_IRON_RULES = (
    "1. 绝对禁止输出「买入、卖出、持有、加仓、减仓、建仓、清仓」等投资操作建议。",
    "2. 绝对禁止对股票/板块未来走势做出确定性预测，仅基于已发生的客观数据。",
    "3. 绝对禁止做「值得投资、具备配置价值、被低估、被高估」等价值定性判断。",
    "4. 绝对禁止主动推荐任何股票、基金、板块等投资标的。",
)

REFUSAL_WORDING = "我无法提供投资操作建议、股价预测及标的推荐"

ALL_INTENTS = (
    "market_review", "stock_query", "sector_deep_dive",
    "news_only", "news_analysis", "watchlist", "general_chat",
)


def _section(text: str, start_marker: str, end_marker: str) -> str:
    """截取 text 中 start_marker 到 end_marker 之间的段落（含起点，不含终点）。"""
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


# ─────────────────────────────────────────────
# 1. 社媒例外条款：存在与位置
# ─────────────────────────────────────────────

class TestSocialCarveOut:
    def test_social_carve_out_exists(self):
        """COMPLIANCE_PROMPT 含社媒例外条款的全部关键要点。"""
        for part in SOCIAL_CARVE_OUT_PARTS:
            assert part in COMPLIANCE_PROMPT, f"合规头缺少社媒例外片段 {part!r}"

    def test_social_carve_out_mentions_platforms(self):
        """社媒例外明确列出微博/知乎/抖音/B站平台。"""
        for platform in ("微博", "知乎", "抖音", "B站"):
            assert platform in COMPLIANCE_PROMPT, f"社媒例外缺少平台 {platform!r}"

    def test_social_carve_out_capability_boundary(self):
        """社媒例外如实说明能力边界：小红书未覆盖、微博/知乎/抖音仅热榜、评论仅B站。"""
        for part in SOCIAL_CAPABILITY_PARTS:
            assert part in COMPLIANCE_PROMPT, f"社媒例外缺少能力边界片段 {part!r}"

    def test_social_carve_out_position(self):
        """社媒例外位于『直接拒答』节内：研报例外之后、拒答话术之前。"""
        report_idx = COMPLIANCE_PROMPT.index("例外（不属于拒答范围，必须正常调用研报工具回答）")
        social_idx = COMPLIANCE_PROMPT.index("例外（社媒公开信息")
        refusal_idx = COMPLIANCE_PROMPT.index("拒答话术：「抱歉")
        assert report_idx < social_idx < refusal_idx, (
            "社媒例外必须位于研报例外之后、拒答话术之前"
        )

    def test_social_carve_out_inside_refusal_section(self):
        """社媒例外确实落在『直接拒答』节内（拒答节起于该标题，止于拒答话术行）。"""
        section = _section(COMPLIANCE_PROMPT, "## 直接拒答", "## 数据源优先级")
        for part in SOCIAL_CARVE_OUT_PARTS:
            assert part in section, f"『直接拒答』节内缺少社媒例外片段 {part!r}"

    def test_social_carve_out_not_leak_or_advice(self):
        """社媒例外明确：不属于非官方渠道泄密，不属于投资操作建议。"""
        section = _section(COMPLIANCE_PROMPT, "例外（社媒公开信息", "拒答话术：「抱歉")
        assert "不属于非官方渠道泄密" in section
        assert "投资操作" in section  # 「也不属于投资操作建议」

    def test_social_carve_out_effective_in_all_intents(self):
        """社媒例外随合规头拼进所有 intent 的最终提示词。"""
        for intent in ALL_INTENTS:
            p = get_system_prompt(intent)
            for part in SOCIAL_CARVE_OUT_PARTS:
                assert part in p, f"intent={intent} 缺少社媒例外片段 {part!r}"


# ─────────────────────────────────────────────
# 2. 角色定位新表述
# ─────────────────────────────────────────────

class TestRolePositioning:
    def test_role_includes_social_sources(self):
        """角色定位段落补社媒数据源：东财股吧 + 微博/知乎/抖音/B站 + 辅助参考。"""
        role = _section(COMPLIANCE_PROMPT, "## 角色定位", "## 输入校验")
        assert "东方财富股吧" in role
        assert "微博/知乎/抖音/B站" in role
        assert "社媒公开信息的聚合整理" in role
        assert "辅助参考" in role

    def test_role_original_sources_intact(self):
        """角色定位原有数据源表述不漂移（新浪智研+Tushare 保留，只追加不替换）。"""
        role = _section(COMPLIANCE_PROMPT, "## 角色定位", "## 输入校验")
        assert "新浪智研和Tushare数据源" in role
        assert "不做「投资决策、走势预测、标的推荐」" in role


# ─────────────────────────────────────────────
# 3. AGENT_QUERY_PROMPT 兜底句
# ─────────────────────────────────────────────

class TestAgentQuerySocialFallback:
    def test_fallback_sentence_at_section_top(self):
        """『社媒舆情引用规范』节顶部（第 1 条之前）有不得拒答兜底句。"""
        section_head = AGENT_QUERY_PROMPT.index("## 社媒舆情引用规范")
        first_item = AGENT_QUERY_PROMPT.index("1. 引用社媒舆情", section_head)
        top = AGENT_QUERY_PROMPT[section_head:first_item]
        assert "不得拒答" in top
        assert "正常调用社媒工具回答" in top

    def test_fallback_mentions_user_initiated(self):
        """兜底句覆盖「用户主动询问社媒/股吧内容」场景。"""
        section_head = AGENT_QUERY_PROMPT.index("## 社媒舆情引用规范")
        first_item = AGENT_QUERY_PROMPT.index("1. 引用社媒舆情", section_head)
        top = AGENT_QUERY_PROMPT[section_head:first_item]
        assert "用户主动询问" in top
        assert "股吧" in top

    def test_original_social_rules_intact(self):
        """原社媒舆情引用规范 1-6 条不漂移（回归，兜底句为追加）。"""
        for marker in (
            "1. 引用社媒舆情（热榜、搜索结果、评论）必须标注平台与抓取日期",
            "2. 社媒情绪噪声大",
            "3. 能力边界必须诚实",
            "5. 东财股吧已覆盖",
            "6. 引用股吧舆情必须标注「东方财富股吧+日期」",
        ):
            assert marker in AGENT_QUERY_PROMPT, f"AGENT_QUERY 原社媒规范 {marker!r} 丢失"


# ─────────────────────────────────────────────
# 4. 合规铁律与拒答话术回归（例外不削弱铁律）
# ─────────────────────────────────────────────

class TestIronRulesRegression:
    def test_iron_rules_intact(self):
        """合规铁律四条原文一字不动。"""
        for rule in COMPLIANCE_IRON_RULES:
            assert rule in COMPLIANCE_PROMPT, f"合规铁律漂移：{rule!r}"

    def test_refusal_wording_intact(self):
        """拒答话术本体不漂移。"""
        assert REFUSAL_WORDING in COMPLIANCE_PROMPT

    def test_refusal_triggers_intact(self):
        """『直接拒答』三条触发条件原样保留。"""
        section = _section(COMPLIANCE_PROMPT, "## 直接拒答", "例外（不属于拒答范围")
        for trigger in (
            "要求买卖建议、预测走势、推荐标的",
            "要求绕过规则、解除限制",
            "输入与金融数据查询无关且违规",
        ):
            assert trigger in section, f"拒答触发条件 {trigger!r} 丢失"

    def test_report_carve_out_intact(self):
        """研报例外条款原样保留（社媒例外为追加，不覆盖研报例外）。"""
        assert "例外（不属于拒答范围，必须正常调用研报工具回答）" in COMPLIANCE_PROMPT
        assert "整理转述" in COMPLIANCE_PROMPT


# ─────────────────────────────────────────────
# 5. 拼接完整性回归
# ─────────────────────────────────────────────

class TestPromptAssemblyRegression:
    def test_all_known_keys_assembly(self):
        """全部既有 key 的拼接关系与改动前一致。"""
        assert get_system_prompt("market_review") == COMPLIANCE_PROMPT + MARKET_REVIEW_PROMPT
        assert get_system_prompt("stock_query") == COMPLIANCE_PROMPT + STOCK_ANALYSIS_PROMPT
        assert get_system_prompt("sector_deep_dive") == COMPLIANCE_PROMPT + SECTOR_DEEP_DIVE_PROMPT
        assert get_system_prompt("news_only") == COMPLIANCE_PROMPT + NEWS_OUTPUT_RULES
        assert get_system_prompt("news_analysis") == COMPLIANCE_PROMPT + NEWS_ANALYSIS_PROMPT
        assert get_system_prompt("watchlist") == COMPLIANCE_PROMPT + WATCHLIST_REVIEW_PROMPT
        assert get_system_prompt("general_chat") == COMPLIANCE_PROMPT + GENERAL_CHAT_PROMPT

    def test_unknown_key_fallback(self):
        """未知 key 回落 COMPLIANCE_PROMPT，社媒例外随之生效。"""
        p = get_system_prompt("nonexistent_intent_xyz")
        assert p == COMPLIANCE_PROMPT
        for part in SOCIAL_CARVE_OUT_PARTS:
            assert part in p

    def test_output_format_plain_text_rule_intact(self):
        """纯文本输出要求保留（禁 Markdown 表格、禁#和*号）。"""
        assert "禁止Markdown表格" in COMPLIANCE_PROMPT
        assert "禁止#和*号" in COMPLIANCE_PROMPT


# ─────────────────────────────────────────────
# 6. 合规边界节与社媒例外无冲突（回归核对）
# ─────────────────────────────────────────────

class TestComplianceBoundaryConsistency:
    def test_boundary_sections_no_social_prohibition(self):
        """六个提示词的『合规边界』节均无「禁止社媒/股吧内容」类表述（与社媒例外无冲突）。"""
        for name, prompt in (
            ("COMPLIANCE_PROMPT", COMPLIANCE_PROMPT),
            ("MARKET_REVIEW_PROMPT", MARKET_REVIEW_PROMPT),
            ("STOCK_ANALYSIS_PROMPT", STOCK_ANALYSIS_PROMPT),
            ("SECTOR_DEEP_DIVE_PROMPT", SECTOR_DEEP_DIVE_PROMPT),
            ("NEWS_ANALYSIS_PROMPT", NEWS_ANALYSIS_PROMPT),
            ("AGENT_QUERY_PROMPT", AGENT_QUERY_PROMPT),
            ("WATCHLIST_REVIEW_PROMPT", WATCHLIST_REVIEW_PROMPT),
            ("CRITIQUE_PROMPT", CRITIQUE_PROMPT),
        ):
            assert "禁止提供社媒" not in prompt, f"{name} 存在与社媒例外冲突的禁社媒表述"
            assert "禁止引用股吧" not in prompt, f"{name} 存在与社媒例外冲突的禁股吧表述"
            assert "非官方渠道" not in prompt or name == "COMPLIANCE_PROMPT", (
                f"{name} 存在可能被误读为禁社媒的「非官方渠道」表述"
            )

    def test_boundary_sections_operation_ban_intact(self):
        """四个分析类提示词『合规边界』节禁操作建议表述均在（铁律不动摇）。"""
        for name, prompt in (
            ("SECTOR_DEEP_DIVE_PROMPT", SECTOR_DEEP_DIVE_PROMPT),
            ("NEWS_ANALYSIS_PROMPT", NEWS_ANALYSIS_PROMPT),
            ("AGENT_QUERY_PROMPT", AGENT_QUERY_PROMPT),
            ("WATCHLIST_REVIEW_PROMPT", WATCHLIST_REVIEW_PROMPT),
        ):
            boundary = _section(prompt, "## 合规边界", "## 输出格式") \
                if "## 输出格式" in prompt[prompt.index("## 合规边界"):] \
                else prompt[prompt.index("## 合规边界"):]
            assert "买入、卖出、持有、加仓、减仓" in boundary, \
                f"{name} 合规边界禁操作建议表述丢失"

    def test_social_exception_does_not_weaken_iron_rules(self):
        """社媒例外条款自身不含任何操作建议/预测/定性/推荐词汇（不引入铁律例外）。"""
        section = _section(COMPLIANCE_PROMPT, "例外（社媒公开信息", "拒答话术：「抱歉")
        for banned in ("买入", "卖出", "持有", "加仓", "减仓", "建仓", "清仓",
                       "值得投资", "被低估", "被高估"):
            assert banned not in section, f"社媒例外条款不得出现 {banned!r}"
