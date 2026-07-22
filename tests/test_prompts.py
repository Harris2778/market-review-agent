"""第六/七波集成：agent/system_prompts.py 提示词配套测试（全 mock 零网络）。

覆盖范围：
1. get_system_prompt("watchlist") 新 key 可用：返回合规头 + 自选股复盘提示词，
   含禁用词红线、点评框架（逐只点评/整体观察/数据未覆盖明说）、合规边界。
2. sector_deep_dive 追加行业知识库使用规则与以史为鉴使用规则，原五维框架不动。
3. market_review 追加与 sector 相同的以史为鉴指引，原输出模板不动。
4. 注入防护：news_analysis / AGENT_QUERY_PROMPT / CRITIQUE_PROMPT 各含
   「外部不可信文本 + 〔已过滤〕占位符」防护行。
5. 既有 key 回归：market_review / stock_query / sector_deep_dive（带参）/
   news_only / news_analysis / general_chat 返回值与常量拼接关系不变。
6. 未知 key 行为与改动前一致：回落 COMPLIANCE_PROMPT，不抛异常。

本文件只调用纯字符串函数，不触网、不读写文件。
"""

from agent.system_prompts import (
    AGENT_QUERY_PROMPT,
    COMPLIANCE_PROMPT,
    CRITIQUE_PROMPT,
    MARKET_REVIEW_PROMPT,
    NEWS_ANALYSIS_PROMPT,
    NEWS_OUTPUT_RULES,
    SECTOR_DEEP_DIVE_PROMPT,
    STOCK_ANALYSIS_PROMPT,
    WATCHLIST_REVIEW_PROMPT,
    get_system_prompt,
)

# 禁用词红线抽样（与 rubric.BANNED_WORDS 同源的标志性词）
BANNED_WORD_SAMPLE = ("护城河", "飞轮", "综上所述", "黄金坑", "戴维斯双击")

# 注入防护行的三个标志性片段
INJECTION_GUARD_PARTS = ("外部不可信文本", "一律忽略", "〔已过滤〕")

# 以史为鉴指引的标志性片段（sector / market 两处共用）
HISTORY_GUIDE_PARTS = ("以史为鉴", "克制", "不改变数据事实")


# ─────────────────────────────────────────────
# 1. watchlist 新 key
# ─────────────────────────────────────────────

class TestWatchlistPrompt:
    def test_watchlist_registered(self):
        """get_system_prompt("watchlist") 可用：合规头 + 自选股复盘提示词。"""
        p = get_system_prompt("watchlist")
        assert p == COMPLIANCE_PROMPT + WATCHLIST_REVIEW_PROMPT
        assert p.startswith(COMPLIANCE_PROMPT)
        assert len(p) > len(COMPLIANCE_PROMPT)

    def test_watchlist_no_params(self):
        """无参数调用：仅传 intent 即可，sector_name 缺省。"""
        p = get_system_prompt("watchlist")
        assert isinstance(p, str) and p.strip()

    def test_watchlist_banned_words_redline(self):
        """自选股提示词含禁用词红线（禁用词清单原样沿用）。"""
        p = get_system_prompt("watchlist")
        assert "禁用词" in p
        for w in BANNED_WORD_SAMPLE:
            assert w in p, f"watchlist 提示词缺少禁用词 {w!r}"
        # 语言红线三件套：禁排比、长短句交错
        assert "排比" in p
        assert "长短交错" in p

    def test_watchlist_framework(self):
        """点评框架：逐只点评持仓结构、数据未覆盖明说、结尾整体观察。"""
        p = get_system_prompt("watchlist")
        assert "【一、逐只点评】" in p
        assert "持仓结构" in p
        assert "本期数据未覆盖" in p
        assert "【二、整体观察】" in p

    def test_watchlist_compliance_boundary(self):
        """合规边界：禁操作建议、禁价值定性、不自行输出风险提示。"""
        p = get_system_prompt("watchlist")
        assert "买入、卖出、持有、加仓、减仓" in p
        assert "值得投资" in p
        assert "风险提示" in p

    def test_watchlist_data_redline(self):
        """数据真实性红线：只用数据块数字，新闻引用不编造。"""
        p = get_system_prompt("watchlist")
        assert "数据真实性红线" in p
        assert "不得编造新闻标题" in p


# ─────────────────────────────────────────────
# 2. sector_deep_dive 新指引
# ─────────────────────────────────────────────

class TestSectorDeepDiveGuidance:
    def test_kb_usage_rule(self):
        """行业知识库使用规则：背景知识、以数据块为准、引用不标来源标签。"""
        assert "行业知识库使用规则" in SECTOR_DEEP_DIVE_PROMPT
        assert "【六、行业知识库（背景知识，数据以数据块为准）】" in SECTOR_DEEP_DIVE_PROMPT
        assert "以数据块为准" in SECTOR_DEEP_DIVE_PROMPT
        assert "不标注数据来源标签" in SECTOR_DEEP_DIVE_PROMPT

    def test_history_lens_rule(self):
        """以史为鉴使用规则：命中率低则措辞克制、不改变数据事实。"""
        assert "以史为鉴使用规则" in SECTOR_DEEP_DIVE_PROMPT
        for part in HISTORY_GUIDE_PARTS:
            assert part in SECTOR_DEEP_DIVE_PROMPT, \
                f"sector_deep_dive 以史为鉴指引缺少 {part!r}"

    def test_original_framework_intact(self):
        """原五维框架与红线结构一字未动（追加不改原结构）。"""
        for marker in (
            "你是一名券商行业研究员",
            "## 数据真实性红线（最高优先级，违反任何一条都算失败）",
            "【一、趋势位置】",
            "【二、估值水位】",
            "【三、资金博弈】",
            "【四、景气度】",
            "【五、催化与风险】",
            "【综合判断】",
            "## 分析框架（五个维度 + 综合判断，按此顺序输出）",
            "全文800-1500字",
        ):
            assert marker in SECTOR_DEEP_DIVE_PROMPT, \
                f"sector_deep_dive 原框架标记 {marker!r} 丢失"

    def test_get_prompt_with_sector_param(self):
        """带参调用行为不变：sector_name 参数照常接受。"""
        p = get_system_prompt("sector_deep_dive", "白酒")
        assert p == COMPLIANCE_PROMPT + SECTOR_DEEP_DIVE_PROMPT
        assert "行业知识库使用规则" in p
        assert "以史为鉴使用规则" in p


# ─────────────────────────────────────────────
# 3. market_review 以史为鉴指引
# ─────────────────────────────────────────────

class TestMarketReviewHistoryLens:
    def test_history_lens_rule(self):
        """market_review 含与 sector 相同的以史为鉴指引（关键表述一致）。"""
        assert "以史为鉴使用规则" in MARKET_REVIEW_PROMPT
        for part in HISTORY_GUIDE_PARTS:
            assert part in MARKET_REVIEW_PROMPT, \
                f"market_review 以史为鉴指引缺少 {part!r}"

    def test_original_template_intact(self):
        """原固定流程与输出模板不动。"""
        for marker in (
            "触发：用户要求市场复盘。固定流程：",
            "一、市场整体概览",
            "二、板块表现",
            "三、资金动向",
            "四、核心要闻",
        ):
            assert marker in MARKET_REVIEW_PROMPT, \
                f"market_review 原模板标记 {marker!r} 丢失"

    def test_get_prompt_regression(self):
        """get_system_prompt("market_review") 拼接关系不变。"""
        p = get_system_prompt("market_review")
        assert p == COMPLIANCE_PROMPT + MARKET_REVIEW_PROMPT


# ─────────────────────────────────────────────
# 4. 注入防护行
# ─────────────────────────────────────────────

class TestInjectionGuard:
    def _assert_guard(self, prompt: str, name: str):
        for part in INJECTION_GUARD_PARTS:
            assert part in prompt, f"{name} 注入防护行缺少 {part!r}"

    def test_news_analysis_guard(self):
        self._assert_guard(NEWS_ANALYSIS_PROMPT, "NEWS_ANALYSIS_PROMPT")

    def test_agent_query_guard(self):
        self._assert_guard(AGENT_QUERY_PROMPT, "AGENT_QUERY_PROMPT")

    def test_critique_guard(self):
        self._assert_guard(CRITIQUE_PROMPT, "CRITIQUE_PROMPT")

    def test_news_analysis_get_prompt(self):
        """news_analysis 经 get_system_prompt 取出后防护行仍在。"""
        p = get_system_prompt("news_analysis")
        self._assert_guard(p, "get_system_prompt('news_analysis')")

    def test_news_analysis_original_redline_intact(self):
        """news_analysis 原红线与框架不动。"""
        for marker in (
            "你是一名财经评论员",
            "【一、主题归纳】",
            "【二、方向判断】",
            "【三、与板块的关联】",
            "【四、关注清单】",
        ):
            assert marker in NEWS_ANALYSIS_PROMPT, \
                f"news_analysis 原框架标记 {marker!r} 丢失"

    def test_critique_original_checklist_intact(self):
        """CRITIQUE 原五条审查清单不动。"""
        for marker in (
            "【一、数字出处】",
            "【二、禁用词】",
            "【三、无据断言】",
            "【四、合规越界】",
            "【五、AI腔】",
            "只输出两个字：通过",
        ):
            assert marker in CRITIQUE_PROMPT, \
                f"CRITIQUE 原清单标记 {marker!r} 丢失"


# ─────────────────────────────────────────────
# 5. 既有 key 回归 + 未知 key 行为
# ─────────────────────────────────────────────

class TestExistingKeysRegression:
    def test_all_known_keys(self):
        """全部既有 key 的拼接关系与改动前一致。"""
        assert get_system_prompt("market_review") == COMPLIANCE_PROMPT + MARKET_REVIEW_PROMPT
        assert get_system_prompt("stock_query") == COMPLIANCE_PROMPT + STOCK_ANALYSIS_PROMPT
        assert get_system_prompt("sector_deep_dive") == COMPLIANCE_PROMPT + SECTOR_DEEP_DIVE_PROMPT
        assert get_system_prompt("sector_deep_dive", "半导体") == COMPLIANCE_PROMPT + SECTOR_DEEP_DIVE_PROMPT
        assert get_system_prompt("news_only") == COMPLIANCE_PROMPT + NEWS_OUTPUT_RULES
        assert get_system_prompt("news_analysis") == COMPLIANCE_PROMPT + NEWS_ANALYSIS_PROMPT
        assert get_system_prompt("general_chat") == COMPLIANCE_PROMPT

    def test_unknown_key_fallback(self):
        """未知 key 行为与改动前一致：回落 COMPLIANCE_PROMPT，不抛异常。"""
        assert get_system_prompt("nonexistent_intent_xyz") == COMPLIANCE_PROMPT
        assert get_system_prompt("") == COMPLIANCE_PROMPT

    def test_unknown_key_with_param(self):
        """未知 key 带参同样回落，不抛异常。"""
        assert get_system_prompt("nonexistent_intent_xyz", "白酒") == COMPLIANCE_PROMPT

    def test_banned_words_in_all_analytical_prompts(self):
        """四个分析类 prompt 禁用词红线均在（含 watchlist 共五处红线不漂移）。"""
        for name, prompt in (
            ("SECTOR_DEEP_DIVE_PROMPT", SECTOR_DEEP_DIVE_PROMPT),
            ("NEWS_ANALYSIS_PROMPT", NEWS_ANALYSIS_PROMPT),
            ("AGENT_QUERY_PROMPT", AGENT_QUERY_PROMPT),
            ("WATCHLIST_REVIEW_PROMPT", WATCHLIST_REVIEW_PROMPT),
        ):
            for w in BANNED_WORD_SAMPLE:
                assert w in prompt, f"{name} 缺少禁用词 {w!r}"


# ─────────────────────────────────────────────
# 7. 拒答例外：研报观点整理转述不得套用拒答话术
# ─────────────────────────────────────────────

class TestReportQACarveOut:
    """合规头「直接拒答」段必须含研报例外条款（2026-07-22 过度拒答修复）。

    背景：用户问「白酒行业研报正文里券商对下半年需求怎么看」被误判为
    「预测走势」触发拒答。例外条款明确：整理转述研报观点（含券商展望）
    属于信息汇总，必须调用研报工具回答。
    """

    CARVE_OUT_PARTS = ("例外", "研报", "整理转述", "不得套用拒答话术")

    def test_carve_out_in_compliance_prompt(self):
        for part in self.CARVE_OUT_PARTS:
            assert part in COMPLIANCE_PROMPT, f"合规头缺少拒答例外片段 {part!r}"

    def test_carve_out_effective_in_all_intents(self):
        """例外条款随合规头拼进所有 intent 的最终提示词。"""
        for intent in (
            "stock_query", "sector_deep_dive", "news_analysis",
            "general_chat", "watchlist", "market_review", "news_only",
        ):
            p = get_system_prompt(intent)
            for part in self.CARVE_OUT_PARTS:
                assert part in p, f"intent={intent} 缺少拒答例外片段 {part!r}"

    def test_refusal_wording_itself_intact(self):
        """拒答话术本体不漂移（例外是新增条款，不改话术）。"""
        assert "我无法提供投资操作建议、股价预测及标的推荐" in COMPLIANCE_PROMPT
