"""tests/test_news_fetch_layer.py — 抓取层修复专项测试（工程师B）。

覆盖范围：
1. _truncate_at_boundary：标点边界截断——句末/分句标点优先、过靠前标点回退、
   无标点硬切、短文本原样、非 str 容错；核心断言「绝不在句子中间截断」。
2. _sanitize_news_text 的 max_len 截断走边界逻辑（带标点文本不再拦腰切断）。
3. fetch_tushare_news：权限不足置进程级 denied 标记并短路后续调用（省配额）；
   非权限异常不置标记、照常降级 sina 源。
4. fetch_mcp_news：无 title 条目用 content 边界截断做标题且完整 content 保留；
   limit>20 自动翻页、跨页去重、不足页提前停止。
5. fetch_cls_telegraph：title 为空时 brief 边界截断做标题；summary 字段随条目导出；
   HTTP 非 200 安全降级。
6. fetch_eastmoney_news/page2：标题完整保留、超长边界截断；HTTP 非 200 安全降级。
7. fetch_news_pool 板块模式：智研双源加深抓取（各 60 条）、
   content/summary/brief 字段随统一条目保留、弃用源（东财/新浪/财联社/Tushare）
   任何模式下都不再被调用。

规则：全部 mock 网络层（requests / _mcp_call / _get_pro / 五个 fetch 函数），
零真实 API 调用；全局标记 _TUSHARE_NEWS_DENIED 用 monkeypatch 隔离。
"""

from unittest.mock import MagicMock, patch, call

import pytest

from agent import data_fetcher
from agent.data_fetcher import _truncate_at_boundary, _sanitize_news_text


# ════════════════════════════════════════════════════════════════
# 1. _truncate_at_boundary：标点边界截断
# ════════════════════════════════════════════════════════════════


def _assert_boundary_cut(original: str, out: str, max_len: int):
    """断言 out 是 original 的边界截断：原文前缀 + 截点是标点 + …结尾。"""
    assert out.endswith("…"), f"截断结果应以 … 结尾: {out!r}"
    body = out[:-1]
    assert original.startswith(body), f"截断结果不是原文前缀: {out!r}"
    assert len(body) <= max_len, f"截断后仍超长: {len(body)} > {max_len}"
    nxt = original[len(body):len(body) + 1]
    assert nxt in "。！？!?；;，,、：:.", (
        f"截断点不在标点边界（拦腰截断），下一字符: {nxt!r}\n结果: {out!r}"
    )


class TestTruncateAtBoundary:

    def test_short_text_returned_as_is(self):
        assert _truncate_at_boundary("完整短标题", 80) == "完整短标题"

    def test_exact_length_returned_as_is(self):
        text = "字" * 80
        assert _truncate_at_boundary(text, 80) == text

    def test_cuts_at_last_sentence_end(self):
        """超限文本在最后一句句末截断 + …，不出现半句话。"""
        text = "央行开展5000亿元MLF操作。流动性合理充裕。" + "后续" * 100
        out = _truncate_at_boundary(text, 30)
        # 截断点必须是完整句子（30字窗口内最后一个句末标点是「充裕」后的「。」）
        assert out == "央行开展5000亿元MLF操作。流动性合理充裕…"
        _assert_boundary_cut(text, out, 30)

    def test_falls_back_to_clause_punctuation(self):
        """无句末标点时退到分句标点（，、：）。"""
        text = "上海地区生产总值同比增长，增速居全国前列" + "数据" * 100
        out = _truncate_at_boundary(text, 30)
        assert out == "上海地区生产总值同比增长…"
        _assert_boundary_cut(text, out, 30)

    def test_prefers_latest_boundary_in_window(self):
        """窗口内多个边界时取最靠后的，尽量保留内容。"""
        text = "第一句。第二句，第三句更长一些。" + "尾巴" * 100
        out = _truncate_at_boundary(text, 20)
        assert out == "第一句。第二句，第三句更长一些…"

    def test_no_punctuation_hard_cut(self):
        """无标点时硬切（与旧行为一致），追加 …。"""
        text = "正" * 600
        out = _truncate_at_boundary(text, 500)
        assert len(out) == 501 and out.endswith("…")
        assert out[:-1] == text[:500]

    def test_too_early_punctuation_ignored(self):
        """标点位置过靠前（< max_len 的 1/3）会导致片段过短，跳过它硬切。"""
        text = "短。然后是一长段没有任何标点的内容" + "字" * 200
        out = _truncate_at_boundary(text, 100)
        # 「。」在第 1 位（< 33），不应在此处截断
        assert out != "短…"
        assert out.endswith("…") and len(out[:-1]) == 100

    def test_english_sentence_end(self):
        text = "The Fed held rates steady. Markets rallied strongly " + "on " * 100
        out = _truncate_at_boundary(text, 40)
        assert out == "The Fed held rates steady…"

    def test_decimal_point_not_a_boundary(self):
        """小数点不算句末边界（2788.5 不能切成 2788…）。"""
        text = "上海GDP达2788.5亿元同比增长" + "数据" * 100
        out = _truncate_at_boundary(text, 20)
        assert not out.endswith("2788…"), f"小数点被当作句末边界: {out!r}"

    def test_non_str_and_bad_max_len(self):
        assert _truncate_at_boundary(None, 80) == ""
        assert _truncate_at_boundary("", 80) == ""
        assert _truncate_at_boundary("abc", 0) == "abc"
        assert _truncate_at_boundary("abc", -1) == "abc"


class TestSanitizeBoundaryTruncation:

    def test_sanitize_max_len_uses_boundary(self):
        """净化截断不再拦腰：带标点长文本在句末结束。"""
        text = "证监会召开系统工作会议，部署明年重点工作。" + "内容" * 300
        out = _sanitize_news_text(text, max_len=500)
        assert out.endswith("…")
        assert "，部署明年重点工作" in out  # 第一逗号句完整保留
        # 不应在「内容内容内容…」中段截出半个词就结束（截点应贴近 500 但有标点优先）
        assert len(out) <= 501

    def test_sanitize_hard_cut_unchanged_without_punctuation(self):
        """无标点长文本行为与旧版一致：500 硬切 + …。"""
        text = "正" * 600
        out = _sanitize_news_text(text, max_len=500)
        assert len(out) == 501 and out[:-1] == text[:500]


# ════════════════════════════════════════════════════════════════
# 2. fetch_tushare_news：权限标记与配额保护
# ════════════════════════════════════════════════════════════════


class TestTushareNewsPermission:

    PERM_MSG = "抱歉，您没有接口(news)访问权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。"

    def test_permission_error_sets_denied_and_skips_fallback(self, monkeypatch):
        """权限异常 → 置 denied 标记、返回 []，且不再降级尝试 sina 源（同权限白试）。"""
        monkeypatch.setattr(data_fetcher, "_TUSHARE_NEWS_DENIED", False)
        pro = MagicMock()
        pro.news.side_effect = Exception(self.PERM_MSG)
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            items = data_fetcher.fetch_tushare_news("20260721", 40)
        assert items == []
        assert data_fetcher._TUSHARE_NEWS_DENIED is True
        # cls 源抛权限错后只调用了 1 次（没有再走 sina 源降级）
        assert pro.news.call_count == 1

    def test_denied_flag_short_circuits_later_calls(self, monkeypatch):
        """denied 标记置位后，后续调用直接返回 []，完全不再触碰 tushare（省配额）。"""
        monkeypatch.setattr(data_fetcher, "_TUSHARE_NEWS_DENIED", True)
        with patch.object(data_fetcher, "_get_pro") as get_pro:
            assert data_fetcher.fetch_tushare_news("20260721", 40) == []
            assert data_fetcher.fetch_tushare_news("20260722", 40) == []
        get_pro.assert_not_called()

    def test_transient_error_does_not_set_flag(self, monkeypatch):
        """普通网络异常不置标记，且照常降级尝试 sina 源。"""
        monkeypatch.setattr(data_fetcher, "_TUSHARE_NEWS_DENIED", False)
        pro = MagicMock()
        pro.news.side_effect = TimeoutError("connection timeout")
        with patch.object(data_fetcher, "_get_pro", return_value=pro):
            items = data_fetcher.fetch_tushare_news("20260721", 40)
        assert items == []
        assert data_fetcher._TUSHARE_NEWS_DENIED is False
        # cls 失败 + sina 降级各一次
        assert pro.news.call_count == 2


# ════════════════════════════════════════════════════════════════
# 3. fetch_mcp_news：content 边界截断 + 翻页 + 去重
# ════════════════════════════════════════════════════════════════


def _mcp_payload(rows):
    return {"result": {"data": {"data": rows}}}


class TestMcpNews:

    def test_title_fallback_uses_boundary_not_mid_sentence(self, monkeypatch):
        """无 title 条目：标题取 content 标点边界截断，完整 content 另存。"""
        monkeypatch.setenv("SINA_MCP_TOKEN", "fake-token")
        long_content = (
            "上海地区生产总值（GDP）达2788亿元，同比增长5.2%。"
            "其中金融业增加值贡献显著，银行业存贷款余额双升。" + "补充" * 100
        )
        rows = [{"title": "", "content": long_content, "ctime": "2026-07-22 09:00:00"}]
        with patch.object(data_fetcher, "_mcp_call", return_value=_mcp_payload(rows)):
            items = data_fetcher.fetch_mcp_news("银行", 20)
        assert len(items) == 1
        _assert_boundary_cut(long_content, items[0]["title"], 80)
        # 完整 content 保留供行业关键词匹配
        assert items[0]["content"].startswith("上海地区生产总值")
        assert len(items[0]["content"]) <= 501

    def test_short_content_used_verbatim_without_ellipsis(self, monkeypatch):
        monkeypatch.setenv("SINA_MCP_TOKEN", "fake-token")
        rows = [{"title": "", "content": "央行开展5000亿元MLF操作", "ctime": "2026-07-22 09:00:00"}]
        with patch.object(data_fetcher, "_mcp_call", return_value=_mcp_payload(rows)):
            items = data_fetcher.fetch_mcp_news("银行", 20)
        assert items[0]["title"] == "央行开展5000亿元MLF操作"
        assert not items[0]["title"].endswith("…")

    def test_pagination_and_cross_page_dedup(self, monkeypatch):
        """limit=40 → 翻 2 页；跨页重复标题只保留一条。"""
        monkeypatch.setenv("SINA_MCP_TOKEN", "fake-token")
        monkeypatch.setattr(data_fetcher.time, "sleep", lambda s: None)

        def fake_call(name, args):
            assert name == "qNewsSearch"
            if args["page"] == 1:
                return _mcp_payload([
                    {"title": f"第1页新闻{i}号", "ctime": "2026-07-22 09:00:00"}
                    for i in range(20)
                ])
            return _mcp_payload(
                [{"title": "第1页新闻0号", "ctime": "2026-07-22 09:01:00"}]  # 跨页重复
                + [{"title": f"第2页新闻{i}号", "ctime": "2026-07-22 09:02:00"} for i in range(19)]
            )

        with patch.object(data_fetcher, "_mcp_call", side_effect=fake_call) as m:
            items = data_fetcher.fetch_mcp_news("银行", 40)
        assert m.call_count == 2
        titles = [it["title"] for it in items]
        assert len(titles) == len(set(titles)), f"跨页重复标题未去重: {titles}"
        assert len(items) == 39  # 40 条候选 - 1 条跨页重复

    def test_early_stop_when_page_not_full(self, monkeypatch):
        """某页返回不足 20 条 → 判定没有更多结果，不再继续翻页。"""
        monkeypatch.setenv("SINA_MCP_TOKEN", "fake-token")
        rows = [{"title": f"快讯{i}号", "ctime": "2026-07-22 09:00:00"} for i in range(5)]
        with patch.object(data_fetcher, "_mcp_call", return_value=_mcp_payload(rows)) as m:
            items = data_fetcher.fetch_mcp_news("银行", 60)
        assert len(items) == 5
        assert m.call_count == 1, f"不足页应提前停止，实际调用 {m.call_count} 次"

    def test_empty_result_stops_immediately(self, monkeypatch):
        monkeypatch.setenv("SINA_MCP_TOKEN", "fake-token")
        with patch.object(data_fetcher, "_mcp_call", return_value=_mcp_payload([])) as m:
            assert data_fetcher.fetch_mcp_news("银行", 60) == []
        assert m.call_count == 1


# ════════════════════════════════════════════════════════════════
# 4. fetch_cls_telegraph：brief 边界截断 + summary 导出 + HTTP 容错
# ════════════════════════════════════════════════════════════════


def _mock_cls_resp(roll, errno=0, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"errno": errno, "data": {"roll_data": roll}}
    return resp


class TestClsTelegraph:

    def test_brief_fallback_boundary_truncation_and_summary(self, monkeypatch):
        monkeypatch.setattr(data_fetcher.time, "sleep", lambda s: None)
        long_brief = (
            "财联社7月22日电，央行今日开展5000亿元中期借贷便利操作，"
            "中标利率持平于2.5%。银行业流动性保持合理充裕。" + "扩展" * 100
        )
        roll = [{"title": "", "brief": long_brief, "ctime": 1784196000}]
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = _mock_cls_resp(roll)
            items = data_fetcher.fetch_cls_telegraph(20)
        assert len(items) == 1
        _assert_boundary_cut(long_brief, items[0]["title"], 80)
        # summary 与 brief 同义导出，供行业关键词匹配
        assert items[0]["summary"] == items[0]["brief"]
        assert items[0]["summary"].startswith("财联社7月22日电")

    def test_short_brief_verbatim(self, monkeypatch):
        monkeypatch.setattr(data_fetcher.time, "sleep", lambda s: None)
        roll = [{"title": "", "brief": "央行宣布降准0.5个百分点", "ctime": 1784196000}]
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = _mock_cls_resp(roll)
            items = data_fetcher.fetch_cls_telegraph(20)
        assert items[0]["title"] == "央行宣布降准0.5个百分点"
        assert not items[0]["title"].endswith("…")

    def test_http_error_degrades_safely(self):
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = _mock_cls_resp([], status=403)
            assert data_fetcher.fetch_cls_telegraph(20) == []


# ════════════════════════════════════════════════════════════════
# 5. fetch_eastmoney_news：标题完整保留 + HTTP 容错
# ════════════════════════════════════════════════════════════════


def _mock_em(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"data": {"fastNewsList": payload}}
    return resp


class TestEastmoneyNews:

    def test_title_kept_complete(self):
        """正常长度标题原样保留（不再 [:150] 预切）。"""
        payload = [{"showTime": "2026-07-22 09:00:00",
                    "title": "央行开展5000亿元MLF操作，利率持平", "summary": "摘要"}]
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = _mock_em(payload)
            items = data_fetcher.fetch_eastmoney_news(5)
        assert items[0]["title"] == "央行开展5000亿元MLF操作，利率持平"

    def test_long_title_boundary_truncated(self):
        """超长标题（>150）在标点边界截断而非硬切。"""
        long_title = "【快讯】" + "详细报道，" * 60 + "完"
        payload = [{"showTime": "2026-07-22 09:00:00", "title": long_title, "summary": ""}]
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = _mock_em(payload)
            items = data_fetcher.fetch_eastmoney_news(5)
        title = items[0]["title"]
        assert len(title) <= 151
        assert title.endswith("…")
        assert not title[:-1].endswith("报"), f"标题在句子中间被切断: {title[-20:]!r}"

    def test_http_error_degrades_safely(self):
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = _mock_em([], status=451)
            assert data_fetcher.fetch_eastmoney_news(5) == []

    def test_page2_http_error_degrades_safely(self):
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = _mock_em([], status=500)
            assert data_fetcher.fetch_eastmoney_news_page2(5) == []


# ════════════════════════════════════════════════════════════════
# 5b. fetch_sina_news：时间取真实 ctime（date 参数失效回归）
# ════════════════════════════════════════════════════════════════


class TestSinaNewsTime:

    def _mock_resp(self, rows):
        resp = MagicMock()
        resp.json.return_value = {"result": {"data": rows}}
        return resp

    def test_time_uses_real_ctime_not_query_date(self):
        """接口 date 参数失效（按日查询返回当天滚动），time 必须取真实 ctime，
        避免把今天的新闻错标到历史日期（48小时分组失真根因）。"""
        rows = [{"title": "央行开展5000亿元MLF操作", "ctime": "1753142400"}]  # 2025-07-22 08:00
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = self._mock_resp(rows)
            items = data_fetcher.fetch_sina_news(5, "2025-07-20")  # 查询历史日期
        assert len(items) == 1
        assert items[0]["time"].startswith("2025-07-22"), (
            f"应使用真实 ctime 而非查询日期: {items[0]['time']!r}"
        )

    def test_missing_ctime_falls_back_to_query_date(self):
        rows = [{"title": "央行开展5000亿元MLF操作"}]
        with patch.object(data_fetcher, "requests") as mreq:
            mreq.get.return_value = self._mock_resp(rows)
            items = data_fetcher.fetch_sina_news(5, "2025-07-20")
        assert items[0]["time"] == "2025-07-20"


# ════════════════════════════════════════════════════════════════
# 6. fetch_news_pool：板块模式加深 + 字段保留 + 旧源剔除
# ════════════════════════════════════════════════════════════════


class TestNewsPoolSectorMode:

    def _patch_all(self, monkeypatch, mcp=None, flash=None):
        monkeypatch.setattr(data_fetcher, "fetch_mcp_news",
                            MagicMock(return_value=mcp or []))
        monkeypatch.setattr(data_fetcher, "fetch_mcp_flash",
                            MagicMock(return_value=flash or []))

    def test_sector_mode_deepens_sources(self, monkeypatch):
        """板块查询：智研双源加深抓取（各 60 条），mcp 以行业关键词搜索。"""
        self._patch_all(monkeypatch)
        pool = data_fetcher.fetch_news_pool(sector_keywords=["银行"], days=3)

        data_fetcher.fetch_mcp_news.assert_called_once_with("银行", 60)
        data_fetcher.fetch_mcp_flash.assert_called_once_with(60)
        assert set(pool.keys()) == {"mcp", "flash"}

    def test_non_sector_mode_uses_shallow_limit(self, monkeypatch):
        """全市场查询：不加深，维持默认上限（各 30 条），mcp 用 "A股" 搜索词。"""
        self._patch_all(monkeypatch)
        data_fetcher.fetch_news_pool()

        data_fetcher.fetch_mcp_news.assert_called_once_with("A股", 30)
        data_fetcher.fetch_mcp_flash.assert_called_once_with(30)

    def test_unified_entries_preserve_body_fields(self, monkeypatch):
        """统一条目保留 content/summary/brief，供下游行业关键词匹配（误杀修复）。"""
        mcp = [{"title": "银行快讯标题", "time": "2026-07-22 11:00:00",
                "source": "智研", "content": "完整正文内容"}]
        flash = [{"title": "智研快讯：降准落地", "time": "2026-07-22 10:00:00",
                  "source": "智研快讯", "brief": "释放长期资金", "summary": "释放长期资金"}]
        self._patch_all(monkeypatch, mcp=mcp, flash=flash)
        pool = data_fetcher.fetch_news_pool(sector_keywords=["银行"], days=3)

        assert pool["mcp"][0]["content"] == "完整正文内容"
        assert pool["flash"][0]["brief"] == "释放长期资金"
        assert pool["flash"][0]["summary"] == "释放长期资金"

    def test_deprecated_sources_not_called(self, monkeypatch):
        """东财/新浪/财联社/Tushare 已弃用：池子任何模式下都不调用这些源。"""
        self._patch_all(monkeypatch)
        for fname in ("fetch_sina_news", "fetch_eastmoney_news",
                      "fetch_eastmoney_news_page2", "fetch_cls_telegraph",
                      "fetch_tushare_news"):
            monkeypatch.setattr(data_fetcher, fname, MagicMock(return_value=[]))
        pool = data_fetcher.fetch_news_pool(sector_keywords=["银行"], days=3)

        for fname in ("fetch_sina_news", "fetch_eastmoney_news",
                      "fetch_eastmoney_news_page2", "fetch_cls_telegraph",
                      "fetch_tushare_news"):
            getattr(data_fetcher, fname).assert_not_called()
        assert set(pool.keys()) == {"mcp", "flash"}

    def test_sector_pool_covers_two_days(self, monkeypatch):
        """48小时覆盖回归：两天的新闻都进池（修复『只覆盖一天』的数据层前提）。"""
        mcp = [{"title": "今日银行快讯", "time": "2026-07-22 09:00:00", "source": "智研"}]
        flash = [{"title": "昨日银行快讯", "time": "2026-07-21 15:00:00", "source": "智研快讯"}]
        self._patch_all(monkeypatch, mcp=mcp, flash=flash)
        pool = data_fetcher.fetch_news_pool(sector_keywords=["银行"], days=3)
        days = {it["time"][:10] for items in pool.values() for it in items}
        assert days == {"2026-07-22", "2026-07-21"}
