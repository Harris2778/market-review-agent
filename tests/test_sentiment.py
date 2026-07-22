"""agent/sentiment.py 社交情绪采集与打分层测试（全 mock 零网络）。

覆盖范围：
1. 人气榜 fetch_hot_rank：实测形态（data 直 list、sc 带 SH/SZ 前缀、无 name、
   hisRc→rank_change、无 total 翻页终止）+ 旧嵌套形态兼容、字段缺失逐条跳过、
   HTTP 异常/5xx/非法 JSON 降级、fetch_stock_names 名称回填与失败降级。
2. 名称回填 fetch_stock_names：secid 市场前缀映射、data.diff list/dict 双形态、
   50 一批分批限速、单批失败保留已解析部分。
3. 人气历史 fetch_hot_rank_history：srcSecurityCode 带市场前缀（实测定案）、
   calcTime 日期键、data 直 list + 嵌套双形态、days 截尾、非法代码入参。
4. 涨跌停池 fetch_limit_up_pools：三池字段映射、炸板率/最高连板、时钟注入、
   单池/全部失败降级 + note、限速 sleep。
5. 词典打分 score_news_sentiment：利好/利空/中性、同词只计一次、饱和截断、
   自定义 scorer 注入与异常降级、非 dict 跳过、原字段保留。
6. 温度公式 _calc_temperature 确定性（手算期望值）与标签分档边界。
7. get_market_sentiment：子源组合、当日缓存命中（不重复 HTTP）、降级 notes。
8. get_stock_sentiment：在榜/不在榜、历史回退、trend 三向判定、全失败不抛。

人气/历史/名称端点响应 fixture 按 2026-07-22 实测定案结构构造（同时保留
旧嵌套形态兼容用例）；涨跌停池 fixture 为实测正常结构。
所有 HTTP 由 fake http_get/http_post 注入，sleep 由 fake 注入，绝不触达真实网络。
"""

import json
from datetime import date

import pytest

import agent.sentiment as st


# ── 公共工具 ──

class FakeResp:
    """最小 response 替身：status_code + JSON body。"""

    def __init__(self, payload=None, status_code=200, raw_text=None):
        self.status_code = status_code
        self._payload = payload
        if raw_text is not None:
            self.text = raw_text
        else:
            self.text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def make_http(handler, calls=None):
    """构造 fake http_get/http_post：handler(url, kw) -> FakeResp 或抛异常。"""
    def http(url, **kw):
        if calls is not None:
            calls.append((url, kw))
        return handler(url, kw)
    return http


def make_sleep(record):
    """构造 fake sleep：记录每次休眠秒数。"""
    def _sleep(seconds):
        record.append(seconds)
    return _sleep


@pytest.fixture(autouse=True)
def _reset_cache():
    """每个用例前后清空市场缓存，避免跨用例污染。"""
    st._clear_market_cache()
    yield
    st._clear_market_cache()


# ── fixture：2026-07-22 实测定案结构 + 旧嵌套形态（兼容用例）──

def hot_rank_page_nested(rows, total=None):
    """旧公开文档嵌套形态：data.data + total（兼容用例）。"""
    if total is None:
        total = len(rows)
    return {"rc": 0, "rt": 11, "data": {"data": rows, "total": total}}


def hot_rank_page_real(rows):
    """实测形态（2026-07-22）：payload['data'] 直接为 list，无 total。"""
    return {"rc": 0, "rt": 11, "data": rows}


def hot_rank_row(sc, n, rk):
    """旧形态条目（带 name，无 hisRc）。"""
    return {"sc": sc, "n": n, "rk": rk, "pct": 1.23}


def hot_rank_row_real(sc, rk, his_rc=None):
    """实测条目：sc 带 SH/SZ 前缀、无 name、hisRc 为排名变化（可选）。"""
    row = {"sc": sc, "rk": rk, "rc": 0}
    if his_rc is not None:
        row["hisRc"] = his_rc
    return row


# 实测形态历史：data 直 list，日期键 calcTime、rank 键 rank
HIS_PAYLOAD_REAL = {"status": 0, "data": [
    {"calcTime": "2025-01-06", "rank": 40},
    {"calcTime": "2025-01-07", "rank": 30},
    {"calcTime": "2025-01-08", "rank": 20},
    {"calcTime": "2025-01-09", "rank": 10},
    {"calcTime": "2025-01-10", "rank": 5},
]}

# 旧嵌套形态历史：data.data，日期键 d、rank 键 rk
HIS_PAYLOAD_NESTED = {"rc": 0, "data": {"data": [
    {"d": "2025-01-09", "rk": 10},
    {"d": "2025-01-10", "rk": 5},
]}}


def pool_payload(rows):
    return {"rc": 0, "rt": 17, "data": {"pool": rows, "tc": len(rows)}}


ZT_ROWS = [
    {"c": "600519", "n": "贵州茅台", "p": 1680000, "zdp": 10.0,
     "fbt": "093001", "lbt": "093001", "fund": 5.2e8, "hybk": "白酒", "lbc": 3},
    {"c": "000001", "n": "平安银行", "p": 12345, "zdp": 9.98,
     "fbt": "093500", "lbt": "140000", "fund": 2.1e8, "hybk": "银行", "lbc": 1},
]
DT_ROWS = [
    {"c": "002594", "n": "比亚迪", "p": 250000, "zdp": -10.0,
     "fund": 1.0e8, "hybk": "汽车整车"},
]
ZB_ROWS = [
    {"c": "300750", "n": "宁德时代", "p": 180000, "zdp": 5.5,
     "fbt": "094000", "lbt": "103000", "fund": 8.0e7, "hybk": "电池", "zbc": 2},
]


def pools_http(zt=ZT_ROWS, dt=DT_ROWS, zb=ZB_ROWS, fail=None, calls=None):
    """构造三池 fake http_get；fail 可传 URL 子串集合强制抛异常。"""
    fail = fail or set()

    def handler(url, kw):
        if any(key in url for key in fail):
            raise ConnectionError("模拟网络故障")
        if "getTopicZTPool" in url:
            return FakeResp(pool_payload(zt))
        if "getTopicDTPool" in url:
            return FakeResp(pool_payload(dt))
        if "getTopicZBPool" in url:
            return FakeResp(pool_payload(zb))
        raise AssertionError(f"未预期 URL: {url}")

    return make_http(handler, calls)


def no_backfill(monkeypatch):
    """关闭名称回填（测试排名解析本身时不依赖名称接口）。"""
    monkeypatch.setattr(st, "fetch_stock_names", lambda codes, **kw: {})


# ═══════════════════════════════════════════
# 0. 市场前缀推导
# ═══════════════════════════════════════════

class TestMarketPrefix:
    @pytest.mark.parametrize("code,expect", [
        ("600519", "SH"), ("688981", "SH"), ("900901", "SH"),
        ("000001", "SZ"), ("300750", "SZ"), ("200011", "SZ"),
        ("430047", "BJ"), ("830799", "BJ"), ("920001", "BJ"),
        ("110000", "SH"),   # 拿不准的段默认 SH（规则见函数注释）
    ])
    def test_prefix_rules(self, code, expect):
        assert st._market_prefix(code) == expect


# ═══════════════════════════════════════════
# 1. fetch_hot_rank
# ═══════════════════════════════════════════

class TestFetchHotRank:
    def test_parse_real_structure(self, monkeypatch):
        """实测形态：data 直 list、sc 带前缀、hisRc→rank_change、名称回填。"""
        monkeypatch.setattr(st, "fetch_stock_names",
                            lambda codes, **kw: {"000938": "中油资本"})
        rows = [hot_rank_row_real("SZ000938", 1, his_rc=12),
                hot_rank_row_real("SH600519", 2)]
        http_post = make_http(lambda url, kw: FakeResp(hot_rank_page_real(rows)))
        result = st.fetch_hot_rank(limit=10, http_post=http_post, sleep=make_sleep([]))
        assert result[0] == {"rank": 1, "code": "000938", "name": "中油资本",
                             "source": "eastmoney_hotrank", "rank_change": 12}
        # 无 hisRc 的条目不携带 rank_change 键；名称接口未返回则留空串
        assert result[1] == {"rank": 2, "code": "600519", "name": "",
                             "source": "eastmoney_hotrank"}
        assert "rank_change" not in result[1]

    def test_parse_nested_structure_compat(self, monkeypatch):
        """旧嵌套形态（data.data + total + 条目带 name）仍兼容，不触发回填。"""
        called = []
        monkeypatch.setattr(st, "fetch_stock_names",
                            lambda codes, **kw: called.append(codes) or {})
        rows = [hot_rank_row("1.600519", "贵州茅台", 1),
                hot_rank_row("0.000001", "平安银行", 2)]
        http_post = make_http(lambda url, kw: FakeResp(hot_rank_page_nested(rows)))
        result = st.fetch_hot_rank(limit=10, http_post=http_post, sleep=make_sleep([]))
        assert result == [
            {"rank": 1, "code": "600519", "name": "贵州茅台", "source": "eastmoney_hotrank"},
            {"rank": 2, "code": "000001", "name": "平安银行", "source": "eastmoney_hotrank"},
        ]
        assert called == []          # 名称齐全时不调用回填

    def test_backfill_failure_keeps_empty_names(self, monkeypatch, caplog):
        def boom(codes, **kw):
            raise RuntimeError("名称接口挂了")
        monkeypatch.setattr(st, "fetch_stock_names", boom)
        rows = [hot_rank_row_real("SZ000938", 1)]
        http_post = make_http(lambda url, kw: FakeResp(hot_rank_page_real(rows)))
        with caplog.at_level("WARNING"):
            result = st.fetch_hot_rank(http_post=http_post, sleep=make_sleep([]))
        assert result[0]["name"] == ""
        assert result[0]["code"] == "000938"     # 排名数据不受回填失败影响
        assert any("名称回填失败" in r.message for r in caplog.records)

    def test_pagination_real_no_total_stops_on_short_page(self, monkeypatch):
        """实测无 total：首页不足 pageSize 即判定最后一页，不再请求第二页。"""
        no_backfill(monkeypatch)
        calls = []
        rows = [hot_rank_row_real(f"SH6000{i:02d}", i + 1) for i in range(10)]
        http_post = make_http(lambda url, kw: FakeResp(hot_rank_page_real(rows)), calls)
        result = st.fetch_hot_rank(limit=13, http_post=http_post, sleep=make_sleep([]))
        assert [r["rank"] for r in result] == list(range(1, 11))
        assert len(calls) == 1              # 首页 10 条 < pageSize=13 → 终止

    def test_pagination_real_full_page_continues(self, monkeypatch):
        """实测无 total：首页满 pageSize 继续翻页，凑满 limit 后截断。"""
        no_backfill(monkeypatch)
        calls = []
        pages = {
            1: hot_rank_page_real([hot_rank_row_real(f"SH60{i:04d}", i + 1)
                                   for i in range(100)]),
            2: hot_rank_page_real([hot_rank_row_real(f"SZ00{i:04d}", 101 + i)
                                   for i in range(30)]),
        }

        def handler(url, kw):
            return FakeResp(pages[kw["json"]["pageNo"]])

        sleeps = []
        result = st.fetch_hot_rank(limit=120, http_post=make_http(handler, calls),
                                   sleep=make_sleep(sleeps))
        assert [r["rank"] for r in result] == list(range(1, 121))
        assert len(calls) == 2              # 首页满 100 条 → 翻第二页
        assert len(sleeps) == 1

    def test_pagination_real_empty_page_terminates(self, monkeypatch):
        no_backfill(monkeypatch)
        calls = []
        pages = {1: hot_rank_page_real([hot_rank_row_real("SH600519", 1)]),
                 2: hot_rank_page_real([])}
        # limit 放大迫使尝试翻页；首页 1 条 == pageSize=1 时需继续探测
        def handler(url, kw):
            return FakeResp(pages[kw["json"]["pageNo"]])
        result = st.fetch_hot_rank(limit=1, http_post=make_http(handler, calls),
                                   sleep=make_sleep([]))
        assert [r["rank"] for r in result] == [1]
        assert len(calls) == 1              # limit 已达，不再请求第二页

    def test_pagination_nested_total_terminates(self, monkeypatch):
        """旧嵌套形态仍按 total 终止。"""
        no_backfill(monkeypatch)
        calls = []
        pages = {
            1: hot_rank_page_nested([hot_rank_row("1.600519", "A", 1),
                                     hot_rank_row("1.600000", "B", 2)], total=4),
            2: hot_rank_page_nested([hot_rank_row("0.000001", "C", 3),
                                     hot_rank_row("0.000002", "D", 4)], total=4),
        }

        def handler(url, kw):
            return FakeResp(pages[kw["json"]["pageNo"]])

        http_post = make_http(handler, calls)
        sleeps = []
        result = st.fetch_hot_rank(limit=3, http_post=http_post, sleep=make_sleep(sleeps))
        assert [r["rank"] for r in result] == [1, 2, 3]
        assert len(calls) == 2
        assert calls[1][1]["json"]["pageNo"] == 2
        assert len(sleeps) == 1

    def test_payload_structure(self, monkeypatch):
        no_backfill(monkeypatch)
        calls = []
        http_post = make_http(lambda url, kw: FakeResp(hot_rank_page_real([])), calls)
        st.fetch_hot_rank(limit=5, http_post=http_post, sleep=make_sleep([]))
        url, kw = calls[0]
        assert url == st.EM_HOT_RANK_URL
        payload = kw["json"]
        assert payload["appId"] == "appId01"
        assert payload["globalId"] == "786e4c21-70dc-435a-93bb-38"
        assert payload["marketType"] == ""
        assert payload["pageNo"] == 1
        assert payload["pageSize"] == 5

    def test_missing_fields_skipped(self, monkeypatch):
        no_backfill(monkeypatch)
        rows = [
            hot_rank_row_real("SZ000938", 1),
            {"rk": 2},                              # 缺 sc
            {"sc": "XYZ", "rk": 4},                 # 代码正则提取失败
            {"sc": "SH601398"},                     # 缺 rk
            "不是字典",
        ]
        http_post = make_http(lambda url, kw: FakeResp(hot_rank_page_real(rows)))
        result = st.fetch_hot_rank(limit=10, http_post=http_post, sleep=make_sleep([]))
        assert [(r["rank"], r["code"]) for r in result] == [(1, "000938")]

    def test_empty_data_returns_empty(self, monkeypatch):
        no_backfill(monkeypatch)
        http_post = make_http(lambda url, kw: FakeResp({"rc": 0, "data": []}))
        assert st.fetch_hot_rank(http_post=http_post, sleep=make_sleep([])) == []

    def test_data_not_list_or_dict_returns_empty(self, monkeypatch):
        no_backfill(monkeypatch)
        http_post = make_http(lambda url, kw: FakeResp({"rc": 0, "data": None}))
        assert st.fetch_hot_rank(http_post=http_post, sleep=make_sleep([])) == []

    def test_http_exception_returns_empty(self, monkeypatch, caplog):
        no_backfill(monkeypatch)
        def boom(url, kw):
            raise ConnectionError("断网")
        with caplog.at_level("WARNING"):
            result = st.fetch_hot_rank(http_post=make_http(boom), sleep=make_sleep([]))
        assert result == []
        assert any("hotrank" in r.message for r in caplog.records)

    def test_http_500_returns_empty(self, monkeypatch):
        no_backfill(monkeypatch)
        http_post = make_http(lambda url, kw: FakeResp(None, status_code=500))
        assert st.fetch_hot_rank(http_post=http_post, sleep=make_sleep([])) == []

    def test_invalid_json_returns_empty(self, monkeypatch):
        no_backfill(monkeypatch)
        http_post = make_http(lambda url, kw: FakeResp(None, raw_text="<html>错误页</html>"))
        assert st.fetch_hot_rank(http_post=http_post, sleep=make_sleep([])) == []

    def test_never_raises_on_weird_payload(self, monkeypatch):
        no_backfill(monkeypatch)
        http_post = make_http(lambda url, kw: FakeResp({"data": {"data": {"not": "list"}}}))
        assert st.fetch_hot_rank(http_post=http_post, sleep=make_sleep([])) == []


# ═══════════════════════════════════════════
# 2. fetch_stock_names（名称批量回填）
# ═══════════════════════════════════════════

class TestFetchStockNames:
    def test_secid_prefix_mapping(self):
        calls = []
        payload = {"data": {"diff": [
            {"f12": "600519", "f14": "贵州茅台"},
            {"f12": "000001", "f14": "平安银行"},
            {"f12": "830799", "f14": "艾融软件"},
        ]}}
        http_get = make_http(lambda url, kw: FakeResp(payload), calls)
        result = st.fetch_stock_names(["600519", "000001", "830799"],
                                      http_get=http_get, sleep=make_sleep([]))
        assert result == {"600519": "贵州茅台", "000001": "平安银行",
                          "830799": "艾融软件"}
        url, kw = calls[0]
        assert url == st.EM_ULIST_URL
        assert kw["params"]["fields"] == "f12,f14"
        assert kw["params"]["secids"] == "1.600519,0.000001,0.830799"

    def test_diff_dict_form_compat(self):
        payload = {"data": {"diff": {
            "600519": {"f12": "600519", "f14": "贵州茅台"},
            "000001": {"f12": "000001", "f14": "平安银行"},
        }}}
        http_get = make_http(lambda url, kw: FakeResp(payload))
        result = st.fetch_stock_names(["600519", "000001"],
                                      http_get=http_get, sleep=make_sleep([]))
        assert result == {"600519": "贵州茅台", "000001": "平安银行"}

    def test_batching_over_50(self):
        calls = []
        sleeps = []
        codes = [f"6000{i:02d}" for i in range(60)]

        def handler(url, kw):
            n = len(kw["params"]["secids"].split(","))
            return FakeResp({"data": {"diff": [
                {"f12": sid.split(".")[1], "f14": f"股{sid}"}
                for sid in kw["params"]["secids"].split(",")]}})

        result = st.fetch_stock_names(codes, http_get=make_http(handler, calls),
                                      sleep=make_sleep(sleeps))
        assert len(calls) == 2               # 50 + 10 两批
        assert len(sleeps) == 1              # 首批不限速
        assert len(result) == 60

    def test_batch_failure_keeps_partial(self):
        calls = []
        codes = [f"6000{i:02d}" for i in range(60)]

        def handler(url, kw):
            if len(calls) > 1:               # 第二批抛异常（calls 先入列再回调）
                raise ConnectionError("故障")
            return FakeResp({"data": {"diff": [
                {"f12": sid.split(".")[1], "f14": "名称"}
                for sid in kw["params"]["secids"].split(",")]}})

        result = st.fetch_stock_names(codes, http_get=make_http(handler, calls),
                                      sleep=make_sleep([]))
        assert len(result) == 50             # 仅第一批解析成功

    def test_invalid_codes_filtered_and_empty(self):
        calls = []
        http_get = make_http(lambda url, kw: FakeResp({"data": {"diff": []}}), calls)
        assert st.fetch_stock_names(["abc", "", None], http_get=http_get,
                                    sleep=make_sleep([])) == {}
        assert calls == []                   # 无有效代码不发请求

    def test_http_error_returns_empty(self):
        http_get = make_http(lambda url, kw: FakeResp(None, status_code=500))
        assert st.fetch_stock_names(["600519"], http_get=http_get,
                                    sleep=make_sleep([])) == {}

    def test_rows_missing_fields_skipped(self):
        payload = {"data": {"diff": [
            {"f12": "600519"},                            # 缺 f14
            {"f14": "无名氏"},                            # 缺 f12
            {"f12": "000001", "f14": "平安银行"},
        ]}}
        http_get = make_http(lambda url, kw: FakeResp(payload))
        assert st.fetch_stock_names(["600519", "000001"], http_get=http_get,
                                    sleep=make_sleep([])) == {"000001": "平安银行"}


# ═══════════════════════════════════════════
# 3. fetch_hot_rank_history
# ═══════════════════════════════════════════

class TestFetchHotRankHistory:
    def test_parse_real_structure_and_days_tail(self):
        """实测形态：data 直 list + calcTime 日期键 + rank 键。"""
        http_post = make_http(lambda url, kw: FakeResp(HIS_PAYLOAD_REAL))
        result = st.fetch_hot_rank_history("600519", days=3,
                                           http_post=http_post, sleep=make_sleep([]))
        assert result == [
            {"date": "2025-01-08", "rank": 20, "code": "600519"},
            {"date": "2025-01-09", "rank": 10, "code": "600519"},
            {"date": "2025-01-10", "rank": 5, "code": "600519"},
        ]

    def test_nested_structure_compat(self):
        """旧嵌套形态（data.data + d/rk 键）仍兼容。"""
        http_post = make_http(lambda url, kw: FakeResp(HIS_PAYLOAD_NESTED))
        result = st.fetch_hot_rank_history("600519", http_post=http_post,
                                           sleep=make_sleep([]))
        assert result == [
            {"date": "2025-01-09", "rank": 10, "code": "600519"},
            {"date": "2025-01-10", "rank": 5, "code": "600519"},
        ]

    @pytest.mark.parametrize("code,expect", [
        ("600519", "SH600519"), ("1.600519", "SH600519"),
        ("000001", "SZ000001"), ("830799", "BJ830799"),
    ])
    def test_payload_uses_src_security_code_with_prefix(self, code, expect):
        calls = []
        http_post = make_http(lambda url, kw: FakeResp(HIS_PAYLOAD_REAL), calls)
        st.fetch_hot_rank_history(code, http_post=http_post, sleep=make_sleep([]))
        assert calls[0][0] == st.EM_HOT_RANK_HIS_URL
        assert calls[0][1]["json"]["srcSecurityCode"] == expect
        assert "stockCode" not in calls[0][1]["json"]

    def test_invalid_code_returns_empty(self, caplog):
        with caplog.at_level("WARNING"):
            assert st.fetch_hot_rank_history("不是代码", http_post=make_http(None)) == []
        assert any("非法代码" in r.message for r in caplog.records)

    def test_missing_fields_rows_skipped(self):
        payload = {"data": [
            {"calcTime": "2025-01-10", "rank": 5},
            {"calcTime": "2025-01-09"},               # 缺 rank
            {"rank": 3},                              # 缺 calcTime
            {"calcTime": "01-08", "rank": 2},         # 日期非法
        ]}
        http_post = make_http(lambda url, kw: FakeResp(payload))
        result = st.fetch_hot_rank_history("600519", http_post=http_post,
                                           sleep=make_sleep([]))
        assert result == [{"date": "2025-01-10", "rank": 5, "code": "600519"}]

    def test_http_failure_returns_empty(self):
        def boom(url, kw):
            raise TimeoutError("超时")
        http_post = make_http(boom)
        assert st.fetch_hot_rank_history("600519", http_post=http_post,
                                         sleep=make_sleep([])) == []


# ═══════════════════════════════════════════
# 4. fetch_limit_up_pools
# ═══════════════════════════════════════════

class TestFetchLimitUpPools:
    def test_parse_all_pools_and_stats(self):
        result = st.fetch_limit_up_pools(date="2025-01-10",
                                         http_get=pools_http(), sleep=make_sleep([]))
        assert result["date"] == "2025-01-10"
        assert result["source"] == "eastmoney_ztpool"
        assert result["notes"] == []
        zt0 = result["zt"][0]
        assert zt0["code"] == "600519"
        assert zt0["name"] == "贵州茅台"
        assert zt0["price"] == 1680.0            # p 千分之一还原
        assert zt0["pct"] == 10.0
        assert zt0["seal_amount"] == 5.2e8
        assert zt0["industry"] == "白酒"
        assert zt0["lianban"] == 3
        assert zt0["first_seal"] == "093001"
        stats = result["stats"]
        assert stats["zt_count"] == 2
        assert stats["dt_count"] == 1
        assert stats["zb_count"] == 1
        assert stats["炸板率"] == round(1 / 3 * 100, 2)   # zb / (zt + zb)
        assert stats["最高连板"] == 3

    def test_request_params(self):
        calls = []
        st.fetch_limit_up_pools(date="20250110", http_get=pools_http(calls=calls),
                                sleep=make_sleep([]))
        urls = [c[0] for c in calls]
        assert st.EM_ZT_POOL_URL in urls and st.EM_DT_POOL_URL in urls and st.EM_ZB_POOL_URL in urls
        params = calls[0][1]["params"]
        assert params["ut"] == "7eea3edcaed734bea9cbfc24409ed989"
        assert params["dpt"] == "wz.ztzt"
        assert params["date"] == "20250110"

    def test_date_none_uses_injected_clock(self, monkeypatch):
        monkeypatch.setattr(st, "_today_ymd", lambda: "20250601")
        calls = []
        result = st.fetch_limit_up_pools(http_get=pools_http(calls=calls),
                                         sleep=make_sleep([]))
        assert result["date"] == "2025-06-01"
        assert all(c[1]["params"]["date"] == "20250601" for c in calls)

    def test_single_pool_failure_degrades(self):
        result = st.fetch_limit_up_pools(date="2025-01-10",
                                         http_get=pools_http(fail={"getTopicZTPool"}),
                                         sleep=make_sleep([]))
        assert result["zt"] == []
        assert len(result["dt"]) == 1 and len(result["zb"]) == 1
        assert any("ztpool" in n for n in result["notes"])
        assert result["stats"]["zt_count"] == 0

    def test_all_pools_failure_degrades(self):
        result = st.fetch_limit_up_pools(
            date="2025-01-10",
            http_get=pools_http(fail={"ZTPool", "DTPool", "ZBPool"}),
            sleep=make_sleep([]))
        assert result["zt"] == [] and result["dt"] == [] and result["zb"] == []
        assert len(result["notes"]) == 3
        assert result["stats"] == {"zt_count": 0, "dt_count": 0, "zb_count": 0,
                                   "炸板率": 0.0, "最高连板": 0}

    def test_missing_fields_default(self):
        zt = [{"c": "600519", "n": "贵州茅台"}]   # 缺 p/zdp/fund/hybk/lbc/fbt/lbt
        result = st.fetch_limit_up_pools(date="2025-01-10",
                                         http_get=pools_http(zt=zt, dt=[], zb=[]),
                                         sleep=make_sleep([]))
        item = result["zt"][0]
        assert item["price"] is None
        assert item["pct"] == 0.0
        assert item["seal_amount"] == 0.0
        assert item["industry"] == ""
        assert item["lianban"] == 0
        assert result["stats"]["最高连板"] == 0

    def test_invalid_json_pool_degrades(self):
        def handler(url, kw):
            if "getTopicZTPool" in url:
                return FakeResp(None, raw_text="not json")
            return FakeResp(pool_payload([]))
        result = st.fetch_limit_up_pools(date="2025-01-10",
                                         http_get=make_http(handler),
                                         sleep=make_sleep([]))
        assert result["zt"] == []
        assert any("ztpool" in n for n in result["notes"])

    def test_empty_pools_stats(self):
        result = st.fetch_limit_up_pools(date="2025-01-10",
                                         http_get=pools_http(zt=[], dt=[], zb=[]),
                                         sleep=make_sleep([]))
        assert result["stats"]["炸板率"] == 0.0
        assert result["stats"]["最高连板"] == 0

    def test_rate_limit_sleeps_between_requests(self):
        sleeps = []
        st.fetch_limit_up_pools(date="2025-01-10", http_get=pools_http(),
                                sleep=make_sleep(sleeps))
        assert len(sleeps) == 2                  # 3 次 GET：首请求不限速
        assert all(s >= st.DEFAULT_RATE for s in sleeps)

    def test_row_without_code_skipped(self):
        zt = [{"n": "无代码"}, {"c": "600519", "n": "贵州茅台", "lbc": 2}]
        result = st.fetch_limit_up_pools(date="2025-01-10",
                                         http_get=pools_http(zt=zt, dt=[], zb=[]),
                                         sleep=make_sleep([]))
        assert [i["code"] for i in result["zt"]] == ["600519"]


# ═══════════════════════════════════════════
# 5. score_news_sentiment
# ═══════════════════════════════════════════

class TestScoreNewsSentiment:
    def test_bullish_text(self):
        items = [{"title": "贵州茅台涨停，业绩增长超预期，获北向资金净流入"}]
        result = st.score_news_sentiment(items)
        assert result[0]["sentiment"] == "利好"
        assert result[0]["sentiment_score"] > 0
        assert set(result[0]["hits"]) == {"涨停", "业绩增长", "超预期", "净流入"}

    def test_bearish_text(self):
        items = [{"title": "某公司爆雷遭立案调查，股价跌停，面临退市风险"}]
        result = st.score_news_sentiment(items)
        assert result[0]["sentiment"] == "利空"
        assert result[0]["sentiment_score"] < 0
        assert "爆雷" in result[0]["hits"] and "跌停" in result[0]["hits"]

    def test_neutral_no_hits(self):
        result = st.score_news_sentiment([{"title": "今日天气晴朗"}])
        assert result[0]["sentiment"] == "中性"
        assert result[0]["sentiment_score"] == 0.0
        assert result[0]["hits"] == []

    def test_mixed_offsets_by_weight(self):
        # 涨停(1.2) vs 减持(0.9) → 净 +0.3 / 3 = 0.1 → 阈值边界为中性
        result = st.score_news_sentiment([{"title": "涨停后大股东减持"}])
        assert result[0]["sentiment_score"] == round(0.3 / st.SENTI_NORM, 4)
        assert result[0]["sentiment"] == "中性"

    def test_saturation_clamps_to_one(self):
        title = " ".join(st.BULL_LEXICON.keys())   # 全部利好词
        result = st.score_news_sentiment([{"title": title}])
        assert result[0]["sentiment_score"] == 1.0

    def test_same_word_counts_once(self):
        once = st.score_news_sentiment([{"title": "涨停"}])[0]
        twice = st.score_news_sentiment([{"title": "涨停涨停涨停"}])[0]
        assert once["sentiment_score"] == twice["sentiment_score"]
        assert once["hits"] == ["涨停"]

    def test_content_and_summary_scored(self):
        items = [{"title": "公司公告", "content": "净利润预增，订单饱满"}]
        result = st.score_news_sentiment(items)
        assert result[0]["sentiment"] == "利好"
        assert set(result[0]["hits"]) == {"预增", "订单饱满"}

    def test_original_fields_preserved(self):
        items = [{"title": "涨停", "url": "http://x", "publish_date": "2025-01-10"}]
        result = st.score_news_sentiment(items)
        assert result[0]["url"] == "http://x"
        assert result[0]["publish_date"] == "2025-01-10"
        assert items[0].get("sentiment") is None    # 不污染入参

    def test_non_dict_items_skipped(self):
        result = st.score_news_sentiment(["字符串", None, {"title": "涨停"}, 42])
        assert len(result) == 1

    def test_custom_scorer_injected(self):
        def scorer(item):
            return {"sentiment": "利好", "sentiment_score": 0.88, "hits": ["自定义"]}
        result = st.score_news_sentiment([{"title": "随便"}], scorer=scorer)
        assert result[0]["sentiment"] == "利好"
        assert result[0]["sentiment_score"] == 0.88
        assert result[0]["hits"] == ["自定义"]

    def test_custom_scorer_partial_fields_filled(self):
        result = st.score_news_sentiment([{"title": "x"}], scorer=lambda it: {"sentiment_score": -0.5})
        assert result[0]["sentiment"] == "利空"     # 由 score 推导
        assert result[0]["hits"] == []

    def test_custom_scorer_exception_degrades(self):
        def boom(item):
            raise RuntimeError("模型炸了")
        result = st.score_news_sentiment([{"title": "涨停"}], scorer=boom)
        assert result[0]["sentiment"] == "中性"
        assert result[0]["sentiment_score"] == 0.0


# ═══════════════════════════════════════════
# 6. 温度公式与标签
# ═══════════════════════════════════════════

class TestTemperature:
    def test_formula_deterministic_handcomputed(self):
        stats = {"zt_count": 40, "dt_count": 15, "zb_count": 0,
                 "炸板率": 50.0, "最高连板": 3}
        # 20 + 35*0.5 - 25*0.5 + 20*0.5 + 20*0.5 = 45.0
        assert st._calc_temperature(stats) == 45.0

    def test_formula_extreme_hot(self):
        stats = {"zt_count": 100, "dt_count": 0, "zb_count": 0,
                 "炸板率": 0.0, "最高连板": 8}
        # 20 + 35 + 0 + 20 + 20 = 95
        assert st._calc_temperature(stats) == 95.0

    def test_formula_extreme_cold(self):
        stats = {"zt_count": 0, "dt_count": 50, "zb_count": 0,
                 "炸板率": 100.0, "最高连板": 0}
        # 20 + 0 - 25 + 0 + 0 = -5 → clamp 0
        assert st._calc_temperature(stats) == 0.0

    def test_formula_empty_stats(self):
        # 20 + 0 - 0 + 20 + 0 = 40（炸板率 0 视为封板质量满分）
        assert st._calc_temperature({}) == 40.0

    @pytest.mark.parametrize("temp,label", [
        (95.0, "亢奋"), (80.0, "亢奋"),
        (79.9, "活跃"), (60.0, "活跃"),
        (59.9, "中性"), (40.0, "中性"),
        (39.9, "低迷"), (20.0, "低迷"),
        (19.9, "冰点"), (0.0, "冰点"),
    ])
    def test_label_boundaries(self, temp, label):
        assert st._temperature_label(temp) == label


# ═══════════════════════════════════════════
# 7. get_market_sentiment
# ═══════════════════════════════════════════

def market_http_post(calls=None):
    rows = [hot_rank_row(f"1.60051{i}", f"股{i}", i + 1) for i in range(10)]
    return make_http(lambda url, kw: FakeResp(hot_rank_page_nested(rows)), calls)


class TestGetMarketSentiment:
    def test_full_composition(self, monkeypatch):
        monkeypatch.setattr(st, "_today_ymd", lambda: "20250110")
        monkeypatch.setattr(st, "_today_iso", lambda: "2025-01-10")
        news = [{"title": "甲公司预增超预期"}, {"title": "乙公司爆雷"}, {"title": "丙公告"}]
        result = st.get_market_sentiment(http_get=pools_http(),
                                         http_post=market_http_post(),
                                         sleep=make_sleep([]),
                                         news_items=news)
        assert result["date"] == "2025-01-10"
        # stats: zt=2, dt=1, zb=1, 炸板率=33.33, 最高连板=3
        # 温度 = 20 + 35*(2/80) - 25*(1/30) + 20*(1-0.3333) + 20*(3/6) ≈ 43.3
        assert result["temperature"] == round(
            20 + 35 * (2 / 80) - 25 * (1 / 30) + 20 * (1 - round(1 / 3 * 100, 2) / 100) + 20 * 0.5, 1)
        assert result["temperature_label"] == "中性"
        assert len(result["hot_rank_top"]) == 10
        assert result["news_sentiment"] == {"利好": 1, "利空": 1, "中性": 1}
        assert set(result["sources"]) == {"eastmoney_ztpool", "eastmoney_hotrank", "news_injected"}

    def test_hot_rank_top_names_backfilled(self, monkeypatch):
        """实测形态人气榜（无 name）经回填后 hot_rank_top 能显示名称。"""
        monkeypatch.setattr(st, "_today_ymd", lambda: "20250110")
        monkeypatch.setattr(st, "_today_iso", lambda: "2025-01-10")
        monkeypatch.setattr(st, "fetch_stock_names",
                            lambda codes, **kw: {c: f"名称{c}" for c in codes})
        rows = [hot_rank_row_real(f"SZ0000{i:02d}", i + 1) for i in range(10)]
        http_post = make_http(lambda url, kw: FakeResp(hot_rank_page_real(rows)))
        result = st.get_market_sentiment(http_get=pools_http(), http_post=http_post,
                                         sleep=make_sleep([]))
        assert result["hot_rank_top"][0]["name"] == "名称000000"
        assert all(it["name"] for it in result["hot_rank_top"])

    def test_cache_hit_avoids_repeat_http(self, monkeypatch):
        monkeypatch.setattr(st, "_today_ymd", lambda: "20250110")
        monkeypatch.setattr(st, "_today_iso", lambda: "2025-01-10")
        get_calls, post_calls = [], []
        first = st.get_market_sentiment(http_get=pools_http(calls=get_calls),
                                        http_post=market_http_post(post_calls),
                                        sleep=make_sleep([]))
        assert len(get_calls) == 3 and len(post_calls) == 1
        second = st.get_market_sentiment(http_get=pools_http(calls=get_calls),
                                         http_post=market_http_post(post_calls),
                                         sleep=make_sleep([]))
        assert len(get_calls) == 3 and len(post_calls) == 1   # 缓存命中，无新请求
        assert second == first

    def test_cache_keyed_by_date(self):
        st.get_market_sentiment(date="2025-01-10", http_get=pools_http(),
                                http_post=market_http_post(), sleep=make_sleep([]))
        get_calls = []
        st.get_market_sentiment(date="2025-01-11", http_get=pools_http(calls=get_calls),
                                http_post=market_http_post(), sleep=make_sleep([]))
        assert len(get_calls) == 3                            # 不同日期不命中缓存

    def test_pool_failure_degrades_with_notes(self, monkeypatch):
        monkeypatch.setattr(st, "_today_ymd", lambda: "20250110")
        monkeypatch.setattr(st, "_today_iso", lambda: "2025-01-10")
        result = st.get_market_sentiment(
            http_get=pools_http(fail={"ZTPool", "DTPool", "ZBPool"}),
            http_post=market_http_post(), sleep=make_sleep([]))
        assert result["stats"]["zt_count"] == 0
        assert len(result["notes"]) >= 3
        assert result["temperature"] == st._calc_temperature(result["stats"])

    def test_hot_rank_failure_degrades(self, monkeypatch):
        monkeypatch.setattr(st, "_today_ymd", lambda: "20250110")
        monkeypatch.setattr(st, "_today_iso", lambda: "2025-01-10")
        def boom(url, kw):
            raise ConnectionError("人气榜挂了")
        result = st.get_market_sentiment(http_get=pools_http(),
                                         http_post=make_http(boom),
                                         sleep=make_sleep([]))
        assert result["hot_rank_top"] == []
        assert any("人气榜" in n for n in result["notes"])
        assert "eastmoney_hotrank" not in result["sources"]

    def test_no_news_sources_omitted(self, monkeypatch):
        monkeypatch.setattr(st, "_today_ymd", lambda: "20250110")
        monkeypatch.setattr(st, "_today_iso", lambda: "2025-01-10")
        result = st.get_market_sentiment(http_get=pools_http(),
                                         http_post=market_http_post(),
                                         sleep=make_sleep([]))
        assert result["news_sentiment"] == {"利好": 0, "利空": 0, "中性": 0}
        assert "news_injected" not in result["sources"]


# ═══════════════════════════════════════════
# 8. get_stock_sentiment
# ═══════════════════════════════════════════

class TestGetStockSentiment:
    def _post(self, board_rows=None, his_payload=None, calls=None):
        board_rows = board_rows if board_rows is not None else [
            hot_rank_row("1.600519", "贵州茅台", 5)]
        his_payload = his_payload if his_payload is not None else HIS_PAYLOAD_REAL

        def handler(url, kw):
            if "getAllCurrentList" in url:
                return FakeResp(hot_rank_page_nested(board_rows))
            if "getHisList" in url:
                return FakeResp(his_payload)
            raise AssertionError(f"未预期 URL: {url}")

        return make_http(handler, calls)

    def test_full_composition_trend_up(self):
        # 历史均值 (40+30+20+10+5)/5 = 21，当前 5 < 21 → 上升
        result = st.get_stock_sentiment("600519", days=30, http_post=self._post())
        assert result["code"] == "600519"
        assert result["hot_rank"] == {"latest": 5, "history_avg": 21.0, "trend": "上升"}
        assert result["sources"] == ["eastmoney_hotrank"]
        assert result["notes"] == []

    def test_trend_down(self):
        his = {"data": [{"calcTime": "2025-01-09", "rank": 3},
                        {"calcTime": "2025-01-10", "rank": 4}]}
        # 均值 3.5，当前 5 > 3.5 → 下降
        result = st.get_stock_sentiment("600519", http_post=self._post(his_payload=his))
        assert result["hot_rank"]["trend"] == "下降"
        assert result["hot_rank"]["history_avg"] == 3.5

    def test_trend_flat(self):
        his = {"data": [{"calcTime": "2025-01-10", "rank": 5}]}
        result = st.get_stock_sentiment("600519", http_post=self._post(his_payload=his))
        assert result["hot_rank"]["trend"] == "平稳"

    def test_not_on_board_uses_history_latest(self):
        board = [hot_rank_row("1.600000", "浦发银行", 1)]
        result = st.get_stock_sentiment("600519", http_post=self._post(board_rows=board))
        assert result["hot_rank"]["latest"] == 5            # 回退历史最新
        assert any("未进入人气榜" in n for n in result["notes"])

    def test_no_board_no_history_trend_unknown(self):
        his = {"data": []}
        result = st.get_stock_sentiment(
            "600519", http_post=self._post(board_rows=[], his_payload=his))
        assert result["hot_rank"] == {"latest": None, "history_avg": None, "trend": "未知"}

    def test_news_distribution(self):
        news = [{"title": "预增"}, {"title": "违约"}, {"title": "中性"}, {"title": "回购增持"}]
        result = st.get_stock_sentiment("600519", http_post=self._post(), news_items=news)
        assert result["news_sentiment"] == {"利好": 2, "利空": 1, "中性": 1}
        assert "news_injected" in result["sources"]

    def test_all_sources_fail_never_raises(self):
        def boom(url, kw):
            raise ConnectionError("全挂")
        result = st.get_stock_sentiment("600519", http_post=make_http(boom),
                                        news_items=[{"title": "预增"}])
        assert result["hot_rank"]["latest"] is None
        assert result["hot_rank"]["trend"] == "未知"
        assert result["news_sentiment"]["利好"] == 1      # 本地打分不依赖网络
        assert len(result["notes"]) >= 2

    def test_invalid_code_degrades(self):
        result = st.get_stock_sentiment("abc", http_post=make_http(None))
        assert result["code"] == "abc"
        assert result["hot_rank"]["trend"] == "未知"
        assert result["sources"] == []
        assert any("非法" in n for n in result["notes"])
