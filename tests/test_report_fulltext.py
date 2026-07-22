"""scripts/report_fulltext.py 研报全文获取层测试（全 mock 零网络）。

覆盖范围（研报库 v2 全文层，全局契约 1/2/7/8）：
1. 分节：惯用节名分节、「节名：内容」同行、无结构退化单节「正文」。
2. 新浪：列表页/详情页解析正确性（内联 HTML fixture，GB2312 字节输入），
   sections 结构（投资要点/盈利预测与投资建议/风险提示），垃圾输入返回 None。
3. 东财：encode_url → pdf.dfcfw.com 直链拼法、EO_Bot JS 挑战页 cookie
   求解（真实捕获样本固化 fixture）、pymupdf 现场生成中文微型 PDF 字节
   fixture 解析分节、非 PDF 字节返回 None。
4. 存储：upsert_fulltext 幂等（两次写入一行）、非法记录跳过。
5. 管道：东财+新浪双源端到端（fake http_get 路由）、EO_Bot 挑战重取成功、
   单篇失败记 failed 跳过不拖垮批次、已有全文的候选被排除、days 过滤、
   新浪未匹配计 skipped。
6. 限速：东财/新浪独立限速门，jitter=0 时 sleep 调用次数与时长断言。
7. CLI：main 经 fake 跑通、返回 0、结尾统计打印正确。

所有 HTTP 由 fake http_get 注入，sleep 由 fake 注入，绝不触达真实网络；
库文件一律落 pytest tmp_path，绝不触碰真实 data/reports.db。
"""

import json
import re
import sqlite3
from datetime import date
from types import SimpleNamespace
from urllib.parse import quote

import pytest

import scripts.report_fulltext as rf
from agent import report_library

TODAY = date.today().isoformat()


# ── 内联 fixture：按 2026-07-22 真实抓取样本固化 ──

SINA_LIST_HTML = f"""<html><body><table>
<tr><th scope="col">序号</th><th scope="col">标题</th><th scope="col">类型</th>
<th scope="col" class="t04">发布日期</th><th scope="col" class="t05">机构</th>
<th scope="col" class="t06">研究员</th></tr>
<tr>
  <td>1</td>
  <td class="tal f14">
    <a target="_blank" title="西麦食品(002956)：六五战略目标清晰 粉类产品成长可期"
       href="//stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/838041262338/index.phtml">
       西麦食品(002956)：六五战略目标清晰 粉类产品成长可期
    </a>
  </td>
  <td>公司</td>
  <td>{TODAY}</td>
  <td><a href="/stock/go.php/vReport_List/kind/search/index.phtml?t1=1&orgname=xxx">
      <div class="fname">中邮证券</div></a></td>
  <td>蔡雪昱/张子健</td>
</tr>
<tr>
  <td>2</td>
  <td class="tal f14">
    <a target="_blank" title="半导体行业：国产替代加速"
       href="//stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/838039380266/index.phtml">
       半导体行业：国产替代加速
    </a>
  </td>
  <td>行业</td>
  <td>{TODAY}</td>
  <td>长江证券股份有限公司</td>
  <td>陈亮</td>
</tr>
</table></body></html>"""

SINA_DETAIL_HTML = """<html><body>
<div class="content">
<h1>西麦食品(002956)：六五战略目标清晰 粉类产品成长可期</h1>
<div class="creab">
  <span>类别：公司</span><span>机构：中邮证券有限责任公司</span><span>日期：2026-07-22</span>
</div>
<div class="blk_container">
  <p>　　投资要点<br /><br />&nbsp;&nbsp;&nbsp;
　　六五战略目标清晰，存量增量双线驱动规模扩容。<br /><br />&nbsp;&nbsp;&nbsp;
　　粉类业务打造第二增长曲线，旺季催化释放业绩高弹性。<br /><br />&nbsp;&nbsp;&nbsp;
　　盈利预测与投资建议<br /><br />&nbsp;&nbsp;&nbsp;
　　我们预计2026-2028 年公司营收27.85/32.92/38.27 亿元，维持买入评级。<br /><br />&nbsp;&nbsp;&nbsp;
　　风险提示：<br /><br />&nbsp;&nbsp;&nbsp;
　　粉类业务拓展不及预期；成本波动风险；食品安全风险。</p>
</div>
</div></body></html>"""

# EO_Bot 挑战页真实样本（2026-07-22 捕获；常量每期动态，此处固化一份）
EO_BOT_JS = (
    "<script>function a(a){function n(){for(var a={wQzOV:_0x649a(\"0x4\"),"
    "iTyzs:function(a,n){return a+n}},n=a[_0x649a(\"0x5\")][_0x649a(\"0x6\")]"
    "(\"|\"),e=0;;){switch(n[e++]){case\"0\":t+=\"EO_Bot_Ssid=\";continue;"
    "case\"1\":return t;case\"2\":t+=\"\";continue;"
    "case\"3\":t=a[_0x649a(\"0x7\")](t,3888971776);continue;"
    "case\"4\":var t=\"\";continue}break}}"
    "var e={WTKkN:2866969970,bOYDu:516246026,dtzqS:function(a,n){return a+n},"
    "wyeCN:628044687,pCQRM:function(a){return a()}},t=0;"
    "return t+=e[_0x649a(\"0x0\")],t+=e[_0x649a(\"0x1\")],"
    "t=e[_0x649a(\"0x2\")](t,e[_0x649a(\"0x3\")]),[t,e[_0x649a(\"0x8\")](n)][a]}"
    "document[_0x649a(\"0x9\")]=\"__tst_status=\"+a(0)+\"#;\","
    "document[_0x649a(\"0x9\")]=a(1)+\";\",setTimeout(_0x649a(\"0xa\"),0x4b0);"
    "</script>"
)


def _gb(text: str) -> bytes:
    """fixture 文本 → GB2312 字节（模拟真实线上编码）。"""
    return text.encode("gb2312", errors="replace")


def _make_pdf_bytes(lines):
    """pymupdf 现场生成中文微型 PDF 字节 fixture（用完即弃）。"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for i, line in enumerate(lines):
        page.insert_text((72, y), line, fontname="china-s",
                         fontsize=14 if i % 2 == 0 else 11)
        y += 30
    blob = doc.tobytes()
    doc.close()
    return blob


def _resp(content, status_code=200):
    return SimpleNamespace(status_code=status_code, content=content)


def _seed_reports(db_path, rows):
    """经 v1 存储层播种 reports 表（真实模块、tmp 库文件）。"""
    return report_library.upsert_reports(rows, str(db_path))


# ═══════════════════════════════════════════
# 分节纯函数
# ═══════════════════════════════════════════

class TestSplitSections:
    def test_known_headers_split(self):
        text = ("投资要点\n营收有望翻倍。\n粉类业务放量。\n"
                "盈利预测\n预计 2026 年 EPS 0.66 元。\n风险提示\n成本波动风险。")
        sections = rf.split_sections(text)
        assert [s["name"] for s in sections] == ["投资要点", "盈利预测", "风险提示"]
        assert "营收有望翻倍" in sections[0]["text"]
        assert "粉类业务放量" in sections[0]["text"]
        assert sections[2]["text"] == "成本波动风险。"

    def test_inline_header_with_content(self):
        text = "公司主业稳健。\n风险提示：需求不及预期、竞争加剧。"
        sections = rf.split_sections(text)
        assert [s["name"] for s in sections] == ["正文", "风险提示"]
        assert sections[1]["text"] == "需求不及预期、竞争加剧。"

    def test_fallback_single_body(self):
        sections = rf.split_sections("第一段无节名内容。\n第二段依然无节名。")
        assert sections == [{"name": "正文", "text": "第一段无节名内容。\n第二段依然无节名。"}]

    def test_empty_input(self):
        assert rf.split_sections("") == []
        assert rf.split_sections(None) == []
        assert rf.split_sections("   \n  ") == []


# ═══════════════════════════════════════════
# 新浪解析
# ═══════════════════════════════════════════

class TestSinaParsers:
    def test_parse_list_gb2312_bytes(self):
        entries = rf.parse_sina_list(_gb(SINA_LIST_HTML))
        assert len(entries) == 2
        first = entries[0]
        assert first["rptid"] == "838041262338"
        assert first["title"] == "西麦食品(002956)：六五战略目标清晰 粉类产品成长可期"
        assert first["date"] == TODAY
        assert first["category"] == "公司"
        assert "中邮证券" in first["org"]
        assert entries[1]["rptid"] == "838039380266"

    def test_parse_detail_sections_gb2312_bytes(self):
        parsed = rf.parse_sina_detail(_gb(SINA_DETAIL_HTML))
        assert parsed is not None
        assert parsed["title"] == "西麦食品(002956)：六五战略目标清晰 粉类产品成长可期"
        names = [s["name"] for s in parsed["sections"]]
        assert names == ["投资要点", "盈利预测与投资建议", "风险提示"]
        body = parsed["sections"][0]["text"]
        assert "六五战略目标清晰" in body and "粉类业务打造第二增长曲线" in body
        assert "食品安全风险" in parsed["sections"][2]["text"]
        # fulltext = 各节文本拼接
        assert parsed["fulltext"] == "\n".join(s["text"] for s in parsed["sections"])

    def test_parse_detail_str_input(self):
        """str 输入（非字节）同样可解析。"""
        parsed = rf.parse_sina_detail(SINA_DETAIL_HTML)
        assert parsed is not None
        assert parsed["sections"][0]["name"] == "投资要点"

    def test_parse_garbage_returns_none(self):
        assert rf.parse_sina_detail(_gb("<html><body>无正文容器</body></html>")) is None
        assert rf.parse_sina_detail(b"") is None
        assert rf.parse_sina_list(b"") == []


# ═══════════════════════════════════════════
# 东财 PDF
# ═══════════════════════════════════════════

class TestEastmoneyPdf:
    def test_pdf_url_uses_info_code(self):
        """P0 修复终版：直链用 info_code（akshare 源码拼法），不用 encode_url
        （含 / 时 Tomcat 原样 404、quote(%2F) 400 双死路，2026-07-22 实测）。"""
        url = rf.eastmoney_pdf_url("AP202607221827246877")
        assert url == "https://pdf.dfcfw.com/pdf/H3_AP202607221827246877_1.pdf"
        assert "/" not in url.split("/pdf/H3_", 1)[1]

    def test_solve_eo_bot_challenge(self):
        cookie = rf.solve_eo_bot_challenge(EO_BOT_JS)
        # __tst_status = 2866969970 + 516246026 + 628044687 = 4011260683
        assert cookie == "__tst_status=4011260683#; EO_Bot_Ssid=3888971776"

    def test_solve_eo_bot_challenge_bytes_input(self):
        assert rf.solve_eo_bot_challenge(EO_BOT_JS.encode("utf-8")) == \
            "__tst_status=4011260683#; EO_Bot_Ssid=3888971776"

    def test_solve_eo_bot_challenge_non_challenge(self):
        assert rf.solve_eo_bot_challenge("<html>正常页面</html>") is None
        assert rf.solve_eo_bot_challenge(b"") is None

    def test_parse_pdf_fulltext(self):
        blob = _make_pdf_bytes([
            "投资要点",
            "公司营收有望翻倍增长，维持买入评级。",
            "风险提示",
            "成本波动风险；食品安全风险。",
        ])
        parsed = rf.parse_pdf_fulltext(blob)
        assert parsed is not None
        names = [s["name"] for s in parsed["sections"]]
        assert names == ["投资要点", "风险提示"]
        assert "买入评级" in parsed["sections"][0]["text"]
        assert "食品安全风险" in parsed["fulltext"]

    def test_parse_pdf_fulltext_no_structure_fallback(self):
        blob = _make_pdf_bytes(["这是一段没有节名的研报正文内容。"])
        parsed = rf.parse_pdf_fulltext(blob)
        assert parsed is not None
        assert [s["name"] for s in parsed["sections"]] == ["正文"]

    def test_parse_pdf_fulltext_bad_bytes(self):
        assert rf.parse_pdf_fulltext(b"<script>not a pdf</script>") is None
        assert rf.parse_pdf_fulltext(b"") is None
        assert rf.parse_pdf_fulltext(None) is None


# ═══════════════════════════════════════════
# 存储：upsert 幂等
# ═══════════════════════════════════════════

class TestUpsertFulltext:
    def _record(self, code="AP001", source="eastmoney", text="全文内容"):
        return {"info_code": code, "source": source, "fulltext": text,
                "sections": [{"name": "正文", "text": text}],
                "fetched_at": "2026-07-22T10:00:00"}

    def test_upsert_idempotent(self, tmp_path):
        db = tmp_path / "reports.db"
        assert rf.upsert_fulltext([self._record()], str(db)) == 1
        assert rf.upsert_fulltext([self._record()], str(db)) == 1  # 重复写入
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT info_code, source, fulltext, sections_json, fetched_at "
                "FROM report_fulltext").fetchall()
        finally:
            conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "AP001" and rows[0][1] == "eastmoney"
        sections = json.loads(rows[0][3])
        assert sections == [{"name": "正文", "text": "全文内容"}]

    def test_upsert_skips_invalid(self, tmp_path):
        db = tmp_path / "reports.db"
        bad = [{"info_code": "", "fulltext": "x"},
               {"info_code": "AP002", "fulltext": ""},
               "not-a-dict"]
        assert rf.upsert_fulltext(bad, str(db)) == 0

    def test_upsert_sections_default_fallback(self, tmp_path):
        """缺 sections 时自动退化为单节「正文」且 fetched_at 自动补。"""
        db = tmp_path / "reports.db"
        n = rf.upsert_fulltext(
            [{"info_code": "AP003", "fulltext": "只有全文没有分节"}], str(db))
        assert n == 1
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT sections_json, fetched_at FROM report_fulltext "
                "WHERE info_code='AP003'").fetchone()
        finally:
            conn.close()
        assert json.loads(row[0]) == [{"name": "正文", "text": "只有全文没有分节"}]
        assert re.match(r"^\d{4}-\d{2}-\d{2}T", row[1])


# ═══════════════════════════════════════════
# 抓取管道（fake http_get 路由）
# ═══════════════════════════════════════════

EM_REPORT = {
    "info_code": "AP202607221827246877", "title": "西麦食品：六五战略点评",
    "org": "中邮证券", "publish_date": TODAY, "source": "eastmoney",
    "encode_url": "bkYktvs9RKNTrOLqkt2RHe9yTKiXfpoQwpc+lFr1vzQ=",
}
SINA_REPORT = {
    "info_code": "stockstar:abc123", "title": "西麦食品(002956)：六五战略目标清晰 粉类产品成长可期",
    "org": "中邮证券", "publish_date": TODAY, "source": "stockstar",
}


def _router(pdf_map=None, sina=True, list_pages=1):
    """构造 fake http_get：按 URL 路由到东财 PDF / 新浪列表 / 新浪详情。
    pdf_map: {encode_url: content-or-(status, content)}；缺省返回标准微型 PDF。"""
    calls = []

    def fake_http_get(url, **kw):
        calls.append((url, kw))
        if "pdf.dfcfw.com" in url:
            for enc, content in (pdf_map or {}).items():
                # encode_url 在 URL 中以 quote(safe='') 形式出现
                if enc in url or quote(enc, safe="") in url:
                    if isinstance(content, tuple):
                        # (首响应, 挑战后响应)：按是否带 Cookie 头区分
                        first, second = content
                        if (kw.get("headers") or {}).get("Cookie"):
                            return _resp(second)
                        return _resp(first)
                    return _resp(content)
            return _resp(_make_pdf_bytes(["投资要点", "标准 PDF 正文内容。"]))
        if "vReport_List" in url:
            m = re.search(r"p=(\d+)", url)
            if m and int(m.group(1)) > list_pages:
                return _resp(_gb("<html><body></body></html>"))
            return _resp(_gb(SINA_LIST_HTML)) if sina else _resp(b"", 500)
        if "vReport_Show" in url:
            return _resp(_gb(SINA_DETAIL_HTML))
        raise AssertionError(f"未预期 URL：{url}")

    fake_http_get.calls = calls
    return fake_http_get


class TestFetchPipeline:
    def test_both_sources_end_to_end(self, tmp_path):
        db = tmp_path / "reports.db"
        _seed_reports(db, [EM_REPORT, SINA_REPORT])
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=_router(), sleep=lambda s: None)
        assert stats == {"candidates": 2, "fetched": 2, "upserted": 2,
                         "failed": 0, "skipped": 0}
        conn = sqlite3.connect(str(db))
        try:
            rows = {r[0]: (r[1], r[2]) for r in conn.execute(
                "SELECT info_code, source, fulltext FROM report_fulltext")}
        finally:
            conn.close()
        assert rows[EM_REPORT["info_code"]][0] == "eastmoney"
        assert "标准 PDF 正文内容" in rows[EM_REPORT["info_code"]][1]
        assert rows[SINA_REPORT["info_code"]][0] == "sina"
        assert "六五战略目标清晰" in rows[SINA_REPORT["info_code"]][1]

    def test_eo_bot_challenge_retry(self, tmp_path):
        """首响应为挑战页 → 解 cookie 重取 → 成功入库。"""
        db = tmp_path / "reports.db"
        _seed_reports(db, [EM_REPORT])
        pdf = _make_pdf_bytes(["投资要点", "挑战后拿到的正文。"])
        router = _router(pdf_map={EM_REPORT["info_code"]: (EO_BOT_JS, pdf)})
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=router, sleep=lambda s: None)
        assert stats["upserted"] == 1 and stats["failed"] == 0
        cookie_calls = [kw for url, kw in router.calls
                        if "pdf.dfcfw.com" in url and
                        (kw.get("headers") or {}).get("Cookie")]
        assert len(cookie_calls) == 1
        assert "EO_Bot_Ssid=3888971776" in cookie_calls[0]["headers"]["Cookie"]

    def test_single_failure_skipped(self, tmp_path):
        """单篇 400 记 failed 跳过，不拖垮其余批次。"""
        db = tmp_path / "reports.db"
        bad = dict(EM_REPORT, info_code="AP_BAD")
        _seed_reports(db, [bad, EM_REPORT])

        def fake(url, **kw):
            if "AP_BAD" in url:  # 直链含 info_code，坏篇恒 400
                return _resp(b"", 400)
            return _resp(_make_pdf_bytes(["投资要点", "好篇正文。"]))

        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=fake, sleep=lambda s: None)
        assert stats["candidates"] == 2
        assert stats["failed"] == 1 and stats["upserted"] == 1

    def test_url_uses_info_code_not_encode_url(self, tmp_path):
        """P0 回归：encode_url 含 / 的东财行，请求 URL 必须含 info_code
        且不得出现 encode_url 原样斜杠路径。"""
        db = tmp_path / "reports.db"
        row = dict(EM_REPORT,
                   encode_url="4PqOz2Xj2NJxO5S/+q3G3QmvCyITKYbKbXnPqrrXqSY=")
        _seed_reports(db, [row])
        router = _router()
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=router, sleep=lambda s: None)
        assert stats["upserted"] == 1
        pdf_urls = [url for url, _ in router.calls if "pdf.dfcfw.com" in url]
        assert pdf_urls and all(
            "H3_AP202607221827246877_1.pdf" in u for u in pdf_urls)
        assert not any("4PqOz2Xj2NJxO5S/" in u for u in pdf_urls)

    def test_eastmoney_row_without_encode_url_still_pdf_channel(self, tmp_path):
        """encode_url 为空的东财行也走 PDF 通道（直链只用 info_code）。"""
        db = tmp_path / "reports.db"
        _seed_reports(db, [dict(EM_REPORT, encode_url="")])
        router = _router()
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=router, sleep=lambda s: None)
        assert stats["upserted"] == 1
        assert any("pdf.dfcfw.com" in url for url, _ in router.calls)

    def test_existing_fulltext_excluded(self, tmp_path):
        db = tmp_path / "reports.db"
        _seed_reports(db, [EM_REPORT, SINA_REPORT])
        rf.upsert_fulltext([{"info_code": EM_REPORT["info_code"],
                             "fulltext": "已有全文"}], str(db))
        router = _router()
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=router, sleep=lambda s: None)
        assert stats["candidates"] == 1 and stats["upserted"] == 1
        assert not any("pdf.dfcfw.com" in url for url, _ in router.calls)

    def test_days_filter(self, tmp_path):
        db = tmp_path / "reports.db"
        old = dict(EM_REPORT, info_code="AP_OLD", publish_date="2020-01-01")
        _seed_reports(db, [old, EM_REPORT])
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=_router(), sleep=lambda s: None)
        assert stats["candidates"] == 1 and stats["upserted"] == 1

    def test_sina_no_match_counts_skipped(self, tmp_path):
        db = tmp_path / "reports.db"
        nomatch = dict(SINA_REPORT, title="列表里根本不存在的研报标题")
        _seed_reports(db, [nomatch])
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=_router(), sleep=lambda s: None)
        assert stats == {"candidates": 1, "fetched": 0, "upserted": 0,
                         "failed": 0, "skipped": 1}

    def test_ids_filter_and_limit(self, tmp_path):
        db = tmp_path / "reports.db"
        _seed_reports(db, [EM_REPORT, SINA_REPORT])
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  ids=[EM_REPORT["info_code"]],
                                  http_get=_router(), sleep=lambda s: None)
        assert stats["candidates"] == 1
        stats2 = rf.fetch_fulltext(db_path=str(db), days=30, limit=1,
                                   http_get=_router(), sleep=lambda s: None)
        assert stats2["candidates"] <= 1  # 另一篇已入库，候选至多剩 1

    def test_empty_db_no_exception(self, tmp_path):
        db = tmp_path / "reports.db"  # 库文件都不存在
        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=_router(), sleep=lambda s: None)
        assert stats == {"candidates": 0, "fetched": 0, "upserted": 0,
                         "failed": 0, "skipped": 0}


# ═══════════════════════════════════════════
# 新浪标题匹配（含短标题约束兜底）
# ═══════════════════════════════════════════

class TestSinaMatching:
    def _entries(self):
        return [
            {"rptid": "1", "title": "晨会纪要", "org": "东吴证券股份有限公司",
             "date": TODAY, "category": "公司"},
            {"rptid": "2", "title": "晨会纪要", "org": "中邮证券",
             "date": TODAY, "category": "公司"},
        ]

    def test_short_title_matched_by_org_and_date(self):
        """泛化短标题（晨会纪要）：标题+日期+机构三约束齐备可匹配。"""
        row = {"title": "晨会纪要", "org": "东吴证券", "publish_date": TODAY}
        e = rf._match_sina_entry(row, self._entries())
        assert e is not None and e["rptid"] == "1"

    def test_short_title_wrong_org_rejected(self):
        row = {"title": "晨会纪要", "org": "国泰君安", "publish_date": TODAY}
        assert rf._match_sina_entry(row, self._entries()) is None

    def test_short_title_wrong_date_rejected(self):
        row = {"title": "晨会纪要", "org": "东吴证券",
               "publish_date": "2020-01-01"}
        assert rf._match_sina_entry(row, self._entries()) is None

    def test_empty_title_rejected(self):
        assert rf._match_sina_entry({"title": "", "org": "x"}, self._entries()) is None


# ═══════════════════════════════════════════
# 东财 WAF 节点重试链
# ═══════════════════════════════════════════

class TestEastmoneyRetry:
    def test_retry_400_then_challenge_then_pdf(self, tmp_path):
        """重试链：400（WAF 节点）→ 挑战页（友好节点）→ cookie 重取命中 PDF。"""
        db = tmp_path / "reports.db"
        _seed_reports(db, [EM_REPORT])
        pdf = _make_pdf_bytes(["投资要点", "重试后拿到的正文。"])
        calls = []
        seq = [_resp(b"<html>waf</html>", 400), _resp(EO_BOT_JS.encode("utf-8"))]

        def fake(url, **kw):
            calls.append((url, kw))
            if seq:
                return seq.pop(0)
            assert (kw.get("headers") or {}).get("Cookie"), "第三轮必须带挑战 cookie"
            return _resp(pdf)

        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=fake, sleep=lambda s: None)
        assert stats["failed"] == 0 and stats["upserted"] == 1
        assert len(calls) == 3  # 400 + 挑战页 + cookie 重取

    def test_attempts_exhausted_bounded(self, tmp_path):
        """持续 400：每篇最多 EM_FETCH_MAX_ATTEMPTS 次请求后记 failed 跳过。"""
        db = tmp_path / "reports.db"
        _seed_reports(db, [EM_REPORT])
        calls = []

        def fake(url, **kw):
            calls.append(url)
            return _resp(b"<html>waf</html>", 400)

        stats = rf.fetch_fulltext(db_path=str(db), days=30,
                                  http_get=fake, sleep=lambda s: None)
        assert stats == {"candidates": 1, "fetched": 0, "upserted": 0,
                         "failed": 1, "skipped": 0}
        assert len(calls) == rf.EM_FETCH_MAX_ATTEMPTS


# ═══════════════════════════════════════════
# 限速门
# ═══════════════════════════════════════════

class TestRateGate:
    def test_independent_gates_and_sleep(self, tmp_path):
        """jitter=0：东财两次请求间 sleep 恰好 1 次 1.0s；新浪门独立，
        其首个请求不 sleep（即便东财已请求过）。"""
        db = tmp_path / "reports.db"
        other_em = dict(EM_REPORT, info_code="AP202607221827243644",
                        encode_url="OTHERENC")
        _seed_reports(db, [EM_REPORT, other_em, SINA_REPORT])
        sleeps = []
        stats = rf.fetch_fulltext(db_path=str(db), days=30, jitter=0.0,
                                  http_get=_router(), sleep=sleeps.append)
        assert stats["upserted"] == 3
        # 东财 2 请求 → 1 次 sleep；新浪 列表2页（翻页条件放宽后 p2 空页止）
        # +详情1 → 2 次 sleep；两源门各自独立
        assert sleeps == [1.0, 1.0, 1.0]

    def test_gate_first_request_free(self):
        sleeps = []
        gate = rf.RateGate(sleep=sleeps.append, rate=1.0, jitter=0.5)
        gate.wait()
        gate.wait()
        assert len(sleeps) == 1 and 1.0 <= sleeps[0] <= 1.5


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

class TestCli:
    def test_main_with_fakes(self, tmp_path, capsys):
        db = tmp_path / "reports.db"
        _seed_reports(db, [EM_REPORT, SINA_REPORT])
        rc = rf.main(["--days", "30", "--db-path", str(db)],
                     http_get=_router(), sleep=lambda s: None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "候选 2 篇" in out
        assert "抓取 2 篇，入库 2 篇，失败 0 篇" in out

    def test_main_empty(self, tmp_path, capsys):
        db = tmp_path / "reports.db"
        rc = rf.main(["--db-path", str(db)],
                     http_get=_router(), sleep=lambda s: None)
        assert rc == 0
        assert "候选 0 篇" in capsys.readouterr().out
