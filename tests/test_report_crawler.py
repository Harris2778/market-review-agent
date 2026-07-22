"""scripts/report_crawler.py 研报多源爬虫测试（全 mock 零网络）。

覆盖范围：
1. 东财：字段映射（EPS/目标价字符串转 float、空串→None、publishDate 截断、
   author 去 id 前缀、infoCode 原值、ratingChange 首次/维持）、行业回退、
   翻页终止（TotalPage / 空页）、HTTP 异常与 5xx fail-safe。
2. 慧博：标题元数据容错（个股/无代码/无日期/非法日期/外资机构）、表格+
   侧边栏解析、作者行回填、[详细] 与公司公告链接排除、日期倒序提前终止。
3. 洞见：信封解析、publishAt 截断、401 状态码/响应体 code 被拒即停。
4. 证券之星：五栏目 GBK 字节解析（时间戳截断）、结构化表格 13/14 格行对
   （评级/评级变动/目标价/EPS/标题从摘要行提取）。
5. 契约：四个源产出记录均为 16 字段、info_code 合成规则（契约 2）。
6. 机制：限速 sleep 注入断言、upsert 注入落库与异常容错、
   CLI 经 fake 存储层跑通且统计正确、单源失败继续其他源。

所有 HTTP 由 fake http_get 注入，sleep 由 fake 注入，绝不触达真实网络。
"""

import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

import scripts.report_crawler as rc


# ── 公共工具 ──

TODAY = date.today().strftime("%Y-%m-%d")


class FakeResp:
    """最小 response 替身：status_code + content（bytes）。"""

    def __init__(self, content=b"", status_code=200):
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.content = content
        self.status_code = status_code
        self.text = content.decode("utf-8", errors="replace")


def make_http(handler, calls=None):
    """构造 fake http_get：handler(url, kw) -> FakeResp 或抛异常。"""
    def http_get(url, **kw):
        if calls is not None:
            calls.append((url, kw))
        return handler(url, kw)
    return http_get


def make_sleep(record):
    """构造 fake sleep：记录每次休眠秒数。"""
    def _sleep(seconds):
        record.append(seconds)
    return _sleep


def em_item(**over):
    """东财个股研报条目（按 2026-07-22 实测结构构造）。"""
    item = {
        "title": "主业企稳，切入硅基新材料领域打开增长天花板",
        "stockName": "宏柏新材", "stockCode": "605366",
        "orgSName": "中航证券", "publishDate": "2026-07-21 00:00:00.000",
        "infoCode": "AP202607211827224241",
        "predictThisYearEps": "0.0000000000", "predictNextYearEps": "0.4200000000",
        "indvInduName": "化学制品", "industryName": "", "industryCode": "",
        "emRatingName": "增持", "emRatingValue": "2",
        "lastEmRatingName": "", "lastEmRatingValue": "", "ratingChange": 2,
        "author": ["11000408132.曾帅"], "researcher": "曾帅",
        "indvAimPriceT": "", "indvAimPriceL": "",
        "encodeUrl": "ZKa/54az+XeMHjmCHDK1LP80BZCrkbUSswV+CYOGjE8=",
    }
    item.update(over)
    return item


def em_payload(items, total_page=1, hits=None):
    return {"hits": len(items) if hits is None else hits, "size": len(items),
            "data": items, "TotalPage": total_page, "pageNo": 1, "currentYear": 2026}


# ═══════════════════════════════════════════
# 1. 东财解析
# ═══════════════════════════════════════════

def test_parse_eastmoney_payload_stock_fields():
    """个股研报：字段全映射 + 类型转换。"""
    rec = rc.parse_eastmoney_payload(em_payload([em_item()]))[0]
    assert rec["info_code"] == "AP202607211827224241"  # 东财用 infoCode 原值
    assert rec["title"] == "主业企稳，切入硅基新材料领域打开增长天花板"
    assert rec["org"] == "中航证券"
    assert rec["author"] == "曾帅"  # author 列表 '11000408132.曾帅' 去 id 前缀
    assert rec["publish_date"] == "2026-07-21"  # publishDate 截断
    assert rec["stock_code"] == "605366"
    assert rec["stock_name"] == "宏柏新材"
    assert rec["industry"] == "化学制品"
    assert rec["rating"] == "增持"
    assert rec["rating_change"] == "首次覆盖"  # ratingChange=2 + last 为空
    assert rec["eps_this_year"] == 0.0  # '0.0000000000' → 0.0（真实零值，不是 None）
    assert rec["eps_next_year"] == 0.42
    assert rec["target_price_high"] is None  # 空串 → None
    assert rec["target_price_low"] is None
    assert rec["encode_url"].startswith("ZKa/")
    assert rec["source"] == "eastmoney"


def test_parse_eastmoney_rating_maintain_target_price_and_multi_authors():
    """ratingChange=3 → 维持；目标价字符串转 float；多作者合并。"""
    item = em_item(
        title="原料药量价齐升，宠物药开辟空间", stockName="回盛生物", stockCode="300871",
        emRatingName="买入", emRatingValue="1", lastEmRatingName="买入",
        lastEmRatingValue="1", ratingChange=3,
        author=["11000390435.陈翼", "11000259468.彭海兰"],
        indvAimPriceT="25.60", indvAimPriceL="20.10",
        predictThisYearEps="1.53", predictNextYearEps="1.97",
    )
    rec = rc.parse_eastmoney_payload(em_payload([item]))[0]
    assert rec["rating_change"] == "维持"
    assert rec["target_price_high"] == 25.6
    assert rec["target_price_low"] == 20.1
    assert rec["author"] == "陈翼,彭海兰"
    assert rec["eps_this_year"] == 1.53


def test_parse_eastmoney_industry_report_and_bad_input():
    """行业研报（qType=1 形态）：indvInduName 空时回退 industryName；
    无评级变动码时按前后评级推导；非法输入返回 []。"""
    item = em_item(
        title="公用环保行业202607第3期：垃圾焚烧发电项目中标梳理",
        stockName="", stockCode="", indvInduName="",
        industryName="电力", industryCode="428",
        emRatingName="增持", lastEmRatingName="", ratingChange="",
        author=[], researcher="",
    )
    rec = rc.parse_eastmoney_payload(em_payload([item]))[0]
    assert rec["stock_code"] == "" and rec["stock_name"] == ""
    assert rec["industry"] == "电力"  # industryName 回退
    assert rec["rating_change"] == "首次覆盖"  # 无码推导：cur 有值 + last 空
    assert rec["author"] == ""
    # 非法输入容错
    assert rc.parse_eastmoney_payload(None) == []
    assert rc.parse_eastmoney_payload({"data": "not-a-list"}) == []
    assert rc.parse_eastmoney_payload({"data": [{"title": ""}, "junk", {}]}) == []


# ═══════════════════════════════════════════
# 2. 东财翻页与 fail-safe
# ═══════════════════════════════════════════

def test_eastmoney_pagination_terminates_at_total_page():
    """翻页：TotalPage=2 → 恰好 2 次请求后终止，两页记录合并产出。"""
    pages = {
        1: em_payload([em_item(), em_item(title="第二篇", infoCode="AP2")], total_page=2, hits=3),
        2: em_payload([em_item(title="第三篇", infoCode="AP3")], total_page=2, hits=3),
    }
    calls = []

    def handler(url, kw):
        return FakeResp(json.dumps(pages[kw["params"]["pageNo"]],
                                   ensure_ascii=False))

    src = rc.EastmoneySource(qtypes=(0,), http_get=make_http(handler, calls),
                             sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-21", "2026-07-21"))
    assert len(records) == 3
    assert len(calls) == 2
    assert calls[0][1]["params"]["pageNo"] == 1
    assert calls[1][1]["params"]["pageNo"] == 2
    assert calls[0][1]["params"]["beginTime"] == "2026-07-21"
    assert src.stats["pages"] == 2 and src.stats["fetched"] == 3


def test_eastmoney_stops_on_empty_page_and_iterates_qtypes():
    """空 data 即停；qType 全分类遍历（空分类自然空跑，各占 1 次请求）。"""
    calls = []

    def handler(url, kw):
        q = kw["params"]["qType"]
        if q == 0:
            return FakeResp(json.dumps(em_payload([em_item()], total_page=1),
                                       ensure_ascii=False))
        return FakeResp(json.dumps(em_payload([], total_page=0, hits=0),
                                   ensure_ascii=False))

    src = rc.EastmoneySource(http_get=make_http(handler, calls), sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-21", "2026-07-21"))
    assert len(records) == 1
    assert len(calls) == 4  # qType 0/1/2/3 各 1 次（1/2/3 空跑）
    assert [c[1]["params"]["qType"] for c in calls] == [0, 1, 2, 3]


def test_eastmoney_http_failure_is_fail_safe():
    """http_get 抛异常 / HTTP 5xx：不抛出、产出空、记 stats。"""
    src = rc.EastmoneySource(qtypes=(0,),
                             http_get=make_http(lambda u, k: (_ for _ in ()).throw(
                                 ConnectionError("boom"))),
                             sleep=make_sleep([]))
    assert list(src.iter_records("2026-07-21", "2026-07-21")) == []

    src2 = rc.EastmoneySource(qtypes=(0,),
                              http_get=make_http(lambda u, k: FakeResp(b"err", 500)),
                              sleep=make_sleep([]))
    assert list(src2.iter_records("2026-07-21", "2026-07-21")) == []
    assert src2.stats["pages"] == 1

    # 非 JSON 响应同样降级
    src3 = rc.EastmoneySource(qtypes=(0,),
                              http_get=make_http(lambda u, k: FakeResp(b"<html>")),
                              sleep=make_sleep([]))
    assert list(src3.iter_records("2026-07-21", "2026-07-21")) == []


# ═══════════════════════════════════════════
# 3. 慧博标题元数据与列表解析
# ═══════════════════════════════════════════

def test_parse_hibor_metadata_stock_title():
    """「券商-个股-代码-标题-YYMMDD」完整解析。"""
    meta = rc.parse_hibor_metadata(
        "东吴证券-璞泰来-603659-2026H1业绩预告点评：负极、隔膜放量贡献盈利弹性-260720")
    assert meta == {"org": "东吴证券", "stock_name": "璞泰来", "stock_code": "603659",
                    "title": "2026H1业绩预告点评：负极、隔膜放量贡献盈利弹性",
                    "publish_date": "2026-07-20"}


def test_parse_hibor_metadata_tolerant_fallbacks():
    """无代码标题（个股字段置空）、标题含连字符、外资机构、非法输入。"""
    meta = rc.parse_hibor_metadata("粤开证券-【粤开宏观】坚定看好A股：拥挤交易退潮与新一轮蓄力-260721")
    assert meta["org"] == "粤开证券"
    assert meta["stock_code"] == "" and meta["stock_name"] == ""
    assert meta["title"] == "【粤开宏观】坚定看好A股：拥挤交易退潮与新一轮蓄力"
    assert meta["publish_date"] == "2026-07-21"

    # 标题内部含 '-'（取最后一个 -YYMMDD 为日期）
    meta2 = rc.parse_hibor_metadata("国信证券-CXO行业系列专题报告（5）：新分子蓬勃发展-260720")
    assert meta2["org"] == "国信证券" and meta2["publish_date"] == "2026-07-20"

    # 外资机构英文名也按券商段解析
    meta3 = rc.parse_hibor_metadata("TD Cowen-Q2 REIT Preview-260714")
    assert meta3["org"] == "TD Cowen" and meta3["stock_code"] == ""

    # 无日期尾缀 / 非法日期 / 空输入 → None（跳过该条目）
    assert rc.parse_hibor_metadata("天键股份：2025年年度报告（更正后）") is None
    assert rc.parse_hibor_metadata("XX证券-某标题-261399") is None  # 13 月非法
    assert rc.parse_hibor_metadata("") is None
    assert rc.parse_hibor_metadata(None) is None


HIBOR_HTML = """<html><body>
<table class="tab_ltnew" id="tableList">
<tr><td><span class="tab_lta"><a href="/data/0994c8805d9b9123bda23cc965137404.html">国金证券-中国人保-601319-预计Q2利润高增长，低估值下有望迎来估值修复-260721</a></span></td></tr>
<tr><td><a href="/data/0994c8805d9b9123bda23cc965137404.html">[详细]</a> 经营分析：财险保费稳健增长</td></tr>
<tr><td>2026-07-21分享者：mrsi******ood作者：张三评级：页数：30 页</td></tr>
<tr><td><a href="/data/b416c2e8d62618df6dab24b007b1aca2.html">开源证券-基础化工行业周报：钛白粉景气上行-260721</a></td></tr>
<tr><td><a href="/data/b416c2e8d62618df6dab24b007b1aca2.html">[详细]</a></td></tr>
<tr><td>2026-07-21分享者：abc作者：李四评级：页数：12 页</td></tr>
</table>
<div class="rt-list-div hot-list"><ul>
<li><a href="/data/2f6b291f75ee2dd3c445709df697eccd.html" title="国信证券-CXO行业系列专题报告（5）：新分子蓬勃发展-260720">3.国信证券-CXO…</a></li>
</ul></div>
<a href="/report/621c3ee86874e6ec2cd0ee18d7b55faf.html" title="天键股份：2025年年度报告（更正后）">公告</a>
</body></html>"""


def test_parse_hibor_list_table_sidebar_and_exclusions():
    """表格标题行 + 侧边栏 title 属性 + 作者回填；[详细] 与公告链接排除。"""
    records = rc.parse_hibor_list(HIBOR_HTML)
    assert len(records) == 3
    by_org = {r["org"]: r for r in records}
    r1 = by_org["国金证券"]
    assert r1["stock_name"] == "中国人保" and r1["stock_code"] == "601319"
    assert r1["title"] == "预计Q2利润高增长，低估值下有望迎来估值修复"
    assert r1["publish_date"] == "2026-07-21"
    assert r1["author"] == "张三"  # 元数据行回填
    assert r1["source"] == "hibor"
    r2 = by_org["开源证券"]
    assert r2["stock_code"] == "" and r2["author"] == "李四"
    r3 = by_org["国信证券"]  # 侧边栏：title 属性承载元数据，无作者行
    assert r3["publish_date"] == "2026-07-20" and r3["author"] == ""
    # 公告链接（/report/{md5}.html）被排除
    assert all("天键股份" not in r["title"] for r in records)


def test_hibor_iter_records_early_stop_on_old_dates():
    """列表日期倒序：本页最新一篇已早于 start → 不再翻页。"""
    old_html = HIBOR_HTML.replace("-260721", "-260610").replace("-260720", "-260609")
    calls = []
    src = rc.HiborSource(categories=((1, "公司调研"),), max_pages=3,
                         http_get=make_http(lambda u, k: FakeResp(old_html), calls),
                         sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-20", "2026-07-22"))
    assert records == []  # 全部早于窗口
    assert len(calls) == 1  # 第 1 页即提前终止


def test_hibor_iter_records_paginates_and_filters():
    """正常翻页：page0 用 /microns_1.html，page1 用 /microns_1_1.html；
    窗口外条目被过滤。"""
    page1_html = HIBOR_HTML.replace("-260721", "-260715").replace("-260720", "-260714")
    pages = {"/microns_1.html": HIBOR_HTML, "/microns_1_1.html": page1_html}
    calls = []

    def handler(url, kw):
        for suffix, html in pages.items():
            if url.endswith(suffix):
                return FakeResp(html)
        return FakeResp("<html><body></body></html>")

    src = rc.HiborSource(categories=((1, "公司调研"),), max_pages=2,
                         http_get=make_http(handler, calls), sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-20", "2026-07-22"))
    assert len(records) == 3  # 仅第 1 页（07-20/07-21）落在窗口
    assert len(calls) == 2
    assert calls[0][0].endswith("/microns_1.html")
    assert calls[1][0].endswith("/microns_1_1.html")


# ═══════════════════════════════════════════
# 4. 洞见研报
# ═══════════════════════════════════════════

def djy_payload(items, code=200):
    return {"code": code, "message": "success" if code == 200 else "登录以访问更多数据",
            "data": {"data": items, "meta": {"itemCount": 10001}}}


def djy_item(**over):
    item = {"id": 4618290, "typeId": 10, "title": "贵州茅台2026年一季报点评：稳健增长",
            "orgName": "中金公司", "authors": "张三,李四",
            "publishAt": "2026-07-21T08:00:00.000Z", "stockName": "贵州茅台",
            "fileUrl": "https://storage.djyanbao.com/dj-docs/pdfs/abc.pdf",
            "fileSize": 3409544, "pageTotal": 38}
    item.update(over)
    return item


def test_parse_djyanbao_payload_fields():
    """信封 data.data 解析；publishAt 截断；stockName None 容错；非法输入 []。"""
    payload = djy_payload([djy_item(), djy_item(id=1, title="无股行业报告",
                                                stockName=None, authors="",
                                                publishAt="2026-07-20T16:00:00.000Z")])
    records = rc.parse_djyanbao_payload(payload)
    assert len(records) == 2
    r1 = records[0]
    assert r1["title"] == "贵州茅台2026年一季报点评：稳健增长"
    assert r1["org"] == "中金公司"
    assert r1["author"] == "张三,李四"
    assert r1["publish_date"] == "2026-07-21"  # '...T08:00:00.000Z' 截断
    assert r1["stock_name"] == "贵州茅台" and r1["stock_code"] == ""
    assert r1["encode_url"].endswith("abc.pdf")  # 匿名 403 私有桶，仅作 v2 标识
    assert r1["rating"] == "" and r1["eps_this_year"] is None
    assert r1["source"] == "djyanbao"
    assert r1["info_code"].startswith("djyanbao:")
    assert records[1]["stock_name"] == ""  # None → 空串
    assert rc.parse_djyanbao_payload(None) == []
    assert rc.parse_djyanbao_payload({"data": None}) == []


def test_djyanbao_stops_on_http_401():
    """匿名深页 HTTP 401（登录以访问更多数据）→ 即停，已抓页照常产出。"""
    pages = {
        1: FakeResp(json.dumps(djy_payload([djy_item()]), ensure_ascii=False)),
        2: FakeResp(json.dumps({"code": 401, "message": "登录以访问更多数据"},
                               ensure_ascii=False), status_code=401),
    }
    calls = []
    src = rc.DjyanbaoSource(http_get=make_http(lambda u, k: pages[k["params"]["page"]], calls),
                            sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-21", "2026-07-21"))
    assert len(records) == 1
    assert len(calls) == 2  # page1 通、page2 被拒即停
    assert src.stats["fetched"] == 1


def test_djyanbao_stops_on_body_code_401_and_empty_page():
    """200 但响应体 code=401 → 即停；空 data 页 → 即停。"""
    src = rc.DjyanbaoSource(
        http_get=make_http(lambda u, k: FakeResp(
            json.dumps({"code": 401, "message": "登录以访问更多数据", "data": None},
                       ensure_ascii=False))),
        sleep=make_sleep([]))
    assert list(src.iter_records("2026-07-21", "2026-07-21")) == []
    assert src.stats["pages"] == 1

    src2 = rc.DjyanbaoSource(
        http_get=make_http(lambda u, k: FakeResp(
            json.dumps(djy_payload([]), ensure_ascii=False))),
        sleep=make_sleep([]))
    assert list(src2.iter_records("2026-07-21", "2026-07-21")) == []
    assert src2.stats["pages"] == 1


def test_djyanbao_date_filter_is_local():
    """列表非日期序：窗口外条目本地过滤，抓取继续到上限（不因日期提前终止）。"""
    pages = {
        1: djy_payload([djy_item(), djy_item(id=2, publishAt="2025-09-17T07:10:38.000Z",
                                             title="旧报告")]),
        2: djy_payload([djy_item(id=3, title="另一篇窗口内报告")]),
    }
    src = rc.DjyanbaoSource(max_pages=2,
                            http_get=make_http(lambda u, k: FakeResp(
                                json.dumps(pages[k["params"]["page"]],
                                           ensure_ascii=False))),
                            sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-21", "2026-07-21"))
    assert [r["title"] for r in records] == ["贵州茅台2026年一季报点评：稳健增长",
                                             "另一篇窗口内报告"]  # 2025 年旧报告被过滤
    assert src.stats["pages"] == 2  # 两页都抓（不因日期提前终止）


# ═══════════════════════════════════════════
# 5. 证券之星（GBK 字节输入）
# ═══════════════════════════════════════════

SS_COLUMN_GBK = """<html><body><div class="list"><ul>
<li><span>2026-07-21 21:53:00</span><a href="https://stock.stockstar.com/JC2026072100044042.shtml">主业企稳，切入硅基新材料领域打开增长天花板</a></li>
<li><span>2026-07-21 17:28:00</span><a href="https://stock.stockstar.com/JC2026072100040166.shtml">原料药量价齐升，宠物药开辟空间</a></li>
<li><a href="https://stock.stockstar.com/list/3491.shtml">更多研报</a></li>
</ul></div></body></html>""".encode("gbk")


def test_parse_stockstar_columns_gbk_bytes():
    """五栏目列表：GBK 字节解码、时间戳截断为日期、无日期条目跳过。"""
    records = rc.parse_stockstar_columns(SS_COLUMN_GBK)
    assert len(records) == 2
    r1 = records[0]
    assert r1["title"] == "主业企稳，切入硅基新材料领域打开增长天花板"
    assert r1["publish_date"] == "2026-07-21"  # '2026-07-21 21:53:00' 截断
    assert r1["org"] == "" and r1["stock_code"] == ""  # 列表无机构/个股字段
    assert r1["source"] == "stockstar"
    assert r1["info_code"].startswith("stockstar:")


SS_DATA_ALL_GBK = """<html><body><table>
<tr><th>序号</th><th>证券代码</th><th>证券简称</th><th>研究机构</th><th>最新评级</th><th>目标价</th><th>报告日收盘价</th><th>预期涨幅</th><th>盈利预测</th><th>报告日期</th><th>报告摘要</th></tr>
<tr><th>26年EPS</th><th>27年EPS</th><th>28年EPS</th></tr>
<tr><td>1</td><td>605366</td><td>宏柏新材</td><td>中航证券</td><td>增持</td><td>-</td><td>7.31</td><td>-</td><td>0.00</td><td>0.42</td><td>0.81</td><td>2026-07-21</td><td>报告摘要</td></tr>
<tr><td colspan="13">主业企稳，切入硅基新材料领域打开增长天花板宏柏新材(605366)营收小幅增长</td></tr>
<tr><td>2</td><td>920176</td><td>维琪科技</td><td>华金证券</td><td>-</td><td>-</td><td>22.16</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2026-07-21</td><td>报告摘要</td></tr>
<tr><td colspan="13">新股覆盖研究：维琪科技维琪科技(920176)投资要点</td></tr>
</table></body></html>""".encode("gbk")

SS_DATA_IH_GBK = """<html><body><table>
<tr><th>序号</th><th>证券代码</th><th>证券简称</th><th>研究机构</th><th>最新评级</th><th>评级变动</th><th>目标价</th><th>报告日收盘价</th><th>预期涨幅</th><th>盈利预测</th><th>报告日期</th><th>报告摘要</th></tr>
<tr><td>1</td><td>603071</td><td>物产环能</td><td>中邮证券</td><td>买入</td><td>调高</td><td>30.50</td><td>11.41</td><td>-</td><td>1.44</td><td>1.52</td><td>1.63</td><td>2026-07-09</td><td>报告摘要</td></tr>
<tr><td colspan="14">热电联产稳健增长，积极布局新能源物产环能(603071)投资要点</td></tr>
</table></body></html>""".encode("gbk")


def test_parse_stockstar_data_table_all_13_cells():
    """指标速递（13 格）：评级/EPS/目标价映射、'-' 占位处理、标题从摘要行提取。"""
    records = rc.parse_stockstar_data_table(SS_DATA_ALL_GBK)
    assert len(records) == 2
    r1 = records[0]
    assert r1["stock_code"] == "605366" and r1["stock_name"] == "宏柏新材"
    assert r1["org"] == "中航证券" and r1["rating"] == "增持"
    assert r1["rating_change"] == ""  # data_all 无评级变动列
    assert r1["target_price_high"] is None and r1["target_price_low"] is None
    assert r1["eps_this_year"] == 0.0 and r1["eps_next_year"] == 0.42
    assert r1["publish_date"] == "2026-07-21"
    assert r1["title"] == "主业企稳，切入硅基新材料领域打开增长天花板"  # 摘要行前缀
    assert r1["source"] == "stockstar"
    r2 = records[1]  # 北交所新股：评级/EPS 全 '-' 占位
    assert r2["stock_code"] == "920176" and r2["rating"] == ""
    assert r2["eps_this_year"] is None and r2["eps_next_year"] is None
    assert r2["title"] == "新股覆盖研究：维琪科技"


def test_parse_stockstar_data_table_ih_14_cells():
    """评级调高（14 格含评级变动列）：列右移对齐、目标价单值→高低同值。"""
    records = rc.parse_stockstar_data_table(SS_DATA_IH_GBK)
    assert len(records) == 1
    r = records[0]
    assert r["stock_code"] == "603071" and r["org"] == "中邮证券"
    assert r["rating"] == "买入" and r["rating_change"] == "调高"
    assert r["target_price_high"] == 30.5 and r["target_price_low"] == 30.5
    assert r["eps_this_year"] == 1.44 and r["eps_next_year"] == 1.52
    assert r["publish_date"] == "2026-07-09"
    assert r["title"] == "热电联产稳健增长，积极布局新能源"


def test_stockstar_iter_records_covers_columns_and_data_pages():
    """iter_records 覆盖五栏目 + 三结构化栏目；单分支失败继续其他分支。"""

    def handler(url, kw):
        if "report_list/report1.htm" in url:
            return FakeResp(SS_COLUMN_GBK)
        if "report_list/report2.htm" in url:
            return FakeResp(b"err", 500)  # 单栏目失败
        if "report/data_all.htm" in url:
            return FakeResp(SS_DATA_ALL_GBK)
        return FakeResp("<html><body></body></html>".encode("gbk"))

    src = rc.StockstarSource(http_get=make_http(handler), sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-09", "2026-07-21"))
    assert len(records) == 4  # 栏目 2 条 + 指标速递 2 条
    assert src.stats["pages"] == 8  # 5 栏目 + 3 结构化页


# ═══════════════════════════════════════════
# 6. 契约完整性（16 字段 + info_code 规则）
# ═══════════════════════════════════════════

def test_record_contract_16_fields_and_info_code_rule():
    """四个源产出的每条记录：恰好 16 个契约字段、类型合法、
    非东财源 info_code = f"{source}:" + sha1(title+org+publish_date)[:16]。"""
    records = (
        rc.parse_eastmoney_payload(em_payload([em_item()]))
        + rc.parse_hibor_list(HIBOR_HTML)
        + rc.parse_djyanbao_payload(djy_payload([djy_item()]))
        + rc.parse_stockstar_columns(SS_COLUMN_GBK)
        + rc.parse_stockstar_data_table(SS_DATA_ALL_GBK)
    )
    assert len(records) >= 6
    expected_keys = set(rc.RECORD_FIELDS)
    assert len(rc.RECORD_FIELDS) == 16
    for rec in records:
        assert set(rec.keys()) == expected_keys
        assert rec["source"] in ("eastmoney", "hibor", "djyanbao", "stockstar")
        assert isinstance(rec["publish_date"], str) and rec["publish_date"][:4].isdigit()
        assert rec["stock_code"] == "" or (
            len(rec["stock_code"]) == 6 and rec["stock_code"].isdigit())
        for f in ("eps_this_year", "eps_next_year",
                  "target_price_high", "target_price_low"):
            assert rec[f] is None or isinstance(rec[f], float)
        if rec["source"] == "eastmoney":
            assert rec["info_code"].startswith("AP")  # infoCode 原值
        else:
            digest = hashlib.sha1(
                (rec["title"] + rec["org"] + rec["publish_date"]).encode("utf-8")
            ).hexdigest()[:16]
            assert rec["info_code"] == f"{rec['source']}:{digest}"


# ═══════════════════════════════════════════
# 7. 限速与 upsert 机制
# ═══════════════════════════════════════════

def test_rate_limit_sleep_injected_between_requests():
    """限速：同一轮内第 2 次请求起每次先 sleep(rate + 0~0.5 抖动)。"""
    sleeps = []
    pages = {
        1: em_payload([em_item()], total_page=2, hits=2),
        2: em_payload([em_item(title="第二篇", infoCode="AP2")], total_page=2, hits=2),
    }
    src = rc.EastmoneySource(qtypes=(0,), rate=1.0,
                             http_get=make_http(
                                 lambda u, k: FakeResp(json.dumps(
                                     pages[k["params"]["pageNo"]], ensure_ascii=False))),
                             sleep=make_sleep(sleeps))
    list(src.iter_records("2026-07-21", "2026-07-21"))
    assert len(sleeps) == 1  # 2 次请求 → 1 次限速（首轮首个请求不限速）
    assert 1.0 <= sleeps[0] <= 1.5


def test_upsert_injection_writes_and_failure_tolerated(caplog):
    """iter_records 注入 upsert：按页批落库并累计写入条数；
    upsert 抛异常时记 warning、记录照常产出。"""
    written = []

    def fake_upsert(records):
        written.extend(records)
        return len(records)

    src = rc.EastmoneySource(qtypes=(0,),
                             http_get=make_http(lambda u, k: FakeResp(
                                 json.dumps(em_payload([em_item()], total_page=1),
                                            ensure_ascii=False))),
                             sleep=make_sleep([]))
    records = list(src.iter_records("2026-07-21", "2026-07-21", upsert=fake_upsert))
    assert len(records) == 1 and written == records
    assert src.stats["upserted"] == 1 and src.stats["fetched"] == 1

    def boom_upsert(records):
        raise RuntimeError("db locked")

    src2 = rc.EastmoneySource(qtypes=(0,),
                              http_get=make_http(lambda u, k: FakeResp(
                                  json.dumps(em_payload([em_item()], total_page=1),
                                             ensure_ascii=False))),
                              sleep=make_sleep([]))
    with caplog.at_level("WARNING"):
        records2 = list(src2.iter_records("2026-07-21", "2026-07-21", upsert=boom_upsert))
    assert len(records2) == 1  # 落库失败不影响产出
    assert src2.stats["upserted"] == 0


# ═══════════════════════════════════════════
# 8. CLI
# ═══════════════════════════════════════════

def make_fake_lib(store):
    """fake 存储层（契约签名）：init_db(db_path)->str / upsert_reports(records, db)->int。"""
    return SimpleNamespace(
        init_db=lambda db_path=None: store.setdefault("db", db_path or "fake_reports.db"),
        upsert_reports=lambda records, db_path=None: (
            store.setdefault("records", []).extend(records) or len(records)),
    )


def test_cli_runs_with_fake_upsert_and_prints_stats(monkeypatch, capsys):
    """CLI：fake 存储层 + fake http/sleep 跑通，落库条数与打印统计一致。"""
    store = {}
    monkeypatch.setattr(rc, "_load_report_library", lambda: make_fake_lib(store))

    def handler(url, kw):
        if kw.get("params", {}).get("qType") == 0:
            item = em_item(publishDate=f"{TODAY} 00:00:00.000")
            return FakeResp(json.dumps(em_payload([item], total_page=1),
                                       ensure_ascii=False))
        return FakeResp(json.dumps(em_payload([], total_page=0, hits=0),
                                   ensure_ascii=False))

    exit_code = rc.main(["--days", "1", "--sources", "eastmoney", "--rate", "0"],
                        http_get=make_http(handler), sleep=make_sleep([]))
    assert exit_code == 0
    assert len(store["records"]) == 1
    assert store["records"][0]["source"] == "eastmoney"
    out = capsys.readouterr().out
    assert "[eastmoney] 抓取 1 篇，入库 1 篇，请求 4 次" in out
    assert "合计：抓取 1 篇，入库 1 篇" in out


def test_cli_source_failure_continues_others(monkeypatch, caplog):
    """单源失败（HTTP 异常）记 warning 并继续其他源，退出码仍为 0。"""
    store = {}
    monkeypatch.setattr(rc, "_load_report_library", lambda: make_fake_lib(store))
    ss_today = SS_COLUMN_GBK.decode("gbk").replace("2026-07-21", TODAY).encode("gbk")

    def handler(url, kw):
        if "eastmoney.com" in url:
            raise ConnectionError("boom")
        if "report_list/report1.htm" in url:
            return FakeResp(ss_today)
        return FakeResp("<html><body></body></html>".encode("gbk"))

    with caplog.at_level("WARNING"):
        exit_code = rc.main(["--days", "1", "--sources", "eastmoney,stockstar",
                             "--rate", "0"],
                            http_get=make_http(handler), sleep=make_sleep([]))
    assert exit_code == 0
    assert len(store["records"]) == 2  # 证券之星照常落库
    assert all(r["source"] == "stockstar" for r in store["records"])
    assert any("eastmoney" in rec.message for rec in caplog.records)


def test_cli_rejects_all_unknown_sources(monkeypatch, capsys):
    """--sources 全部未知 → 打印提示并返回 2（不触存储层）。"""
    monkeypatch.setattr(rc, "_load_report_library",
                        lambda: pytest.fail("不应触达存储层"))
    exit_code = rc.main(["--sources", "xueqiu,hexun"], http_get=make_http(
        lambda u, k: FakeResp(b"")), sleep=make_sleep([]))
    assert exit_code == 2
    assert "无有效数据源" in capsys.readouterr().out
