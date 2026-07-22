"""agent/report_library.py 研报库存储与检索层（研报库 v1）测试。

覆盖范围：
1. 建库：init_db 幂等；17 列齐全（含 source 列 DEFAULT 'eastmoney'）；两个索引照图纸。
2. 路径解析：显式参数 > REPORTS_DB_PATH > ${DATA_DIR:-data}/reports.db；
   env 调用时惰性读取（import 后改 env 立即生效）。
3. upsert：新增计数；同 info_code 重复 upsert 幂等（行数不膨胀）；
   缺 info_code/非 dict 记录跳过不抛出；空输入返回 0。
4. 冲突合并：高优先级源后到低优先级源 → 高优先级赢；低优先级源后到 → 不改变；
   旧值空(NULL/空串)由新值回填；新值为空不覆盖旧值。
5. search_reports：days/stock_code(带交易所前缀)/industry/query 各过滤维度；
   publish_date DESC 排序；limit 夹取；total 不受 limit 截断；
   空库/文件不存在/无表文件均返回 total=0 合法结构。
6. rating_summary：评级分布/目标价区间/EPS 均值(2位)/最新 3 篇；
   全空字段 → None/空结构；空库 total=0。
7. stock_code 入库归一化：SH600519 存为 600519。

规则（与项目其他测试一致）：全 mock 零网络；REPORTS_DB_PATH 一律
monkeypatch 到 tmp_path，绝不写真实 data/。
"""

import os
import sqlite3
from datetime import date, timedelta

import pytest

import agent.report_library as rl


# ── 工具 ──


def _iso(days_ago: int) -> str:
    """N 天前的 YYYY-MM-DD。"""
    return (date.today() - timedelta(days=days_ago)).isoformat()


def make_record(**kw) -> dict:
    """构造一条合法研报记录（16 字段 + source），按需覆盖。"""
    rec = {
        "info_code": "IC-DEFAULT",
        "title": "贵州茅台：稳健增长",
        "org": "中金公司",
        "author": "张三",
        "publish_date": _iso(1),
        "stock_code": "600519",
        "stock_name": "贵州茅台",
        "industry": "食品饮料",
        "rating": "买入",
        "rating_change": "维持",
        "eps_this_year": 68.5,
        "eps_next_year": 75.0,
        "target_price_high": 2100.0,
        "target_price_low": 1800.0,
        "encode_url": "enc123",
        "source": "eastmoney",
    }
    rec.update(kw)
    return rec


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """把研报库隔离到 tmp_path，绝不触碰真实 data/。返回库路径字符串。"""
    p = tmp_path / "reports.db"
    monkeypatch.setenv(rl.REPORTS_DB_PATH_ENV, str(p))
    return str(p)


def _fetch_row(path: str, info_code: str) -> sqlite3.Row:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM reports WHERE info_code = ?", (info_code,)
        ).fetchone()
    finally:
        conn.close()


# ── 1. 建库与 schema ──


def test_init_db_idempotent_and_schema(db):
    path1 = rl.init_db()
    path2 = rl.init_db()
    assert path1 == path2 == db
    assert os.path.exists(db)

    conn = sqlite3.connect(db)
    try:
        cols = {r[1]: r for r in conn.execute("PRAGMA table_info(reports)")}
    finally:
        conn.close()

    expected = {
        "info_code", "title", "org", "author", "publish_date",
        "stock_code", "stock_name", "industry",
        "rating", "rating_change",
        "eps_this_year", "eps_next_year",
        "target_price_high", "target_price_low",
        "encode_url", "source", "created_at",
    }
    assert expected <= set(cols)
    # source 列带 DEFAULT 'eastmoney'
    assert cols["source"][4] == "'eastmoney'"
    # info_code 为主键
    assert cols["info_code"][5] == 1

    conn = sqlite3.connect(db)
    try:
        indexes = {
            r[1] for r in conn.execute("PRAGMA index_list(reports)")
        }
    finally:
        conn.close()
    assert "idx_reports_stock" in indexes
    assert "idx_reports_industry" in indexes


def test_init_db_creates_missing_parent_dirs(tmp_path, monkeypatch):
    deep = tmp_path / "a" / "b" / "reports.db"
    monkeypatch.setenv(rl.REPORTS_DB_PATH_ENV, str(deep))
    assert rl.init_db() == str(deep)
    assert os.path.exists(deep)


# ── 2. 路径解析 ──


def test_db_path_resolution_priority(tmp_path, monkeypatch):
    explicit = str(tmp_path / "explicit.db")
    env_path = str(tmp_path / "env.db")
    data_dir = str(tmp_path / "datadir")

    # 显式参数 > 环境变量
    monkeypatch.setenv(rl.REPORTS_DB_PATH_ENV, env_path)
    assert rl._db_path(explicit) == explicit
    # 环境变量 > DATA_DIR 推导
    monkeypatch.setenv(rl.DATA_DIR_ENV, data_dir)
    assert rl._db_path() == env_path
    assert rl._db_path(None) == env_path
    # DATA_DIR 推导兜底
    monkeypatch.delenv(rl.REPORTS_DB_PATH_ENV)
    assert rl._db_path() == os.path.join(data_dir, "reports.db")
    # 全缺省 → data/reports.db（相对路径，仅比较字符串不落盘）
    monkeypatch.delenv(rl.DATA_DIR_ENV)
    assert rl._db_path() == os.path.join("data", "reports.db")


def test_db_path_env_read_lazily(tmp_path, monkeypatch):
    """import 后修改 env 立即生效：路径解析不绑定 import 时的环境。"""
    monkeypatch.delenv(rl.REPORTS_DB_PATH_ENV, raising=False)
    first = rl._db_path()
    later = str(tmp_path / "later.db")
    monkeypatch.setenv(rl.REPORTS_DB_PATH_ENV, later)
    assert rl._db_path() == later
    assert rl._db_path() != first


# ── 3. upsert 基本行为 ──


def test_upsert_insert_and_conflict_idempotent(db):
    rl.init_db()
    r1 = make_record(info_code="IC-1")
    r2 = make_record(info_code="IC-2", title="宁德时代：点评",
                     stock_code="300750", stock_name="宁德时代")
    assert rl.upsert_reports([r1, r2]) == 2

    # 同 info_code 同源重复 upsert：返回更新条数 1，总行数不膨胀
    assert rl.upsert_reports([make_record(info_code="IC-1")]) == 1
    assert rl.search_reports(days=365)["total"] == 2

    row = _fetch_row(db, "IC-1")
    assert row["title"] == "贵州茅台：稳健增长"
    assert row["source"] == "eastmoney"
    assert row["created_at"]  # 有写入时间戳


def test_upsert_skips_invalid_records(db):
    rl.init_db()
    assert rl.upsert_reports([]) == 0
    assert rl.upsert_reports(None) == 0
    bad = [
        {"title": "没有 info_code"},
        "not-a-dict",
        None,
        make_record(info_code=""),
        make_record(info_code="IC-OK"),
    ]
    assert rl.upsert_reports(bad) == 1
    assert rl.search_reports(days=365)["total"] == 1


def test_upsert_normalizes_stock_code(db):
    rl.init_db()
    rl.upsert_reports([make_record(info_code="IC-PFX", stock_code="SH600519")])
    row = _fetch_row(db, "IC-PFX")
    assert row["stock_code"] == "600519"
    # 归一化后各形态都能精确命中
    assert rl.search_reports(stock_code="600519")["total"] == 1
    assert rl.search_reports(stock_code="sh600519")["total"] == 1


# ── 4. 跨源冲突合并 ──


def test_conflict_merge_higher_priority_wins_regardless_of_order(db):
    rl.init_db()
    same_ic = "IC-SHARED"

    # 场景一：低优先级（慧博）先到，高优先级（东财）后到 → 东财赢
    rl.upsert_reports([make_record(
        info_code=same_ic, source="hibor", rating="增持", rating_change="调低",
        eps_this_year=60.0, target_price_high=2000.0, target_price_low=1700.0,
    )])
    rl.upsert_reports([make_record(
        info_code=same_ic, source="eastmoney", rating="买入", rating_change="维持",
        eps_this_year=68.5, target_price_high=2100.0, target_price_low=1800.0,
    )])
    row = _fetch_row(db, same_ic)
    assert row["rating"] == "买入"
    assert row["rating_change"] == "维持"
    assert row["eps_this_year"] == 68.5
    assert row["target_price_high"] == 2100.0

    # 场景二：高优先级（东财）先到，低优先级（慧博）后到 → 东财保持不变
    rl.upsert_reports([make_record(
        info_code=same_ic, source="hibor", rating="中性", rating_change="调低",
        eps_this_year=55.0, target_price_high=1900.0, target_price_low=1600.0,
    )])
    row = _fetch_row(db, same_ic)
    assert row["rating"] == "买入"
    assert row["eps_this_year"] == 68.5
    assert row["target_price_low"] == 1800.0

    # 中间档：证券之星低于东财，也改变不了东财的值
    rl.upsert_reports([make_record(
        info_code=same_ic, source="stockstar", rating="卖出",
    )])
    row = _fetch_row(db, same_ic)
    assert row["rating"] == "买入"


def test_conflict_merge_backfill_empty_fields(db):
    rl.init_db()
    same_ic = "IC-BACKFILL"

    # 东财先到但 rating 为空、EPS 为 None
    rl.upsert_reports([make_record(
        info_code=same_ic, source="eastmoney",
        rating="", eps_this_year=None, author="李四",
    )])
    # 慧博后到：旧值空 → 回填；旧值非空（author）→ 低优先级不覆盖
    rl.upsert_reports([make_record(
        info_code=same_ic, source="hibor",
        rating="增持", eps_this_year=61.2, author="王五",
        title="",  # 新值为空 → 不覆盖东财标题
    )])
    row = _fetch_row(db, same_ic)
    assert row["rating"] == "增持"          # 空值回填
    assert row["eps_this_year"] == 61.2     # None 回填
    assert row["author"] == "李四"          # 非空冲突：东财优先级高
    assert row["title"] == "贵州茅台：稳健增长"  # 新值为空不覆盖


# ── 5. search_reports ──


def test_search_days_filter(db):
    rl.init_db()
    rl.upsert_reports([
        make_record(info_code="IC-NEW", publish_date=_iso(5)),
        make_record(info_code="IC-OLD", publish_date=_iso(60)),
    ])
    result = rl.search_reports(days=30)
    assert result["total"] == 1
    assert result["reports"][0]["date"] == _iso(5)
    # 放宽天数则两篇都命中
    assert rl.search_reports(days=90)["total"] == 2


def test_search_stock_code_with_exchange_prefix(db):
    rl.init_db()
    rl.upsert_reports([
        make_record(info_code="IC-MT", stock_code="600519"),
        make_record(info_code="IC-CATL", stock_code="300750",
                    stock_name="宁德时代", title="宁德时代：点评"),
    ])
    for form in ("600519", "sh600519", "SH600519"):
        result = rl.search_reports(stock_code=form)
        assert result["total"] == 1
        assert result["reports"][0]["stock_code"] == "600519"
    assert rl.search_reports(stock_code="sz300750")["total"] == 1


def test_search_industry_like(db):
    rl.init_db()
    rl.upsert_reports([
        make_record(info_code="IC-F1", industry="食品饮料"),
        make_record(info_code="IC-F2", industry="食品", stock_code="000001",
                    stock_name="示例股", title="食品行业点评"),
        make_record(info_code="IC-E1", industry="电子", stock_code="002475",
                    stock_name="立讯精密", title="电子行业点评"),
    ])
    result = rl.search_reports(industry="食品")
    assert result["total"] == 2
    assert rl.search_reports(industry="电子")["total"] == 1
    assert rl.search_reports(industry="不存在的行业")["total"] == 0


def test_search_query_matches_title_stock_name_org(db):
    rl.init_db()
    rl.upsert_reports([
        make_record(info_code="IC-Q1", title="白酒行业深度：复苏可期",
                    stock_code="", stock_name="", industry="食品饮料"),
        make_record(info_code="IC-Q2", org="中信证券", title="不起眼的标题",
                    stock_code="601318", stock_name="中国平安", industry="非银金融"),
        make_record(info_code="IC-Q3", stock_name="比亚迪", stock_code="002594",
                    title="公司点评", org="华泰证券", industry="汽车"),
    ])
    assert rl.search_reports(query="白酒")["total"] == 1       # 命中 title
    assert rl.search_reports(query="中信")["total"] == 1       # 命中 org
    assert rl.search_reports(query="比亚迪")["total"] == 1      # 命中 stock_name
    assert rl.search_reports(query="")["total"] == 3            # 空 query 不过滤
    assert rl.search_reports(query="茅台")["total"] == 0


def test_search_order_and_limit_and_total(db):
    rl.init_db()
    recs = [
        make_record(info_code=f"IC-D{i}", publish_date=_iso(i),
                    title=f"研报{i}", stock_code="600519")
        for i in (10, 1, 5, 3, 7)
    ]
    rl.upsert_reports(recs)

    result = rl.search_reports(stock_code="600519", days=30)
    dates = [r["date"] for r in result["reports"]]
    assert dates == sorted(dates, reverse=True)  # publish_date DESC

    limited = rl.search_reports(stock_code="600519", days=30, limit=2)
    assert limited["total"] == 5                 # total 不受 limit 截断
    assert len(limited["reports"]) == 2
    assert limited["reports"][0]["date"] == _iso(1)  # 最新在前

    # limit 夹取 1-50：0 → 夹到 1；500 → 夹到 50（够装全部 5 篇）
    assert len(rl.search_reports(limit=0)["reports"]) == 1
    assert len(rl.search_reports(limit=500)["reports"]) == 5


def test_search_report_item_fields(db):
    rl.init_db()
    rl.upsert_reports([make_record(info_code="IC-FMT")])
    item = rl.search_reports(stock_code="sh600519")["reports"][0]
    assert item["title"] == "贵州茅台：稳健增长"
    assert item["org"] == "中金公司"
    assert item["author"] == "张三"
    assert item["date"] == _iso(1)
    assert item["rating"] == "买入"
    assert item["rating_change"] == "维持"
    assert item["target_price"] == "1800~2100"
    assert item["eps_forecast"] == 68.5
    assert item["eps_next_year"] == 75.0
    assert item["stock_code"] == "600519"
    assert item["stock_name"] == "贵州茅台"
    assert item["industry"] == "食品饮料"
    assert item["source"] == "eastmoney"

    # 目标价全空 → None；仅单侧 → 单侧字符串
    rl.upsert_reports([make_record(
        info_code="IC-NOTP", target_price_high=None, target_price_low=None,
        publish_date=_iso(0),
    )])
    rl.upsert_reports([make_record(
        info_code="IC-HALFTP", target_price_low=None, target_price_high=1999.5,
        publish_date=_iso(0),
    )])
    reports = rl.search_reports(stock_code="600519")["reports"]
    ic_no = [r for r in reports if r["target_price"] is None]
    assert len(ic_no) == 1
    half = [r for r in reports if r["target_price"] == "1999.5"]
    assert len(half) == 1


def test_search_empty_db_returns_zero(db):
    # 库文件不存在 → total=0 合法结构，不建文件
    assert rl.search_reports(query="茅台") == {"total": 0, "reports": []}
    assert not os.path.exists(db)
    # init 后空库 → 同样结构
    rl.init_db()
    assert rl.search_reports() == {"total": 0, "reports": []}


def test_search_db_file_without_table(tmp_path, monkeypatch):
    """存在但无表的库文件（如损坏/空文件）→ 返回 total=0 而非报错。"""
    p = tmp_path / "broken.db"
    p.touch()
    monkeypatch.setenv(rl.REPORTS_DB_PATH_ENV, str(p))
    assert rl.search_reports()["total"] == 0
    assert rl.rating_summary()["total"] == 0


# ── 6. rating_summary ──


def test_rating_summary_full(db):
    rl.init_db()
    rl.upsert_reports([
        make_record(info_code="IC-S1", rating="买入", eps_this_year=60.0,
                    target_price_low=1800.0, target_price_high=2100.0,
                    publish_date=_iso(0), title="最新一篇", org="中金公司"),
        make_record(info_code="IC-S2", rating="买入", eps_this_year=70.0,
                    target_price_low=1900.0, target_price_high=2200.0,
                    publish_date=_iso(1), title="次新一篇", org="中信证券",
                    source="stockstar"),
        make_record(info_code="IC-S3", rating="增持", eps_this_year=None,
                    target_price_low=None, target_price_high=None,
                    publish_date=_iso(2), title="第三篇", org="华泰证券",
                    source="hibor"),
        make_record(info_code="IC-S4", rating="中性", publish_date=_iso(3),
                    title="第四篇", org="国泰君安", eps_this_year=None),
        # 其他股票：不应计入 600519 的统计
        make_record(info_code="IC-OTHER", stock_code="300750",
                    stock_name="宁德时代", rating="买入",
                    publish_date=_iso(0), title="别的股票"),
    ])
    s = rl.rating_summary(stock_code="sh600519", days=30)
    assert s["total"] == 4
    assert s["rating_dist"] == {"买入": 2, "增持": 1, "中性": 1}
    assert s["target_price_range"] == [1800, 2200]
    assert s["avg_eps_forecast"] == 65.0        # (60+70)/2，忽略 None
    assert len(s["latest_reports"]) == 3        # 最多 3 篇
    assert s["latest_reports"][0] == {
        "title": "最新一篇", "org": "中金公司", "date": _iso(0),
    }
    assert s["latest_reports"][1]["title"] == "次新一篇"
    assert s["latest_reports"][2]["title"] == "第三篇"


def test_rating_summary_all_null_fields(db):
    rl.init_db()
    rl.upsert_reports([make_record(
        info_code="IC-NULL", rating="", eps_this_year=None,
        target_price_low=None, target_price_high=None,
    )])
    s = rl.rating_summary(stock_code="600519")
    assert s["total"] == 1
    assert s["rating_dist"] == {}
    assert s["target_price_range"] is None
    assert s["avg_eps_forecast"] is None
    assert len(s["latest_reports"]) == 1


def test_rating_summary_empty_db(db):
    # 库文件不存在 → total=0 合法结构，不建文件
    assert rl.rating_summary(stock_code="600519") == {
        "total": 0, "rating_dist": {}, "target_price_range": None,
        "avg_eps_forecast": None, "latest_reports": [],
    }
    assert not os.path.exists(db)
    rl.init_db()
    assert rl.rating_summary()["total"] == 0


def test_rating_summary_industry_and_days(db):
    rl.init_db()
    rl.upsert_reports([
        make_record(info_code="IC-I1", industry="食品饮料", publish_date=_iso(2)),
        make_record(info_code="IC-I2", industry="食品饮料", publish_date=_iso(40)),
        make_record(info_code="IC-I3", industry="电子", publish_date=_iso(1),
                    stock_code="002475", stock_name="立讯精密"),
    ])
    s = rl.rating_summary(industry="食品", days=30)
    assert s["total"] == 1                       # 40 天前那篇被 days 排除
    assert rl.rating_summary(industry="食品", days=90)["total"] == 2
