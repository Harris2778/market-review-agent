"""tests/test_social_store.py — social_store SQLite 持久化测试（零网络）。

覆盖：建表幂等、路径解析优先级、upsert 幂等与 hit_count、无效条目跳过、
关键词 LIKE 过滤、平台/时间/limit 过滤、坏路径降级（绝不抛）。
"""

import os
import sqlite3

import pytest

from agent import social_store


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "social.db")


def _post(platform="weibo", post_id="p1", title="标题", content="正文",
          author="作者", metrics=None, url="https://x", published_at=""):
    return {
        "platform": platform,
        "post_id": post_id,
        "title": title,
        "content": content,
        "author": author,
        "metrics": metrics if metrics is not None else {"likes": 5},
        "url": url,
        "published_at": published_at,
        "source": f"{platform}_hot",
    }


# ── 建表与路径解析 ──


class TestInitDb:
    def test_init_creates_table(self, db):
        assert social_store.init_db(db) is True
        with sqlite3.connect(db) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "social_posts" in names

    def test_init_idempotent(self, db):
        assert social_store.init_db(db) is True
        assert social_store.init_db(db) is True

    def test_init_creates_parent_dirs(self, tmp_path):
        deep = str(tmp_path / "a" / "b" / "social.db")
        assert social_store.init_db(deep) is True
        assert os.path.exists(deep)

    def test_path_priority_param_over_env(self, db, tmp_path, monkeypatch):
        env_db = str(tmp_path / "env.db")
        monkeypatch.setenv("SOCIAL_DB_PATH", env_db)
        social_store.init_db(db)
        assert os.path.exists(db)
        assert not os.path.exists(env_db)

    def test_path_env_social_db_path(self, tmp_path, monkeypatch):
        env_db = str(tmp_path / "env.db")
        monkeypatch.setenv("SOCIAL_DB_PATH", env_db)
        assert social_store.init_db() is True
        assert os.path.exists(env_db)

    def test_path_fallback_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SOCIAL_DB_PATH", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "datadir"))
        assert social_store.init_db() is True
        assert os.path.exists(str(tmp_path / "datadir" / "social.db"))


# ── upsert ──


class TestUpsert:
    def test_upsert_returns_count(self, db):
        posts = [_post(post_id="a"), _post(post_id="b")]
        assert social_store.upsert_posts(posts, db_path=db) == 2

    def test_upsert_idempotent_second_call(self, db):
        posts = [_post(post_id="a"), _post(post_id="b")]
        social_store.upsert_posts(posts, db_path=db)
        # 重复调用仍返回处理条数，但总行数不变
        assert social_store.upsert_posts(posts, db_path=db) == 2
        assert len(social_store.query_posts(db_path=db)) == 2

    def test_hit_count_increments_on_repeat(self, db):
        for _ in range(3):
            social_store.upsert_posts([_post(post_id="x")], db_path=db)
        rows = social_store.query_posts(db_path=db)
        assert rows[0]["hit_count"] == 3
        assert rows[0]["first_seen"] <= rows[0]["last_seen"]

    def test_upsert_refreshes_snapshot_fields(self, db):
        social_store.upsert_posts([_post(post_id="x", title="旧")], db_path=db)
        social_store.upsert_posts([_post(post_id="x", title="新")], db_path=db)
        assert social_store.query_posts(db_path=db)[0]["title"] == "新"

    def test_upsert_skips_invalid_entries(self, db):
        posts = [
            _post(post_id="ok"),
            _post(post_id=""),               # 缺 post_id
            _post(platform="", post_id="y"),  # 缺 platform
            "not-a-dict",
            None,
        ]
        assert social_store.upsert_posts(posts, db_path=db) == 1

    def test_upsert_none_and_empty(self, db):
        assert social_store.upsert_posts(None, db_path=db) == 0
        assert social_store.upsert_posts([], db_path=db) == 0

    def test_upsert_non_dict_metrics_falls_back(self, db):
        social_store.upsert_posts([_post(post_id="m", metrics="oops")],
                                  db_path=db)
        assert social_store.query_posts(db_path=db)[0]["metrics"] == {}


# ── query ──


class TestQuery:
    def test_query_returns_dicts_with_metrics(self, db):
        social_store.upsert_posts(
            [_post(post_id="q", metrics={"likes": 9, "views": 100})],
            db_path=db)
        rows = social_store.query_posts(db_path=db)
        assert rows[0]["metrics"] == {"likes": 9, "views": 100}
        assert rows[0]["platform"] == "weibo"
        assert "metrics_json" not in rows[0]

    def test_query_filter_platform(self, db):
        social_store.upsert_posts(
            [_post(platform="weibo", post_id="w"),
             _post(platform="zhihu", post_id="z")], db_path=db)
        rows = social_store.query_posts(platform="zhihu", db_path=db)
        assert len(rows) == 1 and rows[0]["platform"] == "zhihu"

    def test_query_keyword_like_title_and_content(self, db):
        social_store.upsert_posts(
            [_post(post_id="t", title="茅台涨停", content=""),
             _post(post_id="c", title="无关", content="宁德业绩预增"),
             _post(post_id="n", title="无关", content="无关")], db_path=db)
        assert {r["post_id"] for r in
                social_store.query_posts(keyword="茅台", db_path=db)} == {"t"}
        assert {r["post_id"] for r in
                social_store.query_posts(keyword="预增", db_path=db)} == {"c"}

    def test_query_limit(self, db):
        social_store.upsert_posts([_post(post_id=str(i)) for i in range(10)],
                                  db_path=db)
        assert len(social_store.query_posts(limit=3, db_path=db)) == 3

    def test_query_days_filter(self, db):
        social_store.upsert_posts([_post(post_id="fresh")], db_path=db)
        # 手工塞一条 30 天前的旧记录
        old_ts = "2000-01-01T00:00:00+00:00"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO social_posts (platform, post_id, title, "
                "first_seen, last_seen, hit_count) VALUES ('weibo','old','旧',?,?,1)",
                (old_ts, old_ts))
        recent = social_store.query_posts(days=7, db_path=db)
        assert {r["post_id"] for r in recent} == {"fresh"}
        all_rows = social_store.query_posts(days=0, db_path=db)
        assert {r["post_id"] for r in all_rows} == {"fresh", "old"}

    def test_query_empty_db_returns_list(self, db):
        assert social_store.query_posts(db_path=db) == []


# ── 坏路径降级（绝不抛）──


class TestBadPathDegradation:
    @pytest.fixture
    def bad_path(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")  # 以文件当目录 → 必然失败
        return str(blocker / "sub" / "social.db")

    def test_init_bad_path_returns_false(self, bad_path):
        assert social_store.init_db(bad_path) is False

    def test_upsert_bad_path_returns_zero(self, bad_path):
        assert social_store.upsert_posts([_post()], db_path=bad_path) == 0

    def test_query_bad_path_returns_empty(self, bad_path):
        assert social_store.query_posts(db_path=bad_path) == []
