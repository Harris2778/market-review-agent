"""tests/test_sentiment_aggregate.py — 舆情分布聚合与快照趋势层测试（零网络）。

覆盖：
- aggregate_distribution：标签归并（利好→乐观/利空→悲观/未知→中性+unknown）、
  权重计算（metrics.likes 优先、顶层回退、likes+1 基础票）、pct 正确性、
  置信度三档边界（29/30/99/100）、空样本、绝不抛。
- pick_representatives：分桶取样、likes 降序、per_bucket、瘦身截 80 字、
  title/content 回退、空输入。
- 快照：save_snapshot 存取/幂等 REPLACE/缺键降级/坏路径不抛、
  get_trend 趋势三向判定（转向乐观/转向悲观/基本稳定/数据不足）、
  边界 diff=15、days 过滤、坏路径降级、路径解析优先级。
"""

import os
import sqlite3

import pytest

from agent import sentiment_aggregate as sa


def _item(sentiment="乐观", likes=0, title="标题", content="正文",
          platform="guba", url="https://x", nested_metrics=True):
    """构造已打分条目：nested_metrics=True 时 likes 放 metrics 内，否则顶层。"""
    it = {
        "sentiment": sentiment,
        "sentiment_score": 0.5,
        "title": title,
        "content": content,
        "platform": platform,
        "url": url,
    }
    if nested_metrics:
        it["metrics"] = {"likes": likes}
    else:
        it["likes"] = likes
    return it


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "social.db")


# ═══════════════════════════════════════════
# aggregate_distribution：标签归并
# ═══════════════════════════════════════════


class TestAggregateLabels:
    def test_basic_three_buckets(self):
        items = ([_item("乐观") for _ in range(3)]
                 + [_item("悲观") for _ in range(2)]
                 + [_item("中性")])
        r = sa.aggregate_distribution(items)
        assert r["n"] == 6
        assert r["dist"]["乐观"]["count"] == 3
        assert r["dist"]["悲观"]["count"] == 2
        assert r["dist"]["中性"]["count"] == 1
        assert r["dist"]["乐观"]["pct"] == 50.0
        assert r["dist"]["悲观"]["pct"] == pytest.approx(33.3)
        assert r["dist"]["中性"]["pct"] == pytest.approx(16.7)
        assert r["unknown_count"] == 0

    def test_lexicon_bull_merged_to_positive(self):
        """词典标签「利好」归并到「乐观」。"""
        r = sa.aggregate_distribution([_item("利好"), _item("乐观")])
        assert r["dist"]["乐观"]["count"] == 2
        assert r["dist"]["乐观"]["pct"] == 100.0

    def test_lexicon_bear_merged_to_negative(self):
        """词典标签「利空」归并到「悲观」。"""
        r = sa.aggregate_distribution([_item("利空"), _item("悲观"), _item("中性")])
        assert r["dist"]["悲观"]["count"] == 2
        assert r["dist"]["悲观"]["pct"] == pytest.approx(66.7)

    def test_unknown_label_to_neutral_and_counted(self):
        """未知标签归入中性并计入 unknown_count。"""
        r = sa.aggregate_distribution(
            [_item("乐观"), _item("看多"), _item("???")])
        assert r["n"] == 3
        assert r["dist"]["中性"]["count"] == 2
        assert r["unknown_count"] == 2

    def test_missing_sentiment_key_counts_unknown(self):
        """缺失 sentiment 键 → 中性 + unknown。"""
        it = _item("乐观")
        del it["sentiment"]
        r = sa.aggregate_distribution([it])
        assert r["dist"]["中性"]["count"] == 1
        assert r["unknown_count"] == 1

    def test_non_dict_items_skipped(self):
        """非 dict 条目跳过且不计入 n。"""
        r = sa.aggregate_distribution([_item("乐观"), "junk", None, 42])
        assert r["n"] == 1
        assert r["dist"]["乐观"]["count"] == 1

    def test_method_field(self):
        r = sa.aggregate_distribution([_item("乐观")])
        assert r["method"] == "aggregate_v1"


# ═══════════════════════════════════════════
# aggregate_distribution：权重计算
# ═══════════════════════════════════════════


class TestAggregateWeights:
    def test_weighted_dist_metrics_likes(self):
        """权重 = metrics.likes + 1：乐观 9 赞(10 票) vs 悲观 0 赞(1 票)。"""
        items = [_item("乐观", likes=9), _item("悲观", likes=0)]
        r = sa.aggregate_distribution(items)
        assert r["weighted_dist"]["乐观"] == pytest.approx(10 / 11 * 100, abs=0.1)
        assert r["weighted_dist"]["悲观"] == pytest.approx(1 / 11 * 100, abs=0.1)

    def test_top_level_likes_fallback(self):
        """顶层 likes（无 metrics）作为回退。"""
        items = [_item("乐观", likes=3, nested_metrics=False),
                 _item("悲观", likes=0)]
        r = sa.aggregate_distribution(items)
        # 权重 4 : 1 → 80 : 20
        assert r["weighted_dist"]["乐观"] == 80.0
        assert r["weighted_dist"]["悲观"] == 20.0

    def test_zero_likes_still_has_base_vote(self):
        """0 赞也有基础票（权重 1），不会出现除零/全 0。"""
        r = sa.aggregate_distribution([_item("乐观", likes=0)])
        assert r["weighted_dist"]["乐观"] == 100.0

    def test_missing_likes_defaults_zero(self):
        """metrics 无 likes 键 → 按 0 赞处理。"""
        it = _item("乐观")
        it["metrics"] = {"comments": 7}
        r = sa.aggregate_distribution([it, _item("悲观", likes=0)])
        assert r["weighted_dist"]["乐观"] == 50.0
        assert r["weighted_dist"]["悲观"] == 50.0

    def test_negative_likes_clamped(self):
        """负 likes 截断到 0（权重仍为 1）。"""
        r = sa.aggregate_distribution(
            [_item("乐观", likes=-5), _item("悲观", likes=0)])
        assert r["weighted_dist"]["乐观"] == 50.0

    def test_metrics_likes_priority_over_top_level(self):
        """metrics.likes 优先于顶层 likes。"""
        it = _item("乐观", likes=9)          # metrics.likes = 9
        it["likes"] = 1                      # 顶层 likes = 1（应被忽略）
        r = sa.aggregate_distribution([it, _item("悲观", likes=0)])
        assert r["weighted_dist"]["乐观"] == pytest.approx(10 / 11 * 100, abs=0.1)

    def test_custom_weight_key(self):
        """weight_likes_key 可换键（如用 comments 加权）。"""
        items = [_item("乐观"), _item("悲观")]
        items[0]["metrics"] = {"comments": 19}
        r = sa.aggregate_distribution(items, weight_likes_key="comments")
        assert r["weighted_dist"]["乐观"] == pytest.approx(20 / 21 * 100, abs=0.1)


# ═══════════════════════════════════════════
# aggregate_distribution：置信度三档边界
# ═══════════════════════════════════════════


class TestConfidence:
    def test_n29_low(self):
        r = sa.aggregate_distribution([_item() for _ in range(29)])
        assert r["confidence"]["level"] == "低"
        assert "样本不足" in r["confidence"]["reason"]

    def test_n30_medium(self):
        r = sa.aggregate_distribution([_item() for _ in range(30)])
        assert r["confidence"]["level"] == "中"

    def test_n99_medium(self):
        r = sa.aggregate_distribution([_item() for _ in range(99)])
        assert r["confidence"]["level"] == "中"

    def test_n100_high(self):
        r = sa.aggregate_distribution([_item() for _ in range(100)])
        assert r["confidence"]["level"] == "高"


# ═══════════════════════════════════════════
# aggregate_distribution：空样本与防御
# ═══════════════════════════════════════════


class TestEmptySamples:
    def test_empty_list(self):
        assert sa.aggregate_distribution([]) == {"n": 0, "note": "样本为空"}

    def test_none_input(self):
        assert sa.aggregate_distribution(None) == {"n": 0, "note": "样本为空"}

    def test_all_non_dict_is_empty(self):
        assert sa.aggregate_distribution(["a", 1, None])["n"] == 0

    def test_weird_input_never_raises(self):
        """畸形入参（非标 dict、坏 metrics）绝不抛。"""
        r = sa.aggregate_distribution([{"sentiment": None, "metrics": "bad"}])
        assert r["n"] == 1
        assert r["unknown_count"] == 1
        assert r["dist"]["中性"]["count"] == 1


# ═══════════════════════════════════════════
# pick_representatives
# ═══════════════════════════════════════════


class TestRepresentatives:
    def test_per_bucket_top_by_likes_desc(self):
        """每桶按 likes 降序取前 per_bucket 条。"""
        items = [
            _item("乐观", likes=1, title="低赞"),
            _item("乐观", likes=99, title="高赞"),
            _item("乐观", likes=50, title="中赞"),
        ]
        r = sa.pick_representatives(items, per_bucket=2)
        assert [s["likes"] for s in r["乐观"]] == [99, 50]
        assert r["乐观"][0]["text"] == "高赞"

    def test_all_three_buckets_present(self):
        items = [_item("乐观", likes=5), _item("悲观", likes=3),
                 _item("中性", likes=1)]
        r = sa.pick_representatives(items)
        assert set(r.keys()) == {"乐观", "悲观", "中性", "无关"}
        assert len(r["乐观"]) == 1 and len(r["悲观"]) == 1
        assert len(r["中性"]) == 1
        assert r["无关"] == []

    def test_lexicon_labels_bucketed(self):
        """利好/利空归入乐观/悲观桶。"""
        r = sa.pick_representatives([_item("利好", likes=8),
                                     _item("利空", likes=6)])
        assert len(r["乐观"]) == 1 and r["乐观"][0]["likes"] == 8
        assert len(r["悲观"]) == 1 and r["悲观"][0]["likes"] == 6

    def test_slim_fields_and_truncation(self):
        """瘦身：仅 text/likes/platform/url；text 截 80 字。"""
        long_title = "长" * 200
        r = sa.pick_representatives([_item("乐观", likes=2, title=long_title)])
        s = r["乐观"][0]
        assert set(s.keys()) == {"text", "likes", "platform", "url"}
        assert len(s["text"]) == 80
        assert s["platform"] == "guba"
        assert s["url"] == "https://x"

    def test_content_fallback_when_title_empty(self):
        """title 为空时回退 content。"""
        r = sa.pick_representatives(
            [_item("中性", likes=1, title="", content="评论正文")])
        assert r["中性"][0]["text"] == "评论正文"

    def test_default_per_bucket_is_two(self):
        items = [_item("悲观", likes=i) for i in range(5)]
        r = sa.pick_representatives(items)
        assert len(r["悲观"]) == 2
        assert [s["likes"] for s in r["悲观"]] == [4, 3]

    def test_empty_items_empty_buckets(self):
        r = sa.pick_representatives([])
        assert r == {"乐观": [], "悲观": [], "中性": [], "无关": []}

    def test_unknown_label_goes_neutral(self):
        r = sa.pick_representatives([_item("看多", likes=4)])
        assert len(r["中性"]) == 1

    def test_likes_from_metrics_in_slim(self):
        """瘦身样本 likes 来自 metrics。"""
        r = sa.pick_representatives([_item("乐观", likes=42)])
        assert r["乐观"][0]["likes"] == 42

    def test_never_raises_on_junk(self):
        r = sa.pick_representatives([None, "x", {"sentiment": "乐观"}])
        assert len(r["乐观"]) == 1

    def test_irrelevant_bucket_populated(self):
        """「无关」条目进入无关桶，不挤占中性桶。"""
        r = sa.pick_representatives([_item("无关", likes=7),
                                     _item("中性", likes=2)])
        assert len(r["无关"]) == 1 and r["无关"][0]["likes"] == 7
        assert len(r["中性"]) == 1


# ═══════════════════════════════════════════
# 四桶分布与 bull_bear 多空比
# ═══════════════════════════════════════════


class TestFourBuckets:
    def test_dist_has_four_buckets(self):
        """dist/weighted_dist 均为四桶（乐观/悲观/中性/无关）。"""
        r = sa.aggregate_distribution([_item("乐观")])
        assert set(r["dist"]) == {"乐观", "悲观", "中性", "无关"}
        assert set(r["weighted_dist"]) == {"乐观", "悲观", "中性", "无关"}

    def test_irrelevant_counted_own_bucket(self):
        items = ([_item("乐观") for _ in range(2)]
                 + [_item("悲观")]
                 + [_item("中性")]
                 + [_item("无关") for _ in range(4)])
        r = sa.aggregate_distribution(items)
        assert r["n"] == 8
        assert r["dist"]["无关"]["count"] == 4
        assert r["dist"]["无关"]["pct"] == 50.0
        assert r["dist"]["乐观"]["pct"] == 25.0
        assert r["unknown_count"] == 0

    def test_irrelevant_weight_counted_in_weighted_dist(self):
        """无关桶权重照常统计（likes+1 基础票）。"""
        items = [_item("乐观", likes=0), _item("无关", likes=9)]
        r = sa.aggregate_distribution(items)
        # 权重 1 : 10 → 乐观 9.1 / 无关 90.9
        assert r["weighted_dist"]["无关"] == pytest.approx(10 / 11 * 100, abs=0.1)
        assert r["weighted_dist"]["乐观"] == pytest.approx(1 / 11 * 100, abs=0.1)

    def test_bucket_order(self):
        assert sa.BUCKETS == ("乐观", "悲观", "中性", "无关")


class TestBullBear:
    def test_basic_ratio_on_bull_bear_subset(self):
        """多空比仅在乐观+悲观子集上计算，两者之和为 100。"""
        items = ([_item("乐观") for _ in range(3)]
                 + [_item("悲观")]
                 + [_item("中性") for _ in range(2)]
                 + [_item("无关") for _ in range(4)])
        r = sa.aggregate_distribution(items)
        bb = r["bull_bear"]
        assert bb == {"乐观_pct": 75.0, "悲观_pct": 25.0}
        assert bb["乐观_pct"] + bb["悲观_pct"] == 100.0

    def test_neutral_and_irrelevant_excluded(self):
        """中性/无关条目不参与多空比分母。"""
        items = [_item("乐观"), _item("中性"), _item("无关")]
        r = sa.aggregate_distribution(items)
        assert r["bull_bear"] == {"乐观_pct": 100.0, "悲观_pct": 0.0}

    def test_all_neutral_edge_none_with_note(self):
        """全中性边界：乐观+悲观样本数为 0 → 双 None + note。"""
        r = sa.aggregate_distribution([_item("中性"), _item("无关")])
        bb = r["bull_bear"]
        assert bb["乐观_pct"] is None and bb["悲观_pct"] is None
        assert bb["note"] == "样本中无明确多空观点"

    def test_all_unknown_labels_also_none(self):
        """未知标签归中性后同样无多空观点。"""
        r = sa.aggregate_distribution([_item("???"), _item("看多")])
        assert r["bull_bear"]["乐观_pct"] is None
        assert r["unknown_count"] == 2

    def test_only_bear_side(self):
        r = sa.aggregate_distribution([_item("悲观"), _item("利空")])
        assert r["bull_bear"] == {"乐观_pct": 0.0, "悲观_pct": 100.0}


# ═══════════════════════════════════════════
# save_snapshot / get_trend
# ═══════════════════════════════════════════


def _snap(platform="guba", target="600519", date="2026-07-20", n=50,
          pos=60.0, neu=25.0, neg=15.0, w_pos=55.0, w_neu=30.0, w_neg=15.0):
    return {"platform": platform, "target": target, "date": date, "n": n,
            "pos": pos, "neu": neu, "neg": neg,
            "w_pos": w_pos, "w_neu": w_neu, "w_neg": w_neg}


class TestSnapshots:
    def test_save_and_read_roundtrip(self, db):
        assert sa.save_snapshot(_snap(), db_path=db) is True
        r = sa.get_trend("guba", "600519", db_path=db)
        assert len(r["series"]) == 1
        s = r["series"][0]
        assert s["date"] == "2026-07-20"
        assert s["n"] == 50
        assert s["dist"] == {"乐观": 60.0, "中性": 25.0, "悲观": 15.0}
        assert s["weighted"] == {"乐观": 55.0, "中性": 30.0, "悲观": 15.0}

    def test_idempotent_replace(self, db):
        """同主键重复写入 → REPLACE 覆盖，仅一行且取新值。"""
        assert sa.save_snapshot(_snap(pos=60.0), db_path=db) is True
        assert sa.save_snapshot(_snap(pos=75.0), db_path=db) is True
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT * FROM sentiment_snapshots").fetchall()
        assert len(rows) == 1
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["series"][0]["dist"]["乐观"] == 75.0

    def test_shares_social_db(self, db):
        """复用 social.db：同一文件内建 sentiment_snapshots 表。"""
        sa.save_snapshot(_snap(), db_path=db)
        with sqlite3.connect(db) as conn:
            name = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='sentiment_snapshots'").fetchone()
        assert name is not None

    def test_save_missing_keys_returns_false(self, db):
        """缺 platform/target/date → False，不抛。"""
        assert sa.save_snapshot({"platform": "guba"}, db_path=db) is False
        assert sa.save_snapshot({}, db_path=db) is False

    def test_save_accepts_aggregate_output_shape(self, db):
        """兼容 aggregate_distribution 输出形态（dist/weighted_dist 嵌套）。"""
        agg = {"n": 40,
               "dist": {"乐观": {"count": 28, "pct": 70.0},
                        "中性": {"count": 8, "pct": 20.0},
                        "悲观": {"count": 4, "pct": 10.0}},
               "weighted_dist": {"乐观": 80.0, "中性": 15.0, "悲观": 5.0}}
        snap = dict(agg, platform="bilibili", target="600519",
                    date="2026-07-21")
        assert sa.save_snapshot(snap, db_path=db) is True
        r = sa.get_trend("bilibili", "600519", db_path=db)
        s = r["series"][0]
        assert s["dist"]["乐观"] == 70.0
        assert s["weighted"]["乐观"] == 80.0

    def test_save_bad_path_returns_false(self, tmp_path):
        """坏路径（目录不可建）→ False，绝不抛。"""
        bad = str(tmp_path / "nonexist_parent_file")
        # 把一个已存在的文件当目录用，mkdir 必败
        f = tmp_path / "afile"
        f.write_text("x")
        assert sa.save_snapshot(_snap(), db_path=str(f / "sub" / "x.db")) is False

    def test_get_trend_missing_db(self, tmp_path):
        """库文件不存在 → 数据不足安全值，绝不抛。"""
        r = sa.get_trend("guba", "600519",
                         db_path=str(tmp_path / "none.db"))
        assert r == {"series": [], "direction": "数据不足",
                     "note": "快照库不存在，暂无历史数据"}

    def test_get_trend_empty_platform(self, db):
        r = sa.get_trend("", "600519", db_path=db)
        assert r["direction"] == "数据不足"
        assert r["series"] == []


class TestTrendDirection:
    def test_shift_to_positive(self, db):
        """乐观占比 +20pct → 转向乐观。"""
        sa.save_snapshot(_snap(date="2026-07-20", pos=50.0), db_path=db)
        sa.save_snapshot(_snap(date="2026-07-21", pos=70.0), db_path=db)
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["direction"] == "转向乐观"
        assert "+20.0pct" in r["note"]

    def test_shift_to_negative(self, db):
        """乐观占比 -25pct → 转向悲观。"""
        sa.save_snapshot(_snap(date="2026-07-20", pos=60.0), db_path=db)
        sa.save_snapshot(_snap(date="2026-07-21", pos=35.0), db_path=db)
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["direction"] == "转向悲观"

    def test_stable(self, db):
        """乐观占比 +5pct → 基本稳定。"""
        sa.save_snapshot(_snap(date="2026-07-20", pos=60.0), db_path=db)
        sa.save_snapshot(_snap(date="2026-07-21", pos=65.0), db_path=db)
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["direction"] == "基本稳定"

    def test_boundary_diff_exactly_15_is_stable(self, db):
        """差值恰好 15pct（未超阈值）→ 基本稳定。"""
        sa.save_snapshot(_snap(date="2026-07-20", pos=50.0), db_path=db)
        sa.save_snapshot(_snap(date="2026-07-21", pos=65.0), db_path=db)
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["direction"] == "基本稳定"

    def test_single_snapshot_insufficient(self, db):
        sa.save_snapshot(_snap(), db_path=db)
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["direction"] == "数据不足"
        assert len(r["series"]) == 1

    def test_zero_n_snapshot_not_valid(self, db):
        """n=0 的快照不计为有效快照。"""
        sa.save_snapshot(_snap(date="2026-07-20", n=0, pos=10.0), db_path=db)
        sa.save_snapshot(_snap(date="2026-07-21", pos=90.0), db_path=db)
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["direction"] == "数据不足"

    def test_uses_latest_two_snapshots(self, db):
        """三天数据：取最近两个有效快照判定（后两天 +20pct）。"""
        sa.save_snapshot(_snap(date="2026-07-19", pos=90.0), db_path=db)
        sa.save_snapshot(_snap(date="2026-07-20", pos=50.0), db_path=db)
        sa.save_snapshot(_snap(date="2026-07-21", pos=70.0), db_path=db)
        r = sa.get_trend("guba", "600519", db_path=db)
        assert r["direction"] == "转向乐观"
        assert [s["date"] for s in r["series"]] == [
            "2026-07-19", "2026-07-20", "2026-07-21"]

    def test_days_limit(self, db):
        """days 限制返回条数（取最近 N 天）。"""
        for i in range(5):
            sa.save_snapshot(_snap(date=f"2026-07-1{i}", pos=50.0 + i),
                             db_path=db)
        r = sa.get_trend("guba", "600519", days=3, db_path=db)
        assert len(r["series"]) == 3
        assert r["series"][0]["date"] == "2026-07-12"

    def test_target_isolation(self, db):
        """不同 target 快照互不影响。"""
        sa.save_snapshot(_snap(target="600519", date="2026-07-20",
                               pos=50.0), db_path=db)
        sa.save_snapshot(_snap(target="600519", date="2026-07-21",
                               pos=70.0), db_path=db)
        sa.save_snapshot(_snap(target="000001", date="2026-07-20",
                               pos=10.0), db_path=db)
        r = sa.get_trend("guba", "000001", db_path=db)
        assert r["direction"] == "数据不足"
        assert len(r["series"]) == 1


class TestPathResolution:
    def test_env_social_db_path(self, tmp_path, monkeypatch):
        """路径解析优先级：env SOCIAL_DB_PATH > 默认。"""
        env_db = str(tmp_path / "env_social.db")
        monkeypatch.setenv("SOCIAL_DB_PATH", env_db)
        assert sa.save_snapshot(_snap()) is True
        assert os.path.exists(env_db)
        r = sa.get_trend("guba", "600519")
        assert len(r["series"]) == 1

    def test_explicit_db_path_beats_env(self, tmp_path, monkeypatch):
        """db_path 参数优先级高于环境变量。"""
        env_db = str(tmp_path / "env.db")
        arg_db = str(tmp_path / "arg.db")
        monkeypatch.setenv("SOCIAL_DB_PATH", env_db)
        assert sa.save_snapshot(_snap(), db_path=arg_db) is True
        assert os.path.exists(arg_db)
        assert not os.path.exists(env_db)

    def test_default_data_dir_fallback(self, tmp_path, monkeypatch):
        """无参数无 SOCIAL_DB_PATH → ${DATA_DIR:-data}/social.db。"""
        monkeypatch.delenv("SOCIAL_DB_PATH", raising=False)
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "dd"))
        assert sa.save_snapshot(_snap()) is True
        assert os.path.exists(str(tmp_path / "dd" / "social.db"))
