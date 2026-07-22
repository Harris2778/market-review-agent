#!/usr/bin/env python3
"""课程点评综合总结批量生成脚本（校园知识库 · thucourse_summary）。

流程：从 campus_kb 读取全部 thucourse_review 条目 → 按课程分组 →
逐课程调用 agent.review_summary.summarize_course_reviews 生成综合总结 →
build_summary_entry 构造 thucourse_summary 条目 → upsert 回 campus_kb。

CLI 用法：
    /usr/local/bin/python3 scripts/generate_course_summaries.py            # 全量 fallback
    /usr/local/bin/python3 scripts/generate_course_summaries.py --use-llm  # 走 LLM（见 _make_llm_fn TODO）
    /usr/local/bin/python3 scripts/generate_course_summaries.py --limit 20 --db data/campus_kb.db

纪律：
- campus_kb 惰性导入（_load_campus_kb），测试 monkeypatch 注入 fake，
  不形成对实体模块/真实 db 的硬依赖；
- --use-llm 默认关（纯 fallback 确定性摘要）；开启时经 _make_llm_fn()
  装配 llm_fn——LLM 真实接线由 Stage 3 编排方完成，此处仅留工厂位置；
- 逐课程限速 sleep 可注入（测试用 fake）；
- 单课程失败记 warning 并继续，核心函数绝不向调用方抛异常。
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 保证从项目根可导入 agent 包（脚本可被任意 cwd 调用）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import review_summary  # noqa: E402  本 Worker 自有模块，安全

# 读取 thucourse_review 条目的分页上限（search_kb limit）
_FETCH_LIMIT = 100000
# 默认逐课程限速间隔（秒）
DEFAULT_INTERVAL = 0.5

# 课程标识的 metadata 候选键（与 thucourse 爬虫 Worker 的 metadata 约定对齐，
# 取首个命中的非空值；均缺失时退回 source_id 解析，再退回 title）
_SQID_KEYS = ("course_sqid", "course_id", "sqid")
_TITLE_KEYS = ("course_title", "course_name", "course")


def _load_campus_kb():
    """惰性导入知识库存储层（仅 CLI 落库路径；测试 monkeypatch 本函数注入
    fake，不对 agent.campus_kb 实体形成硬依赖）。失败返回 None。"""
    try:
        from agent import campus_kb
        return campus_kb
    except Exception:
        logger.warning("agent.campus_kb 导入失败", exc_info=True)
        return None


def _make_llm_fn():
    """--use-llm 时装配 llm_fn(prompt: str) -> str 的工厂位置。

    TODO(Stage 3 编排方接线)：此处读取项目既有 LLM 配置（环境变量
    DEEPSEEK_API_KEY / AGENT_API_KEY 等，或复用 agent 层统一的 LLM client
    封装）构造真实 llm_fn。本阶段刻意不 import orchestrator / tools，
    避免与接线 Worker 产生文件冲突。

    返回 None 表示装配失败/未接线：调用方按「llm_fn=None」处理，
    自动走 fallback 确定性摘要，保证脚本在任何环境下可用。
    """
    logger.warning("--use-llm 已开启，但 llm_fn 工厂尚未接线（Stage 3 TODO），"
                   "本次运行降级为 fallback 确定性摘要")
    return None


def _parse_metadata(entry: dict) -> dict:
    """条目 metadata_json 容错解析；任何失败返回 {}。"""
    raw = entry.get("metadata_json")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def course_key(entry: dict) -> Tuple[str, str]:
    """从点评条目推导 (course_sqid, course_title) 分组键。

    优先 metadata_json 的 course_sqid/course_id/sqid；缺失时按
    source_id 形如 'thucourse:review:{sqid}[:...]' 解析；再缺失退回
    'title:{title}'（同标题视为同课程）。title 同理从 metadata 取，
    缺失退回条目 title 原值。绝不抛异常。"""
    meta = _parse_metadata(entry)
    sqid = ""
    for key in _SQID_KEYS:
        value = str(meta.get(key) or "").strip()
        if value:
            sqid = value
            break
    if not sqid:
        source_id = str(entry.get("source_id") or "")
        parts = source_id.split(":")
        # thucourse:review:{sqid}[:...] / thucourse:course:{sqid}
        if len(parts) >= 3 and parts[0] == "thucourse" and parts[2]:
            sqid = parts[2]
    title = ""
    for key in _TITLE_KEYS:
        value = str(meta.get(key) or "").strip()
        if value:
            title = value
            break
    if not title:
        title = str(entry.get("title") or "").strip()
    if not sqid:
        sqid = f"title:{title or 'unknown'}"
    return sqid, title


def group_reviews_by_course(reviews: List[dict]) -> Dict[str, dict]:
    """点评条目按课程分组：{sqid: {"title": str, "reviews": [...]}}。
    同课程取首个非空 title。返回 dict 键按 sqid 排序后的插入序（确定性）。"""
    groups: Dict[str, dict] = {}
    for entry in reviews:
        if not isinstance(entry, dict):
            continue
        sqid, title = course_key(entry)
        group = groups.setdefault(sqid, {"title": title, "reviews": []})
        if not group["title"] and title:
            group["title"] = title
        group["reviews"].append(entry)
    return {k: groups[k] for k in sorted(groups)}


def run(kb, *, db_path=None, use_llm: bool = False, llm_fn=None,
        limit: Optional[int] = None, interval: float = DEFAULT_INTERVAL,
        sleep=None) -> dict:
    """主流程：读取点评 → 分组 → 逐课程生成总结并 upsert。

    kb 为 campus_kb 模块（或测试注入的 fake，需提供 search_kb /
    upsert_entries）。返回统计 dict：{courses, reviews, upserted,
    llm_used, failed}。单课程失败记 warning 继续；绝不抛异常。"""
    sleep_fn = sleep or time.sleep
    stats = {"courses": 0, "reviews": 0, "upserted": 0,
             "llm_used": False, "failed": 0}

    effective_llm = llm_fn if (use_llm or llm_fn is not None) else None
    if use_llm and effective_llm is None:
        effective_llm = _make_llm_fn()
    stats["llm_used"] = effective_llm is not None

    try:
        list_fn = getattr(kb, "list_entries", None)
        if callable(list_fn):
            # 实体 campus_kb：批量列出（search_kb 空查询按契约返回 []，
            # 且 limit 会被夹取到 100，无法用于全量取数）
            reviews = list_fn("thucourse_review", limit=_FETCH_LIMIT,
                              db_path=db_path)
        else:
            # 测试注入 fake（无 list_entries）：约定 search_kb("", source=...)
            # 返回该来源全量条目
            reviews = kb.search_kb("", source="thucourse_review",
                                   limit=_FETCH_LIMIT, db_path=db_path)
    except Exception:
        logger.warning("读取 thucourse_review 条目失败", exc_info=True)
        return stats
    if not isinstance(reviews, list):
        logger.warning("点评读取返回非列表（%r），终止", type(reviews))
        return stats
    stats["reviews"] = len(reviews)

    groups = group_reviews_by_course(reviews)
    sqids = list(groups)
    if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0:
        sqids = sqids[:limit]

    for i, sqid in enumerate(sqids):
        if i > 0 and interval > 0:
            sleep_fn(interval)
        group = groups[sqid]
        title = group["title"]
        try:
            summary = review_summary.summarize_course_reviews(
                title, group["reviews"], llm_fn=effective_llm)
            entry = review_summary.build_summary_entry(sqid, title, summary)
            written = kb.upsert_entries([entry], db_path=db_path)
            if isinstance(written, int) and not isinstance(written, bool):
                stats["upserted"] += written
            stats["courses"] += 1
            logger.info("课程 %s（%s）：%d 条点评 → 总结已入库（method=%s）",
                        sqid, title, len(group["reviews"]), summary["method"])
        except Exception:
            stats["failed"] += 1
            logger.warning("课程 %s（%s）总结生成/入库失败，跳过",
                           sqid, title, exc_info=True)
    return stats


def main(argv=None, *, sleep=None) -> int:
    """CLI 入口。sleep 仅供测试注入；生产走 time.sleep。"""
    parser = argparse.ArgumentParser(
        description="课程点评综合总结批量生成：thucourse_review → thucourse_summary")
    parser.add_argument("--use-llm", action="store_true",
                        help="启用 LLM 摘要（默认关，纯 fallback；LLM 接线见 "
                             "_make_llm_fn 的 Stage 3 TODO）")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理课程数（调试用；缺省全量）")
    parser.add_argument("--db", dest="db_path", default=None,
                        help="SQLite 路径（缺省走 campus_kb 默认解析）")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="逐课程限速间隔秒数（默认 0.5）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="输出 DEBUG 日志")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    kb = _load_campus_kb()
    if kb is None:
        print("知识库 agent.campus_kb 不可用（详见日志），终止。")
        return 1
    try:
        kb.init_db(args.db_path)
    except Exception:
        logger.warning("知识库初始化失败", exc_info=True)
        print("知识库初始化失败（详见日志），终止。")
        return 1

    stats = run(kb, db_path=args.db_path, use_llm=args.use_llm,
                limit=args.limit, interval=args.interval, sleep=sleep)
    print(f"合计：读取点评 {stats['reviews']} 条，处理课程 {stats['courses']} 门，"
          f"入库总结 {stats['upserted']} 条，失败 {stats['failed']} 门，"
          f"LLM {'启用' if stats['llm_used'] else '未启用（fallback）'}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
