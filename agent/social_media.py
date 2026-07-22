"""agent/social_media.py 社媒舆情门面（微舆 BettaFish 式爬取层的统一入口）。

端点 2026-07-22 实测定案（详见 research/social_endpoints_recon.md）：
    微博仅热搜（ajax/side/hotSearch）、知乎仅热榜（topstory/hot-list）、
    B 站全通（square/popular/搜索/评论，搜索与评论需 buvid3 预热）、
    抖音仅热榜（hot/search/list/，脆弱红利需 newsnow 兜底）、
    小红书无可用无登录端点 v1 缺席、聚合器 newsnow.busiyi.world 四源兜底。
    东方财富股吧为个股舆情专用通道（get_guba_buzz，不做搜索/评论/热榜，
    不进 PLATFORM_MODULES），端点定案详见 research/guba_endpoints_recon.md。

合规注记：仅公开数据、低频访问（≤1 req/s + 抖动）、无登录自动化、
    不破解任何签名；平台条款/风控策略变动可能导致端点失效，
    运行期一律以 warning 日志观测降级，绝不抛出异常。

全局契约 —— 统一 Post 字典：
    {platform: str('weibo'/'zhihu'/'douyin'/'bilibili'),
     post_id: str, title: str, content: str(可空串), author: str(可空串),
     metrics: dict（只放实际拿到的键 likes/comments/shares/views/heat/favorites/coins）,
     url: str, published_at: str(ISO 或空串), source: str(端点标识如 'weibo_hot')}

并行施工隔离：本门面**绝不 import 兄弟平台模块的具体函数**，一律
    importlib 惰性导入模块对象 + getattr 探测能力；模块不存在 / 函数
    缺失即降级跳过并记 note。测试用 monkeypatch 把假模块塞进
    sys.modules（'social_weibo' 或 'agent.social_weibo' 均可识别）。
"""

import importlib
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from . import social_store
from .sentiment import score_news_sentiment

logger = logging.getLogger(__name__)

# 平台名 → 模块名（模块由兄弟 Worker 并行交付；缺失即降级）
PLATFORM_MODULES: Dict[str, str] = {
    "weibo": "social_weibo",
    "douyin": "social_douyin",
    "bilibili": "social_bilibili",
    "zhihu": "social_zhihu",
}
AGGREGATOR_MODULE = "social_aggregator"

# v1 明确缺席的平台及中文缺席原因（调用方问到时进 notes）
UNSUPPORTED_PLATFORMS: Dict[str, str] = {
    "xiaohongshu": (
        "小红书 v1 暂不支持：无可用无登录公开端点（搜索/详情 API 需 x-s 签名，"
        "超出合规边界；newsnow 聚合器亦无小红书源），v2 再评估接入。"
    ),
}

# A 股 6 位代码：沪 60/68、深 00/30/20、北交所/新三板 8/4 开头
_CODE_RE = re.compile(r"(?<![\d./])((?:60|68|00|30|20)\d{4}|[84]\d{5})(?!\d)")
# 价格/金额语境：代码后紧跟（可空白的）元/块/% 等量词时判为价格而非股票代码
_PRICE_CONTEXT_RE = re.compile(r"[\s　]*(元|块|%|万元|亿元|美元|人民币|港币)")

_XHS_KEY = "xiaohongshu"


# ── 惰性模块加载与能力探测 ──


def _load_module(module_name: str):
    """惰性导入模块对象；优先 sys.modules（测试注入），再 importlib。

    sys.modules 命中优先（裸名先于 'agent.' 前缀，便于测试注入覆盖
    真实模块的进程内缓存）；未命中时依次 import 'agent.<name>' 与
    '<name>'；全部失败返回 None。绝不抛出。
    """
    for fullname in (module_name, f"agent.{module_name}"):
        mod = sys.modules.get(fullname)
        if mod is not None:
            return mod
    for fullname in (f"agent.{module_name}", module_name):
        try:
            return importlib.import_module(fullname)
        except Exception:
            continue
    return None


def _get_capability(mod, func_name: str) -> Optional[Callable]:
    """getattr 探测模块能力；缺失或不可调用返回 None。"""
    if mod is None:
        return None
    fn = getattr(mod, func_name, None)
    return fn if callable(fn) else None


def _safe_fetch(fn: Callable, *args, **kwargs) -> List[dict]:
    """调用抓取函数并兜底：异常/非 list 返回 []；dict 条目透传。绝不抛。"""
    try:
        result = fn(*args, **kwargs)
    except TypeError:
        # 签名不兼容（如聚合器只收 platform 一个位置参数）时降级重试
        try:
            result = fn(*args)
        except Exception as e:
            logger.warning("社媒抓取调用失败（按空结果降级）: %s", e)
            return []
    except Exception as e:
        logger.warning("社媒抓取调用失败（按空结果降级）: %s", e)
        return []
    if not isinstance(result, list):
        return []
    return [p for p in result if isinstance(p, dict)]


def _normalize_platforms(platforms) -> List[str]:
    if platforms is None:
        return list(PLATFORM_MODULES.keys())
    out = []
    for p in platforms or []:
        key = str(p).strip().lower()
        if key and key not in out:
            out.append(key)
    return out


def _dedup_key(post: dict) -> tuple:
    platform = str(post.get("platform") or "").strip().lower()
    post_id = str(post.get("post_id") or "").strip()
    # post_id 缺失时退化为标题去重，避免把不同平台的无 ID 帖子全部合并
    return (platform, post_id or str(post.get("title") or ""))


def _merge_dedup(batches: List[List[dict]]) -> List[dict]:
    seen, merged = set(), []
    for batch in batches:
        for p in batch:
            key = _dedup_key(p)
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)
    return merged


def _check_unsupported(platform: str, notes: List[str]) -> bool:
    """平台在缺席名单时记录中文 note，返回 True 表示已处理（跳过）。"""
    if platform in UNSUPPORTED_PLATFORMS:
        notes.append(f"{platform}: {UNSUPPORTED_PLATFORMS[platform]}")
        return True
    return False


# ── 热榜聚合 ──


def get_hot_all(platforms=None, limit: int = 10,
                use_store: bool = True, sleep=None) -> dict:
    """逐平台拉热榜，直连空则自动降级聚合器兜底，合并去重后返回。绝不抛。

    返回 {date, platforms: {平台: 条数}, posts: [...], sources: {平台: 来源},
    notes: [中文说明...]}；sources 取值 'direct' / 'aggregator' / 'none'。
    use_store=True 时对合并结果 best-effort 落盘（social_store.upsert_posts）。
    """
    notes: List[str] = []
    counts: Dict[str, int] = {}
    sources: Dict[str, str] = {}
    batches: List[List[dict]] = []
    try:
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 10
        for plat in _normalize_platforms(platforms):
            if _check_unsupported(plat, notes):
                continue
            module_name = PLATFORM_MODULES.get(plat)
            if module_name is None:
                notes.append(f"{plat}: 未知平台，已跳过"
                             f"（支持：{'/'.join(PLATFORM_MODULES)}）")
                continue
            fetch = _get_capability(_load_module(module_name), "fetch_hot")
            posts: List[dict] = []
            source = "none"
            if fetch is None:
                notes.append(f"{plat}: 平台模块未就绪或缺少 fetch_hot，"
                             "直连跳过，尝试聚合器兜底")
            else:
                posts = _safe_fetch(fetch, limit=limit, sleep=sleep)
                if posts:
                    source = "direct"
            if not posts:
                agg_fetch = _get_capability(_load_module(AGGREGATOR_MODULE),
                                            "fetch_hot")
                if agg_fetch is not None:
                    agg_posts = _safe_fetch(agg_fetch, plat,
                                            limit=limit, sleep=sleep)
                    for p in agg_posts:
                        if not str(p.get("platform") or "").strip():
                            p["platform"] = plat
                    posts = agg_posts
                    if posts:
                        source = "aggregator"
            if not posts:
                if source == "none":
                    notes.append(f"{plat}: 直连与聚合器兜底均为空，本轮缺席")
            counts[plat] = len(posts)
            sources[plat] = source
            batches.append(posts)
        merged = _merge_dedup(batches)
        if use_store and merged:
            try:
                social_store.upsert_posts(merged)
            except Exception as e:
                logger.warning("舆情落盘失败（best-effort，忽略）: %s", e)
        return {
            "date": datetime.now().date().isoformat(),
            "platforms": counts,
            "posts": merged,
            "sources": sources,
            "notes": notes,
        }
    except Exception as e:
        logger.warning("get_hot_all 异常（返回空骨架）: %s", e, exc_info=True)
        return {
            "date": datetime.now().date().isoformat(),
            "platforms": counts,
            "posts": _merge_dedup(batches),
            "sources": sources,
            "notes": notes + [f"门面内部异常，结果可能不完整: {e}"],
        }


# ── 关键词搜索分发 ──


def search_all(keyword, platforms=None, limit: int = 10, sleep=None) -> dict:
    """只对提供 search 能力的平台分发关键词搜索。绝不抛。

    v1 实际仅 B 站提供 search（视频+专栏，需 buvid3 预热）；其余平台进
    notes 说明「该平台搜索暂不支持」。返回 {keyword, date, platforms,
    posts, sources, notes}。
    """
    notes: List[str] = []
    counts: Dict[str, int] = {}
    sources: Dict[str, str] = {}
    batches: List[List[dict]] = []
    kw = str(keyword or "").strip()
    try:
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 10
        if not kw:
            notes.append("搜索关键词为空，未执行任何搜索")
        else:
            for plat in _normalize_platforms(platforms):
                if _check_unsupported(plat, notes):
                    continue
                module_name = PLATFORM_MODULES.get(plat)
                if module_name is None:
                    notes.append(f"{plat}: 未知平台，已跳过"
                                 f"（支持：{'/'.join(PLATFORM_MODULES)}）")
                    continue
                search = _get_capability(_load_module(module_name), "search")
                if search is None:
                    notes.append(f"{plat}: 该平台搜索暂不支持，已跳过")
                    continue
                posts = _safe_fetch(search, kw, limit=limit, sleep=sleep)
                counts[plat] = len(posts)
                sources[plat] = "direct" if posts else "none"
                if not posts:
                    notes.append(f"{plat}: 搜索无结果或端点降级")
                batches.append(posts)
        return {
            "keyword": kw,
            "date": datetime.now().date().isoformat(),
            "platforms": counts,
            "posts": _merge_dedup(batches),
            "sources": sources,
            "notes": notes,
        }
    except Exception as e:
        logger.warning("search_all 异常（返回空骨架）: %s", e, exc_info=True)
        return {
            "keyword": kw,
            "date": datetime.now().date().isoformat(),
            "platforms": counts,
            "posts": _merge_dedup(batches),
            "sources": sources,
            "notes": notes + [f"门面内部异常，结果可能不完整: {e}"],
        }


# ── 股票关联提取 ──


def _load_watchlist() -> List[dict]:
    """惰性读 agent/watchlist.py 的 list_stocks；读不到返回 []。绝不抛。"""
    try:
        from . import watchlist as wl
    except Exception as e:
        logger.warning("watchlist 模块不可用（跳过自选股匹配）: %s", e)
        return []
    try:
        stocks = wl.list_stocks()
        return stocks if isinstance(stocks, list) else []
    except Exception as e:
        logger.warning("watchlist 读取失败（跳过自选股匹配）: %s", e)
        return []


def extract_stock_mentions(posts, watchlist=None, extra_names=None) -> dict:
    """从 title+content 提取股票关联，返回 {代码/名称: {count, sample_titles[:3]}}。

    三类匹配：
    1. 6 位数字代码正则（60/68/00/30/20/8/4 开头）；价格语境
       （如 "600519 元"、"300750块"、"涨幅 5%600123元" 尾缀量词）不计。
    2. watchlist（list[dict] 含 code/name，可注入；None 时惰性读
       agent/watchlist.py，读不到跳过）——代码与名称都归并到 6 位代码键。
    3. extra_names 注入名单（如东财人气榜 top 名称），按名称键统计。

    同一帖子对同一键只计一次；sample_titles 最多 3 条去重。绝不抛。
    """
    mentions: Dict[str, dict] = {}

    def bump(key: str, title: str) -> None:
        entry = mentions.setdefault(key, {"count": 0, "sample_titles": []})
        entry["count"] += 1
        t = str(title or "").strip()
        if t and len(entry["sample_titles"]) < 3 \
                and t not in entry["sample_titles"]:
            entry["sample_titles"].append(t)

    try:
        wl = watchlist if watchlist is not None else _load_watchlist()
        # (名称, 归并键)：watchlist 有代码时名称归并到代码键
        name_rules: List[tuple] = []
        for item in wl or []:
            if not isinstance(item, dict):
                continue
            code_digits = re.sub(r"\D", "", str(item.get("code") or ""))
            name = str(item.get("name") or "").strip()
            key = code_digits or name
            if name and key:
                name_rules.append((name, key))
        for n in extra_names or []:
            name = str(n).strip()
            if name:
                name_rules.append((name, name))
        # 长名优先，避免短名（如 "银行"）抢先吞掉长名帖子的语义归属
        name_rules.sort(key=lambda r: len(r[0]), reverse=True)

        for p in posts or []:
            if not isinstance(p, dict):
                continue
            title = str(p.get("title") or "")
            text = f"{title}\n{p.get('content') or ''}"
            hit_keys = set()
            for m in _CODE_RE.finditer(text):
                if _PRICE_CONTEXT_RE.match(text, m.end()):
                    continue  # 价格/金额语境，非股票代码
                hit_keys.add(m.group(1))
            for name, key in name_rules:
                if name in text:
                    hit_keys.add(key)
            for key in hit_keys:
                bump(key, title)
        return mentions
    except Exception as e:
        logger.warning("extract_stock_mentions 异常（返回已有部分结果）: %s",
                       e, exc_info=True)
        return mentions


# ── 舆情热度情感聚合 ──


def aggregate_buzz(posts, scorer: Optional[Callable] = None) -> dict:
    """复用 agent.sentiment.score_news_sentiment 对 Post 列表打情感分并聚合。

    Post 转成 score_news_sentiment 吃的 {title, content} 形态；scorer 可
    注入自定义打分函数（ scorer(item)->dict ），透传给底层。
    返回 {total, sentiment: {利好, 利空, 中性}, by_platform: {平台: {...}},
    avg_score}。绝不抛。
    """
    empty = {"total": 0,
             "sentiment": {"利好": 0, "利空": 0, "中性": 0},
             "by_platform": {},
             "avg_score": 0.0}
    try:
        valid = [p for p in posts or [] if isinstance(p, dict)]
        if not valid:
            return empty
        items = [{"title": str(p.get("title") or ""),
                  "content": str(p.get("content") or "")} for p in valid]
        scored = score_news_sentiment(items, scorer=scorer)
        dist = {"利好": 0, "利空": 0, "中性": 0}
        by_platform: Dict[str, dict] = {}
        score_sum = 0.0
        for post, s in zip(valid, scored):
            label = s.get("sentiment")
            if label in dist:
                dist[label] += 1
            try:
                score_sum += float(s.get("sentiment_score") or 0.0)
            except (TypeError, ValueError):
                pass
            plat = str(post.get("platform") or "unknown")
            bucket = by_platform.setdefault(
                plat, {"total": 0,
                       "sentiment": {"利好": 0, "利空": 0, "中性": 0}})
            bucket["total"] += 1
            if label in bucket["sentiment"]:
                bucket["sentiment"][label] += 1
        total = len(scored)
        return {
            "total": total,
            "sentiment": dist,
            "by_platform": by_platform,
            "avg_score": round(score_sum / total, 4) if total else 0.0,
        }
    except Exception as e:
        logger.warning("aggregate_buzz 异常（返回空骨架）: %s", e, exc_info=True)
        return empty


# ── 股吧个股舆情专用通道（东方财富股吧）──

GUBA_MODULE = "social_guba"
GUBA_SOURCE = "eastmoney_guba"
_GUBA_CONTENT_MAX = 200  # 帖子正文截断长度（防上下文爆炸）


def _normalize_guba_code(code) -> str:
    """股吧代码归一：提取数字部分，恰为 6 位 A 股代码才有效，否则返回空串。"""
    digits = re.sub(r"\D", "", str(code or ""))
    return digits if len(digits) == 6 else ""


def _slim_guba_post(post) -> Optional[dict]:
    """股吧帖子瘦身：platform/post_id/title/content截200/metrics/url/
    published_at/source；非 dict 输入返回 None（调用方过滤）。绝不抛。"""
    if not isinstance(post, dict):
        return None
    metrics = post.get("metrics")
    return {
        "platform": str(post.get("platform") or "guba"),
        "post_id": str(post.get("post_id") or ""),
        "title": str(post.get("title") or ""),
        "content": str(post.get("content") or "")[:_GUBA_CONTENT_MAX],
        "metrics": metrics if isinstance(metrics, dict) else {},
        "url": str(post.get("url") or ""),
        "published_at": str(post.get("published_at") or ""),
        "source": str(post.get("source") or ""),
    }


def get_guba_buzz(code, limit: int = 20, enrich: int = 3, sleep=None) -> dict:
    """东方财富股吧个股舆情：帖子列表 → 详情富化（正文+点赞）→ 情感聚合。绝不抛。

    定位：**个股舆情专用通道**。股吧端点 2026-07-22 实测定案（详见
    research/guba_endpoints_recon.md）只有个股吧帖子列表（gbapi
    Articlelist）与帖子详情（HTML SSR post_article，唯一点赞来源）两条
    免登录路径；关键词搜索/评论抓取/全站热榜全部判死，因此股吧**不进
    PLATFORM_MODULES**，不参与 get_hot_all/search_all 分发，只经本函数
    （及 get_stock_sentiment 工具的 guba 增强块）触达。

    合规注记：仅公开数据、低频访问（≤1 req/s + 抖动，由 social_guba
    内部限速保证）、无登录；模块缺席/能力缺失/抓取失败/列表为空一律
    warning + 中文说明进 notes 降级，绝不抛出异常。

    流程：惰性加载 social_guba（缺席即降级）→ fetch_bar_posts(code,
    limit=limit, sleep=sleep) → enrich_posts(posts, top_n=enrich,
    sleep=sleep) 回填 content 与 metrics.likes（enrich<=0 跳过富化）→
    复用 aggregate_buzz 对（富化后）帖子打情感分。

    返回 {code, posts(瘦身：platform/post_id/title/content截200/metrics/
    url/published_at/source), buzz, sources: ['eastmoney_guba'], notes}。
    """
    notes: List[str] = []
    empty_buzz = {"total": 0,
                  "sentiment": {"利好": 0, "利空": 0, "中性": 0},
                  "by_platform": {},
                  "avg_score": 0.0}

    def _out(code6: str, posts: List[dict], buzz: dict) -> dict:
        return {"code": code6, "posts": posts, "buzz": buzz,
                "sources": [GUBA_SOURCE], "notes": notes}

    try:
        code6 = _normalize_guba_code(code)
        if not code6:
            notes.append(f"股吧代码非法（需 6 位 A 股代码）：{code!r}，本轮未抓取")
            return _out("", [], empty_buzz)
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 20
        try:
            enrich = int(enrich)
        except (TypeError, ValueError):
            enrich = 3
        mod = _load_module(GUBA_MODULE)
        if mod is None:
            notes.append("股吧模块未就绪（agent/social_guba.py 缺席），本轮无股吧数据")
            return _out(code6, [], empty_buzz)
        fetch = _get_capability(mod, "fetch_bar_posts")
        if fetch is None:
            notes.append("股吧模块缺少 fetch_bar_posts 能力，本轮无股吧数据")
            return _out(code6, [], empty_buzz)
        posts = _safe_fetch(fetch, code6, limit=limit, sleep=sleep)
        if not posts:
            notes.append(f"股吧 {code6} 本轮未抓到帖子（端点降级或吧内无新帖）")
            return _out(code6, [], empty_buzz)
        if enrich <= 0:
            notes.append("enrich=0，跳过详情富化（帖子无正文与点赞数）")
        else:
            enrich_fn = _get_capability(mod, "enrich_posts")
            if enrich_fn is None:
                notes.append("股吧模块缺少 enrich_posts 能力，帖子无正文与点赞数")
            else:
                try:
                    enriched = enrich_fn(posts, top_n=enrich, sleep=sleep)
                except TypeError:
                    # 签名不兼容时降级为仅位置参数重试
                    try:
                        enriched = enrich_fn(posts)
                    except Exception as e:  # noqa: BLE001 - 富化失败降级
                        logger.warning("股吧详情富化失败（保留列表原始数据）: %s", e)
                        enriched = None
                        notes.append(f"股吧详情富化失败（{e}），帖子无正文与点赞数")
                except Exception as e:  # noqa: BLE001 - 富化失败降级
                    logger.warning("股吧详情富化失败（保留列表原始数据）: %s", e)
                    enriched = None
                    notes.append(f"股吧详情富化失败（{e}），帖子无正文与点赞数")
                if enriched is not None:
                    if isinstance(enriched, list) and enriched:
                        posts = [p for p in enriched if isinstance(p, dict)]
                    else:
                        notes.append("股吧详情富化返回为空，保留列表原始数据")
        if not posts:
            notes.append("股吧帖子经富化后为空，本轮无股吧数据")
            return _out(code6, [], empty_buzz)
        buzz = aggregate_buzz(posts)
        slim = [s for s in (_slim_guba_post(p) for p in posts) if s]
        return _out(code6, slim, buzz)
    except Exception as e:  # noqa: BLE001 - 门面绝不抛出
        logger.warning("get_guba_buzz 异常（返回空骨架）: %s", e, exc_info=True)
        notes.append(f"股吧舆情通道内部异常，结果可能不完整: {e}")
        return _out(_normalize_guba_code(code), [], empty_buzz)


# ── 情绪分布主入口（舆情「分布化」改造 · 并行接线层）──
#
# 定位：把「引用个别帖子/评论」的轶事式舆情升级为「整体情绪分布」为主体
# （样本量 n + 乐观/中性/悲观占比 + 置信度 + 趋势 + 代表性样本点缀）。
#
# 并行契约（兄弟 Worker 交付，一律惰性 importlib+getattr 探测，缺席降级进 notes）：
#   - sentiment_llm.score_texts_batch(texts, client=None, ...) ->
#       [{index, label(乐观/悲观/中性/无关), score, method:'llm'|'fallback'}]（契约绝不抛）
#   - sentiment_aggregate.aggregate_distribution(items) ->
#       {n, dist, weighted_dist, confidence, method}
#     / pick_representatives(items, per_bucket=2)
#     / save_snapshot(snapshot, db_path=None)
#     / get_trend(platform, target, days=7, db_path=None)
#   - 同文件采集扩容 Worker：collect_guba_samples(code, post_limit, enrich, sleep)
#     -> {code, posts, notes}；collect_keyword_samples(keyword, video_limit,
#     comments_per_video, sleep) -> {keyword, videos_used, comments, notes}
#
# 打分路径：use_llm 且 sentiment_llm 就绪 → LLM 批量打分并回填条目
#   sentiment/sentiment_score；否则降级 agent.sentiment.score_news_sentiment
#   词典打分（scorer 默认，标签 利好/利空/中性 由聚合层归并为 乐观/悲观/中性）。
#   method 标注 'llm' / 'lexicon' / 'mixed'（合并路径双平台打分方式不一致时）。

SENTIMENT_LLM_MODULE = "sentiment_llm"
SENTIMENT_AGG_MODULE = "sentiment_aggregate"
_DIST_BUCKETS = ("乐观", "悲观", "中性", "无关")
_DIST_TEXT_MAX = 100        # 打分输入文本（title+content）截断长度
_DIST_REPS_PER_BUCKET = 2   # pick_representatives 每桶代表样本数
_DIST_TREND_DAYS = 7        # 情绪趋势回看天数

# 采样深度档（全局契约）：standard 维持既有量级；deep 扩容
_DEPTH_STANDARD = "standard"
_DEPTH_DEEP = "deep"
_DEEP_POST_LIMIT = 300          # deep 档股吧帖子采样条数
_DEEP_COMMENT_LIMIT = 400       # deep 档关键词评论总量上限
_DEEP_VIDEO_LIMIT = 15          # deep 档搜索视频数（补偿匿名每视频仅约 3 条热评）
_DEEP_COMMENTS_PER_VIDEO = 50   # deep 档每视频评论数
_DEFAULT_SINCE_DAYS = 7         # 采样时间窗默认天数

# B 站匿名降级透明化说明（实测 2026-07-23：匿名每视频仅约 3 条热评且多为旧评）
_BILI_ANON_NOTE = ("B站匿名访问每视频仅返回约3条热评，"
                   "配置 BILI_SESSDATA 环境变量可拉取全量评论")


def _bili_sessdata_configured() -> bool:
    """BILI_SESSDATA 环境变量是否已配置。绝不抛。"""
    try:
        return bool((os.environ.get("BILI_SESSDATA") or "").strip())
    except Exception:
        return False


def _dist_text(item) -> str:
    """提取打分输入文本：title + content 拼接后截 _DIST_TEXT_MAX 字。绝不抛。"""
    if not isinstance(item, dict):
        return ""
    text = f"{item.get('title') or ''}\n{item.get('content') or ''}".strip()
    return text[:_DIST_TEXT_MAX]


def _dist_confidence_for(n: int) -> dict:
    """置信度分档（全局契约）：n<30 低（必须声明样本不足）；30≤n<100 中；n≥100 高。"""
    if n < 30:
        return {"level": "低", "reason": f"样本不足（n={n} < 30），分布仅供参考"}
    if n < 100:
        return {"level": "中", "reason": f"样本量中等（n={n}），分布可作参考"}
    return {"level": "高", "reason": f"样本量充足（n={n}），分布较稳健"}


def _empty_dist(n: int = 0) -> dict:
    """分布空骨架：对齐 aggregate_distribution 返回结构（四桶 + bull_bear）。"""
    return {
        "n": n,
        "dist": {b: {"count": 0, "pct": 0.0} for b in _DIST_BUCKETS},
        "weighted_dist": {b: 0.0 for b in _DIST_BUCKETS},
        "bull_bear": {"乐观_pct": None, "悲观_pct": None,
                      "note": "样本中无明确多空观点"},
        "confidence": _dist_confidence_for(n),
        "method": "none",
    }


# ── 采样时间窗工具（since_days 过滤 + window 输出）──


def _parse_sample_time(value):
    """解析样本时间为 aware datetime（UTC）：ISO 字符串 / unix 秒（数值或数字
    字符串）；naive ISO 按 UTC 处理；缺失/非法返回 None。绝不抛。"""
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"\d+(\.\d+)?", text):
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _filter_by_time_window(items, since_days, notes: List[str],
                           label: str) -> List[dict]:
    """按 published_at 过滤近 since_days 天样本（宁可多留不可误杀）。

    时间字段缺失/解析失败的条目保留并计数进 notes 说明；since_days 非法
    回落默认 7 天，<=0 不过滤。绝不抛。
    """
    try:
        days = int(since_days)
    except (TypeError, ValueError):
        days = _DEFAULT_SINCE_DAYS
    if days <= 0:
        return list(items or [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    kept: List[dict] = []
    missing = 0
    dropped = 0
    for it in items or []:
        t = _parse_sample_time(it.get("published_at"))
        if t is None:
            missing += 1
            kept.append(it)  # 时间缺失：保留，不误杀
        elif t >= cutoff:
            kept.append(it)
        else:
            dropped += 1
    if missing:
        notes.append(f"{label}：{missing} 条样本缺少可解析时间，"
                     "已保留未按时间窗过滤")
    if dropped:
        notes.append(f"{label}：按近 {days} 天时间窗过滤掉 {dropped} 条较早样本")
    return kept


def _sample_window(groups) -> dict:
    """从实际样本的 published_at 计算时间窗 {"from": 最早ISO, "to": 最晚ISO}；
    无任何可解析时间 → {"from": None, "to": None}。绝不抛。"""
    times = []
    for _platform, _target, items in groups or []:
        for it in items or []:
            if not isinstance(it, dict):
                continue
            dt = _parse_sample_time(it.get("published_at"))
            if dt is not None:
                times.append(dt)
    if not times:
        return {"from": None, "to": None}
    return {"from": min(times).isoformat(), "to": max(times).isoformat()}


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_dist_samples(code6: str, kw: str, post_limit: int,
                          video_limit: int, comments_per_video: int,
                          comment_total_cap, since_days, sleep,
                          notes: List[str]):
    """采集层惰性调用（采集扩容 Worker 同文件并行交付，globals 探测缺席即降级）。

    返回 [(platform, target, items)]：code 路径 ('guba', code, posts)，
    keyword 路径 ('bilibili', keyword, comments)。since_days 时间窗透传采集
    函数；comment_total_cap 非 None 时对关键词评论总量截断（deep 档用）。
    函数缺席/调用失败/返回结构异常/为空一律记中文 note 降级，绝不抛。
    """
    groups = []
    if code6:
        fn = globals().get("collect_guba_samples")
        if not callable(fn):
            notes.append("股吧采样函数 collect_guba_samples 未就绪"
                         "（采集层并行交付中），股吧样本缺席")
        else:
            res = None
            try:
                res = fn(code6, post_limit=post_limit, sleep=sleep,
                         since_days=since_days)
            except TypeError:
                # 签名不兼容时降级为仅位置参数重试
                try:
                    res = fn(code6)
                except Exception as e:  # noqa: BLE001 - 采集失败降级
                    logger.warning("股吧样本采集失败（降级缺席）: %s", e)
                    notes.append(f"股吧样本采集失败（{e}），股吧样本缺席")
            except Exception as e:  # noqa: BLE001 - 采集失败降级
                logger.warning("股吧样本采集失败（降级缺席）: %s", e)
                notes.append(f"股吧样本采集失败（{e}），股吧样本缺席")
            if isinstance(res, dict):
                for n in res.get("notes") or []:
                    notes.append(f"股吧：{n}")
                posts = [p for p in (res.get("posts") or []) if isinstance(p, dict)]
                if posts:
                    groups.append(("guba", code6, posts))
                else:
                    notes.append("股吧本轮无有效帖子样本")
            elif res is not None:
                notes.append("股吧采样返回结构异常，股吧样本缺席")
    if kw:
        fn = globals().get("collect_keyword_samples")
        if not callable(fn):
            notes.append("关键词采样函数 collect_keyword_samples 未就绪"
                         "（采集层并行交付中），关键词样本缺席")
        else:
            res = None
            try:
                res = fn(kw, video_limit=video_limit,
                         comments_per_video=comments_per_video, sleep=sleep,
                         since_days=since_days)
            except TypeError:
                try:
                    res = fn(kw)
                except Exception as e:  # noqa: BLE001 - 采集失败降级
                    logger.warning("关键词样本采集失败（降级缺席）: %s", e)
                    notes.append(f"关键词样本采集失败（{e}），关键词样本缺席")
            except Exception as e:  # noqa: BLE001 - 采集失败降级
                logger.warning("关键词样本采集失败（降级缺席）: %s", e)
                notes.append(f"关键词样本采集失败（{e}），关键词样本缺席")
            if isinstance(res, dict):
                for n in res.get("notes") or []:
                    notes.append(f"关键词：{n}")
                comments = [c for c in (res.get("comments") or [])
                            if isinstance(c, dict)]
                if comment_total_cap is not None and len(comments) > 0:
                    try:
                        cap = int(comment_total_cap)
                    except (TypeError, ValueError):
                        cap = 0
                    if cap > 0 and len(comments) > cap:
                        notes.append(f"关键词：评论样本超总量上限 {cap} 条，"
                                     f"已截断（原 {len(comments)} 条）")
                        comments = comments[:cap]
                if comments:
                    groups.append(("bilibili", kw, comments))
                else:
                    notes.append("关键词本轮无有效评论样本")
            elif res is not None:
                notes.append("关键词采样返回结构异常，关键词样本缺席")
    return groups


def _score_items_llm(items, client, sleep, platform: str, notes: List[str]) -> bool:
    """LLM 批量打分路径：texts 截 100 字 → score_texts_batch → 按 index 回填
    条目 sentiment（乐观/悲观/中性/无关）与 sentiment_score（夹取 [-1,1]）。

    返回 True=LLM 路径已用；模块缺席/能力缺失/异常/结构异常一律记 note
    并返回 False（调用方降级词典）。契约承诺绝不抛，仍全防御。
    """
    mod = _load_module(SENTIMENT_LLM_MODULE)
    if mod is None:
        notes.append(f"{platform}: LLM 打分模块 sentiment_llm 未就绪，降级词典打分")
        return False
    batch_fn = _get_capability(mod, "score_texts_batch")
    if batch_fn is None:
        notes.append(f"{platform}: LLM 打分能力 score_texts_batch 缺失，降级词典打分")
        return False
    texts = [_dist_text(it) for it in items]
    try:
        results = batch_fn(texts, client=client, sleep=sleep)
    except TypeError:
        try:
            results = batch_fn(texts)
        except Exception as e:  # noqa: BLE001 - LLM 失败降级词典
            logger.warning("%s LLM 批量打分失败（降级词典）: %s", platform, e)
            notes.append(f"{platform}: LLM 批量打分失败（{e}），降级词典打分")
            return False
    except Exception as e:  # noqa: BLE001 - LLM 失败降级词典
        logger.warning("%s LLM 批量打分失败（降级词典）: %s", platform, e)
        notes.append(f"{platform}: LLM 批量打分失败（{e}），降级词典打分")
        return False
    if not isinstance(results, list):
        notes.append(f"{platform}: LLM 批量打分返回结构异常，降级词典打分")
        return False
    by_index = {}
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("index"), int):
            by_index[r["index"]] = r
    fallback_hits = 0
    for i, item in enumerate(items):
        r = by_index.get(i)
        if not isinstance(r, dict):
            continue
        label = str(r.get("label") or "")
        if label in _DIST_BUCKETS:
            item["sentiment"] = label
        item["sentiment_score"] = round(
            max(-1.0, min(1.0, _safe_float(r.get("score")))), 4)
        if r.get("method") == "fallback":
            fallback_hits += 1
    if fallback_hits:
        notes.append(f"{platform}: LLM 打分中 {fallback_hits} 条内部降级为"
                     "兜底打分（method=fallback）")
    return True


def _score_items_lexicon(items, platform: str, notes: List[str]) -> bool:
    """词典打分路径：复用 score_news_sentiment（scorer 默认），把标签/分数回填
    到原条目上（保留 metrics 等其余字段）；标签 利好/利空/中性 由聚合层归并。
    失败记 note 返回 False，绝不抛。"""
    try:
        scored = score_news_sentiment(items)
    except Exception as e:  # noqa: BLE001 - 词典打分失败降级
        logger.warning("%s 词典打分失败（条目 sentiment 留空）: %s", platform, e)
        notes.append(f"{platform}: 词典打分异常（{e}），条目 sentiment 留空")
        return False
    for item, s in zip(items, scored):
        if not isinstance(s, dict):
            continue
        item["sentiment"] = s.get("sentiment")
        item["sentiment_score"] = s.get("sentiment_score", 0.0)
    return True


def _aggregate_group(items, platform: str, notes: List[str]):
    """单平台聚合：aggregate_distribution 算分布 + pick_representatives 取样本。

    返回 (agg, reps)；agg 对齐 aggregate_distribution 结构（失败给空骨架，
    n 兜底为条目数），reps 失败给 []。任何失败记 note，绝不抛。
    """
    mod = _load_module(SENTIMENT_AGG_MODULE)
    agg_fn = _get_capability(mod, "aggregate_distribution")
    agg = None
    if agg_fn is None:
        notes.append(f"{platform}: 聚合能力 aggregate_distribution 未就绪"
                     "（聚合层并行交付中），分布降级为空骨架")
    else:
        noted = False
        try:
            candidate = agg_fn(items)
        except Exception as e:  # noqa: BLE001 - 聚合失败降级空骨架
            logger.warning("%s 分布聚合异常（降级空骨架）: %s", platform, e)
            candidate = None
            noted = True
            notes.append(f"{platform}: 分布聚合异常（{e}），降级为空骨架")
        if isinstance(candidate, dict):
            agg = candidate
        elif not noted:
            notes.append(f"{platform}: 分布聚合返回结构异常或为空，降级为空骨架")
    if agg is None:
        agg = _empty_dist(n=len(items))
    rep_fn = _get_capability(mod, "pick_representatives")
    reps = []
    if rep_fn is None:
        notes.append(f"{platform}: 代表样本能力 pick_representatives 未就绪"
                     "（聚合层并行交付中），无代表性样本")
    else:
        try:
            cand = rep_fn(items, per_bucket=_DIST_REPS_PER_BUCKET)
        except TypeError:
            try:
                cand = rep_fn(items)
            except Exception as e:  # noqa: BLE001 - 代表样本失败降级
                logger.warning("%s 代表样本选取失败（降级空）: %s", platform, e)
                cand = None
                notes.append(f"{platform}: 代表样本选取失败（{e}），降级为空")
        except Exception as e:  # noqa: BLE001 - 代表样本失败降级
            logger.warning("%s 代表样本选取失败（降级空）: %s", platform, e)
            cand = None
            notes.append(f"{platform}: 代表样本选取失败（{e}），降级为空")
        if isinstance(cand, (list, dict)):
            reps = cand
        elif cand is not None:
            notes.append(f"{platform}: 代表样本返回结构异常，降级为空")
    return agg, reps


def _snapshot_and_trend(platform: str, target: str, agg: dict, notes: List[str]):
    """落快照 + 取趋势：save_snapshot 落本轮分布，get_trend 取近 _DIST_TREND_DAYS 天。

    快照/趋势失败只记 note，绝不影响本轮分布返回。返回 trend（失败为 None）。
    """
    mod = _load_module(SENTIMENT_AGG_MODULE)
    snapshot = {
        "platform": platform,
        "target": target,
        "date": datetime.now().date().isoformat(),
        "n": agg.get("n"),
        "dist": agg.get("dist"),
        "weighted_dist": agg.get("weighted_dist"),
        "confidence": agg.get("confidence"),
    }
    save_fn = _get_capability(mod, "save_snapshot")
    if save_fn is None:
        notes.append(f"{platform}: 快照能力 save_snapshot 未就绪，本轮未落快照")
    else:
        try:
            save_fn(snapshot)
        except Exception as e:  # noqa: BLE001 - 快照失败不影响分布
            logger.warning("%s 快照落盘失败（不影响本轮分布）: %s", platform, e)
            notes.append(f"{platform}: 快照落盘失败（{e}），不影响本轮分布")
    trend_fn = _get_capability(mod, "get_trend")
    trend = None
    if trend_fn is None:
        notes.append(f"{platform}: 趋势能力 get_trend 未就绪，本轮无趋势数据")
    else:
        try:
            trend = trend_fn(platform, target, days=_DIST_TREND_DAYS)
        except TypeError:
            try:
                trend = trend_fn(platform, target)
            except Exception as e:  # noqa: BLE001 - 趋势失败降级
                logger.warning("%s 趋势读取失败（本轮无趋势数据）: %s", platform, e)
                notes.append(f"{platform}: 趋势读取失败（{e}），本轮无趋势数据")
        except Exception as e:  # noqa: BLE001 - 趋势失败降级
            logger.warning("%s 趋势读取失败（本轮无趋势数据）: %s", platform, e)
            notes.append(f"{platform}: 趋势读取失败（{e}），本轮无趋势数据")
    return trend


def _merge_group_dists(aggs) -> dict:
    """合并多平台分布（平台分别统计后归并）：count 求和重算 pct，
    weighted_dist 按各平台 n 加权平均，confidence 按合并 n 重新分档。绝不抛。"""
    n = 0
    counts = {b: 0 for b in _DIST_BUCKETS}
    weighted_sum = {b: 0.0 for b in _DIST_BUCKETS}
    for agg in aggs or []:
        if not isinstance(agg, dict):
            continue
        n_i = _safe_int(agg.get("n"))
        n += n_i
        dist = agg.get("dist") if isinstance(agg.get("dist"), dict) else {}
        for b in _DIST_BUCKETS:
            bucket = dist.get(b) if isinstance(dist.get(b), dict) else {}
            counts[b] += _safe_int(bucket.get("count"))
        w = agg.get("weighted_dist") if isinstance(agg.get("weighted_dist"), dict) else {}
        for b in _DIST_BUCKETS:
            weighted_sum[b] += _safe_float(w.get(b)) * n_i
    if n <= 0:
        n = sum(counts.values())  # aggregate 未提供 n 时用 count 合计兜底
    dist = {b: {"count": counts[b],
                "pct": round(counts[b] / n * 100.0, 1) if n else 0.0}
            for b in _DIST_BUCKETS}
    weighted = {b: round(weighted_sum[b] / n, 1) if n else 0.0
                for b in _DIST_BUCKETS}
    pos_neg = counts["乐观"] + counts["悲观"]
    if pos_neg > 0:
        bull_bear = {"乐观_pct": round(counts["乐观"] / pos_neg * 100.0, 1),
                     "悲观_pct": round(counts["悲观"] / pos_neg * 100.0, 1)}
    else:
        bull_bear = {"乐观_pct": None, "悲观_pct": None,
                     "note": "样本中无明确多空观点"}
    return {"n": n, "dist": dist, "weighted_dist": weighted,
            "bull_bear": bull_bear,
            "confidence": _dist_confidence_for(n)}


def get_sentiment_distribution(code=None, keyword=None, post_limit: int = 80,
                               comment_limit: int = 120, use_llm: bool = True,
                               client=None, sleep=None, depth: str = "standard",
                               since_days: int = 7) -> dict:
    """个股/关键词社媒情绪分布主入口：采样 → 打分 → 聚合分布 → 快照/趋势。绝不抛。

    路径：
    - code 路径：collect_guba_samples(code, post_limit) 取股吧帖子；
    - keyword 路径：collect_keyword_samples(keyword) 取 B 站评论；
    - 两者都给则两路都采集，平台分别统计后归并（dist 计数求和重算 pct，
      weighted_dist 按平台 n 加权，confidence 按合并 n 重新分档）。

    深度档 depth：standard（缺省）维持既有量级（post_limit/comment_limit
    参数直传，视频 5 个）；deep 扩容为 post_limit=300 / comment_limit=400
    （评论总量上限）/ video_limit=8 / comments_per_video=50。未知档按
    standard 处理并进 notes。
    时间窗 since_days（默认 7）：透传采集层按样本 published_at 过滤近
    since_days 天样本（时间缺失条目保留并记 notes）。

    打分：use_llm 且 sentiment_llm 就绪 → score_texts_batch 批量打分
    （标签回填条目 sentiment/sentiment_score）；否则降级
    sentiment.score_news_sentiment 词典打分（scorer 默认）。method 标注
    'llm' / 'lexicon' / 'mixed'（合并路径双平台打分方式不一致时）。

    返回 {target:{code/keyword}, samples_total, dist（四桶：乐观/悲观/中性/
    无关）, weighted_dist, bull_bear（多空比，仅乐观+悲观子集相对占比）,
    window（{"from"/"to"} 样本实际时间跨度，无时间信息则双 None 并记 notes）,
    confidence, trend, representatives, method, sources, notes}；
    合并路径 trend/representatives 为 {平台: 值} 字典。任何子步骤失败
    降级进 notes，绝不抛异常。
    """
    notes: List[str] = []

    def _out(target, samples_total, dist, weighted_dist, confidence, trend,
             representatives, method, sources, bull_bear=None, window=None):
        return {
            "target": target,
            "samples_total": samples_total,
            "dist": dist,
            "weighted_dist": weighted_dist,
            "bull_bear": bull_bear,
            "window": window if window is not None
                      else {"from": None, "to": None},
            "confidence": confidence,
            "trend": trend,
            "representatives": representatives,
            "method": method,
            "sources": sources,
            "notes": notes,
        }

    try:
        code6 = _normalize_guba_code(code) if code else ""
        kw = str(keyword or "").strip()
        if code and not code6:
            notes.append(f"股吧代码非法（需 6 位 A 股代码）：{code!r}，股吧路径跳过")
        target = {}
        if code6:
            target["code"] = code6
        if kw:
            target["keyword"] = kw
        if not code6 and not kw:
            notes.append("code 与 keyword 均缺失/非法，本轮无样本可打分")
            empty = _empty_dist(0)
            return _out(target, 0, empty["dist"], empty["weighted_dist"],
                        empty["confidence"], None, [], "none", [],
                        bull_bear=empty["bull_bear"])
        # 深度档：standard 维持既有量级；deep 扩容固定配额
        depth_norm = str(depth or _DEPTH_STANDARD).strip().lower()
        if depth_norm not in (_DEPTH_STANDARD, _DEPTH_DEEP):
            notes.append(f"未知深度档 depth={depth!r}，已按 standard 档处理")
            depth_norm = _DEPTH_STANDARD
        try:
            post_limit = max(1, int(post_limit))
        except (TypeError, ValueError):
            post_limit = 80
        try:
            comment_limit = max(1, int(comment_limit))
        except (TypeError, ValueError):
            comment_limit = 120
        if depth_norm == _DEPTH_DEEP:
            post_limit = _DEEP_POST_LIMIT
            comment_total_cap = _DEEP_COMMENT_LIMIT
            video_limit = _DEEP_VIDEO_LIMIT
            comments_per_video = _DEEP_COMMENTS_PER_VIDEO
        else:
            comment_total_cap = None
            video_limit = 5
            comments_per_video = comment_limit
        try:
            since_days = int(since_days)
        except (TypeError, ValueError):
            since_days = _DEFAULT_SINCE_DAYS

        groups = _collect_dist_samples(code6, kw, post_limit, video_limit,
                                       comments_per_video, comment_total_cap,
                                       since_days, sleep, notes)
        if not groups:
            notes.append("本轮无任何有效样本（采集层缺席或为空）")
            empty = _empty_dist(0)
            return _out(target, 0, empty["dist"], empty["weighted_dist"],
                        empty["confidence"], None, [], "none", [],
                        bull_bear=empty["bull_bear"])

        window = _sample_window(groups)
        if window["from"] is None and window["to"] is None:
            notes.append("样本缺少可解析时间，无法计算 window 时间窗")

        per_group = []
        methods = []
        for platform, target_value, items in groups:
            used_llm = _score_items_llm(items, client, sleep, platform, notes) \
                if use_llm else False
            if not used_llm:
                _score_items_lexicon(items, platform, notes)
            methods.append("llm" if used_llm else "lexicon")
            agg, reps = _aggregate_group(items, platform, notes)
            trend = _snapshot_and_trend(platform, target_value, agg, notes)
            per_group.append({
                "platform": platform, "target": target_value, "items": items,
                "agg": agg, "reps": reps, "trend": trend,
            })

        overall_method = methods[0] if len(set(methods)) == 1 else "mixed"
        sources = ["eastmoney_guba" if g["platform"] == "guba" else g["platform"]
                   for g in per_group]

        if len(per_group) == 1:
            g = per_group[0]
            agg = g["agg"]
            empty = _empty_dist(len(g["items"]))
            bull_bear = (agg.get("bull_bear")
                         if isinstance(agg.get("bull_bear"), dict)
                         else empty["bull_bear"])
            return _out(
                target,
                _safe_int(agg.get("n"), len(g["items"])),
                agg.get("dist") if isinstance(agg.get("dist"), dict)
                else empty["dist"],
                agg.get("weighted_dist") if isinstance(agg.get("weighted_dist"), dict)
                else empty["weighted_dist"],
                agg.get("confidence") if isinstance(agg.get("confidence"), dict)
                else empty["confidence"],
                g["trend"], g["reps"], overall_method, sources,
                bull_bear=bull_bear, window=window)
        # 合并路径：平台分别统计后归并
        merged = _merge_group_dists([g["agg"] for g in per_group])
        trends = {g["platform"]: g["trend"] for g in per_group}
        reps = {g["platform"]: g["reps"] for g in per_group}
        return _out(target, merged["n"], merged["dist"], merged["weighted_dist"],
                    merged["confidence"], trends, reps, overall_method, sources,
                    bull_bear=merged["bull_bear"], window=window)
    except Exception as e:  # noqa: BLE001 - 门面绝不抛出
        logger.warning("get_sentiment_distribution 异常（返回空骨架）: %s", e,
                       exc_info=True)
        notes.append(f"情绪分布主流程内部异常，结果为空骨架: {e}")
        empty = _empty_dist(0)
        return _out({}, 0, empty["dist"], empty["weighted_dist"],
                    empty["confidence"], None, [], "none", [],
                    bull_bear=empty["bull_bear"])


# ── 采样扩容：关键词评论样本 + 股吧帖子样本（分布化舆情数据源）──
#
# 面向「整体情绪分布」改造：下游聚合 Worker 基于这两个函数拿到数百条
# 样本后做分布统计，单条引用降级为代表性点缀。契约：
#   collect_keyword_samples → {keyword, videos_used, comments, notes}
#   collect_guba_samples    → {code, posts, notes}
# 两者均绝不抛异常，任何降级进 notes。


def collect_keyword_samples(keyword, video_limit: int = 5,
                            comments_per_video: int = 30, sleep=None,
                            since_days: int = 7) -> dict:
    """B 站关键词评论样本扩容采集：搜索前 N 个视频 → 逐视频拉评论 → 合并。绝不抛。

    流程：惰性加载 B 站模块（search 能力缺失即降级）→ search(keyword,
    limit=video_limit, order="pubdate")（按发布时间倒序采样）→ 逐视频
    fetch_comments(post_id, limit=comments_per_video) → 合并为评论样本列表
    → since_days 时间窗过滤（默认近 7 天；时间缺失/解析失败的条目保留并
    进 notes 说明，宁可多留不可误杀）。

    评论条目为统一 Comment 契约 {platform, post_id, author, content,
    likes, published_at} 另加 source_video 键（源视频标题，供代表性
    引用与追溯）。部分视频评论失败/为空跳过并进 notes，绝不抛。
    匿名降级透明化：环境无 BILI_SESSDATA 时 notes 追加匿名热评受限说明。

    返回 {keyword, videos_used: [{post_id, title, comments}], comments,
    notes}；videos_used 只含实际采到评论的视频。
    """
    notes: List[str] = []
    kw = str(keyword or "").strip()
    videos_used: List[dict] = []
    comments: List[dict] = []

    def _out() -> dict:
        if not _bili_sessdata_configured() and _BILI_ANON_NOTE not in notes:
            notes.append(_BILI_ANON_NOTE)
        return {"keyword": kw, "videos_used": videos_used,
                "comments": comments, "notes": notes}

    try:
        try:
            video_limit = max(1, int(video_limit))
        except (TypeError, ValueError):
            video_limit = 5
        try:
            comments_per_video = max(1, int(comments_per_video))
        except (TypeError, ValueError):
            comments_per_video = 30
        if not kw:
            notes.append("搜索关键词为空，未执行评论采样")
            return _out()
        mod = _load_module(PLATFORM_MODULES["bilibili"])
        search = _get_capability(mod, "search")
        if search is None:
            notes.append("B 站模块未就绪或缺少 search 能力，本轮无评论样本")
            return _out()
        videos = _safe_fetch(search, kw, limit=video_limit,
                             sleep=sleep, order="pubdate")[:video_limit]
        if not videos:
            notes.append(f"关键词「{kw}」搜索无视频结果或端点降级，"
                         "本轮无评论样本")
            return _out()
        fetch_comments = _get_capability(mod, "fetch_comments")
        if fetch_comments is None:
            notes.append("B 站模块缺少 fetch_comments 能力，本轮无评论样本")
            return _out()
        for v in videos:
            pid = str(v.get("post_id") or "").strip()
            title = str(v.get("title") or "").strip()
            if not pid:
                notes.append(f"视频「{title or '?'}」缺少 post_id，"
                             "评论采样跳过")
                continue
            batch = _safe_fetch(fetch_comments, pid,
                                limit=comments_per_video, sleep=sleep)
            if not batch:
                notes.append(f"视频「{title or pid}」评论抓取失败或为空，"
                             "已跳过")
                continue
            for c in batch:
                c["source_video"] = title
            comments.extend(batch)
            videos_used.append({"post_id": pid, "title": title,
                                "comments": len(batch)})
        if not comments:
            notes.append("全部视频评论采样失败，本轮无评论样本")
        else:
            comments[:] = _filter_by_time_window(comments, since_days,
                                                 notes, "关键词")
        return _out()
    except Exception as e:  # noqa: BLE001 - 门面绝不抛出
        logger.warning("collect_keyword_samples 异常（返回已有部分结果）: %s",
                       e, exc_info=True)
        notes.append(f"关键词评论采样内部异常，结果可能不完整: {e}")
        return _out()


def collect_guba_samples(code, post_limit: int = 100, enrich: int = 0,
                         sleep=None, since_days: int = 7) -> dict:
    """股吧帖子样本扩容采集：fetch_bar_posts 分页拉帖，可选 enrich 回填。绝不抛。

    流程：代码归一（6 位 A 股）→ 惰性加载 social_guba →
    fetch_bar_posts(code, limit=post_limit) → enrich>0 时
    enrich_posts(posts, top_n=enrich) 回填 top 正文与点赞 →
    since_days 时间窗过滤（默认近 7 天；时间缺失/解析失败的条目保留并
    进 notes 说明，宁可多留不可误杀）。

    返回 {code, posts, notes}；posts 为统一 Post 契约原样透传（不瘦身、
    不打分，交由下游聚合 Worker 处理）。模块缺席/能力缺失/抓取失败/
    富化失败一律降级进 notes，绝不抛异常。
    """
    notes: List[str] = []
    posts: List[dict] = []

    def _out(code6: str) -> dict:
        return {"code": code6, "posts": posts, "notes": notes}

    try:
        code6 = _normalize_guba_code(code)
        if not code6:
            notes.append(f"股吧代码非法（需 6 位 A 股代码）：{code!r}，"
                         "本轮未采样")
            return _out("")
        try:
            post_limit = max(1, int(post_limit))
        except (TypeError, ValueError):
            post_limit = 100
        try:
            enrich = max(0, int(enrich))
        except (TypeError, ValueError):
            enrich = 0
        mod = _load_module(GUBA_MODULE)
        if mod is None:
            notes.append("股吧模块未就绪（agent/social_guba.py 缺席），"
                         "本轮无帖子样本")
            return _out(code6)
        fetch = _get_capability(mod, "fetch_bar_posts")
        if fetch is None:
            notes.append("股吧模块缺少 fetch_bar_posts 能力，本轮无帖子样本")
            return _out(code6)
        posts.extend(_safe_fetch(fetch, code6, limit=post_limit, sleep=sleep))
        if not posts:
            notes.append(f"股吧 {code6} 本轮未抓到帖子"
                         "（端点降级或吧内无新帖）")
            return _out(code6)
        if enrich > 0:
            enrich_fn = _get_capability(mod, "enrich_posts")
            if enrich_fn is None:
                notes.append("股吧模块缺少 enrich_posts 能力，"
                             "帖子无正文与点赞数")
            else:
                try:
                    enriched = enrich_fn(list(posts), top_n=enrich,
                                         sleep=sleep)
                except TypeError:
                    # 签名不兼容时降级为仅位置参数重试
                    try:
                        enriched = enrich_fn(list(posts))
                    except Exception as e:  # noqa: BLE001 - 富化失败降级
                        logger.warning("股吧详情富化失败（保留列表原始数据）: %s", e)
                        enriched = None
                        notes.append(f"股吧详情富化失败（{e}），"
                                     "帖子无正文与点赞数")
                except Exception as e:  # noqa: BLE001 - 富化失败降级
                    logger.warning("股吧详情富化失败（保留列表原始数据）: %s", e)
                    enriched = None
                    notes.append(f"股吧详情富化失败（{e}），帖子无正文与点赞数")
                if enriched is not None:
                    if isinstance(enriched, list) and enriched:
                        posts[:] = [p for p in enriched if isinstance(p, dict)]
                    else:
                        notes.append("股吧详情富化返回为空，保留列表原始数据")
        posts[:] = _filter_by_time_window(posts, since_days, notes, "股吧")
        return _out(code6)
    except Exception as e:  # noqa: BLE001 - 门面绝不抛出
        logger.warning("collect_guba_samples 异常（返回已有部分结果）: %s",
                       e, exc_info=True)
        notes.append(f"股吧帖子采样内部异常，结果可能不完整: {e}")
        return _out(_normalize_guba_code(code))
