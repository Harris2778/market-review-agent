"""agent/social_media.py 社媒舆情门面（微舆 BettaFish 式爬取层的统一入口）。

端点 2026-07-22 实测定案（详见 research/social_endpoints_recon.md）：
    微博仅热搜（ajax/side/hotSearch）、知乎仅热榜（topstory/hot-list）、
    B 站全通（square/popular/搜索/评论，搜索与评论需 buvid3 预热）、
    抖音仅热榜（hot/search/list/，脆弱红利需 newsnow 兜底）、
    小红书无可用无登录端点 v1 缺席、聚合器 newsnow.busiyi.world 四源兜底。

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
import re
import sys
from datetime import datetime
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
