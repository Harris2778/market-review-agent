"""
第五波 · 主动推送（定时复盘）任务逻辑。

设计原则：
1. 全部函数可独立测试、fail-safe——任何一步失败都降级处理，
   绝不向调用方抛出导致主服务受影响。
2. 对 agent.orchestrator / agent.charts 一律使用「函数体内
   try/except import」消费——这两个文件由其他工程师并行开发，
   模块顶层不引用任何项目内模块，避免 import 期冲突。
3. 时区一律显式使用 Asia/Shanghai（Railway 容器是 UTC，
   依赖容器本地时间会错判推送窗口）。

对外接口：
- should_fire(now, fire_time, last_fired_date)  纯函数，是否到达推送时机
- build_push_payload()                          组装推送内容（文字+图表URL）
- send_push(webhook_url, payload)               POST 到 webhook
- push_tick(...)                                单次检查（定时循环每轮调用）
"""

import logging
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# 推送窗口一律按上海时间判断（A 股交易日历时区）。
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

DEFAULT_FIRE_TIME = "15:40"  # 收盘后默认推送时间
PUSH_MESSAGE = "今日复盘"     # 触发非流式复盘的用户消息
SEND_TIMEOUT_SECONDS = 10.0   # webhook POST 超时


# ── 时间工具 ──

def now_in_shanghai() -> datetime:
    """当前上海时间（容器时区不可信，必须显式指定）。"""
    return datetime.now(SHANGHAI_TZ)


def _to_shanghai(dt: datetime) -> datetime:
    """任意 datetime → 上海时区。naive 视为已是上海本地时间。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SHANGHAI_TZ)
    return dt.astimezone(SHANGHAI_TZ)


def _parse_fire_time(fire_time) -> time:
    """'HH:MM' 字符串或 datetime.time → time。非法输入抛 ValueError。"""
    if isinstance(fire_time, time):
        return fire_time
    if isinstance(fire_time, str):
        parts = fire_time.strip().split(":")
        if len(parts) == 2:
            try:
                hour, minute = int(parts[0]), int(parts[1])
            except ValueError:
                pass
            else:
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return time(hour, minute)
    raise ValueError(f"非法的推送时间格式（应为 'HH:MM'）: {fire_time!r}")


def _latest_weekday(d: date) -> date:
    """周末回退到周五（与 orchestrator 的周末兜底规则一致，不触碰网络）。"""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ── 触发判断 ──

def should_fire(now_shanghai: datetime, fire_time, last_fired_date: Optional[date] = None) -> bool:
    """
    是否应触发推送（纯函数，无任何副作用）。

    三个条件同时满足才返回 True：
    1. 当天是工作日（周一~周五，按上海时间）；
    2. 当前上海时间已越过 fire_time（恰好等于也算越过）；
    3. 当天尚未触发过——调用方每触发一次就回存 last_fired_date，
       传入当天日期即视为已触发（配合每分钟醒一次的循环防重复）。

    fire_time 接受 'HH:MM' 字符串或 datetime.time；非法格式抛 ValueError
    （由调用方 catch 并记日志，不影响服务）。
    """
    now_sh = _to_shanghai(now_shanghai)
    ft = _parse_fire_time(fire_time)
    if now_sh.weekday() >= 5:
        return False
    if now_sh.time() < ft:
        return False
    if last_fired_date is not None and last_fired_date >= now_sh.date():
        return False
    return True


# ── 内容组装 ──

def _chart_path_to_url(path, chart_dir: str) -> Optional[str]:
    """
    图表文件路径 → /charts/<日期子目录>/<文件名> URL。

    优先按相对 chart_dir 计算（保留日期子目录层级，与 StaticFiles
    挂载目录一致）；路径不在 chart_dir 下时退化为 <父目录名>/<文件名>；
    实在无法解析返回 None（调用方跳过该项）。
    """
    try:
        p = Path(path)
    except TypeError:
        return None
    if not p.name:
        return None
    try:
        rel = p.absolute().relative_to(Path(chart_dir).absolute())
        return "/charts/" + rel.as_posix()
    except ValueError:
        pass
    if p.parent.name:
        return f"/charts/{p.parent.name}/{p.name}"
    return f"/charts/{p.name}"


async def build_push_payload() -> dict:
    """
    组装推送内容：{"text": ..., "charts": [...], "trade_date": ...}。

    fail-safe 降级策略（任何一步失败都不抛出）：
    - 文字复盘失败 → text 为空串；
    - agent.charts 不存在 / 快照采集失败 / 图表生成失败 → charts 为空列表，
      绝不阻挡文字推送；
    - trade_date 优先取快照日期，取不到按「上海今天 + 周末回退」兜底。
    """
    payload = {"text": "", "charts": [], "trade_date": ""}

    # ── 1. 文字：走编排层的非流式全市场复盘 ──
    try:
        from agent.orchestrator import get_agent
        agent = get_agent()
        result = await agent.process_message(PUSH_MESSAGE, stream=False)
        if isinstance(result, dict):
            payload["text"] = result.get("content", "") or ""
        else:
            logger.warning("推送：process_message 返回非 dict（%r），文字置空", type(result))
    except Exception:
        logger.error("推送：文字复盘生成失败（降级为空文字）", exc_info=True)

    # ── 2. 图表：可选能力，运行时 try/except import ──
    trade_date = ""
    charts_mod = None
    try:
        from agent import charts as charts_mod  # noqa: F811
    except Exception:
        logger.info("推送：agent.charts 不可用，本次跳过图表", exc_info=True)

    if charts_mod is not None:
        try:
            from agent.orchestrator import collect_market_snapshot
            snapshot = await collect_market_snapshot()
            trade_date = getattr(snapshot, "date", "") or ""

            chart_dir = os.getenv("CHART_DIR", "charts")
            paths = charts_mod.generate_daily_charts(snapshot) or []
            if isinstance(paths, dict):
                paths = list(paths.values())
            urls = []
            for p in paths:
                url = _chart_path_to_url(p, chart_dir)
                if url:
                    urls.append(url)
            payload["charts"] = urls
        except Exception:
            logger.warning("推送：图表生成失败（降级为纯文字）", exc_info=True)

    # ── 3. 交易日兜底 ──
    if not trade_date:
        trade_date = _latest_weekday(now_in_shanghai().date()).strftime("%Y%m%d")
    payload["trade_date"] = trade_date
    return payload


# ── 发送 ──

async def send_push(webhook_url: str, payload: dict) -> bool:
    """
    POST JSON 到 webhook（httpx，10s 超时）。

    2xx → True；超时/网络异常/非 2xx/URL 为空 → False 并记日志，
    绝不抛出。
    """
    if not webhook_url:
        logger.warning("推送：webhook_url 为空，放弃发送")
        return False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            resp = await client.post(webhook_url, json=payload)
        if 200 <= resp.status_code < 300:
            logger.info("推送发送成功：%s（HTTP %d）", webhook_url, resp.status_code)
            return True
        logger.warning("推送发送失败：%s 返回 HTTP %d", webhook_url, resp.status_code)
        return False
    except Exception as e:
        logger.warning("推送发送异常：%s（%s: %s）", webhook_url, type(e).__name__, e, exc_info=True)
        return False


# ── 单次检查（main.py 定时循环每轮调用）──

async def push_tick(
    fire_time=DEFAULT_FIRE_TIME,
    webhook_url: Optional[str] = None,
    last_fired_date: Optional[date] = None,
    now: Optional[datetime] = None,
) -> tuple:
    """
    单次推送检查：到点则生成 payload 并按配置发送/仅记录。

    返回 (fired, new_last_fired_date)：
    - fired=False 时 new_last_fired_date 原样返回；
    - fired=True 时 new_last_fired_date 为当天日期（调用方回存防重复）。

    webhook_url 为空时只生成 payload 记日志（方便以后接入），不发送。
    fire_time 非法时 should_fire 抛 ValueError，由调用方（定时循环）catch。
    """
    now = now or now_in_shanghai()
    if not should_fire(now, fire_time, last_fired_date):
        return False, last_fired_date

    payload = await build_push_payload()
    fired_date = _to_shanghai(now).date()

    if webhook_url:
        ok = await send_push(webhook_url, payload)
        logger.info(
            "定时推送：%s（trade_date=%s, text=%d字, charts=%d）",
            "发送成功" if ok else "发送失败",
            payload.get("trade_date", ""), len(payload.get("text", "")),
            len(payload.get("charts", [])),
        )
    else:
        logger.info(
            "PUSH_WEBHOOK_URL 未配置：仅生成推送内容不发送"
            "（trade_date=%s, text=%d字, charts=%d）",
            payload.get("trade_date", ""), len(payload.get("text", "")),
            len(payload.get("charts", [])),
        )
    return True, fired_date
