"""
市场复盘智能体 — FastAPI 服务端。

提供 OpenAI 兼容的 API 接口，接入清小搭平台。

启动方式：
  python main.py
  或
  uvicorn main:app --host 0.0.0.0 --port 8000

环境变量（必填）：
  DEEPSEEK_API_KEY    DeepSeek API Key
  AGENT_API_KEY        智能体调用密钥（平台 → 你的服务）

环境变量（推荐）：
  TUSHARE_TOKEN       Tushare Pro Token
  FINNHUB_API_KEY     Finnhub API Key
  FRED_API_KEY        FRED API Key
  BRAVE_SEARCH_API_KEY Brave Search API Key

环境变量（可选 · 定时推送与图表静态服务）：
  PUSH_WEBHOOK_URL    定时推送的目标 Webhook URL（未配置则只生成内容记日志，不发送）
  PUSH_TIME           定时推送触发时间，上海时区 HH:MM（默认 15:40，仅工作日触发）
  CHART_DIR           图表文件目录，启动时自动创建并挂载到 /charts（默认 ${DATA_DIR:-data}/charts）

环境变量（可选 · 限流与每日配额）：
  RATE_LIMIT_PER_MIN  每分钟限流（默认 30，仅统计鉴权通过的 /v1/chat/completions 请求）
  QUOTA_DAILY         每日配额（默认 500，按 Asia/Shanghai 自然日重置）

环境变量（可选 · 舆情快照同步）：
  SOCIAL_DB_PATH      舆情快照库路径（默认 ${DATA_DIR:-data}/social.db，
                      /v1/admin/sentiment/snapshots 的落库位置）
  SNAPSHOT_BATCH_MAX  单次快照推送条数上限（默认 500，防爆）

环境变量（可选 · 公开网站 /api/*）：
  WEB_DB_PATH         网站数据库路径（默认 ${DATA_DIR:-data}/webapp.db）
  WEB_MONTHLY_QUOTA   每用户每月定额消息数（默认 100，每月 1 日重置）
  WEB_PACK_SIZE       加油包每包消息数（默认 50，不过期可累积）
"""

import os
import json
import math
import time
import uuid
import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import traceback
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from agent.orchestrator import get_agent, detect_intent
    _agent_loaded = True
    _agent_error = None
except Exception as e:
    _agent_loaded = False
    _agent_error = traceback.format_exc()
    get_agent = None
    detect_intent = None

# 推送模块（agent.push）同为并行开发文件：import 失败只降级，不影响主服务
try:
    from agent import push as push_tasks
    _push_loaded = True
except Exception:
    push_tasks = None
    _push_loaded = False
    logger.error("推送模块 agent.push 加载失败，定时推送将不可用", exc_info=True)

# 舆情快照模块（agent.sentiment_aggregate）：import 失败只降级，
# 快照接收端点照常可用（所有条目记入 failed），不影响主服务
try:
    from agent.sentiment_aggregate import save_snapshot as _save_snapshot
except Exception:
    _save_snapshot = None
    logger.error(
        "舆情快照模块 agent.sentiment_aggregate 加载失败，快照接收端点将全量记 failed",
        exc_info=True,
    )

# 公开网站认证/额度/对话模块（agent.webauth）：纯标准库实现，
# import 失败只降级（/api/* 端点 503），不影响主服务
try:
    from agent import webauth as web_auth
    _webauth_loaded = True
    _webauth_error = None
except Exception:
    web_auth = None
    _webauth_loaded = False
    _webauth_error = traceback.format_exc()
    logger.error("网站认证模块 agent.webauth 加载失败，/api/* 将不可用", exc_info=True)

# ── 配置 ──

AGENT_API_KEY = os.getenv("AGENT_API_KEY")
if not AGENT_API_KEY:
    raise RuntimeError(
        "缺少必填环境变量 AGENT_API_KEY（智能体调用密钥），服务拒绝启动。"
        "请在环境变量或 .env 文件中设置 AGENT_API_KEY 后重启。"
    )
AGENT_NAME = "市场复盘智能体"
AGENT_DESCRIPTION = (
    "A股市场每日复盘智能体，提供全市场31行业覆盖、"
    "宏观新闻S/A/B/C四级权威性分级解读、资金流向分析、"
    "单板块7维度深度聚焦。数据来源：DeepSeek + Tushare + Finnhub + FRED。"
)
AGENT_VERSION = "1.3.0"

# ── FastAPI App ──

PUSH_LOOP_INTERVAL_SECONDS = 60  # 定时推送循环唤醒间隔


async def _push_loop():
    """
    定时推送后台任务：每 60 秒醒一次，工作日越过 PUSH_TIME 即触发一次。

    由 lifespan 在服务启动时创建、关闭时取消。循环体内所有异常都被吞掉
    并记日志（fail-safe），绝不会因推送问题影响主服务。
    """
    if push_tasks is None:
        logger.error("推送模块不可用，定时推送任务直接退出")
        return
    last_fired_date = None  # 当天触发后回存，防同一交易日重复推送
    logger.info(
        "定时推送任务已启动（PUSH_TIME=%s，webhook=%s）",
        os.getenv("PUSH_TIME", push_tasks.DEFAULT_FIRE_TIME),
        "已配置" if os.getenv("PUSH_WEBHOOK_URL") else "未配置（仅生成不发送）",
    )
    while True:
        try:
            _fired, last_fired_date = await push_tasks.push_tick(
                fire_time=os.getenv("PUSH_TIME", push_tasks.DEFAULT_FIRE_TIME),
                webhook_url=(os.getenv("PUSH_WEBHOOK_URL", "").strip() or None),
                last_fired_date=last_fired_date,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("定时推送循环本轮异常（已吞掉，下轮继续）", exc_info=True)
        await asyncio.sleep(PUSH_LOOP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期：启动时拉起定时推送后台任务，关闭时取消并等待退出。"""
    push_task = asyncio.create_task(_push_loop())
    try:
        yield
    finally:
        push_task.cancel()
        try:
            await push_task
        except asyncio.CancelledError:
            pass
        logger.info("定时推送任务已停止")


app = FastAPI(
    title=AGENT_NAME,
    description=AGENT_DESCRIPTION,
    version=AGENT_VERSION,
    lifespan=lifespan,
)

# ── 图表静态文件（第五波可视化产出，目录不存在先创建再挂载）──

# CHART_DIR 显式设置优先；缺省沿用 DATA_DIR 约定（${DATA_DIR:-data}/charts），
# 与 agent/charts.py、agent/push.py 的解析保持一致，挂卷（DATA_DIR=/data）时无需再单设。
CHART_DIR = os.getenv("CHART_DIR") or os.path.join(os.getenv("DATA_DIR") or "data", "charts")
try:
    os.makedirs(CHART_DIR, exist_ok=True)
except OSError:
    logger.warning("图表目录创建失败：%s（/charts 将返回 404）", CHART_DIR, exc_info=True)
# check_dir=False：目录创建失败时也不让主服务启动崩溃
app.mount("/charts", StaticFiles(directory=CHART_DIR, check_dir=False), name="charts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 鉴权 ──

def verify_api_key(request: Request) -> None:
    """验证 API Key。"""
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {AGENT_API_KEY}"

    if auth != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": "未授权：API Key 无效", "code": "invalid_api_key"},
        )


# ── 限流与每日配额（第六波工程化）──
#
# 纯内存实现，零外部依赖。局限：仅对单 worker 进程有效——Railway 单实例部署下
# 计数准确；若未来水平扩容为多实例/多 worker，各进程计数互相独立，需外置共享
# 计数器（如 Redis）才能保证全局一致。


def _env_int(name: str, default: int) -> int:
    """读取正整数环境变量；缺失或非法时回退默认值并记日志（不让配置错误摧毁启动）。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是整数，回退默认值 %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("环境变量 %s=%d 不是正整数，回退默认值 %d", name, value, default)
        return default
    return value


RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 30)  # 每分钟限流阈值
QUOTA_DAILY = _env_int("QUOTA_DAILY", 500)               # 每日配额阈值

_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

_quota_lock = threading.Lock()
_quota_state = {
    "minutes": {},                        # 分钟桶：int(epoch // 60) -> 本桶已用计数
    "daily": {"date": None, "count": 0},  # 每日配额：date 为上海自然日 YYYY-MM-DD
}


def _quota_now() -> float:
    """配额计数使用的当前时间（epoch 秒）。独立成函数，便于测试 patch 注入时间。"""
    return time.time()


def _quota_today(ts: float) -> str:
    """把 epoch 秒按 Asia/Shanghai 时区折算为自然日（YYYY-MM-DD）。"""
    return datetime.fromtimestamp(ts, tz=_SHANGHAI_TZ).date().isoformat()


def _check_rate_and_quota() -> Optional[JSONResponse]:
    """
    对一次【已通过鉴权】的 /v1/chat/completions 请求做限流与每日配额检查。

    - 分钟限流：滑动窗口以"每分钟桶"近似——按 int(epoch//60) 分桶，当前分钟桶
      计数达到 RATE_LIMIT_PER_MIN 即拒绝；跨入下一分钟自动恢复。
    - 每日配额：按 Asia/Shanghai 自然日计数，达到 QUOTA_DAILY 即拒绝；
      跨日后的首次请求自动归零重置。
    - 分钟限流优先于每日配额检查（更瞬时的限制先报）；两项均未超限才一起 +1，
      任一超限返回 429 且不消耗任何计数。
    - 鉴权失败（401）的请求在 verify_api_key 处已被拒绝，到不了这里，不计数。

    返回 None 表示放行；返回 429 JSONResponse 表示拒绝（OpenAI 风格 error 结构）。
    """
    ts = _quota_now()
    minute_bucket = int(ts // 60)
    today = _quota_today(ts)
    with _quota_lock:
        minutes = _quota_state["minutes"]
        minute_used = minutes.get(minute_bucket, 0)
        if minute_used >= RATE_LIMIT_PER_MIN:
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "请求过于频繁，请稍后再试", "code": "rate_limited"}},
            )
        daily = _quota_state["daily"]
        if daily["date"] != today:
            daily["date"] = today
            daily["count"] = 0
        if daily["count"] >= QUOTA_DAILY:
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "今日配额已用完", "code": "quota_exceeded"}},
            )
        minutes[minute_bucket] = minute_used + 1
        daily["count"] += 1
        # 惰性清理过期分钟桶，防长时间运行后 dict 膨胀
        for key in [k for k in minutes if k < minute_bucket]:
            del minutes[key]
    return None


# ── 公开网站静态站点（web/ 目录由前端工程维护，后端只读）──

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _web_index_path() -> str:
    """网站首页 index.html 路径。独立成函数便于测试 monkeypatch。"""
    return os.path.join(WEB_DIR, "index.html")


# check_dir=False：web/ 目录可能暂不存在（前端尚未部署），不让主服务启动崩溃
app.mount("/static", StaticFiles(directory=WEB_DIR, check_dir=False), name="static")

# ── OpenAI 兼容端点 ──

@app.get("/")
async def web_index():
    """
    公开网站首页：web/index.html 存在时返回静态页面；
    前端尚未部署（web/ 目录缺失）时返回 503 占位，不影响 API 可用性。
    """
    index_path = _web_index_path()
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        status_code=503,
        content={"error": "web_not_deployed", "detail": "网站前端尚未部署"},
    )


@app.get("/api/root-info")
async def root_info():
    """服务健康检查（原根路由 / 的 JSON 信息，网站上线后挪到此处）。"""
    result = {
        "service": AGENT_NAME,
        "version": AGENT_VERSION,
        "status": "running" if _agent_loaded else "error",
        "time": datetime.now().isoformat(),
    }
    if not _agent_loaded:
        logger.error("智能体加载失败:\n%s", _agent_error)
        result["error"] = "智能体加载失败，详情见服务端日志"
    return result


@app.get("/v1/models")
async def list_models():
    """列出可用模型（OpenAI 兼容格式）。"""
    return {
        "object": "list",
        "data": [
            {
                "id": "market-review-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "financial-intelligence",
            }
        ],
    }


@app.get("/v1/usage", dependencies=[Depends(verify_api_key)])
async def usage_stats():
    """
    配额用量查询（需 Bearer 鉴权；查询本身不消耗限流与配额计数）。

    返回当前分钟窗口已用/上限、今日（Asia/Shanghai 自然日）已用/上限。
    计数器跨分钟/跨日的重置是惰性的：本端点读取时按当前时间折算，
    不依赖 /v1/chat/completions 请求触发。
    """
    ts = _quota_now()
    minute_bucket = int(ts // 60)
    today = _quota_today(ts)
    with _quota_lock:
        minute_used = _quota_state["minutes"].get(minute_bucket, 0)
        daily = _quota_state["daily"]
        today_used = daily["count"] if daily["date"] == today else 0
    return {
        "today_used": today_used,
        "daily_quota": QUOTA_DAILY,
        "minute_used": minute_used,
        "rate_limit": RATE_LIMIT_PER_MIN,
    }


# ── 研报库管理端点（挂卷后库文件同步：本机爬取 → 上传生产卷）──

_REPORTS_DB_MAX_BYTES = _env_int("REPORTS_DB_UPLOAD_MAX_MB", 200) * 1024 * 1024


def _reports_db_path() -> str:
    """研报库路径解析（复用 agent.report_library 的惰性解析契约，失败时按 DATA_DIR 推导）。"""
    try:
        from agent.report_library import _db_path
        return _db_path(None)
    except Exception:  # noqa: BLE001 - 路径解析兜底，绝不影响端点可用性
        return os.path.join(os.getenv("DATA_DIR") or "data", "reports.db")


@app.get("/v1/admin/reports-db/info", dependencies=[Depends(verify_api_key)])
async def reports_db_info():
    """研报库状态查询（需 Bearer 鉴权；查询本身不消耗限流与配额计数）。"""
    path = _reports_db_path()
    info: dict = {"path": path, "exists": os.path.exists(path)}
    if info["exists"]:
        info["size_bytes"] = os.path.getsize(path)
        try:
            import sqlite3
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                info["total_reports"] = conn.execute(
                    "SELECT COUNT(*) FROM reports"
                ).fetchone()[0]
                info["latest_publish_date"] = conn.execute(
                    "SELECT MAX(publish_date) FROM reports"
                ).fetchone()[0]
        except Exception as e:  # noqa: BLE001 - 库损坏/无表时如实上报而非 500
            info["stats_error"] = str(e)
    return info


@app.post("/v1/admin/reports-db", dependencies=[Depends(verify_api_key)])
async def upload_reports_db(request: Request):
    """
    上传研报库 SQLite 文件（整体替换，需 Bearer 鉴权；不消耗限流与配额计数）。

    请求体为数据库文件原始字节流（Content-Type: application/octet-stream）。
    写入采用 临时文件 + os.replace 原子替换，正在进行的查询不会读到半截文件。
    用途：研报在本地 Mac 爬取（东财等源对海外 IP 不友好），产出库文件后
    上传到生产卷（DATA_DIR=/data 时落 /data/reports.db）。
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="请求体为空：需要 SQLite 文件字节流")
    if len(body) > _REPORTS_DB_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过上限（{_REPORTS_DB_MAX_BYTES // 1024 // 1024}MB）",
        )
    if not body.startswith(b"SQLite format 3\x00"):
        raise HTTPException(status_code=400, detail="不是合法的 SQLite 数据库文件")
    path = _reports_db_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".uploading"
        with open(tmp_path, "wb") as f:
            f.write(body)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.error("研报库写入失败 path=%s: %s", path, e, exc_info=True)
        try:
            os.unlink(path + ".uploading")
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"研报库写入失败：{e}")
    logger.info("研报库已更新 path=%s size=%d", path, len(body))
    return {"ok": True, "path": path, "size_bytes": len(body)}


# ── 舆情快照接收端点（本地 Mac 每日抓取 → 推送生产实例落库）──

_SNAPSHOT_BATCH_MAX = _env_int("SNAPSHOT_BATCH_MAX", 500)  # 单次推送快照条数上限

# 快照中可强制转 float 的数值字段（缺省字段交给 save_snapshot 的默认值/映射逻辑）
_SNAPSHOT_FLOAT_FIELDS = ("n", "pos", "neu", "neg", "w_pos", "w_neu", "w_neg")


def _validate_snapshot(item) -> tuple:
    """
    校验并清洗单条快照。返回 (clean, reason)：
    - 通过：clean 为可直接传给 save_snapshot 的 dict，reason 为 None；
    - 失败：clean 为 None，reason 为失败原因字符串。

    规则：platform/target/date 必填且为非空字符串；n/pos/neu/neg/w_pos/w_neu/w_neg
    若提供必须可强制转为有限 float（bool/None/NaN/Inf/非数值字符串均判非法）。
    """
    if not isinstance(item, dict):
        return None, "条目不是 JSON 对象"
    missing = [
        k for k in ("platform", "target", "date")
        if not isinstance(item.get(k), str) or not item.get(k).strip()
    ]
    if missing:
        return None, f"platform/target/date 必填且须为非空字符串（缺失/非法: {','.join(missing)}）"
    clean = {k: item[k].strip() for k in ("platform", "target", "date")}
    for k in _SNAPSHOT_FLOAT_FIELDS:
        value = item.get(k)
        if value is None:
            continue  # 未提供：交给 save_snapshot 的默认/映射逻辑
        if isinstance(value, bool):
            return None, f"字段 {k} 须为数值，收到 bool"
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None, f"字段 {k} 无法强制转为数值: {value!r}"
        if not math.isfinite(num):
            return None, f"字段 {k} 不是有限数值: {value!r}"
        clean[k] = num
    return clean, None


@app.post("/v1/admin/sentiment/snapshots", dependencies=[Depends(verify_api_key)])
async def ingest_sentiment_snapshots(request: Request):
    """
    批量接收舆情快照并落库（需 Bearer 鉴权；与研报库管理端点一致，
    不消耗限流与每日配额计数）。

    请求体 {"snapshots": [快照dict, ...]}，每条经 _validate_snapshot 校验后调
    agent.sentiment_aggregate.save_snapshot（幂等 INSERT OR REPLACE，
    主键 platform+target+date，路径解析 db_path > env SOCIAL_DB_PATH >
    ${DATA_DIR:-data}/social.db）。用途：舆情在本地 Mac 抓取打分，每日推送快照
    到 Railway 生产实例供趋势查询。

    - 逐条校验，非法条目记入 failed 不中断整批；
    - 空列表合法（saved=0）；条数超过 SNAPSHOT_BATCH_MAX 整批 400 拒绝；
    - 绝不抛 500：未预期异常兜底为 200 + failed 说明。
    """
    saved = 0
    failed: list = []
    try:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")
        snapshots = body.get("snapshots") if isinstance(body, dict) else None
        if not isinstance(snapshots, list):
            raise HTTPException(
                status_code=400,
                detail='请求体须为 {"snapshots": [快照dict, ...]}，snapshots 为数组',
            )
        if len(snapshots) > _SNAPSHOT_BATCH_MAX:
            raise HTTPException(
                status_code=400,
                detail=f"快照条数 {len(snapshots)} 超过上限 {_SNAPSHOT_BATCH_MAX}",
            )
        for i, item in enumerate(snapshots):
            try:
                clean, reason = _validate_snapshot(item)
                if clean is None:
                    failed.append({"index": i, "reason": reason})
                    continue
                if _save_snapshot is None:
                    failed.append({"index": i, "reason": "快照模块未加载，无法落盘"})
                    continue
                if _save_snapshot(clean):
                    saved += 1
                else:
                    failed.append({"index": i, "reason": "save_snapshot 落盘失败（返回 False）"})
            except Exception as e:  # noqa: BLE001 - 单条未预期异常不中断整批
                logger.error("快照条目处理异常 index=%d: %s", i, e, exc_info=True)
                failed.append({"index": i, "reason": f"未预期异常: {e}"})
        logger.info("舆情快照批量落库：saved=%d failed=%d", saved, len(failed))
        return {"ok": True, "saved": saved, "failed": failed}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - 铁律：兜底为 200 + failed，绝不 500
        logger.error("快照接收端点未预期异常（兜底 200）: %s", e, exc_info=True)
        failed.append({"index": -1, "reason": f"服务端未预期异常: {e}"})
        return {"ok": True, "saved": saved, "failed": failed}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI 兼容的对话接口。

    请求体格式：
    {
        "model": "market-review-agent",
        "messages": [{"role": "user", "content": "今日复盘"}],
        "stream": false
    }
    """
    verify_api_key(request)

    # 限流与每日配额：仅本端点计数；鉴权失败（401）的请求上一行已拒绝，不计数
    quota_response = _check_rate_and_quota()
    if quota_response is not None:
        return quota_response

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages 数组不能为空")

    user_message = messages[-1].get("content", "") if messages else ""
    stream = body.get("stream", False)
    # 多轮对话：提取完整历史（过滤 system 消息，去掉最后一条当前消息，最多保留最近10轮）
    history = _extract_history(messages)

    if not _agent_loaded:
        raise HTTPException(status_code=503, detail=f"智能体加载失败: {_agent_error[-200:] if _agent_error else 'unknown'}")

    agent = get_agent()

    if stream:
        return StreamingResponse(
            _stream_chat_completion(agent, user_message, body.get("model", "market-review-agent"), history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            result = await agent.process_message(user_message, stream=False, history=history)
        except Exception as e:
            logger.error("非流式对话处理失败: %s", e, exc_info=True)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "上游模型调用失败，请稍后重试",
                        "type": "upstream_error",
                        "code": "agent_process_error",
                    }
                },
            )

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        return JSONResponse({
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "market-review-agent"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("content", ""),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def _extract_history(messages: list) -> list:
    """
    从 OpenAI 格式 messages 提取对话历史。

    - 过滤掉 system 消息（编排层使用自己的系统提示词）；
    - 去掉最后一条（当前用户消息，已单独作为 user_message 传递）；
    - 只保留 role 为 user/assistant 且 content 为非空字符串的条目；
    - 最多保留最近 10 轮（20 条），防 token 膨胀。
    """
    history = []
    for msg in messages[:-1]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content})
    return history[-20:]


async def _stream_chat_completion(agent, user_message: str, model: str, history: list = None):
    """SSE 流式输出。"""
    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # 立即发送 role chunk，防止连接超时
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

        # 加载提示语分级：只有 detect_intent 判定为重数据采集意图
        # （market_review / sector_deep_dive，采集+成文耗时长）才发等待提示；
        # 其他意图（闲聊/新闻/个股/简单查询等）直接出答案，不发提示。
        intent = None
        if detect_intent is not None:
            try:
                intent, _ = detect_intent(user_message)
            except Exception:
                logger.warning("流式提示语分级的意图判定异常（按不发提示处理）", exc_info=True)
                intent = None
        if intent in ("market_review", "sector_deep_dive"):
            # 显示预计等待时间
            warm = agent.cache_warm
            seconds = "15-30" if warm else "30-40"
            hint = f"正在采集市场数据并生成分析报告，请稍候..（{'首次' if not warm else ''}约需{seconds}秒）\n\n"
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': hint}, 'finish_reason': None}]})}\n\n"

        # 流式输出内容
        async for content_chunk in await agent.process_message(user_message, stream=True, history=history):
            if content_chunk:
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content_chunk},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # 免责条款
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': '\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，不构成任何投资建议。市场有风险，投资需谨慎。'}, 'finish_reason': None}]})}\n\n"

        # 发送结束 chunk
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {
            "error": {"message": str(e), "type": "internal_error", "code": "stream_error"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


# ── 公开网站 /api/* 端点 ──
#
# 与 /v1/* 的 verify_api_key（管理端点专用）完全隔离：/api/* 使用
# agent.webauth 的 Bearer token（users/tokens 表），不参与限流与每日配额，
# 额度走 webauth 的月度定额 + 加油包体系。

def get_web_user(request: Request) -> dict:
    """
    /api/* 的 Bearer token 鉴权依赖。
    解析 Authorization: Bearer <token> → {"id", "username"}；无效/过期 → 401。
    """
    if not _webauth_loaded:
        raise HTTPException(status_code=503, detail="网站认证模块不可用")
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    user = web_auth.resolve_token(token)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "未授权：token 无效或已过期", "code": "invalid_token"},
        )
    return user


@app.post("/api/auth/register")
async def web_register(request: Request):
    """注册：{username, password} → 200 {token, user:{username}}；用户名重复 409。"""
    if not _webauth_loaded:
        raise HTTPException(status_code=503, detail="网站认证模块不可用")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")
    try:
        token = web_auth.register(body.get("username"), body.get("password"))
    except web_auth.UserExistsError:
        raise HTTPException(status_code=409, detail="用户名已存在")
    except web_auth.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    user = web_auth.resolve_token(token)
    return {"token": token, "user": {"username": user["username"]}}


@app.post("/api/auth/login")
async def web_login(request: Request):
    """登录：{username, password} → 200 {token, user:{username}}；失败 401。"""
    if not _webauth_loaded:
        raise HTTPException(status_code=503, detail="网站认证模块不可用")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")
    try:
        token = web_auth.login(body.get("username"), body.get("password"))
    except web_auth.AuthError:
        raise HTTPException(
            status_code=401,
            detail={"error": "用户名或密码错误", "code": "invalid_credentials"},
        )
    except web_auth.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    user = web_auth.resolve_token(token)
    return {"token": token, "user": {"username": user["username"]}}


@app.post("/api/auth/logout")
async def web_logout(request: Request):
    """登出：删除 Bearer token（幂等）→ 200 {}。"""
    if not _webauth_loaded:
        raise HTTPException(status_code=503, detail="网站认证模块不可用")
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    web_auth.logout(token)
    return {}


@app.get("/api/me")
async def web_me(user: dict = Depends(get_web_user)):
    """当前用户信息 + 额度快照（月度定额 / 加油包 / 次月 1 日重置日）。"""
    quota = web_auth.get_quota(user["id"])
    return {"username": user["username"], **quota}


@app.get("/api/conversations")
async def web_list_conversations(user: dict = Depends(get_web_user)):
    """对话列表：置顶优先，同组内按 updated_at 降序，每项含 pinned 布尔字段。"""
    return web_auth.list_conversations(user["id"])


@app.post("/api/conversations")
async def web_create_conversation(request: Request, user: dict = Depends(get_web_user)):
    """新建对话：{title} → {id, title, created_at, updated_at}。"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")
    try:
        return web_auth.create_conversation(user["id"], body.get("title"))
    except web_auth.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def web_get_conversation(conversation_id: str, user: dict = Depends(get_web_user)):
    """对话详情（含消息）；不存在或不属于当前用户一律 404。"""
    conv = web_auth.get_conversation(user["id"], conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conv


@app.patch("/api/conversations/{conversation_id}")
async def web_update_conversation(
    conversation_id: str, request: Request, user: dict = Depends(get_web_user)
):
    """
    修改对话标题 / 置顶状态：{title?, pinned?}（至少其一）
    → 200 {id, title, pinned, created_at, updated_at}。

    - title 去空白后 1-60 字符，否则 400；两者都缺 400
    - 不存在或不属于当前用户一律 404
    - updated_at 仅在改 title 时刷新，单独 pin/unpin 不动
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体须为 JSON 对象")
    try:
        result = web_auth.update_conversation(
            user["id"],
            conversation_id,
            title=body.get("title"),
            pinned=body.get("pinned"),
        )
    except web_auth.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return result


@app.delete("/api/conversations/{conversation_id}")
async def web_delete_conversation(conversation_id: str, user: dict = Depends(get_web_user)):
    """删除对话；不存在或不属于当前用户一律 404。"""
    if not web_auth.delete_conversation(user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="对话不存在")
    return {}


@app.post("/api/topup")
async def web_topup(request: Request, user: dict = Depends(get_web_user)):
    """
    加油包充值：{pack_count:1} → 200 {pack_credits, total_remaining}。
    当前为 mock 支付直接到账。
    # TODO(支付): 接真实支付渠道（ Stripe / 微信 / 支付宝 ），
    #   此处改为创建支付订单，到账在支付回调中确认后调 web_auth.topup。
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")
    try:
        quota = web_auth.topup(user["id"], body.get("pack_count", 1))
    except web_auth.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"pack_credits": quota["pack_credits"], "total_remaining": quota["total_remaining"]}


async def _stream_web_chat(agent, user_message: str, history: list,
                           conversation_id: str, user_id: int):
    """
    网站 SSE 流式输出（OpenAI chunk 格式），并在流正常完成后：
    把助手完整回复落库到 messages 表 + 扣减 1 次额度。
    流中途异常：只发 error chunk，不落库、不扣额度。
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = "market-review-agent"
    full_content = ""

    def _chunk(delta: dict, finish_reason=None) -> str:
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        # 立即发送 role chunk，防止连接超时
        yield _chunk({"role": "assistant"})

        async for content_chunk in await agent.process_message(
            user_message, stream=True, history=history
        ):
            if content_chunk:
                full_content += content_chunk
                yield _chunk({"content": content_chunk})

        # 免责条款（与 /v1 流式行为一致，并入助手回复落库）
        disclaimer = (
            "\n\n风险提示：以上内容仅为客观数据整理与公开信息分析，"
            "不构成任何投资建议。市场有风险，投资需谨慎。"
        )
        full_content += disclaimer
        yield _chunk({"content": disclaimer})

        # 流正常完成：先落库 + 扣额度，再发结束 chunk
        web_auth.add_message(conversation_id, "assistant", full_content)
        web_auth.consume_quota(user_id)

        yield _chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("网站流式对话失败 conversation=%s: %s", conversation_id, e, exc_info=True)
        error_data = {
            "error": {"message": str(e), "type": "internal_error", "code": "stream_error"}
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def web_chat(request: Request, user: dict = Depends(get_web_user)):
    """
    网站对话端点：{conversation_id, message} → SSE 流（text/event-stream）。

    - 额度耗尽 → HTTP 429 JSON {error:"quota_exhausted", total_remaining:0, reset_date}
    - 用户消息立即落库；助手回复在流正常完成后落库并扣 1 次额度
    - 对话不存在或不属于当前用户 → 404
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    conversation_id = body.get("conversation_id")
    message = body.get("message")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise HTTPException(status_code=400, detail="conversation_id 不能为空")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    conv = web_auth.get_conversation(user["id"], conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    quota = web_auth.get_quota(user["id"])
    if quota["total_remaining"] <= 0:
        return JSONResponse(
            status_code=429,
            content={
                "error": "quota_exhausted",
                "total_remaining": 0,
                "reset_date": quota["reset_date"],
            },
        )

    if not _agent_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"智能体加载失败: {_agent_error[-200:] if _agent_error else 'unknown'}",
        )

    agent = get_agent()
    history = web_auth.get_history(conversation_id)
    # 用户消息先落库（即使后续流失败，用户提问也保留在对话里）
    web_auth.add_message(conversation_id, "user", message.strip())

    return StreamingResponse(
        _stream_web_chat(agent, message.strip(), history, conversation_id, user["id"]),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 调试端点 ──

@app.get("/debug/mcp-test", dependencies=[Depends(verify_api_key)])
async def debug_mcp_test(tool: str = "cnMarketUpdownDistribution"):
    """测试任意MCP工具——返回原始响应。"""
    import requests, os
    token = os.getenv("SINA_MCP_TOKEN","")
    base = "https://mcp.finance.sina.com.cn/mcp-http"
    r = requests.post(f"{base}?token={token}", json={
        "jsonrpc":"2.0","method":"initialize","id":1,
        "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"a","version":"1"}}
    }, timeout=15)
    sid = r.headers.get("Mcp-Session-Id","")
    r2 = requests.post(f"{base}?token={token}", json={
        "jsonrpc":"2.0","method":"tools/call","id":2,
        "params":{"name": tool, "arguments": {}}
    }, headers={"Mcp-Session-Id":sid}, timeout=30)
    return {"tool": tool, "response": str(r2.json())[:1500]}


@app.get("/debug/hot", dependencies=[Depends(verify_api_key)])
async def debug_hot():
    """热搜原始响应。"""
    import requests, os
    token = os.getenv("SINA_MCP_TOKEN","")
    base = "https://mcp.finance.sina.com.cn/mcp-http"
    r = requests.post(f"{base}?token={token}", json={
        "jsonrpc":"2.0","method":"initialize","id":1,
        "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"a","version":"1"}}
    }, timeout=15)
    sid = r.headers.get("Mcp-Session-Id","")
    r2 = requests.post(f"{base}?token={token}", json={
        "jsonrpc":"2.0","method":"tools/call","id":2,
        "params":{"name":"globalStockHotBoard","arguments":{"type":"hot","market":"cn","num":5,"page":1}}
    }, headers={"Mcp-Session-Id":sid}, timeout=30)
    return {"text": str(r2.json())[:1000]}


@app.get("/debug/sina-news", dependencies=[Depends(verify_api_key)])
async def debug_sina_news():
    """测试新浪历史新闻是否能拉取。"""
    from agent.data_fetcher import fetch_sina_news
    d1 = "2026-07-20"
    d2 = "2026-07-19"
    items1 = fetch_sina_news(30, d1)
    items2 = fetch_sina_news(30, d2)
    return {
        "d1_count": len(items1),
        "d2_count": len(items2),
        "d1_sample": [i["title"][:60] for i in items1[:3]],
        "d2_sample": [i["title"][:60] for i in items2[:3]],
    }


@app.get("/debug/sector-stocks", dependencies=[Depends(verify_api_key)])
async def debug_sector_stocks(sector: str = "食品饮料"):
    """测试板块成分股数据获取。"""
    from agent.data_fetcher import fetch_sector_stock_detail
    today = datetime.now().strftime("%Y%m%d")
    detail = fetch_sector_stock_detail(sector, today)
    return {"sector": sector, "detail": detail}


@app.get("/debug/derivatives", dependencies=[Depends(verify_api_key)])
async def debug_derivatives():
    """测试衍生品数据权限。"""
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        return {"status": "no_token"}

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()
    results = {}

    tests = [
        ("opt_daily", lambda: pro.opt_daily(trade_date="20260718")),
        ("opt_basic", lambda: pro.opt_basic(exchange="SSE")),
        ("fut_daily", lambda: pro.fut_daily(trade_date="20260718")),
        ("fut_holding", lambda: pro.fut_holding(trade_date="20260718")),
    ]

    for name, fn in tests:
        try:
            df = fn()
            if df is not None and not df.empty:
                results[name] = {"status": "ok", "rows": len(df), "cols": list(df.columns)[:8]}
            else:
                results[name] = {"status": "empty"}
        except Exception as e:
            results[name] = {"status": "fail", "error": str(e)[:120]}

    return {"derivatives": results}


@app.get("/debug/macro", dependencies=[Depends(verify_api_key)])
async def debug_macro():
    """测试 Tushare 宏观数据 + 个股基本面接口权限。"""
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        return {"status": "no_token"}

    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()
    results = {}

    # 宏观数据
    macro_tests = [
        ("cn_cpi", lambda: pro.cn_cpi(start_m="202606", end_m="202607")),
        ("cn_ppi", lambda: pro.cn_ppi(start_m="202606", end_m="202607")),
        ("cn_pmi", lambda: pro.cn_pmi(start_m="202606", end_m="202607")),
        ("cn_m", lambda: pro.cn_m(start_m="202606", end_m="202607")),
        ("cn_gdp", lambda: pro.cn_gdp(start_q="2025Q1", end_q="2026Q1")),
        ("sf_month", lambda: pro.sf_month(start_m="202606", end_m="202607")),
        ("daily_basic", lambda: pro.daily_basic(ts_code="000001.SZ", trade_date="20260718")),
    ]

    for name, fn in macro_tests:
        try:
            df = fn()
            if df is not None and not df.empty:
                results[name] = {"status": "ok", "rows": len(df), "columns": list(df.columns)[:8]}
            else:
                results[name] = {"status": "empty"}
        except Exception as e:
            results[name] = {"status": "fail", "error": str(e)[:120]}

    return {"macro_test": results}


@app.get("/debug/mcp-news", dependencies=[Depends(verify_api_key)])
async def debug_mcp_news():
    """测试MCP连通性+新闻搜索。"""
    import requests as req, traceback, os
    token = os.getenv("SINA_MCP_TOKEN", "")
    result = {"token_exists": bool(token), "steps": []}
    try:
        base = "https://mcp.finance.sina.com.cn/mcp-http"
        r = req.post(f"{base}?token={token}", json={
            "jsonrpc":"2.0","method":"initialize","id":1,
            "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"a","version":"1"}}
        }, timeout=15)
        result["init_status"] = r.status_code
        sid = r.headers.get("Mcp-Session-Id","")
        result["session"] = bool(sid)
        if sid:
            r2 = req.post(f"{base}?token={token}", json={
                "jsonrpc":"2.0","method":"tools/call","id":2,
                "params":{"name":"qNewsSearch","arguments":{"keyword":"银行","num":5,"page":1}}
            }, headers={"Mcp-Session-Id":sid}, timeout=30)
            d = r2.json()
            content = d.get("result",{}).get("content",[])
            if content:
                text = content[0].get("text","")
                data = json.loads(text)
                items = data.get("result",{}).get("data",{}).get("data",[])
                result["count"] = len(items)
                result["sample"] = [(i.get("title","") or i.get("content",""))[:60] for i in items[:3]]
        return result
    except Exception as e:
        logger.warning("/debug/mcp-news 失败: %s", e, exc_info=True)
        return {"error": str(e)[:200]}


@app.get("/debug/stock-all", dependencies=[Depends(verify_api_key)])
async def debug_stock_all():
    """测试个股全流程。"""
    import traceback
    try:
        from agent.data_fetcher import fetch_stock_quote, fetch_stock_kline, fetch_stock_news
        q = fetch_stock_quote("cn","sh600519")
        k = fetch_stock_kline("cn","sh600519",5)
        n = fetch_stock_news("sh600519","cn",5)
        return {"quote": bool(q), "kline": len(k), "news": len(n)}
    except Exception as e:
        logger.warning("/debug/stock-all 失败: %s", e, exc_info=True)
        return {"error": str(e)[:200]}


@app.get("/debug/futures", dependencies=[Depends(verify_api_key)])
async def debug_futures():
    """测试期货+个股API。"""
    from agent.data_fetcher import fetch_futures, fetch_stock_quote
    f = fetch_futures("gn","AU0")
    s = fetch_stock_quote("cn","sh600519")
    return {"futures": f, "stock": s}


@app.get("/debug/news-count", dependencies=[Depends(verify_api_key)])
async def debug_news_count():
    """检查新闻数据是否进入了snapshot。"""
    import asyncio
    from datetime import datetime
    from agent.data_fetcher import (
        fetch_eastmoney_news, fetch_sina_news, fetch_eastmoney_news_page2
    )
    loop = asyncio.get_event_loop()
    d1 = "2026-07-20"
    d2 = "2026-07-19"
    em1 = await loop.run_in_executor(None, fetch_eastmoney_news, 80)
    em2 = await loop.run_in_executor(None, fetch_eastmoney_news_page2, 80)
    sina1 = await loop.run_in_executor(None, fetch_sina_news, 30, d1)
    sina2 = await loop.run_in_executor(None, fetch_sina_news, 30, d2)
    return {
        "em_p1": len(em1 or []),
        "em_p2": len(em2 or []),
        "sina_d1": len(sina1 or []),
        "sina_d2": len(sina2 or []),
        "total_sina": len(sina1 or []) + len(sina2 or []),
    }


@app.get("/debug/pipeline", dependencies=[Depends(verify_api_key)])
async def debug_pipeline():
    """测试完整数据采集管线。"""
    from agent.orchestrator import _get_latest_trade_date
    from datetime import datetime

    today = datetime.now()
    trade_date = _get_latest_trade_date(today)
    date_str = trade_date.strftime("%Y%m%d")

    # 运行实际数据采集
    from agent.data_fetcher import (
        fetch_a_share_indices, fetch_shenwan_sectors,
        fetch_fund_flows, fetch_global_indices,
        fetch_us_macro, fetch_cls_telegraph,
    )

    results = {}

    # A股指数
    idx = fetch_a_share_indices(date_str)
    results["indices"] = {
        "date_used": date_str,
        "count": len(idx),
        "sample": dict(list(idx.items())[:3]) if idx else "EMPTY",
    }

    # 行业
    sec = fetch_shenwan_sectors(date_str)
    results["sectors"] = {
        "count": len(sec),
        "sample": sec[:3] if sec else "EMPTY",
    }

    # 资金
    flow = fetch_fund_flows(date_str)
    results["fund_flows"] = flow if flow else "EMPTY"

    # 全球
    gidx = fetch_global_indices()
    results["global"] = {
        "count": len(gidx),
        "sample": dict(list(gidx.items())[:3]) if gidx else "EMPTY",
    }

    # 宏观
    macro = fetch_us_macro()
    results["macro"] = macro if macro else "EMPTY"

    # 新闻
    news = fetch_cls_telegraph(5)
    results["news_cls"] = f"{len(news)} items" if news else "EMPTY"

    return {
        "pipeline_test": results,
        "dates": {
            "today": today.strftime("%Y%m%d"),
            "trade_date_used": date_str,
        },
    }


@app.get("/debug/tushare", dependencies=[Depends(verify_api_key)])
async def debug_tushare():
    """测试 Tushare API 连通性，返回详细错误信息。"""
    import traceback, requests
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        return {"status": "no_token", "error": "TUSHARE_TOKEN 未设置"}

    results = {
        "token_len": len(token),
        "token_mask": f"{token[:4]}...{token[-4:]}" if len(token) >= 12 else "(too short)",
    }
    # 裸 HTTP 直连 api.tushare.pro：区分 网络/IP封锁 vs SDK/本地环境问题
    try:
        r = requests.post(
            "http://api.tushare.pro",
            json={"api_name": "trade_cal", "token": token,
                  "params": {"exchange": "SSE", "start_date": "20260720", "end_date": "20260723"},
                  "fields": "cal_date,is_open"},
            timeout=20,
        )
        results["raw_http"] = {"status": r.status_code, "body_head": r.text[:400]}
    except Exception:
        results["raw_http"] = {"status": "error", "error": traceback.format_exc()[-400:]}
    try:
        from agent.data_fetcher import _ensure_writable_home
        _ensure_writable_home()
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()

        # 交易日历（最简单的接口）
        try:
            df = pro.trade_cal(exchange="SSE", start_date="20260720", end_date="20260724")
            results["trade_cal"] = {
                "status": "ok",
                "rows": len(df) if df is not None else 0,
            }
        except Exception as e:
            results["trade_cal"] = {"status": "fail", "error": str(e)[:200]}

        # 指数行情
        try:
            df = pro.index_daily(ts_code="000001.SH", start_date="20260717", end_date="20260720")
            results["index_daily"] = {
                "status": "ok",
                "rows": len(df) if df is not None else 0,
            }
        except Exception as e:
            results["index_daily"] = {"status": "fail", "error": str(e)[:200]}

        # 申万行业
        try:
            df = pro.sw_daily(trade_date="20260717")
            results["sw_daily"] = {
                "status": "ok",
                "rows": len(df) if df is not None else 0,
            }
        except Exception as e:
            results["sw_daily"] = {"status": "fail", "error": str(e)[:200]}

        # 资金流向
        try:
            df = pro.moneyflow_hsgt(start_date="20260717", end_date="20260718")
            results["moneyflow"] = {
                "status": "ok",
                "rows": len(df) if df is not None else 0,
            }
        except Exception as e:
            results["moneyflow"] = {"status": "fail", "error": str(e)[:200]}

    except Exception as e:
        results["init"] = {"status": "fail", "error": str(e)[:200]}

    return {"tushare": results}

@app.get("/health")
async def health_check():
    """详细健康检查。"""
    import sys
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "agent": {
            "model": "deepseek-chat",
            "capabilities": [
                "market_daily_review",
                "sector_deep_dive",
                "general_chat",
            ],
        },
        "apis": {
            "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
            "tushare": bool(os.getenv("TUSHARE_TOKEN")),
            "finnhub": bool(os.getenv("FINNHUB_API_KEY")),
            "fred": bool(os.getenv("FRED_API_KEY")),
            "brave_search": bool(os.getenv("BRAVE_SEARCH_API_KEY")),
        },
    }


# ── 启动 ──

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"[{AGENT_NAME}] 启动中...")
    print(f"  模型: DeepSeek (deepseek-chat)")
    print(f"  端口: {port}")
    print(f"  API Key: {'已设置' if os.getenv('AGENT_API_KEY') else '使用默认值'}")
    print(f"  Tushare: {'已配置' if os.getenv('TUSHARE_TOKEN') else '未配置'}")
    print(f"  Finnhub: {'已配置' if os.getenv('FINNHUB_API_KEY') else '未配置'}")
    print(f"  FRED: {'已配置' if os.getenv('FRED_API_KEY') else '未配置'}")
    print(f"  定时推送: PUSH_TIME={os.getenv('PUSH_TIME', '15:40')}，"
          f"webhook={'已配置' if os.getenv('PUSH_WEBHOOK_URL') else '未配置（仅生成不发送）'}")
    print(f"  图表目录: {CHART_DIR} → /charts")
    uvicorn.run(app, host="0.0.0.0", port=port)
