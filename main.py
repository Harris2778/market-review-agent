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
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import traceback

try:
    from agent.orchestrator import get_agent, detect_intent
    _agent_loaded = True
    _agent_error = None
except Exception as e:
    _agent_loaded = False
    _agent_error = traceback.format_exc()
    get_agent = None
    detect_intent = None

# ── 配置 ──

AGENT_API_KEY = os.getenv("AGENT_API_KEY", "market-review-agent-key")
AGENT_NAME = "市场复盘智能体"
AGENT_DESCRIPTION = (
    "A股市场每日复盘智能体，提供全市场31行业覆盖、"
    "宏观新闻S/A/B/C四级权威性分级解读、资金流向分析、"
    "单板块7维度深度聚焦。数据来源：DeepSeek + Tushare + Finnhub + FRED。"
)
AGENT_VERSION = "1.1.0"

# ── FastAPI App ──

app = FastAPI(
    title=AGENT_NAME,
    description=AGENT_DESCRIPTION,
    version=AGENT_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


# ── OpenAI 兼容端点 ──

@app.get("/")
async def root():
    """服务健康检查。"""
    result = {
        "service": AGENT_NAME,
        "version": AGENT_VERSION,
        "status": "running" if _agent_loaded else "error",
        "time": datetime.now().isoformat(),
    }
    if not _agent_loaded:
        result["error"] = _agent_error[-500:] if _agent_error else "unknown"
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

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体格式错误，需要 JSON")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages 数组不能为空")

    user_message = messages[-1].get("content", "") if messages else ""
    stream = body.get("stream", False)

    if not _agent_loaded:
        raise HTTPException(status_code=503, detail=f"智能体加载失败: {_agent_error[-200:] if _agent_error else 'unknown'}")

    agent = get_agent()

    if stream:
        return StreamingResponse(
            _stream_chat_completion(agent, user_message, body.get("model", "market-review-agent")),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await agent.process_message(user_message, stream=False)

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


async def _stream_chat_completion(agent, user_message: str, model: str):
    """SSE 流式输出。"""
    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # 立即发送 role chunk，防止连接超时
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

        # 显示预计等待时间
        warm = agent.cache_warm
        seconds = "15-30" if warm else "30-40"
        hint = f"正在采集市场数据并生成分析报告，请稍候..（{'首次' if not warm else ''}约需{seconds}秒）\n\n"
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': hint}, 'finish_reason': None}]})}\n\n"

        # 流式输出内容
        async for content_chunk in await agent.process_message(user_message, stream=True):
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
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': '\n\n风险提示：以上内容仅为行情数据复盘，不构成任何投资建议。本智能体由AI驱动，市场数据来源于公开信息，分析结论仅供参考。智能体开发同学与以上内容无任何责任关系。市场有风险，投资需谨慎。'}, 'finish_reason': None}]})}\n\n"

        # 发送结束 chunk
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {
            "error": {"message": str(e), "type": "internal_error", "code": "stream_error"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


# ── 调试端点 ──

@app.get("/debug/mcp-test")
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


@app.get("/debug/hot")
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


@app.get("/debug/sina-news")
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


@app.get("/debug/sector-stocks")
async def debug_sector_stocks(sector: str = "食品饮料"):
    """测试板块成分股数据获取。"""
    from agent.data_fetcher import fetch_sector_stock_detail
    today = datetime.now().strftime("%Y%m%d")
    detail = fetch_sector_stock_detail(sector, today)
    return {"sector": sector, "detail": detail}


@app.get("/debug/derivatives")
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


@app.get("/debug/macro")
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


@app.get("/debug/mcp-news")
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
        return {"error": str(e)[:200], "trace": traceback.format_exc()[-300:]}


@app.get("/debug/stock-all")
async def debug_stock_all():
    """测试个股全流程。"""
    from agent.data_fetcher import fetch_stock_quote, fetch_stock_kline, fetch_stock_news
    q = fetch_stock_quote("cn","sh600519")
    k = fetch_stock_kline("cn","sh600519",5)
    n = fetch_stock_news("sh600519","cn",5)
    return {"quote": bool(q), "kline": len(k), "news": len(n), "k_sample": k[:2] if k else [], "n_sample": n[:2] if n else []}


@app.get("/debug/futures")
async def debug_futures():
    """测试期货+个股API。"""
    import os as _os
    token = _os.getenv("SINA_MCP_TOKEN","")[:10]
    from agent.data_fetcher import fetch_futures, fetch_stock_quote
    f = fetch_futures("gn","AU0")
    s = fetch_stock_quote("cn","sh600519")
    return {"token_prefix": token, "futures": f, "stock": s}


@app.get("/debug/news-count")
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


@app.get("/debug/pipeline")
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
        fetch_us_macro, fetch_cls_news,
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
    news = fetch_cls_news(5)
    results["news_cls"] = f"{len(news)} items" if news else "EMPTY"

    return {
        "pipeline_test": results,
        "dates": {
            "today": today.strftime("%Y%m%d"),
            "trade_date_used": date_str,
        },
    }


@app.get("/debug/tushare")
async def debug_tushare():
    """测试 Tushare API 连通性，返回详细错误信息。"""
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        return {"status": "no_token", "error": "TUSHARE_TOKEN 未设置"}

    results = {}
    try:
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

    return {"tushare": results, "proxy_env": {
        k: os.environ.get(k) for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY"]
    }}

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
    uvicorn.run(app, host="0.0.0.0", port=port)
