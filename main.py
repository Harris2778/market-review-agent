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
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from agent.orchestrator import get_agent, detect_intent

# ── 配置 ──

AGENT_API_KEY = os.getenv("AGENT_API_KEY", "market-review-agent-key")
AGENT_NAME = "市场复盘智能体"
AGENT_DESCRIPTION = (
    "A股市场每日复盘智能体，提供全市场31行业覆盖、"
    "宏观新闻S/A/B/C四级权威性分级解读、资金流向分析、"
    "单板块7维度深度聚焦。数据来源：DeepSeek + Tushare + Finnhub + FRED。"
)
AGENT_VERSION = "1.0.0"

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
    return {
        "service": AGENT_NAME,
        "version": AGENT_VERSION,
        "status": "running",
        "time": datetime.now().isoformat(),
    }


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

        # 发送第一个 chunk（role）
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

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

        # 发送结束 chunk
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {
            "error": {"message": str(e), "type": "internal_error", "code": "stream_error"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


# ── 健康检查 ──

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
