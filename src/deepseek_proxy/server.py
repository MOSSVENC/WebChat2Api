"""
FastAPI HTTP 服务器 — OpenAI 兼容端点

端点:
- GET  /v1/models
- POST /v1/chat/completions
- GET  /health
"""

from __future__ import annotations

import json
import uuid
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import ProxyConfig
from .client import DsClient
from .auth import create_auth_provider, AuthProvider, TokenAuthProvider
from .pow_solver import create_solver, PowSolver
from .sessions import (
    create_session_strategy, SessionStrategy,
    ReuseSessionStrategy, cleanup_session,
)
from .openai_adapter import OpenAIAdapter
from .sse_parser import full_sse_pipeline

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 全局状态 (应用生命周期内)
# ═══════════════════════════════════════════════════════════

class AppState:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.client: Optional[DsClient] = None
        self.solver: Optional[PowSolver] = None
        self.auth: Optional[AuthProvider] = None
        self.token: Optional[str] = None
        self.session_strategy: Optional[SessionStrategy] = None
        self.adapter: Optional[OpenAIAdapter] = None
        self._refresh_task: Optional[asyncio.Task] = None

    async def init(self):
        """初始化所有组件"""
        # 1. HTTP 客户端
        logger.info("初始化 HTTP 客户端...")
        self.client = DsClient(self.config)

        # 2. 认证
        logger.info("初始化认证 (mode=%s)...", self.config.auth_mode.value)
        self.auth = create_auth_provider(self.config, self.client)
        self.token = await self.auth.authenticate()
        logger.info("认证成功, token: %s...", self.token[:20])

        # 3. PoW 求解器
        logger.info("下载 WASM 并初始化 PoW 求解器...")
        self.solver = await create_solver(self.config.wasm_url)
        logger.info("PoW 求解器初始化完成")

        # 4. 会话策略
        logger.info("初始化会话策略 (mode=%s)...", self.config.session_mode.value)
        self.session_strategy = create_session_strategy(
            self.config, self.client, self.solver, self.token,
            self.config.model_types,
        )
        if isinstance(self.session_strategy, ReuseSessionStrategy):
            await self.session_strategy.init(self.config.model_types)
            logger.info("会话池初始化完成")

        # 5. 适配器
        self.adapter = OpenAIAdapter(self.config, self.session_strategy)

        # 6. 启动 token 定期刷新 (仅 TOKEN 模式)
        if isinstance(self.auth, TokenAuthProvider):
            self._refresh_task = asyncio.create_task(self._token_refresh_loop())
            logger.info("Token 自动刷新已启动 (间隔 %ds)", self.config.token_refresh_interval)

        logger.info("DeepSeek Proxy 初始化完成")

    async def _token_refresh_loop(self):
        """定期验证 token 有效性，失败时尝试刷新"""
        while True:
            try:
                await asyncio.sleep(self.config.token_refresh_interval)
                if self.token is None:
                    break
                logger.debug("验证 token 有效性...")
                new_token = await self.auth.refresh(self.token)  # type: ignore
                if new_token and new_token != self.token:
                    logger.info("Token 已刷新")
                    self.token = new_token
                elif new_token is None:
                    logger.error("Token 刷新失败 — 后续请求可能失败")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Token 刷新循环异常: %s", e)
                await asyncio.sleep(60)

    async def shutdown(self):
        """清理资源"""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        if self.session_strategy:
            await self.session_strategy.cleanup()
        if self.client:
            await self.client.close()


_app_state: Optional[AppState] = None


def get_state() -> AppState:
    if _app_state is None:
        raise RuntimeError("AppState not initialized")
    return _app_state


# ═══════════════════════════════════════════════════════════
# 模型列表 (从枚举生成，避免硬编码重复)
# ═══════════════════════════════════════════════════════════

_MODEL_REGISTRY: list[dict] = [
    {"id": "deepseek-v4-flash", "object": "model", "owned_by": "deepseek"},
    {"id": "deepseek-v4-pro", "object": "model", "owned_by": "deepseek"},
]


# ═══════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════

def create_app(config: ProxyConfig) -> FastAPI:
    global _app_state

    app = FastAPI(
        title="DeepSeek Proxy",
        description="OpenAI-compatible API proxy for DeepSeek chat",
        version="0.2.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key 验证中间件
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        state = _app_state
        if state and state.config.api_tokens:
            auth_header = request.headers.get("Authorization", "")
            token = auth_header.replace("Bearer ", "").strip()

            if not token or token not in state.config.api_tokens:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid API key", "type": "auth_error"}},
                )

        return await call_next(request)

    # ── 端点 ─────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": list(_MODEL_REGISTRY)}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        state = get_state()

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        messages = body.get("messages", [])
        model = body.get("model", state.config.default_model.value)
        stream = body.get("stream", True)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        response_format = body.get("response_format")
        web_search = body.get("web_search")
        reasoning_effort = body.get("reasoning_effort")
        jailbreak = body.get("jailbreak")

        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        try:
            resp = await state.adapter.chat(
                messages=messages,
                model=model,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                web_search=web_search,
                reasoning_effort=reasoning_effort,
                jailbreak=jailbreak,
            )
        except Exception as e:
            logger.error("Chat failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        if not stream:
            return await _handle_non_stream(resp, model)

        return StreamingResponse(
            _event_stream(resp, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# ═══════════════════════════════════════════════════════════
# 流式/非流式处理
# ═══════════════════════════════════════════════════════════

async def _handle_non_stream(resp, model: str) -> dict:
    """非流式: 累积内容 + usage 后返回"""
    full_content = ""
    usage_data: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async for chunk in full_sse_pipeline(resp.aiter_bytes(), model=model, include_usage=True):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            full_content += delta["content"]
        # 提取 usage
        if "usage" in chunk and chunk["usage"]:
            usage_data = chunk["usage"]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_content},
            "finish_reason": "stop",
        }],
        "usage": usage_data,
    }


async def _event_stream(resp, model: str):
    """流式 SSE 生成器"""
    try:
        async for chunk in full_sse_pipeline(resp.aiter_bytes(), model=model):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception:
        logger.exception("Stream error")
        error_chunk = {"error": {"message": "Stream error", "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        session_id = getattr(resp, "_session_id", None)
        if session_id:
            state = get_state()
            await cleanup_session(state.client, state.token, session_id)


# ═══════════════════════════════════════════════════════════
# 初始化入口 (在 main.py 中调用)
# ═══════════════════════════════════════════════════════════

async def init_server(config: ProxyConfig):
    global _app_state
    _app_state = AppState(config)
    await _app_state.init()
    return _app_state


async def shutdown_server():
    if _app_state:
        await _app_state.shutdown()
