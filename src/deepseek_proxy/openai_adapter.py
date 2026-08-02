"""
OpenAI 兼容请求适配 — 接收 OpenAI 格式请求, 转换为 DeepSeek 调用

对应 ds-free-api: openai_adapter/ + openai_adapter/request/
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from .config import ProxyConfig
from .prompt import build_chatml_prompt
from .sessions import SessionStrategy
from .jailbreak import get_jailbreak_prompt

logger = logging.getLogger(__name__)

# 已知模型别名 → (model_type, thinking, search)
_KNOWN_MODELS: dict[str, tuple[str, bool, bool]] = {
    "deepseek-v4-flash": ("default", False, False),
    "deepseek-v3":       ("default", False, False),
    "deepseek-v3.2":     ("default", False, False),
    "default":           ("default", False, False),
    "deepseek-v4-pro":   ("default", True,  True),
}


class OpenAIAdapter:
    """将 OpenAI /v1/chat/completions 请求适配到 DeepSeek 网页 API"""

    def __init__(self, config: ProxyConfig, session: SessionStrategy):
        self.config = config
        self.session = session

    def _resolve_model(self, model: str) -> tuple[str, bool, bool]:
        """解析模型名 → (model_type, thinking_enabled, search_enabled)"""
        model_lower = model.lower()

        # 精确匹配已知模型
        if model_lower in _KNOWN_MODELS:
            return _KNOWN_MODELS[model_lower]

        # 模糊匹配 (搜索类)
        if "search" in model_lower:
            return "default", False, True

        # 模糊匹配 (思考/推理类)
        if any(kw in model_lower for kw in ("think", "reasoner", "r1", "o1", "o3")):
            return "default", True, False

        # 未知模型 — 日志警告，fallback 到 default
        logger.warning("Unknown model '%s', falling back to default (deepseek-v4-flash)", model)
        return "default", False, False

    def build_prompt(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        response_format: Optional[dict] = None,
        web_search: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        jailbreak: Optional[str] = None,
    ) -> str:
        """构建 ChatML prompt

        如果指定了 jailbreak 模板，会在 messages 最前面注入
        一条 system message 作为破限指令。
        """
        jailbreak_prompt = get_jailbreak_prompt(jailbreak)
        if jailbreak_prompt:
            messages = [
                {"role": "system", "content": jailbreak_prompt},
                *messages,
            ]

        return build_chatml_prompt(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    async def chat(
        self,
        messages: list[dict],
        model: str = "deepseek-v4-flash",
        stream: bool = True,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        response_format: Optional[dict] = None,
        web_search: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        jailbreak: Optional[str] = None,
    ):
        """执行一次对话请求"""
        model_type, default_thinking, default_search = self._resolve_model(model)

        thinking_enabled = default_thinking
        search_enabled = default_search

        if reasoning_effort is not None:
            thinking_enabled = True
        if web_search is not None:
            search_enabled = bool(web_search)

        prompt = self.build_prompt(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            web_search=web_search,
            reasoning_effort=reasoning_effort,
            jailbreak=jailbreak,
        )

        resp = await self.session.execute(
            prompt=prompt,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            model_type=model_type,
        )

        return resp
