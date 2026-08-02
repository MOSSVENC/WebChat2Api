"""
DeepSeek Proxy — 配置文件

支持环境变量覆盖，也可直接修改下方 CONFIG 对象。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class AuthMode(str, Enum):
    PASSWORD = "password"       # 邮箱/手机号 + 密码自动登录
    TOKEN = "token"             # 手动提供 Bearer Token


class SessionMode(str, Enum):
    REUSE = "reuse"             # 初始化时创建 session, 永久复用 edit_message
    NEW = "new"                 # 每次请求新建 session + completion


class DeepSeekModel(str, Enum):
    DEFAULT = "default"                # deepseek-v4-flash（非思考模式）
    DEFAULT_SEARCH = "default_search"  # deepseek-v4-flash + 搜索


@dataclass
class ProxyConfig:
    # === DeepSeek API 连接 ===
    api_base: str = "https://chat.deepseek.com/api/v0"
    wasm_url: str = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"

    # === HTTP 请求头 ===
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    )
    client_version: str = "1.8.0"
    client_platform: str = "web"
    app_version: str = "20241129.1"
    client_locale: str = "zh-CN"

    # === 认证 ===
    auth_mode: AuthMode = AuthMode.TOKEN

    # --- AuthMode.PASSWORD 专用 ---
    account_email: str = ""
    account_mobile: str = ""
    account_area_code: str = ""
    account_password: str = ""

    # --- AuthMode.TOKEN 专用 ---
    user_token: str = ""

    # === 会话策略 ===
    session_mode: SessionMode = SessionMode.REUSE

    # === 服务器 ===
    server_host: str = "127.0.0.1"
    server_port: int = 5317
    api_tokens: list[str] = field(default_factory=list)  # 空 = 不鉴权

    # === 模型 ===
    default_model: DeepSeekModel = DeepSeekModel.DEFAULT
    model_types: list[str] = field(default_factory=lambda: ["default"])

    # === 超时 (秒) ===
    http_timeout: float = 120.0
    token_refresh_interval: int = 3000  # 秒 (50 分钟, 保守)

    @classmethod
    def from_dict(cls, d: dict) -> ProxyConfig:
        """从字典构造, 覆盖默认值"""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in fields}
        # 处理枚举
        if "auth_mode" in kwargs and isinstance(kwargs["auth_mode"], str):
            kwargs["auth_mode"] = AuthMode(kwargs["auth_mode"])
        if "session_mode" in kwargs and isinstance(kwargs["session_mode"], str):
            kwargs["session_mode"] = SessionMode(kwargs["session_mode"])
        if "default_model" in kwargs and isinstance(kwargs["default_model"], str):
            kwargs["default_model"] = DeepSeekModel(kwargs["default_model"])
        return cls(**kwargs)


def _env(key: str, default: str = "") -> str:
    """从环境变量读取，不存在则返回 default"""
    return os.environ.get(key, default)


def _env_list(key: str, default: str = "") -> list[str]:
    """从环境变量读取逗号分隔列表"""
    val = os.environ.get(key, default)
    return [s.strip() for s in val.split(",") if s.strip()]


# === 默认全局配置 (可通过环境变量覆盖) ===
CONFIG = ProxyConfig(
    # 认证: 改为 TOKEN 模式则填 user_token
    auth_mode=AuthMode(_env("DS_AUTH_MODE", "token")),

    # 账号密码登录 (auth_mode=password 时使用)
    account_email=_env("DS_EMAIL"),
    account_mobile=_env("DS_MOBILE"),
    account_area_code=_env("DS_AREA_CODE"),
    account_password=_env("DS_PASSWORD"),

    # Bearer Token (auth_mode=token 时使用)
    # ⚠️ 不要在代码中硬编码 token，请通过 DS_TOKEN 环境变量传入
    user_token=_env("DS_TOKEN"),

    # 会话策略
    session_mode=SessionMode(_env("DS_SESSION_MODE", "reuse")),

    # 服务器
    server_host=_env("DS_HOST", "127.0.0.1"),
    server_port=int(_env("DS_PORT", "5317")),
    api_tokens=_env_list("DS_API_TOKENS"),
)
