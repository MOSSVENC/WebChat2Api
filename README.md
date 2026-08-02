# WebChat2Api

> DeepSeek 网页 API 反代 — 无需 Electron，纯 Python 实现

提供 **OpenAI 兼容端点**，可直接替代 OpenAI API base URL 使用。

![yes](yes.png)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **OpenAI 兼容接口** | `/v1/chat/completions`、`/v1/models`、`/health` |
| **双模型** | `deepseek-v4-flash`（快速对话）、`deepseek-v4-pro`（深度推理） |
| **双认证** | Password（邮箱/手机号 + 密码自动登录）或 Bearer Token |
| **双会话** | REUSE（session 永久复用）或 NEW（每次新建） |
| **PoW 反爬** | 自动下载 WASM 并求解 DeepSeekHashV1 工作量证明 |
| **流式响应** | SSE 流式输出，支持 `reasoning_content`（思考过程） |
| **Tool Calls** | 函数调用支持（`<tool_calls>` XML 解析） |
| **环境变量配置** | 敏感信息不入代码，支持 `.env` 文件 |
| **CORS** | 跨域请求开箱即用 |

---

## 快速开始

### 环境要求

- Python 3.12+
- UV 包管理器（推荐）

### 安装

```bash
git clone https://github.com/MOSSVENC/WebChat2Api.git
cd WebChat2Api

# 推荐: 使用 UV
uv sync
uv run deepseek-proxy

# 或: pip 安装
pip install -e .
deepseek-proxy

# 或: 直接运行
pip install -r requirements.txt
python -m deepseek_proxy.main
```

### 配置

复制环境变量模板并填入你的 token：

```bash
cp .env.example .env
# 编辑 .env，填入 DS_TOKEN（从浏览器 F12 开发者工具获取）
```

或直接编辑 `src/deepseek_proxy/config.py`。

服务将在 `http://127.0.0.1:5317` 启动。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DS_AUTH_MODE` | `token` | 认证方式：`token` 或 `password` |
| `DS_TOKEN` | — | Bearer Token（TOKEN 模式） |
| `DS_EMAIL` | — | 邮箱（PASSWORD 模式） |
| `DS_MOBILE` | — | 手机号（PASSWORD 模式） |
| `DS_AREA_CODE` | — | 区号（PASSWORD 模式） |
| `DS_PASSWORD` | — | 密码（PASSWORD 模式） |
| `DS_SESSION_MODE` | `reuse` | 会话策略：`reuse` 或 `new` |
| `DS_HOST` | `127.0.0.1` | 监听地址 |
| `DS_PORT` | `5317` | 监听端口 |
| `DS_API_TOKENS` | — | API Key 鉴权（逗号分隔，留空=不鉴权） |

---

## API 端点

### 健康检查

```bash
curl http://127.0.0.1:5317/health
# → {"status": "ok"}
```

### 模型列表

```bash
curl http://127.0.0.1:5317/v1/models
```

### 对话补全（流式）

```bash
curl -X POST http://127.0.0.1:5317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Say hi"}],
    "stream": true
  }'
```

### 对话补全（非流式）

```bash
curl -X POST http://127.0.0.1:5317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Say hi"}],
    "stream": false
  }'
```

### Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-token",
    base_url="http://127.0.0.1:5317/v1"
)

# 快速对话
stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# 深度推理（输出含 reasoning_content）
stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True
)
```

### 高级参数

```python
# 函数调用
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[...],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }],
    stream=True
)

# JSON 输出格式
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[...],
    response_format={"type": "json_object"},
    stream=False
)

# 网页搜索
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[...],
    extra_body={"web_search": True},
    stream=True
)

# 破限模板
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[...],
    extra_body={"jailbreak": "deepseek_unlock"},
    stream=True
)
```

---

## 模型映射

| 请求模型名 | 实际模型 | 模式 | 备注 |
|-----------|---------|------|------|
| `deepseek-v4-flash` | DeepSeek-V4-Flash | 快速对话（默认） | **推荐** |
| `deepseek-v4-pro` | DeepSeek-V4-Pro | 深度推理（默认思考） | **推荐** |
| `deepseek-chat` | DeepSeek-V4-Flash | 快速对话 | ⚠️ 将于 2026/07/24 弃用 |
| `deepseek-reasoner` | DeepSeek-V4-Pro | 思考模式 | ⚠️ 将于 2026/07/24 弃用 |

---

## 项目结构

```
WebChat2Api/
├── src/deepseek_proxy/
│   ├── __init__.py          模块入口，导出核心类型
│   ├── config.py            配置（支持环境变量 + .env）
│   ├── client.py            DsClient HTTP 客户端
│   ├── auth.py              双通道认证（密码 / Token）
│   ├── pow_solver.py        PoW 求解（WASM + wasmtime）
│   ├── sessions.py          会话管理（复用 / 新建）
│   ├── prompt.py            ChatML prompt 构建
│   ├── sse_parser.py        SSE 流解析 Pipeline（3 层）
│   ├── openai_adapter.py    OpenAI → DeepSeek 请求适配
│   ├── jailbreak.py         破限模板系统（5 个预设）
│   ├── server.py            FastAPI 服务器 + token 自动刷新
│   └── main.py              入口
├── pyproject.toml           打包配置
├── requirements.txt         依赖声明
├── .env.example             环境变量模板
└── README.md
```

---

## 技术栈

| 组件 | 用途 |
|------|------|
| FastAPI | HTTP 服务器，路由与中间件 |
| httpx | 异步 HTTP 客户端，支持 SOCKS 代理 |
| uvicorn | ASGI 服务器 |
| pydantic | 数据模型与验证 |
| wasmtime | WebAssembly 运行时（PoW 求解） |

---

## 注意事项

- 本项目仅供学习和研究使用，请遵守 DeepSeek 服务条款
- Bearer Token 从浏览器开发者工具获取（登录 chat.deepseek.com 后按 F12）
- PoW 求解会消耗 CPU 资源，首次启动需要下载 WASM 文件
- 生产环境建议配置 `DS_API_TOKENS` 进行 API Key 鉴权
- REUSE 模式下 session 永久复用，适合长对话；NEW 模式适合独立请求
- Token 自动刷新在 TOKEN 模式下默认启用（每 50 分钟校验一次）

---

## License

MIT
