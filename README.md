# DeepSeek Proxy

DeepSeek 网页 API 反代，提供 OpenAI 兼容端点，可直接替代 OpenAI API base URL 使用。

无需 Electron，纯 Python 实现，支持 DeepSeek-V4-Flash（快速对话）和 DeepSeek-V4-Pro（深度推理）模型。

---

## 功能特性

- **OpenAI 兼容接口** — 适配 `/v1/chat/completions`、`/v1/models`、`/health`
- **双模型支持** — DeepSeek-V4-Flash（`deepseek-v4-flash`）和 DeepSeek-V4-Pro（`deepseek-v4-pro`）
- **双认证通道** — Password（邮箱/手机号 + 密码自动登录）或 Bearer Token
- **双会话策略** — REUSE（初始化创建 session 永久复用）或 NEW（每次请求新建 session）
- **PoW 反爬** — 自动下载 WASM 模块并求解工作量证明
- **流式响应** — Server-Sent Events 流式输出，支持 `reasoning_content`（V4-Pro 推理过程）
- **CORS 支持** — 跨域请求开箱即用
- **环境变量配置** — 敏感信息（Token）通过环境变量传入，无需硬编码

---

## 快速开始

### 环境要求

- Python 3.12+
- UV 包管理器（推荐）

### 安装

```bash
# 克隆
git clone https://github.com/MOSSVENC/chat2api_reborn.git
cd chat2api_reborn

# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 Token
# DS_TOKEN=your-bearer-token-here

# 使用 UV 安装依赖并运行
uv sync
uv run deepseek-proxy
```

服务将在 `http://127.0.0.1:5317` 启动。

### 全局安装（可选）

```bash
uv pip install -e .
deepseek-proxy
```

---

## 配置

### 环境变量（推荐）

```bash
# 认证模式
DS_AUTH_MODE=token  # 或 password

# Token 模式
DS_TOKEN=your-bearer-token

# Password 模式
DS_EMAIL=example@mail.com
DS_PASSWORD=your-password

# 会话策略
DS_SESSION_MODE=reuse  # 或 new

# 服务器
DS_HOST=127.0.0.1
DS_PORT=5317

# API Key 鉴权 (逗号分隔，留空=不鉴权)
DS_API_TOKENS=your-api-key1,your-api-key2
```

### 配置参数说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DS_AUTH_MODE` | `token` | 认证方式 |
| `DS_TOKEN` | — | Bearer Token（auth_mode=token 时） |
| `DS_EMAIL` | — | 邮箱（auth_mode=password 时） |
| `DS_PASSWORD` | — | 密码（auth_mode=password 时） |
| `DS_SESSION_MODE` | `reuse` | REUSE 复用 session；NEW 每次新建 |
| `DS_HOST` | `127.0.0.1` | 监听地址 |
| `DS_PORT` | `5317` | 监听端口 |
| `DS_API_TOKENS` | — | API Key 鉴权列表，留空=不鉴权 |

---

## API 端点

```
GET  /health             健康检查
GET  /v1/models          可用模型列表
POST /v1/chat/completions  对话补全（流式）
```

### 请求示例（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-token",
    base_url="http://127.0.0.1:5317/v1"  # 指向本服务
)

# DeepSeek-V4-Flash 快速对话（默认）
stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")

# DeepSeek-V4-Pro 深度推理（输出包含 reasoning_content）
stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True
)
```

### curl 示例

```bash
curl -X POST http://127.0.0.1:5317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say hi"}],"max_tokens":50}'
```

---

## 模型映射

| OpenAI 模型名 | 模型 | 类型 | 备注 |
|--------------|------|------|------|
| `deepseek-v4-flash` | DeepSeek-V4-Flash | 快速对话（默认） | 推荐使用 |
| `deepseek-v4-pro` | DeepSeek-V4-Pro | 深度推理（默认开启思考） | 推荐使用 |

---

## 项目结构

```
src/deepseek_proxy/
├── __init__.py          模块入口，导出核心类型
├── config.py            配置定义 (支持环境变量覆盖)
├── client.py            DsClient HTTP 客户端
├── auth.py              双通道认证 (密码 / Token)
├── pow_solver.py        PoW 求解 (WASM + wasmtime)
├── sessions.py          会话管理 (复用 / 新建策略)
├── prompt.py            ChatML prompt 构建
├── sse_parser.py        SSE 流解析 Pipeline
├── jailbreak.py         破限模板系统
├── openai_adapter.py    OpenAI 请求 → DeepSeek 请求适配
├── server.py            FastAPI 服务器，定义所有端点
└── main.py              入口，asyncio 事件循环 + cli()
```

---

## 注意事项

- 本项目仅供学习和研究使用，请遵守 DeepSeek 服务条款
- **不要在代码中硬编码 Token**，请通过环境变量或 `.env` 文件传入
- PoW 求解会消耗 CPU 资源，首次启动需要下载 WASM 文件
- 生产环境建议配置 API Key 鉴权 `DS_API_TOKENS`
- REUSE 模式下 session 永久复用，适合长对话；NEW 模式适合独立请求

---

## License

MIT
