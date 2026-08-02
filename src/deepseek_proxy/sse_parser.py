"""
SSE 流解析 Pipeline — 三层转换: Byte → SSE Event → DsFrame → OpenAI Chunk

对应 ds-free-api:
- openai_adapter/response/sse_parser.rs  (SseStream)
- openai_adapter/response/state.rs       (StateStream → DsFrame)
- openai_adapter/response/converter.rs   (ConverterStream → ChatCompletionChunk)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional, AsyncIterator, Any


# ═══════════════════════════════════════════════════════════
# Layer 1: SSE Event
# ═══════════════════════════════════════════════════════════

MAX_BUF_SIZE = 4 * 1024 * 1024  # 4 MB，防止 OOM

_RE_SEARCH_PREFIX = re.compile(r'^(SEARCH|WEB_SEARCH|SEARCHING)\s*', re.IGNORECASE)
_TOOL_CALLS_CLOSE = "</tool_calls>"


@dataclass
class SseEvent:
    event: Optional[str] = None
    data: str = ""


async def parse_sse_stream(byte_stream) -> AsyncIterator[SseEvent]:
    """Layer 1: 原始字节流 → SSE 事件"""
    buf = bytearray()

    async for chunk in byte_stream:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        buf.extend(chunk)

        if len(buf) > MAX_BUF_SIZE:
            raise RuntimeError(f"SSE buffer overflow: {len(buf)} bytes > {MAX_BUF_SIZE}")

        while b"\n\n" in buf:
            idx = buf.index(b"\n\n")
            raw_event = bytes(buf[:idx])
            del buf[:idx + 2]
            evt = _parse_raw_event(raw_event.decode("utf-8", errors="replace"))
            if evt:
                yield evt

    if buf.strip():
        evt = _parse_raw_event(bytes(buf).decode("utf-8", errors="replace"))
        if evt:
            yield evt


def _parse_raw_event(text: str) -> Optional[SseEvent]:
    """解析单个 SSE 事件块"""
    event = None
    data_lines: list[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if not data_lines:
        return None

    return SseEvent(event=event, data="\n".join(data_lines))


# ═══════════════════════════════════════════════════════════
# Layer 2: DsFrame (DeepSeek Frame)
# ═══════════════════════════════════════════════════════════

class DsFrameType(Enum):
    ROLE = "role"
    THINK_DELTA = "think_delta"
    CONTENT_DELTA = "content_delta"
    STATUS = "status"
    USAGE = "usage"
    FINISH = "finish"


@dataclass
class DsFrame:
    type: DsFrameType
    value: Any = None


FRAG_THINK = "THINK"
FRAG_RESPONSE = "RESPONSE"
FRAG_ANSWER = "ANSWER"


def _fragment_to_frame(ty: str, content: str) -> Optional[DsFrame]:
    if ty == FRAG_THINK:
        return DsFrame(type=DsFrameType.THINK_DELTA, value=content)
    if ty in (FRAG_RESPONSE, FRAG_ANSWER):
        return DsFrame(type=DsFrameType.CONTENT_DELTA, value=content)
    return None


class DsState:
    """DeepSeek Patch 状态机 — 维护 p/o/v 路径操作状态"""

    def __init__(self):
        self.current_path: Optional[str] = None
        self.fragments: list[dict] = []  # [{type, content}] — 跟踪 fragment 类型
        self.status: Optional[str] = None
        self.accumulated_token_usage: Optional[int] = None

    def apply_event(self, evt: SseEvent) -> list[DsFrame]:
        frames: list[DsFrame] = []

        if evt.event == "ready":
            frames.append(DsFrame(type=DsFrameType.ROLE))
        elif evt.event == "finish":
            frames.append(DsFrame(type=DsFrameType.FINISH))

        if evt.data and evt.data.strip():
            try:
                val = json.loads(evt.data)
            except json.JSONDecodeError:
                return frames
            frames.extend(self._apply_patch_value(val))

        return frames

    def _apply_patch_value(self, val: dict) -> list[DsFrame]:
        frames: list[DsFrame] = []
        has_p = "p" in val
        op = val.get("o")

        if has_p:
            p = val["p"]
            if isinstance(p, str):
                self.current_path = p

        v = val.get("v")
        if v is None:
            return frames

        if has_p or op is not None:
            path = self.current_path
            if not path:
                return frames

            if path == "response" and op == "BATCH":
                if isinstance(v, list):
                    for item in v:
                        frames.extend(self._apply_patch_value(item))
            else:
                frames.extend(self._apply_path(path, op, v))

        elif self.current_path:
            frames.extend(self._apply_path(self.current_path, "APPEND", v))
        else:
            self._apply_snapshot(v, frames)

        return frames

    def _apply_snapshot(self, v: Any, frames: list[DsFrame]) -> None:
        if isinstance(v, dict):
            resp = v.get("response", v)
            if isinstance(resp, dict):
                # 记录 fragment 类型供后续 APPEND 判断
                frags = resp.get("fragments", [])
                if isinstance(frags, list):
                    self.fragments = [
                        {"type": f.get("type", ""), "content": f.get("content", "")}
                        for f in frags if isinstance(f, dict)
                    ]
                for frag in frags if isinstance(frags, list) else []:
                    ty = frag.get("type", "")
                    content = frag.get("content", "")
                    f = _fragment_to_frame(ty, content)
                    if f:
                        frames.append(f)
                status = resp.get("status")
                if status:
                    self.status = status
                    if status == "FINISHED":
                        frames.append(DsFrame(type=DsFrameType.STATUS, value="FINISHED"))
                usage = resp.get("usage")
                if usage and isinstance(usage, dict):
                    tu = usage.get("total_token_usage")
                    if tu is not None:
                        self.accumulated_token_usage = tu
                        frames.append(DsFrame(type=DsFrameType.USAGE, value=tu))
            elif isinstance(resp, str):
                frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=resp))
        elif isinstance(v, str):
            frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=v))

    def _apply_path(self, path: str, op: Optional[str], v: Any) -> list[DsFrame]:
        frames: list[DsFrame] = []

        if "usage" in path and "total_token_usage" in path:
            if isinstance(v, (int, float)):
                self.accumulated_token_usage = int(v)
                frames.append(DsFrame(type=DsFrameType.USAGE, value=int(v)))
            return frames

        if path.endswith("/status"):
            status_val = v if isinstance(v, str) else str(v)
            self.status = status_val
            frames.append(DsFrame(type=DsFrameType.STATUS, value=status_val))
            return frames

        if "/content" in path:
            if isinstance(v, str):
                text = v
            elif isinstance(v, dict) and "value" in v:
                text = v["value"]
            else:
                text = str(v) if v else ""

            # fragments/<idx>/content 路径: 按路径索引取 fragment 类型分流
            if "/fragments/" in path and self.fragments:
                idx = self._extract_frag_idx(path)
                target = self._fragment_at(idx)
                if target is not None:
                    ty = target.get("type", "")
                    # 累积到对应 fragment 内容
                    target["content"] = target.get("content", "") + text
                    if ty == FRAG_THINK:
                        frames.append(DsFrame(type=DsFrameType.THINK_DELTA, value=text))
                        return frames
                    elif ty in (FRAG_RESPONSE, FRAG_ANSWER):
                        frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=text))
                        return frames

            # fragments APPEND (新 fragment 加入)
            if path == "response/fragments" and isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        ty = item.get("type", "")
                        content = item.get("content", "")
                        if not isinstance(content, str):
                            content = ""
                        self.fragments.append({"type": ty, "content": content})
                        if content:
                            f = _fragment_to_frame(ty, content)
                            if f:
                                frames.append(f)
                return frames

            frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=text))
            return frames

        if isinstance(v, str):
            frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=v))
        elif isinstance(v, dict):
            content = v.get("content", "")
            if content:
                frames.append(DsFrame(type=DsFrameType.CONTENT_DELTA, value=str(content)))

        return frames



    @staticmethod
    def _extract_frag_idx(path: str) -> Optional[int]:
        """从路径提取 fragment 索引: response/fragments/0/content → 0, -1 → -1"""
        parts = path.split("/")
        for i, p in enumerate(parts):
            if p == "fragments" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    def _fragment_at(self, idx: Optional[int]):
        """按索引取 fragment，支持 -1 表示最后一个"""
        if idx is None or not self.fragments:
            return None
        try:
            return self.fragments[idx]
        except IndexError:
            return None


async def parse_dsframe_stream(sse_stream: AsyncIterator[SseEvent]) -> AsyncIterator[DsFrame]:
    """Layer 2: SSE 事件流 → DsFrame 流"""
    state = DsState()

    async for evt in sse_stream:
        frames = state.apply_event(evt)
        for frame in frames:
            yield frame


# ═══════════════════════════════════════════════════════════
# Layer 3: DsFrame → OpenAI Chunk
# ═══════════════════════════════════════════════════════════

@dataclass
class Delta:
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list] = None


def make_chunk(delta: Delta, model: str = "deepseek-chat",
               finish_reason: Optional[str] = None) -> dict:
    d: dict = {}
    if delta.role is not None:
        d["role"] = delta.role
    if delta.content is not None:
        d["content"] = delta.content
    if delta.reasoning_content is not None:
        d["reasoning_content"] = delta.reasoning_content
    if delta.tool_calls is not None:
        d["tool_calls"] = delta.tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": d,
            "finish_reason": finish_reason,
        }],
    }


def make_usage_chunk(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion.chunk",
        "model": "",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def convert_to_openai_chunks(
    ds_frame_stream: AsyncIterator[DsFrame],
    model: str = "deepseek-chat",
    include_usage: bool = False,
    prompt_tokens: int = 0,
) -> AsyncIterator[dict]:
    """Layer 3: DsFrame → OpenAI ChatCompletionChunk"""

    finished = False
    usage_sent = False
    usage_value: Optional[int] = None

    # Tool Call 缓冲
    tool_calls_active = False
    tool_buffer = ""

    try:
        async for frame in ds_frame_stream:
            if frame.type == DsFrameType.ROLE:
                yield make_chunk(Delta(role="assistant"), model=model)

            elif frame.type == DsFrameType.THINK_DELTA:
                text = _clean_text(frame.value)
                if text:
                    yield make_chunk(Delta(reasoning_content=text), model=model)

            elif frame.type == DsFrameType.CONTENT_DELTA:
                text = _clean_text(frame.value)
                if not text:
                    continue

                # ── Tool Call: 累积缓冲，检测 </tool_calls> 闭合标签 ──
                if tool_calls_active:
                    tool_buffer += text
                    close_idx = tool_buffer.find(_TOOL_CALLS_CLOSE)
                    if close_idx >= 0:
                        tool_calls_active = False
                        tc_json = tool_buffer[:close_idx].strip()
                        parsed = _parse_tool_calls_json(tc_json)
                        if parsed:
                            tc_deltas = _build_tool_call_deltas(parsed)
                            yield make_chunk(Delta(tool_calls=tc_deltas), model=model)
                        after = tool_buffer[close_idx + len(_TOOL_CALLS_CLOSE):]
                        tool_buffer = ""
                        if after.strip():
                            yield make_chunk(Delta(content=after), model=model)
                    continue

                # 检测 <tool_calls> 开始标签
                open_idx = text.find("<tool_calls>")
                if open_idx >= 0:
                    tool_calls_active = True
                    before = text[:open_idx]
                    after_tag = text[open_idx + len("<tool_calls>"):]
                    if before.strip():
                        yield make_chunk(Delta(content=before), model=model)
                    tool_buffer = after_tag
                    # 检查是否同一 chunk 内就闭合
                    close_idx = tool_buffer.find(_TOOL_CALLS_CLOSE)
                    if close_idx >= 0:
                        tool_calls_active = False
                        tc_json = tool_buffer[:close_idx].strip()
                        parsed = _parse_tool_calls_json(tc_json)
                        if parsed:
                            tc_deltas = _build_tool_call_deltas(parsed)
                            yield make_chunk(Delta(tool_calls=tc_deltas), model=model)
                        after = tool_buffer[close_idx + len(_TOOL_CALLS_CLOSE):]
                        tool_buffer = ""
                        if after.strip():
                            yield make_chunk(Delta(content=after), model=model)
                    continue

                yield make_chunk(Delta(content=text), model=model)

            elif frame.type == DsFrameType.STATUS:
                status = str(frame.value) if frame.value else ""
                if status == "FINISHED" and not finished:
                    finished = True
                    yield make_chunk(Delta(), model=model, finish_reason="stop")

            elif frame.type == DsFrameType.USAGE:
                usage_value = int(frame.value) if frame.value else 0
                if finished and include_usage and not usage_sent:
                    usage_sent = True
                    yield make_usage_chunk(prompt_tokens, usage_value)

            elif frame.type == DsFrameType.FINISH:
                if not finished:
                    finished = True
                    yield make_chunk(Delta(), model=model, finish_reason="stop")

    finally:
        if finished and include_usage and not usage_sent and usage_value is not None:
            yield make_usage_chunk(prompt_tokens, usage_value)


def _clean_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("FINISHED", "")
    text = _RE_SEARCH_PREFIX.sub('', text)
    return text


def _parse_tool_calls_json(buf: str) -> list[dict]:
    """从 tool_calls 内容中解析 JSON"""
    buf = buf.strip()
    try:
        parsed = json.loads(buf)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        try:
            fixed = buf.replace("'", '"')
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass
    return []


def _build_tool_call_deltas(parsed: list[dict]) -> list[dict]:
    tc_deltas = []
    for i, tc in enumerate(parsed):
        tc_deltas.append({
            "index": i,
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
            },
        })
    return tc_deltas


# ═══════════════════════════════════════════════════════════
# 完整 Pipeline (便捷函数)
# ═══════════════════════════════════════════════════════════

async def full_sse_pipeline(
    byte_stream,
    model: str = "deepseek-chat",
    include_usage: bool = False,
    prompt_tokens: int = 0,
) -> AsyncIterator[dict]:
    """完整 SSE Pipeline: 字节流 → OpenAI Chunks"""
    sse_stream = parse_sse_stream(byte_stream)
    ds_frame_stream = parse_dsframe_stream(sse_stream)

    async for chunk in convert_to_openai_chunks(
        ds_frame_stream,
        model=model,
        include_usage=include_usage,
        prompt_tokens=prompt_tokens,
    ):
        yield chunk
