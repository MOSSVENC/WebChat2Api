"""
手动测试 SSE 解析 Pipeline (不依赖真实 API)
"""
import asyncio
from deepseek_proxy.sse_parser import parse_sse_stream, parse_dsframe_stream, full_sse_pipeline


# 模拟 DeepSeek SSE 响应
MOCK_SSE = (
    b'event: ready\ndata: {}\n\n'
    b'data: {"v":{"response":{"thinking_enabled":false,"fragments":[{"type":"ANSWER","content":"hello"}]}}}\n\n'
    b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" world"}\n\n'
    b'data: {"v":"! I am"}\n\n'
    b'data: {"v":" DeepSeek."}\n\n'
    b'data: {"p":"response/status","o":"APPEND","v":"FINISHED"}\n\n'
    b'event: finish\ndata: {}\n\n'
)

# 模拟 tool_calls 响应 — 内嵌 JSON 需要双重转义
# 实际 DeepSeek API: fragment content 内的 JSON 会被 JSON-encode
# 所以外层 JSON 的 v 值中, " 会变成 \"
MOCK_TOOL_CALLS = (
    b'event: ready\ndata: {}\n\n'
    b'data: {"v":{"response":{"thinking_enabled":false,"fragments":[{"type":"ANSWER","content":"Let me check that."}]}}}\n\n'
    b'data: {"v":"<tool_calls>[{\\\"name\\\":\\\"get_weather\\\",\\\"arguments\\\":{\\\"city\\\":\\\"Beijing\\\"}}]</tool_calls>"}\n\n'
    b'data: {"p":"response/status","o":"APPEND","v":"FINISHED"}\n\n'
    b'event: finish\ndata: {}\n\n'
)


async def test_sse_events():
    print("=== Layer 1: SSE Events ===")
    async def byte_stream():
        yield MOCK_SSE

    count = 0
    async for evt in parse_sse_stream(byte_stream()):
        print(f"  event={evt.event!r}, data={evt.data[:80]}")
        count += 1
    assert count >= 6, f"Expected >= 6 events, got {count}"
    print(f"  \u2713 {count} events parsed")


async def test_dsframes():
    print("\n=== Layer 2: DsFrames ===")
    async def byte_stream():
        yield MOCK_SSE

    frames = []
    async for frame in parse_dsframe_stream(parse_sse_stream(byte_stream())):
        print(f"  {frame.type.value}: {str(frame.value)[:60]}")
        frames.append(frame)
    assert len(frames) >= 4, f"Expected >= 4 frames, got {len(frames)}"
    print(f"  \u2713 {len(frames)} frames parsed")


async def test_full_pipeline():
    print("\n=== Layer 3: Full Pipeline (OpenAI Chunks) ===")
    async def byte_stream():
        yield MOCK_SSE

    chunks = []
    async for chunk in full_sse_pipeline(byte_stream(), model="test"):
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish = choice.get("finish_reason")
        if delta.get("role"):
            print(f"  role: {delta['role']}")
        if delta.get("content"):
            print(f"  content: {delta['content']}")
        if delta.get("reasoning_content"):
            print(f"  reasoning: {delta['reasoning_content'][:50]}")
        if finish:
            print(f"  finish_reason: {finish}")
        chunks.append(chunk)
    assert len(chunks) >= 4, f"Expected >= 4 chunks, got {len(chunks)}"
    print(f"  \u2713 {len(chunks)} chunks produced")


async def test_tool_calls():
    print("\n=== Tool Call Detection ===")
    async def byte_stream():
        yield MOCK_TOOL_CALLS

    chunks = []
    tool_call_chunks = []
    content_chunks = []
    async for chunk in full_sse_pipeline(byte_stream(), model="test"):
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        if delta.get("tool_calls"):
            tool_call_chunks.append(delta["tool_calls"])
            print(f"  tool_calls: {delta['tool_calls']}")
        if delta.get("content"):
            content_chunks.append(delta["content"])
            print(f"  content: {delta['content']}")
        chunks.append(chunk)

    assert len(tool_call_chunks) >= 1, f"Expected tool_calls chunk, got {len(tool_call_chunks)}"
    assert len(content_chunks) >= 1, f"Expected content chunks, got {len(content_chunks)}"
    # 验证 tool call 内容
    tc = tool_call_chunks[0][0]
    assert tc["function"]["name"] == "get_weather", f"Wrong function name: {tc['function']['name']}"
    print(f"  \u2713 Tool call correctly parsed: {tc['function']['name']}")


async def main():
    await test_sse_events()
    print("---")
    await test_dsframes()
    print("---")
    await test_full_pipeline()
    print("---")
    await test_tool_calls()
    print("\n\u2705 All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
