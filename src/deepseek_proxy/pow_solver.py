"""
PoW 求解器 — 通过 wasmtime-py 执行 DeepSeekHashV1 WASM

对应:
- ds-free-api: ds_core/pow.rs (动态签名探测)
- Chat2API: src/main/lib/challenge.ts (硬编码导出名)
"""

from __future__ import annotations

import struct
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .client import ChallengeData, build_pow_header

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 错误
# ═══════════════════════════════════════════════════════════

class PowError(Exception):
    pass


# ═══════════════════════════════════════════════════════════
# PoW 结果
# ═══════════════════════════════════════════════════════════

@dataclass
class PowResult:
    algorithm: str
    challenge: str
    salt: str
    answer: int
    signature: str
    target_path: str

    def to_header(self) -> str:
        return build_pow_header(
            self.algorithm, self.challenge, self.salt,
            self.answer, self.signature, self.target_path,
        )


# ═══════════════════════════════════════════════════════════
# wasmtime-py 求解器
# ═══════════════════════════════════════════════════════════

class PowSolver:
    """基于 wasmtime-py 的 DeepSeekHashV1 PoW 求解器

    动态探测 WASM 导出符号，不硬编码名称。
    首次 solve() 后缓存 linker 提升后续性能。
    """

    def __init__(self, wasm_bytes: bytes):
        self._wasm_bytes = wasm_bytes
        self._engine: Any = None
        self._module: Any = None
        self._linker: Any = None
        self._store_cls: Any = None
        self._wasi_config: Any = None
        self._inited = False

    async def init(self) -> None:
        """初始化 WASM engine"""
        try:
            import wasmtime
        except ImportError:
            raise PowError(
                "wasmtime-py 未安装。请运行: pip install wasmtime"
            )

        self._engine = wasmtime.Engine()
        self._store_cls = wasmtime.Store
        self._wasi_config = wasmtime.WasiConfig()

        self._module = wasmtime.Module(self._engine, self._wasm_bytes)

        # 预构建 linker（含 WASI 定义），后续复用
        linker = wasmtime.Linker(self._engine)
        linker.define_wasi()
        self._linker = linker

        # 动态查找关键导出
        exports = list(self._module.exports)
        self._add_to_stack = self._find_export(exports, "__wbindgen_add_to_stack_pointer")
        self._alloc = self._find_alloc_export(exports)
        self._solve = self._find_export(exports, "wasm_solve")

        self._inited = True

    @staticmethod
    def _find_export(exports, name: str) -> str:
        for exp in exports:
            if exp.name == name:
                return exp.name
        raise PowError(f"{name} not found in WASM")

    @staticmethod
    def _find_alloc_export(exports) -> str:
        for exp in exports:
            if exp.name == "__wbindgen_malloc":
                return exp.name
        for exp in exports:
            if exp.name.startswith("__wbindgen_export_"):
                return exp.name
        raise PowError("allocator export not found in WASM")

    def solve(self, challenge: ChallengeData) -> PowResult:
        """求解 PoW 挑战"""
        if challenge.algorithm != "DeepSeekHashV1":
            raise PowError(f"Unsupported algorithm: {challenge.algorithm}")

        if not self._inited:
            raise PowError("Solver not initialized. Call init() first.")

        import wasmtime

        store = self._store_cls(self._engine, self._wasi_config)
        instance = self._linker.instantiate(store, self._module)

        memory = instance.exports(store)["memory"]
        add_to_stack = instance.exports(store)[self._add_to_stack]
        alloc = instance.exports(store)[self._alloc]
        solve_fn = instance.exports(store)[self._solve]

        base = memory.data_ptr(store)

        retptr = add_to_stack(store, -16)
        prefix = f"{challenge.salt}_{challenge.expire_at}_"

        ptr_challenge, len_challenge = _write_string(store, base, alloc,
            challenge.challenge)
        ptr_prefix, len_prefix = _write_string(store, base, alloc, prefix)

        solve_fn(store, retptr,
            ptr_challenge, len_challenge,
            ptr_prefix, len_prefix,
            float(challenge.difficulty))

        # 读返回值: status (i32 at +0), value (f64 at +8)
        status_bytes = bytes(base[retptr + i] for i in range(4))
        value_bytes = bytes(base[retptr + 8 + i] for i in range(8))

        status = struct.unpack("<i", status_bytes)[0]
        value = struct.unpack("<d", value_bytes)[0]

        add_to_stack(store, 16)

        if status == 0:
            raise PowError("No solution found")

        logger.debug("PoW solved: answer=%d, difficulty=%d", int(value), challenge.difficulty)

        return PowResult(
            algorithm=challenge.algorithm,
            challenge=challenge.challenge,
            salt=challenge.salt,
            answer=int(value),
            signature=challenge.signature,
            target_path=challenge.target_path,
        )


def _write_string(store, base, alloc, text: str) -> tuple[int, int]:
    """WASM 内存写入字符串, 返回 (ptr, len)"""
    data = text.encode("utf-8")
    ptr = alloc(store, len(data), 1)
    for i, b in enumerate(data):
        base[ptr + i] = b
    return ptr, len(data)


# ═══════════════════════════════════════════════════════════
# WASM 下载 & 缓存
# ═══════════════════════════════════════════════════════════

async def fetch_wasm(wasm_url: str) -> bytes:
    """下载 WASM 二进制"""
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.get(wasm_url)
        resp.raise_for_status()
        return resp.content


async def create_solver(wasm_url: str) -> PowSolver:
    """下载 WASM 并创建求解器"""
    wasm_bytes = await fetch_wasm(wasm_url)
    solver = PowSolver(wasm_bytes)
    await solver.init()
    return solver
