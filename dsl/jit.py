from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

from .graph import Graph
from .meta import MetaTensor
from .mlir_emit import emit_mlir
from .mlir_interp import ParsedProgram, parse_mlir
from .tensor import Tensor, ensure_tensor
from .tracer import ProxyTensor, Tracer


@dataclass
class CompiledKernel:
    input_metas: tuple[MetaTensor, ...]
    output_meta: MetaTensor
    graph: Graph
    mlir: str
    program: ParsedProgram


_GLOBAL_STATS = {"hits": 0, "misses": 0}
_ALL_CACHES: list[dict[tuple[Any, ...], CompiledKernel]] = []



def _meta_key(tensors: tuple[Tensor, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((t.dtype.name, t.shape) for t in tensors)



def _guard_pass(metas: tuple[MetaTensor, ...], tensors: tuple[Tensor, ...]) -> bool:
    if len(metas) != len(tensors):
        return False
    return all(m.shape == t.shape and m.dtype == t.dtype for m, t in zip(metas, tensors))



def jit(trace: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Tensor]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Tensor]:
        cache: dict[tuple[Any, ...], CompiledKernel] = {}
        _ALL_CACHES.append(cache)

        @functools.wraps(fn)
        def wrapper(*args: Any) -> Tensor:
            tensors = tuple(ensure_tensor(a) for a in args)
            key = (fn.__qualname__, _meta_key(tensors))

            compiled = cache.get(key)
            if compiled is not None and _guard_pass(compiled.input_metas, tensors):
                _GLOBAL_STATS["hits"] += 1
                wrapper.last_call_cache_hit = True
                out = compiled.program.run([t.data for t in tensors])
                return Tensor(out)

            _GLOBAL_STATS["misses"] += 1
            wrapper.last_call_cache_hit = False

            tracer = Tracer(trace_enabled=trace)
            proxy_args = [tracer.placeholder(t.meta) for t in tensors]
            traced_out = fn(*proxy_args)
            if not isinstance(traced_out, ProxyTensor):
                raise TypeError("demo jit only supports single Tensor output")
            tracer.finalize(traced_out)

            graph = tracer.graph
            graph.verify()
            mlir = emit_mlir(graph)
            program = parse_mlir(mlir)

            compiled = CompiledKernel(
                input_metas=tuple(t.meta for t in tensors),
                output_meta=traced_out.meta,
                graph=graph,
                mlir=mlir,
                program=program,
            )
            cache[key] = compiled

            wrapper.last_graph = graph
            wrapper.last_mlir = mlir

            out = compiled.program.run([t.data for t in tensors])
            return Tensor(out)

        wrapper.last_graph = None
        wrapper.last_mlir = ""
        wrapper.last_call_cache_hit = False
        wrapper.eager = fn
        wrapper._jit_cache = cache
        return wrapper

    return decorator



def jit_cache_stats() -> dict[str, int]:
    return {
        "hits": _GLOBAL_STATS["hits"],
        "misses": _GLOBAL_STATS["misses"],
        "versions": sum(len(c) for c in _ALL_CACHES),
    }
