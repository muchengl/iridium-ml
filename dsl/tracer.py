from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .debug import trace
from .graph import Graph, Op, Value
from .meta import MetaTensor, broadcast_shape, promote_dtype
from .tensor import Tensor



def _as_proxy(x: Any) -> "ProxyTensor":
    if not isinstance(x, ProxyTensor):
        raise TypeError("traced ops only accept ProxyTensor inputs")
    return x



def _normalize_axis(axis: int, rank: int) -> int:
    a = int(axis)
    if a < 0:
        a += rank
    if a < 0 or a >= rank:
        raise ValueError(f"axis {axis} out of range for rank {rank}")
    return a


@dataclass
class ProxyTensor:
    tracer: "Tracer"
    value: Value

    @property
    def meta(self) -> MetaTensor:
        return self.value.meta

    def __add__(self, other: Any) -> "ProxyTensor":
        return self.tracer.emit(Op.ADD, [self, _as_proxy(other)], {})

    def __radd__(self, other: Any) -> "ProxyTensor":
        return _as_proxy(other).__add__(self)

    def __mul__(self, other: Any) -> "ProxyTensor":
        return self.tracer.emit(Op.MUL, [self, _as_proxy(other)], {})

    def __rmul__(self, other: Any) -> "ProxyTensor":
        return _as_proxy(other).__mul__(self)

    def __matmul__(self, other: Any) -> "ProxyTensor":
        return self.tracer.emit(Op.MATMUL, [self, _as_proxy(other)], {})

    def relu(self) -> "ProxyTensor":
        return self.tracer.emit(Op.RELU, [self], {})

    def mean(self, axis: int, keepdim: bool = False) -> "ProxyTensor":
        attrs = {"axis": int(axis), "keepdim": bool(keepdim)}
        return self.tracer.emit(Op.REDUCE_MEAN, [self], attrs)


class Tracer:
    def __init__(self, trace_enabled: bool = False):
        self.graph = Graph()
        self.trace_enabled = trace_enabled

    def placeholder(self, meta: MetaTensor) -> ProxyTensor:
        v = self.graph.add_input(meta)
        trace(f"input %{v.id}: {meta.short()}", self.trace_enabled)
        return ProxyTensor(tracer=self, value=v)

    def emit(self, op: Op, inputs: list[ProxyTensor], attrs: dict[str, Any]) -> ProxyTensor:
        input_values = [i.value for i in inputs]
        out_meta = self._infer(op, [i.meta for i in inputs], attrs)
        out = self.graph.add_node(op=op, inputs=input_values, attrs=attrs, output_meta=out_meta)
        in_ids = ", ".join(f"%{v.id}" for v in input_values)
        trace(f"{op.value}({in_ids}) -> %{out.id} : {out_meta.short()}", self.trace_enabled)
        return ProxyTensor(tracer=self, value=out)

    def finalize(self, outputs: ProxyTensor | tuple[ProxyTensor, ...] | list[ProxyTensor]) -> None:
        if isinstance(outputs, ProxyTensor):
            out_list = [outputs.value]
        else:
            out_list = [_as_proxy(x).value for x in outputs]
        self.graph.set_outputs(out_list)

    def _infer(self, op: Op, inputs: list[MetaTensor], attrs: dict[str, Any]) -> MetaTensor:
        if op in (Op.ADD, Op.MUL):
            if len(inputs) != 2:
                raise ValueError(f"{op.value} expects 2 inputs")
            shape = broadcast_shape(inputs[0].shape, inputs[1].shape)
            dtype = promote_dtype(inputs[0].dtype, inputs[1].dtype)
            return MetaTensor(shape=shape, dtype=dtype)
        if op == Op.RELU:
            if len(inputs) != 1:
                raise ValueError("relu expects 1 input")
            return inputs[0]
        if op == Op.MATMUL:
            if len(inputs) != 2:
                raise ValueError("matmul expects 2 inputs")
            a, b = inputs
            if len(a.shape) != 2 or len(b.shape) != 2:
                raise ValueError("matmul only supports 2D tensors")
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
            dtype = promote_dtype(a.dtype, b.dtype)
            return MetaTensor(shape=(a.shape[0], b.shape[1]), dtype=dtype)
        if op == Op.REDUCE_MEAN:
            if len(inputs) != 1:
                raise ValueError("reduce_mean expects 1 input")
            x = inputs[0]
            axis = _normalize_axis(int(attrs["axis"]), len(x.shape))
            keepdim = bool(attrs.get("keepdim", False))
            if keepdim:
                out_shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape))
            else:
                out_shape = tuple(d for i, d in enumerate(x.shape) if i != axis)
            return MetaTensor(shape=out_shape, dtype=x.dtype)
        raise NotImplementedError(f"unsupported op for infer: {op}")



def relu(x: Tensor | ProxyTensor) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.relu()
    if isinstance(x, Tensor):
        return x.relu()
    arr = np.asarray(x)
    return Tensor(np.maximum(arr, 0))



def mean(x: Tensor | ProxyTensor, axis: int, keepdim: bool = False) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.mean(axis=axis, keepdim=keepdim)
    if isinstance(x, Tensor):
        return x.mean(axis=axis, keepdim=keepdim)
    arr = np.asarray(x)
    return Tensor(np.mean(arr, axis=axis, keepdims=keepdim))
