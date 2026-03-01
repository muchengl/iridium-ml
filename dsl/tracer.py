from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .debug import trace
from .graph import Graph, Op, Value
from .meta import MetaTensor, broadcast_shape, promote_dtype
from .tensor import Tensor, _infer_reshape, _normalize_flatten_dims, _pair


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


def _reshape_tuple(shape: tuple[int, ...]) -> dict[str, int]:
    attrs: dict[str, int] = {"shape_rank": len(shape)}
    for i, d in enumerate(shape):
        attrs[f"shape_{i}"] = int(d)
    return attrs


def _shape_from_attrs(attrs: dict[str, Any]) -> tuple[int, ...]:
    rank = int(attrs["shape_rank"])
    return tuple(int(attrs[f"shape_{i}"]) for i in range(rank))


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

    def tanh(self) -> "ProxyTensor":
        return self.tracer.emit(Op.TANH, [self], {})

    def mean(self, axis: int, keepdim: bool = False) -> "ProxyTensor":
        attrs = {"axis": int(axis), "keepdim": bool(keepdim)}
        return self.tracer.emit(Op.REDUCE_MEAN, [self], attrs)

    def conv2d(
        self,
        weight: "ProxyTensor",
        bias: "ProxyTensor | None" = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> "ProxyTensor":
        stride_hw = _pair(stride, "stride", min_value=1)
        pad_hw = _pair(padding, "padding", min_value=0)
        attrs = {
            "stride_h": stride_hw[0],
            "stride_w": stride_hw[1],
            "pad_h": pad_hw[0],
            "pad_w": pad_hw[1],
        }
        ins = [self, _as_proxy(weight)]
        if bias is not None:
            ins.append(_as_proxy(bias))
        return self.tracer.emit(Op.CONV2D, ins, attrs)

    def avg_pool2d(
        self,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
    ) -> "ProxyTensor":
        kernel_hw = _pair(kernel, "kernel", min_value=1)
        stride_hw = kernel_hw if stride is None else _pair(stride, "stride", min_value=1)
        attrs = {
            "kernel_h": kernel_hw[0],
            "kernel_w": kernel_hw[1],
            "stride_h": stride_hw[0],
            "stride_w": stride_hw[1],
        }
        return self.tracer.emit(Op.AVG_POOL2D, [self], attrs)

    def max_pool2d(
        self,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
    ) -> "ProxyTensor":
        kernel_hw = _pair(kernel, "kernel", min_value=1)
        stride_hw = kernel_hw if stride is None else _pair(stride, "stride", min_value=1)
        attrs = {
            "kernel_h": kernel_hw[0],
            "kernel_w": kernel_hw[1],
            "stride_h": stride_hw[0],
            "stride_w": stride_hw[1],
        }
        return self.tracer.emit(Op.MAX_POOL2D, [self], attrs)

    def flatten(self, start_dim: int = 1, end_dim: int = -1) -> "ProxyTensor":
        attrs = {"start_dim": int(start_dim), "end_dim": int(end_dim)}
        return self.tracer.emit(Op.FLATTEN, [self], attrs)

    def reshape(self, *shape: int | tuple[int, ...]) -> "ProxyTensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            target_shape = tuple(int(d) for d in shape[0])
        else:
            target_shape = tuple(int(d) for d in shape)
        return self.tracer.emit(Op.RESHAPE, [self], _reshape_tuple(target_shape))


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
        if op in (Op.RELU, Op.TANH):
            if len(inputs) != 1:
                raise ValueError(f"{op.value} expects 1 input")
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
        if op == Op.CONV2D:
            if len(inputs) not in (2, 3):
                raise ValueError("conv2d expects 2 or 3 inputs")
            x = inputs[0]
            w = inputs[1]
            if len(x.shape) != 4 or len(w.shape) != 4:
                raise ValueError("conv2d expects x and w to be rank-4 NCHW tensors")
            n, c_in, h, w_in = x.shape
            c_out, c_w, kh, kw = w.shape
            if c_in != c_w:
                raise ValueError(f"conv2d channel mismatch: {c_in} vs {c_w}")
            sh = int(attrs["stride_h"])
            sw = int(attrs["stride_w"])
            ph = int(attrs["pad_h"])
            pw = int(attrs["pad_w"])
            h_out = (h + 2 * ph - kh) // sh + 1
            w_out = (w_in + 2 * pw - kw) // sw + 1
            if h_out <= 0 or w_out <= 0:
                raise ValueError(f"conv2d produced invalid output shape {(h_out, w_out)}")

            out_shape = (n, c_out, h_out, w_out)
            dtype = promote_dtype(x.dtype, w.dtype)
            if len(inputs) == 3:
                b = inputs[2]
                if len(b.shape) == 1:
                    if b.shape[0] != c_out:
                        raise ValueError(
                            f"conv2d bias size mismatch: expected {c_out}, got {b.shape[0]}"
                        )
                elif len(b.shape) == 4:
                    broadcast_shape(out_shape, b.shape)
                else:
                    raise ValueError("conv2d bias must be rank-1 or rank-4")
                dtype = promote_dtype(dtype, b.dtype)
            return MetaTensor(shape=out_shape, dtype=dtype)
        if op in (Op.AVG_POOL2D, Op.MAX_POOL2D):
            if len(inputs) != 1:
                raise ValueError(f"{op.value} expects 1 input")
            x = inputs[0]
            if len(x.shape) != 4:
                raise ValueError(f"{op.value} expects rank-4 NCHW tensor")
            n, c, h, w = x.shape
            kh = int(attrs["kernel_h"])
            kw = int(attrs["kernel_w"])
            sh = int(attrs["stride_h"])
            sw = int(attrs["stride_w"])
            h_out = (h - kh) // sh + 1
            w_out = (w - kw) // sw + 1
            if h_out <= 0 or w_out <= 0:
                raise ValueError(f"{op.value} produced invalid output shape {(h_out, w_out)}")
            return MetaTensor(shape=(n, c, h_out, w_out), dtype=x.dtype)
        if op == Op.FLATTEN:
            if len(inputs) != 1:
                raise ValueError("flatten expects 1 input")
            x = inputs[0]
            s, e = _normalize_flatten_dims(int(attrs["start_dim"]), int(attrs["end_dim"]), len(x.shape))
            pre = x.shape[:s]
            mid = int(np.prod(x.shape[s : e + 1], dtype=np.int64))
            post = x.shape[e + 1 :]
            return MetaTensor(shape=pre + (mid,) + post, dtype=x.dtype)
        if op == Op.RESHAPE:
            if len(inputs) != 1:
                raise ValueError("reshape expects 1 input")
            x = inputs[0]
            target = _shape_from_attrs(attrs)
            out_shape = _infer_reshape(x.shape, target)
            return MetaTensor(shape=out_shape, dtype=x.dtype)
        raise NotImplementedError(f"unsupported op for infer: {op}")


def relu(x: Tensor | ProxyTensor) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.relu()
    if isinstance(x, Tensor):
        return x.relu()
    arr = np.asarray(x)
    return Tensor(np.maximum(arr, 0))


def tanh(x: Tensor | ProxyTensor) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.tanh()
    if isinstance(x, Tensor):
        return x.tanh()
    arr = np.asarray(x)
    return Tensor(np.tanh(arr))


def mean(x: Tensor | ProxyTensor, axis: int, keepdim: bool = False) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.mean(axis=axis, keepdim=keepdim)
    if isinstance(x, Tensor):
        return x.mean(axis=axis, keepdim=keepdim)
    arr = np.asarray(x)
    return Tensor(np.mean(arr, axis=axis, keepdims=keepdim))


def conv2d(
    x: Tensor | ProxyTensor,
    weight: Tensor | ProxyTensor,
    bias: Tensor | ProxyTensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.conv2d(_as_proxy(weight), None if bias is None else _as_proxy(bias), stride, padding)
    if isinstance(x, Tensor):
        return x.conv2d(weight, bias=bias, stride=stride, padding=padding)
    return Tensor(np.asarray(x)).conv2d(weight, bias=bias, stride=stride, padding=padding)


def avg_pool2d(
    x: Tensor | ProxyTensor,
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.avg_pool2d(kernel=kernel, stride=stride)
    if isinstance(x, Tensor):
        return x.avg_pool2d(kernel=kernel, stride=stride)
    return Tensor(np.asarray(x)).avg_pool2d(kernel=kernel, stride=stride)


def max_pool2d(
    x: Tensor | ProxyTensor,
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.max_pool2d(kernel=kernel, stride=stride)
    if isinstance(x, Tensor):
        return x.max_pool2d(kernel=kernel, stride=stride)
    return Tensor(np.asarray(x)).max_pool2d(kernel=kernel, stride=stride)


def flatten(x: Tensor | ProxyTensor, start_dim: int = 1, end_dim: int = -1) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.flatten(start_dim=start_dim, end_dim=end_dim)
    if isinstance(x, Tensor):
        return x.flatten(start_dim=start_dim, end_dim=end_dim)
    return Tensor(np.asarray(x)).flatten(start_dim=start_dim, end_dim=end_dim)


def reshape(x: Tensor | ProxyTensor, *shape: int | tuple[int, ...]) -> Tensor | ProxyTensor:
    if isinstance(x, ProxyTensor):
        return x.reshape(*shape)
    if isinstance(x, Tensor):
        return x.reshape(*shape)
    return Tensor(np.asarray(x)).reshape(*shape)
