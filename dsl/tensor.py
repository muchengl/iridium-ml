from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .meta import MetaTensor



def _to_array(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x)


def _pair(v: int | tuple[int, int], name: str, min_value: int = 1) -> tuple[int, int]:
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(f"{name} must have 2 values, got {v}")
        out = (int(v[0]), int(v[1]))
    else:
        i = int(v)
        out = (i, i)
    if out[0] < min_value or out[1] < min_value:
        if min_value == 0:
            raise ValueError(f"{name} must be non-negative, got {v}")
        raise ValueError(f"{name} must be positive, got {v}")
    return out


def _normalize_flatten_dims(start_dim: int, end_dim: int, rank: int) -> tuple[int, int]:
    s = int(start_dim)
    e = int(end_dim)
    if s < 0:
        s += rank
    if e < 0:
        e += rank
    if s < 0 or s >= rank:
        raise ValueError(f"start_dim {start_dim} out of range for rank {rank}")
    if e < 0 or e >= rank:
        raise ValueError(f"end_dim {end_dim} out of range for rank {rank}")
    if s > e:
        raise ValueError(f"flatten start_dim {s} cannot be greater than end_dim {e}")
    return s, e


def _infer_reshape(shape: tuple[int, ...], new_shape: tuple[int, ...]) -> tuple[int, ...]:
    out = [int(d) for d in new_shape]
    neg_idx = [i for i, d in enumerate(out) if d == -1]
    if len(neg_idx) > 1:
        raise ValueError("reshape only supports a single -1 dimension")
    if any(d == 0 or d < -1 for d in out):
        raise ValueError(f"invalid reshape target shape {new_shape}")

    in_elems = int(np.prod(shape))
    if neg_idx:
        known = int(np.prod([d for d in out if d != -1], dtype=np.int64))
        if known == 0 or in_elems % known != 0:
            raise ValueError(f"cannot infer reshape target {new_shape} from input shape {shape}")
        out[neg_idx[0]] = in_elems // known
    out_elems = int(np.prod(out, dtype=np.int64))
    if out_elems != in_elems:
        raise ValueError(f"reshape element mismatch: {shape} -> {tuple(out)}")
    return tuple(out)


def _conv2d_nchw(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    if x.ndim != 4 or w.ndim != 4:
        raise ValueError("conv2d expects x and w to be 4D NCHW tensors")
    n, c_in, h, ww = x.shape
    c_out, c_w, kh, kw = w.shape
    if c_in != c_w:
        raise ValueError(f"conv2d channel mismatch: x has {c_in}, w has {c_w}")

    sh, sw = stride
    ph, pw = padding
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (ww + 2 * pw - kw) // sw + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"conv2d produced non-positive output shape {(h_out, w_out)} for input {(h, ww)}"
        )

    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    out_dtype = np.result_type(x.dtype, w.dtype, b.dtype if b is not None else x.dtype)
    out = np.zeros((n, c_out, h_out, w_out), dtype=out_dtype)

    for ni in range(n):
        for co in range(c_out):
            for ho in range(h_out):
                hs = ho * sh
                for wo_i in range(w_out):
                    ws = wo_i * sw
                    window = x_pad[ni, :, hs : hs + kh, ws : ws + kw]
                    out[ni, co, ho, wo_i] = np.sum(window * w[co])
    if b is not None:
        if b.ndim == 1:
            if b.shape[0] != c_out:
                raise ValueError(f"conv2d bias size mismatch: expected {c_out}, got {b.shape[0]}")
            out += b.reshape(1, c_out, 1, 1)
        elif b.ndim == 4:
            out += b
        else:
            raise ValueError("conv2d bias must be rank-1 [C_out] or rank-4 broadcastable tensor")
    return out


def _pool2d_nchw(
    x: np.ndarray,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    mode: str,
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"{mode}_pool2d expects a 4D NCHW tensor")
    n, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = stride
    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"{mode}_pool2d produced non-positive output shape {(h_out, w_out)} for input {(h, w)}"
        )

    out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)
    for ni in range(n):
        for ci in range(c):
            for ho in range(h_out):
                hs = ho * sh
                for wo_i in range(w_out):
                    ws = wo_i * sw
                    window = x[ni, ci, hs : hs + kh, ws : ws + kw]
                    if mode == "avg":
                        out[ni, ci, ho, wo_i] = np.mean(window)
                    elif mode == "max":
                        out[ni, ci, ho, wo_i] = np.max(window)
                    else:
                        raise ValueError(f"unsupported pool mode {mode}")
    return out


@dataclass
class Tensor:
    data: np.ndarray

    def __init__(self, data: Any):
        self.data = np.asarray(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def meta(self) -> MetaTensor:
        return MetaTensor.from_array(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def __array__(self) -> np.ndarray:
        return self.data

    def __add__(self, other: Any) -> "Tensor":
        return Tensor(self.data + _to_array(other))

    def __radd__(self, other: Any) -> "Tensor":
        return Tensor(_to_array(other) + self.data)

    def __mul__(self, other: Any) -> "Tensor":
        return Tensor(self.data * _to_array(other))

    def __rmul__(self, other: Any) -> "Tensor":
        return Tensor(_to_array(other) * self.data)

    def __matmul__(self, other: Any) -> "Tensor":
        return Tensor(self.data @ _to_array(other))

    def relu(self) -> "Tensor":
        return Tensor(np.maximum(self.data, 0))

    def tanh(self) -> "Tensor":
        return Tensor(np.tanh(self.data))

    def mean(self, axis: int, keepdim: bool = False) -> "Tensor":
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdim))

    def conv2d(
        self,
        weight: Any,
        bias: Any | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> "Tensor":
        x = self.data
        w = _to_array(weight)
        b = _to_array(bias) if bias is not None else None
        stride_hw = _pair(stride, "stride", min_value=1)
        pad_hw = _pair(padding, "padding", min_value=0)
        return Tensor(_conv2d_nchw(x, w, b, stride_hw, pad_hw))

    def avg_pool2d(
        self,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
    ) -> "Tensor":
        k_hw = _pair(kernel, "kernel", min_value=1)
        s_hw = k_hw if stride is None else _pair(stride, "stride", min_value=1)
        return Tensor(_pool2d_nchw(self.data, k_hw, s_hw, mode="avg"))

    def max_pool2d(
        self,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
    ) -> "Tensor":
        k_hw = _pair(kernel, "kernel", min_value=1)
        s_hw = k_hw if stride is None else _pair(stride, "stride", min_value=1)
        return Tensor(_pool2d_nchw(self.data, k_hw, s_hw, mode="max"))

    def flatten(self, start_dim: int = 1, end_dim: int = -1) -> "Tensor":
        rank = self.data.ndim
        s, e = _normalize_flatten_dims(start_dim, end_dim, rank)
        pre = self.data.shape[:s]
        mid = int(np.prod(self.data.shape[s : e + 1], dtype=np.int64))
        post = self.data.shape[e + 1 :]
        return Tensor(self.data.reshape(pre + (mid,) + post))

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            target = tuple(int(d) for d in shape[0])
        else:
            target = tuple(int(d) for d in shape)
        out_shape = _infer_reshape(self.data.shape, target)
        return Tensor(self.data.reshape(out_shape))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name}, data={self.data!r})"



def ensure_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)
