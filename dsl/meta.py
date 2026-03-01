from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Shape = tuple[int, ...]


_DTYPE_TO_MLIR = {
    np.dtype("float32"): "f32",
    np.dtype("float64"): "f64",
    np.dtype("int32"): "i32",
    np.dtype("int64"): "i64",
}



def normalize_shape(shape: Iterable[int]) -> Shape:
    out = tuple(int(d) for d in shape)
    for d in out:
        if d < 0:
            raise ValueError(f"shape must be static non-negative ints, got {shape}")
    return out



def normalize_dtype(dtype: np.dtype | str | type) -> np.dtype:
    return np.dtype(dtype)



def promote_dtype(a: np.dtype, b: np.dtype) -> np.dtype:
    return np.result_type(a, b)



def broadcast_shape(a: Shape, b: Shape) -> Shape:
    ra = list(reversed(a))
    rb = list(reversed(b))
    out: list[int] = []
    n = max(len(ra), len(rb))
    for i in range(n):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        elif da == db:
            out.append(da)
        else:
            raise ValueError(f"cannot broadcast shapes {a} and {b}")
    return tuple(reversed(out))


@dataclass(frozen=True)
class MetaTensor:
    shape: Shape
    dtype: np.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", normalize_shape(self.shape))
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "MetaTensor":
        return cls(shape=arr.shape, dtype=arr.dtype)

    def to_mlir_tensor_type(self) -> str:
        dtype_str = _DTYPE_TO_MLIR.get(self.dtype)
        if dtype_str is None:
            raise ValueError(f"unsupported dtype for MLIR emit: {self.dtype}")
        if self.shape:
            shape_str = "x".join(str(d) for d in self.shape)
            return f"tensor<{shape_str}x{dtype_str}>"
        return f"tensor<{dtype_str}>"

    def short(self) -> str:
        return f"shape={self.shape}, dtype={self.dtype.name}"
