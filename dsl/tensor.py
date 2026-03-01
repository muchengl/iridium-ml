from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .meta import MetaTensor



def _to_array(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x)


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

    def mean(self, axis: int, keepdim: bool = False) -> "Tensor":
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdim))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name}, data={self.data!r})"



def ensure_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)
