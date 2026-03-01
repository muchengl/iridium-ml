from __future__ import annotations

import numpy as np

from dsl import Tensor, jit, mean, relu

# Case metadata used by the dynamic test loader.
CASE_NAME = "mean_pool_head"
INPUT_ORDER = ("x", "w", "b")
RTOL = 1e-5
ATOL = 1e-5



def get_numpy_inputs() -> dict[str, np.ndarray]:
    # Sequence mean pooling followed by a tiny linear head.
    # x: [batch, seq_len], pooled: [batch, 1], out: [batch, 2]
    return {
        "x": np.array([[1.0, 0.0, 2.0], [-1.0, 3.0, 1.0]], dtype=np.float32),
        "w": np.array([[0.8, -0.6]], dtype=np.float32),
        "b": np.array([0.2, 0.1], dtype=np.float32),
    }


@jit(trace=False)
def kernel(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    pooled = mean(x, axis=1, keepdim=True)
    return relu(pooled @ w + b)



def eager_numpy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    pooled = np.mean(inputs["x"], axis=1, keepdims=True)
    return np.maximum(pooled @ inputs["w"] + inputs["b"], 0)



def to_tensor_args(inputs: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
    return tuple(Tensor(inputs[name]) for name in INPUT_ORDER)
