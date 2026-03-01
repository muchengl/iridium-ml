from __future__ import annotations

import numpy as np

from dsl import Tensor, jit, relu

# Case metadata used by the dynamic test loader.
CASE_NAME = "mlp_block"
INPUT_ORDER = ("x", "w1", "b1", "w2", "b2")
RTOL = 1e-5
ATOL = 1e-5



def get_numpy_inputs() -> dict[str, np.ndarray]:
    # A two-layer feed-forward block (very common in MLP and transformer FFN).
    return {
        "x": np.array([[0.4, -1.1, 0.7], [1.3, 0.2, -0.5]], dtype=np.float32),
        "w1": np.array(
            [[0.2, -0.4, 0.1, 0.6], [-0.7, 0.3, 0.5, -0.2], [0.9, -0.8, 0.2, 0.4]],
            dtype=np.float32,
        ),
        "b1": np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float32),
        "w2": np.array(
            [[0.3, -0.5], [0.6, 0.1], [-0.2, 0.4], [0.7, -0.3]],
            dtype=np.float32,
        ),
        "b2": np.array([0.05, -0.1], dtype=np.float32),
    }


@jit(trace=False)
def kernel(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
    hidden = relu(x @ w1 + b1)
    return hidden @ w2 + b2



def eager_numpy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    hidden = np.maximum(inputs["x"] @ inputs["w1"] + inputs["b1"], 0)
    return hidden @ inputs["w2"] + inputs["b2"]



def to_tensor_args(inputs: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
    return tuple(Tensor(inputs[name]) for name in INPUT_ORDER)
