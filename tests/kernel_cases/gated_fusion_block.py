from __future__ import annotations

import numpy as np

from dsl import Tensor, jit, relu

# Case metadata used by the dynamic test loader.
CASE_NAME = "gated_fusion_block"
INPUT_ORDER = ("x", "wa", "ba", "wb", "bb", "scale")
RTOL = 1e-5
ATOL = 1e-5



def get_numpy_inputs() -> dict[str, np.ndarray]:
    # A compact gated fusion pattern:
    # gate = relu(x @ wa + ba)
    # value = x @ wb + bb
    # out = gate * value * scale
    return {
        "x": np.array([[0.5, -0.2, 1.1], [-1.5, 0.7, 0.3]], dtype=np.float32),
        "wa": np.array(
            [[0.4, -0.1, 0.2], [0.3, 0.5, -0.6], [0.7, -0.4, 0.8]],
            dtype=np.float32,
        ),
        "ba": np.array([0.1, 0.0, -0.2], dtype=np.float32),
        "wb": np.array(
            [[-0.3, 0.2, 0.9], [0.6, -0.7, 0.1], [0.4, 0.5, -0.2]],
            dtype=np.float32,
        ),
        "bb": np.array([0.05, -0.1, 0.2], dtype=np.float32),
        "scale": np.array([1.0, 0.5, 1.5], dtype=np.float32),
    }


@jit(trace=False)
def kernel(x: Tensor, wa: Tensor, ba: Tensor, wb: Tensor, bb: Tensor, scale: Tensor) -> Tensor:
    gate = relu(x @ wa + ba)
    value = x @ wb + bb
    return gate * value * scale



def eager_numpy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    gate = np.maximum(inputs["x"] @ inputs["wa"] + inputs["ba"], 0)
    value = inputs["x"] @ inputs["wb"] + inputs["bb"]
    return gate * value * inputs["scale"]



def to_tensor_args(inputs: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
    return tuple(Tensor(inputs[name]) for name in INPUT_ORDER)
