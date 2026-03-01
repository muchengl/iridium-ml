from __future__ import annotations

import numpy as np

from dsl import Tensor, jit, relu

# Case metadata used by the dynamic test loader.
CASE_NAME = "featurewise_affine_relu"
INPUT_ORDER = ("x", "gamma", "beta")
RTOL = 1e-5
ATOL = 1e-5



def get_numpy_inputs() -> dict[str, np.ndarray]:
    # This pattern is common in inference stacks: feature-wise affine transform
    # followed by a non-linearity.
    return {
        "x": np.array([[1.2, -0.7, 0.3, -2.0], [0.1, 0.9, -1.3, 2.2]], dtype=np.float32),
        "gamma": np.array([0.5, 2.0, -1.5, 0.25], dtype=np.float32),
        "beta": np.array([0.1, -0.2, 0.0, 0.3], dtype=np.float32),
    }


@jit(trace=False)
def kernel(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    return relu(x * gamma + beta)



def eager_numpy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(inputs["x"] * inputs["gamma"] + inputs["beta"], 0)



def to_tensor_args(inputs: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
    return tuple(Tensor(inputs[name]) for name in INPUT_ORDER)
