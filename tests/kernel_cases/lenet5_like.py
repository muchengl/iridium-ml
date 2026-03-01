from __future__ import annotations

import numpy as np

from dsl import Tensor, avg_pool2d, conv2d, flatten, jit, tanh

# Case metadata used by the dynamic test loader.
CASE_NAME = "lenet5_like"
INPUT_ORDER = ("x", "w1", "b1", "w2", "b2", "wf", "bf")
RTOL = 1e-5
ATOL = 1e-5


def get_numpy_inputs() -> dict[str, np.ndarray]:
    x = (np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8) / 32.0) - 1.0
    w1 = np.array(
        [
            [[[0.2, -0.1, 0.0], [0.1, 0.3, -0.2], [0.0, -0.2, 0.2]]],
            [[[-0.1, 0.2, 0.1], [0.0, -0.3, 0.2], [0.2, 0.1, -0.1]]],
        ],
        dtype=np.float32,
    )
    b1 = np.array([0.05, -0.02], dtype=np.float32)

    w2 = np.array(
        [
            [
                [[0.1, -0.2, 0.1], [0.0, 0.2, -0.1], [0.1, 0.0, 0.2]],
                [[-0.1, 0.1, 0.0], [0.2, -0.2, 0.1], [0.0, 0.1, -0.1]],
            ],
            [
                [[-0.2, 0.1, 0.2], [0.1, 0.0, -0.1], [0.2, -0.1, 0.0]],
                [[0.1, 0.2, -0.2], [0.0, -0.1, 0.2], [-0.1, 0.1, 0.0]],
            ],
            [
                [[0.05, -0.05, 0.1], [0.1, 0.0, -0.1], [0.0, 0.05, 0.0]],
                [[-0.1, 0.0, 0.1], [0.05, -0.05, 0.0], [0.1, 0.0, -0.1]],
            ],
        ],
        dtype=np.float32,
    )
    b2 = np.array([0.01, -0.03, 0.02], dtype=np.float32)

    wf = np.array(
        [[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]],
        dtype=np.float32,
    )
    bf = np.array([0.05, -0.1], dtype=np.float32)

    return {"x": x, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "wf": wf, "bf": bf}


@jit(trace=False)
def kernel(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, wf: Tensor, bf: Tensor) -> Tensor:
    h1 = tanh(conv2d(x, w1, b1, stride=1, padding=0))
    h1 = avg_pool2d(h1, kernel=2, stride=2)
    h2 = tanh(conv2d(h1, w2, b2, stride=1, padding=0))
    h2 = flatten(h2, start_dim=1)
    return h2 @ wf + bf


def eager_numpy(inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = Tensor(inputs["x"])
    w1 = Tensor(inputs["w1"])
    b1 = Tensor(inputs["b1"])
    w2 = Tensor(inputs["w2"])
    b2 = Tensor(inputs["b2"])
    wf = Tensor(inputs["wf"])
    bf = Tensor(inputs["bf"])

    h1 = tanh(conv2d(x, w1, b1, stride=1, padding=0))
    h1 = avg_pool2d(h1, kernel=2, stride=2)
    h2 = tanh(conv2d(h1, w2, b2, stride=1, padding=0))
    h2 = flatten(h2, start_dim=1)
    out = h2 @ wf + bf
    return out.numpy()


def to_tensor_args(inputs: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
    return tuple(Tensor(inputs[name]) for name in INPUT_ORDER)
