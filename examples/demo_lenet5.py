from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsl import Tensor, avg_pool2d, conv2d, flatten, jit, tanh


@jit(trace=False)
def lenet5_like(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    wf: Tensor,
    bf: Tensor,
) -> Tensor:
    h1 = tanh(conv2d(x, w1, b1, stride=1, padding=0))
    h1 = avg_pool2d(h1, kernel=2, stride=2)
    h2 = tanh(conv2d(h1, w2, b2, stride=1, padding=0))
    h2 = flatten(h2, start_dim=1)
    return h2 @ wf + bf


def main() -> None:
    rng = np.random.default_rng(7)

    x = Tensor(rng.normal(size=(1, 1, 8, 8)).astype(np.float32))
    w1 = Tensor(rng.normal(size=(2, 1, 3, 3)).astype(np.float32))
    b1 = Tensor(rng.normal(size=(2,)).astype(np.float32))
    w2 = Tensor(rng.normal(size=(3, 2, 3, 3)).astype(np.float32))
    b2 = Tensor(rng.normal(size=(3,)).astype(np.float32))
    wf = Tensor(rng.normal(size=(3, 2)).astype(np.float32))
    bf = Tensor(rng.normal(size=(2,)).astype(np.float32))

    out = lenet5_like(x, w1, b1, w2, b2, wf, bf)

    print("output:")
    print(out.numpy())

    print("\ngraph.pretty():")
    print(lenet5_like.last_graph.pretty())

    print("\nMLIR:")
    print(lenet5_like.last_mlir)


if __name__ == "__main__":
    main()
