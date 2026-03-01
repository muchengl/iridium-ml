from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsl import Tensor, jit, relu
from dsl.mlir_interp import parse_mlir


@jit(trace=False)
def tiny_kernel(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return relu(x @ w + b)



def main() -> None:
    rng = np.random.default_rng(1)
    x = Tensor(rng.normal(size=(2, 3)).astype(np.float32))
    w = Tensor(rng.normal(size=(3, 2)).astype(np.float32))
    b = Tensor(rng.normal(size=(2,)).astype(np.float32))

    out = tiny_kernel(x, w, b)
    mlir_text = tiny_kernel.last_mlir
    print("MLIR module:\n")
    print(mlir_text)

    program = parse_mlir(mlir_text)
    out_interp = program.run([x.numpy(), w.numpy(), b.numpy()])

    print("\ninterpreter output:")
    print(out_interp)
    print("\nmax abs diff:", float(np.max(np.abs(out_interp - out.numpy()))))


if __name__ == "__main__":
    main()
