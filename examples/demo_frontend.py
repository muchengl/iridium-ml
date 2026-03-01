from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsl import Tensor, jit, jit_cache_stats, mean, relu


@jit(trace=True)
def kernel_elementwise(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return relu(x * w + b)


@jit(trace=True)
def kernel_reduce(x: Tensor) -> Tensor:
    return mean(x, axis=1, keepdim=True)


@jit(trace=True)
def kernel_matmul(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return relu(x @ w + b)



def _max_diff(a: Tensor, b: Tensor) -> float:
    return float(np.max(np.abs(a.numpy() - b.numpy())))



def run_case(name: str, fn, args: tuple[Tensor, ...]) -> None:
    print(f"\n=== {name} ===")

    out1 = fn(*args)
    eager1 = fn.eager(*args)
    print("first run output:")
    print(out1.numpy())
    print(f"max |mlir - eager| = {_max_diff(out1, eager1):.6f}")

    if fn.last_graph is not None:
        print("\ngraph.pretty():")
        print(fn.last_graph.pretty())
    if fn.last_mlir:
        print("\nMLIR:")
        print(fn.last_mlir)

    out2 = fn(*args)
    eager2 = fn.eager(*args)
    print(f"\nsecond run cache_hit={fn.last_call_cache_hit}")
    print(f"max |mlir - eager| = {_max_diff(out2, eager2):.6f}")



def main() -> None:
    rng = np.random.default_rng(0)

    x = Tensor(rng.normal(size=(2, 3)).astype(np.float32))
    w = Tensor(rng.normal(size=(3,)).astype(np.float32))
    b = Tensor(rng.normal(size=(3,)).astype(np.float32))
    run_case("elementwise + broadcast", kernel_elementwise, (x, w, b))

    xr = Tensor(rng.normal(size=(2, 4)).astype(np.float32))
    run_case("reduction mean(axis=1, keepdim=True)", kernel_reduce, (xr,))

    xm = Tensor(rng.normal(size=(2, 3)).astype(np.float32))
    wm = Tensor(rng.normal(size=(3, 4)).astype(np.float32))
    bm = Tensor(rng.normal(size=(4,)).astype(np.float32))
    run_case("matmul + activation", kernel_matmul, (xm, wm, bm))

    print("\ncache stats:", jit_cache_stats())


if __name__ == "__main__":
    main()
