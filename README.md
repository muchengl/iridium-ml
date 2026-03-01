# Iridium

**iridium** is a compact, tracing-based ML compiler in Python 3.11.
It demonstrates the full forward path of a tiny compiler stack:
Python kernel -> Proxy tracing -> SSA Graph IR -> textual MLIR -> custom MLIR interpreter (NumPy backend).

This project is intentionally minimal, but the core architecture matches what a larger ML compiler/runtime system would need.

## Why This Project

`iridium` is designed as a practical skeleton for experimenting with ML system ideas:

- Frontend tracing with tensor metadata only (shape/dtype)
- Explicit graph IR with verification
- Debug-friendly graph introspection
- Deterministic textual IR generation (MLIR-like)
- End-to-end execution through an interpreter
- JIT cache with input guards

## Features

- Proxy-based tracing (`ProxyTensor`, `Tracer`)
- SSA Graph IR (`Value`, `Node`, `Graph`)
- Static shape and dtype inference
- Supported ops (MVP):
  - elementwise: `add`, `mul`
  - unary: `relu`, `tanh`
  - reduction: `reduce_mean(axis, keepdim)`
  - linear algebra: `matmul` (2D)
  - CNN: `conv2d` (NCHW), `avg_pool2d`, `max_pool2d`
  - shape: `flatten`, `reshape`
- Graph debug tools:
  - `graph.pretty()`
  - `graph.to_dot(path)`
  - trace logs via `@jit(trace=True)`
- MLIR textual emission (`toy.*` op subset)
- Lightweight MLIR-subset interpreter (no LLVM / official MLIR runtime)
- JIT compile cache keyed by function name + input shapes/dtypes

## Repository Layout

```text
repo/
  README.md
  pyproject.toml
  dsl/
    __init__.py
    tensor.py
    meta.py
    graph.py
    tracer.py
    jit.py
    debug.py
    mlir_emit.py
    mlir_interp.py
  examples/
    demo_frontend.py
    demo_mlir.py
    demo_lenet5.py
    lenet5_mnist_benchmark.py
  tests/
    test_demo.py
    kernel_cases/
    expected_outputs/
```

## Quick Start

```bash
python examples/demo_frontend.py
python examples/demo_mlir.py
python examples/demo_lenet5.py
pytest -q
```

## Frontend Kernel Style

`iridium` aims to feel close to NumPy/PyTorch-style authoring:

1. Elementwise + broadcast:
   `y = relu(x * w + b)`
2. Reduction:
   `y = mean(x, axis=1, keepdim=True)`
3. Matmul + activation:
   `y = relu(x @ w + b)`

These are implemented in `examples/demo_frontend.py` with `@jit`.

## Execution Flow

On first call to a jitted kernel:

1. Build placeholders from input metadata (`shape`, `dtype`)
2. Trace Python ops into Graph IR
3. Verify graph
4. Emit textual MLIR (`module` + `func.func` + `toy.*` ops)
5. Parse and run with the custom interpreter
6. Cache compiled program under guard key

On second call with the same input signature:

- Guard passes -> cache hit -> skip retracing/re-emission

## Dynamic Test Cases

Tests are dynamically discovered and validated:

- Kernel definitions: `tests/kernel_cases/*.py`
- Expected outputs: `tests/expected_outputs/*.json`
- Test entry: `tests/test_demo.py`

Current practical kernels:

- `featurewise_affine_relu`
- `mlp_block`
- `mean_pool_head`
- `gated_fusion_block`
- `lenet5_like`

Each kernel case module exposes:

- `CASE_NAME`
- `get_numpy_inputs()`
- `kernel(...)` (with `@jit`)
- `eager_numpy(inputs)`
- `to_tensor_args(inputs)`

The test runner checks:

- first-run compile/cache miss
- second-run cache hit
- numeric match vs eager + expected output files

## Example Graph IR (`graph.pretty()`)

```text
graph(
  input 0: %0 : shape=(2, 3), dtype=float32
  input 1: %1 : shape=(3,), dtype=float32
  input 2: %2 : shape=(3,), dtype=float32
) {
  %3 = mul(%0, %1) : shape=(2, 3), dtype=float32
  %4 = add(%3, %2) : shape=(2, 3), dtype=float32
  %5 = relu(%4) : shape=(2, 3), dtype=float32
  return %5
}
```

## Example Emitted MLIR

```mlir
module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3xf32> {
    %3 = toy.mul %arg0, %arg1 : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
    %4 = toy.add %3, %arg2 : tensor<2x3xf32>, tensor<3xf32> -> tensor<2x3xf32>
    %5 = toy.relu %4 : tensor<2x3xf32> -> tensor<2x3xf32>
    return %5 : tensor<2x3xf32>
  }
}
```

## Scope and Limits

- Forward-only demo (no autograd, no training)
- Static shapes only
- CPU only
- NumPy for runtime numerics
- Interpreter supports only the emitted MLIR subset

## Next Extensions

Natural directions if you want to evolve `iridium`:

1. Add constants and literal handling in IR/MLIR
2. Add more ops (`sub`, `div`, `exp`, `softmax`, `layernorm`)
3. Add simple optimization passes (constant folding, dead code elimination)
4. Add multi-output function support
5. Add a lower-level backend target beyond the interpreter
6. Add lower-level backend and scheduling passes for better LeNet/CNN performance

## LeNet-5 Support

LeNet-5 style forward is now supported in the DSL frontend and IR stack.

- Frontend API:
  - `conv2d(x, w, b=None, stride=1, padding=0)`
  - `avg_pool2d(x, kernel=2, stride=2)`
  - `max_pool2d(x, kernel=2, stride=2)` (for common modern LeNet variants)
  - `flatten(x, start_dim=1, end_dim=-1)`
  - `reshape(x, *shape)`
  - `tanh(x)` / `relu(x)`
- IR ops:
  - `conv2d`, `avg_pool2d`, `max_pool2d`, `flatten`, `reshape`, `tanh`
- Examples and tests:
  - `examples/demo_lenet5.py`
  - `tests/kernel_cases/lenet5_like.py`
  - `tests/expected_outputs/lenet5_like.json`

## Online LeNet-5 + MNIST Benchmark

`examples/lenet5_mnist_benchmark.py` downloads:

- a LeNet-style `.pth` checkpoint from the internet
- the MNIST test set

Then it runs inference with the DSL kernel and reports:

- first-run latency (compile + run)
- steady-state latency
- throughput
- top-1 accuracy on the sampled batch

Install optional deps first:

```bash
pip install torch torchvision
```

Run benchmark:

```bash
python examples/lenet5_mnist_benchmark.py --num-samples 256 --iters 5
```
