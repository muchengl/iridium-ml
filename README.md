# Tracing-based ML DSL Demo

This repository is a minimal runnable tracing-style ML DSL demo. Users write kernels in Python, and the first call triggers Proxy tracing to build an SSA graph IR, emits a tiny textual MLIR module, then executes it with a custom MLIR-subset interpreter backed by NumPy. JIT cache and shape/dtype guards are used to reuse compiled variants.

## Run

```bash
python examples/demo_frontend.py
python examples/demo_mlir.py
pytest -q
```

## Dynamic Test Cases

Tests are organized as dynamically loaded practical kernel cases:

- Kernel definitions: `tests/kernel_cases/*.py`
- Expected outputs: `tests/expected_outputs/*.json`
- Test entry: `tests/test_demo.py`

Current practical kernels:

- `featurewise_affine_relu`: per-feature affine transform + ReLU
- `mlp_block`: two-layer MLP block
- `mean_pool_head`: mean pooling + linear head + ReLU
- `gated_fusion_block`: gated/value branches with elementwise fusion

Each `kernel_cases/*.py` module must expose:

- `CASE_NAME`
- `get_numpy_inputs()`
- `kernel(...)` (decorated with `@jit`)
- `eager_numpy(inputs)`
- `to_tensor_args(inputs)`

`tests/test_demo.py` automatically:

- scans kernel cases
- matches the expected JSON with the same case name
- checks first-run cache miss and second-run cache hit
- validates DSL outputs against eager and expected values

## Key Concepts

- Proxy tracing: `ProxyTensor` intercepts `+/*/@/relu/mean` during function execution and records `Node`s.
- Graph IR: `Graph(inputs, nodes, outputs)` where each `Value` is SSA and carries `MetaTensor(shape, dtype)`.
- MLIR emit: `dsl/mlir_emit.py` converts the graph into textual IR with `module + func.func + toy.*`.
- MLIR interp: `dsl/mlir_interp.py` parses and interprets that MLIR subset.
- Cache & guards: `@jit` uses `(fn_qualname, dtypes, shapes)` as the key and validates reuse with shape/dtype guards.

## Example graph.pretty output

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

## Example MLIR output

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
