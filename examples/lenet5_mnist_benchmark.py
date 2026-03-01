from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.request
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsl import Tensor, conv2d, flatten, jit, max_pool2d, relu

DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/icaros-usc/pyribs/master/tutorials/mnist/mnist_classifier.pth"
)


# JIT entry:
# - first call: trace Python kernel -> Graph IR -> MLIR text -> parse to interpreter program
# - later calls (same shape/dtype): cache hit and directly run parsed program
@jit(trace=False)
def lenet_kernel(
    x: Tensor,
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
    fc3_w: Tensor,
    fc3_b: Tensor,
) -> Tensor:
    h = relu(conv2d(x, conv1_w, conv1_b, stride=1, padding=0))
    h = max_pool2d(h, kernel=2, stride=2)
    h = relu(conv2d(h, conv2_w, conv2_b, stride=1, padding=0))
    h = max_pool2d(h, kernel=2, stride=2)
    h = flatten(h, start_dim=1)
    h = relu(h @ fc1_w + fc1_b)
    h = relu(h @ fc2_w + fc2_b)
    return h @ fc3_w + fc3_b


def _download(url: str, dst: pathlib.Path) -> pathlib.Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    dst.write_bytes(data)
    return dst


def _normalize_state_dict(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]
        elif all(hasattr(v, "shape") for v in ckpt.values()):
            sd = ckpt
        else:
            raise ValueError("checkpoint format not supported")
    else:
        raise ValueError("checkpoint must be a dict-like object")

    out: dict[str, Any] = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _pick_by_shape(items: dict[str, np.ndarray], shape: tuple[int, ...], used: set[str]) -> np.ndarray:
    for k, v in items.items():
        if k in used:
            continue
        if tuple(v.shape) == shape:
            used.add(k)
            return v
    raise KeyError(f"no parameter with shape {shape} found in checkpoint")


def load_lenet_weights(model_url: str, cache_dir: pathlib.Path) -> dict[str, Tensor]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("torch is required for loading the online checkpoint") from exc

    model_path = _download(model_url, cache_dir / "mnist_lenet5.pth")
    ckpt = torch.load(model_path, map_location="cpu")
    sd = _normalize_state_dict(ckpt)

    np_sd: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):
            arr = v.detach().cpu().numpy().astype(np.float32)
            np_sd[k] = arr

    used: set[str] = set()

    def take(*names: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        for n in names:
            if n in np_sd:
                used.add(n)
                return np_sd[n]
        if shape is not None:
            return _pick_by_shape(np_sd, shape, used)
        raise KeyError(f"missing parameter names: {names}")

    conv1_w = take("conv1.weight", "features.0.weight", shape=(6, 1, 5, 5))
    conv1_b = take("conv1.bias", "features.0.bias", shape=(6,))
    conv2_w = take("conv2.weight", "features.3.weight", shape=(16, 6, 5, 5))
    conv2_b = take("conv2.bias", "features.3.bias", shape=(16,))

    fc1_w_t = take("fc1.weight", "classifier.0.weight", shape=(120, 256))
    fc1_b = take("fc1.bias", "classifier.0.bias", shape=(120,))
    fc2_w_t = take("fc2.weight", "classifier.2.weight", shape=(84, 120))
    fc2_b = take("fc2.bias", "classifier.2.bias", shape=(84,))
    fc3_w_t = take("fc3.weight", "classifier.4.weight", shape=(10, 84))
    fc3_b = take("fc3.bias", "classifier.4.bias", shape=(10,))

    return {
        "conv1_w": Tensor(conv1_w),
        "conv1_b": Tensor(conv1_b),
        "conv2_w": Tensor(conv2_w),
        "conv2_b": Tensor(conv2_b),
        "fc1_w": Tensor(fc1_w_t.T.copy()),
        "fc1_b": Tensor(fc1_b),
        "fc2_w": Tensor(fc2_w_t.T.copy()),
        "fc2_b": Tensor(fc2_b),
        "fc3_w": Tensor(fc3_w_t.T.copy()),
        "fc3_b": Tensor(fc3_b),
    }


def load_mnist_batch(cache_dir: pathlib.Path, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("torchvision is required for downloading MNIST") from exc

    ds = MNIST(root=str(cache_dir), train=False, download=True)
    num = min(num_samples, len(ds))

    images: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(num):
        img, label = ds[i]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.1307) / 0.3081
        images.append(arr[None, :, :])
        labels.append(int(label))

    x = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def run(args: argparse.Namespace) -> None:
    cache_dir = pathlib.Path(args.cache_dir).resolve()

    weights = load_lenet_weights(args.model_url, cache_dir)
    x, y = load_mnist_batch(cache_dir, args.num_samples)

    x_tensor = Tensor(x)
    weight_args = (
        weights["conv1_w"],
        weights["conv1_b"],
        weights["conv2_w"],
        weights["conv2_b"],
        weights["fc1_w"],
        weights["fc1_b"],
        weights["fc2_w"],
        weights["fc2_b"],
        weights["fc3_w"],
        weights["fc3_b"],
    )

    # First JIT call: includes compile path (trace + MLIR emit/parse) and one interpreter execution.
    t0 = time.perf_counter()
    logits_first = lenet_kernel(x_tensor, *weight_args).numpy()
    first_ms = (time.perf_counter() - t0) * 1000.0

    pred_first = np.argmax(logits_first, axis=1)
    acc_first = float(np.mean(pred_first == y))

    # Steady-state calls: expected to be cache hits, still executed by the MLIR interpreter backend.
    times: list[float] = []
    for _ in range(args.iters):
        ts = time.perf_counter()
        logits = lenet_kernel(x_tensor, *weight_args).numpy()
        times.append(time.perf_counter() - ts)

    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))

    avg_ms = float(np.mean(times) * 1000.0)
    throughput = float(args.num_samples / np.mean(times))

    print("LeNet-5 MNIST benchmark (DSL JIT + toy MLIR interpreter)")
    print(f"model_url: {args.model_url}")
    print(f"samples: {args.num_samples}")
    print(f"first run (compile + run): {first_ms:.2f} ms, acc={acc_first:.4f}")
    print(f"steady avg latency: {avg_ms:.2f} ms")
    print(f"steady throughput: {throughput:.2f} samples/s")
    print(f"steady accuracy: {acc:.4f}")
    # True means this call used JIT cache (no retrace/re-emit); execution remains interpreter-based.
    print(f"cache hit on last call: {lenet_kernel.last_call_cache_hit}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LeNet-5 checkpoint + MNIST and benchmark DSL inference")
    parser.add_argument("--model-url", type=str, default=DEFAULT_MODEL_URL, help="Remote .pth checkpoint URL")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(ROOT / ".cache" / "lenet5"),
        help="Directory for model and dataset cache",
    )
    parser.add_argument("--num-samples", type=int, default=256, help="Number of test samples")
    parser.add_argument("--iters", type=int, default=5, help="Steady-state benchmark iterations")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
