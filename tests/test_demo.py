from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
KERNEL_DIR = TESTS_DIR / "kernel_cases"
EXPECTED_DIR = TESTS_DIR / "expected_outputs"


def _load_module(path: Path) -> ModuleType:
    module_name = f"tests.kernel_cases.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load kernel module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_cases() -> list[tuple[str, ModuleType, Path]]:
    cases: list[tuple[str, ModuleType, Path]] = []
    for kernel_path in sorted(KERNEL_DIR.glob("*.py")):
        if kernel_path.name.startswith("_"):
            continue
        module = _load_module(kernel_path)
        case_name = getattr(module, "CASE_NAME", kernel_path.stem)
        expected_path = EXPECTED_DIR / f"{case_name}.json"
        if not expected_path.exists():
            raise FileNotFoundError(f"missing expected output for case {case_name}: {expected_path}")
        cases.append((case_name, module, expected_path))
    if not cases:
        raise RuntimeError("no kernel cases found")
    return cases


def _load_expected(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    expected = np.asarray(payload["values"], dtype=np.float32)
    declared_shape = tuple(int(d) for d in payload["shape"])
    if expected.shape != declared_shape:
        raise ValueError(f"shape mismatch in expected file {path}: {expected.shape} vs {declared_shape}")
    return expected


CASES = _discover_cases()


@pytest.mark.parametrize(
    "case_name,module,expected_path",
    [pytest.param(name, mod, exp_path, id=name) for name, mod, exp_path in CASES],
)
def test_practical_kernels_dynamic(case_name: str, module: ModuleType, expected_path: Path) -> None:
    # Each case module provides:
    # - get_numpy_inputs()
    # - kernel(...): jitted DSL kernel
    # - eager_numpy(inputs)
    # - to_tensor_args(inputs)
    inputs = module.get_numpy_inputs()
    args = module.to_tensor_args(inputs)

    out_first = module.kernel(*args)
    assert module.kernel.last_call_cache_hit is False
    assert module.kernel.last_mlir

    out_second = module.kernel(*args)
    assert module.kernel.last_call_cache_hit is True

    eager = np.asarray(module.eager_numpy(inputs), dtype=np.float32)
    expected = _load_expected(expected_path)

    rtol = float(getattr(module, "RTOL", 1e-5))
    atol = float(getattr(module, "ATOL", 1e-5))

    np.testing.assert_allclose(out_first.numpy(), eager, rtol=rtol, atol=atol)
    np.testing.assert_allclose(out_first.numpy(), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(out_second.numpy(), expected, rtol=rtol, atol=atol)


def test_expected_files_match_discovered_cases() -> None:
    case_names = {name for name, _, _ in CASES}
    expected_names = {p.stem for p in EXPECTED_DIR.glob("*.json")}
    assert case_names == expected_names
