from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ParsedOp:
    result: str
    op: str
    inputs: list[str]
    attrs: dict[str, Any]


@dataclass
class ParsedProgram:
    arg_names: list[str]
    ops: list[ParsedOp]
    ret_name: str

    def run(self, inputs: list[np.ndarray]) -> np.ndarray:
        if len(inputs) != len(self.arg_names):
            raise ValueError(f"expected {len(self.arg_names)} inputs, got {len(inputs)}")

        env: dict[str, np.ndarray] = {}
        for name, arr in zip(self.arg_names, inputs):
            env[name] = np.asarray(arr)

        for inst in self.ops:
            vals = [env[name] for name in inst.inputs]
            if inst.op == "toy.add":
                out = vals[0] + vals[1]
            elif inst.op == "toy.mul":
                out = vals[0] * vals[1]
            elif inst.op == "toy.relu":
                out = np.maximum(vals[0], 0)
            elif inst.op == "toy.matmul":
                if vals[0].ndim != 2 or vals[1].ndim != 2:
                    raise ValueError("toy.matmul only supports 2D")
                out = vals[0] @ vals[1]
            elif inst.op == "toy.reduce_mean":
                axis = int(inst.attrs["axis"])
                keepdim = bool(inst.attrs.get("keepdim", False))
                out = np.mean(vals[0], axis=axis, keepdims=keepdim)
            else:
                raise NotImplementedError(f"unsupported op in interpreter: {inst.op}")
            env[inst.result] = out

        if self.ret_name not in env:
            raise ValueError(f"return value {self.ret_name} was never defined")
        return env[self.ret_name]



def _parse_bool_or_int(s: str) -> Any:
    v = s.strip()
    if v == "true":
        return True
    if v == "false":
        return False
    return int(v)



def _parse_attrs(attr_text: str) -> dict[str, Any]:
    text = attr_text.strip()
    if not text:
        return {}
    out: dict[str, Any] = {}
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for part in parts:
        key, value = [x.strip() for x in part.split("=", 1)]
        out[key] = _parse_bool_or_int(value)
    return out



def _parse_signature(sig_line: str) -> list[str]:
    # func.func @main(%arg0: tensor<...>, %arg1: tensor<...>) -> tensor<...> {
    start = sig_line.index("(") + 1
    end = sig_line.index(")", start)
    args_str = sig_line[start:end].strip()
    if not args_str:
        return []
    chunks = [c.strip() for c in args_str.split(",") if c.strip()]
    names = [chunk.split(":", 1)[0].strip() for chunk in chunks]
    return names



def parse_mlir(text: str) -> ParsedProgram:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    sig_line = next((ln for ln in lines if ln.startswith("func.func @main(")), None)
    if sig_line is None:
        raise ValueError("could not find func.func @main signature")
    arg_names = _parse_signature(sig_line)

    in_body = False
    ops: list[ParsedOp] = []
    ret_name: str | None = None

    for ln in lines:
        if ln.startswith("func.func @main("):
            in_body = True
            continue
        if not in_body:
            continue
        if ln == "}":
            in_body = False
            continue
        if ln.startswith("return "):
            ret_name = ln.split()[1]
            continue
        if "=" not in ln:
            continue

        lhs, rhs = ln.split("=", 1)
        result = lhs.strip()

        rhs = rhs.strip()
        op_name, rest = rhs.split(" ", 1)
        before_type = rest.split(":", 1)[0].strip()

        attrs: dict[str, Any] = {}
        if "{" in before_type:
            inputs_text, attr_part = before_type.split("{", 1)
            attr_text = attr_part.rsplit("}", 1)[0]
            attrs = _parse_attrs(attr_text)
            inputs_text = inputs_text.strip()
        else:
            inputs_text = before_type

        inputs = [x.strip() for x in inputs_text.split(",") if x.strip()]
        ops.append(ParsedOp(result=result, op=op_name, inputs=inputs, attrs=attrs))

    if ret_name is None:
        raise ValueError("could not parse return from MLIR text")

    return ParsedProgram(arg_names=arg_names, ops=ops, ret_name=ret_name)



def run_mlir(text: str, inputs: list[np.ndarray]) -> np.ndarray:
    program = parse_mlir(text)
    return program.run(inputs)
