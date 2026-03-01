from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .meta import MetaTensor


class Op(str, Enum):
    ADD = "add"
    MUL = "mul"
    RELU = "relu"
    REDUCE_MEAN = "reduce_mean"
    MATMUL = "matmul"


@dataclass(frozen=True)
class Value:
    id: int
    meta: MetaTensor


@dataclass
class Node:
    op: Op
    inputs: list[Value]
    attrs: dict[str, Any]
    output: Value


@dataclass
class Graph:
    inputs: list[Value] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
    _next_value_id: int = 0

    def new_value(self, meta: MetaTensor) -> Value:
        v = Value(id=self._next_value_id, meta=meta)
        self._next_value_id += 1
        return v

    def add_input(self, meta: MetaTensor) -> Value:
        v = self.new_value(meta)
        self.inputs.append(v)
        return v

    def add_node(self, op: Op, inputs: list[Value], attrs: dict[str, Any], output_meta: MetaTensor) -> Value:
        out = self.new_value(output_meta)
        self.nodes.append(Node(op=op, inputs=inputs, attrs=dict(attrs), output=out))
        return out

    def set_outputs(self, outputs: list[Value]) -> None:
        if not outputs:
            raise ValueError("graph must have at least one output")
        self.outputs = outputs

    def verify(self) -> None:
        defined = {v.id for v in self.inputs}
        if len(defined) != len(self.inputs):
            raise ValueError("duplicate graph input values")
        for idx, node in enumerate(self.nodes):
            for inp in node.inputs:
                if inp.id not in defined:
                    raise ValueError(f"node {idx} uses undefined input value %{inp.id}")
            if node.output.id in defined:
                raise ValueError(f"node {idx} redefines value %{node.output.id}")
            defined.add(node.output.id)
        for out in self.outputs:
            if out.id not in defined:
                raise ValueError(f"graph output %{out.id} is undefined")

    def pretty(self) -> str:
        from .debug import graph_pretty

        return graph_pretty(self)

    def to_dot(self, path: str) -> None:
        from .debug import graph_to_dot

        graph_to_dot(self, path)
