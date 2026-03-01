from __future__ import annotations

from typing import Any

from .graph import Graph, Node



def _format_attrs(attrs: dict[str, Any]) -> str:
    if not attrs:
        return ""
    body = ", ".join(f"{k}={v}" for k, v in attrs.items())
    return f" [{body}]"



def _value_label(value_id: int) -> str:
    return f"%{value_id}"



def graph_pretty(graph: Graph) -> str:
    lines: list[str] = []
    lines.append("graph(")
    for i, inp in enumerate(graph.inputs):
        lines.append(f"  input {i}: {_value_label(inp.id)} : {inp.meta.short()}")
    lines.append(") {")
    for node in graph.nodes:
        inp_str = ", ".join(_value_label(v.id) for v in node.inputs)
        lines.append(
            f"  {_value_label(node.output.id)} = {node.op.value}({inp_str})"
            f"{_format_attrs(node.attrs)} : {node.output.meta.short()}"
        )
    out_str = ", ".join(_value_label(v.id) for v in graph.outputs)
    lines.append(f"  return {out_str}")
    lines.append("}")
    return "\n".join(lines)



def _node_name(idx: int) -> str:
    return f"op{idx}"



def _dot_node_line(node_id: str, label: str, shape: str = "ellipse") -> str:
    return f'  {node_id} [label="{label}", shape={shape}];'



def graph_to_dot(graph: Graph, path: str) -> None:
    lines: list[str] = ["digraph G {"]
    for inp in graph.inputs:
        label = f"%{inp.id}\\n{inp.meta.shape}\\n{inp.meta.dtype.name}"
        lines.append(_dot_node_line(f"v{inp.id}", label))
    for idx, node in enumerate(graph.nodes):
        nid = _node_name(idx)
        label = node.op.value
        if node.attrs:
            attrs = ", ".join(f"{k}={v}" for k, v in node.attrs.items())
            label = f"{label}\\n{attrs}"
        lines.append(_dot_node_line(nid, label, shape="box"))
        for inp in node.inputs:
            lines.append(f"  v{inp.id} -> {nid};")
        out = node.output
        out_label = f"%{out.id}\\n{out.meta.shape}\\n{out.meta.dtype.name}"
        lines.append(_dot_node_line(f"v{out.id}", out_label))
        lines.append(f"  {nid} -> v{out.id};")
    for out in graph.outputs:
        lines.append(f"  v{out.id} [peripheries=2];")
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")



def trace(msg: str, enabled: bool) -> None:
    if enabled:
        print(f"[trace] {msg}")
