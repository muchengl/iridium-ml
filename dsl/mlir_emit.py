from __future__ import annotations

from .graph import Graph, Op



def _fmt_attrs(attrs: dict[str, object]) -> str:
    if not attrs:
        return ""

    def _fmt(v: object) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    body = ", ".join(f"{k} = {_fmt(v)}" for k, v in attrs.items())
    return f" {{{body}}}"



def _op_name(op: Op) -> str:
    return f"toy.{op.value}"



def emit_mlir(graph: Graph, func_name: str = "main") -> str:
    if len(graph.outputs) != 1:
        raise ValueError("this demo emitter only supports single-output graphs")

    name_of: dict[int, str] = {}
    arg_defs: list[str] = []
    for idx, inp in enumerate(graph.inputs):
        name = f"%arg{idx}"
        name_of[inp.id] = name
        arg_defs.append(f"{name}: {inp.meta.to_mlir_tensor_type()}")

    ret_ty = graph.outputs[0].meta.to_mlir_tensor_type()

    lines: list[str] = ["module {"]
    lines.append(f"  func.func @{func_name}({', '.join(arg_defs)}) -> {ret_ty} {{")

    for node in graph.nodes:
        out_name = f"%{node.output.id}"
        name_of[node.output.id] = out_name
        input_names = [name_of[v.id] for v in node.inputs]
        input_tys = [v.meta.to_mlir_tensor_type() for v in node.inputs]
        attrs = _fmt_attrs(node.attrs)
        inputs_str = ", ".join(input_names)
        tys_str = ", ".join(input_tys)
        out_ty = node.output.meta.to_mlir_tensor_type()
        lines.append(
            f"    {out_name} = {_op_name(node.op)} {inputs_str}{attrs} : {tys_str} -> {out_ty}"
        )

    ret_name = name_of[graph.outputs[0].id]
    lines.append(f"    return {ret_name} : {ret_ty}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
