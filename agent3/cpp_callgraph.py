from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tree_sitter import Language, Node, Parser
from tree_sitter_cpp import language as cpp_language

from agent3.fs_utils import CPP_EXTS, iter_files, safe_read_text


@dataclass(frozen=True)
class FunctionInfo:
    key: str
    label: str
    relpath: str


@dataclass(frozen=True)
class CallEdge:
    src_key: str
    dst_key: str


def _set_parser_language(parser: Parser) -> None:
    raw = cpp_language()
    # tree-sitter-cpp returns a PyCapsule; tree_sitter.Parser expects tree_sitter.Language.
    lang: Language
    if isinstance(raw, Language):
        lang = raw
    else:
        # tree_sitter>=0.22 supports constructing Language from a capsule.
        lang = Language(raw)  # type: ignore[arg-type]

    # tree-sitter API differs by version; support both.
    if hasattr(parser, "set_language"):
        parser.set_language(lang)  # type: ignore[attr-defined]
    else:
        parser.language = lang  # type: ignore[assignment]


def _node_text(source_bytes: bytes, node: Node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _first_ident_text(source_bytes: bytes, node: Node) -> str | None:
    """
    Best-effort extraction of an identifier-like token from a node subtree.
    """
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in {
            "identifier",
            "field_identifier",
            "type_identifier",
            "namespace_identifier",
            "scoped_identifier",
        }:
            t = _node_text(source_bytes, n).strip()
            return t if t else None
        if n.type == "qualified_identifier":
            # Prefer the last child identifier
            for c in reversed(n.children):
                if c.type in {"identifier", "field_identifier", "type_identifier"}:
                    t = _node_text(source_bytes, c).strip()
                    if t:
                        return t
        # DFS
        for c in reversed(n.children):
            stack.append(c)
    return None


def _scope_stack_names(source_bytes: bytes, scope_stack: list[Node]) -> str:
    names: list[str] = []
    for n in scope_stack:
        name_node = n.child_by_field_name("name")
        if name_node is None:
            continue
        t = _node_text(source_bytes, name_node).strip()
        if t:
            names.append(t)
    return "::".join(names)


def _callee_name(source_bytes: bytes, call_node: Node) -> str | None:
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    # call_expression.function can be identifier / field_expression / qualified_identifier, etc.
    if fn.type in {"identifier", "field_identifier"}:
        t = _node_text(source_bytes, fn).strip()
        return t if t else None
    if fn.type == "field_expression":
        # Prefer final field_identifier (method call like obj.foo()).
        for c in reversed(fn.children):
            if c.type in {"field_identifier", "identifier"}:
                t = _node_text(source_bytes, c).strip()
                if t:
                    return t
    return _first_ident_text(source_bytes, fn)


def _stable_key(relpath: str, label: str, start_byte: int) -> str:
    h = hashlib.sha1(f"{relpath}:{label}:{start_byte}".encode("utf-8")).hexdigest()[:12]
    return f"fn_{h}"


def _walk(node: Node) -> Iterable[Node]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for c in reversed(n.children):
            stack.append(c)


def build_callgraph(
    *,
    project_path: Path,
    scope: Path | None = None,
    exclude_dir_names: set[str] | None = None,
) -> tuple[dict[str, FunctionInfo], list[tuple[str, str]]]:
    """
    Returns:
      - functions: key -> FunctionInfo
      - edges_by_name: list[(src_key, callee_name)] (callee_name unresolved)
    """
    project_path = project_path.resolve()
    scope_path = (scope or project_path).resolve()

    parser = Parser()
    _set_parser_language(parser)

    functions: dict[str, FunctionInfo] = {}
    edges_by_name: list[tuple[str, str]] = []

    for rec in iter_files(scope_path, include_exts=CPP_EXTS, exclude_dir_names=exclude_dir_names):
        source = safe_read_text(rec.path)
        if not source:
            continue
        src_bytes = source.encode("utf-8", errors="ignore")
        tree = parser.parse(src_bytes)
        root = tree.root_node

        # Traverse with a simple scope stack (namespaces/classes) to improve labels.
        scope_stack: list[Node] = []

        def visit(n: Node):
            nonlocal scope_stack

            is_scope = n.type in {
                "namespace_definition",
                "class_specifier",
                "struct_specifier",
                "union_specifier",
            }
            if is_scope:
                scope_stack.append(n)

            if n.type in {"function_definition", "constructor_or_destructor_definition"}:
                decl = n.child_by_field_name("declarator") or n.child_by_field_name("name")
                name = _first_ident_text(src_bytes, decl) if decl is not None else None
                if name:
                    scope_prefix = _scope_stack_names(src_bytes, scope_stack)
                    label_base = f"{scope_prefix}::{name}" if scope_prefix else name
                    label = f"{label_base} ({rec.relpath})"
                    key = _stable_key(rec.relpath, label_base, n.start_byte)
                    functions[key] = FunctionInfo(key=key, label=label, relpath=rec.relpath)

                    # Collect call expressions inside this function definition.
                    for sub in _walk(n):
                        if sub.type == "call_expression":
                            callee = _callee_name(src_bytes, sub)
                            if callee:
                                edges_by_name.append((key, callee))

            for c in n.children:
                visit(c)

            if is_scope:
                scope_stack.pop()

        visit(root)

    return functions, edges_by_name


def resolve_edges(
    functions: dict[str, FunctionInfo],
    edges_by_name: list[tuple[str, str]],
) -> list[CallEdge]:
    # Build an index from "short name" (last segment) to function keys.
    name_to_keys: dict[str, list[str]] = {}
    for k, fn in functions.items():
        # label contains "(relpath)" so use the part before space + '(' for name resolution.
        base = fn.label.split(" (", 1)[0]
        short = base.split("::")[-1]
        name_to_keys.setdefault(short, []).append(k)

    edges: list[CallEdge] = []
    for src_key, callee_name in edges_by_name:
        dst_candidates = name_to_keys.get(callee_name, [])
        for dst_key in dst_candidates:
            if dst_key != src_key:
                edges.append(CallEdge(src_key=src_key, dst_key=dst_key))
    return edges


