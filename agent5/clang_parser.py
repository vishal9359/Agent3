"""
Clang AST + CFG Parser for C++ Projects (Stage 1).

This module provides NO-LLM deterministic parsing of C++ code using Clang's
Python bindings. It extracts:
- Full AST for all translation units
- Control-Flow Graphs (CFG) per function
- Call relationships (for understanding only)
- Leaf-level execution units
- Guard conditions, state mutations, error exits
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import clang.cindex as clang
    from clang.cindex import (
        Config,
        CursorKind,
        Index,
        TranslationUnit,
        TypeKind,
    )

    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False


@dataclass
class CFGNode:
    """Control Flow Graph node."""

    id: str
    kind: str  # "statement", "decision", "call", "return", "exit"
    code: str
    line: int
    successors: list[str] = field(default_factory=list)
    is_guard: bool = False  # Guard condition
    is_state_mutation: bool = False  # Changes state
    is_error_exit: bool = False  # Error path
    is_leaf: bool = False  # Leaf execution unit


@dataclass
class FunctionInfo:
    """Function information extracted from AST."""

    name: str
    file_path: str
    line_start: int
    line_end: int
    return_type: str
    parameters: list[tuple[str, str]]  # [(name, type), ...]
    calls: list[str]  # Called function names
    cfg: list[CFGNode]  # Control Flow Graph
    is_leaf: bool = False  # True if no calls to other functions


@dataclass
class ASTContext:
    """Complete AST context for a project."""

    project_path: Path
    translation_units: dict[str, Any] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)  # name -> info
    call_graph: dict[str, list[str]] = field(default_factory=dict)  # caller -> callees
    reverse_call_graph: dict[str, list[str]] = field(default_factory=dict)  # callee -> callers


class ClangParser:
    """Clang-based C++ AST and CFG parser."""

    def __init__(self, project_path: Path, include_paths: list[Path] | None = None):
        """
        Initialize Clang parser.

        Args:
            project_path: Root path of the C++ project
            include_paths: Additional include directories
        """
        if not CLANG_AVAILABLE:
            msg = (
                "libclang not available. Install with: pip install libclang\n"
                "Also ensure clang/llvm is installed on your system."
            )
            raise RuntimeError(msg)

        self.project_path = project_path
        self.include_paths = include_paths or []
        self.index = Index.create()
        self.context = ASTContext(project_path=project_path)

    def parse_project(self, file_patterns: list[str] | None = None) -> ASTContext:
        """
        Parse entire C++ project.

        Args:
            file_patterns: Glob patterns for C++ files (e.g., ["*.cpp", "*.cc"])

        Returns:
            Complete AST context
        """
        if file_patterns is None:
            file_patterns = ["*.cpp", "*.cc", "*.cxx", "*.c++"]

        cpp_files = []
        for pattern in file_patterns:
            cpp_files.extend(self.project_path.rglob(pattern))

        for cpp_file in cpp_files:
            self._parse_file(cpp_file)

        self._build_call_graphs()
        self._identify_leaf_functions()

        return self.context

    def _parse_file(self, file_path: Path) -> None:
        """Parse a single C++ file and extract functions."""
        args = ["-x", "c++", "-std=c++17"]
        for include_path in self.include_paths:
            args.extend(["-I", str(include_path)])

        try:
            tu = self.index.parse(str(file_path), args=args)
            self.context.translation_units[str(file_path)] = tu

            # Extract functions from this translation unit
            self._extract_functions(tu.cursor, str(file_path))

        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")

    def _extract_functions(self, cursor, file_path: str) -> None:
        """Recursively extract functions from AST."""
        if cursor.kind == CursorKind.FUNCTION_DECL and cursor.is_definition():
            # Extract function information
            func_name = cursor.spelling
            return_type = cursor.result_type.spelling if cursor.result_type else "void"

            params = []
            for arg in cursor.get_arguments():
                params.append((arg.spelling, arg.type.spelling))

            line_start = cursor.extent.start.line
            line_end = cursor.extent.end.line

            # Extract function calls
            calls = self._extract_calls(cursor)

            # Build CFG
            cfg = self._build_cfg(cursor)

            func_info = FunctionInfo(
                name=func_name,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                return_type=return_type,
                parameters=params,
                calls=calls,
                cfg=cfg,
            )

            # Use qualified name if inside namespace/class
            qualified_name = self._get_qualified_name(cursor)
            self.context.functions[qualified_name] = func_info

        # Recurse into children
        for child in cursor.get_children():
            if child.location.file and child.location.file.name == file_path:
                self._extract_functions(child, file_path)

    def _get_qualified_name(self, cursor) -> str:
        """Get fully qualified function name (e.g., namespace::ClassName::funcName)."""
        parts = []
        current = cursor

        while current:
            if current.kind in {
                CursorKind.FUNCTION_DECL,
                CursorKind.CXX_METHOD,
                CursorKind.CLASS_DECL,
                CursorKind.STRUCT_DECL,
                CursorKind.NAMESPACE,
            }:
                if current.spelling:
                    parts.insert(0, current.spelling)
            current = current.semantic_parent

        return "::".join(parts) if parts else cursor.spelling

    def _extract_calls(self, cursor) -> list[str]:
        """Extract all function calls from a function body."""
        calls = []

        def visit(node):
            if node.kind == CursorKind.CALL_EXPR:
                if node.referenced and node.referenced.spelling:
                    calls.append(node.referenced.spelling)
            for child in node.get_children():
                visit(child)

        visit(cursor)
        return calls

    def _build_cfg(self, cursor) -> list[CFGNode]:
        """
        Build Control Flow Graph for a function.

        This is a simplified CFG builder that identifies:
        - Sequential statements
        - Decision points (if/switch)
        - Loops
        - Function calls
        - Return statements
        - Guard conditions
        - State mutations
        """
        nodes = []
        node_id = 0

        def create_node(kind: str, code: str, line: int, **kwargs) -> CFGNode:
            nonlocal node_id
            node = CFGNode(
                id=f"node_{node_id}",
                kind=kind,
                code=code,
                line=line,
                **kwargs,
            )
            node_id += 1
            return node

        def visit(node, parent_id: str | None = None):
            """Visit AST nodes and build CFG."""
            nonlocal nodes

            if node.kind == CursorKind.IF_STMT:
                # Decision node
                condition = self._get_source_text(node)
                cfg_node = create_node(
                    "decision",
                    condition,
                    node.location.line,
                    is_guard=self._is_guard_condition(node),
                )
                nodes.append(cfg_node)

                # Connect to parent
                if parent_id:
                    for n in nodes:
                        if n.id == parent_id:
                            n.successors.append(cfg_node.id)

                # Visit children (then/else branches)
                for child in node.get_children():
                    visit(child, cfg_node.id)

            elif node.kind in {CursorKind.WHILE_STMT, CursorKind.FOR_STMT, CursorKind.DO_STMT}:
                # Loop node
                loop_code = self._get_source_text(node)
                cfg_node = create_node("decision", loop_code, node.location.line)
                nodes.append(cfg_node)

                if parent_id:
                    for n in nodes:
                        if n.id == parent_id:
                            n.successors.append(cfg_node.id)

                for child in node.get_children():
                    visit(child, cfg_node.id)

            elif node.kind == CursorKind.CALL_EXPR:
                # Function call
                call_name = node.spelling or "unknown"
                cfg_node = create_node("call", call_name, node.location.line)
                nodes.append(cfg_node)

                if parent_id:
                    for n in nodes:
                        if n.id == parent_id:
                            n.successors.append(cfg_node.id)

            elif node.kind == CursorKind.RETURN_STMT:
                # Return statement
                return_code = self._get_source_text(node)
                cfg_node = create_node(
                    "return",
                    return_code,
                    node.location.line,
                    is_error_exit=self._is_error_return(node),
                )
                nodes.append(cfg_node)

                if parent_id:
                    for n in nodes:
                        if n.id == parent_id:
                            n.successors.append(cfg_node.id)

            elif node.kind in {
                CursorKind.BINARY_OPERATOR,
                CursorKind.UNARY_OPERATOR,
                CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
            }:
                # Potential state mutation
                stmt_code = self._get_source_text(node)
                cfg_node = create_node(
                    "statement",
                    stmt_code,
                    node.location.line,
                    is_state_mutation=True,
                )
                nodes.append(cfg_node)

                if parent_id:
                    for n in nodes:
                        if n.id == parent_id:
                            n.successors.append(cfg_node.id)

            else:
                # Generic statement
                for child in node.get_children():
                    visit(child, parent_id)

        # Start visiting from function body
        visit(cursor)

        # Mark leaf nodes (nodes with no successors)
        for node in nodes:
            if not node.successors:
                node.is_leaf = True

        return nodes

    def _get_source_text(self, cursor) -> str:
        """Extract source code text for a cursor."""
        try:
            tokens = list(cursor.get_tokens())
            if tokens:
                return " ".join(t.spelling for t in tokens)
        except Exception:
            pass
        return cursor.spelling or ""

    def _is_guard_condition(self, cursor) -> bool:
        """Detect if an if-statement is a guard condition (validation/check)."""
        code = self._get_source_text(cursor).lower()
        guard_keywords = ["nullptr", "null", "invalid", "error", "fail", "empty", "!"]
        return any(kw in code for kw in guard_keywords)

    def _is_error_return(self, cursor) -> bool:
        """Detect if a return statement is an error exit."""
        code = self._get_source_text(cursor).lower()
        error_keywords = ["false", "nullptr", "null", "-1", "error", "fail"]
        return any(kw in code for kw in error_keywords)

    def _build_call_graphs(self) -> None:
        """Build forward and reverse call graphs."""
        for func_name, func_info in self.context.functions.items():
            self.context.call_graph[func_name] = func_info.calls
            for callee in func_info.calls:
                if callee not in self.context.reverse_call_graph:
                    self.context.reverse_call_graph[callee] = []
                self.context.reverse_call_graph[callee].append(func_name)

    def _identify_leaf_functions(self) -> None:
        """Identify leaf functions (functions that don't call other project functions)."""
        project_functions = set(self.context.functions.keys())

        for func_name, func_info in self.context.functions.items():
            # Check if any called function is in the project
            has_project_calls = any(call in project_functions for call in func_info.calls)
            func_info.is_leaf = not has_project_calls

    def find_function(
        self, function_name: str, file_path: Path | None = None
    ) -> FunctionInfo | None:
        """
        Find a function in the parsed project.

        Args:
            function_name: Name of the function
            file_path: Optional file path to disambiguate

        Returns:
            FunctionInfo if found, None otherwise
        """
        matches = []

        for qualified_name, func_info in self.context.functions.items():
            if function_name in qualified_name:
                if file_path is None or func_info.file_path == str(file_path):
                    matches.append(func_info)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            if file_path:
                raise ValueError(
                    f"Multiple matches for {function_name} in {file_path}: "
                    f"{[f.name for f in matches]}"
                )
            else:
                raise ValueError(
                    f"Ambiguous function name {function_name}. Found in: "
                    f"{[f.file_path for f in matches]}. "
                    "Please specify --entry-file to disambiguate."
                )

        return None

    def export_context(self, output_path: Path) -> None:
        """Export AST context to JSON for debugging."""
        data = {
            "project_path": str(self.context.project_path),
            "functions": {
                name: {
                    "name": func.name,
                    "file_path": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "return_type": func.return_type,
                    "parameters": func.parameters,
                    "calls": func.calls,
                    "is_leaf": func.is_leaf,
                    "cfg_node_count": len(func.cfg),
                }
                for name, func in self.context.functions.items()
            },
            "call_graph": self.context.call_graph,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

