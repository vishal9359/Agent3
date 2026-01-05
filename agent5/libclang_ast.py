"""
AST builder using libclang.

This module builds AST from C++ codebase using libclang and generates JSON output
with function information, call relationships, and LLM-generated descriptions/flowcharts.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from clang import cindex
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama

from agent5.config import SETTINGS

# Configure libclang library path
# Default to common Linux paths, can be overridden via environment variable
CLANG_LIB_PATH = os.getenv("CLANG_LIB_PATH", "/usr/lib/llvm-18/lib/libclang.so")
try:
    cindex.Config.set_library_file(CLANG_LIB_PATH)
except Exception:
    # Try alternative paths
    for alt_path in [
        "/usr/lib/llvm-19/lib/libclang.so",
        "/usr/lib/llvm-17/lib/libclang.so",
        "/usr/lib/x86_64-linux-gnu/libclang.so.1",
    ]:
        try:
            cindex.Config.set_library_file(alt_path)
            break
        except Exception:
            continue

SUPPORTED_EXT = (".c", ".cpp", ".cc", ".cxx")
llm = ChatOllama(model="qwen3", temperature=0.1, base_url=SETTINGS.ollama_base_url)


def is_cpp_file(path: str) -> bool:
    """Check if file is a C++ source file."""
    return path.endswith(SUPPORTED_EXT)


def get_module_name(file_path: str, root_dir: str) -> str:
    """Get module name from file path relative to root directory."""
    rel = os.path.relpath(file_path, root_dir)
    no_ext = os.path.splitext(rel)[0]
    return ".".join(no_ext.split(os.sep))


def node_uid(cursor: cindex.Cursor) -> str:
    """Generate stable unique identifier for functions/methods."""
    loc = cursor.location
    name = cursor.spelling or "<anonymous>"
    file_name = loc.file.name if loc.file else "<unknown>"
    return f"{name}:{file_name}:{loc.line}"


def _get_fully_qualified_name(cursor: cindex.Cursor) -> str:
    """Get fully qualified function name including namespace/class."""
    name = cursor.spelling or "<anonymous>"
    
    # Try to get semantic parent (namespace/class)
    parent = cursor.semantic_parent
    if parent and parent.kind in (
        cindex.CursorKind.NAMESPACE,
        cindex.CursorKind.CLASS_DECL,
        cindex.CursorKind.STRUCT_DECL,
    ):
        parent_name = parent.spelling
        if parent_name:
            return f"{parent_name}::{name}"
    
    return name


def extract_node_info(cursor: cindex.Cursor, file_path: str, module_name: str) -> dict:
    """Extract node information including LLM-generated description and flowchart."""
    extent = cursor.extent
    
    # Read function code
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        l = lines[extent.start.line - 1 : extent.end.line]
    
    function_code = "".join(l)
    
    # Generate description using LLM
    description_prompt = (
        "You are a C++ Project Documentation Expert. "
        "Provide a Requirement Description for the given function written in C++. "
        "Be concise and don't invent anything. "
        f"Function:\n{function_code}"
    )
    messages = [HumanMessage(content=description_prompt)]
    description_response = llm.invoke(messages)
    
    # Generate flowchart using LLM
    flowchart_prompt = (
        "You are a C++ Project Documentation Expert. "
        "Generate a Mermaid flowchart for the given function written in C++. "
        "Be concise and don't invent anything. "
        f"Function:\n{function_code}"
    )
    messages = [HumanMessage(content=flowchart_prompt)]
    flowchart_response = llm.invoke(messages)
    
    # Get both simple and fully qualified names
    simple_name = cursor.spelling or "<anonymous>"
    qualified_name = _get_fully_qualified_name(cursor)
    
    return {
        "uid": node_uid(cursor),
        "name": simple_name,
        "qualified_name": qualified_name,
        "line_start": extent.start.line,
        "column_start": extent.start.column,
        "line_end": extent.end.line,
        "column_end": extent.end.column,
        "file_name": file_path,
        "module_name": module_name,
        "description": getattr(description_response, "content", str(description_response)),
        "flowchart": getattr(flowchart_response, "content", str(flowchart_response)),
        "callees": [],
        "callers": [],
    }


visited: dict[str, bool] = {}


def visit(
    cursor: cindex.Cursor,
    file_path: str,
    module_name: str,
    nodes: dict[str, dict],
    call_edges: defaultdict[str, set],
    current_fn: str | None,
) -> None:
    """AST traversal with call extraction."""
    if cursor.location.file and cursor.location.file.name != file_path:
        return
    
    fqn = module_name + ":" + file_path + ":" + (cursor.spelling or "<anonymous>")
    if fqn in visited:
        return
    else:
        visited[fqn] = True
    
    # Function / method / class
    if cursor.kind in (
        cindex.CursorKind.FUNCTION_DECL,
        cindex.CursorKind.CXX_METHOD,
    ):
        if cursor.spelling:
            uid = node_uid(cursor)
            if uid not in nodes:
                nodes[uid] = extract_node_info(cursor, file_path, module_name)
            current_fn = uid
    
    # Call expression
    if cursor.kind == cindex.CursorKind.CALL_EXPR and current_fn:
        ref = cursor.referenced
        if ref and ref.spelling and ref.location.file:
            callee_uid = node_uid(ref)
            call_edges[current_fn].add(callee_uid)
    
    for child in cursor.get_children():
        visit(child, file_path, module_name, nodes, call_edges, current_fn)


def parse_file(
    index: cindex.Index,
    file_path: str,
    root_dir: str,
    compile_args: list[str],
    nodes: dict[str, dict],
    call_edges: defaultdict[str, set],
) -> None:
    """Parse a single C++ file and extract AST information."""
    module_name = get_module_name(file_path, root_dir)
    
    tu = index.parse(
        file_path,
        args=compile_args,
        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    
    visit(tu.cursor, file_path, module_name, nodes, call_edges, None)


def parse_codebase(root_dir: str | Path, compile_args: list[str] | None = None) -> list[dict]:
    """
    Parse entire C++ codebase and generate AST with call graph.
    
    Args:
        root_dir: Root directory of the C++ project
        compile_args: Compilation arguments (default: ["-std=c++17"])
        
    Returns:
        List of node dictionaries with AST information
    """
    root_dir = str(root_dir)
    compile_args = compile_args or ["-std=c++17"]
    index = cindex.Index.create()
    
    nodes: dict[str, dict] = {}
    call_edges: defaultdict[str, set] = defaultdict(set)
    
    # Reset visited for each codebase parse
    visited.clear()
    
    for root, _, files in os.walk(root_dir):
        for f in files:
            if is_cpp_file(f):
                path = os.path.join(root, f)
                try:
                    parse_file(index, path, root_dir, compile_args, nodes, call_edges)
                except Exception as e:
                    print(f"[WARN] Failed to parse {path}: {e}")
    
    # Build callers from callees
    for caller_uid, callees in call_edges.items():
        for callee_uid in callees:
            if caller_uid in nodes:
                nodes[caller_uid]["callees"].append({"uid": callee_uid})
            if callee_uid in nodes:
                nodes[callee_uid]["callers"].append({"uid": caller_uid})
    
    return list(nodes.values())


def build_ast_json(root_dir: str | Path, output_path: str | Path, compile_args: list[str] | None = None) -> Path:
    """
    Build AST and save to JSON file.
    
    Args:
        root_dir: Root directory of the C++ project
        output_path: Path to output JSON file
        compile_args: Compilation arguments
        
    Returns:
        Path to the generated JSON file
    """
    output_path = Path(output_path)
    ast = parse_codebase(root_dir, compile_args)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ast, f, indent=2)
    
    print(f"AST with call graph written to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="C++ codebase root")
    parser.add_argument("-o", "--output", default="ast_with_calls.json")
    parser.add_argument("--compile-args", nargs="*", default=["-std=c++17"], help="Compilation arguments")
    args = parser.parse_args()
    
    build_ast_json(args.path, args.output, args.compile_args)


