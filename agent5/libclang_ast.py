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


visited_first_pass: dict[str, bool] = {}
visited_second_pass: dict[str, bool] = {}


def _extract_callee_name_from_call(cursor: cindex.Cursor) -> str | None:
    """Extract function name from a call expression cursor."""
    # Method 1: Try referenced cursor (most reliable)
    ref = cursor.referenced
    if ref and ref.spelling:
        return ref.spelling
    
    # Method 2: Extract from call expression structure
    children = list(cursor.get_children())
    if not children:
        return None
    
    # The function part is typically the first child (before arguments)
    func_cursor = children[0]
    
    # Try direct spelling
    if func_cursor.spelling:
        return func_cursor.spelling
    
    # For member function calls (obj.method()), traverse to find method name
    def find_method_name(c: cindex.Cursor) -> str | None:
        """Recursively find method name in call expression."""
        if c.kind == cindex.CursorKind.FIELD_IDENTIFIER:
            return c.spelling
        if c.kind == cindex.CursorKind.MEMBER_REF_EXPR:
            for child in c.get_children():
                if child.kind == cindex.CursorKind.FIELD_IDENTIFIER:
                    return child.spelling
        if c.kind == cindex.CursorKind.DECL_REF_EXPR:
            return c.spelling
        if c.spelling:
            return c.spelling
        
        # Recurse into children
        for child in c.get_children():
            result = find_method_name(child)
            if result:
                return result
        return None
    
    method_name = find_method_name(func_cursor)
    if method_name:
        return method_name
    
    # Method 3: Try to get display name
    try:
        display_name = cursor.displayname
        if display_name:
            # Remove arguments if present: "func(args)" -> "func"
            if "(" in display_name:
                return display_name.split("(")[0].strip()
            return display_name
    except Exception:
        pass
    
    return None


def visit(
    cursor: cindex.Cursor,
    file_path: str,
    module_name: str,
    nodes: dict[str, dict],
    call_edges: defaultdict[str, set],
    current_fn: str | None,
    is_second_pass: bool = False,
) -> str | None:
    """
    AST traversal with call extraction.
    
    Args:
        is_second_pass: If True, we're in the second pass (extracting calls)
    
    Returns:
        Updated current_fn (for proper tracking across recursion)
    """
    # Skip if cursor is from a different file
    if cursor.location.file and cursor.location.file.name != file_path:
        return current_fn
    
    # Use appropriate visited dictionary based on pass
    # For second pass, we need to traverse all nodes to find calls, so we use a more lenient check
    if is_second_pass:
        # In second pass, only track call expressions to avoid processing the same call twice
        # Allow full traversal of function bodies
        if cursor.kind == cindex.CursorKind.CALL_EXPR:
            fqn = f"{file_path}:{cursor.location.line}:{cursor.location.column}"
            if fqn in visited_second_pass:
                return current_fn
            visited_second_pass[fqn] = True
        # For other nodes in second pass, don't use visited check (allow full traversal)
    else:
        # First pass: use normal visited check to avoid processing same node twice
        fqn = f"{file_path}:{cursor.location.line}:{cursor.location.column}:{cursor.kind}"
        if fqn in visited_first_pass:
            return current_fn
        visited_first_pass[fqn] = True
    
    # Function / method / class declaration
    if cursor.kind in (
        cindex.CursorKind.FUNCTION_DECL,
        cindex.CursorKind.CXX_METHOD,
        cindex.CursorKind.CONSTRUCTOR,
        cindex.CursorKind.DESTRUCTOR,
    ):
        if cursor.spelling:
            uid = node_uid(cursor)
            if uid not in nodes:
                # First pass: extract full node info
                if not is_second_pass:
                    nodes[uid] = extract_node_info(cursor, file_path, module_name)
                else:
                    # Second pass: create minimal entry if missing
                    nodes[uid] = {
                        "uid": uid,
                        "name": cursor.spelling,
                        "qualified_name": _get_fully_qualified_name(cursor),
                        "line_start": cursor.extent.start.line if cursor.extent else 0,
                        "column_start": cursor.extent.start.column if cursor.extent else 0,
                        "line_end": cursor.extent.end.line if cursor.extent else 0,
                        "column_end": cursor.extent.end.column if cursor.extent else 0,
                        "file_name": file_path,
                        "module_name": module_name,
                        "description": "",
                        "flowchart": "",
                        "callees": [],
                        "callers": [],
                    }
            # CRITICAL: Always set current_fn when we encounter a function, even if already in nodes
            # This is essential for the second pass to track which function we're inside
            current_fn = uid
    
    # Call expression - extract callee
    if cursor.kind == cindex.CursorKind.CALL_EXPR:
        if not current_fn:
            # Debug: found a call but no current function context
            if is_second_pass and len(visited_second_pass) < 100:  # Limit debug output
                print(f"[DEBUG] Found CALL_EXPR at {cursor.location.line} but current_fn is None")
            return current_fn
        
        # Try multiple methods to get the callee
        callee_name = None
        callee_uid = None
        
        # Method 1: Use referenced cursor (works for direct function calls)
        ref = cursor.referenced
        if ref and ref.spelling:
            # Check if this is a function declaration we know about
            if ref.kind in (
                cindex.CursorKind.FUNCTION_DECL,
                cindex.CursorKind.CXX_METHOD,
                cindex.CursorKind.CONSTRUCTOR,
                cindex.CursorKind.DESTRUCTOR,
            ):
                callee_uid = node_uid(ref)
                callee_name = ref.spelling
            else:
                # It's a reference but not a function decl - might be a declaration
                # Try to find it in our nodes or create a placeholder
                callee_name = ref.spelling
                # Try to construct UID from the reference
                if ref.location.file:
                    ref_uid = node_uid(ref)
                    if ref_uid in nodes:
                        callee_uid = ref_uid
        
        # Method 2: Extract from call expression structure
        if not callee_name:
            callee_name = _extract_callee_name_from_call(cursor)
        
        # If we found a callee name, try to match it to existing nodes
        if callee_name and not callee_uid:
            # Try to find match in nodes by name (simple or qualified)
            for node_uid_key, node_data in nodes.items():
                node_name = node_data.get("name", "")
                node_qualified = node_data.get("qualified_name", "")
                
                # Match by simple name
                if node_name == callee_name:
                    callee_uid = node_uid_key
                    break
                # Match by qualified name
                if node_qualified and node_qualified == callee_name:
                    callee_uid = node_uid_key
                    break
                # Match by qualified name ending (e.g., "Prg::AllocateApply" matches "AllocateApply")
                if node_qualified and "::" in node_qualified:
                    qualified_parts = node_qualified.split("::")
                    if len(qualified_parts) >= 2 and qualified_parts[-1] == callee_name:
                        callee_uid = node_uid_key
                        break
                # Reverse match: if callee is "Prg::AllocateApply", match "AllocateApply"
                if "::" in callee_name:
                    callee_parts = callee_name.split("::")
                    if len(callee_parts) >= 2 and callee_parts[-1] == node_name:
                        callee_uid = node_uid_key
                        break
        
        # Record the call edge (will be resolved in parse_codebase)
        if callee_uid:
            call_edges[current_fn].add(callee_uid)
        elif callee_name:
            # Store by name for later resolution in parse_codebase
            call_edges[current_fn].add(f"NAME:{callee_name}")
        
        # Debug output (limit to avoid spam)
        if is_second_pass:
            caller_name = nodes.get(current_fn, {}).get("name", current_fn) if current_fn in nodes else current_fn
            # Print first 50 calls found
            if sum(len(v) for v in call_edges.values()) <= 50:
                print(f"[DEBUG] Found call: {caller_name} -> {callee_name or callee_uid or 'unknown'}")
    
    # Recursively process children
    for child in cursor.get_children():
        current_fn = visit(child, file_path, module_name, nodes, call_edges, current_fn, is_second_pass)
    
    return current_fn


def parse_file(
    index: cindex.Index,
    file_path: str,
    root_dir: str,
    compile_args: list[str],
    nodes: dict[str, dict],
    call_edges: defaultdict[str, set],
    is_second_pass: bool = False,
) -> None:
    """Parse a single C++ file and extract AST information."""
    module_name = get_module_name(file_path, root_dir)
    
    tu = index.parse(
        file_path,
        args=compile_args,
        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    
    visit(tu.cursor, file_path, module_name, nodes, call_edges, None, is_second_pass)


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
    name_to_uids: defaultdict[str, list[str]] = defaultdict(list)  # Map function names to UIDs
    
    # Reset visited dictionaries for each codebase parse
    visited_first_pass.clear()
    visited_second_pass.clear()
    
    # First pass: Collect all function declarations across all files
    print("[INFO] Collecting function declarations...")
    for root, _, files in os.walk(root_dir):
        for f in files:
            if is_cpp_file(f):
                path = os.path.join(root, f)
                try:
                    module_name = get_module_name(path, root_dir)
                    tu = index.parse(
                        path,
                        args=compile_args,
                        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
                    )
                    
                    def collect_functions(cursor: cindex.Cursor) -> None:
                        if cursor.location.file and cursor.location.file.name != path:
                            return
                        
                        if cursor.kind in (
                            cindex.CursorKind.FUNCTION_DECL,
                            cindex.CursorKind.CXX_METHOD,
                            cindex.CursorKind.CONSTRUCTOR,
                            cindex.CursorKind.DESTRUCTOR,
                        ):
                            if cursor.spelling:
                                uid = node_uid(cursor)
                                if uid not in nodes:
                                    nodes[uid] = extract_node_info(cursor, path, module_name)
                                    # Index by name for lookup
                                    name = cursor.spelling
                                    name_to_uids[name].append(uid)
                                    # Also index by qualified name
                                    qualified = _get_fully_qualified_name(cursor)
                                    if qualified != name:
                                        name_to_uids[qualified].append(uid)
                        
                        for child in cursor.get_children():
                            collect_functions(child)
                    
                    collect_functions(tu.cursor)
                except Exception as e:
                    print(f"[WARN] Failed to parse {path}: {e}")
    
    print(f"[INFO] Found {len(nodes)} functions. Extracting call relationships...")
    
    # Second pass: Extract call relationships
    for root, _, files in os.walk(root_dir):
        for f in files:
            if is_cpp_file(f):
                path = os.path.join(root, f)
                try:
                    parse_file(index, path, root_dir, compile_args, nodes, call_edges, is_second_pass=True)
                except Exception as e:
                    print(f"[WARN] Failed to parse {path} for call extraction: {e}")
    
    total_raw_edges = sum(len(callees) for callees in call_edges.values())
    print(f"[INFO] Extracted {total_raw_edges} raw call edges from AST")
    print(f"[INFO] Found calls in {len(call_edges)} functions")
    
    if total_raw_edges == 0:
        print("[WARN] No call edges found! This might indicate:")
        print("  - Functions don't call other functions")
        print("  - Call extraction is not working properly")
        print("  - current_fn is not being set correctly")
    
    # Resolve call edges: match by name if UID doesn't exist
    resolved_edges: defaultdict[str, set[str]] = defaultdict(set)
    unresolved_calls = 0
    
    for caller_uid, callees in call_edges.items():
        if caller_uid not in nodes:
            print(f"[WARN] Caller UID {caller_uid} not found in nodes, skipping")
            continue  # Skip if caller not in nodes
            
        for callee_ref in callees:
            callee_uid = None
            callee_name = None
            
            # If it's already a UID, use it
            if isinstance(callee_ref, str) and not callee_ref.startswith("NAME:"):
                if callee_ref in nodes:
                    callee_uid = callee_ref
                else:
                    # Try to find by name
                    # Extract name from UID format: "name:file:line"
                    parts = callee_ref.split(":", 2)
                    if len(parts) >= 1:
                        callee_name = parts[0]
            
            # If it's a name placeholder, extract the name
            elif isinstance(callee_ref, str) and callee_ref.startswith("NAME:"):
                callee_name = callee_ref[5:]  # Remove "NAME:" prefix
            
            # Try to resolve by name
            if callee_name and not callee_uid:
                # Try exact name match
                if callee_name in name_to_uids:
                    # Use first match (prefer exact match)
                    callee_uid = name_to_uids[callee_name][0]
                else:
                    # Try partial matching (for qualified names)
                    # e.g., "AllocateApply" should match "Prg::AllocateApply"
                    for name_key, uid_list in name_to_uids.items():
                        # Check if callee_name matches the function part of qualified name
                        if "::" in name_key:
                            qualified_parts = name_key.split("::")
                            if len(qualified_parts) >= 2 and qualified_parts[-1] == callee_name:
                                callee_uid = uid_list[0]
                                break
                        # Also try reverse: if callee is "Prg::AllocateApply", match "AllocateApply"
                        if "::" in callee_name:
                            callee_parts = callee_name.split("::")
                            if len(callee_parts) >= 2:
                                func_part = callee_parts[-1]
                                if name_key == func_part or (name_key in name_to_uids and func_part in name_key):
                                    callee_uid = uid_list[0]
                                    break
            
            # Record the resolved edge
            if callee_uid and callee_uid in nodes:
                resolved_edges[caller_uid].add(callee_uid)
            elif callee_name:
                unresolved_calls += 1
                # Debug: print unresolved calls
                if unresolved_calls <= 10:  # Limit output
                    print(f"[DEBUG] Unresolved call: {caller_uid} -> {callee_name}")
    
    print(f"[INFO] Resolved {len(resolved_edges)} caller functions with {sum(len(callees) for callees in resolved_edges.values())} total relationships")
    
    # Build callees and callers lists (avoid duplicates)
    total_edges = 0
    for caller_uid, callee_uids in resolved_edges.items():
        if caller_uid in nodes:
            for callee_uid in callee_uids:
                if callee_uid in nodes:
                    # Add to caller's callees (avoid duplicates)
                    callees_list = nodes[caller_uid]["callees"]
                    if not any(c.get("uid") == callee_uid for c in callees_list):
                        callees_list.append({"uid": callee_uid})
                        total_edges += 1
                    
                    # Add to callee's callers (avoid duplicates)
                    callers_list = nodes[callee_uid]["callers"]
                    if not any(c.get("uid") == caller_uid for c in callers_list):
                        callers_list.append({"uid": caller_uid})
    
    print(f"[INFO] Built call graph with {total_edges} call relationships")
    print(f"[INFO] Functions with callees: {sum(1 for n in nodes.values() if len(n.get('callees', [])) > 0)}")
    print(f"[INFO] Functions with callers: {sum(1 for n in nodes.values() if len(n.get('callers', [])) > 0)}")
    if unresolved_calls > 0:
        print(f"[WARN] {unresolved_calls} call relationships could not be resolved (may be external functions, templates, or system calls)")
    
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


