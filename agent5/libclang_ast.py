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
from typing import Any

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


def _analyze_ast_node_bottom_up(cursor: cindex.Cursor, semantic_info: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze AST node bottom-up, building semantic understanding incrementally.
    
    This implements a DocAgent-inspired approach where meaning is built at leaf nodes
    first and aggregated while backtracking up the tree.
    """
    node_info = {
        "type": str(cursor.kind),
        "spelling": cursor.spelling or "",
        "line": cursor.location.line if cursor.location else 0,
        "semantic": "",
        "children": []
    }
    
    # First, recursively process children (bottom-up traversal)
    children_semantics = []
    for child in cursor.get_children():
        child_info = _analyze_ast_node_bottom_up(child, semantic_info)
        children_semantics.append(child_info)
        node_info["children"].append(child_info)
    
    # Now build semantic understanding at this level based on children
    kind = cursor.kind
    
    # Leaf nodes: extract basic semantic information
    if kind == cindex.CursorKind.DECL_REF_EXPR:
        node_info["semantic"] = f"reference to {cursor.spelling}"
        if cursor.spelling:
            semantic_info["variables"].add(cursor.spelling)
    
    elif kind == cindex.CursorKind.INTEGER_LITERAL:
        node_info["semantic"] = f"integer literal: {cursor.spelling}"
    
    elif kind == cindex.CursorKind.STRING_LITERAL:
        node_info["semantic"] = f"string literal: {cursor.spelling}"
    
    elif kind == cindex.CursorKind.CALL_EXPR:
        callee_name = cursor.spelling or (cursor.referenced.spelling if cursor.referenced else "unknown")
        node_info["semantic"] = f"function call: {callee_name}"
        semantic_info["function_calls"].add(callee_name)
        # Aggregate arguments from children
        args = [c["semantic"] for c in children_semantics if c["semantic"]]
        if args:
            node_info["semantic"] += f" with args: {', '.join(args)}"
    
    elif kind == cindex.CursorKind.BINARY_OPERATOR:
        # Get operator token
        tokens = list(cursor.get_tokens())
        op_token = tokens[1].spelling if len(tokens) > 1 else "?"
        left = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "value"
        right = children_semantics[1]["semantic"] if len(children_semantics) > 1 else "value"
        node_info["semantic"] = f"{left} {op_token} {right}"
        # Special handling for assignment
        if op_token == "=":
            semantic_info["operations"].append(f"assign {left} = {right}")
    
    elif kind == cindex.CursorKind.UNARY_OPERATOR:
        tokens = list(cursor.get_tokens())
        op_token = tokens[0].spelling if len(tokens) > 0 else "?"
        operand = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "value"
        node_info["semantic"] = f"{op_token}{operand}"
    
    elif kind == cindex.CursorKind.IF_STMT:
        condition = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "condition"
        node_info["semantic"] = f"if ({condition})"
        semantic_info["control_flow"].append("if")
        if len(children_semantics) > 1:
            then_part = children_semantics[1]["semantic"]
            node_info["semantic"] += f" then {then_part}"
        if len(children_semantics) > 2:
            else_part = children_semantics[2]["semantic"]
            node_info["semantic"] += f" else {else_part}"
    
    elif kind == cindex.CursorKind.WHILE_STMT:
        condition = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "condition"
        body = children_semantics[1]["semantic"] if len(children_semantics) > 1 else "body"
        node_info["semantic"] = f"while ({condition}) do {body}"
        semantic_info["control_flow"].append("while")
    
    elif kind == cindex.CursorKind.FOR_STMT:
        init = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "init"
        condition = children_semantics[1]["semantic"] if len(children_semantics) > 1 else "condition"
        increment = children_semantics[2]["semantic"] if len(children_semantics) > 2 else "increment"
        body = children_semantics[3]["semantic"] if len(children_semantics) > 3 else "body"
        node_info["semantic"] = f"for ({init}; {condition}; {increment}) do {body}"
        semantic_info["control_flow"].append("for")
    
    elif kind == cindex.CursorKind.RETURN_STMT:
        value = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "void"
        node_info["semantic"] = f"return {value}"
        semantic_info["control_flow"].append("return")
    
    elif kind == cindex.CursorKind.COMPOUND_STMT:
        # Aggregate all statements in the compound
        stmts = [c["semantic"] for c in children_semantics if c["semantic"]]
        node_info["semantic"] = "; ".join(stmts) if stmts else "{}"
    
    elif kind == cindex.CursorKind.VAR_DECL:
        var_name = cursor.spelling or "variable"
        var_type = cursor.type.spelling if cursor.type else "unknown"
        init = children_semantics[0]["semantic"] if len(children_semantics) > 0 else None
        if init:
            node_info["semantic"] = f"declare {var_type} {var_name} = {init}"
        else:
            node_info["semantic"] = f"declare {var_type} {var_name}"
        semantic_info["variables"].add(var_name)
    
    elif kind == cindex.CursorKind.PARM_DECL:
        param_name = cursor.spelling or "param"
        param_type = cursor.type.spelling if cursor.type else "unknown"
        node_info["semantic"] = f"{param_type} {param_name}"
        semantic_info["parameters"].add(f"{param_type} {param_name}")
    
    elif kind == cindex.CursorKind.MEMBER_REF_EXPR:
        obj = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "object"
        member = cursor.spelling or "member"
        node_info["semantic"] = f"{obj}.{member}"
    
    elif kind == cindex.CursorKind.ARRAY_SUBSCRIPT_EXPR:
        array = children_semantics[0]["semantic"] if len(children_semantics) > 0 else "array"
        index = children_semantics[1]["semantic"] if len(children_semantics) > 1 else "index"
        node_info["semantic"] = f"{array}[{index}]"
    
    # Aggregate semantic information from children if not already set
    if not node_info["semantic"] and children_semantics:
        # Default: aggregate children semantics
        child_semantics = [c["semantic"] for c in children_semantics if c["semantic"]]
        if child_semantics:
            node_info["semantic"] = " ".join(child_semantics)
    
    return node_info


def _build_semantic_representation(cursor: cindex.Cursor) -> dict[str, Any]:
    """
    Build a semantic representation of the function using bottom-up aggregation.
    """
    semantic_info = {
        "variables": set(),
        "function_calls": set(),
        "control_flow": [],
        "parameters": set(),
        "operations": [],
        "ast_structure": None
    }
    
    # Build AST analysis bottom-up
    ast_structure = _analyze_ast_node_bottom_up(cursor, semantic_info)
    semantic_info["ast_structure"] = ast_structure
    
    # Convert sets to lists for JSON serialization
    semantic_info["variables"] = list(semantic_info["variables"])
    semantic_info["function_calls"] = list(semantic_info["function_calls"])
    semantic_info["parameters"] = list(semantic_info["parameters"])
    
    return semantic_info


def _generate_description_from_semantics(function_code: str, semantic_info: dict[str, Any], function_name: str) -> str:
    """
    Generate function description using bottom-up semantic understanding.
    """
    # Build semantic context
    params = ", ".join(semantic_info.get("parameters", [])) if semantic_info.get("parameters") else "none"
    vars_used = ", ".join(list(semantic_info.get("variables", []))[:10])  # Limit to 10 vars
    calls = ", ".join(list(semantic_info.get("function_calls", []))[:10])  # Limit to 10 calls
    control_flow = ", ".join(semantic_info.get("control_flow", [])) if semantic_info.get("control_flow") else "linear"
    operations = ", ".join(semantic_info.get("operations", [])[:10])  # Limit to 10 operations
    
    # Extract main semantic flow from AST structure (get the function body semantic)
    ast_structure = semantic_info.get("ast_structure", {})
    main_flow = ast_structure.get("semantic", "")
    
    # If main flow is too long, try to get a summary from children
    if len(main_flow) > 800:
        # Get semantic from direct children (statements)
        children = ast_structure.get("children", [])
        child_semantics = [c.get("semantic", "") for c in children if c.get("semantic")]
        if child_semantics:
            main_flow = "; ".join(child_semantics[:10])  # Top 10 statements
        else:
            main_flow = main_flow[:800] + "..."
    
    prompt = f"""You are a C++ Project Documentation Expert using a DocAgent-inspired bottom-up semantic understanding approach.

Function: {function_name}

Semantic Analysis (Built bottom-up from AST):
- Parameters: {params}
- Variables used: {vars_used}
- Function calls: {calls}
- Control flow structures: {control_flow}
- Key operations: {operations}

Semantic Flow (aggregated from leaf nodes):
{main_flow}

Full Function Code:
{function_code}

Based on the bottom-up semantic analysis above (where meaning was built from leaf nodes and aggregated upward), provide a comprehensive, accurate Requirement Description for this function.

The description should:
1. Explain what the function does based on the semantic understanding derived from AST analysis
2. Describe the control flow and logic patterns detected
3. Mention key operations, variable usage, and function calls identified
4. Be precise and accurate - don't invent anything not present in the semantic analysis
5. Focus on the actual behavior derived from the bottom-up semantic aggregation

Provide a clear, detailed Requirement Description:"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


def _clean_and_validate_flowchart(flowchart: str) -> str:
    """
    Clean and validate Mermaid flowchart syntax.
    Fixes common issues like invalid node IDs, missing connections, etc.
    """
    import re
    
    # Remove markdown code blocks
    flowchart = flowchart.strip()
    if flowchart.startswith("```"):
        lines = flowchart.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        flowchart = "\n".join(lines)
    
    flowchart = flowchart.strip()
    
    # Ensure it starts with flowchart TD (preferred) or graph TD
    if not flowchart.startswith("flowchart") and not flowchart.startswith("graph"):
        flowchart = "flowchart TD\n" + flowchart
    elif flowchart.startswith("graph"):
        # Replace graph with flowchart for better compatibility
        flowchart = flowchart.replace("graph TD", "flowchart TD", 1)
        flowchart = flowchart.replace("graph LR", "flowchart LR", 1)
    
    # Fix common syntax issues
    lines = flowchart.split("\n")
    cleaned_lines = []
    node_id_map = {}  # Map old invalid IDs to new simple IDs
    next_id_letter = ord('A')
    
    for line in lines:
        original_line = line
        line = line.strip()
        if not line:
            continue
        
        # Skip comments
        if line.startswith("%%") or line.startswith("//"):
            continue
        
        # Fix node connections - ensure proper arrow syntax
        if "-->" in line:
            # Normalize arrow spacing
            line = re.sub(r'\s*-->\s*', ' --> ', line)
        
        # Fix node IDs - replace invalid IDs (with spaces, special chars) with simple ones
        # Pattern: NodeID[Label] or NodeID{Label}
        def replace_node_id(match):
            node_id = match.group(1)
            # If node ID is invalid (has spaces, special chars, or is multi-word)
            if ' ' in node_id or not re.match(r'^[A-Za-z][A-Za-z0-9]*$', node_id):
                # Use mapping to ensure consistency
                if node_id not in node_id_map:
                    # Generate simple ID: A, B, C, etc.
                    new_id = chr(next_id_letter)
                    node_id_map[node_id] = new_id
                    nonlocal next_id_letter
                    next_id_letter += 1
                    if next_id_letter > ord('Z'):
                        next_id_letter = ord('A')  # Wrap around
                else:
                    new_id = node_id_map[node_id]
                return new_id + match.group(2)  # new_id + [ or {
            return match.group(0)  # Keep original if valid
        
        # Replace node definitions: NodeID[ or NodeID{
        line = re.sub(r'([A-Za-z0-9_\s]+)([\[{])', replace_node_id, line)
        
        # Also fix node references in arrows (standalone node IDs)
        if " --> " in line:
            parts = line.split(" --> ")
            for i, part in enumerate(parts):
                part = part.strip()
                # If part is just a node ID (no brackets), check if it needs fixing
                if not '[' in part and not '{' in part and part:
                    # Check if it's a valid simple ID
                    if not re.match(r'^[A-Za-z][A-Za-z0-9]*$', part):
                        # Map to simple ID
                        if part not in node_id_map:
                            new_id = chr(next_id_letter)
                            node_id_map[part] = new_id
                            next_id_letter += 1
                            if next_id_letter > ord('Z'):
                                next_id_letter = ord('A')
                        else:
                            new_id = node_id_map[part]
                        parts[i] = new_id
                    else:
                        parts[i] = part
            line = " --> ".join(parts)
        
        cleaned_lines.append(line)
    
    flowchart = "\n".join(cleaned_lines)
    
    # Final validation: ensure it starts correctly
    if not (flowchart.startswith("flowchart") or flowchart.startswith("graph")):
        flowchart = "flowchart TD\n" + flowchart
    
    return flowchart.strip()


def _generate_flowchart_from_semantics(function_code: str, semantic_info: dict[str, Any], function_name: str, description: str) -> str:
    """
    Generate Mermaid flowchart using bottom-up semantic understanding and description.
    """
    # Build semantic context
    params = ", ".join(semantic_info.get("parameters", [])) if semantic_info.get("parameters") else "none"
    control_flow = semantic_info.get("control_flow", [])
    calls = semantic_info.get("function_calls", [])
    operations = semantic_info.get("operations", [])
    
    # Extract control flow structure
    control_flow_str = ", ".join(control_flow) if control_flow else "linear"
    calls_str = ", ".join(list(calls)[:10]) if calls else "none"
    operations_str = ", ".join(list(operations)[:10]) if operations else "none"
    
    # Extract main semantic flow from AST structure
    ast_structure = semantic_info.get("ast_structure", {})
    main_flow = ast_structure.get("semantic", "")
    
    # Get structured flow from children for better flowchart generation
    children = ast_structure.get("children", [])
    structured_flow = []
    for child in children[:15]:  # Limit to 15 top-level statements
        child_semantic = child.get("semantic", "")
        child_type = child.get("type", "")
        if child_semantic:
            structured_flow.append(f"{child_type}: {child_semantic[:100]}")
    
    structured_flow_str = "\n".join(structured_flow) if structured_flow else main_flow[:500]
    
    prompt = f"""You are a C++ Project Documentation Expert using a DocAgent-inspired bottom-up semantic understanding approach.

Function: {function_name}
Parameters: {params}
Control flow detected: {control_flow_str}
Function calls: {calls_str}
Key operations: {operations_str}

Semantic Analysis (Built bottom-up from AST):
{structured_flow_str}

Function Description (MUST MATCH THIS):
{description[:1500]}

Full Function Code:
{function_code}

Based on the bottom-up semantic analysis and the function description above, generate a VALID, COMPLETE Mermaid flowchart that EXACTLY matches the description.

CRITICAL REQUIREMENTS:
1. Start with "flowchart TD" (not "graph TD")
2. Use simple node IDs (A, B, C, D, etc.) - NO spaces, special chars, or multi-word IDs
3. Every node must be connected - no orphaned nodes
4. All if/else/switch branches must have proper Yes/No labels and ALL paths must connect to End
5. All control flow structures must be properly closed
6. Use proper Mermaid syntax: NodeID[Label] --> NextNodeID
7. For conditions: NodeID{{Decision?}} -->|Yes| YesNode
8. For conditions: NodeID{{Decision?}} -->|No| NoNode
9. Ensure ALL paths eventually reach an End node
10. The flowchart MUST visualize exactly what the description says - if description mentions try/catch, show it; if it mentions multiple if branches, show all of them

Example valid syntax:
flowchart TD
    A[Start] --> B[Action]
    B --> C{{Decision?}}
    C -->|Yes| D[Action if Yes]
    C -->|No| E[Action if No]
    D --> F[End]
    E --> F

Generate ONLY valid Mermaid flowchart code starting with "flowchart TD". No markdown, no explanations, no code blocks:"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    flowchart = getattr(response, "content", str(response))
    
    # Clean up and validate the flowchart
    flowchart = _clean_and_validate_flowchart(flowchart)
    
    return flowchart


def extract_node_info(cursor: cindex.Cursor, file_path: str, module_name: str) -> dict:
    """Extract node information including LLM-generated description and flowchart using bottom-up semantic analysis."""
    extent = cursor.extent
    
    # Read function code
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        l = lines[extent.start.line - 1 : extent.end.line]
    
    function_code = "".join(l)
    
    # Get function name
    simple_name = cursor.spelling or "<anonymous>"
    qualified_name = _get_fully_qualified_name(cursor)
    
    # Build semantic representation using bottom-up approach
    print(f"[INFO] Analyzing semantics for function: {simple_name}")
    semantic_info = _build_semantic_representation(cursor)
    
    # Generate description using semantic understanding (must be first)
    print(f"[INFO] Generating description for: {simple_name}")
    description = _generate_description_from_semantics(function_code, semantic_info, qualified_name)
    
    # Generate flowchart using semantic understanding AND description (to ensure they match)
    print(f"[INFO] Generating flowchart for: {simple_name}")
    flowchart = _generate_flowchart_from_semantics(function_code, semantic_info, qualified_name, description)
    
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
        "description": description,
        "flowchart": flowchart,
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


