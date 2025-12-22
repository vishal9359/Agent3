"""
Call graph builder for C++ projects.

This module analyzes cross-function dependencies and builds a call graph
for bottom-up semantic aggregation. It identifies function boundaries, 
call relationships, and control flow structures.

Key Concepts:
- Call Graph: Directed graph where nodes are functions and edges are calls
- Basic Blocks: Atomic semantic units (statements without branches)
- Control Flow: Decision points, loops, returns within functions
- Cross-file Analysis: Resolves calls across multiple files

Pipeline:
    C++ Files → AST Analysis → Call Graph → Leaf Functions → Bottom-up Aggregation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser
from tree_sitter_cpp import language as cpp_language

# Import centralized project exclusion configuration
from agent5.fs_utils import is_in_project_scope, PROJECT_EXCLUDE_DIRS, PROJECT_EXCLUDE_PATTERNS


@dataclass
class FunctionInfo:
    """Metadata about a function in the codebase."""
    
    name: str
    file_path: Path
    namespace: str | None = None
    class_name: str | None = None
    qualified_name: str | None = None  # Full qualified name (namespace::class::function)
    start_line: int = 0
    end_line: int = 0
    is_leaf: bool = False  # True if function makes no calls
    calls: list[str] = field(default_factory=list)  # List of called function names
    called_by: list[str] = field(default_factory=list)  # List of caller function names
    body_statements: list[Any] | None = None  # AST nodes representing function body statements
    ast_root: Any | None = None  # Root AST node for the function
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallGraphNode:
    """A node in the call graph."""
    
    func_info: FunctionInfo
    level: int = 0  # Distance from leaf nodes (0 = leaf)
    semantic_summary: str | None = None  # LLM-generated summary (for aggregation)


@dataclass
class CallGraph:
    """Call graph for a C++ project."""
    
    nodes: dict[str, CallGraphNode]  # Map: qualified_name -> node
    edges: list[tuple[str, str]]  # List of (caller, callee) pairs
    
    def get_leaf_nodes(self) -> list[CallGraphNode]:
        """Get all leaf nodes (functions with no calls)."""
        return [node for node in self.nodes.values() if node.func_info.is_leaf]
    
    def get_nodes_at_level(self, level: int) -> list[CallGraphNode]:
        """Get all nodes at a specific level."""
        return [node for node in self.nodes.values() if node.level == level]
    
    def get_callees(self, func_name: str) -> list[CallGraphNode]:
        """Get all functions called by the given function."""
        callees = []
        func_info = self.nodes.get(func_name)
        if func_info:
            for callee_name in func_info.func_info.calls:
                if callee_name in self.nodes:
                    callees.append(self.nodes[callee_name])
        return callees
    
    def compute_levels(self) -> None:
        """
        Compute levels for all nodes (bottom-up).
        Leaf nodes are level 0, nodes calling only leaf nodes are level 1, etc.
        """
        # Start with leaf nodes
        work_queue = self.get_leaf_nodes()
        for node in work_queue:
            node.level = 0
        
        visited = {node.func_info.qualified_name for node in work_queue}
        
        # Process level by level
        current_level = 0
        while work_queue:
            # Find all callers of current level nodes
            next_level_funcs = set()
            for node in work_queue:
                for caller_name in node.func_info.called_by:
                    if caller_name not in visited:
                        next_level_funcs.add(caller_name)
            
            if not next_level_funcs:
                break
            
            current_level += 1
            work_queue = []
            
            for func_name in next_level_funcs:
                if func_name in self.nodes:
                    node = self.nodes[func_name]
                    
                    # Check if all callees have been assigned levels
                    callees = self.get_callees(func_name)
                    if all(c.level >= 0 for c in callees):
                        max_callee_level = max((c.level for c in callees), default=-1)
                        node.level = max_callee_level + 1
                        work_queue.append(node)
                        visited.add(func_name)


def _set_parser_language(parser: Parser) -> None:
    """Set the C++ language for the tree-sitter parser."""
    raw = cpp_language()
    lang: Language
    if isinstance(raw, Language):
        lang = raw
    else:
        lang = Language(raw)  # type: ignore[arg-type]
    
    if hasattr(parser, "set_language"):
        parser.set_language(lang)  # type: ignore[attr-defined]
    else:
        parser.language = lang  # type: ignore[assignment]


def _node_text(source_bytes: bytes, node: Node) -> str:
    """Extract text from a tree-sitter node."""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _get_identifier(source_bytes: bytes, node: Node | None) -> str | None:
    """Extract identifier from a node."""
    if node is None:
        return None
    
    if node.type in {"identifier", "field_identifier", "type_identifier"}:
        return _node_text(source_bytes, node).strip()
    
    # Traverse to find identifier
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in {"identifier", "field_identifier", "type_identifier"}:
            text = _node_text(source_bytes, n).strip()
            if text:
                return text
        for c in reversed(n.children):
            stack.append(c)
    
    return None


def _extract_namespace(node: Node, source_bytes: bytes) -> str | None:
    """Extract namespace from a namespace node."""
    for child in node.children:
        if child.type in {"identifier", "namespace_identifier"}:
            return _node_text(source_bytes, child).strip()
    return None


def _extract_class_name(node: Node, source_bytes: bytes) -> str | None:
    """Extract class/struct name from a class node."""
    for child in node.children:
        if child.type in {"type_identifier", "identifier"}:
            return _node_text(source_bytes, child).strip()
    return None


def _extract_calls_from_node(source_bytes: bytes, node: Node) -> list[str]:
    """Extract function call names from a node (recursively)."""
    calls = []
    
    def visit(n: Node) -> None:
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn:
                callee = _get_identifier(source_bytes, fn)
                if callee:
                    calls.append(callee)
        
        for child in n.children:
            visit(child)
    
    visit(node)
    return calls


def _analyze_function(
    source_bytes: bytes,
    fn_node: Node,
    file_path: Path,
    namespace: str | None,
    class_name: str | None,
) -> FunctionInfo | None:
    """Analyze a function definition and extract metadata."""
    # Get function name
    decl = fn_node.child_by_field_name("declarator") or fn_node.child_by_field_name("name")
    if not decl:
        return None
    
    func_name = _get_identifier(source_bytes, decl)
    if not func_name:
        return None
    
    # Build qualified name
    qualified_parts = []
    if namespace:
        qualified_parts.append(namespace)
    if class_name:
        qualified_parts.append(class_name)
    qualified_parts.append(func_name)
    qualified_name = "::".join(qualified_parts)
    
    # Get location
    start_line = fn_node.start_point[0] + 1
    end_line = fn_node.end_point[0] + 1
    
    # Extract calls
    body = fn_node.child_by_field_name("body")
    calls = []
    if body:
        calls = _extract_calls_from_node(source_bytes, body)
    
    # Determine if leaf
    is_leaf = len(calls) == 0
    
    return FunctionInfo(
        name=func_name,
        file_path=file_path,
        namespace=namespace,
        class_name=class_name,
        qualified_name=qualified_name,
        start_line=start_line,
        end_line=end_line,
        is_leaf=is_leaf,
        calls=calls,
    )


def _analyze_file(file_path: Path, project_root: Path) -> list[FunctionInfo]:
    """
    Analyze a single C++ file and extract function information.
    
    Only files that are inside the project root AND not in excluded directories
    are analyzed. This enforces a hard project boundary for AST analysis.
    
    Args:
        file_path: Path to the C++ file to analyze
        project_root: Root path of the project (for exclusion checking)
    
    Returns:
        List of FunctionInfo objects, or empty list if file should be excluded
    """
    # Use centralized exclusion check - skip files outside project scope
    if not is_in_project_scope(file_path, project_root):
        return []
    try:
        source_code = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    
    parser = Parser()
    _set_parser_language(parser)
    
    src_bytes = source_code.encode("utf-8", errors="ignore")
    tree = parser.parse(src_bytes)
    root = tree.root_node
    
    functions = []
    
    def visit(node: Node, namespace: str | None = None, class_name: str | None = None) -> None:
        """Recursively visit nodes and extract functions."""
        # Handle namespaces
        if node.type == "namespace_definition":
            ns_name = _extract_namespace(node, src_bytes)
            new_namespace = f"{namespace}::{ns_name}" if namespace else ns_name
            for child in node.children:
                visit(child, new_namespace, class_name)
            return
        
        # Handle classes/structs
        if node.type in {"class_specifier", "struct_specifier"}:
            cls_name = _extract_class_name(node, src_bytes)
            new_class_name = cls_name
            for child in node.children:
                visit(child, namespace, new_class_name)
            return
        
        # Handle functions
        if node.type in {"function_definition", "constructor_or_destructor_definition"}:
            func_info = _analyze_function(src_bytes, node, file_path, namespace, class_name)
            if func_info:
                functions.append(func_info)
            return
        
        # Recurse into children
        for child in node.children:
            visit(child, namespace, class_name)
    
    visit(root)
    return functions


def build_call_graph(project_path: Path) -> CallGraph:
    """
    Build a call graph for a C++ project.
    
    Args:
        project_path: Root directory of the C++ project
        
    Returns:
        CallGraph with all functions and their relationships
    """
    project_path = project_path.resolve()

    # Collect all C++ files (using centralized exclusion)
    cpp_files: list[Path] = []
    for ext in ["*.cpp", "*.cc", "*.cxx", "*.c", "*.hpp", "*.h", "*.hxx"]:
        for candidate in project_path.rglob(ext):
            # Use centralized exclusion check
            if is_in_project_scope(candidate, project_path):
                cpp_files.append(candidate)
    
    # Analyze all files
    all_functions: dict[str, FunctionInfo] = {}
    
    for file_path in cpp_files:
        functions = _analyze_file(file_path, project_path)
        for func in functions:
            # Use qualified name as key
            all_functions[func.qualified_name] = func
    
    # Build edges and update called_by relationships (project-only)
    edges: list[tuple[str, str]] = []
    
    for func in all_functions.values():
        for callee_name in func.calls:
            # Try to resolve callee (exact match, or fuzzy match)
            resolved_callee = None
            
            # 1. Exact match
            if callee_name in all_functions:
                resolved_callee = callee_name
            
            # 2. Try to match with current namespace
            if not resolved_callee and func.namespace:
                candidate = f"{func.namespace}::{callee_name}"
                if candidate in all_functions:
                    resolved_callee = candidate
            
            # 3. Try to match with current class
            if not resolved_callee and func.class_name:
                if func.namespace:
                    candidate = f"{func.namespace}::{func.class_name}::{callee_name}"
                else:
                    candidate = f"{func.class_name}::{callee_name}"
                if candidate in all_functions:
                    resolved_callee = candidate
            
            # 4. Fuzzy match (ends with callee_name)
            if not resolved_callee:
                for qname in all_functions.keys():
                    if qname.endswith(f"::{callee_name}") or qname == callee_name:
                        resolved_callee = qname
                        break
            
            if resolved_callee:
                edges.append((func.qualified_name, resolved_callee))
                all_functions[resolved_callee].called_by.append(func.qualified_name)

    # Leaf detection: consider ONLY calls to project-defined functions.
    # A function is leaf if it does not call any other project function.
    project_callees = {callee for _, callee in edges}
    project_callers = {caller for caller, _ in edges}

    for qname, func in all_functions.items():
        # If this function never appears as a caller in project edges, it's a leaf.
        func.is_leaf = qname not in project_callers
    
    # Create call graph nodes
    nodes = {
        qname: CallGraphNode(func_info=func, level=-1)
        for qname, func in all_functions.items()
    }
    
    # Create call graph
    call_graph = CallGraph(nodes=nodes, edges=edges)
    
    # Compute levels (bottom-up)
    call_graph.compute_levels()
    
    return call_graph


def find_entry_function(
    call_graph: CallGraph,
    function_name: str,
    file_path: Path | None = None,
) -> CallGraphNode | None:
    """
    Find an entry function in the call graph.
    
    Args:
        call_graph: The call graph
        function_name: Function name to find (simple or qualified)
        file_path: Optional file path for disambiguation
        
    Returns:
        CallGraphNode or None if not found
    """
    function_name_clean = function_name.strip()
    function_parts = function_name_clean.split("::")
    
    # Exact match
    if function_name_clean in call_graph.nodes:
        node = call_graph.nodes[function_name_clean]
        if file_path is None or node.func_info.file_path == str(file_path):
            return node
    
    # Improved matching logic
    candidates = []
    for node in call_graph.nodes.values():
        qualified_name = node.func_info.qualified_name or node.func_info.name
        qualified_name_clean = qualified_name.strip()
        qualified_parts = qualified_name_clean.split("::")
        
        is_match = False
        
        # Exact match
        if qualified_name_clean == function_name_clean:
            is_match = True
        # If function_name is qualified (e.g., "Manager::OnApply")
        elif len(function_parts) > 1:
            if qualified_name_clean == function_name_clean:
                is_match = True
            elif qualified_name_clean.endswith("::" + function_name_clean):
                is_match = True
            # Check if last components match
            elif len(qualified_parts) >= len(function_parts):
                if qualified_parts[-len(function_parts):] == function_parts:
                    is_match = True
        else:
            # Simple name (e.g., "OnApply") - match at end or as last component
            if qualified_name_clean.endswith("::" + function_name_clean):
                is_match = True
            elif qualified_parts and qualified_parts[-1] == function_name_clean:
                is_match = True
        
        if is_match:
            if file_path is None or node.func_info.file_path == str(file_path):
                candidates.append(node)
    
    if len(candidates) == 1:
        return candidates[0]
    
    if len(candidates) > 1:
        # Prefer exact qualified name match
        for node in candidates:
            qualified_name = node.func_info.qualified_name or node.func_info.name
            if qualified_name.strip() == function_name_clean:
                return node
        # Prefer exact simple name match
        for node in candidates:
            if node.func_info.name == function_name_clean:
                return node
        return candidates[0]
    
    return None


