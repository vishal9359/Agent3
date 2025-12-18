"""
Enhanced scenario extraction for C++ code.

This module implements deterministic, rule-based scenario extraction from C++ code
using AST + CFG (Control Flow Graph) analysis. The extraction follows strict
scenario boundary rules to avoid noise and produce reliable Scenario Flow Models (SFM).

Pipeline:
    C++ Code → AST + CFG → Scenario Extraction → SFM (JSON) → Mermaid

Key Principles:
- DETERMINISTIC: No guessing, rule-based only
- FAIL FAST: If SFM cannot be built reliably, refuse to proceed
- SEMANTIC: Collapse function calls into semantic actions
- BOUNDED: Include only scenario-relevant nodes (exclude logging, metrics, utilities)

Detail Levels (v3):
- HIGH: Only top-level business steps (minimal detail)
- MEDIUM: Include validations, decisions, state-changing operations (default)
- DEEP: Expand critical sub-operations affecting control flow or persistent state
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser
from tree_sitter_cpp import language as cpp_language


class DetailLevel(Enum):
    """Detail level for scenario extraction."""
    
    HIGH = "high"  # Only top-level business steps
    MEDIUM = "medium"  # Include validations, decisions, state changes (default)
    DEEP = "deep"  # Expand critical sub-operations


@dataclass
class SFMNode:
    """A node in the Scenario Flow Model."""
    
    id: str
    node_type: str  # 'terminator', 'process', 'decision', 'io'
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SFMEdge:
    """An edge in the Scenario Flow Model."""
    
    src: str
    dst: str
    label: str | None = None


@dataclass
class ScenarioFlowModel:
    """
    Scenario Flow Model (SFM) - a deterministic, validated flow representation.
    
    This is the authoritative model that must exist before any LLM is called.
    """
    
    nodes: list[SFMNode]
    edges: list[SFMEdge]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type,
                    "label": n.label,
                    **n.metadata,
                }
                for n in self.nodes
            ],
            "edges": [
                {"src": e.src, "dst": e.dst, **({"label": e.label} if e.label else {})}
                for e in self.edges
            ],
            "metadata": self.metadata,
        }
    
    def validate(self) -> None:
        """
        Validate the SFM structure. Raises RuntimeError if invalid.
        
        Requirements:
        - Exactly 1 start node
        - At least 1 end node
        - All edges reference valid nodes
        - No orphaned nodes (except start/end)
        """
        node_ids = {n.id for n in self.nodes}
        
        # Check for start and end
        starts = [n for n in self.nodes if n.id == "start"]
        ends = [n for n in self.nodes if n.id == "end"]
        
        if len(starts) != 1:
            raise RuntimeError(
                f"SFM validation failed: Expected exactly 1 start node, found {len(starts)}"
            )
        
        if len(ends) < 1:
            raise RuntimeError("SFM validation failed: No end node found")
        
        # Check all edges reference valid nodes
        for edge in self.edges:
            if edge.src not in node_ids:
                raise RuntimeError(f"SFM validation failed: Edge references invalid src node '{edge.src}'")
            if edge.dst not in node_ids:
                raise RuntimeError(f"SFM validation failed: Edge references invalid dst node '{edge.dst}'")
        
        # Check for orphaned nodes (all nodes except start should have incoming edges)
        has_incoming = {e.dst for e in self.edges}
        has_outgoing = {e.src for e in self.edges}
        
        for node in self.nodes:
            if node.id in {"start", "end"}:
                continue
            if node.id not in has_incoming and node.id not in has_outgoing:
                raise RuntimeError(
                    f"SFM validation failed: Orphaned node '{node.id}' (no incoming or outgoing edges)"
                )


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


def _normalize_ws(s: str, limit: int = 140) -> str:
    """Normalize whitespace in a string."""
    out = " ".join(s.replace("\t", " ").split())
    return out if len(out) <= limit else out[: limit - 1] + "…"


def _sanitize_label(label: str) -> str:
    """
    Make labels human-readable and Mermaid-safe.
    
    Rules:
    - Remove C++ operators/punctuation
    - Simplify common patterns
    - Keep semantic meaning
    """
    s = label.strip()
    
    # Common condition patterns -> semantic
    s = re.sub(r"\.empty\(\)\s*==\s*true", " empty", s)
    s = re.sub(r"\.empty\(\)", " empty", s)
    s = re.sub(r"\s*==\s*true\b", "", s)
    s = re.sub(r"\s*==\s*false\b", " not", s)
    s = re.sub(r"\s*!=\s*nullptr\b", " exists", s)
    s = re.sub(r"\s*==\s*nullptr\b", " is null", s)
    
    # Return statements
    m = re.match(r"^return\s+(.+)$", s)
    if m:
        ret = m.group(1).strip().rstrip(";")
        ret = ret.split("::")[-1]
        s = f"Return {ret}"
    
    # new/delete
    if "new " in s:
        s = "Create object"
    if s.startswith("delete "):
        s = "Free object"
    
    # Strip namespace qualifiers
    s = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*::", "", s)
    
    # Remove risky punctuation
    s = re.sub(r"[;:*!=<>|&%^~`]", " ", s)
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = s.replace('"', "").replace("'", "")
    
    # Collapse whitespace
    s = " ".join(s.split())
    if not s:
        s = "Step"
    
    # Keep labels short
    if len(s) > 60:
        s = s[:57] + "..."
    
    return s


def _is_noise_call(name: str) -> bool:
    """
    Check if a function call is noise (logging, metrics, etc.).
    
    Scenario boundary rule: Exclude logging, metrics, debugging.
    """
    n = name.lower()
    noise_keywords = (
        "log",
        "spdlog",
        "printf",
        "fprintf",
        "sprintf",
        "cout",
        "cerr",
        "trace",
        "metric",
        "stats",
        "telemetry",
        "debug",
        "info",
        "warn",
        "warning",
        "error",
        "assert",
        "check",
        "verify",
    )
    return any(kw in n for kw in noise_keywords)


def _classify_call(callee: str, detail_level: DetailLevel = DetailLevel.MEDIUM) -> tuple[bool, str, str]:
    """
    Classify a function call into a semantic action based on detail level.
    
    CRITICAL: Detail level MUST produce structural differences in the flowchart!
    
    Args:
        callee: Function name
        detail_level: Level of detail to include
        
    Returns:
        (include: bool, semantic_label: str, category: str)
        
    Scenario boundary rule: Include only business-relevant calls.
    Collapse the call into a single semantic step (never descend).
    
    Categories and inclusion rules:
    - business: Core business logic
      * HIGH: Only major business operations (create, execute, handle, process)
      * MEDIUM+: All business operations
    - validation: Input/data validation
      * HIGH: EXCLUDED
      * MEDIUM+: INCLUDED
    - state: State-changing operations  
      * HIGH: Only major state changes (create, delete, init)
      * MEDIUM+: All state changes
    - critical: Critical sub-operations (lookups, reads, loads)
      * HIGH: EXCLUDED
      * MEDIUM: EXCLUDED
      * DEEP: INCLUDED
    - utility: Utility functions (NEVER included)
    """
    c = callee.strip()
    lc = c.lower()
    
    # Always exclude noise
    if not c or _is_noise_call(c):
        return False, "", "utility"
    
    # HIGH level: ONLY major business operations
    major_business_verbs = [
        ("create", "Create", "business_major"),
        ("execute", "Execute", "business_major"),
        ("handle", "Handle", "business_major"),
        ("process", "Process", "business_major"),
    ]
    
    # MEDIUM level: More business operations  
    minor_business_verbs = [
        ("make", "Create", "business_minor"),
        ("build", "Build", "business_minor"),
        ("run", "Run", "business_minor"),
        ("send", "Send", "business_minor"),
        ("receive", "Receive", "business_minor"),
        ("perform", "Perform", "business_minor"),
    ]
    
    # MEDIUM+ level: Validations
    validation_verbs = [
        ("parse", "Parse", "validation"),
        ("check", "Validate", "validation"),
        ("validate", "Validate", "validation"),
        ("isvalid", "Validate", "validation"),
        ("verify", "Validate", "validation"),
    ]
    
    # HIGH level: Major state changes only
    major_state_verbs = [
        ("create", "Create", "state_major"),  # Duplicate with business, but check state context
        ("delete", "Delete", "state_major"),
        ("init", "Initialize", "state_major"),
        ("destroy", "Destroy", "state_major"),
    ]
    
    # MEDIUM+ level: All state changes
    minor_state_verbs = [
        ("set", "Set", "state_minor"),
        ("update", "Update", "state_minor"),
        ("add", "Add", "state_minor"),
        ("insert", "Insert", "state_minor"),
        ("remove", "Remove", "state_minor"),
        ("erase", "Erase", "state_minor"),
        ("open", "Open", "state_minor"),
        ("close", "Close", "state_minor"),
        ("start", "Start", "state_minor"),
        ("stop", "Stop", "state_minor"),
        ("save", "Save", "state_minor"),
        ("write", "Write", "state_minor"),
    ]
    
    # DEEP level ONLY: Critical sub-operations
    critical_verbs = [
        ("get", "Lookup", "critical"),
        ("fetch", "Fetch", "critical"),
        ("retrieve", "Retrieve", "critical"),
        ("read", "Read", "critical"),
        ("load", "Load", "critical"),
        ("query", "Query", "critical"),
        ("lookup", "Lookup", "critical"),
        ("find", "Find", "critical"),
    ]
    
    # Try to match and categorize
    all_verbs = (major_business_verbs + minor_business_verbs + 
                 validation_verbs + major_state_verbs + 
                 minor_state_verbs + critical_verbs)
    
    for key, verb, category in all_verbs:
        if key in lc:
            # Extract object from function name
            obj = re.sub(key, "", c, flags=re.IGNORECASE)
            obj = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", obj)
            obj = re.sub(r"[_\-]+", " ", obj)
            obj = " ".join(obj.split()).strip()
            
            # STRICT inclusion rules based on detail level
            include = False
            
            if detail_level == DetailLevel.HIGH:
                # HIGH: ONLY major business and major state operations
                if category in {"business_major", "state_major"}:
                    include = True
                # Everything else: EXCLUDED
                
            elif detail_level == DetailLevel.MEDIUM:
                # MEDIUM: Business + validations + state (but NOT critical)
                if category in {"business_major", "business_minor", 
                               "validation", "state_major", "state_minor"}:
                    include = True
                # critical: EXCLUDED
                
            elif detail_level == DetailLevel.DEEP:
                # DEEP: Include everything except utility
                if category != "utility":
                    include = True
            
            label = f"{verb} {obj}".strip() if obj else verb
            return include, label, category
    
    # Default: treat as utility, exclude
    return False, "", "utility"


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


def _extract_callee_name(source_bytes: bytes, call_node: Node) -> str | None:
    """Extract the name of the called function."""
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    
    if fn.type in {"identifier", "field_identifier"}:
        return _node_text(source_bytes, fn).strip()
    
    if fn.type == "field_expression":
        # Method call: obj.method()
        for c in reversed(fn.children):
            if c.type in {"field_identifier", "identifier"}:
                text = _node_text(source_bytes, c).strip()
                if text:
                    return text
    
    return _get_identifier(source_bytes, fn)


def _should_include_declaration(text: str, detail_level: DetailLevel = DetailLevel.MEDIUM) -> bool:
    """
    Check if a declaration should be included in the scenario based on detail level.
    
    CRITICAL: Must produce structural differences based on detail level!
    
    Scenario boundary rule: Include argument parsing, config, important state.
    """
    t = text.lower()
    
    # HIGH: ONLY top-level configurations (very selective)
    if detail_level == DetailLevel.HIGH:
        high_keywords = ("config", "manager", "service", "controller", "handler")
        # Must match AND be an initialization/assignment
        if any(kw in t for kw in high_keywords):
            # Only if it looks like initialization (has '=' or 'new')
            return ('=' in text or 'new ' in text.lower())
        return False
    
    # MEDIUM: Include arguments, parameters, return values
    if detail_level == DetailLevel.MEDIUM:
        medium_keywords = ("argv", "argc", "arg", "option", "param", "config", 
                          "ret", "result", "status", "error", "manager", "handler")
        return any(kw in t for kw in medium_keywords)
    
    # DEEP: Include more details (data structures, buffers, contexts)
    if detail_level == DetailLevel.DEEP:
        deep_keywords = ("argv", "argc", "arg", "option", "param", "config", 
                        "ret", "result", "status", "error",
                        "data", "buffer", "request", "response", "context", 
                        "ptr", "obj", "item", "entry", "record")
        return any(kw in t for kw in deep_keywords)
    
    return False


class SFMBuilder:
    """
    Builder for Scenario Flow Models.
    
    Implements deterministic, rule-based extraction with strict validation.
    """
    
    def __init__(self, max_steps: int = 30, detail_level: DetailLevel = DetailLevel.MEDIUM):
        self.max_steps = max_steps
        self.detail_level = detail_level
        self.nodes: dict[str, SFMNode] = {}
        self.edges: list[SFMEdge] = []
        self._counter = 0
        self._step_count = 0
        
        # Always start with start and end nodes
        self.add_node("start", "terminator", "Start")
        self.add_node("end", "terminator", "End")
    
    def _new_id(self, prefix: str) -> str:
        """Generate a new unique node ID."""
        self._counter += 1
        return f"{prefix}{self._counter}"
    
    def add_node(self, node_id: str, node_type: str, label: str, **metadata) -> str:
        """Add a node to the SFM."""
        self.nodes[node_id] = SFMNode(
            id=node_id,
            node_type=node_type,
            label=label,
            metadata=metadata,
        )
        return node_id
    
    def add_process(self, label: str, **metadata) -> str | None:
        """
        Add a process node.
        
        Returns None if max_steps exceeded (caller must handle gracefully).
        """
        if self._step_count >= self.max_steps:
            return None
        self._step_count += 1
        nid = self._new_id("p")
        return self.add_node(nid, "process", label, **metadata)
    
    def add_decision(self, label: str, **metadata) -> str | None:
        """
        Add a decision node.
        
        Returns None if max_steps exceeded (caller must handle gracefully).
        """
        if self._step_count >= self.max_steps:
            return None
        self._step_count += 1
        nid = self._new_id("d")
        return self.add_node(nid, "decision", label, **metadata)
    
    def add_edge(self, src: str, dst: str, label: str | None = None) -> None:
        """Add an edge to the SFM."""
        self.edges.append(SFMEdge(src=src, dst=dst, label=label))
    
    def build(self) -> ScenarioFlowModel:
        """Build and validate the final SFM."""
        sfm = ScenarioFlowModel(
            nodes=list(self.nodes.values()),
            edges=self.edges,
        )
        sfm.validate()
        return sfm


def extract_scenario_from_function(
    source_code: str,
    function_name: str | None = None,
    *,
    max_steps: int = 30,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
) -> ScenarioFlowModel:
    """
    Extract a Scenario Flow Model from a C++ function.
    
    This is the core deterministic extraction that MUST succeed before any LLM is called.
    
    Args:
        source_code: C++ source code
        function_name: Name of the entry function (auto-detect if None)
        max_steps: Maximum number of steps in the scenario
        detail_level: Level of detail (HIGH, MEDIUM, DEEP)
        
    Returns:
        ScenarioFlowModel
        
    Raises:
        RuntimeError: If SFM cannot be built reliably
    """
    parser = Parser()
    _set_parser_language(parser)
    
    src_bytes = source_code.encode("utf-8", errors="ignore")
    tree = parser.parse(src_bytes)
    root = tree.root_node
    
    # Find the entry function
    fn_node = _find_function(src_bytes, root, function_name)
    if fn_node is None:
        # List available functions for better error message
        available_funcs = []
        
        def collect_func_names(node: Node) -> None:
            if node.type in {"function_definition", "constructor_or_destructor_definition"}:
                decl = node.child_by_field_name("declarator") or node.child_by_field_name("name")
                if decl:
                    fn_name = _get_identifier(src_bytes, decl)
                    if fn_name:
                        available_funcs.append(fn_name)
            for child in node.children:
                collect_func_names(child)
        
        collect_func_names(root)
        
        error_msg = f"Cannot find entry function"
        if function_name:
            error_msg += f" '{function_name}'"
        error_msg += " in source code"
        
        if available_funcs:
            error_msg += f"\n\nAvailable functions in file:\n"
            for i, fn in enumerate(available_funcs[:20], 1):
                error_msg += f"  {i}. {fn}\n"
            if len(available_funcs) > 20:
                error_msg += f"  ... and {len(available_funcs) - 20} more\n"
            error_msg += f"\nUse: --function <function_name>"
        else:
            error_msg += "\n\nNo function definitions found in file."
        
        raise RuntimeError(error_msg)
    
    # Build SFM from the function
    builder = SFMBuilder(max_steps=max_steps, detail_level=detail_level)
    _extract_from_function_body(src_bytes, fn_node, builder)
    
    return builder.build()


def _find_function(source_bytes: bytes, root: Node, name: str | None) -> Node | None:
    """
    Find a function definition node by name.
    Recursively searches through namespaces, classes, etc.
    """
    functions = []
    
    # Recursively collect all function definitions
    def collect_functions(node: Node) -> None:
        if node.type in {"function_definition", "constructor_or_destructor_definition"}:
            functions.append(node)
        
        # Recurse into children
        for child in node.children:
            collect_functions(child)
    
    collect_functions(root)
    
    if not functions:
        return None
    
    if name is None:
        # Return first function if no name specified
        return functions[0]
    
    # Find by name (exact match or partial match)
    name_lower = name.lower()
    
    # First try exact match
    for fn in functions:
        decl = fn.child_by_field_name("declarator") or fn.child_by_field_name("name")
        if decl:
            fn_name = _get_identifier(source_bytes, decl)
            if fn_name and fn_name == name:
                return fn
    
    # Try case-insensitive match
    for fn in functions:
        decl = fn.child_by_field_name("declarator") or fn.child_by_field_name("name")
        if decl:
            fn_name = _get_identifier(source_bytes, decl)
            if fn_name and fn_name.lower() == name_lower:
                return fn
    
    # Try partial match (contains)
    for fn in functions:
        decl = fn.child_by_field_name("declarator") or fn.child_by_field_name("name")
        if decl:
            fn_name = _get_identifier(source_bytes, decl)
            if fn_name and name_lower in fn_name.lower():
                return fn
    
    return None


def _extract_from_function_body(
    source_bytes: bytes,
    fn_node: Node,
    builder: SFMBuilder,
) -> None:
    """Extract scenario from a function body."""
    body = fn_node.child_by_field_name("body")
    if body is None:
        raise RuntimeError("Function has no body")
    
    # Process statements
    frontier = ["start"]
    frontier = _process_block(source_bytes, body, builder, frontier, None)
    
    # Connect remaining frontier to end
    for node_id in frontier:
        if node_id != "end":
            builder.add_edge(node_id, "end")


def _process_block(
    source_bytes: bytes,
    block_node: Node,
    builder: SFMBuilder,
    frontier: list[str],
    branch_label: str | None,
) -> list[str]:
    """Process a block of statements."""
    if not frontier:
        return []
    
    stmts = _get_statements(block_node)
    if not stmts:
        return frontier
    
    current = frontier
    first = True
    
    for stmt in stmts:
        current = _process_statement(
            source_bytes,
            stmt,
            builder,
            current,
            branch_label if first else None,
        )
        first = False
        if not current:
            break
    
    return current


def _get_statements(block_node: Node) -> list[Node]:
    """Get statement nodes from a compound statement."""
    if block_node.type != "compound_statement":
        return [block_node]
    
    stmts = []
    for child in block_node.children:
        if child.type not in {"{", "}"}:
            stmts.append(child)
    
    return stmts


def _connect_frontier(
    builder: SFMBuilder,
    frontier: list[str],
    dst: str,
    label: str | None = None,
) -> None:
    """Connect all nodes in frontier to destination."""
    for src in frontier:
        if src != "end":
            builder.add_edge(src, dst, label if len(frontier) == 1 else None)


def _process_statement(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process a single statement and return new frontier."""
    if not frontier or builder._step_count >= builder.max_steps:
        _connect_frontier(builder, frontier, "end", incoming_label)
        return []
    
    # If statement
    if stmt.type == "if_statement":
        return _process_if(source_bytes, stmt, builder, frontier, incoming_label)
    
    # Return statement
    elif stmt.type == "return_statement":
        return _process_return(source_bytes, stmt, builder, frontier, incoming_label)
    
    # Throw statement
    elif stmt.type == "throw_statement":
        return _process_throw(source_bytes, stmt, builder, frontier, incoming_label)
    
    # Loop statements
    elif stmt.type in {"while_statement", "for_statement"}:
        return _process_loop(source_bytes, stmt, builder, frontier, incoming_label)
    
    # Switch statement
    elif stmt.type == "switch_statement":
        return _process_switch(source_bytes, stmt, builder, frontier, incoming_label)
    
    # Call expression
    elif stmt.type == "expression_statement":
        return _process_expression(source_bytes, stmt, builder, frontier, incoming_label)
    
    # Declaration
    elif stmt.type == "declaration":
        return _process_declaration(source_bytes, stmt, builder, frontier, incoming_label)
    
    # Default: ignore and continue
    return frontier


def _process_if(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process an if statement."""
    cond = stmt.child_by_field_name("condition")
    cond_text = _node_text(source_bytes, cond) if cond else "condition"
    cond_text = _sanitize_label(_normalize_ws(cond_text))
    
    # Add decision node
    d = builder.add_decision(cond_text.rstrip("?") + "?")
    if not d:
        _connect_frontier(builder, frontier, "end")
        return []
    
    _connect_frontier(builder, frontier, d, incoming_label)
    
    # Process then branch
    cons = stmt.child_by_field_name("consequence")
    then_front = _process_block(source_bytes, cons, builder, [d], "YES") if cons else []
    
    # Process else branch
    alt = stmt.child_by_field_name("alternative")
    else_front = _process_block(source_bytes, alt, builder, [d], "NO") if alt else []
    
    # Ensure both branches have nodes
    if not then_front:
        then_node = builder.add_process("Proceed")
        if then_node:
            builder.add_edge(d, then_node, "YES")
            then_front = [then_node]
    
    if not else_front:
        else_node = builder.add_process("Proceed")
        if else_node:
            builder.add_edge(d, else_node, "NO")
            else_front = [else_node]
    
    # Merge branches
    merged = []
    for node in then_front + else_front:
        if node and node != "end" and node not in merged:
            merged.append(node)
    
    return merged


def _process_return(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process a return statement."""
    text = _normalize_ws(_node_text(source_bytes, stmt))
    label = _sanitize_label(text if text else "Return")
    
    nid = builder.add_process(label)
    if not nid:
        _connect_frontier(builder, frontier, "end", incoming_label)
        return []
    
    _connect_frontier(builder, frontier, nid, incoming_label)
    builder.add_edge(nid, "end")
    return []


def _process_throw(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process a throw statement."""
    text = _normalize_ws(_node_text(source_bytes, stmt))
    label = _sanitize_label(text if text else "Throw exception")
    
    nid = builder.add_process(label)
    if not nid:
        _connect_frontier(builder, frontier, "end", incoming_label)
        return []
    
    _connect_frontier(builder, frontier, nid, incoming_label)
    builder.add_edge(nid, "end")
    return []


def _process_loop(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process a loop statement (simplified)."""
    cond = stmt.child_by_field_name("condition")
    cond_text = _node_text(source_bytes, cond) if cond else "loop condition"
    cond_text = _sanitize_label(_normalize_ws(cond_text))
    
    d = builder.add_decision(cond_text.rstrip("?") + "?")
    if not d:
        _connect_frontier(builder, frontier, "end")
        return []
    
    _connect_frontier(builder, frontier, d, incoming_label)
    
    # Process loop body
    body = stmt.child_by_field_name("body")
    body_front = _process_block(source_bytes, body, builder, [d], "YES") if body else []
    
    if not body_front:
        body_node = builder.add_process("Loop body")
        if body_node:
            builder.add_edge(d, body_node, "YES")
            body_front = [body_node]
    
    # Loop back
    for node in body_front:
        if node and node != "end":
            builder.add_edge(node, d, "loop")
    
    # Exit loop
    exit_node = builder.add_process("Continue")
    if exit_node:
        builder.add_edge(d, exit_node, "NO")
        return [exit_node]
    
    return []


def _process_switch(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process a switch statement (simplified)."""
    val = stmt.child_by_field_name("value")
    val_text = _node_text(source_bytes, val) if val else "switch"
    val_text = _sanitize_label(_normalize_ws(val_text))
    
    d = builder.add_decision(f"{val_text}?")
    if not d:
        _connect_frontier(builder, frontier, "end")
        return []
    
    _connect_frontier(builder, frontier, d, incoming_label)
    
    # Simplified: just create a continuation node
    cont = builder.add_process("Handle case")
    if cont:
        builder.add_edge(d, cont, "case")
        return [cont]
    
    return []


def _process_expression(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process an expression statement."""
    # Check if it's a call expression
    call = None
    for child in stmt.children:
        if child.type == "call_expression":
            call = child
            break
    
    if call:
        callee = _extract_callee_name(source_bytes, call)
        if callee:
            include, label, category = _classify_call(callee, builder.detail_level)
            if include:
                nid = builder.add_process(_sanitize_label(label))
                if not nid:
                    _connect_frontier(builder, frontier, "end", incoming_label)
                    return []
                _connect_frontier(builder, frontier, nid, incoming_label)
                return [nid]
    
    # Not an interesting call, continue
    return frontier


def _process_declaration(
    source_bytes: bytes,
    stmt: Node,
    builder: SFMBuilder,
    frontier: list[str],
    incoming_label: str | None,
) -> list[str]:
    """Process a declaration statement."""
    text = _normalize_ws(_node_text(source_bytes, stmt))
    
    if not _should_include_declaration(text, builder.detail_level):
        return frontier
    
    label = _sanitize_label(text)
    nid = builder.add_process(label)
    if not nid:
        _connect_frontier(builder, frontier, "end", incoming_label)
        return []
    
    _connect_frontier(builder, frontier, nid, incoming_label)
    return [nid]


def extract_scenario_from_project(
    project_path: Path,
    function_name: str,
    file_path: Path | None = None,
    *,
    max_steps: int = 50,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
) -> tuple[ScenarioFlowModel, dict[str, Any]]:
    """
    Extract a Scenario Flow Model from a C++ project using bottom-up aggregation.
    
    This is the VERSION 4 implementation that uses:
    - Call graph analysis for cross-file dependencies
    - Bottom-up semantic aggregation (DocAgent-inspired)
    - LLM-assisted semantic understanding
    - Deterministic SFM construction from aggregated semantics
    
    Pipeline:
        Project → Call Graph → Leaf Summaries → Bottom-up Aggregation → 
        Entry Function Semantic → SFM Construction → Validated SFM
    
    Args:
        project_path: Root directory of the C++ project
        function_name: Entry function name
        file_path: Optional file path for entry function disambiguation
        max_steps: Maximum number of steps in the scenario
        detail_level: Level of detail (HIGH, MEDIUM, DEEP)
        
    Returns:
        Tuple of (ScenarioFlowModel, semantic_metadata)
        
    Raises:
        RuntimeError: If call graph cannot be built or SFM cannot be constructed
    """
    from agent5.call_graph_builder import build_call_graph, find_entry_function
    from agent5.semantic_aggregator import (
        build_semantic_hierarchy,
        generate_scenario_description,
    )
    
    # Step 1: Build call graph
    call_graph = build_call_graph(project_path)
    
    if not call_graph.nodes:
        raise RuntimeError(f"No functions found in project: {project_path}")
    
    # Step 2: Find entry function
    entry_node = find_entry_function(call_graph, function_name, file_path)
    
    if not entry_node:
        # List available functions
        available = list(call_graph.nodes.keys())[:20]
        error_msg = f"Cannot find entry function '{function_name}' in project"
        if available:
            error_msg += f"\n\nAvailable functions (showing first 20):\n"
            for i, fn in enumerate(available, 1):
                error_msg += f"  {i}. {fn}\n"
            error_msg += f"\nTotal functions: {len(call_graph.nodes)}"
        raise RuntimeError(error_msg)
    
    # Step 3: Build semantic hierarchy (bottom-up aggregation)
    semantic_summaries = build_semantic_hierarchy(
        call_graph,
        entry_node,
        detail_level=detail_level,
    )
    
    entry_summary = semantic_summaries.get(entry_node.func_info.qualified_name)
    
    if not entry_summary:
        raise RuntimeError(
            f"Failed to build semantic summary for entry function: {function_name}"
        )
    
    # Step 4: Convert semantic summary to SFM
    # This is where we translate high-level semantic understanding into
    # a structured flow model suitable for flowchart generation
    builder = SFMBuilder(max_steps=max_steps, detail_level=detail_level)
    
    last_node = "start"
    node_counter = 0
    
    # Add key operations as process nodes
    for op in entry_summary.key_operations[:max_steps - len(entry_summary.decisions) - 2]:
        node_counter += 1
        nid = builder.add_process(op)
        if nid:
            builder.add_edge(last_node, nid)
            last_node = nid
    
    # Add decision points
    for decision in entry_summary.decisions[:max_steps - node_counter - 2]:
        node_counter += 1
        d = builder.add_decision(decision if decision.endswith("?") else f"{decision}?")
        if d:
            builder.add_edge(last_node, d)
            
            # Create branches
            yes_node = builder.add_process("Handle yes case")
            no_node = builder.add_process("Handle no case")
            
            if yes_node and no_node:
                builder.add_edge(d, yes_node, "YES")
                builder.add_edge(d, no_node, "NO")
                
                # Merge branches
                merge_node = builder.add_process("Continue")
                if merge_node:
                    builder.add_edge(yes_node, merge_node)
                    builder.add_edge(no_node, merge_node)
                    last_node = merge_node
                else:
                    builder.add_edge(yes_node, "end")
                    builder.add_edge(no_node, "end")
                    last_node = None
                    break
    
    # Connect to end
    if last_node and last_node != "end":
        builder.add_edge(last_node, "end")
    
    # Build and validate SFM
    sfm = builder.build()
    
    # Generate scenario description
    scenario_desc = generate_scenario_description(entry_summary, semantic_summaries)
    
    # Metadata for debugging and documentation
    metadata = {
        "entry_function": entry_node.func_info.qualified_name,
        "entry_file": str(entry_node.func_info.file_path),
        "call_graph_depth": entry_node.level,
        "reachable_functions": len(semantic_summaries),
        "scenario_description": scenario_desc,
        "semantic_summary": {
            "summary": entry_summary.summary,
            "key_operations": entry_summary.key_operations,
            "decisions": entry_summary.decisions,
            "state_changes": entry_summary.state_changes,
        },
    }
    
    return sfm, metadata

