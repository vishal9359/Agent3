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
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser
from tree_sitter_cpp import language as cpp_language


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


def _classify_call(callee: str) -> tuple[bool, str]:
    """
    Classify a function call into a semantic action.
    
    Returns:
        (include: bool, semantic_label: str)
        
    Scenario boundary rule: Include only business-relevant calls.
    Collapse the call into a single semantic step (never descend).
    """
    c = callee.strip()
    lc = c.lower()
    
    if not c or _is_noise_call(c):
        return False, ""
    
    # Map common verbs to semantic actions
    verb_map = [
        ("parse", "Parse"),
        ("check", "Validate"),
        ("validate", "Validate"),
        ("isvalid", "Validate"),
        ("verify", "Validate"),
        ("get", "Lookup"),
        ("fetch", "Fetch"),
        ("retrieve", "Retrieve"),
        ("set", "Set"),
        ("update", "Update"),
        ("add", "Add"),
        ("insert", "Insert"),
        ("remove", "Remove"),
        ("delete", "Delete"),
        ("erase", "Erase"),
        ("create", "Create"),
        ("make", "Create"),
        ("build", "Build"),
        ("open", "Open"),
        ("close", "Close"),
        ("init", "Initialize"),
        ("start", "Start"),
        ("stop", "Stop"),
        ("execute", "Execute"),
        ("run", "Run"),
        ("handle", "Handle"),
        ("process", "Process"),
        ("send", "Send"),
        ("receive", "Receive"),
        ("read", "Read"),
        ("write", "Write"),
        ("load", "Load"),
        ("save", "Save"),
    ]
    
    for key, verb in verb_map:
        if key in lc:
            # Extract object from function name
            obj = re.sub(key, "", c, flags=re.IGNORECASE)
            obj = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", obj)
            obj = re.sub(r"[_\-]+", " ", obj)
            obj = " ".join(obj.split()).strip()
            
            if obj:
                return True, f"{verb} {obj}".strip()
            else:
                return True, verb
    
    # Default: exclude unknown calls (treat as utility)
    return False, ""


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


def _should_include_declaration(text: str) -> bool:
    """
    Check if a declaration should be included in the scenario.
    
    Scenario boundary rule: Include argument parsing, config, important state.
    """
    t = text.lower()
    include_keywords = ("argv", "argc", "arg", "option", "param", "config", "ret", "result", "status")
    return any(kw in t for kw in include_keywords)


class SFMBuilder:
    """
    Builder for Scenario Flow Models.
    
    Implements deterministic, rule-based extraction with strict validation.
    """
    
    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps
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
) -> ScenarioFlowModel:
    """
    Extract a Scenario Flow Model from a C++ function.
    
    This is the core deterministic extraction that MUST succeed before any LLM is called.
    
    Args:
        source_code: C++ source code
        function_name: Name of the entry function (auto-detect if None)
        max_steps: Maximum number of steps in the scenario
        
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
        raise RuntimeError(
            f"Cannot find entry function{' ' + function_name if function_name else ''} in source code"
        )
    
    # Build SFM from the function
    builder = SFMBuilder(max_steps=max_steps)
    _extract_from_function_body(src_bytes, fn_node, builder)
    
    return builder.build()


def _find_function(source_bytes: bytes, root: Node, name: str | None) -> Node | None:
    """Find a function definition node by name."""
    functions = []
    
    for child in root.children:
        if child.type in {"function_definition", "constructor_or_destructor_definition"}:
            functions.append(child)
    
    if not functions:
        return None
    
    if name is None:
        # Return first function if no name specified
        return functions[0]
    
    # Find by name
    for fn in functions:
        decl = fn.child_by_field_name("declarator") or fn.child_by_field_name("name")
        if decl:
            fn_name = _get_identifier(source_bytes, decl)
            if fn_name == name:
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
            include, label = _classify_call(callee)
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
    
    if not _should_include_declaration(text):
        return frontier
    
    label = _sanitize_label(text)
    nid = builder.add_process(label)
    if not nid:
        _connect_frontier(builder, frontier, "end", incoming_label)
        return []
    
    _connect_frontier(builder, frontier, nid, incoming_label)
    return [nid]

