from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from agent3.cpp_callgraph import CallEdge, FunctionInfo, build_callgraph, resolve_edges
from agent3.config import SETTINGS
from agent3.ollama_compat import get_chat_ollama


@dataclass(frozen=True)
class MermaidGraph:
    mermaid: str
    node_count: int
    edge_count: int


def _escape_label(label: str) -> str:
    # Mermaid node labels can break on quotes/brackets; keep it simple.
    return (
        label.replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
    )


def _prune_graph(
    functions: dict[str, FunctionInfo],
    edges: list[CallEdge],
    *,
    entry: str | None,
    max_nodes: int,
) -> tuple[dict[str, FunctionInfo], list[CallEdge]]:
    if max_nodes <= 0 or len(functions) <= max_nodes:
        return functions, edges

    adj: dict[str, set[str]] = defaultdict(set)
    rev: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e.src_key].add(e.dst_key)
        rev[e.dst_key].add(e.src_key)

    # Choose seeds
    seeds: list[str] = []
    if entry:
        entry_l = entry.lower()
        for k, fn in functions.items():
            if entry_l in fn.label.lower():
                seeds.append(k)

    if not seeds:
        # Fallback: top-degree nodes
        degrees = []
        for k in functions:
            degrees.append((len(adj.get(k, set())) + len(rev.get(k, set())), k))
        degrees.sort(reverse=True)
        seeds = [k for _, k in degrees[: min(5, len(degrees))]]

    keep: set[str] = set()
    q = deque(seeds)
    while q and len(keep) < max_nodes:
        cur = q.popleft()
        if cur in keep:
            continue
        keep.add(cur)
        for nxt in list(adj.get(cur, set())) + list(rev.get(cur, set())):
            if nxt not in keep and len(keep) < max_nodes:
                q.append(nxt)

    pruned_functions = {k: functions[k] for k in keep if k in functions}
    pruned_edges = [e for e in edges if e.src_key in keep and e.dst_key in keep]
    return pruned_functions, pruned_edges


def to_mermaid(
    functions: dict[str, FunctionInfo],
    edges: list[CallEdge],
    *,
    entry: str | None = None,
    max_nodes: int = 120,
) -> MermaidGraph:
    f2, e2 = _prune_graph(functions, edges, entry=entry, max_nodes=max_nodes)

    lines: list[str] = ["flowchart TD"]
    # Declare nodes
    for k in sorted(f2.keys()):
        label = _escape_label(f2[k].label)
        lines.append(f'  {k}["{label}"]')
    # Declare edges
    seen = set()
    for e in e2:
        tup = (e.src_key, e.dst_key)
        if tup in seen:
            continue
        if e.src_key not in f2 or e.dst_key not in f2:
            continue
        seen.add(tup)
        lines.append(f"  {e.src_key} --> {e.dst_key}")

    return MermaidGraph(mermaid="\n".join(lines) + "\n", node_count=len(f2), edge_count=len(seen))


def generate_flowchart_mermaid(
    *,
    project_path: Path,
    scope: Path | None = None,
    entry: str | None = None,
    max_nodes: int = 120,
) -> MermaidGraph:
    functions, edges_by_name = build_callgraph(project_path=project_path, scope=scope)
    edges = resolve_edges(functions, edges_by_name)
    return to_mermaid(functions, edges, entry=entry, max_nodes=max_nodes)


def write_flowchart(
    *,
    project_path: Path,
    scope: Path | None,
    out: Path,
    entry: str | None = None,
    max_nodes: int = 120,
) -> MermaidGraph:
    g = generate_flowchart_mermaid(
        project_path=project_path, scope=scope, entry=entry, max_nodes=max_nodes
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(g.mermaid, encoding="utf-8")
    return g


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # Remove leading fence line and trailing fence if present.
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def _format_context_for_llm(docs, max_chars: int = 30_000) -> str:
    parts: list[str] = []
    total = 0
    for d in docs:
        src = d.metadata.get("relpath") or d.metadata.get("source") or "unknown"
        chunk = d.page_content
        block = f"\n---\nFILE: {src}\n{chunk}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).strip()


def _count_mermaid_edges_nodes(mermaid: str) -> tuple[int, int]:
    # Best-effort counts for CLI display.
    node_ids: set[str] = set()
    edges = 0
    for line in mermaid.splitlines():
        t = line.strip()
        if not t or t.startswith("%%"):
            continue
        if "-->" in t:
            edges += 1
            left = t.split("-->", 1)[0].strip()
            # extract id before any brackets
            node_ids.add(left.split("[", 1)[0].split("(", 1)[0].split("{", 1)[0].strip())
            right = t.split("-->", 1)[1].strip()
            node_ids.add(right.split("[", 1)[0].split("(", 1)[0].split("{", 1)[0].strip())
        else:
            # Node declaration like: A["label"] / A([Start]) / A{Decision}
            head = t.split()[0]
            if any(x in head for x in ("[", "(", "{")):
                node_ids.add(head.split("[", 1)[0].split("(", 1)[0].split("{", 1)[0].strip())
    node_ids.discard("flowchart")
    node_ids.discard("graph")
    return len([n for n in node_ids if n]), edges


SCENARIO_TRANSLATE_SYSTEM_PROMPT = """You are a diagram translator.
You will receive a Scenario Flow Model (SFM) as JSON. The SFM is authoritative.

Task: translate the SFM to Mermaid flowchart code ONLY.

Output rules (STRICT):
- Output ONLY Mermaid, starting with: flowchart TD
- No markdown, no bullets, no headings, no explanations.
- Use these shapes:
  - Terminator: id([Label])
  - Process:    id["Label"]
  - Decision:   id{"Label?"}
  - I/O:        id[/"Label"/]
- Every node must be explicitly declared (typed) using one of the shapes above.
- Edges may include labels using: -->|YES| or -->|NO| etc.
"""


def _extract_mermaid_only(text: str) -> str | None:
    """
    Extract Mermaid flowchart block starting at the first 'flowchart'.
    Returns None if no Mermaid found.
    """
    t = _strip_code_fences(text)
    lines = t.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("flowchart"):
            start_idx = i
            break
    if start_idx is None:
        return None
    mermaid = "\n".join(lines[start_idx:]).strip()
    # Reject obvious non-mermaid prose / markdown after extraction
    bad_markers = ("###", "- **", "- ", "* ", "```")
    for l in mermaid.splitlines():
        if l.strip().startswith(bad_markers):
            return None
    return mermaid + "\n"


def _ensure_start_end_terminators(mermaid: str) -> str:
    lines = mermaid.splitlines()
    if not lines:
        return "flowchart TD\n  start([Start])\n  end([End])\n"
    if not lines[0].lstrip().startswith("flowchart"):
        lines = ["flowchart TD"] + lines
    s = "\n".join(lines)
    if "start([" not in s:
        lines = [lines[0], "  start([Start])"] + lines[1:]
        s = "\n".join(lines)
    if "end([" not in s:
        lines.append("  end([End])")
    return "\n".join(lines).strip() + "\n"


def _normalize_mermaid_common_syntax(mermaid: str) -> str:
    """
    Normalize Mermaid that some models output but Mermaid Live rejects.
    - Convert terminators like start(["Start"]) -> start([Start])
    - Convert decisions like d1{"cond?"} -> d1{cond?}
    """
    import re

    out = mermaid

    # start(["Start"]) / end(["End"]) -> start([Start]) / end([End])
    out = re.sub(r'(\bstart)\(\["([^"]+)"\]\)', r"\1([\2])", out)
    out = re.sub(r'(\bend)\(\["([^"]+)"\]\)', r"\1([\2])", out)

    # Generic terminator with quoted label: id(["Label"]) -> id([Label])
    out = re.sub(r'(\b[A-Za-z_][A-Za-z0-9_]*)\(\["([^"]+)"\]\)', r"\1([\2])", out)

    # Decision with quoted label: id{"Label"} -> id{Label}
    out = re.sub(r'(\b[A-Za-z_][A-Za-z0-9_]*)\{\s*"([^"]+)"\s*\}', r"\1{\2}", out)

    return out


def _validate_mermaid_shapes_only(mermaid: str) -> None:
    """
    Fail if the diagram contains untyped nodes or markdown/prose.
    This is intentionally strict to keep output deterministic.
    """
    import re

    def _is_bare_id(s: str) -> bool:
        return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s) is not None

    def _parse_edge_endpoints(t: str) -> tuple[str, str] | None:
        if "-->" not in t:
            return None
        left, right = t.split("-->", 1)
        left = left.strip()
        right = right.strip()
        # Handle edge labels:  a -->|YES| b
        if right.startswith("|"):
            # find closing pipe
            try:
                _, rest = right[1:].split("|", 1)
            except ValueError:
                return None
            right = rest.strip()
        # Reject inline node definitions in edges (e.g. end([End]) or x["lbl"]).
        if any(ch in left for ch in "[](){}\"/") or any(ch in right for ch in "[](){}\"/"):
            return None
        if not _is_bare_id(left) or not _is_bare_id(right):
            return None
        return left, right

    for line in mermaid.splitlines():
        t = line.strip()
        if not t or t.startswith("%%") or t.startswith("flowchart"):
            continue
        if t.startswith(("#", "- ", "* ")):
            raise ValueError("Mermaid output contains markdown/prose.")
        # Accept edges and explicit node declarations only.
        if "-->" in t:
            if _parse_edge_endpoints(t) is None:
                raise ValueError("Mermaid output contains inline node definitions in an edge.")
            continue
        # Node declarations must include a shape marker.
        if not any(x in t for x in ('["', "([", "{", '[/"')):
            raise ValueError("Mermaid output contains an untyped node declaration.")


def _set_parser_language(parser) -> None:
    """
    Minimal tree-sitter-cpp language setup (mirrors agent3.cpp_callgraph behavior).
    """
    from tree_sitter import Language
    from tree_sitter_cpp import language as cpp_language

    raw = cpp_language()
    if isinstance(raw, Language):
        lang = raw
    else:
        lang = Language(raw)  # type: ignore[arg-type]

    if hasattr(parser, "set_language"):
        parser.set_language(lang)  # type: ignore[attr-defined]
    else:
        parser.language = lang  # type: ignore[assignment]


def _node_text(source_bytes: bytes, node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _first_ident_text(source_bytes: bytes, node) -> str | None:
    """
    Best-effort identifier extraction from a declarator subtree.
    """
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
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
            for c in reversed(n.children):
                if c.type in {"identifier", "field_identifier"}:
                    t = _node_text(source_bytes, c).strip()
                    if t:
                        return t
        for c in reversed(n.children):
            stack.append(c)
    return None


def _normalize_ws(s: str, limit: int = 140) -> str:
    out = " ".join(s.replace("\t", " ").split())
    return out if len(out) <= limit else out[: limit - 1] + "…"


def _callee_name_from_call(source_bytes: bytes, call_node) -> str | None:
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    # identifier / field_expression / qualified_identifier
    if fn.type in {"identifier", "field_identifier"}:
        t = _node_text(source_bytes, fn).strip()
        return t if t else None
    if fn.type == "field_expression":
        for c in reversed(fn.children):
            if c.type in {"field_identifier", "identifier"}:
                t = _node_text(source_bytes, c).strip()
                if t:
                    return t
    return _first_ident_text(source_bytes, fn)


def _is_noise_call(name: str) -> bool:
    n = name.lower()
    noise = (
        "log",
        "spdlog",
        "printf",
        "fprintf",
        "cout",
        "cerr",
        "trace",
        "metric",
        "stats",
        "telemetry",
        "debug",
        "info",
        "warn",
        "error",
    )
    return any(tok in n for tok in noise)


class _SFMBuilder:
    def __init__(self, *, max_steps: int):
        self.max_steps = max_steps
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self._i = 0
        self._steps = 0

        self.add_node("start", "terminator", "Start")
        self.add_node("end", "terminator", "End")

    def _new_id(self, prefix: str) -> str:
        self._i += 1
        return f"{prefix}{self._i}"

    def add_node(self, node_id: str, node_type: str, label: str) -> str:
        self.nodes[node_id] = {"id": node_id, "type": node_type, "label": label}
        return node_id

    def add_process(self, label: str) -> str:
        if self._steps >= self.max_steps:
            # Signal "stop" to callers; they must terminate.
            return ""
        self._steps += 1
        nid = self._new_id("p")
        return self.add_node(nid, "process", label)

    def add_decision(self, label: str) -> str:
        if self._steps >= self.max_steps:
            return ""
        self._steps += 1
        nid = self._new_id("d")
        return self.add_node(nid, "decision", label)

    def add_edge(self, src: str, dst: str, label: str | None = None) -> None:
        e = {"src": src, "dst": dst}
        if label:
            e["label"] = label
        self.edges.append(e)

    def to_json(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": self.edges}


def _sfm_to_mermaid(sfm: dict) -> str:
    nodes = {n["id"]: n for n in sfm.get("nodes", [])}
    edges = sfm.get("edges", [])
    lines: list[str] = ["flowchart TD"]

    # Declare nodes
    for nid, n in nodes.items():
        t = n.get("type")
        lbl = _escape_label(str(n.get("label", "")))
        if t == "terminator":
            lines.append(f"  {nid}([{lbl}])")
        elif t == "decision":
            lines.append(f"  {nid}{{{lbl}}}")
        elif t == "io":
            lines.append(f"  {nid}[/\"{lbl}\"/]")
        else:
            lines.append(f"  {nid}[\"{lbl}\"]")

    # Declare edges
    for e in edges:
        src = e["src"]
        dst = e["dst"]
        lab = e.get("label")
        if lab:
            lines.append(f"  {src} -->|{_escape_label(str(lab))}| {dst}")
        else:
            lines.append(f"  {src} --> {dst}")

    mermaid = "\n".join(lines).strip() + "\n"
    mermaid = _ensure_start_end_terminators(mermaid)
    _validate_mermaid_shapes_only(mermaid)
    return mermaid


def _translate_sfm_with_llm(
    *,
    sfm: dict,
    max_steps: int,
    chat_model: str,
    ollama_base_url: str,
) -> str | None:
    """
    LLM is allowed ONLY after SFM exists. If the LLM violates output rules, return None.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_chat_ollama(model=chat_model, base_url=ollama_base_url)
    user_msg = (
        "Scenario Flow Model (JSON):\n"
        f"{sfm}\n\n"
        f"Constraint: keep ~<= {max_steps} steps.\n"
    )
    resp = llm.invoke([SystemMessage(content=SCENARIO_TRANSLATE_SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    mermaid = _extract_mermaid_only(getattr(resp, "content", str(resp)) or "")
    if not mermaid:
        return None
    try:
        mermaid = _normalize_mermaid_common_syntax(mermaid)
        mermaid = _ensure_start_end_terminators(mermaid)
        _validate_mermaid_shapes_only(mermaid)
        return mermaid
    except Exception:
        return None


def _tokenize(s: str) -> set[str]:
    import re

    # Split on non-alnum and camelCase boundaries
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    parts = re.split(r"[^a-zA-Z0-9]+", s2)
    return {p.lower() for p in parts if p and len(p) >= 3}


def _collect_functions(source_bytes: bytes, root) -> list[tuple[str, str, object, int]]:
    """
    Returns list of (short_name, qualified_name, node, approx_size_chars)
    """
    funcs: list[tuple[str, str, object, int]] = []

    # Maintain a simple scope stack for namespaces/classes to build qualified names.
    scope_stack: list[object] = []

    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        n, state = stack.pop()

        is_scope = n.type in {
            "namespace_definition",
            "class_specifier",
            "struct_specifier",
            "union_specifier",
        }

        if state == 0:
            if is_scope:
                scope_stack.append(n)
            if n.type in {"function_definition", "constructor_or_destructor_definition"}:
                decl = n.child_by_field_name("declarator") or n.child_by_field_name("name")
                name = _first_ident_text(source_bytes, decl) if decl is not None else None
                if name:
                    # Resolve scope names
                    scopes: list[str] = []
                    for sc in scope_stack:
                        name_node = sc.child_by_field_name("name")
                        if name_node is None:
                            continue
                        t = _node_text(source_bytes, name_node).strip()
                        if t:
                            scopes.append(t)
                    qname = "::".join(scopes + [name]) if scopes else name

                    body = n.child_by_field_name("body")
                    size = 0
                    if body is not None:
                        try:
                            size = max(0, body.end_byte - body.start_byte)
                        except Exception:
                            size = 0
                    funcs.append((name, qname, n, size))

            # Post-order marker
            stack.append((n, 1))
            for c in reversed(getattr(n, "children", [])):
                stack.append((c, 0))
        else:
            if is_scope and scope_stack and scope_stack[-1] is n:
                scope_stack.pop()

    return funcs


def _score_entry_candidate(
    *,
    scenario: str,
    focus_path: Path,
    short_name: str,
    qualified_name: str,
    size_bytes: int,
) -> int:
    score = 0

    # Strong preference for main when present.
    if short_name == "main":
        score += 100

    # Token overlap with scenario + filename.
    scen_toks = _tokenize(scenario)
    file_toks = _tokenize(focus_path.stem)
    fn_toks = _tokenize(qualified_name)
    overlap = (scen_toks | file_toks) & fn_toks
    score += 5 * len(overlap)

    # Common entry-like verbs.
    entryish = {"execute", "run", "handle", "process", "start", "create", "open", "init", "parse"}
    if any(t in fn_toks for t in entryish):
        score += 12

    # Bigger bodies are more likely to be scenario orchestration.
    if size_bytes >= 2000:
        score += 6
    elif size_bytes >= 800:
        score += 3

    return score


def _select_entry_function(
    source_bytes: bytes,
    root,
    *,
    scenario: str,
    focus_path: Path,
    entry_fn: str | None,
) -> tuple[str, object]:
    """
    Return (function_name, function_node). Raises if ambiguous.
    """
    funcs = _collect_functions(source_bytes, root)

    if not funcs:
        raise RuntimeError("No function definitions found in focus file.")

    if entry_fn:
        for short, qname, node, _sz in funcs:
            if entry_fn == qname or entry_fn == short:
                return qname, node
        raise RuntimeError(f"Entry function '{entry_fn}' not found in focus file.")

    if len(funcs) == 1:
        return funcs[0][1], funcs[0][2]

    scored: list[tuple[int, str, str, object]] = []
    for short, qname, node, sz in funcs:
        s = _score_entry_candidate(
            scenario=scenario, focus_path=focus_path, short_name=short, qualified_name=qname, size_bytes=sz
        )
        scored.append((s, qname, short, node))

    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    # Confidence gate: refuse if we can't choose confidently.
    if best[0] < 10:
        tops = ", ".join([f"{s}:{q}" for s, q, _short, _node in scored[:10]])
        raise RuntimeError(
            "Unable to auto-detect scenario entry function confidently.\n"
            "Please set --focus to a file that contains the scenario entrypoint (ideally one main function), "
            "or optionally provide --entry_fn to override.\n"
            f"Candidates (score:name): {tops}"
        )
    if second is not None and (best[0] - second[0]) <= 3:
        tops = ", ".join([f"{s}:{q}" for s, q, _short, _node in scored[:10]])
        raise RuntimeError(
            "Ambiguous scenario entry function in focus file (auto-detection not confident).\n"
            "Please set --focus to a narrower file, or optionally provide --entry_fn.\n"
            f"Candidates (score:name): {tops}"
        )

    return best[1], best[3]


def _iter_statement_nodes(block_node) -> list[object]:
    # compound_statement children include '{', '}' and statements; keep statement-like children.
    out: list[object] = []
    for c in getattr(block_node, "children", []):
        if c.type in {"{", "}"}:
            continue
        out.append(c)
    return out


def _stmt_is_call_expr(stmt) -> object | None:
    # expression_statement -> call_expression
    if stmt.type == "expression_statement":
        for c in stmt.children:
            if c.type == "call_expression":
                return c
    return None


def _build_sfm_from_function(
    *,
    source_bytes: bytes,
    fn_node,
    max_steps: int,
) -> dict:
    b = _SFMBuilder(max_steps=max_steps)

    body = fn_node.child_by_field_name("body")
    if body is None:
        raise RuntimeError("Selected entry function has no body.")

    prev = "start"

    def add_step(label: str) -> None:
        nonlocal prev
        if prev == "end":
            return
        nid = b.add_process(label)
        if not nid:
            b.add_edge(prev, "end")
            prev = "end"
            return
        b.add_edge(prev, nid)
        prev = nid

    def add_return(label: str) -> None:
        nonlocal prev
        if prev == "end":
            return
        nid = b.add_process(label)
        if not nid:
            b.add_edge(prev, "end")
            prev = "end"
            return
        b.add_edge(prev, nid)
        b.add_edge(nid, "end")
        prev = "end"

    def handle_stmt(stmt) -> None:
        nonlocal prev
        if b._steps >= b.max_steps:
            b.add_edge(prev, "end")
            prev = "end"
            return

        # if (...) { ... } else { ... }
        if stmt.type == "if_statement":
            cond = stmt.child_by_field_name("condition")
            cond_txt = _normalize_ws(_node_text(source_bytes, cond)) if cond is not None else "condition"
            d = b.add_decision(cond_txt.rstrip("?") + "?")
            if not d:
                b.add_edge(prev, "end")
                prev = "end"
                return
            b.add_edge(prev, d)

            merge = b.add_process("Continue")
            if not merge:
                # No room to model; terminate.
                b.add_edge(d, "end")
                prev = "end"
                return

            cons = stmt.child_by_field_name("consequence")
            alt = stmt.child_by_field_name("alternative")

            # THEN branch
            prev = d
            if cons is not None:
                _handle_stmt_or_block(cons, edge_label="YES")
            else:
                b.add_edge(d, merge, "YES")
            # if branch ended at end, skip merge edge
            if prev != "end":
                b.add_edge(prev, merge)

            # ELSE
            prev = d
            if alt is not None:
                _handle_stmt_or_block(alt, edge_label="NO")
                if prev != "end":
                    b.add_edge(prev, merge)
            else:
                b.add_edge(d, merge, "NO")

            prev = merge
            return

        # return ...
        if stmt.type == "return_statement":
            txt = _normalize_ws(_node_text(source_bytes, stmt))
            add_return(txt if txt else "Return")
            prev = "end"
            return

        if stmt.type in {"throw_statement"}:
            txt = _normalize_ws(_node_text(source_bytes, stmt))
            add_return(txt if txt else "Throw")
            prev = "end"
            return

        # while/for loops (simplified)
        if stmt.type in {"while_statement", "for_statement"}:
            cond = stmt.child_by_field_name("condition")
            cond_txt = _normalize_ws(_node_text(source_bytes, cond)) if cond is not None else "loop condition"
            d = b.add_decision(cond_txt.rstrip("?") + "?")
            b.add_edge(prev, d)
            merge = b.add_process("Continue")

            body_n = stmt.child_by_field_name("body")
            if body_n is not None:
                prev = d
                _handle_stmt_or_block(body_n, edge_label="YES")
                if prev != "end":
                    b.add_edge(prev, d, "loop")
            b.add_edge(d, merge, "NO")
            prev = merge
            return

        # switch_statement
        if stmt.type == "switch_statement":
            val = stmt.child_by_field_name("value")
            val_txt = _normalize_ws(_node_text(source_bytes, val)) if val is not None else "switch"
            d = b.add_decision(f"{val_txt}?")
            b.add_edge(prev, d)
            merge = b.add_process("Continue")
            body_n = stmt.child_by_field_name("body")
            if body_n is not None:
                # Very simplified: treat each case as a branch to merge.
                for c in body_n.children:
                    if c.type in {"case_statement", "default_statement"}:
                        lab = _normalize_ws(_node_text(source_bytes, c.children[0])) if c.children else "case"
                        prev = d
                        b.add_edge(d, merge, lab)
                b.add_edge(d, merge, "default")
            prev = merge
            return

        # calls / assignments / declarations as process steps
        call = _stmt_is_call_expr(stmt)
        if call is not None:
            callee = _callee_name_from_call(source_bytes, call) or "call"
            if _is_noise_call(callee):
                return
            add_step(f"Call {callee}()")
            return

        if stmt.type in {"declaration"}:
            txt = _normalize_ws(_node_text(source_bytes, stmt))
            add_step(txt)
            return

        if stmt.type in {"expression_statement"}:
            txt = _normalize_ws(_node_text(source_bytes, stmt))
            if txt:
                add_step(txt)
            return

        # Fallback: keep a compact representation
        txt = _normalize_ws(_node_text(source_bytes, stmt))
        if txt:
            add_step(txt)

    def _handle_stmt_or_block(node, *, edge_label: str | None = None) -> None:
        nonlocal prev
        if prev == "end":
            return
        # Connect from current prev to first node in this branch with label
        if node.type == "compound_statement":
            stmts = _iter_statement_nodes(node)
            if not stmts:
                return
            first = stmts[0]
            # create a tiny "branch" anchor
            anchor = b.add_process("Branch")
            if not anchor:
                b.add_edge(prev, "end", edge_label)
                prev = "end"
                return
            b.add_edge(prev, anchor, edge_label)
            prev = anchor
            for st in stmts:
                handle_stmt(st)
        else:
            anchor = b.add_process("Branch")
            if not anchor:
                b.add_edge(prev, "end", edge_label)
                prev = "end"
                return
            b.add_edge(prev, anchor, edge_label)
            prev = anchor
            handle_stmt(node)

    # walk top-level statements
    for st in _iter_statement_nodes(body):
        if prev == "end":
            break
        handle_stmt(st)

    # Normal fallthrough to End
    if prev != "end":
        b.add_edge(prev, "end")

    sfm = b.to_json()
    # Validate basic invariants
    starts = [n for n in sfm["nodes"] if n["id"] == "start"]
    ends = [n for n in sfm["nodes"] if n["id"] == "end"]
    if len(starts) != 1 or len(ends) < 1:
        raise RuntimeError("Failed to build a valid Scenario Flow Model (missing Start/End).")
    return sfm


def generate_scenario_flowchart_mermaid(
    *,
    project_path: Path,
    scenario: str,
    collection: str | None = None,
    focus: Path | None = None,
    entry_fn: str | None = None,
    k: int = 12,
    detail: str = "high",
    max_steps: int = 26,
    use_llm: bool = True,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidGraph:
    """
    Scenario-driven flowchart:
    - Build deterministic Scenario Flow Model (SFM) from AST + control structures.
    - If SFM cannot be built, FAIL FAST and DO NOT call the LLM.
    - Optionally translate SFM -> Mermaid using the LLM (translator only); otherwise do deterministic translation.
    """
    project_path = project_path.resolve()
    base_url = ollama_base_url or SETTINGS.ollama_base_url
    if focus is None:
        raise RuntimeError("Scenario flowchart requires --focus to build a deterministic Scenario Flow Model.")

    fp = (project_path / focus).resolve() if not focus.is_absolute() else focus.resolve()
    try:
        source = fp.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        raise RuntimeError(f"Unable to read focus file: {fp}") from e

    from tree_sitter import Parser

    parser = Parser()
    _set_parser_language(parser)
    src_bytes = source.encode("utf-8", errors="ignore")
    tree = parser.parse(src_bytes)
    root = tree.root_node

    _fn_name, fn_node = _select_entry_function(
        src_bytes, root, scenario=scenario, focus_path=fp, entry_fn=entry_fn
    )

    # Deterministic SFM (must succeed; otherwise fail fast).
    sfm = _build_sfm_from_function(source_bytes=src_bytes, fn_node=fn_node, max_steps=max_steps)

    # Translate SFM -> Mermaid (LLM optional, translator-only).
    mermaid: str | None = None
    if use_llm:
        mermaid = _translate_sfm_with_llm(
            sfm=sfm,
            max_steps=max_steps,
            chat_model=chat_model or SETTINGS.ollama_chat_model,
            ollama_base_url=base_url,
        )
    if not mermaid:
        mermaid = _sfm_to_mermaid(sfm)

    nodes, edges = _count_mermaid_edges_nodes(mermaid)
    return MermaidGraph(mermaid=mermaid, node_count=nodes, edge_count=edges)


def write_scenario_flowchart(
    *,
    project_path: Path,
    out: Path,
    scenario: str,
    collection: str | None = None,
    focus: Path | None = None,
    entry_fn: str | None = None,
    k: int = 12,
    detail: str = "high",
    max_steps: int = 26,
    use_llm: bool = True,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidGraph:
    g = generate_scenario_flowchart_mermaid(
        project_path=project_path,
        scenario=scenario,
        collection=collection,
        focus=focus,
        entry_fn=entry_fn,
        k=k,
        detail=detail,
        max_steps=max_steps,
        use_llm=use_llm,
        chat_model=chat_model,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(g.mermaid, encoding="utf-8")
    return g


