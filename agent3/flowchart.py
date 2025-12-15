from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from agent3.cpp_callgraph import CallEdge, FunctionInfo, build_callgraph, resolve_edges
from agent3.config import SETTINGS
from agent3.ollama_compat import get_chat_ollama
from agent3.vectorstore import get_vectorstore


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


SCENARIO_SYSTEM_PROMPT = """You are a senior C++ engineer who draws scenario-driven flowcharts.
You will be given a user scenario (what the user is trying to do) and code context snippets.

Goal: output a Mermaid flowchart that looks like a whiteboard execution flow (Start/End, decisions, key steps),
NOT a call graph. Model the runtime steps a user triggers for the scenario.

Rules:
- Output ONLY Mermaid (no prose), starting with: flowchart TD
- Must include Start and End nodes.
- Use decision diamonds { } when behavior branches.
- Use short, meaningful step labels (CLI parse args, validate, call backend API, handle error, print result, etc.).
- If the context is insufficient, include a node that says what code is missing (e.g., "Need: <file/function>") and end.
"""


def generate_scenario_flowchart_mermaid(
    *,
    project_path: Path,
    scenario: str,
    collection: str | None = None,
    focus: Path | None = None,
    k: int = 12,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidGraph:
    """
    Generate a scenario-driven (execution) flowchart using LLM + code context.
    - If `collection` is provided, uses indexed retrieval.
    - If `focus` is provided, injects that file's full content first (highest signal).
    """
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, SystemMessage

    project_path = project_path.resolve()
    base_url = ollama_base_url or SETTINGS.ollama_base_url
    llm = get_chat_ollama(model=chat_model or SETTINGS.ollama_chat_model, base_url=base_url)

    docs: list[Document] = []
    if focus is not None:
        fp = (project_path / focus).resolve() if not focus.is_absolute() else focus.resolve()
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        rel = str(fp.relative_to(project_path)).replace("\\", "/") if project_path in fp.parents else str(fp)
        if text:
            docs.append(Document(page_content=text, metadata={"relpath": rel, "source": str(fp)}))

    if collection:
        vs = get_vectorstore(collection, embed_model=embed_model, ollama_base_url=base_url)
        # Pull a bit more than k if focus wasn't provided.
        kk = k if docs else max(k, 16)
        try:
            docs.extend(vs.similarity_search(f"{scenario}\nFOCUS: {focus or ''}", k=kk))
        except Exception:
            pass
        # Try MMR for diversity.
        try:
            mmr = getattr(vs, "max_marginal_relevance_search", None)
            if callable(mmr):
                docs.extend(mmr(f"{scenario}\nFOCUS: {focus or ''}", k=kk, fetch_k=max(32, kk * 4)))
        except Exception:
            pass

    # Dedupe by relpath + content hash
    seen: set[tuple[str, int]] = set()
    deduped: list[Document] = []
    for d in docs:
        src = str(d.metadata.get("relpath") or d.metadata.get("source") or "unknown")
        key = (src, hash(d.page_content))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    context = _format_context_for_llm(deduped)
    user_msg = f"Scenario:\n{scenario}\n\nProject root:\n{project_path}\n\nContext:\n{context}\n"
    resp = llm.invoke([SystemMessage(content=SCENARIO_SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    mermaid = _strip_code_fences(getattr(resp, "content", str(resp)))
    if not mermaid.lstrip().startswith("flowchart"):
        mermaid = "flowchart TD\n" + mermaid.strip() + "\n"

    # Ensure Start/End exist (best-effort).
    if "Start" not in mermaid and "start" not in mermaid:
        mermaid = "flowchart TD\n  start([Start])\n" + "\n".join(mermaid.splitlines()[1:]) + "\n"
    if "End" not in mermaid and "end" not in mermaid:
        mermaid = mermaid.rstrip() + "\n  end([End])\n"

    nodes, edges = _count_mermaid_edges_nodes(mermaid)
    return MermaidGraph(mermaid=mermaid if mermaid.endswith("\n") else mermaid + "\n", node_count=nodes, edge_count=edges)


def write_scenario_flowchart(
    *,
    project_path: Path,
    out: Path,
    scenario: str,
    collection: str | None = None,
    focus: Path | None = None,
    k: int = 12,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidGraph:
    g = generate_scenario_flowchart_mermaid(
        project_path=project_path,
        scenario=scenario,
        collection=collection,
        focus=focus,
        k=k,
        chat_model=chat_model,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(g.mermaid, encoding="utf-8")
    return g


