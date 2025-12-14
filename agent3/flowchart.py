from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from agent3.cpp_callgraph import CallEdge, FunctionInfo, build_callgraph, resolve_edges


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


