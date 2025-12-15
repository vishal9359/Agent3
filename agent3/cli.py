from __future__ import annotations

import argparse
from pathlib import Path

from agent3.logging_utils import console


def _path(p: str) -> Path:
    return Path(p).expanduser()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent3", description="C++ RAG + flowchart agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index a C++ project into Chroma")
    p_index.add_argument("--project_path", required=True, type=_path)
    p_index.add_argument("--collection", required=True)
    p_index.add_argument("--scope", type=_path, default=None)
    p_index.add_argument("--chunk_size", type=int, default=2000)
    p_index.add_argument("--chunk_overlap", type=int, default=200)
    p_index.add_argument("--clear", action="store_true", help="Clear the collection first")
    p_index.add_argument(
        "--ollama_base_url",
        default=None,
        help="Override Ollama base URL (default: env OLLAMA_BASE_URL / http://localhost:11434)",
    )
    p_index.add_argument(
        "--embed_model",
        default=None,
        help="Override Ollama embedding model (default: env OLLAMA_EMBED_MODEL)",
    )

    p_ask = sub.add_parser("ask", help="Ask a question using RAG over an indexed collection")
    p_ask.add_argument("--collection", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--k", type=int, default=8)
    p_ask.add_argument(
        "--ollama_base_url",
        default=None,
        help="Override Ollama base URL (default: env OLLAMA_BASE_URL / http://localhost:11434)",
    )
    p_ask.add_argument(
        "--chat_model",
        "--model",
        dest="chat_model",
        default=None,
        help="Override Ollama chat model (default: env OLLAMA_CHAT_MODEL / qwen3)",
    )
    p_ask.add_argument(
        "--embed_model",
        default=None,
        help="Override Ollama embedding model for retrieval (default: env OLLAMA_EMBED_MODEL)",
    )

    p_fc = sub.add_parser("flowchart", help="Generate a Mermaid flowchart for a C++ project/module")
    p_fc.add_argument("--project_path", required=True, type=_path)
    p_fc.add_argument("--scope", required=False, type=_path, default=None, help="Module/scope path within the project (default: project_path)")
    p_fc.add_argument("--out", required=True, type=_path, help="Output .mmd file")
    p_fc.add_argument("--entry", default=None, help="Substring to choose an entry function (optional)")
    p_fc.add_argument("--max_nodes", type=int, default=120)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "index":
        from agent3.indexer import index_project

        index_project(
            project_path=args.project_path,
            collection=args.collection,
            scope=args.scope,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            clear_collection=args.clear,
            ollama_base_url=args.ollama_base_url,
            embed_model=args.embed_model,
        )
        return 0

    if args.cmd == "ask":
        from agent3.rag_graph import ask

        resp = ask(
            collection=args.collection,
            question=args.question,
            k=args.k,
            ollama_base_url=args.ollama_base_url,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
        )
        console.print(resp)
        return 0

    if args.cmd == "flowchart":
        from agent3.flowchart import write_flowchart

        g = write_flowchart(
            project_path=args.project_path,
            scope=args.scope,
            out=args.out,
            entry=args.entry,
            max_nodes=args.max_nodes,
        )
        console.print(
            f"[green]Wrote[/green] {args.out}  (nodes={g.node_count}, edges={g.edge_count})"
        )
        return 0

    parser.error("Unknown command")
    return 2


