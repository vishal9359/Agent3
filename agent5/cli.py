"""Command-line interface for Agent5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent5.logging_utils import console


def _path(p: str) -> Path:
    """Convert string to Path."""
    return Path(p).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent5",
        description="Enhanced C++ Project Understanding and Flowchart Generation Agent",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Index a C++ project using AST-aware chunking",
    )
    index_parser.add_argument(
        "--project_path",
        required=True,
        type=_path,
        help="Path to the C++ project root",
    )
    index_parser.add_argument(
        "--collection",
        required=True,
        help="Name of the collection to create/update",
    )
    index_parser.add_argument(
        "--scope",
        type=_path,
        default=None,
        help="Optional scope path to limit indexing (default: project_path)",
    )
    index_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before indexing",
    )
    index_parser.add_argument(
        "--ollama_base_url",
        default=None,
        help="Ollama base URL (default: env OLLAMA_BASE_URL or http://localhost:11434)",
    )
    index_parser.add_argument(
        "--embed_model",
        default=None,
        help="Embedding model name (default: env OLLAMA_EMBED_MODEL)",
    )
    
    # Ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question about an indexed C++ project",
    )
    ask_parser.add_argument(
        "--collection",
        required=True,
        help="Name of the indexed collection",
    )
    ask_parser.add_argument(
        "--question",
        required=True,
        help="Question to ask",
    )
    ask_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)",
    )
    ask_parser.add_argument(
        "--project_path",
        type=_path,
        default=None,
        help="Project root path (for resolving --focus)",
    )
    ask_parser.add_argument(
        "--focus",
        type=_path,
        default=None,
        help="Focus on a specific file (relative to project_path or absolute)",
    )
    ask_parser.add_argument(
        "--chat_model",
        default=None,
        help="Chat model name (default: env OLLAMA_CHAT_MODEL)",
    )
    ask_parser.add_argument(
        "--embed_model",
        default=None,
        help="Embedding model name (default: env OLLAMA_EMBED_MODEL)",
    )
    ask_parser.add_argument(
        "--ollama_base_url",
        default=None,
        help="Ollama base URL",
    )
    
    # Flowchart command
    flowchart_parser = subparsers.add_parser(
        "flowchart",
        help="Generate a Mermaid flowchart from C++ code",
    )
    flowchart_parser.add_argument(
        "--file",
        required=True,
        type=_path,
        help="Path to the C++ source file",
    )
    flowchart_parser.add_argument(
        "--out",
        required=True,
        type=_path,
        help="Output path for the .mmd file",
    )
    flowchart_parser.add_argument(
        "--function",
        default=None,
        help="Name of the entry function (auto-detect if not specified)",
    )
    flowchart_parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum number of steps in the flowchart (default: 30)",
    )
    flowchart_parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Use LLM for translation (optional, has deterministic fallback)",
    )
    flowchart_parser.add_argument(
        "--chat_model",
        default=None,
        help="Chat model name for LLM translation",
    )
    flowchart_parser.add_argument(
        "--ollama_base_url",
        default=None,
        help="Ollama base URL",
    )
    
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        if args.command == "index":
            from agent5.indexer import index_project
            
            console.print("\n[bold cyan]═══ Agent5: Index C++ Project ═══[/bold cyan]\n")
            
            num_docs = index_project(
                project_path=args.project_path,
                collection=args.collection,
                scope=args.scope,
                clear_collection_first=args.clear,
                ollama_base_url=args.ollama_base_url,
                embed_model=args.embed_model,
            )
            
            console.print(f"\n[bold green]✓ Successfully indexed {num_docs} documents[/bold green]")
            console.print(f"[cyan]Collection:[/cyan] {args.collection}\n")
            return 0
        
        elif args.command == "ask":
            from agent5.rag_system import ask_question
            
            console.print("\n[bold cyan]═══ Agent5: Ask Question ═══[/bold cyan]\n")
            console.print(f"[yellow]Question:[/yellow] {args.question}\n")
            
            answer = ask_question(
                collection=args.collection,
                question=args.question,
                k=args.k,
                chat_model=args.chat_model,
                embed_model=args.embed_model,
                ollama_base_url=args.ollama_base_url,
                focus_file=args.focus,
                project_path=args.project_path,
            )
            
            console.print("[bold green]Answer:[/bold green]")
            console.print(answer)
            console.print()
            return 0
        
        elif args.command == "flowchart":
            from agent5.flowchart import write_flowchart
            
            console.print("\n[bold cyan]═══ Agent5: Generate Flowchart ═══[/bold cyan]\n")
            console.print(f"[cyan]Input:[/cyan] {args.file}")
            console.print(f"[cyan]Output:[/cyan] {args.out}")
            
            if args.function:
                console.print(f"[cyan]Function:[/cyan] {args.function}")
            else:
                console.print("[cyan]Function:[/cyan] Auto-detect")
            
            console.print(f"[cyan]Max steps:[/cyan] {args.max_steps}")
            console.print(f"[cyan]Use LLM:[/cyan] {args.use_llm}")
            console.print()
            
            console.print("[yellow]Extracting Scenario Flow Model (SFM)...[/yellow]")
            
            flowchart = write_flowchart(
                output_path=args.out,
                file_path=args.file,
                function_name=args.function,
                max_steps=args.max_steps,
                use_llm=args.use_llm,
                chat_model=args.chat_model,
                ollama_base_url=args.ollama_base_url,
            )
            
            console.print(
                f"\n[bold green]✓ Flowchart generated successfully[/bold green]"
            )
            console.print(f"[green]  Nodes:[/green] {flowchart.node_count}")
            console.print(f"[green]  Edges:[/green] {flowchart.edge_count}")
            console.print(f"[green]  Output:[/green] {args.out}")
            
            if flowchart.sfm:
                sfm_path = args.out.with_suffix(".sfm.json")
                console.print(f"[green]  SFM (debug):[/green] {sfm_path}")
            
            console.print()
            return 0
        
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if "--debug" in sys.argv:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())

