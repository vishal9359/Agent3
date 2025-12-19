"""
CLI for Agent5 V4: DocAgent-Inspired Bottom-Up Semantic Aggregation

This CLI provides the agent5-v4 command for generating flowcharts
using the V4 pipeline with bottom-up semantic aggregation.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from agent5.logging_utils import console, get_logger, setup_logging

logger = get_logger(__name__)


def _path(p: str) -> Path:
    """Convert string to Path."""
    return Path(p).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for V4 pipeline."""
    parser = argparse.ArgumentParser(
        prog="agent5-v4",
        description="Agent5 V4: DocAgent-Inspired Bottom-Up Semantic Aggregation for C++ Flowcharts",
    )
    
    parser.add_argument(
        "--project-root",
        type=_path,
        default=None,
        help="Root path that defines the project boundary for AST/CFG/semantics (default: current working directory).",
    )
    
    parser.add_argument(
        "--project-path",
        required=True,
        type=_path,
        help="Path to the C++ project root (defines analysis scope)",
    )
    
    parser.add_argument(
        "--entry-function",
        default=None,
        help="Entry function name (e.g., 'CreateVolume'). If ambiguous, use with --entry-file. If omitted, auto-detects.",
    )
    
    parser.add_argument(
        "--entry-file",
        type=_path,
        default=None,
        help="File containing entry function (for disambiguation). This does NOT limit analysis scope.",
    )
    
    parser.add_argument(
        "--detail-level",
        choices=["high", "medium", "deep"],
        default="medium",
        help="Detail level: 'high' (business-level only), 'medium' (default, includes validations and state changes), 'deep' (expanded critical sub-operations)",
    )
    
    parser.add_argument(
        "--out",
        required=True,
        type=_path,
        help="Output path for the .mmd file",
    )
    
    parser.add_argument(
        "--include-paths",
        default=None,
        help="Comma-separated include paths for Clang (e.g., '/usr/include,/usr/local/include')",
    )
    
    parser.add_argument(
        "--llm-model",
        default="llama3.2:3b",
        help="Ollama LLM model to use (default: llama3.2:3b)",
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM (use deterministic fallbacks only)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves intermediate artifacts to project_path/output/)",
    )
    
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama base URL (default: http://localhost:11434)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for V4 CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    try:
        from agent5.pipeline import generate_flowchart_from_project
        
        console.print("\n[bold cyan]═══ Agent5 V4: DocAgent-Inspired Pipeline ═══[/bold cyan]\n")
        
        # Display configuration
        console.print("[bold]Configuration:[/bold]")
        # Determine project root (hard AST boundary)
        project_root = args.project_root or args.project_path or Path.cwd()
        if not project_root.is_dir():
            console.print(f"[bold red]Error:[/bold red] Invalid project root (not a directory): {project_root}")
            return 1

        console.print(f"  [cyan]Project root:[/cyan] {project_root}")
        console.print(f"  [cyan]Project path:[/cyan] {args.project_path}")
        console.print(f"  [cyan]Entry function:[/cyan] {args.entry_function or 'Auto-detect'}")
        if args.entry_file:
            console.print(f"  [cyan]Entry file:[/cyan] {args.entry_file}")
            console.print(f"  [yellow]  → Entry file is used ONLY for disambiguation[/yellow]")
            console.print(f"  [yellow]  → Analysis scope = entire project path[/yellow]")
        console.print(f"  [cyan]Detail level:[/cyan] {args.detail_level}")
        console.print(f"  [cyan]LLM enabled:[/cyan] {not args.no_llm}")
        if not args.no_llm:
            console.print(f"  [cyan]LLM model:[/cyan] {args.llm_model}")
        console.print(f"  [cyan]Debug mode:[/cyan] {args.debug}")
        console.print(f"  [cyan]Output:[/cyan] {args.out}")
        
        # Build include paths
        include_paths = None
        if args.include_paths:
            include_paths = [Path(p.strip()) for p in args.include_paths.split(',')]
            console.print(f"  [cyan]Include paths:[/cyan] {args.include_paths}")
        
        console.print()
        
        # Run pipeline
        console.print("[bold yellow]Running 6-Stage Pipeline:[/bold yellow]")
        console.print("  [yellow]Stage 1:[/yellow] Full AST Construction (Clang)")
        console.print("  [yellow]Stage 2:[/yellow] Leaf-Level Semantic Extraction")
        console.print("  [yellow]Stage 3:[/yellow] Bottom-Up Semantic Aggregation")
        console.print("  [yellow]Stage 4:[/yellow] Scenario Flow Model Construction")
        console.print("  [yellow]Stage 5:[/yellow] Detail-Level Filtering")
        console.print("  [yellow]Stage 6:[/yellow] Mermaid Translation")
        console.print()
        
        mermaid = generate_flowchart_from_project(
            project_path=project_root,
            entry_function=args.entry_function,
            entry_file=args.entry_file,
            detail_level=args.detail_level,
            output_path=args.out,
            chat_model=args.llm_model,
            use_llm=not args.no_llm,
            include_paths=include_paths,
            debug=args.debug,
        )
        
        console.print()
        console.print("[bold green]✓ Flowchart generated successfully![/bold green]")
        console.print(f"[green]  Output:[/green] {args.out}")
        
        if args.debug:
            debug_dir = args.project_path / "debug"
            console.print(f"[green]  Debug artifacts:[/green] {debug_dir}/")
            console.print(f"    • ast_context.json")
            console.print(f"    • function_summary.json")
            console.print(f"    • sfm.json")
        
        console.print()
        
        # Preview flowchart
        console.print("[bold]Flowchart Preview (first 20 lines):[/bold]")
        lines = mermaid.split('\n')
        for line in lines[:20]:
            console.print(f"  {line}")
        if len(lines) > 20:
            console.print(f"  ... ({len(lines) - 20} more lines)")
        console.print()
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if args.debug:
            import traceback
            console.print("\n[bold red]Traceback:[/bold red]")
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
