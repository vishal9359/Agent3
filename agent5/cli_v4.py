"""
CLI for Agent5 V4: DocAgent-Inspired Bottom-Up Semantic Aggregation

New commands for V4 pipeline:
- flowchart-v4: Generate flowcharts using the new V4 pipeline
"""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

import click

from .logging_utils import setup_logging
from .sfm_constructor import DetailLevel
from .v4_pipeline import V4Pipeline

logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli_v4(verbose: bool):
    """Agent5 V4: C++ Flowchart Generator with Bottom-Up Semantic Aggregation"""
    setup_logging(verbose=verbose)


@cli_v4.command(name='flowchart')
@click.option(
    '--project-path',
    required=True,
    type=click.Path(exists=True),
    help='Path to C++ project root'
)
@click.option(
    '--entry-function',
    required=True,
    help='Name of entry function for the scenario'
)
@click.option(
    '--entry-file',
    type=str,
    help='File path to disambiguate entry function (optional but recommended)'
)
@click.option(
    '--detail-level',
    type=click.Choice(['high', 'medium', 'deep'], case_sensitive=False),
    default='medium',
    help='Detail level for flowchart: high (business only), medium (+ validations/state), deep (+ critical sub-ops)'
)
@click.option(
    '--scenario-name',
    type=str,
    help='Custom name for the scenario (default: function name)'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output file for Mermaid flowchart (default: stdout)'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    help='Directory to save intermediate outputs (SFM, summaries, etc.)'
)
@click.option(
    '--model',
    default='llama3.2:3b',
    help='Ollama model for LLM stages (default: llama3.2:3b)'
)
@click.option(
    '--no-llm-translator',
    is_flag=True,
    help='Use deterministic rule-based Mermaid translation instead of LLM'
)
def flowchart_v4(
    project_path: str,
    entry_function: str,
    entry_file: Optional[str],
    detail_level: str,
    scenario_name: Optional[str],
    output: Optional[str],
    output_dir: Optional[str],
    model: str,
    no_llm_translator: bool
):
    """
    Generate a flowchart using V4 pipeline (DocAgent-inspired bottom-up aggregation).
    
    The V4 pipeline performs:
    1. Full AST + CFG construction (Clang)
    2. Leaf-level semantic extraction (rule-based)
    3. Bottom-up semantic aggregation (LLM-assisted)
    4. Scenario Flow Model construction
    5. Detail-level filtering
    6. Mermaid translation (LLM or rule-based)
    
    Examples:
    
    \b
    # Medium detail flowchart for processRequest in server.cpp
    agent5-v4 flowchart --project-path ./myproject \\
        --entry-function processRequest \\
        --entry-file src/server.cpp \\
        --detail-level medium \\
        --output flowchart.mmd
    
    \b
    # High-level overview with intermediate outputs saved
    agent5-v4 flowchart --project-path ./myproject \\
        --entry-function main \\
        --detail-level high \\
        --output-dir ./analysis \\
        --output flowchart_high.mmd
    
    \b
    # Deep detail without LLM translator (faster)
    agent5-v4 flowchart --project-path ./myproject \\
        --entry-function handleTransaction \\
        --entry-file transaction.cpp \\
        --detail-level deep \\
        --no-llm-translator
    """
    try:
        # Convert detail level string to enum
        detail_enum = DetailLevel[detail_level.upper()]
        
        # Initialize V4 pipeline
        click.echo(f"Initializing V4 pipeline for project: {project_path}")
        pipeline = V4Pipeline(
            project_path=project_path,
            model_name=model,
            use_llm_translator=not no_llm_translator
        )
        
        # Generate flowchart
        click.echo(f"\nGenerating {detail_level} detail flowchart for '{entry_function}'...")
        
        mermaid_code = pipeline.generate_flowchart(
            entry_function=entry_function,
            entry_file=entry_file,
            detail_level=detail_enum,
            scenario_name=scenario_name,
            output_dir=output_dir
        )
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(mermaid_code)
            click.echo(f"\n✓ Flowchart saved to: {output}")
        else:
            click.echo("\n" + "=" * 70)
            click.echo("MERMAID FLOWCHART:")
            click.echo("=" * 70)
            click.echo(mermaid_code)
            click.echo("=" * 70)
        
        # Show stats
        stats = pipeline.get_pipeline_stats()
        click.echo("\nPipeline Statistics:")
        click.echo(f"  Files analyzed:      {stats['files_analyzed']}")
        click.echo(f"  Functions found:     {stats['functions_found']}")
        click.echo(f"  Call relations:      {stats['call_relations']}")
        click.echo(f"  Leaf functions:      {stats['leaf_functions']}")
        click.echo(f"  Semantic summaries:  {stats['semantic_summaries']}")
        
        click.echo("\n✓ Flowchart generation complete!")
        
    except Exception as e:
        logger.exception("Flowchart generation failed")
        click.echo(f"\n✗ Error: {e}", err=True)
        sys.exit(1)


@cli_v4.command(name='analyze')
@click.option(
    '--project-path',
    required=True,
    type=click.Path(exists=True),
    help='Path to C++ project root'
)
@click.option(
    '--output-dir',
    required=True,
    type=click.Path(),
    help='Directory to save analysis outputs'
)
@click.option(
    '--model',
    default='llama3.2:3b',
    help='Ollama model for semantic aggregation'
)
def analyze_project(project_path: str, output_dir: str, model: str):
    """
    Analyze entire C++ project and export semantic summaries.
    
    This command runs stages 1-3 of the V4 pipeline:
    - Full AST + CFG construction
    - Leaf-level semantic extraction
    - Bottom-up semantic aggregation for ALL functions
    
    Useful for understanding a codebase before generating flowcharts.
    """
    try:
        from .clang_analyzer import ClangAnalyzer
        from .semantic_extractor import LeafSemanticExtractor
        from .semantic_aggregator import SemanticAggregator
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Analyzing project: {project_path}")
        
        # Stage 1: AST Construction
        click.echo("\n[1/3] Constructing AST + CFG...")
        analyzer = ClangAnalyzer(project_path)
        analyzer.analyze_project()
        click.echo(f"  ✓ {len(analyzer.function_cfgs)} functions analyzed")
        
        # Stage 2: Semantic Extraction
        click.echo("\n[2/3] Extracting semantic actions...")
        extractor = LeafSemanticExtractor()
        total_actions = 0
        for cfg in analyzer.function_cfgs.values():
            actions = extractor.extract_from_cfg(cfg)
            total_actions += len(actions)
        click.echo(f"  ✓ {total_actions} semantic actions extracted")
        
        # Stage 3: Semantic Aggregation (for all functions)
        click.echo("\n[3/3] Aggregating semantic summaries...")
        aggregator = SemanticAggregator(model_name=model)
        
        # Process each function independently
        for func_name, cfg in analyzer.function_cfgs.items():
            actions = extractor.get_actions_for_function(func_name)
            # For now, just summarize leaf functions
            if cfg.is_leaf:
                try:
                    summary = aggregator._summarize_leaf_function(cfg, actions)
                    aggregator.summaries[func_name] = summary
                except Exception as e:
                    logger.warning(f"Failed to summarize {func_name}: {e}")
        
        click.echo(f"  ✓ {len(aggregator.summaries)} summaries generated")
        
        # Export results
        summaries_file = output_path / "semantic_summaries.json"
        aggregator.export_summaries(str(summaries_file))
        click.echo(f"\n✓ Analysis complete! Results saved to: {output_dir}")
        click.echo(f"  - Semantic summaries: {summaries_file}")
        
    except Exception as e:
        logger.exception("Analysis failed")
        click.echo(f"\n✗ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli_v4()


# =============================================================================
# Alternative argparse-based CLI (used as entry point for agent5-v4 command)
# =============================================================================

import argparse

from agent5.logging_utils import console, get_logger

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
        default="llama2:7b",
        help="Ollama LLM model to use (default: llama2:7b)",
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
    
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for V4 CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        from agent5.pipeline import generate_flowchart
        from agent5.config import set_ollama_base_url
        
        console.print("\n[bold cyan]═══ Agent5 V4: DocAgent-Inspired Pipeline ═══[/bold cyan]\n")
        
        # Set Ollama base URL if provided
        if args.ollama_base_url:
            set_ollama_base_url(args.ollama_base_url)
        
        # Display configuration
        console.print("[bold]Configuration:[/bold]")
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
        
        # Build compile args
        compile_args = ['-std=c++17', f'-I{args.project_path}']
        if args.include_paths:
            for path in args.include_paths.split(','):
                compile_args.append(f'-I{path.strip()}')
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
        
        mermaid = generate_flowchart(
            project_path=str(args.project_path),
            entry_function=args.entry_function,
            entry_file=str(args.entry_file) if args.entry_file else None,
            detail_level=args.detail_level,
            output_path=str(args.out),
            llm_model=args.llm_model,
            save_intermediate=args.debug,
            use_llm=not args.no_llm,
        )
        
        console.print()
        console.print("[bold green]✓ Flowchart generated successfully![/bold green]")
        console.print(f"[green]  Output:[/green] {args.out}")
        
        if args.debug:
            output_dir = args.project_path / "output"
            console.print(f"[green]  Debug artifacts:[/green] {output_dir}/")
            console.print(f"    • call_graph.json")
            console.print(f"    • cfgs.json")
            console.print(f"    • semantics.json")
            console.print(f"    • aggregated.json")
            console.print(f"    • sfm.json")
            console.print(f"    • sfm_{args.detail_level}.json")
        
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
