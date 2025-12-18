"""
V4 Command-Line Interface

Enhanced CLI with support for:
- Bottom-up semantic aggregation pipeline
- Detail levels (high|medium|deep)
- Entry-point disambiguation (--entry-function and --entry-file)
- Project-wide analysis scope
"""

import argparse
import sys
from pathlib import Path

from agent5.pipeline_v4 import V4Pipeline
from agent5.logging_utils import get_logger

logger = get_logger(__name__)


def create_v4_parser() -> argparse.ArgumentParser:
    """Create argument parser for V4 CLI"""
    parser = argparse.ArgumentParser(
        description="Agent5 V4: C++ Scenario-Based Flowchart Generator with Bottom-Up Semantic Aggregation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate high-level flowchart for main function
  python -m agent5.cli_v4 flowchart /path/to/project --entry-function main --detail-level high
  
  # Generate detailed flowchart with file disambiguation
  python -m agent5.cli_v4 flowchart /path/to/project --entry-function processRequest --entry-file server.cpp --detail-level deep
  
  # Generate medium-detail flowchart (default)
  python -m agent5.cli_v4 flowchart /path/to/project --entry-function authenticate
  
  # Clear cache and regenerate
  python -m agent5.cli_v4 flowchart /path/to/project --entry-function main --no-cache

Detail Levels:
  high   - Business-level steps only, minimal decisions, suitable for architecture overview
  medium - Include all validations, decisions, and state-changing operations (default)
  deep   - Expand critical sub-operations that affect control flow or persistent state

Entry Point Resolution:
  --entry-function FUNC       - Required: Name of the entry point function
  --entry-file FILE           - Optional: File path to disambiguate if multiple functions have the same name
  
  If --entry-file is provided, it is used ONLY to locate the correct entry point.
  The analysis scope is ALWAYS the entire project path.
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Flowchart command
    flowchart_parser = subparsers.add_parser(
        "flowchart",
        help="Generate scenario-based flowchart"
    )
    flowchart_parser.add_argument(
        "project_path",
        type=str,
        help="Path to C++ project directory"
    )
    flowchart_parser.add_argument(
        "--entry-function",
        type=str,
        required=True,
        help="Entry point function name (required)"
    )
    flowchart_parser.add_argument(
        "--entry-file",
        type=str,
        help="File containing entry function (optional, for disambiguation)"
    )
    flowchart_parser.add_argument(
        "--detail-level",
        type=str,
        choices=["high", "medium", "deep"],
        default="medium",
        help="Detail level for flowchart (default: medium)"
    )
    flowchart_parser.add_argument(
        "--output",
        type=str,
        help="Output file path for Mermaid diagram (default: stdout)"
    )
    flowchart_parser.add_argument(
        "--scenario-name",
        type=str,
        help="Optional scenario name"
    )
    flowchart_parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="LLM model name (default: llama3.2:3b)"
    )
    flowchart_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of intermediate results"
    )
    flowchart_parser.add_argument(
        "--compile-flags",
        type=str,
        nargs="+",
        help="Additional C++ compile flags (e.g., -I/path/to/includes)"
    )
    
    # Clear cache command
    clear_parser = subparsers.add_parser(
        "clear-cache",
        help="Clear cached intermediate results"
    )
    clear_parser.add_argument(
        "project_path",
        type=str,
        help="Path to C++ project directory"
    )
    
    return parser


def cmd_flowchart(args):
    """Execute flowchart generation command"""
    logger.info(f"Generating flowchart for project: {args.project_path}")
    logger.info(f"Entry function: {args.entry_function}")
    if args.entry_file:
        logger.info(f"Entry file: {args.entry_file}")
    logger.info(f"Detail level: {args.detail_level}")
    
    # Validate project path
    project_path = Path(args.project_path)
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        return 1
    
    # Setup compile flags
    compile_flags = ["-std=c++17"]
    if args.compile_flags:
        compile_flags.extend(args.compile_flags)
    
    try:
        # Initialize V4 pipeline
        pipeline = V4Pipeline(
            project_path=str(project_path),
            model_name=args.model,
            compile_flags=compile_flags
        )
        
        # Run pipeline
        mermaid_code = pipeline.run_full_pipeline(
            entry_function=args.entry_function,
            entry_file=args.entry_file,
            detail_level=args.detail_level,
            scenario_name=args.scenario_name,
            use_cache=not args.no_cache
        )
        
        # Output result
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(mermaid_code)
            logger.info(f"Flowchart saved to: {output_path}")
        else:
            print("\n" + "=" * 80)
            print("GENERATED MERMAID FLOWCHART")
            print("=" * 80)
            print(mermaid_code)
            print("=" * 80)
        
        return 0
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error generating flowchart: {e}", exc_info=True)
        return 1


def cmd_clear_cache(args):
    """Execute cache clearing command"""
    project_path = Path(args.project_path)
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        return 1
    
    try:
        pipeline = V4Pipeline(project_path=str(project_path))
        pipeline.clear_cache()
        logger.info("Cache cleared successfully")
        return 0
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = create_v4_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "flowchart":
        return cmd_flowchart(args)
    elif args.command == "clear-cache":
        return cmd_clear_cache(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

