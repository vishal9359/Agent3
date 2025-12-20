"""
Pipeline integration for V4 DocAgent-inspired flowchart generation.

This module provides the bridge between the CLI and the docagent_pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .docagent_pipeline import DocAgentPipeline
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)


def generate_flowchart_from_project(
    project_path: Path,
    entry_function: str,
    entry_file: Path | None = None,
    detail_level: str = "medium",
    output_path: Path | None = None,
    chat_model: str | None = None,
    use_llm: bool = True,
    include_paths: list[Path] | None = None,
    debug: bool = False,
) -> str:
    """
    Generate a flowchart from a C++ project using the V4 DocAgent pipeline.
    
    Args:
        project_path: Root path of the C++ project
        entry_function: Name of the entry function
        entry_file: Optional file to disambiguate function
        detail_level: "high", "medium", or "deep"
        output_path: Optional path to save the Mermaid file
        chat_model: Optional chat model name (defaults to qwen2.5-coder:7b)
        use_llm: Whether to use LLM for Mermaid translation
        include_paths: Additional include directories for Clang
        debug: Whether to save debug artifacts
    
    Returns:
        Mermaid flowchart code
    """
    setup_logging()
    
    # Default model
    llm_model = chat_model or "qwen2.5-coder:7b"
    
    logger.info("=" * 80)
    logger.info("V4 DOCAGENT PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Project: {project_path}")
    logger.info(f"Entry function: {entry_function}")
    if entry_file:
        logger.info(f"Entry file: {entry_file}")
    logger.info(f"Detail level: {detail_level}")
    logger.info(f"LLM model: {llm_model}")
    logger.info(f"Use LLM translation: {use_llm}")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = DocAgentPipeline(
        project_path=project_path,
        include_paths=include_paths,
        llm_model=llm_model,
        llm_base_url="http://localhost:11434"
    )
    
    # Generate flowchart
    mermaid_code = pipeline.generate_flowchart(
        entry_function=entry_function,
        entry_file=entry_file,
        detail_level=detail_level,
        use_llm_translation=use_llm
    )
    
    # Save debug artifacts if requested
    if debug:
        debug_dir = project_path / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        # Save AST context
        if pipeline.project_ast:
            ast_context = {
                "functions": list(pipeline.project_ast.functions.keys()),
                "translation_units": pipeline.project_ast.translation_unit_files,
                "call_graph_size": len(pipeline.project_ast.call_graph),
            }
            with open(debug_dir / "ast_context.json", "w") as f:
                json.dump(ast_context, f, indent=2)
            logger.info(f"Saved AST context to {debug_dir / 'ast_context.json'}")
        
        # Save function summary
        if pipeline.function_summary:
            with open(debug_dir / "function_summary.json", "w") as f:
                json.dump(pipeline.function_summary.to_dict(), f, indent=2)
            logger.info(f"Saved function summary to {debug_dir / 'function_summary.json'}")
        
        # Save SFM
        if pipeline.sfm:
            with open(debug_dir / "sfm.json", "w") as f:
                json.dump(pipeline.sfm.to_dict(), f, indent=2)
            logger.info(f"Saved SFM to {debug_dir / 'sfm.json'}")
    
    # Write output file if path is provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        logger.info(f"Wrote Mermaid flowchart to {output_path}")
    
    return mermaid_code
