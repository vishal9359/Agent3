"""
Version 4 Scenario Pipeline - DocAgent-inspired Bottom-Up Aggregation

This module implements the complete 6-stage pipeline:
Stage 1: Full AST Construction (NO LLM)
Stage 2: Leaf-Level Semantic Extraction (BOTTOM LEVEL)
Stage 3: Bottom-Up Backtracking & Semantic Aggregation (LLM-ASSISTED)
Stage 4: Scenario Flow Model Construction (SINGLE SOURCE OF TRUTH)
Stage 5: Detail-Level Filtering (RULE-BASED)
Stage 6: Mermaid Translation (LLM STRICT TRANSLATOR)

This provides cross-file, project-wide scenario analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent5.clang_ast_extractor import ClangASTExtractor
from agent5.bottom_up_aggregator import BottomUpAggregator
from agent5.sfm_constructor import DetailLevel, SFMConstructor, ScenarioFlowModel
from agent5.logging_utils import get_logger

logger = get_logger(__name__)


def extract_scenario_from_project(
    project_path: Path,
    function_name: str,
    file_path: Path | None = None,
    *,
    max_steps: int = 30,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    compile_flags: list[str] | None = None,
) -> tuple[ScenarioFlowModel, dict[str, Any]]:
    """
    Extract scenario flow model from a C++ project using bottom-up aggregation.
    
    This implements the complete Version 4 pipeline:
    1. Parse entire project with Clang AST
    2. Extract leaf-level semantics
    3. Aggregate bottom-up through call graph
    4. Construct Scenario Flow Model
    5. Filter by detail level
    
    Args:
        project_path: Root path of the C++ project
        function_name: Entry function name
        file_path: Optional file path to disambiguate entry function
        max_steps: Maximum steps in the scenario
        detail_level: Detail level (HIGH, MEDIUM, DEEP)
        compile_flags: Optional C++ compile flags
        
    Returns:
        Tuple of (ScenarioFlowModel, metadata)
        
    Raises:
        RuntimeError: If any stage fails
    """
    logger.info(f"🚀 Version 4 Pipeline: Bottom-Up Scenario Extraction")
    logger.info(f"   Project: {project_path}")
    logger.info(f"   Entry: {function_name}")
    logger.info(f"   Detail: {detail_level.value}")
    
    # Stage 1: Full AST Construction (NO LLM)
    logger.info(f"📋 Stage 1: Clang AST + CFG Extraction")
    extractor = ClangASTExtractor(str(project_path), compile_flags=compile_flags)
    project_ast = extractor.extract_project()
    
    # Resolve entry function
    entry_function_name = _resolve_entry_function(
        project_ast,
        function_name,
        file_path
    )
    
    if not entry_function_name:
        raise RuntimeError(
            f"Cannot find entry function '{function_name}' in project.\n"
            f"Available functions: {', '.join(list(project_ast.functions.keys())[:10])}"
        )
    
    logger.info(f"   ✓ Resolved entry: {entry_function_name}")
    
    # Stage 2 & 3: Leaf-Level Semantic Extraction + Bottom-Up Aggregation
    logger.info(f"🔬 Stage 2-3: Semantic Extraction & Aggregation")
    aggregator = BottomUpAggregator(project_ast)
    aggregated_semantics = aggregator.aggregate_from_entry_point(entry_function_name)
    
    # Stage 4: Scenario Flow Model Construction
    logger.info(f"📐 Stage 4: Scenario Flow Model Construction")
    constructor = SFMConstructor()
    full_sfm = constructor.construct_from_aggregated_semantics(aggregated_semantics)
    
    # Stage 5: Detail-Level Filtering
    logger.info(f"🎚️ Stage 5: Detail-Level Filtering ({detail_level.value})")
    filtered_sfm = full_sfm.filter_by_detail_level(detail_level)
    
    # Prepare metadata
    metadata = {
        "entry_function": entry_function_name,
        "detail_level": detail_level.value,
        "total_functions_analyzed": len(aggregated_semantics.function_summaries),
        "critical_path_length": len(aggregated_semantics.critical_path),
        "scenario_description": aggregated_semantics.scenario_description,
    }
    
    logger.info(f"✅ Pipeline Complete!")
    logger.info(f"   Analyzed {metadata['total_functions_analyzed']} functions")
    logger.info(f"   Generated {len(filtered_sfm.nodes)} scenario nodes")
    
    return filtered_sfm, metadata


def _resolve_entry_function(
    project_ast,
    function_name: str,
    file_path: Path | None = None
) -> str | None:
    """
    Resolve entry function from function name and optional file path.
    
    Rules:
    - If both file_path and function_name provided: strict resolution
    - If only function_name: must be unambiguous
    - Returns qualified function name or None
    """
    # Try exact match first
    if function_name in project_ast.functions:
        # Check file constraint if provided
        if file_path:
            func_info = project_ast.functions[function_name]
            if func_info.file_path == str(file_path):
                return function_name
        else:
            return function_name
    
    # Try partial match (contains function_name)
    matches = []
    for qualified_name, func_info in project_ast.functions.items():
        if function_name in qualified_name:
            if file_path is None or func_info.file_path == str(file_path):
                matches.append(qualified_name)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Ambiguous
        if file_path:
            raise ValueError(
                f"Multiple matches for function '{function_name}' in file {file_path}:\n"
                + "\n".join(f"  - {m}" for m in matches)
                + "\n\nPlease use the fully qualified name."
            )
        else:
            raise ValueError(
                f"Ambiguous function name '{function_name}'. Found matches:\n"
                + "\n".join(f"  - {m} ({project_ast.functions[m].file_path})" for m in matches)
                + "\n\nPlease specify --entry-file to disambiguate."
            )
    
    return None


# Export for compatibility with existing code
def extract_scenario_from_function_v4(
    source_code: str,
    function_name: str | None = None,
    *,
    max_steps: int = 30,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
) -> ScenarioFlowModel:
    """
    Single-file scenario extraction (legacy compatibility).
    
    This is a simplified version that doesn't use the full bottom-up pipeline.
    For new code, use extract_scenario_from_project instead.
    """
    # Fall back to tree-sitter based extraction
    from agent5.scenario_extractor import extract_scenario_from_function
    
    return extract_scenario_from_function(
        source_code,
        function_name=function_name,
        max_steps=max_steps,
        detail_level=detail_level
    )

