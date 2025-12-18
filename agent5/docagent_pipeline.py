"""
DocAgent-Inspired Bottom-Up Pipeline for C++ Scenario Flowchart Generation.

This module orchestrates the complete 6-stage pipeline:
1. Full AST Construction (Clang)
2. Leaf-Level Semantic Extraction
3. Bottom-Up Backtracking & Semantic Aggregation
4. Scenario Flow Model Construction
5. Detail-Level Filtering
6. Mermaid Translation

Entry Point:
- User specifies entry function and optionally entry file for disambiguation
- Analysis scope is determined by project path, NOT entry file

Critical Constraints:
- NO function-call diagrams
- NO recursive visual expansion
- NO LLM creativity in logic
- Bottom-up understanding allowed
- Backtracking aggregation allowed
- Scenario Flow Model is authoritative
"""
from __future__ import annotations

import logging
from pathlib import Path

from .bottom_up_aggregator import aggregate_semantics
from .clang_ast_extractor import extract_ast_for_project
from .leaf_semantic_extractor import extract_leaf_semantics
from .mermaid_translator import translate_to_mermaid
from .sfm_builder import DetailLevel, build_scenario_flow_model

logger = logging.getLogger(__name__)


class DocAgentPipeline:
    """
    Complete pipeline for DocAgent-inspired C++ flowchart generation.
    """
    
    def __init__(
        self,
        project_path: Path,
        include_paths: list[Path] | None = None,
        llm_model: str = "qwen2.5-coder:7b",
        llm_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the pipeline.
        
        Args:
            project_path: Root path of the C++ project (defines analysis scope)
            include_paths: Additional include directories for Clang
            llm_model: Ollama model name for semantic aggregation
            llm_base_url: Ollama base URL
        """
        self.project_path = Path(project_path)
        self.include_paths = include_paths or []
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        
        # Pipeline state
        self.project_ast = None
        self.leaf_semantics = None
        self.function_summary = None
        self.sfm = None
        
        logger.info(f"Initialized DocAgent Pipeline for project: {project_path}")
    
    def generate_flowchart(
        self,
        entry_function: str,
        entry_file: Path | None = None,
        detail_level: str = "medium",
        use_llm_translation: bool = True
    ) -> str:
        """
        Generate a Mermaid flowchart for the given entry point.
        
        This executes all 6 stages of the pipeline.
        
        Args:
            entry_function: Name of the entry function (can be simple or qualified)
            entry_file: Optional file path to disambiguate function (NOT analysis scope)
            detail_level: Detail level (high/medium/deep)
            use_llm_translation: Whether to use LLM for Mermaid translation
        
        Returns:
            Mermaid flowchart code
        """
        logger.info("=" * 80)
        logger.info("STARTING DOCAGENT PIPELINE")
        logger.info("=" * 80)
        
        # Stage 1: Full AST Construction
        logger.info("\n[STAGE 1] Full AST Construction using Clang")
        logger.info("-" * 80)
        self.project_ast = self._stage1_extract_ast(entry_file)
        
        # Resolve entry function
        resolved_entry = self._resolve_entry_function(entry_function, entry_file)
        logger.info(f"Resolved entry function: {resolved_entry}")
        
        # Stage 2: Leaf-Level Semantic Extraction
        logger.info("\n[STAGE 2] Leaf-Level Semantic Extraction")
        logger.info("-" * 80)
        self.leaf_semantics = self._stage2_extract_leaf_semantics()
        
        # Stage 3: Bottom-Up Aggregation
        logger.info("\n[STAGE 3] Bottom-Up Backtracking & Semantic Aggregation")
        logger.info("-" * 80)
        self.function_summary = self._stage3_aggregate_semantics(resolved_entry)
        
        # Stage 4: Scenario Flow Model Construction
        logger.info("\n[STAGE 4] Scenario Flow Model Construction")
        logger.info("-" * 80)
        self.sfm = self._stage4_build_sfm()
        
        # Stage 5: Detail-Level Filtering
        logger.info("\n[STAGE 5] Detail-Level Filtering")
        logger.info("-" * 80)
        filtered_sfm = self._stage5_filter_by_detail_level(detail_level)
        
        # Stage 6: Mermaid Translation
        logger.info("\n[STAGE 6] Mermaid Translation")
        logger.info("-" * 80)
        mermaid_code = self._stage6_translate_to_mermaid(filtered_sfm, use_llm_translation)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return mermaid_code
    
    def _stage1_extract_ast(self, entry_file: Path | None = None) -> any:
        """
        Stage 1: Extract full AST using Clang.
        
        Args:
            entry_file: If provided, parse at least this file (but still parse entire project)
        
        Returns:
            ProjectAST
        """
        logger.info(f"Parsing C++ project: {self.project_path}")
        
        # Determine which files to parse
        cpp_files = None
        if entry_file:
            logger.info(f"Entry file specified: {entry_file}")
            logger.info("Note: Entry file is for disambiguation only. Analyzing entire project.")
        
        # Parse entire project
        project_ast = extract_ast_for_project(
            project_path=self.project_path,
            include_paths=self.include_paths,
            cpp_files=cpp_files
        )
        
        logger.info(f"✓ Extracted AST for {len(project_ast.translation_units)} translation units")
        logger.info(f"✓ Found {len(project_ast.functions)} functions")
        logger.info(f"✓ Built {len(project_ast.cfgs)} control flow graphs")
        logger.info(f"✓ Identified {len(project_ast.call_graph)} call relationships")
        
        return project_ast
    
    def _resolve_entry_function(self, entry_function: str, entry_file: Path | None = None) -> str:
        """
        Resolve entry function name to fully qualified name.
        
        Args:
            entry_function: Function name (simple or qualified)
            entry_file: Optional file for disambiguation
        
        Returns:
            Fully qualified function name
        """
        logger.info(f"Resolving entry function: {entry_function}")
        
        # Find all matching functions
        matches = []
        for func_name in self.project_ast.functions.keys():
            # Check for exact match
            if func_name == entry_function:
                matches.append(func_name)
            # Check if simple name matches
            elif func_name.endswith("::" + entry_function):
                matches.append(func_name)
            # Check if it matches anywhere
            elif entry_function in func_name:
                matches.append(func_name)
        
        if not matches:
            available = "\n".join(f"  - {name}" for name in list(self.project_ast.functions.keys())[:20])
            raise ValueError(
                f"Entry function '{entry_function}' not found in project.\n"
                f"Available functions (first 20):\n{available}"
            )
        
        # If entry_file is specified, filter matches to that file
        if entry_file:
            file_matches = []
            for func_name in matches:
                func_cursor = self.project_ast.functions[func_name]
                if func_cursor.location.file and Path(func_cursor.location.file.name) == entry_file:
                    file_matches.append(func_name)
            
            if file_matches:
                matches = file_matches
            else:
                logger.warning(f"No functions matching '{entry_function}' found in {entry_file}")
        
        # If still ambiguous, error out
        if len(matches) > 1:
            matches_str = "\n".join(f"  - {name}" for name in matches)
            raise ValueError(
                f"Entry function '{entry_function}' is ambiguous. Found {len(matches)} matches:\n{matches_str}\n"
                f"Please specify --entry-file to disambiguate."
            )
        
        return matches[0]
    
    def _stage2_extract_leaf_semantics(self) -> dict:
        """
        Stage 2: Extract leaf-level semantic actions.
        
        Returns:
            Dictionary mapping function_name -> (block_id -> BlockSemantics)
        """
        logger.info("Extracting semantic actions from AST/CFG...")
        
        leaf_semantics = extract_leaf_semantics(self.project_ast)
        
        logger.info(f"✓ Extracted semantics for {len(leaf_semantics)} functions")
        
        # Log statistics
        total_actions = sum(
            len(block_sem.actions)
            for func_sems in leaf_semantics.values()
            for block_sem in func_sems.values()
        )
        logger.info(f"✓ Identified {total_actions} atomic semantic actions")
        
        return leaf_semantics
    
    def _stage3_aggregate_semantics(self, entry_function: str) -> any:
        """
        Stage 3: Perform bottom-up semantic aggregation.
        
        Args:
            entry_function: Resolved entry function name
        
        Returns:
            FunctionSummary
        """
        logger.info(f"Performing bottom-up aggregation from: {entry_function}")
        
        function_summary = aggregate_semantics(
            project_ast=self.project_ast,
            leaf_semantics=self.leaf_semantics,
            entry_function=entry_function,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url
        )
        
        logger.info(f"✓ Generated semantic summary for {entry_function}")
        logger.info(f"  Purpose: {function_summary.purpose}")
        logger.info(f"  Preconditions: {len(function_summary.preconditions)}")
        logger.info(f"  Control Flow: {len(function_summary.control_flow)}")
        logger.info(f"  State Changes: {len(function_summary.state_changes)}")
        logger.info(f"  Dependencies: {len(function_summary.dependencies)}")
        
        return function_summary
    
    def _stage4_build_sfm(self) -> any:
        """
        Stage 4: Build Scenario Flow Model from function summary.
        
        Returns:
            ScenarioFlowModel
        """
        logger.info("Building Scenario Flow Model...")
        
        sfm = build_scenario_flow_model(self.function_summary)
        
        # Validate
        is_valid, errors = sfm.validate()
        if not is_valid:
            logger.error(f"SFM validation failed: {errors}")
            raise ValueError(f"Generated invalid SFM: {errors}")
        
        logger.info(f"✓ Built SFM with {len(sfm.nodes)} nodes and {len(sfm.edges)} edges")
        logger.info(f"  Start node: {sfm.start_node}")
        logger.info(f"  End nodes: {len(sfm.end_nodes)}")
        
        return sfm
    
    def _stage5_filter_by_detail_level(self, detail_level: str) -> any:
        """
        Stage 5: Filter SFM by detail level.
        
        Args:
            detail_level: "high", "medium", or "deep"
        
        Returns:
            Filtered ScenarioFlowModel
        """
        detail_enum = DetailLevel(detail_level.lower())
        logger.info(f"Filtering SFM for detail level: {detail_enum.value}")
        
        filtered_sfm = self.sfm.filter_by_detail_level(detail_enum)
        
        logger.info(f"✓ Filtered SFM: {len(filtered_sfm.nodes)} nodes, {len(filtered_sfm.edges)} edges")
        
        # Validate filtered SFM
        is_valid, errors = filtered_sfm.validate()
        if not is_valid:
            logger.warning(f"Filtered SFM has issues: {errors}")
            # Don't fail here, as filtering might disconnect some paths
        
        return filtered_sfm
    
    def _stage6_translate_to_mermaid(self, sfm: any, use_llm: bool = True) -> str:
        """
        Stage 6: Translate SFM to Mermaid.
        
        Args:
            sfm: Scenario Flow Model
            use_llm: Whether to use LLM for translation
        
        Returns:
            Mermaid flowchart code
        """
        logger.info(f"Translating SFM to Mermaid (use_llm={use_llm})...")
        
        mermaid_code = translate_to_mermaid(
            sfm=sfm,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            use_llm=use_llm
        )
        
        logger.info(f"✓ Generated Mermaid flowchart ({len(mermaid_code)} characters)")
        
        return mermaid_code


def generate_flowchart(
    project_path: Path,
    entry_function: str,
    entry_file: Path | None = None,
    detail_level: str = "medium",
    include_paths: list[Path] | None = None,
    llm_model: str = "qwen2.5-coder:7b",
    llm_base_url: str = "http://localhost:11434",
    use_llm_translation: bool = True
) -> str:
    """
    Convenience function to generate a flowchart using the DocAgent pipeline.
    
    Args:
        project_path: Root path of the C++ project
        entry_function: Name of the entry function
        entry_file: Optional file to disambiguate function
        detail_level: "high", "medium", or "deep"
        include_paths: Additional include directories
        llm_model: Ollama model name
        llm_base_url: Ollama base URL
        use_llm_translation: Whether to use LLM for Mermaid translation
    
    Returns:
        Mermaid flowchart code
    """
    pipeline = DocAgentPipeline(
        project_path=project_path,
        include_paths=include_paths,
        llm_model=llm_model,
        llm_base_url=llm_base_url
    )
    
    return pipeline.generate_flowchart(
        entry_function=entry_function,
        entry_file=entry_file,
        detail_level=detail_level,
        use_llm_translation=use_llm_translation
    )

