"""
Version 4 Pipeline - DocAgent-Inspired Bottom-Up Semantic Aggregation

This module orchestrates the complete V4 pipeline:
1. Stage 1: Full AST Construction (Clang)
2. Stage 2: Leaf-Level Semantic Extraction
3. Stage 3: Bottom-Up Aggregation (LLM-assisted)
4. Stage 4: Scenario Flow Model Construction
5. Stage 5: Detail-Level Filtering
6. Stage 6: Mermaid Translation

The pipeline produces documentation-quality scenario flowcharts.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .bottom_up_aggregator import BottomUpAggregator
from .clang_ast_parser import CLANG_AVAILABLE, ClangASTParser, ProjectAST
from .leaf_semantic_extractor import LeafSemanticExtractor
from .mermaid_generator import MermaidFlowchart, MermaidGenerator
from .sfm_constructor import DetailLevel, ScenarioFlowModel, construct_sfm

logger = logging.getLogger(__name__)


class V4Pipeline:
    """
    Complete V4 pipeline for documentation-quality flowchart generation.
    
    This pipeline uses a DocAgent-inspired bottom-up understanding strategy.
    """

    def __init__(
        self,
        project_path: Path,
        chat_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ):
        """
        Initialize the V4 pipeline.
        
        Args:
            project_path: Root path of the C++ project
            chat_model: Ollama chat model for LLM-assisted aggregation
            ollama_base_url: Ollama server URL
        """
        self.project_path = project_path
        self.chat_model = chat_model
        self.ollama_base_url = ollama_base_url
        
        # Pipeline components
        self.ast_parser: Optional[ClangASTParser] = None
        self.project_ast: Optional[ProjectAST] = None
        self.leaf_extractor: Optional[LeafSemanticExtractor] = None
        self.aggregator: Optional[BottomUpAggregator] = None
        self.sfm: Optional[ScenarioFlowModel] = None
    
    def run(
        self,
        entry_file: Path,
        entry_function: Optional[str] = None,
        detail_level: str = "medium",
        output_path: Optional[Path] = None,
    ) -> MermaidFlowchart:
        """
        Run the complete V4 pipeline.
        
        Args:
            entry_file: Path to file containing entry function (for locating only)
            entry_function: Entry function name (auto-detect if None)
            detail_level: Detail level (high|medium|deep)
            output_path: Optional path to save .mmd file
        
        Returns:
            MermaidFlowchart object
        """
        logger.info("=" * 60)
        logger.info("V4 Pipeline: DocAgent-Inspired Bottom-Up Semantic Aggregation")
        logger.info("=" * 60)
        
        # Stage 1: Full AST Construction
        logger.info("\n[Stage 1] Full AST Construction (NO LLM)")
        self._stage1_ast_construction()
        
        # Stage 2: Leaf-Level Semantic Extraction
        logger.info("\n[Stage 2] Leaf-Level Semantic Extraction")
        leaf_semantics = self._stage2_leaf_extraction()
        
        # Resolve entry function
        resolved_entry = self._resolve_entry_function(entry_file, entry_function)
        
        # Stage 3: Bottom-Up Aggregation
        logger.info(f"\n[Stage 3] Bottom-Up Aggregation from entry: {resolved_entry}")
        aggregated_semantics = self._stage3_bottom_up_aggregation(resolved_entry, leaf_semantics)
        
        # Stage 4: Scenario Flow Model Construction
        logger.info("\n[Stage 4] Scenario Flow Model Construction")
        self.sfm = self._stage4_sfm_construction(resolved_entry, aggregated_semantics, leaf_semantics)
        
        # Stage 5: Detail-Level Filtering
        logger.info(f"\n[Stage 5] Detail-Level Filtering (level={detail_level})")
        filtered_sfm = self._stage5_detail_filtering(self.sfm, DetailLevel[detail_level.upper()])
        
        # Stage 6: Mermaid Translation
        logger.info("\n[Stage 6] Mermaid Translation (LLM STRICT TRANSLATOR)")
        flowchart = self._stage6_mermaid_translation(filtered_sfm, use_llm=bool(self.chat_model))
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(flowchart.to_mermaid(), encoding="utf-8")
            logger.info(f"\n✓ Flowchart saved to: {output_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("V4 Pipeline Complete")
        logger.info("=" * 60)
        
        return flowchart
    
    def _stage1_ast_construction(self):
        """Stage 1: Parse entire C++ project using Clang AST."""
        if not CLANG_AVAILABLE:
            logger.warning(
                "Clang AST not available - falling back to tree-sitter (limited functionality)"
            )
            # TODO: Implement tree-sitter fallback
            raise RuntimeError(
                "Clang AST is required for V4 pipeline. Please install libclang."
            )
        
        logger.info(f"Parsing C++ project at: {self.project_path}")
        self.ast_parser = ClangASTParser(self.project_path)
        self.project_ast = self.ast_parser.parse_project()
        
        logger.info(f"  ✓ Parsed {len(self.project_ast.function_cfgs)} functions")
        logger.info(f"  ✓ Found {len(self.project_ast.leaf_functions)} leaf functions")
        logger.info(f"  ✓ Identified {len(self.project_ast.entry_points)} potential entry points")
    
    def _stage2_leaf_extraction(self):
        """Stage 2: Extract atomic semantic actions at leaf level."""
        logger.info("Extracting leaf-level semantics...")
        
        self.leaf_extractor = LeafSemanticExtractor(self.project_ast)
        leaf_semantics = self.leaf_extractor.extract_semantics()
        
        logger.info(f"  ✓ Extracted semantics for {len(leaf_semantics)} functions")
        
        return leaf_semantics
    
    def _resolve_entry_function(
        self, entry_file: Path, entry_function: Optional[str]
    ) -> str:
        """
        Resolve the entry function name.
        
        Args:
            entry_file: File containing entry function (for disambiguation)
            entry_function: Function name (or None for auto-detect)
        
        Returns:
            Resolved fully-qualified function name
        
        Raises:
            ValueError: If function cannot be resolved
        """
        # If function name provided, search for it
        if entry_function:
            # Search in specified file first
            candidates = []
            
            for func_name, cfg in self.project_ast.function_cfgs.items():
                if entry_function in func_name:
                    # Check if it's in the specified file
                    if str(entry_file) in cfg.file_path or entry_file.name in cfg.file_path:
                        return func_name
                    candidates.append(func_name)
            
            # If not found in specified file, check all candidates
            if len(candidates) == 1:
                logger.warning(
                    f"Function '{entry_function}' not found in {entry_file}, "
                    f"using match from {self.project_ast.function_cfgs[candidates[0]].file_path}"
                )
                return candidates[0]
            elif len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous function name '{entry_function}'. Found {len(candidates)} matches:\n"
                    + "\n".join(f"  - {c}" for c in candidates)
                    + "\nPlease specify --entry-file to disambiguate."
                )
            else:
                # Try exact match
                if entry_function in self.project_ast.function_cfgs:
                    return entry_function
                
                raise ValueError(
                    f"Function '{entry_function}' not found. "
                    f"Available functions in {entry_file}:\n"
                    + "\n".join(
                        f"  - {name}"
                        for name, cfg in self.project_ast.function_cfgs.items()
                        if str(entry_file) in cfg.file_path or entry_file.name in cfg.file_path
                    )
                )
        
        # Auto-detect: use entry points from AST
        if self.project_ast.entry_points:
            # Prefer functions in the specified file
            for func_name in self.project_ast.entry_points:
                cfg = self.project_ast.function_cfgs[func_name]
                if str(entry_file) in cfg.file_path or entry_file.name in cfg.file_path:
                    logger.info(f"Auto-detected entry function: {func_name}")
                    return func_name
            
            # Use any entry point
            func_name = next(iter(self.project_ast.entry_points))
            logger.info(f"Auto-detected entry function: {func_name}")
            return func_name
        
        raise ValueError(
            "Could not auto-detect entry function. Please specify --function explicitly."
        )
    
    def _stage3_bottom_up_aggregation(self, entry_function: str, leaf_semantics):
        """Stage 3: Bottom-up semantic aggregation (LLM-assisted)."""
        logger.info(f"Starting bottom-up aggregation from: {entry_function}")
        
        self.aggregator = BottomUpAggregator(
            project_ast=self.project_ast,
            leaf_semantics=leaf_semantics,
            chat_model=self.chat_model,
            ollama_base_url=self.ollama_base_url,
        )
        
        aggregated_semantics = self.aggregator.aggregate(entry_function)
        
        logger.info(f"  ✓ Aggregated {len(aggregated_semantics)} functions")
        
        # Log entry function summary
        entry_summary = self.aggregator.get_entry_summary(entry_function)
        if entry_summary:
            logger.info(f"  Entry function summary: {entry_summary.semantic_summary}")
        
        return aggregated_semantics
    
    def _stage4_sfm_construction(
        self, entry_function: str, aggregated_semantics, leaf_semantics
    ) -> ScenarioFlowModel:
        """Stage 4: Construct Scenario Flow Model."""
        logger.info(f"Constructing Scenario Flow Model for: {entry_function}")
        
        sfm = construct_sfm(entry_function, aggregated_semantics, leaf_semantics)
        
        logger.info(f"  ✓ SFM constructed with {len(sfm.steps)} steps")
        logger.info(f"  ✓ Scenario: {sfm.scenario_name}")
        
        return sfm
    
    def _stage5_detail_filtering(
        self, sfm: ScenarioFlowModel, detail_level: DetailLevel
    ) -> ScenarioFlowModel:
        """Stage 5: Filter SFM by detail level."""
        logger.info(f"Filtering SFM for detail level: {detail_level.value}")
        
        # Create filtered SFM
        filtered_sfm = ScenarioFlowModel(
            entry_function=sfm.entry_function,
            scenario_name=sfm.scenario_name,
            start_step=sfm.start_step,
            metadata=sfm.metadata,
        )
        
        # Filter steps by detail level
        for step_id, step in sfm.steps.items():
            if detail_level in step.detail_levels:
                filtered_sfm.steps[step_id] = step
        
        # Update end steps
        filtered_sfm.end_steps = [
            end_id for end_id in sfm.end_steps if end_id in filtered_sfm.steps
        ]
        
        # Clean up references to filtered-out steps
        for step in filtered_sfm.steps.values():
            step.next_steps = [
                next_id for next_id in step.next_steps if next_id in filtered_sfm.steps
            ]
            if step.on_fail and step.on_fail not in filtered_sfm.steps:
                step.on_fail = None
            if step.on_success and step.on_success not in filtered_sfm.steps:
                step.on_success = None
        
        logger.info(
            f"  ✓ Filtered from {len(sfm.steps)} to {len(filtered_sfm.steps)} steps"
        )
        
        return filtered_sfm
    
    def _stage6_mermaid_translation(
        self, sfm: ScenarioFlowModel, use_llm: bool
    ) -> MermaidFlowchart:
        """Stage 6: Translate SFM to Mermaid (LLM as strict translator)."""
        logger.info("Translating SFM to Mermaid...")
        
        generator = MermaidGenerator(
            chat_model=self.chat_model if use_llm else None,
            ollama_base_url=self.ollama_base_url if use_llm else None,
        )
        
        flowchart = generator.generate_from_sfm(sfm)
        
        logger.info(f"  ✓ Generated Mermaid flowchart with {len(flowchart.nodes)} nodes")
        
        return flowchart
    
    def export_sfm_json(self, output_path: Path):
        """Export SFM as JSON for inspection/debugging."""
        if not self.sfm:
            raise ValueError("No SFM available - run pipeline first")
        
        from .sfm_constructor import SFMConstructor
        
        # Create a temporary constructor just to use its to_dict method
        constructor = SFMConstructor("", {}, {})
        constructor.sfm = self.sfm
        
        sfm_dict = constructor.to_dict()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(sfm_dict, indent=2), encoding="utf-8")
        
        logger.info(f"SFM exported to: {output_path}")


def generate_v4_flowchart(
    project_path: Path,
    entry_file: Path,
    entry_function: Optional[str] = None,
    detail_level: str = "medium",
    output_path: Optional[Path] = None,
    chat_model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
) -> MermaidFlowchart:
    """
    Convenience function to run the V4 pipeline.
    
    Args:
        project_path: Root path of the C++ project
        entry_file: Path to file containing entry function
        entry_function: Entry function name (auto-detect if None)
        detail_level: Detail level (high|medium|deep)
        output_path: Optional path to save .mmd file
        chat_model: Ollama chat model name
        ollama_base_url: Ollama server URL
    
    Returns:
        MermaidFlowchart object
    """
    pipeline = V4Pipeline(
        project_path=project_path,
        chat_model=chat_model,
        ollama_base_url=ollama_base_url,
    )
    
    return pipeline.run(
        entry_file=entry_file,
        entry_function=entry_function,
        detail_level=detail_level,
        output_path=output_path,
    )
