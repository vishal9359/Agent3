"""
V4 Pipeline Integration

Orchestrates all 6 stages of the bottom-up semantic aggregation pipeline:
1. Full AST Construction (Clang)
2. Leaf-Level Semantic Extraction
3. Bottom-Up Semantic Aggregation (LLM-assisted)
4. Scenario Flow Model Construction
5. Detail-Level Filtering
6. Mermaid Translation

This is the main entry point for V4 flowchart generation.
"""

import os
from pathlib import Path
from typing import Optional

from agent5.clang_ast_extractor import ClangASTExtractor, ProjectAST
from agent5.leaf_semantic_extractor import LeafSemanticExtractor
from agent5.semantic_aggregator import SemanticAggregator, AggregatedSemantics
from agent5.sfm_constructor import SFMConstructor, ScenarioFlowModel, DetailLevel
from agent5.detail_filter import DetailLevelFilter
from agent5.mermaid_translator_v4 import MermaidTranslatorV4
from agent5.logging_utils import get_logger

logger = get_logger(__name__)


class V4Pipeline:
    """
    Complete V4 pipeline orchestrator.
    
    Implements the DocAgent-inspired bottom-up approach for C++ flowchart generation.
    """
    
    def __init__(
        self,
        project_path: str,
        model_name: str = "llama3.2:3b",
        compile_flags: Optional[list] = None,
        cache_dir: Optional[str] = None
    ):
        self.project_path = Path(project_path)
        self.model_name = model_name
        self.compile_flags = compile_flags or ["-std=c++17"]
        self.cache_dir = Path(cache_dir) if cache_dir else self.project_path / ".agent5_cache"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        # Pipeline state
        self.project_ast: Optional[ProjectAST] = None
        self.leaf_semantics: Optional[dict] = None
        self.aggregated_semantics: Optional[AggregatedSemantics] = None
    
    def run_full_pipeline(
        self,
        entry_function: str,
        entry_file: Optional[str] = None,
        detail_level: str = "medium",
        scenario_name: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Run the complete V4 pipeline and return Mermaid flowchart.
        
        Args:
            entry_function: Name of entry point function
            entry_file: Optional file path to disambiguate entry function
            detail_level: "high", "medium", or "deep"
            scenario_name: Optional name for the scenario
            use_cache: Whether to use cached intermediate results
        
        Returns:
            Mermaid flowchart code as string
        """
        logger.info("=" * 80)
        logger.info("V4 PIPELINE: Bottom-Up Semantic Aggregation")
        logger.info("=" * 80)
        
        # Stage 1: Full AST Construction
        if use_cache and (self.cache_dir / "project_ast.json").exists():
            logger.info("Stage 1: Loading cached AST")
            self.project_ast = self._load_cached_ast()
        else:
            logger.info("Stage 1: Full AST Construction with Clang")
            extractor = ClangASTExtractor(str(self.project_path), self.compile_flags)
            self.project_ast = extractor.extract_project()
            if use_cache:
                extractor.save_to_file(self.project_ast, str(self.cache_dir / "project_ast.json"))
        
        # Stage 2: Leaf-Level Semantic Extraction
        if use_cache and (self.cache_dir / "leaf_semantics.json").exists():
            logger.info("Stage 2: Loading cached leaf semantics")
            self.leaf_semantics = self._load_cached_leaf_semantics()
        else:
            logger.info("Stage 2: Leaf-Level Semantic Extraction")
            leaf_extractor = LeafSemanticExtractor(self.project_ast)
            self.leaf_semantics = leaf_extractor.extract_leaf_semantics()
            if use_cache:
                leaf_extractor.save_to_file(self.leaf_semantics, str(self.cache_dir / "leaf_semantics.json"))
        
        # Stage 3: Bottom-Up Semantic Aggregation
        if use_cache and (self.cache_dir / "aggregated_semantics.json").exists():
            logger.info("Stage 3: Loading cached aggregated semantics")
            self.aggregated_semantics = self._load_cached_aggregated_semantics()
        else:
            logger.info("Stage 3: Bottom-Up Semantic Aggregation (LLM-assisted)")
            aggregator = SemanticAggregator(
                self.project_ast,
                self.leaf_semantics,
                self.model_name
            )
            self.aggregated_semantics = aggregator.aggregate()
            if use_cache:
                aggregator.save_to_file(self.aggregated_semantics, str(self.cache_dir / "aggregated_semantics.json"))
        
        # Resolve entry function
        resolved_entry = self._resolve_entry_function(entry_function, entry_file)
        
        # Stage 4: Scenario Flow Model Construction
        logger.info("Stage 4: Scenario Flow Model Construction")
        sfm_constructor = SFMConstructor(self.project_ast, self.aggregated_semantics)
        sfm = sfm_constructor.construct_scenario(resolved_entry, scenario_name)
        
        # Save SFM for debugging
        if use_cache:
            sfm_constructor.save_to_file(sfm, str(self.cache_dir / "scenario_flow_model.json"))
        
        # Stage 5: Detail-Level Filtering
        logger.info("Stage 5: Detail-Level Filtering")
        detail_enum = self._parse_detail_level(detail_level)
        filter_engine = DetailLevelFilter()
        filtered_sfm = filter_engine.filter(sfm, detail_enum)
        
        # Stage 6: Mermaid Translation
        logger.info("Stage 6: Mermaid Translation")
        translator = MermaidTranslatorV4(self.model_name)
        mermaid_code = translator.translate(filtered_sfm)
        
        logger.info("=" * 80)
        logger.info("V4 PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return mermaid_code
    
    def _resolve_entry_function(self, function_name: str, file_path: Optional[str]) -> str:
        """
        Resolve entry function qualified name.
        
        Rules:
        - If file_path provided: find function in that file
        - If only function_name: search all functions
        - If ambiguous: raise error
        - If not found: raise error
        """
        candidates = []
        
        for qualified_name, func_info in self.project_ast.functions.items():
            # Match by function name
            if func_info.name == function_name or qualified_name == function_name:
                # If file_path specified, must match
                if file_path:
                    if Path(func_info.file_path).name == Path(file_path).name:
                        candidates.append(qualified_name)
                else:
                    candidates.append(qualified_name)
        
        if len(candidates) == 0:
            raise ValueError(f"Entry function '{function_name}' not found in project")
        elif len(candidates) == 1:
            logger.info(f"Resolved entry function: {candidates[0]}")
            return candidates[0]
        else:
            # Ambiguous
            raise ValueError(
                f"Ambiguous entry function '{function_name}'. Found {len(candidates)} matches:\n" +
                "\n".join(f"  - {c}" for c in candidates) +
                "\n\nPlease specify --entry-file to disambiguate."
            )
    
    def _parse_detail_level(self, level_str: str) -> DetailLevel:
        """Parse detail level string to enum"""
        level_map = {
            "high": DetailLevel.HIGH,
            "medium": DetailLevel.MEDIUM,
            "deep": DetailLevel.DEEP
        }
        return level_map.get(level_str.lower(), DetailLevel.MEDIUM)
    
    def _load_cached_ast(self) -> ProjectAST:
        """Load cached AST (placeholder - would need proper deserialization)"""
        # For now, re-extract
        logger.warning("AST caching not fully implemented, re-extracting")
        extractor = ClangASTExtractor(str(self.project_path), self.compile_flags)
        return extractor.extract_project()
    
    def _load_cached_leaf_semantics(self) -> dict:
        """Load cached leaf semantics (placeholder)"""
        logger.warning("Leaf semantics caching not fully implemented, re-extracting")
        leaf_extractor = LeafSemanticExtractor(self.project_ast)
        return leaf_extractor.extract_leaf_semantics()
    
    def _load_cached_aggregated_semantics(self) -> AggregatedSemantics:
        """Load cached aggregated semantics (placeholder)"""
        logger.warning("Aggregated semantics caching not fully implemented, re-aggregating")
        aggregator = SemanticAggregator(self.project_ast, self.leaf_semantics, self.model_name)
        return aggregator.aggregate()
    
    def clear_cache(self):
        """Clear all cached intermediate results"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared cache directory: {self.cache_dir}")



