"""
V4 Pipeline: DocAgent-Inspired Bottom-Up Semantic Aggregation

This module integrates all stages of the V4 pipeline:
1. Clang AST + CFG extraction
2. Leaf-level semantic extraction
3. Bottom-up semantic aggregation (LLM-assisted)
4. Scenario Flow Model construction
5. Detail-level filtering
6. Mermaid translation

Entry point for generating documentation-quality flowcharts from C++ projects.
"""

import logging
import json
from pathlib import Path
from typing import Optional

from .clang_analyzer import ClangAnalyzer
from .semantic_extractor import LeafSemanticExtractor
from .semantic_aggregator import SemanticAggregator
from .sfm_constructor import SFMConstructor, DetailLevel
from .sfm_filter import SFMFilter
from .mermaid_translator import MermaidTranslator

logger = logging.getLogger(__name__)


class V4Pipeline:
    """
    Complete V4 pipeline for C++ flowchart generation.
    
    Uses DocAgent-inspired bottom-up semantic aggregation for deep understanding,
    while maintaining scenario-based presentation.
    """
    
    def __init__(
        self,
        project_path: str,
        model_name: str = "llama3.2:3b",
        use_llm_translator: bool = True
    ):
        """
        Initialize the V4 pipeline
        
        Args:
            project_path: Root path of C++ project
            model_name: Ollama model for LLM stages
            use_llm_translator: Whether to use LLM for Mermaid translation
        """
        self.project_path = Path(project_path).resolve()
        if not self.project_path.is_dir():
            raise ValueError(f"Invalid project root (not a directory): {self.project_path}")
        self.model_name = model_name
        
        # Initialize pipeline components
        logger.info("Initializing V4 pipeline components...")
        
        # Stage 1: Clang Analyzer
        self.analyzer = ClangAnalyzer(str(self.project_path))
        
        # Stage 2: Leaf Semantic Extractor
        self.semantic_extractor = LeafSemanticExtractor()
        
        # Stage 3: Semantic Aggregator (LLM-assisted)
        self.aggregator = SemanticAggregator(model_name=model_name)
        
        # Stage 4: SFM Constructor
        self.sfm_constructor = SFMConstructor()
        
        # Stage 5: SFM Filter
        self.sfm_filter = SFMFilter()
        
        # Stage 6: Mermaid Translator
        self.mermaid_translator = MermaidTranslator(
            model_name=model_name,
            use_llm=use_llm_translator
        )
        
        logger.info("V4 pipeline initialized successfully")
    
    def generate_flowchart(
        self,
        entry_function: str,
        entry_file: Optional[str] = None,
        detail_level: DetailLevel = DetailLevel.MEDIUM,
        scenario_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Generate a flowchart for a C++ scenario using the V4 pipeline.
        
        Args:
            entry_function: Name of the entry function
            entry_file: Optional file path to disambiguate entry function
            detail_level: Level of detail (HIGH, MEDIUM, DEEP)
            scenario_name: Optional name for the scenario
            output_dir: Optional directory to save intermediate outputs
            
        Returns:
            Mermaid flowchart code as string
        """
        logger.info("=" * 70)
        logger.info("V4 PIPELINE: Starting flowchart generation")
        logger.info(f"Entry function: {entry_function}")
        logger.info(f"Entry file: {entry_file or 'auto-detect'}")
        logger.info(f"Detail level: {detail_level.value}")
        logger.info("=" * 70)
        
        # Create output directory if specified
        output_path = None
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: Full AST Construction
        logger.info("\n[STAGE 1] Full AST Construction (NO LLM)")
        logger.info("-" * 70)
        
        file_patterns = ['*.cpp', '*.cc', '*.cxx', '*.c++']
        if entry_file:
            # If entry file specified, analyze it and related files
            file_patterns.append(entry_file)
        
        self.analyzer.analyze_project(file_patterns)
        logger.info(f"✓ Analyzed {len(self.analyzer.translation_units)} files")
        logger.info(f"✓ Built {len(self.analyzer.function_cfgs)} function CFGs")
        logger.info(f"✓ Extracted {len(self.analyzer.call_graph)} call relations")
        
        # Resolve entry function
        qualified_entry = self._resolve_entry_function(entry_function, entry_file)
        if not qualified_entry:
            raise ValueError(f"Cannot find entry function '{entry_function}' in project")
        
        logger.info(f"✓ Resolved entry function: {qualified_entry}")
        
        # Stage 2: Leaf-Level Semantic Extraction
        logger.info("\n[STAGE 2] Leaf-Level Semantic Extraction (BOTTOM LEVEL)")
        logger.info("-" * 70)
        
        total_actions = 0
        for func_name, cfg in self.analyzer.function_cfgs.items():
            actions = self.semantic_extractor.extract_from_cfg(cfg)
            total_actions += len(actions)
        
        logger.info(f"✓ Extracted {total_actions} semantic actions from all functions")
        
        # Stage 3: Bottom-Up Semantic Aggregation
        logger.info("\n[STAGE 3] Bottom-Up Backtracking & Semantic Aggregation (LLM-ASSISTED)")
        logger.info("-" * 70)
        
        summaries = self.aggregator.aggregate(
            self.analyzer,
            self.semantic_extractor,
            qualified_entry
        )
        
        logger.info(f"✓ Generated {len(summaries)} function semantic summaries")
        
        # Export summaries if output dir specified
        if output_path:
            summaries_file = output_path / "semantic_summaries.json"
            self.aggregator.export_summaries(str(summaries_file))
            logger.info(f"✓ Exported summaries to {summaries_file}")
        
        # Get entry function summary
        entry_summary = self.aggregator.get_summary(qualified_entry)
        if not entry_summary:
            raise ValueError(f"Failed to generate summary for entry function '{qualified_entry}'")
        
        # Stage 4: Scenario Flow Model Construction
        logger.info("\n[STAGE 4] Scenario Flow Model Construction (SINGLE SOURCE OF TRUTH)")
        logger.info("-" * 70)
        
        sfm = self.sfm_constructor.construct(
            entry_function=qualified_entry,
            semantic_summary=entry_summary,
            scenario_name=scenario_name or entry_function
        )
        
        logger.info(f"✓ Constructed SFM with {len(sfm.nodes)} nodes")
        logger.info(f"  - Start node: {sfm.start_node_id}")
        logger.info(f"  - End nodes: {', '.join(sfm.end_node_ids)}")
        
        # Validate SFM
        errors = sfm.validate()
        if errors:
            logger.warning(f"SFM validation warnings: {errors}")
        else:
            logger.info("✓ SFM validation passed")
        
        # Export SFM if output dir specified
        if output_path:
            sfm_file = output_path / "scenario_flow_model.json"
            self.sfm_constructor.export_sfm(sfm, str(sfm_file))
            logger.info(f"✓ Exported SFM to {sfm_file}")
        
        # Stage 5: Detail-Level Filtering
        logger.info("\n[STAGE 5] Detail-Level Filtering (RULE-BASED)")
        logger.info("-" * 70)
        
        filtered_sfm = self.sfm_filter.filter(sfm, detail_level)
        
        logger.info(f"✓ Filtered to {len(filtered_sfm.nodes)} nodes at {detail_level.value} level")
        
        # Export filtered SFM if output dir specified
        if output_path:
            filtered_sfm_file = output_path / f"sfm_filtered_{detail_level.value}.json"
            with open(filtered_sfm_file, 'w') as f:
                json.dump(filtered_sfm.to_dict(), f, indent=2)
            logger.info(f"✓ Exported filtered SFM to {filtered_sfm_file}")
        
        # Stage 6: Mermaid Translation
        logger.info("\n[STAGE 6] Mermaid Translation (LLM STRICT TRANSLATOR)")
        logger.info("-" * 70)
        
        mermaid_code = self.mermaid_translator.translate(filtered_sfm)
        
        logger.info(f"✓ Generated Mermaid flowchart ({len(mermaid_code)} characters)")
        
        # Save Mermaid code if output dir specified
        if output_path:
            mermaid_file = output_path / f"flowchart_{detail_level.value}.mmd"
            with open(mermaid_file, 'w') as f:
                f.write(mermaid_code)
            logger.info(f"✓ Saved Mermaid code to {mermaid_file}")
        
        logger.info("\n" + "=" * 70)
        logger.info("V4 PIPELINE: Flowchart generation complete!")
        logger.info("=" * 70)
        
        return mermaid_code
    
    def _resolve_entry_function(
        self,
        function_name: str,
        entry_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Resolve entry function to its qualified name.
        
        Args:
            function_name: Simple or qualified function name
            entry_file: Optional file path to disambiguate
            
        Returns:
            Qualified function name, or None if not found/ambiguous
        """
        # Find all matching functions
        matches = []
        
        for qualified_name in self.analyzer.function_cfgs.keys():
            # Check if function name matches
            if qualified_name == function_name or qualified_name.endswith(f"::{function_name}"):
                cfg = self.analyzer.function_cfgs[qualified_name]
                
                # If entry_file specified, check file match
                if entry_file:
                    if Path(cfg.file_path).name == Path(entry_file).name:
                        matches.append(qualified_name)
                else:
                    matches.append(qualified_name)
        
        if len(matches) == 0:
            logger.error(f"No function named '{function_name}' found")
            logger.info("Available functions:")
            for qname in sorted(self.analyzer.function_cfgs.keys())[:20]:
                logger.info(f"  - {qname}")
            return None
        
        if len(matches) == 1:
            return matches[0]
        
        if len(matches) > 1:
            if entry_file:
                logger.error(f"Multiple matches for '{function_name}' in '{entry_file}':")
            else:
                logger.error(f"Ambiguous function name '{function_name}'. Please specify --entry-file:")
            
            for match in matches:
                cfg = self.analyzer.function_cfgs[match]
                logger.info(f"  - {match} in {cfg.file_path}")
            
            return None
        
        return None
    
    def get_pipeline_stats(self) -> dict:
        """Get statistics about the pipeline execution"""
        return {
            "files_analyzed": len(self.analyzer.translation_units),
            "functions_found": len(self.analyzer.function_cfgs),
            "call_relations": len(self.analyzer.call_graph),
            "leaf_functions": len(self.analyzer.get_leaf_functions()),
            "semantic_summaries": len(self.aggregator.summaries)
        }
