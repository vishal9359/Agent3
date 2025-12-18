"""
Main Pipeline Orchestrator

This module orchestrates the complete 6-stage pipeline for flowchart generation:

Stage 1: Full AST Construction (Clang)
Stage 2: Leaf-Level Semantic Extraction
Stage 3: Bottom-Up Semantic Aggregation
Stage 4: Scenario Flow Model Construction
Stage 5: Detail-Level Filtering
Stage 6: Mermaid Translation

Entry point: generate_flowchart()
"""

import os
from pathlib import Path
from typing import Optional, List
import json

from agent5.clang_ast_parser import ClangASTParser
from agent5.cfg_builder import build_project_cfgs
from agent5.semantic_extractor import extract_project_semantics
from agent5.aggregator import BottomUpAggregator
from agent5.sfm_builder import SFMBuilder, save_sfm_to_file
from agent5.detail_filter import apply_detail_level
from agent5.flowchart_v4 import generate_flowchart_from_sfm
from agent5.logging_utils import get_logger

logger = get_logger(__name__)


class FlowchartPipeline:
    """
    Orchestrates the complete pipeline for flowchart generation.
    """
    
    def __init__(self, project_path: str, llm_model: str = "llama2:7b", 
                 compile_args: Optional[List[str]] = None):
        """
        Initialize the pipeline.
        
        Args:
            project_path: Path to the C++ project
            llm_model: Name of the Ollama LLM model to use
            compile_args: Optional compilation arguments for Clang
        """
        self.project_path = Path(project_path)
        self.llm_model = llm_model
        self.compile_args = compile_args
        
        # Pipeline stages
        self.ast_parser = None
        self.call_graph = None
        self.cfgs = None
        self.semantics_map = None
        self.aggregator = None
        
        logger.info(f"Initialized FlowchartPipeline for project: {project_path}")
    
    def generate_flowchart(self, entry_function: Optional[str] = None, 
                          entry_file: Optional[str] = None,
                          detail_level: str = "medium",
                          output_path: Optional[str] = None,
                          save_intermediate: bool = False,
                          use_llm: bool = True) -> str:
        """
        Generate flowchart for a scenario.
        
        Args:
            entry_function: Name of the entry function (optional if auto-detect)
            entry_file: Path to the file containing entry function (for disambiguation)
            detail_level: Detail level ("high", "medium", or "deep")
            output_path: Optional path to save the Mermaid file
            save_intermediate: Whether to save intermediate artifacts (SFM, semantics, etc.)
            use_llm: Whether to use LLM for translation and aggregation
            
        Returns:
            Mermaid flowchart string
        """
        logger.info("=" * 80)
        logger.info("STARTING FLOWCHART GENERATION PIPELINE")
        logger.info("=" * 80)
        
        # Stage 1: Full AST Construction
        logger.info("STAGE 1: Full AST Construction")
        self.ast_parser = ClangASTParser(str(self.project_path), self.compile_args)
        self.call_graph = self.ast_parser.parse_project()
        
        if save_intermediate:
            self._save_call_graph()
        
        # Resolve entry function
        resolved_entry = self._resolve_entry_function(entry_function, entry_file)
        logger.info(f"Resolved entry function: {resolved_entry}")
        
        # Get detailed AST for functions in the call path
        logger.info("Extracting detailed ASTs for reachable functions...")
        self._extract_detailed_asts(resolved_entry)
        
        # Stage 1b: Build CFGs
        logger.info("STAGE 1B: Control Flow Graph Construction")
        self.cfgs = build_project_cfgs(self.call_graph)
        
        if save_intermediate:
            self._save_cfgs()
        
        # Stage 2: Leaf-Level Semantic Extraction
        logger.info("STAGE 2: Leaf-Level Semantic Extraction")
        self.semantics_map = extract_project_semantics(self.call_graph, self.cfgs)
        
        if save_intermediate:
            self._save_semantics()
        
        # Stage 3: Bottom-Up Semantic Aggregation
        logger.info("STAGE 3: Bottom-Up Semantic Aggregation")
        self.aggregator = BottomUpAggregator(llm_model=self.llm_model)
        aggregated = self.aggregator.aggregate_from_entry(resolved_entry, self.call_graph, self.semantics_map)
        
        if not aggregated:
            raise ValueError(f"Failed to aggregate semantics for entry function: {resolved_entry}")
        
        if save_intermediate:
            self._save_aggregated(aggregated)
        
        # Stage 4: Scenario Flow Model Construction
        logger.info("STAGE 4: Scenario Flow Model Construction")
        sfm_builder = SFMBuilder()
        sfm = sfm_builder.build_sfm(resolved_entry, aggregated)
        
        if save_intermediate:
            save_sfm_to_file(sfm, str(self.project_path / "output" / "sfm.json"))
        
        # Stage 5: Detail-Level Filtering
        logger.info(f"STAGE 5: Detail-Level Filtering (level={detail_level})")
        filtered_sfm = apply_detail_level(sfm, detail_level)
        
        if save_intermediate:
            save_sfm_to_file(filtered_sfm, str(self.project_path / "output" / f"sfm_{detail_level}.json"))
        
        # Stage 6: Mermaid Translation
        logger.info("STAGE 6: Mermaid Translation")
        mermaid = generate_flowchart_from_sfm(filtered_sfm, output_path, use_llm=use_llm, llm_model=self.llm_model)
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return mermaid
    
    def _resolve_entry_function(self, entry_function: Optional[str], 
                               entry_file: Optional[str]) -> str:
        """
        Resolve entry function name.
        
        Rules:
        - If both entry_function and entry_file provided: strict resolution
        - If only entry_function provided: error if ambiguous
        - If neither provided: auto-detect using AST evidence
        """
        if entry_function and entry_file:
            # Strict resolution
            candidates = []
            for func_name, func_info in self.call_graph.functions.items():
                if entry_function in func_name and func_info.file_path == entry_file:
                    candidates.append(func_name)
            
            if len(candidates) == 0:
                raise ValueError(f"Function '{entry_function}' not found in file '{entry_file}'")
            elif len(candidates) == 1:
                return candidates[0]
            else:
                raise ValueError(f"Ambiguous function name '{entry_function}' in '{entry_file}'. "
                               f"Candidates: {candidates}")
        
        elif entry_function:
            # Find function by name only
            candidates = [func_name for func_name in self.call_graph.functions 
                         if entry_function in func_name]
            
            if len(candidates) == 0:
                raise ValueError(f"Function '{entry_function}' not found in project")
            elif len(candidates) == 1:
                return candidates[0]
            else:
                raise ValueError(f"Ambiguous function name '{entry_function}'. "
                               f"Found in multiple locations: {candidates}. "
                               f"Please specify --entry-file to disambiguate.")
        
        else:
            # Auto-detect entry point
            if self.call_graph.entry_points:
                # Use the first entry point
                entry = list(self.call_graph.entry_points)[0]
                logger.info(f"Auto-detected entry point: {entry}")
                return entry
            else:
                # Use any function with "main" in the name
                main_candidates = [func_name for func_name in self.call_graph.functions 
                                 if "main" in func_name.lower()]
                if main_candidates:
                    logger.info(f"Auto-detected main function: {main_candidates[0]}")
                    return main_candidates[0]
                else:
                    raise ValueError("Could not auto-detect entry point. Please specify --entry-function.")
    
    def _extract_detailed_asts(self, entry_function: str):
        """Extract detailed ASTs for all functions reachable from entry point"""
        # Get all reachable functions
        reachable = self._get_reachable_functions(entry_function)
        
        logger.info(f"Extracting detailed ASTs for {len(reachable)} reachable functions...")
        
        for func_name in reachable:
            func_info = self.call_graph.functions.get(func_name)
            if func_info and not func_info.ast_root:
                try:
                    self.ast_parser.get_function_ast(func_name, func_info.file_path)
                except Exception as e:
                    logger.warning(f"Failed to extract AST for {func_name}: {e}")
    
    def _get_reachable_functions(self, entry_function: str) -> set:
        """Get all functions reachable from entry function"""
        reachable = set()
        visited = set()
        
        def dfs(func_name: str):
            if func_name in visited or func_name not in self.call_graph.functions:
                return
            visited.add(func_name)
            reachable.add(func_name)
            
            for callee in self.call_graph.functions[func_name].calls:
                dfs(callee)
        
        dfs(entry_function)
        return reachable
    
    def _save_call_graph(self):
        """Save call graph to JSON file"""
        output_dir = self.project_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        call_graph_data = {
            "functions": {
                func_name: {
                    "name": func_info.name,
                    "qualified_name": func_info.qualified_name,
                    "file_path": func_info.file_path,
                    "line_number": func_info.line_number,
                    "is_leaf": func_info.is_leaf,
                    "calls": list(func_info.calls),
                    "called_by": list(func_info.called_by)
                }
                for func_name, func_info in self.call_graph.functions.items()
            },
            "leaf_functions": list(self.call_graph.leaf_functions),
            "entry_points": list(self.call_graph.entry_points)
        }
        
        output_path = output_dir / "call_graph.json"
        with open(output_path, 'w') as f:
            json.dump(call_graph_data, f, indent=2)
        
        logger.info(f"Saved call graph to {output_path}")
    
    def _save_cfgs(self):
        """Save CFGs to JSON file"""
        output_dir = self.project_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        cfgs_data = {}
        for func_name, cfg in self.cfgs.items():
            cfgs_data[func_name] = {
                "function_name": cfg.function_name,
                "entry_block_id": cfg.entry_block_id,
                "exit_block_ids": list(cfg.exit_block_ids),
                "blocks": {
                    str(block_id): {
                        "id": block.id,
                        "has_validation": block.has_validation,
                        "has_state_mutation": block.has_state_mutation,
                        "has_side_effect": block.has_side_effect,
                        "is_guard": block.is_guard,
                        "is_error_exit": block.is_error_exit,
                        "guard_condition": block.guard_condition,
                        "successors": list(block.successors),
                        "predecessors": list(block.predecessors)
                    }
                    for block_id, block in cfg.blocks.items()
                }
            }
        
        output_path = output_dir / "cfgs.json"
        with open(output_path, 'w') as f:
            json.dump(cfgs_data, f, indent=2)
        
        logger.info(f"Saved CFGs to {output_path}")
    
    def _save_semantics(self):
        """Save semantic extraction results to JSON file"""
        output_dir = self.project_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        semantics_data = {
            func_name: semantics.to_dict()
            for func_name, semantics in self.semantics_map.items()
        }
        
        output_path = output_dir / "semantics.json"
        with open(output_path, 'w') as f:
            json.dump(semantics_data, f, indent=2)
        
        logger.info(f"Saved semantics to {output_path}")
    
    def _save_aggregated(self, aggregated):
        """Save aggregated semantics to JSON file"""
        output_dir = self.project_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "aggregated.json"
        with open(output_path, 'w') as f:
            json.dump(aggregated.to_dict(), f, indent=2)
        
        logger.info(f"Saved aggregated semantics to {output_path}")


def generate_flowchart(project_path: str,
                      entry_function: Optional[str] = None,
                      entry_file: Optional[str] = None,
                      detail_level: str = "medium",
                      output_path: Optional[str] = None,
                      llm_model: str = "llama2:7b",
                      save_intermediate: bool = False,
                      use_llm: bool = True) -> str:
    """
    Main entry point for flowchart generation.
    
    Args:
        project_path: Path to the C++ project
        entry_function: Name of the entry function (optional)
        entry_file: Path to the file containing entry function (for disambiguation)
        detail_level: Detail level ("high", "medium", or "deep")
        output_path: Optional path to save the Mermaid file
        llm_model: Name of the Ollama LLM model to use
        save_intermediate: Whether to save intermediate artifacts
        use_llm: Whether to use LLM for translation and aggregation
        
    Returns:
        Mermaid flowchart string
    """
    pipeline = FlowchartPipeline(project_path, llm_model=llm_model)
    return pipeline.generate_flowchart(
        entry_function=entry_function,
        entry_file=entry_file,
        detail_level=detail_level,
        output_path=output_path,
        save_intermediate=save_intermediate,
        use_llm=use_llm
    )
