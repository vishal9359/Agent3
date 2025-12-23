"""
Stage 3: Bottom-Up Backtracking & Semantic Aggregation (LLM-ASSISTED)

This module implements the DocAgent-inspired bottom-up semantic aggregation:
- Start from leaf functions
- Generate local semantic summaries using LLM (based strictly on extracted AST facts)
- Move upward in the call graph
- Combine child summaries
- Elide non-critical operations
- Preserve control-flow and state semantics

Rules:
- Function calls are summarized, not expanded
- Aggregation is semantic, not structural
- No new logic introduced by LLM
- Output is hierarchical semantic understanding, NOT diagrams
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .clang_analyzer import ClangAnalyzer, FunctionCFG, CallRelation
from .semantic_extractor import LeafSemanticExtractor, SemanticAction, SemanticActionType

logger = logging.getLogger(__name__)


@dataclass
class FunctionSemanticSummary:
    """
    Semantic summary of a function's behavior.
    This is the output of aggregation for each function.
    """
    function_name: str
    qualified_name: str
    
    # High-level semantic description
    purpose: str  # What does this function do?
    preconditions: List[str] = field(default_factory=list)  # What must be true before calling?
    postconditions: List[str] = field(default_factory=list)  # What is true after execution?
    
    # Control flow semantics
    decision_points: List[str] = field(default_factory=list)  # Key decisions made
    early_exits: List[str] = field(default_factory=list)  # Conditions for early return
    error_conditions: List[str] = field(default_factory=list)  # Error scenarios
    
    # State impact
    state_mutations: List[str] = field(default_factory=list)  # State changes
    side_effects: List[str] = field(default_factory=list)  # Irreversible side effects
    
    # Dependencies (for upward aggregation)
    calls_summary: Dict[str, str] = field(default_factory=dict)  # {callee: semantic_effect}
    
    # Metadata
    is_leaf: bool = False
    complexity: str = "simple"  # simple, moderate, complex
    criticality: str = "normal"  # low, normal, high, critical
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class SemanticAggregator:
    """
    Performs bottom-up semantic aggregation using LLM.
    
    The LLM is used ONLY to:
    1. Summarize extracted semantic actions into natural language
    2. Combine multiple child summaries into a parent summary
    3. Identify semantic patterns
    
    The LLM is NOT allowed to:
    - Invent logic not present in the code
    - Make assumptions about behavior
    - Add information not derivable from AST
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the semantic aggregator
        
        Args:
            model_name: Ollama model to use for LLM summarization
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,  # Deterministic
            format="json"  # Request JSON output
        )
        
        self.summaries: Dict[str, FunctionSemanticSummary] = {}
        
        # Prompts for different stages
        self.leaf_summary_prompt = ChatPromptTemplate.from_template(
            """You are a C++ code analysis assistant. Your task is to create a semantic summary of a function based STRICTLY on the provided semantic actions extracted from its AST.

DO NOT invent logic. DO NOT make assumptions. ONLY describe what is explicitly present.

Function: {function_name}
Parameters: {parameters}
Return Type: {return_type}

Semantic Actions (extracted from AST):
{semantic_actions}

Your task:
1. Write a concise "purpose" statement (one sentence)
2. List preconditions (what must be true before calling)
3. List postconditions (what is guaranteed after execution)
4. List key decision points
5. List conditions for early exits
6. List error conditions
7. List state mutations
8. List side effects

Respond in JSON format:
{{
  "purpose": "...",
  "preconditions": [...],
  "postconditions": [...],
  "decision_points": [...],
  "early_exits": [...],
  "error_conditions": [...],
  "state_mutations": [...],
  "side_effects": [...],
  "complexity": "simple|moderate|complex",
  "criticality": "low|normal|high|critical"
}}

Be concise and precise. Focus on WHAT, not HOW."""
        )
        
        self.aggregation_prompt = ChatPromptTemplate.from_template(
            """You are a C++ code analysis assistant. Your task is to aggregate semantic summaries of child functions into a parent function summary.

Parent Function: {function_name}
Parameters: {parameters}
Return Type: {return_type}

Parent's Direct Semantic Actions:
{parent_actions}

Child Function Summaries:
{child_summaries}

Your task:
1. Combine child summaries with parent's actions
2. Create a unified semantic summary
3. Elevate important child behavior to parent level
4. Elide trivial operations (logging, metrics, utilities)
5. Preserve control flow and state semantics

Rules:
- Function calls are SUMMARIZED, not expanded
- Focus on business logic and state changes
- Mention child functions semantically, not structurally
- Keep it concise

Respond in JSON format:
{{
  "purpose": "...",
  "preconditions": [...],
  "postconditions": [...],
  "decision_points": [...],
  "early_exits": [...],
  "error_conditions": [...],
  "state_mutations": [...],
  "side_effects": [...],
  "calls_summary": {{"callee_name": "semantic_effect", ...}},
  "complexity": "simple|moderate|complex",
  "criticality": "low|normal|high|critical"
}}"""
        )
        
        logger.info(f"SemanticAggregator initialized with model: {model_name}")
    
    def aggregate(
        self,
        analyzer: ClangAnalyzer,
        extractor: LeafSemanticExtractor,
        entry_function: str
    ) -> Dict[str, FunctionSemanticSummary]:
        """
        Perform bottom-up semantic aggregation from leaf functions to entry point.
        
        Args:
            analyzer: ClangAnalyzer with CFG data
            extractor: LeafSemanticExtractor with semantic actions
            entry_function: Qualified name of entry function
            
        Returns:
            Dictionary of FunctionSemanticSummary for all analyzed functions
        """
        logger.info(f"Starting bottom-up aggregation from entry: {entry_function}")
        
        # Build call graph
        call_graph = self._build_call_graph(analyzer)
        
        # Perform topological sort (bottom-up order)
        processing_order = self._topological_sort(call_graph, entry_function)
        
        logger.info(f"Processing {len(processing_order)} functions in bottom-up order")
        
        # Process each function in bottom-up order
        for func_name in processing_order:
            cfg = analyzer.get_function_cfg(func_name)
            if not cfg:
                logger.warning(f"No CFG found for {func_name}, skipping")
                continue
            
            actions = extractor.get_actions_for_function(func_name)
            
            if cfg.is_leaf or func_name not in call_graph or not call_graph[func_name]:
                # Leaf function: summarize from actions only
                summary = self._summarize_leaf_function(cfg, actions)
            else:
                # Non-leaf: aggregate child summaries
                child_funcs = call_graph.get(func_name, [])
                child_summaries = {
                    child: self.summaries.get(child)
                    for child in child_funcs
                    if child in self.summaries
                }
                summary = self._aggregate_function(cfg, actions, child_summaries)
            
            self.summaries[func_name] = summary
            logger.debug(f"Created summary for: {func_name}")
        
        return self.summaries
    
    def _build_call_graph(self, analyzer: ClangAnalyzer) -> Dict[str, List[str]]:
        """Build a dictionary mapping each function to its callees"""
        graph: Dict[str, List[str]] = {}
        
        for relation in analyzer.call_graph:
            if relation.caller not in graph:
                graph[relation.caller] = []
            graph[relation.caller].append(relation.callee)
        
        return graph
    
    def _topological_sort(
        self, 
        call_graph: Dict[str, List[str]], 
        entry_function: str
    ) -> List[str]:
        """
        Perform topological sort to get bottom-up processing order.
        
        Returns list of function names, with leaf functions first.
        """
        # Build reverse graph (callee -> callers)
        reverse_graph: Dict[str, List[str]] = {}
        all_funcs = set([entry_function])
        
        for caller, callees in call_graph.items():
            all_funcs.add(caller)
            for callee in callees:
                all_funcs.add(callee)
                if callee not in reverse_graph:
                    reverse_graph[callee] = []
                reverse_graph[callee].append(caller)
        
        # Compute in-degree (number of callees)
        in_degree = {func: len(call_graph.get(func, [])) for func in all_funcs}
        
        # Start with leaf functions (in-degree 0)
        queue = [func for func, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            func = queue.pop(0)
            result.append(func)
            
            # Update callers
            for caller in reverse_graph.get(func, []):
                in_degree[caller] -= 1
                if in_degree[caller] == 0:
                    queue.append(caller)
        
        return result
    
    def _summarize_leaf_function(
        self,
        cfg: FunctionCFG,
        actions: List[SemanticAction]
    ) -> FunctionSemanticSummary:
        """
        Summarize a leaf function using LLM.
        
        Args:
            cfg: Function CFG
            actions: Extracted semantic actions
            
        Returns:
            FunctionSemanticSummary
        """
        # Format semantic actions for LLM
        actions_text = self._format_actions(actions)
        
        # Format parameters
        params_text = ", ".join(f"{name}: {typ}" for name, typ in cfg.parameters)
        
        # Call LLM
        messages = self.leaf_summary_prompt.format_messages(
            function_name=cfg.qualified_name,
            parameters=params_text or "none",
            return_type=cfg.return_type,
            semantic_actions=actions_text
        )
        
        try:
            response = self.llm.invoke(messages)
            summary_data = json.loads(response.content)
            
            # Create summary object
            summary = FunctionSemanticSummary(
                function_name=cfg.function_name,
                qualified_name=cfg.qualified_name,
                purpose=summary_data.get("purpose", ""),
                preconditions=summary_data.get("preconditions", []),
                postconditions=summary_data.get("postconditions", []),
                decision_points=summary_data.get("decision_points", []),
                early_exits=summary_data.get("early_exits", []),
                error_conditions=summary_data.get("error_conditions", []),
                state_mutations=summary_data.get("state_mutations", []),
                side_effects=summary_data.get("side_effects", []),
                is_leaf=True,
                complexity=summary_data.get("complexity", "simple"),
                criticality=summary_data.get("criticality", "normal")
            )
            # Enrich/repair summary from concrete actions if LLM output is sparse
            self._enrich_summary_from_actions(summary, actions)

            return summary
            
        except Exception as e:
            logger.error(f"LLM summarization failed for {cfg.qualified_name}: {e}")
            # Fallback: create basic summary from actions
            return self._create_fallback_summary(cfg, actions, is_leaf=True)
    
    def _aggregate_function(
        self,
        cfg: FunctionCFG,
        actions: List[SemanticAction],
        child_summaries: Dict[str, Optional[FunctionSemanticSummary]]
    ) -> FunctionSemanticSummary:
        """
        Aggregate a non-leaf function by combining child summaries.
        
        Args:
            cfg: Function CFG
            actions: Direct semantic actions of this function
            child_summaries: Summaries of called functions
            
        Returns:
            FunctionSemanticSummary
        """
        # Format parent's actions
        parent_actions_text = self._format_actions(actions)
        
        # Format child summaries
        child_summaries_text = self._format_child_summaries(child_summaries)
        
        # Format parameters
        params_text = ", ".join(f"{name}: {typ}" for name, typ in cfg.parameters)
        
        # Call LLM
        messages = self.aggregation_prompt.format_messages(
            function_name=cfg.qualified_name,
            parameters=params_text or "none",
            return_type=cfg.return_type,
            parent_actions=parent_actions_text,
            child_summaries=child_summaries_text
        )
        
        try:
            response = self.llm.invoke(messages)
            summary_data = json.loads(response.content)
            
            # Create summary object
            summary = FunctionSemanticSummary(
                function_name=cfg.function_name,
                qualified_name=cfg.qualified_name,
                purpose=summary_data.get("purpose", ""),
                preconditions=summary_data.get("preconditions", []),
                postconditions=summary_data.get("postconditions", []),
                decision_points=summary_data.get("decision_points", []),
                early_exits=summary_data.get("early_exits", []),
                error_conditions=summary_data.get("error_conditions", []),
                state_mutations=summary_data.get("state_mutations", []),
                side_effects=summary_data.get("side_effects", []),
                calls_summary=summary_data.get("calls_summary", {}),
                is_leaf=False,
                complexity=summary_data.get("complexity", "moderate"),
                criticality=summary_data.get("criticality", "normal")
            )
            # Enrich/repair summary from concrete actions if LLM output is sparse
            self._enrich_summary_from_actions(summary, actions)

            return summary
            
        except Exception as e:
            logger.error(f"LLM aggregation failed for {cfg.qualified_name}: {e}")
            # Fallback: combine actions and child summaries naively
            return self._create_fallback_summary(cfg, actions, is_leaf=False, child_summaries=child_summaries)
    
    def _enrich_summary_from_actions(
        self,
        summary: FunctionSemanticSummary,
        actions: List[SemanticAction],
    ) -> None:
        """
        Ensure summary has enough structure by deriving fields from concrete actions.
        
        This is critical when the LLM returns very sparse JSON (e.g. empty lists) even
        though we have rich SemanticAction data. We **only** fill fields that are
        currently empty, so any non-empty LLM output is preserved.
        """
        if not actions:
            return

        # If LLM already populated these lists, don't override them.
        needs_pre = not summary.preconditions
        needs_post = not summary.postconditions
        needs_decisions = not summary.decision_points
        needs_early = not summary.early_exits
        needs_errors = not summary.error_conditions
        needs_state = not summary.state_mutations
        needs_side = not summary.side_effects

        if not any(
            [
                needs_pre,
                needs_post,
                needs_decisions,
                needs_early,
                needs_errors,
                needs_state,
                needs_side,
            ]
        ):
            # Nothing to do – LLM already produced a rich summary
            return

        for action in actions:
            atype = action.action_type
            effect = action.effect

            # Validation / permission checks → preconditions + decisions
            if atype in (
                SemanticActionType.VALIDATION,
                SemanticActionType.PERMISSION_CHECK,
            ):
                if needs_pre:
                    summary.preconditions.append(effect)
                if needs_decisions:
                    summary.decision_points.append(effect)

            # State mutation → state_mutations (+ postconditions as a coarse proxy)
            if atype == SemanticActionType.STATE_MUTATION:
                if needs_state:
                    summary.state_mutations.append(effect)
                if needs_post:
                    summary.postconditions.append(effect)

            # Side effects → side_effects
            if atype == SemanticActionType.SIDE_EFFECT:
                if needs_side:
                    summary.side_effects.append(effect)

            # Early exits → early_exits (and sometimes error_conditions)
            if atype == SemanticActionType.EARLY_EXIT:
                if needs_early:
                    summary.early_exits.append(effect)
                if action.is_error_path and needs_errors:
                    summary.error_conditions.append(effect)

            # Error handling → error_conditions
            if atype == SemanticActionType.ERROR_HANDLING:
                if needs_errors:
                    summary.error_conditions.append(effect)

        # Truncate to keep SFM readable
        summary.preconditions = summary.preconditions[:5]
        summary.postconditions = summary.postconditions[:5]
        summary.decision_points = summary.decision_points[:5]
        summary.early_exits = summary.early_exits[:5]
        summary.error_conditions = summary.error_conditions[:5]
        summary.state_mutations = summary.state_mutations[:5]
        summary.side_effects = summary.side_effects[:5]
    
    def _format_actions(self, actions: List[SemanticAction]) -> str:
        """Format semantic actions for LLM consumption"""
        if not actions:
            return "No semantic actions identified."
        
        lines = []
        for i, action in enumerate(actions, 1):
            lines.append(f"{i}. [{action.action_type.value}] {action.effect}")
            if action.control_impact:
                lines.append(f"   - Affects control flow")
            if action.state_impact:
                lines.append(f"   - Modifies state")
            if action.is_critical:
                lines.append(f"   - CRITICAL")
        
        return "\n".join(lines)
    
    def _format_child_summaries(
        self, 
        child_summaries: Dict[str, Optional[FunctionSemanticSummary]]
    ) -> str:
        """Format child function summaries for LLM consumption"""
        if not child_summaries:
            return "No child functions called."
        
        lines = []
        for func_name, summary in child_summaries.items():
            if summary:
                lines.append(f"\n{func_name}:")
                lines.append(f"  Purpose: {summary.purpose}")
                if summary.decision_points:
                    lines.append(f"  Decisions: {', '.join(summary.decision_points[:3])}")
                if summary.state_mutations:
                    lines.append(f"  State changes: {', '.join(summary.state_mutations[:3])}")
                if summary.side_effects:
                    lines.append(f"  Side effects: {', '.join(summary.side_effects[:3])}")
            else:
                lines.append(f"\n{func_name}: (no summary available)")
        
        return "\n".join(lines)
    
    def _create_fallback_summary(
        self,
        cfg: FunctionCFG,
        actions: List[SemanticAction],
        is_leaf: bool,
        child_summaries: Optional[Dict[str, Optional[FunctionSemanticSummary]]] = None
    ) -> FunctionSemanticSummary:
        """Create a basic summary when LLM fails"""
        
        # Extract key information from actions
        purpose = f"Performs {cfg.function_name}"
        decision_points = [a.effect for a in actions if a.action_type == SemanticActionType.VALIDATION]
        early_exits = [a.effect for a in actions if a.action_type == SemanticActionType.EARLY_EXIT]
        error_conditions = [a.effect for a in actions if a.action_type == SemanticActionType.ERROR_HANDLING]
        state_mutations = [a.effect for a in actions if a.action_type == SemanticActionType.STATE_MUTATION]
        side_effects = [a.effect for a in actions if a.action_type == SemanticActionType.SIDE_EFFECT]
        
        calls_summary = {}
        if child_summaries:
            for func_name, summary in child_summaries.items():
                if summary:
                    calls_summary[func_name] = summary.purpose
        
        return FunctionSemanticSummary(
            function_name=cfg.function_name,
            qualified_name=cfg.qualified_name,
            purpose=purpose,
            decision_points=decision_points[:5],  # Limit to 5
            early_exits=early_exits[:5],
            error_conditions=error_conditions[:5],
            state_mutations=state_mutations[:5],
            side_effects=side_effects[:5],
            calls_summary=calls_summary,
            is_leaf=is_leaf,
            complexity="moderate",
            criticality="normal"
        )
    
    def get_summary(self, qualified_name: str) -> Optional[FunctionSemanticSummary]:
        """Retrieve semantic summary for a specific function"""
        return self.summaries.get(qualified_name)
    
    def export_summaries(self, output_path: str) -> None:
        """Export all summaries to a JSON file"""
        data = {name: summary.to_dict() for name, summary in self.summaries.items()}
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.summaries)} summaries to {output_path}")
