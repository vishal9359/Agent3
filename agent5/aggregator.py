"""
Stage 3: Bottom-Up Backtracking & Semantic Aggregation

This module implements DocAgent-inspired bottom-up semantic aggregation:
- Start from leaf functions
- Generate local semantic summaries using LLM (based strictly on extracted AST facts)
- Move upward in call graph
- Combine child summaries
- Elide non-critical operations
- Preserve control-flow and state semantics
- Continue backtracking until reaching entry function

Function calls are summarized, not expanded.
Aggregation is semantic, not structural.
No new logic introduced by LLM.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import json

from agent5.clang_ast_parser import CallGraph, FunctionInfo
from agent5.semantic_extractor import FunctionSemantics, SemanticAction, SemanticActionType
from agent5.logging_utils import get_logger
from agent5.ollama_compat import get_ollama_llm

logger = get_logger(__name__)


@dataclass
class AggregatedSemantics:
    """
    Aggregated semantic understanding of a function.
    This combines local semantics with summaries of called functions.
    """
    function_name: str
    is_leaf: bool
    level: int  # Level in call graph (0 = leaf, increases going up)
    local_actions: List[SemanticAction] = field(default_factory=list)
    aggregated_summary: str = ""  # High-level semantic summary
    control_flow_summary: str = ""  # Summary of control flow behavior
    state_impact_summary: str = ""  # Summary of state mutations
    child_summaries: Dict[str, str] = field(default_factory=dict)  # Called function -> summary
    critical_operations: List[str] = field(default_factory=list)  # Critical ops to preserve
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "functionName": self.function_name,
            "isLeaf": self.is_leaf,
            "level": self.level,
            "localActions": [action.to_dict() for action in self.local_actions],
            "aggregatedSummary": self.aggregated_summary,
            "controlFlowSummary": self.control_flow_summary,
            "stateImpactSummary": self.state_impact_summary,
            "childSummaries": self.child_summaries,
            "criticalOperations": self.critical_operations
        }


class BottomUpAggregator:
    """
    Aggregates semantic understanding bottom-up through the call graph.
    This is Stage 3: Bottom-Up Backtracking & Semantic Aggregation.
    """
    
    def __init__(self, llm_model: str = "llama2:7b"):
        """
        Initialize the aggregator.
        
        Args:
            llm_model: Name of the Ollama LLM model to use for summarization
        """
        self.llm = get_ollama_llm(llm_model)
        self.aggregated_cache: Dict[str, AggregatedSemantics] = {}
    
    def aggregate_from_entry(self, entry_function: str, call_graph: CallGraph, 
                            semantics_map: Dict[str, FunctionSemantics]) -> AggregatedSemantics:
        """
        Aggregate semantic understanding starting from an entry function.
        Uses bottom-up backtracking: processes leaf functions first, then moves upward.
        
        Args:
            entry_function: Name of the entry point function
            call_graph: Call graph of the project
            semantics_map: Map of function names to their leaf-level semantics
            
        Returns:
            AggregatedSemantics for the entry function
        """
        logger.info(f"Starting bottom-up aggregation from entry function: {entry_function}")
        
        # Clear cache
        self.aggregated_cache = {}
        
        # Perform topological sort to determine processing order (bottom-up)
        processing_order = self._topological_sort_bottom_up(entry_function, call_graph)
        
        logger.info(f"Processing order determined: {len(processing_order)} functions to process")
        
        # Process functions in bottom-up order
        for func_name in processing_order:
            if func_name in semantics_map:
                self._aggregate_function(func_name, call_graph, semantics_map)
        
        # Return aggregated semantics for entry function
        result = self.aggregated_cache.get(entry_function)
        if result:
            logger.info(f"Aggregation complete. Summary: {result.aggregated_summary[:200]}...")
        else:
            logger.warning(f"No aggregated semantics found for entry function {entry_function}")
        
        return result
    
    def _topological_sort_bottom_up(self, entry_function: str, call_graph: CallGraph) -> List[str]:
        """
        Topological sort to determine processing order (bottom-up).
        Leaf functions come first, entry function comes last.
        """
        # Build subgraph reachable from entry function
        reachable = self._get_reachable_functions(entry_function, call_graph)
        
        # Compute in-degree for each function (number of callers within subgraph)
        in_degree = {func: 0 for func in reachable}
        for func in reachable:
            if func in call_graph.functions:
                for callee in call_graph.functions[func].calls:
                    if callee in reachable:
                        in_degree[callee] = in_degree.get(callee, 0) + 1
        
        # Kahn's algorithm for topological sort
        queue = [func for func in reachable if in_degree[func] == 0]
        result = []
        
        while queue:
            # Process all nodes with in-degree 0
            current = queue.pop(0)
            result.append(current)
            
            # Reduce in-degree of callees
            if current in call_graph.functions:
                for callee in call_graph.functions[current].calls:
                    if callee in reachable:
                        in_degree[callee] -= 1
                        if in_degree[callee] == 0:
                            queue.append(callee)
        
        return result
    
    def _get_reachable_functions(self, entry_function: str, call_graph: CallGraph) -> Set[str]:
        """Get all functions reachable from entry function"""
        reachable = set()
        visited = set()
        
        def dfs(func_name: str):
            if func_name in visited or func_name not in call_graph.functions:
                return
            visited.add(func_name)
            reachable.add(func_name)
            
            for callee in call_graph.functions[func_name].calls:
                dfs(callee)
        
        dfs(entry_function)
        return reachable
    
    def _aggregate_function(self, func_name: str, call_graph: CallGraph, 
                           semantics_map: Dict[str, FunctionSemantics]):
        """Aggregate semantics for a single function"""
        if func_name in self.aggregated_cache:
            return  # Already processed
        
        logger.debug(f"Aggregating function: {func_name}")
        
        func_info = call_graph.functions.get(func_name)
        local_semantics = semantics_map.get(func_name)
        
        if not func_info or not local_semantics:
            logger.warning(f"Missing information for function {func_name}")
            return
        
        # Create aggregated semantics object
        aggregated = AggregatedSemantics(
            function_name=func_name,
            is_leaf=func_info.is_leaf,
            level=0,  # Will be updated
            local_actions=local_semantics.actions
        )
        
        # Collect summaries of called functions
        for callee in func_info.calls:
            if callee in self.aggregated_cache:
                callee_summary = self.aggregated_cache[callee].aggregated_summary
                aggregated.child_summaries[callee] = callee_summary
                aggregated.level = max(aggregated.level, self.aggregated_cache[callee].level + 1)
        
        # Generate aggregated summary using LLM
        aggregated.aggregated_summary = self._generate_summary(aggregated, local_semantics)
        
        # Generate control flow summary
        aggregated.control_flow_summary = self._summarize_control_flow(aggregated, local_semantics)
        
        # Generate state impact summary
        aggregated.state_impact_summary = self._summarize_state_impact(aggregated, local_semantics)
        
        # Identify critical operations
        aggregated.critical_operations = self._identify_critical_operations(aggregated, local_semantics)
        
        # Cache result
        self.aggregated_cache[func_name] = aggregated
        
        logger.debug(f"Aggregated {func_name}: {aggregated.aggregated_summary[:100]}...")
    
    def _generate_summary(self, aggregated: AggregatedSemantics, 
                         local_semantics: FunctionSemantics) -> str:
        """Generate high-level semantic summary using LLM"""
        # Build prompt from extracted facts
        prompt = self._build_summary_prompt(aggregated, local_semantics)
        
        try:
            response = self.llm.invoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            return summary.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed for {aggregated.function_name}: {e}")
            # Fallback: deterministic summary
            return self._generate_deterministic_summary(aggregated, local_semantics)
    
    def _build_summary_prompt(self, aggregated: AggregatedSemantics, 
                             local_semantics: FunctionSemantics) -> str:
        """Build LLM prompt for semantic summarization"""
        prompt = f"""You are a technical documentation assistant. Summarize the semantic behavior of a C++ function based ONLY on the provided facts.

Function: {aggregated.function_name}
Is Leaf Function: {aggregated.is_leaf}

LOCAL ACTIONS (extracted from AST):
"""
        
        for i, action in enumerate(aggregated.local_actions, 1):
            prompt += f"{i}. {action.action_type.value}: {action.effect}\n"
            if action.control_impact:
                prompt += f"   - Affects control flow\n"
            if action.state_impact:
                prompt += f"   - Affects state\n"
        
        if aggregated.child_summaries:
            prompt += "\nCALLED FUNCTIONS (summaries):\n"
            for callee, summary in aggregated.child_summaries.items():
                prompt += f"- {callee}: {summary}\n"
        
        prompt += """
TASK:
Generate a concise semantic summary (2-3 sentences) that describes:
1. What this function does at a high level
2. Key control flow decisions
3. Important state changes or side effects

RULES:
- Base your summary ONLY on the provided facts
- Do NOT invent logic or behavior not shown in the facts
- Focus on semantic meaning, not implementation details
- Collapse utility operations into high-level descriptions
- Preserve critical control flow and state semantics

Summary:"""
        
        return prompt
    
    def _generate_deterministic_summary(self, aggregated: AggregatedSemantics, 
                                       local_semantics: FunctionSemantics) -> str:
        """Generate deterministic summary as fallback"""
        parts = []
        
        # Count action types
        validations = sum(1 for a in aggregated.local_actions if a.action_type == SemanticActionType.VALIDATION)
        state_changes = sum(1 for a in aggregated.local_actions if a.action_type == SemanticActionType.STATE_MUTATION)
        decisions = sum(1 for a in aggregated.local_actions if a.action_type == SemanticActionType.DECISION)
        side_effects = sum(1 for a in aggregated.local_actions if a.action_type == SemanticActionType.IRREVERSIBLE_SIDE_EFFECT)
        
        parts.append(f"Function {aggregated.function_name}")
        
        if validations > 0:
            parts.append(f"performs {validations} validation(s)")
        if decisions > 0:
            parts.append(f"has {decisions} decision point(s)")
        if state_changes > 0:
            parts.append(f"modifies state {state_changes} time(s)")
        if side_effects > 0:
            parts.append(f"has {side_effects} side effect(s)")
        
        if aggregated.child_summaries:
            parts.append(f"calls {len(aggregated.child_summaries)} function(s)")
        
        return " | ".join(parts) if len(parts) > 1 else parts[0]
    
    def _summarize_control_flow(self, aggregated: AggregatedSemantics, 
                               local_semantics: FunctionSemantics) -> str:
        """Summarize control flow behavior"""
        decisions = [a for a in aggregated.local_actions if a.action_type == SemanticActionType.DECISION]
        early_exits = [a for a in aggregated.local_actions if a.action_type == SemanticActionType.EARLY_EXIT]
        
        if not decisions and not early_exits:
            return "Linear control flow"
        
        parts = []
        if decisions:
            parts.append(f"{len(decisions)} decision point(s)")
        if early_exits:
            parts.append(f"{len(early_exits)} early exit(s)")
        
        return ", ".join(parts)
    
    def _summarize_state_impact(self, aggregated: AggregatedSemantics, 
                               local_semantics: FunctionSemantics) -> str:
        """Summarize state mutation behavior"""
        state_mutations = [a for a in aggregated.local_actions if a.action_type == SemanticActionType.STATE_MUTATION]
        side_effects = [a for a in aggregated.local_actions if a.action_type == SemanticActionType.IRREVERSIBLE_SIDE_EFFECT]
        
        if not state_mutations and not side_effects:
            return "No state changes"
        
        parts = []
        if state_mutations:
            parts.append(f"{len(state_mutations)} state mutation(s)")
        if side_effects:
            parts.append(f"{len(side_effects)} irreversible side effect(s)")
        
        return ", ".join(parts)
    
    def _identify_critical_operations(self, aggregated: AggregatedSemantics, 
                                     local_semantics: FunctionSemantics) -> List[str]:
        """Identify operations that are critical to preserve in documentation"""
        critical = []
        
        for action in aggregated.local_actions:
            # Critical: operations affecting control flow or state
            if action.control_impact or action.state_impact:
                critical.append(action.effect)
            
            # Critical: validations and permission checks
            if action.action_type in [SemanticActionType.VALIDATION, SemanticActionType.PERMISSION_CHECK]:
                critical.append(action.effect)
            
            # Critical: irreversible side effects
            if action.action_type == SemanticActionType.IRREVERSIBLE_SIDE_EFFECT:
                critical.append(action.effect)
        
        return critical





