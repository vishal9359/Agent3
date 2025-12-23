"""
Bottom-Up Aggregation Engine - Stage 3

This module implements DocAgent-inspired bottom-up semantic aggregation.
It starts from leaf functions and backtraces upward through the call graph,
using LLM to combine child summaries into parent summaries.

Key principles:
- Function calls are summarized, not expanded
- Aggregation is semantic, not structural
- No new logic introduced by LLM
- Produces hierarchical semantic understanding, NOT diagrams
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

# Lazy imports for LangChain - only import when actually used
if TYPE_CHECKING:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

from .clang_ast_parser import CallRelationship, ProjectAST
from .leaf_semantic_extractor import FunctionSemantics, SemanticActionType

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSemantics:
    """
    Aggregated semantic summary for a function.
    
    This is built bottom-up by combining child function summaries.
    """
    function_name: str
    level: int  # 0 = leaf, 1+ = higher levels
    semantic_summary: str  # LLM-generated summary
    control_flow_summary: str  # Control flow semantics
    state_impact_summary: str  # State change semantics
    child_functions: List[str] = field(default_factory=list)
    dominant_operations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    # Internal storage for compatibility attributes
    _local_actions: List[Any] = field(default_factory=list, repr=False)  # Store actions for compatibility
    _child_summaries_dict: Dict[str, str] = field(default_factory=dict, repr=False)  # Store child summaries
    
    @property
    def aggregated_summary(self) -> str:
        """
        Compatibility property: returns semantic_summary for backward compatibility.
        Some code expects 'aggregated_summary' instead of 'semantic_summary'.
        """
        return self.semantic_summary
    
    @property
    def local_actions(self) -> List[Any]:
        """
        Compatibility property: returns local_actions for backward compatibility.
        Some code expects 'local_actions' attribute.
        """
        return self._local_actions
    
    @property
    def is_leaf(self) -> bool:
        """
        Compatibility property: returns True if level is 0 (leaf function).
        Some code expects 'is_leaf' attribute.
        """
        return self.level == 0
    
    @property
    def child_summaries(self) -> Dict[str, str]:
        """
        Compatibility property: returns child_summaries dict for backward compatibility.
        Maps child function names to their summaries.
        """
        return self._child_summaries_dict
    
    @property
    def critical_operations(self) -> List[str]:
        """
        Compatibility property: returns dominant_operations as critical_operations.
        Some code expects 'critical_operations' attribute.
        """
        return self.dominant_operations


@dataclass
class AggregationContext:
    """Context for LLM aggregation prompt."""
    function_name: str
    leaf_semantics: Optional[FunctionSemantics]
    child_summaries: List[AggregatedSemantics]
    call_relationships: List[CallRelationship]


class BottomUpAggregator:
    """
    Bottom-up semantic aggregation engine.
    
    This mirrors DocAgent's approach:
    1. Start at leaf functions
    2. Generate local semantic summaries (LLM-assisted)
    3. Move upward, combining child summaries
    4. Continue until reaching entry function
    """

    def __init__(
        self,
        project_ast: ProjectAST,
        leaf_semantics: Dict[str, FunctionSemantics],
        chat_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ):
        """
        Initialize the aggregator.
        
        Args:
            project_ast: Complete project AST
            leaf_semantics: Leaf-level semantic extractions
            chat_model: Ollama chat model name
            ollama_base_url: Ollama server URL
        """
        self.project_ast = project_ast
        self.leaf_semantics = leaf_semantics
        self.aggregated: Dict[str, AggregatedSemantics] = {}
        
        # Initialize LLM
        self.llm = self._init_llm(chat_model, ollama_base_url)
        self.parser = None  # Will be set in _init_llm if LangChain is available
        
        # Build reverse call graph (callee -> callers)
        self.reverse_call_graph: Dict[str, List[str]] = {}
        self._build_reverse_call_graph()
    
    def _init_llm(
        self, chat_model: Optional[str], ollama_base_url: Optional[str]
    ) -> Optional['ChatOllama']:
        """Initialize LLM for semantic aggregation."""
        if not chat_model:
            logger.warning("No chat model specified - using rule-based aggregation only")
            return None
        
        try:
            # Import LangChain only when actually needed
            from langchain_community.chat_models import ChatOllama
            from langchain_core.output_parsers import StrOutputParser
            
            llm = ChatOllama(
                model=chat_model or "llama3.2:3b",
                base_url=ollama_base_url or "http://localhost:11434",
                temperature=0.1,  # Low temperature for deterministic output
            )
            self.parser = StrOutputParser()
            return llm
        except ImportError as e:
            logger.error(f"LangChain not installed: {e}. Using rule-based aggregation only.")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _build_reverse_call_graph(self):
        """Build reverse call graph (callee -> list of callers)."""
        for rel in self.project_ast.call_graph:
            if rel.callee not in self.reverse_call_graph:
                self.reverse_call_graph[rel.callee] = []
            self.reverse_call_graph[rel.callee].append(rel.caller)
    
    def aggregate(self, entry_function: str) -> Dict[str, AggregatedSemantics]:
        """
        Perform bottom-up aggregation starting from entry function.
        
        Args:
            entry_function: Entry point function name
        
        Returns:
            Dictionary of aggregated semantics for all functions in the call chain
        """
        logger.info(f"Starting bottom-up aggregation from entry: {entry_function}")
        
        # Get all functions reachable from entry
        call_chain = self._get_call_chain(entry_function)
        
        logger.info(f"Call chain contains {len(call_chain)} functions")
        
        # Sort by call depth (leaf functions first)
        sorted_functions = self._topological_sort(call_chain)
        
        # Process in bottom-up order
        for func_name in sorted_functions:
            self._aggregate_function(func_name)
        
        logger.info(f"Aggregation complete - processed {len(self.aggregated)} functions")
        
        return self.aggregated
    
    def _get_call_chain(self, entry_function: str) -> Set[str]:
        """Get all functions reachable from entry function."""
        visited = set()
        stack = [entry_function]
        
        while stack:
            func = stack.pop()
            if func in visited:
                continue
            
            visited.add(func)
            
            # Add callees
            for rel in self.project_ast.call_graph:
                if rel.caller == func:
                    stack.append(rel.callee)
        
        return visited
    
    def _topological_sort(self, functions: Set[str]) -> List[str]:
        """
        Sort functions in topological order (leaf functions first).
        
        This ensures we process dependencies before dependents.
        """
        # Calculate in-degree (number of callees)
        in_degree = {func: 0 for func in functions}
        
        for rel in self.project_ast.call_graph:
            if rel.caller in functions and rel.callee in functions:
                in_degree[rel.callee] += 1
        
        # Start with leaf functions (in-degree 0)
        queue = [func for func, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            func = queue.pop(0)
            result.append(func)
            
            # Decrease in-degree of callers
            if func in self.reverse_call_graph:
                for caller in self.reverse_call_graph[func]:
                    if caller in in_degree:
                        in_degree[caller] -= 1
                        if in_degree[caller] == 0:
                            queue.append(caller)
        
        # Add any remaining functions (cycles)
        for func in functions:
            if func not in result:
                result.append(func)
        
        return result
    
    def _aggregate_function(self, function_name: str):
        """Aggregate semantics for a single function."""
        logger.debug(f"Aggregating: {function_name}")
        
        # Get leaf semantics
        leaf_sem = self.leaf_semantics.get(function_name)
        
        # Get child summaries (callees)
        child_summaries = self._get_child_summaries(function_name)
        
        # Get call relationships
        call_rels = [
            rel
            for rel in self.project_ast.call_graph
            if rel.caller == function_name
        ]
        
        # Determine level
        level = 0 if leaf_sem and leaf_sem.is_leaf else max(
            [cs.level for cs in child_summaries], default=0
        ) + 1
        
        # Create aggregation context
        context = AggregationContext(
            function_name=function_name,
            leaf_semantics=leaf_sem,
            child_summaries=child_summaries,
            call_relationships=call_rels,
        )
        
        # Generate aggregated semantics
        if self.llm and (child_summaries or (leaf_sem and not leaf_sem.is_leaf)):
            # Use LLM for non-trivial aggregation
            aggregated = self._llm_aggregate(context, level)
        else:
            # Use rule-based aggregation for leaf functions
            aggregated = self._rule_based_aggregate(context, level)
        
        self.aggregated[function_name] = aggregated
    
    def _get_child_summaries(self, function_name: str) -> List[AggregatedSemantics]:
        """Get aggregated summaries of all functions called by this function."""
        child_summaries = []
        
        for rel in self.project_ast.call_graph:
            if rel.caller == function_name and rel.callee in self.aggregated:
                child_summaries.append(self.aggregated[rel.callee])
        
        return child_summaries
    
    def _rule_based_aggregate(
        self, context: AggregationContext, level: int
    ) -> AggregatedSemantics:
        """
        Rule-based aggregation for leaf functions or when LLM is unavailable.
        
        This produces a deterministic summary based on extracted semantics.
        """
        leaf_sem = context.leaf_semantics
        
        if not leaf_sem:
            # Build child summaries dict from child_summaries
            child_summaries_dict = {}
            if context.child_summaries:
                for child in context.child_summaries:
                    child_summaries_dict[child.function_name] = child.semantic_summary
            
            return AggregatedSemantics(
                function_name=context.function_name,
                level=level,
                semantic_summary="No semantic information available",
                control_flow_summary="Unknown control flow",
                state_impact_summary="Unknown state impact",
                _local_actions=[],  # Empty actions for compatibility
                _child_summaries_dict=child_summaries_dict,  # Store child summaries for compatibility
            )
        
        # Collect actions
        all_actions = []
        for block_sem in leaf_sem.blocks.values():
            all_actions.extend(block_sem.actions)
        
        # Filter significant actions
        significant_actions = [
            a
            for a in all_actions
            if a.action_type
            not in {
                SemanticActionType.LOGGING,
                SemanticActionType.METRIC,
                SemanticActionType.UTILITY,
            }
        ]
        
        # Build semantic summary
        if not significant_actions:
            semantic_summary = f"{context.function_name}: performs utility operations"
        else:
            action_types = [a.action_type.value for a in significant_actions]
            dominant = max(set(action_types), key=action_types.count)
            semantic_summary = (
                f"{context.function_name}: primarily performs {dominant}"
            )
        
        # Build control flow summary
        control_actions = [a for a in significant_actions if a.control_impact]
        if control_actions:
            control_flow_summary = f"Controls flow via {len(control_actions)} decision points"
        else:
            control_flow_summary = "No control flow impact"
        
        # Build state impact summary
        state_actions = [a for a in significant_actions if a.state_impact]
        if state_actions:
            state_types = [a.action_type.value for a in state_actions]
            state_impact_summary = f"Modifies state: {', '.join(set(state_types))}"
        else:
            state_impact_summary = "No state modification"
        
        # Extract dominant operations
        dominant_ops = list(set([a.effect for a in significant_actions[:5]]))
        
        # Build child summaries dict from child_summaries
        child_summaries_dict = {}
        if context.child_summaries:
            for child in context.child_summaries:
                child_summaries_dict[child.function_name] = child.semantic_summary
        
        return AggregatedSemantics(
            function_name=context.function_name,
            level=level,
            semantic_summary=semantic_summary,
            control_flow_summary=control_flow_summary,
            state_impact_summary=state_impact_summary,
            dominant_operations=dominant_ops,
            confidence=0.8,
            _local_actions=significant_actions,  # Store actions for compatibility
            _child_summaries_dict=child_summaries_dict,  # Store child summaries for compatibility
        )
    
    def _llm_aggregate(
        self, context: AggregationContext, level: int
    ) -> AggregatedSemantics:
        """
        LLM-assisted aggregation.
        
        This combines leaf semantics and child summaries using LLM.
        The LLM acts as a semantic compressor, NOT a logic inferrer.
        """
        prompt = self._build_aggregation_prompt(context)
        
        try:
            # Import LangChain messages only when needed
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt),
            ]
            
            response = self.llm.invoke(messages)
            summary_text = self.parser.invoke(response) if self.parser else str(response)
            
            # Parse LLM response
            aggregated = self._parse_llm_response(
                summary_text, context.function_name, level
            )
            # Enrich aggregated semantics with concrete actions and child summaries
            # so that downstream SFM construction has structured data to work with.

            # 1) Populate _local_actions from leaf semantics (significant actions only)
            significant_actions: List[Any] = []
            if context.leaf_semantics:
                all_actions: List[Any] = []
                for block_sem in context.leaf_semantics.blocks.values():
                    all_actions.extend(getattr(block_sem, "actions", []))
                
                # Filter out logging/metrics/utility actions if they exist on this enum
                for action in all_actions:
                    atype = getattr(action, "action_type", None)
                    if atype is None:
                        continue
                    # Compare by name to stay robust across different SemanticActionType enums
                    atype_name = getattr(atype, "name", str(atype))
                    if atype_name in {"LOGGING", "METRIC", "UTILITY"}:
                        continue
                    significant_actions.append(action)

            aggregated._local_actions = significant_actions

            # 2) Populate _child_summaries_dict from child_summaries
            child_summaries_dict: Dict[str, str] = {}
            if context.child_summaries:
                for child in context.child_summaries:
                    # child is AggregatedSemantics
                    child_summaries_dict[child.function_name] = child.semantic_summary
            aggregated._child_summaries_dict = child_summaries_dict

            # 3) Maintain explicit child_functions list for compatibility
            aggregated.child_functions = [cs.function_name for cs in context.child_summaries]
            
            return aggregated
            
        except Exception as e:
            logger.error(f"LLM aggregation failed for {context.function_name}: {e}")
            # Fallback to rule-based
            return self._rule_based_aggregate(context, level)
    
    def _get_system_prompt(self) -> str:
        """System prompt for LLM aggregation."""
        return """You are a semantic code analyzer performing bottom-up semantic aggregation.

Your task: Given a function's leaf-level semantic actions and summaries of functions it calls,
produce a concise semantic summary.

RULES:
1. Summarize semantics, do NOT expand function calls
2. Preserve control flow semantics (validations, decisions, exits)
3. Preserve state impact semantics (mutations, side effects)
4. Elide non-critical operations (logging, metrics, utilities)
5. Do NOT invent logic - only summarize what is given
6. Output ONLY a JSON object with these fields:
   {
     "semantic_summary": "Brief description of what the function does",
     "control_flow_summary": "How the function controls execution flow",
     "state_impact_summary": "What state changes the function makes",
     "dominant_operations": ["operation1", "operation2", ...]
   }

Keep summaries concise (1-2 sentences each).
"""
    
    def _build_aggregation_prompt(self, context: AggregationContext) -> str:
        """Build prompt for LLM aggregation."""
        prompt_parts = [
            f"Function: {context.function_name}",
            "",
            "=== LEAF-LEVEL SEMANTICS ===",
        ]
        
        # Add leaf semantics
        if context.leaf_semantics:
            prompt_parts.append(self._format_leaf_semantics(context.leaf_semantics))
        else:
            prompt_parts.append("(No leaf semantics - function only calls other functions)")
        
        prompt_parts.append("")
        prompt_parts.append("=== CHILD FUNCTION SUMMARIES ===")
        
        # Add child summaries
        if context.child_summaries:
            for child in context.child_summaries:
                prompt_parts.append(f"\n{child.function_name}:")
                prompt_parts.append(f"  Summary: {child.semantic_summary}")
                prompt_parts.append(f"  Control: {child.control_flow_summary}")
                prompt_parts.append(f"  State: {child.state_impact_summary}")
        else:
            prompt_parts.append("(No child functions)")
        
        prompt_parts.append("")
        prompt_parts.append("=== YOUR TASK ===")
        prompt_parts.append(
            f"Aggregate the above information into a semantic summary for {context.function_name}."
        )
        prompt_parts.append("Output JSON only.")
        
        return "\n".join(prompt_parts)
    
    def _format_leaf_semantics(self, leaf_sem: FunctionSemantics) -> str:
        """Format leaf semantics for prompt."""
        lines = []
        
        for block_id, block_sem in leaf_sem.blocks.items():
            if block_sem.actions:
                lines.append(f"Block {block_id}:")
                for action in block_sem.actions:
                    if action.action_type not in {
                        SemanticActionType.LOGGING,
                        SemanticActionType.METRIC,
                        SemanticActionType.UTILITY,
                    }:
                        lines.append(f"  - {action.action_type.value}: {action.effect}")
        
        return "\n".join(lines) if lines else "(No significant actions)"
    
    def _parse_llm_response(
        self, response: str, function_name: str, level: int
    ) -> AggregatedSemantics:
        """Parse LLM JSON response into AggregatedSemantics."""
        try:
            # Extract JSON from response (handle potential text around it)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                return AggregatedSemantics(
                    function_name=function_name,
                    level=level,
                    semantic_summary=data.get("semantic_summary", ""),
                    control_flow_summary=data.get("control_flow_summary", ""),
                    state_impact_summary=data.get("state_impact_summary", ""),
                    dominant_operations=data.get("dominant_operations", []),
                    confidence=0.9,
                    _local_actions=[],  # Will be populated by caller
                    _child_summaries_dict={},  # Will be populated by caller
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Return a basic summary
            return AggregatedSemantics(
                function_name=function_name,
                level=level,
                semantic_summary=response[:200],
                control_flow_summary="See summary",
                state_impact_summary="See summary",
                confidence=0.5,
                _local_actions=[],  # Will be populated by caller
                _child_summaries_dict={},  # Will be populated by caller
            )
    
    def get_entry_summary(self, entry_function: str) -> Optional[AggregatedSemantics]:
        """Get the aggregated summary for the entry function."""
        return self.aggregated.get(entry_function)
    
    def get_function_summary(self, function_name: str) -> Optional[AggregatedSemantics]:
        """Get aggregated summary for any function."""
        return self.aggregated.get(function_name)
    
    def export_hierarchy(self, entry_function: str) -> Dict:
        """
        Export the semantic hierarchy as a dictionary.
        
        This can be used for visualization or further processing.
        """
        if entry_function not in self.aggregated:
            return {}
        
        def build_tree(func_name: str, visited: Set[str]) -> Dict:
            if func_name in visited:
                return {"function": func_name, "cycle": True}
            
            visited.add(func_name)
            agg = self.aggregated.get(func_name)
            
            if not agg:
                return {"function": func_name, "error": "No aggregation data"}
            
            node = {
                "function": func_name,
                "level": agg.level,
                "summary": agg.semantic_summary,
                "control_flow": agg.control_flow_summary,
                "state_impact": agg.state_impact_summary,
                "children": [],
            }
            
            for child in agg.child_functions:
                node["children"].append(build_tree(child, visited.copy()))
            
            return node
        
        return build_tree(entry_function, set())


@dataclass
class FunctionSummary:
    """
    Compatibility wrapper for aggregated semantics to match expected interface.
    
    This provides the properties expected by downstream stages.
    """
    aggregated: AggregatedSemantics
    
    @property
    def purpose(self) -> str:
        return self.aggregated.semantic_summary
    
    @property
    def preconditions(self) -> List[str]:
        # Extract preconditions from control flow summary if any
        return []
    
    @property
    def control_flow(self) -> List[str]:
        return [self.aggregated.control_flow_summary]
    
    @property
    def state_changes(self) -> List[str]:
        return [self.aggregated.state_impact_summary]
    
    @property
    def dependencies(self) -> List[str]:
        return self.aggregated.child_functions


def aggregate_semantics(
    project_ast: ProjectAST,
    leaf_semantics: Dict[str, FunctionSemantics],
    entry_function: str,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> FunctionSummary:
    """
    Convenience function for bottom-up semantic aggregation.
    
    This wraps the BottomUpAggregator class for easier functional-style usage.
    
    Args:
        project_ast: Complete project AST
        leaf_semantics: Leaf-level semantic extractions
        entry_function: Entry function name to start aggregation from
        llm_model: Optional Ollama model name
        llm_base_url: Optional Ollama server URL
    
    Returns:
        FunctionSummary for the entry function
    """
    aggregator = BottomUpAggregator(
        project_ast=project_ast,
        leaf_semantics=leaf_semantics,
        chat_model=llm_model,
        ollama_base_url=llm_base_url,
    )
    
    # Perform aggregation
    aggregator.aggregate(entry_function)
    
    # Return entry function summary
    entry_summary = aggregator.get_entry_summary(entry_function)
    
    if not entry_summary:
        raise ValueError(f"Failed to aggregate semantics for entry function: {entry_function}")
    
    return FunctionSummary(aggregated=entry_summary)
