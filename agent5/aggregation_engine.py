"""
Bottom-Up Semantic Aggregation Engine (Stage 3).

This module implements the DocAgent-inspired bottom-up backtracking strategy:
1. Start from leaf functions
2. Generate local semantic summaries using LLM (strictly based on AST facts)
3. Move upward in the call graph
4. Combine child summaries
5. Elide non-critical operations
6. Preserve control-flow and state semantics

Rules:
- Function calls are summarized, NOT expanded
- Aggregation is semantic, NOT structural
- NO new logic introduced by LLM
- Output is hierarchical semantic understanding, NOT diagrams
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent5.clang_parser import ASTContext, FunctionInfo
from agent5.semantic_extractor import SemanticAction, SemanticExtractor, SemanticType


@dataclass
class SemanticSummary:
    """
    Semantic summary of a function or code block.

    This is the output of Stage 3 - aggregated semantic understanding.
    """

    function_name: str
    summary: str  # High-level semantic description
    preconditions: list[str]  # What must be true before execution
    postconditions: list[str]  # What is true after execution
    side_effects: list[str]  # Persistent changes
    control_flow: list[str]  # Key decision points
    error_paths: list[str]  # Error handling paths
    child_summaries: dict[str, str] = field(default_factory=dict)  # callee -> summary
    semantic_actions: list[SemanticAction] = field(default_factory=list)
    confidence: float = 1.0


class AggregationEngine:
    """Bottom-up semantic aggregation engine."""

    def __init__(
        self,
        ast_context: ASTContext,
        chat_model: BaseChatModel | None = None,
        use_llm: bool = False,
    ):
        """
        Initialize aggregation engine.

        Args:
            ast_context: Complete AST context from Stage 1
            chat_model: LLM for semantic summarization (optional)
            use_llm: Whether to use LLM for aggregation
        """
        self.context = ast_context
        self.chat_model = chat_model
        self.use_llm = use_llm
        self.extractor = SemanticExtractor()
        self.summaries: dict[str, SemanticSummary] = {}  # function_name -> summary

    def aggregate(self, entry_function: str) -> SemanticSummary:
        """
        Perform bottom-up aggregation starting from entry function.

        Args:
            entry_function: Entry point function name

        Returns:
            Aggregated semantic summary for the entry function
        """
        # Build aggregation order (leaf-first, using topological sort)
        aggregation_order = self._build_aggregation_order(entry_function)

        # Process functions in order (leaf-first)
        for func_name in aggregation_order:
            func_info = self.context.functions.get(func_name)
            if func_info:
                self.summaries[func_name] = self._summarize_function(func_info)

        # Return the entry function's summary
        return self.summaries.get(entry_function) or self._create_empty_summary(entry_function)

    def _build_aggregation_order(self, entry_function: str) -> list[str]:
        """
        Build bottom-up aggregation order using topological sort.

        Returns functions in leaf-first order (dependencies before dependents).
        """
        visited = set()
        order = []

        def visit(func_name: str):
            if func_name in visited:
                return
            visited.add(func_name)

            # Get function info
            func_info = self.context.functions.get(func_name)
            if not func_info:
                return

            # Visit callees first (bottom-up)
            for callee in func_info.calls:
                if callee in self.context.functions:  # Only project functions
                    visit(callee)

            order.append(func_name)

        visit(entry_function)
        return order

    def _summarize_function(self, func_info: FunctionInfo) -> SemanticSummary:
        """
        Generate semantic summary for a function.

        This is where LLM is used (if enabled) for semantic summarization.
        """
        # Stage 2: Extract leaf-level semantic actions
        actions = self.extractor.extract_from_function(func_info)
        actions = self.extractor.filter_utility_actions(actions)

        # Collect child summaries (callees)
        child_summaries = {}
        for callee in func_info.calls:
            if callee in self.summaries:
                child_summaries[callee] = self.summaries[callee].summary

        # Generate summary
        if self.use_llm and self.chat_model:
            summary = self._llm_summarize(func_info, actions, child_summaries)
        else:
            summary = self._rule_based_summarize(func_info, actions, child_summaries)

        return summary

    def _llm_summarize(
        self,
        func_info: FunctionInfo,
        actions: list[SemanticAction],
        child_summaries: dict[str, str],
    ) -> SemanticSummary:
        """
        Use LLM to generate semantic summary.

        CRITICAL: LLM is used ONLY for semantic interpretation, NOT for logic.
        """
        system_prompt = """You are a semantic code analyzer.
Your task is to generate a concise semantic summary of a C++ function based STRICTLY on:
1. Extracted semantic actions (from AST)
2. Summaries of called functions

Rules:
- Do NOT invent logic
- Do NOT expand beyond provided facts
- Focus on WHAT the function does semantically
- Identify preconditions, postconditions, side effects, control flow
- Be concise and precise
- Use natural language, NOT code syntax
"""

        # Build context
        actions_text = "\n".join(
            [f"- {action.type.value}: {action.effect} (line {action.line})" for action in actions]
        )

        child_summaries_text = "\n".join(
            [f"- {callee}: {summary}" for callee, summary in child_summaries.items()]
        )

        user_prompt = f"""Function: {func_info.name}
Return type: {func_info.return_type}
Parameters: {", ".join([f"{name}: {type_}" for name, type_ in func_info.parameters])}

Semantic Actions:
{actions_text}

Called Functions:
{child_summaries_text if child_summaries_text else "(none)"}

Generate a structured semantic summary with:
1. High-level summary (1 sentence)
2. Preconditions (what must be true before)
3. Postconditions (what is true after)
4. Side effects (persistent changes)
5. Control flow (key decision points)
6. Error paths (error handling)

Format as JSON:
{{
  "summary": "...",
  "preconditions": [...],
  "postconditions": [...],
  "side_effects": [...],
  "control_flow": [...],
  "error_paths": [...]
}}
"""

        try:
            response = self.chat_model.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )

            # Parse JSON response
            import json

            result = json.loads(response.content)

            return SemanticSummary(
                function_name=func_info.name,
                summary=result.get("summary", ""),
                preconditions=result.get("preconditions", []),
                postconditions=result.get("postconditions", []),
                side_effects=result.get("side_effects", []),
                control_flow=result.get("control_flow", []),
                error_paths=result.get("error_paths", []),
                child_summaries=child_summaries,
                semantic_actions=actions,
                confidence=0.8,  # LLM-based, slightly lower confidence
            )

        except Exception as e:
            print(f"Warning: LLM summarization failed for {func_info.name}: {e}")
            # Fallback to rule-based
            return self._rule_based_summarize(func_info, actions, child_summaries)

    def _rule_based_summarize(
        self,
        func_info: FunctionInfo,
        actions: list[SemanticAction],
        child_summaries: dict[str, str],
    ) -> SemanticSummary:
        """
        Generate semantic summary using deterministic rules (no LLM).

        This is the fallback method and is used when LLM is disabled.
        """
        # Generate high-level summary
        business_actions = [a for a in actions if a.type == SemanticType.BUSINESS_LOGIC]
        if business_actions:
            summary = f"Function '{func_info.name}' performs: " + ", ".join(
                [a.effect for a in business_actions[:3]]
            )
        else:
            summary = f"Function '{func_info.name}' executes {len(actions)} operations"

        # Extract preconditions
        preconditions = [
            a.effect
            for a in actions
            if a.type in {SemanticType.VALIDATION, SemanticType.PERMISSION_CHECK}
        ]

        # Extract postconditions (state mutations)
        postconditions = [a.effect for a in actions if a.type == SemanticType.STATE_MUTATION]

        # Extract side effects
        side_effects = [a.effect for a in actions if a.type == SemanticType.SIDE_EFFECT]

        # Extract control flow
        control_flow = [a.effect for a in actions if a.type == SemanticType.DECISION and a.control_impact]

        # Extract error paths
        error_paths = [a.effect for a in actions if a.type == SemanticType.EARLY_EXIT]

        return SemanticSummary(
            function_name=func_info.name,
            summary=summary,
            preconditions=preconditions,
            postconditions=postconditions,
            side_effects=side_effects,
            control_flow=control_flow,
            error_paths=error_paths,
            child_summaries=child_summaries,
            semantic_actions=actions,
            confidence=1.0,  # Rule-based, high confidence
        )

    def _create_empty_summary(self, function_name: str) -> SemanticSummary:
        """Create empty summary for missing function."""
        return SemanticSummary(
            function_name=function_name,
            summary=f"Function '{function_name}' not found in project",
            preconditions=[],
            postconditions=[],
            side_effects=[],
            control_flow=[],
            error_paths=[],
        )

    def export_summaries(self, output_path: Path) -> None:
        """Export semantic summaries to JSON for debugging."""
        import json

        data = {
            func_name: {
                "summary": summary.summary,
                "preconditions": summary.preconditions,
                "postconditions": summary.postconditions,
                "side_effects": summary.side_effects,
                "control_flow": summary.control_flow,
                "error_paths": summary.error_paths,
                "child_summaries": summary.child_summaries,
                "action_count": len(summary.semantic_actions),
                "confidence": summary.confidence,
            }
            for func_name, summary in self.summaries.items()
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
