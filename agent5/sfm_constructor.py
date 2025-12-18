"""
Stage 4: Scenario Flow Model Construction (SINGLE SOURCE OF TRUTH)

This module converts aggregated semantic summaries into the Scenario Flow Model (SFM).
The SFM is a deterministic, JSON-based representation that serves as the single source
of truth for diagram generation.

Rules:
- One SFM per entry-point scenario
- Explicit mapping to detail levels (high, medium, deep)
- No Mermaid generation here
- No prose generation here
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set

from .semantic_aggregator import FunctionSemanticSummary, SemanticAggregator
from .semantic_extractor import SemanticAction, SemanticActionType

logger = logging.getLogger(__name__)


class DetailLevel(Enum):
    """Detail levels for flowchart generation"""
    HIGH = "high"  # Business-level steps only
    MEDIUM = "medium"  # All decisions + validations + state changes
    DEEP = "deep"  # Expanded critical sub-operations


class SFMNodeType(Enum):
    """Types of nodes in the Scenario Flow Model"""
    START = "start"
    END = "end"
    PROCESS = "process"  # A single action/operation
    DECISION = "decision"  # A branching point
    SUBPROCESS = "subprocess"  # Represents aggregated child logic
    ERROR = "error"  # Error handling


@dataclass
class SFMNode:
    """A node in the Scenario Flow Model"""
    id: str
    node_type: SFMNodeType
    label: str
    description: str = ""
    
    # Edges
    next_nodes: List[str] = field(default_factory=list)
    
    # For decision nodes
    condition: Optional[str] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None
    
    # Detail level inclusion
    min_detail_level: DetailLevel = DetailLevel.HIGH
    
    # Metadata
    source_function: Optional[str] = None
    is_critical: bool = False
    is_error_path: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert enums to strings
        data['node_type'] = self.node_type.value
        data['min_detail_level'] = self.min_detail_level.value
        return data


@dataclass
class ScenarioFlowModel:
    """
    The complete Scenario Flow Model for a single entry-point scenario.
    This is the authoritative representation of the scenario logic.
    """
    scenario_name: str
    entry_function: str
    start_node_id: str
    end_node_ids: List[str]
    nodes: Dict[str, SFMNode]
    
    # Metadata
    detail_levels_supported: List[DetailLevel] = field(
        default_factory=lambda: [DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP]
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "scenario_name": self.scenario_name,
            "entry_function": self.entry_function,
            "start_node_id": self.start_node_id,
            "end_node_ids": self.end_node_ids,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "detail_levels_supported": [level.value for level in self.detail_levels_supported]
        }
    
    def validate(self) -> List[str]:
        """
        Validate the SFM for completeness and consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check start node exists
        if self.start_node_id not in self.nodes:
            errors.append(f"Start node '{self.start_node_id}' not found in nodes")
        
        # Check all end nodes exist
        for end_id in self.end_node_ids:
            if end_id not in self.nodes:
                errors.append(f"End node '{end_id}' not found in nodes")
        
        # Check all referenced nodes exist
        for node_id, node in self.nodes.items():
            for next_id in node.next_nodes:
                if next_id not in self.nodes:
                    errors.append(f"Node '{node_id}' references non-existent node '{next_id}'")
            
            if node.true_branch and node.true_branch not in self.nodes:
                errors.append(f"Decision node '{node_id}' references non-existent true branch '{node.true_branch}'")
            
            if node.false_branch and node.false_branch not in self.nodes:
                errors.append(f"Decision node '{node_id}' references non-existent false branch '{node.false_branch}'")
        
        # Check that all paths lead to an end node
        reachable = self._find_reachable_nodes()
        for end_id in self.end_node_ids:
            if end_id not in reachable:
                errors.append(f"End node '{end_id}' is not reachable from start")
        
        return errors
    
    def _find_reachable_nodes(self) -> Set[str]:
        """Find all nodes reachable from start"""
        visited = set()
        queue = [self.start_node_id]
        
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            
            visited.add(node_id)
            node = self.nodes.get(node_id)
            
            if node:
                queue.extend(node.next_nodes)
                if node.true_branch:
                    queue.append(node.true_branch)
                if node.false_branch:
                    queue.append(node.false_branch)
        
        return visited


class SFMConstructor:
    """
    Constructs Scenario Flow Models from aggregated semantic summaries.
    
    This is a rule-based, deterministic process that converts semantic
    understanding into a structured flow representation.
    """
    
    def __init__(self):
        self.node_counter = 0
        logger.info("SFMConstructor initialized")
    
    def construct(
        self,
        entry_function: str,
        semantic_summary: FunctionSemanticSummary,
        scenario_name: Optional[str] = None
    ) -> ScenarioFlowModel:
        """
        Construct an SFM from a semantic summary.
        
        Args:
            entry_function: Qualified name of entry function
            semantic_summary: Aggregated semantic summary for the function
            scenario_name: Optional name for the scenario
            
        Returns:
            ScenarioFlowModel
        """
        self.node_counter = 0
        
        if scenario_name is None:
            scenario_name = semantic_summary.function_name
        
        logger.info(f"Constructing SFM for scenario: {scenario_name}")
        
        nodes: Dict[str, SFMNode] = {}
        
        # Create start node
        start_id = self._create_node_id("start")
        start_node = SFMNode(
            id=start_id,
            node_type=SFMNodeType.START,
            label="Start",
            description=f"Entry: {semantic_summary.function_name}",
            min_detail_level=DetailLevel.HIGH
        )
        nodes[start_id] = start_node
        
        # Build flow from semantic summary
        current_id = start_id
        current_id, end_ids = self._build_flow_from_summary(
            semantic_summary,
            current_id,
            nodes
        )
        
        # Create end nodes if not already created
        if not end_ids:
            end_id = self._create_node_id("end")
            end_node = SFMNode(
                id=end_id,
                node_type=SFMNodeType.END,
                label="End",
                description=f"Exit: {semantic_summary.function_name}",
                min_detail_level=DetailLevel.HIGH
            )
            nodes[end_id] = end_node
            
            # Connect current to end
            if current_id and current_id in nodes:
                nodes[current_id].next_nodes.append(end_id)
            
            end_ids = [end_id]
        
        # Create SFM
        sfm = ScenarioFlowModel(
            scenario_name=scenario_name,
            entry_function=entry_function,
            start_node_id=start_id,
            end_node_ids=end_ids,
            nodes=nodes
        )
        
        # Validate
        errors = sfm.validate()
        if errors:
            logger.warning(f"SFM validation warnings: {errors}")
        
        return sfm
    
    def _build_flow_from_summary(
        self,
        summary: FunctionSemanticSummary,
        current_id: str,
        nodes: Dict[str, SFMNode]
    ) -> tuple[str, List[str]]:
        """
        Build flow nodes from a semantic summary.
        
        Returns:
            (last_node_id, end_node_ids)
        """
        end_ids = []
        
        # 1. Add preconditions as validations (MEDIUM level)
        for precond in summary.preconditions[:3]:  # Limit to 3
            node_id = self._create_node_id("validation")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.DECISION,
                label=f"Validate: {self._shorten(precond)}",
                description=precond,
                condition=precond,
                min_detail_level=DetailLevel.MEDIUM,
                is_critical=True,
                source_function=summary.qualified_name
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            
            # Add error branch (DEEP level)
            error_id = self._create_node_id("error")
            error_node = SFMNode(
                id=error_id,
                node_type=SFMNodeType.ERROR,
                label="Validation Failed",
                description=f"Reject: {precond}",
                min_detail_level=DetailLevel.DEEP,
                is_error_path=True
            )
            nodes[error_id] = error_node
            node.false_branch = error_id
            end_ids.append(error_id)
            
            # Continue on true branch
            current_id = node_id
        
        # 2. Add decision points (MEDIUM level)
        for decision in summary.decision_points[:5]:
            node_id = self._create_node_id("decision")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.DECISION,
                label=f"Decision: {self._shorten(decision)}",
                description=decision,
                condition=decision,
                min_detail_level=DetailLevel.MEDIUM,
                source_function=summary.qualified_name
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            current_id = node_id
        
        # 3. Add state mutations (MEDIUM level, but expanded in DEEP)
        for mutation in summary.state_mutations[:5]:
            # For HIGH level, skip mutations
            # For MEDIUM level, show as single step
            # For DEEP level, could expand if it's from a child call
            
            node_id = self._create_node_id("mutation")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.PROCESS,
                label=f"Update: {self._shorten(mutation)}",
                description=mutation,
                min_detail_level=DetailLevel.MEDIUM,
                is_critical=True,
                source_function=summary.qualified_name
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            current_id = node_id
        
        # 4. Add side effects (MEDIUM level)
        for side_effect in summary.side_effects[:3]:
            node_id = self._create_node_id("side_effect")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.PROCESS,
                label=f"Execute: {self._shorten(side_effect)}",
                description=side_effect,
                min_detail_level=DetailLevel.MEDIUM,
                is_critical=True,
                source_function=summary.qualified_name
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            current_id = node_id
        
        # 5. Add child calls as subprocesses (HIGH level for business, DEEP for others)
        for callee, effect in list(summary.calls_summary.items())[:5]:
            # Determine if this is a business call or utility
            is_business = self._is_business_call(callee, effect)
            
            node_id = self._create_node_id("subprocess")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.SUBPROCESS,
                label=self._shorten(effect),
                description=f"{callee}: {effect}",
                min_detail_level=DetailLevel.HIGH if is_business else DetailLevel.DEEP,
                source_function=callee
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            current_id = node_id
        
        # 6. Add error conditions as decision nodes (DEEP level)
        for error_cond in summary.error_conditions[:3]:
            node_id = self._create_node_id("error_check")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.DECISION,
                label=f"Check: {self._shorten(error_cond)}",
                description=error_cond,
                condition=error_cond,
                min_detail_level=DetailLevel.DEEP,
                is_error_path=True,
                source_function=summary.qualified_name
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            
            # Add error exit
            error_id = self._create_node_id("error_exit")
            error_node = SFMNode(
                id=error_id,
                node_type=SFMNodeType.ERROR,
                label="Error Exit",
                description=error_cond,
                min_detail_level=DetailLevel.DEEP,
                is_error_path=True
            )
            nodes[error_id] = error_node
            node.true_branch = error_id
            end_ids.append(error_id)
            
            current_id = node_id
        
        # 7. Add early exits (MEDIUM level)
        for early_exit in summary.early_exits[:2]:
            node_id = self._create_node_id("early_exit")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.DECISION,
                label=f"Check: {self._shorten(early_exit)}",
                description=early_exit,
                min_detail_level=DetailLevel.MEDIUM,
                source_function=summary.qualified_name
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            
            # Add exit node
            exit_id = self._create_node_id("exit")
            exit_node = SFMNode(
                id=exit_id,
                node_type=SFMNodeType.END,
                label="Early Exit",
                description=early_exit,
                min_detail_level=DetailLevel.MEDIUM
            )
            nodes[exit_id] = exit_node
            node.true_branch = exit_id
            end_ids.append(exit_id)
            
            current_id = node_id
        
        # 8. Add postconditions as final checks (DEEP level)
        for postcond in summary.postconditions[:2]:
            node_id = self._create_node_id("postcondition")
            node = SFMNode(
                id=node_id,
                node_type=SFMNodeType.PROCESS,
                label=f"Ensure: {self._shorten(postcond)}",
                description=postcond,
                min_detail_level=DetailLevel.DEEP,
                source_function=summary.qualified_name
            )
            nodes[node_id] = node
            nodes[current_id].next_nodes.append(node_id)
            current_id = node_id
        
        return current_id, end_ids
    
    def _is_business_call(self, callee: str, effect: str) -> bool:
        """Determine if a call is business logic (HIGH level) vs utility (DEEP level)"""
        
        # Utility indicators
        utility_keywords = [
            'log', 'debug', 'trace', 'metric', 'monitor',
            'util', 'helper', 'format', 'convert', 'parse',
            'serialize', 'deserialize', 'alloc', 'free'
        ]
        
        callee_lower = callee.lower()
        effect_lower = effect.lower()
        
        # Check if it's a utility
        if any(kw in callee_lower or kw in effect_lower for kw in utility_keywords):
            return False
        
        # Business indicators
        business_keywords = [
            'create', 'update', 'delete', 'process', 'execute',
            'validate', 'authorize', 'send', 'receive', 'handle'
        ]
        
        if any(kw in callee_lower or kw in effect_lower for kw in business_keywords):
            return True
        
        # Default: assume business logic
        return True
    
    def _create_node_id(self, node_type: str) -> str:
        """Generate a unique node ID"""
        node_id = f"{node_type}_{self.node_counter}"
        self.node_counter += 1
        return node_id
    
    def _shorten(self, text: str, max_length: int = 60) -> str:
        """Shorten text for labels"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def export_sfm(self, sfm: ScenarioFlowModel, output_path: str) -> None:
        """Export SFM to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(sfm.to_dict(), f, indent=2)
        
        logger.info(f"Exported SFM to {output_path}")
