"""
Stage 4: Scenario Flow Model (SFM) Construction

Convert aggregated semantics into a Scenario Flow Model (SFM).
This is the SINGLE SOURCE OF TRUTH for flowchart generation.

SFM Properties:
- One SFM per entry-point scenario
- Explicit mapping to detail levels
- No Mermaid, no prose
- Deterministic structure
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from agent5.aggregator import AggregatedSemantics
from agent5.semantic_extractor import SemanticAction, SemanticActionType
from agent5.logging_utils import get_logger

logger = get_logger(__name__)


class SFMNodeType(Enum):
    """Types of nodes in the Scenario Flow Model"""
    START = "start"
    END = "end"
    DECISION = "decision"
    VALIDATION = "validation"
    STATE_CHANGE = "state_change"
    OPERATION = "operation"
    ERROR_EXIT = "error_exit"


class DetailLevel(Enum):
    """Detail levels for flowchart generation"""
    HIGH = "high"  # Business-level steps only
    MEDIUM = "medium"  # All decisions + validations + state changes
    DEEP = "deep"  # Expanded critical sub-operations affecting control/state


@dataclass
class SFMNode:
    """Node in the Scenario Flow Model"""
    id: str
    node_type: SFMNodeType
    label: str
    description: str = ""
    detail_levels: Set[DetailLevel] = field(default_factory=lambda: {DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP})
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "nodeType": self.node_type.value,
            "label": self.label,
            "description": self.description,
            "detailLevels": [dl.value for dl in self.detail_levels],
            "metadata": self.metadata
        }


@dataclass
class SFMEdge:
    """Edge in the Scenario Flow Model"""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    edge_type: str = "normal"  # "normal", "true", "false", "error"
    detail_levels: Set[DetailLevel] = field(default_factory=lambda: {DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP})
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "from": self.from_node,
            "to": self.to_node,
            "condition": self.condition,
            "edgeType": self.edge_type,
            "detailLevels": [dl.value for dl in self.detail_levels]
        }


@dataclass
class ScenarioFlowModel:
    """
    Scenario Flow Model (SFM) - Single source of truth for flowchart generation.
    
    This model represents the high-level scenario flow, NOT function calls.
    It is built from aggregated semantics and preserves semantic meaning.
    """
    scenario_name: str
    entry_function: str
    nodes: List[SFMNode] = field(default_factory=list)
    edges: List[SFMEdge] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_node(self, node: SFMNode):
        """Add a node to the model"""
        self.nodes.append(node)
    
    def add_edge(self, edge: SFMEdge):
        """Add an edge to the model"""
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[SFMNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "scenarioName": self.scenario_name,
            "entryFunction": self.entry_function,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class SFMBuilder:
    """
    Builds Scenario Flow Model from aggregated semantics.
    This is Stage 4: Scenario Flow Model Construction.
    """
    
    def __init__(self):
        self.next_node_id = 0
    
    def build_sfm(self, entry_function: str, aggregated: AggregatedSemantics) -> ScenarioFlowModel:
        """
        Build Scenario Flow Model from aggregated semantics.
        
        Args:
            entry_function: Name of the entry function
            aggregated: Aggregated semantics for the entry function
            
        Returns:
            ScenarioFlowModel representing the scenario flow
        """
        logger.info(f"Building Scenario Flow Model for: {entry_function}")
        
        self.next_node_id = 0
        
        sfm = ScenarioFlowModel(
            scenario_name=self._generate_scenario_name(entry_function),
            entry_function=entry_function,
            metadata={
                "function": aggregated.function_name,
                "level": aggregated.level,
                "summary": aggregated.aggregated_summary
            }
        )
        
        # Create start node
        start_node = self._create_node(
            SFMNodeType.START,
            f"Start: {self._generate_scenario_name(entry_function)}",
            "Entry point of the scenario",
            detail_levels={DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP}
        )
        sfm.add_node(start_node)
        
        # Convert semantic actions to SFM nodes
        current_node_id = start_node.id
        current_node_id = self._convert_actions_to_nodes(aggregated, sfm, current_node_id)
        
        # Create end node
        end_node = self._create_node(
            SFMNodeType.END,
            "End: Scenario complete",
            "Successful completion of the scenario",
            detail_levels={DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP}
        )
        sfm.add_node(end_node)
        
        # Connect last node to end
        if current_node_id:
            sfm.add_edge(SFMEdge(
                from_node=current_node_id,
                to_node=end_node.id,
                detail_levels={DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP}
            ))
        
        logger.info(f"SFM built with {len(sfm.nodes)} nodes and {len(sfm.edges)} edges")
        
        return sfm
    
    def _generate_scenario_name(self, entry_function: str) -> str:
        """Generate a human-readable scenario name from function name"""
        # Remove namespace/class prefixes
        name = entry_function.split("::")[-1]
        
        # Convert camelCase/PascalCase to words
        import re
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        name = name.replace('_', ' ').title()
        
        return name
    
    def _create_node(self, node_type: SFMNodeType, label: str, description: str = "",
                    detail_levels: Optional[Set[DetailLevel]] = None,
                    metadata: Optional[Dict] = None) -> SFMNode:
        """Create a new SFM node with unique ID"""
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        
        if detail_levels is None:
            detail_levels = {DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP}
        
        return SFMNode(
            id=node_id,
            node_type=node_type,
            label=label,
            description=description,
            detail_levels=detail_levels,
            metadata=metadata or {}
        )
    
    def _convert_actions_to_nodes(self, aggregated: AggregatedSemantics, sfm: ScenarioFlowModel, 
                                 current_node_id: str) -> str:
        """
        Convert semantic actions to SFM nodes and edges.
        Returns the ID of the last node created.
        """
        # Group actions by type and criticality
        business_actions = []
        validation_actions = []
        decision_actions = []
        state_change_actions = []
        critical_actions = []
        
        for action in aggregated.local_actions:
            # Classify action based on type and impact
            if action.action_type == SemanticActionType.VALIDATION:
                validation_actions.append(action)
            elif action.action_type == SemanticActionType.DECISION:
                decision_actions.append(action)
            elif action.action_type in [SemanticActionType.STATE_MUTATION, SemanticActionType.IRREVERSIBLE_SIDE_EFFECT]:
                state_change_actions.append(action)
            elif action.control_impact or action.state_impact:
                critical_actions.append(action)
            else:
                business_actions.append(action)
        
        # Create nodes based on classification
        
        # 1. Validations (included in MEDIUM and DEEP)
        for action in validation_actions:
            node = self._create_validation_node(action)
            sfm.add_node(node)
            sfm.add_edge(SFMEdge(
                from_node=current_node_id,
                to_node=node.id,
                detail_levels={DetailLevel.MEDIUM, DetailLevel.DEEP}
            ))
            current_node_id = node.id
        
        # 2. Decisions (included in all levels, but expanded in MEDIUM/DEEP)
        for action in decision_actions:
            node = self._create_decision_node(action)
            sfm.add_node(node)
            sfm.add_edge(SFMEdge(
                from_node=current_node_id,
                to_node=node.id,
                detail_levels={DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP}
            ))
            current_node_id = node.id
        
        # 3. State changes (included in MEDIUM and DEEP)
        for action in state_change_actions:
            node = self._create_state_change_node(action)
            sfm.add_node(node)
            sfm.add_edge(SFMEdge(
                from_node=current_node_id,
                to_node=node.id,
                detail_levels={DetailLevel.MEDIUM, DetailLevel.DEEP}
            ))
            current_node_id = node.id
        
        # 4. Critical operations (included only in DEEP)
        for action in critical_actions:
            node = self._create_operation_node(action)
            sfm.add_node(node)
            sfm.add_edge(SFMEdge(
                from_node=current_node_id,
                to_node=node.id,
                detail_levels={DetailLevel.DEEP}
            ))
            current_node_id = node.id
        
        # 5. Business operations (HIGH level summary)
        if aggregated.aggregated_summary and not (validation_actions or decision_actions or state_change_actions):
            # If no other actions, create a high-level business operation node
            node = self._create_node(
                SFMNodeType.OPERATION,
                self._generate_scenario_name(aggregated.function_name),
                aggregated.aggregated_summary,
                detail_levels={DetailLevel.HIGH}
            )
            sfm.add_node(node)
            sfm.add_edge(SFMEdge(
                from_node=current_node_id,
                to_node=node.id,
                detail_levels={DetailLevel.HIGH}
            ))
            current_node_id = node.id
        
        return current_node_id
    
    def _create_validation_node(self, action: SemanticAction) -> SFMNode:
        """Create a validation node from semantic action"""
        return self._create_node(
            SFMNodeType.VALIDATION,
            self._clean_action_label(action.effect),
            action.effect,
            detail_levels={DetailLevel.MEDIUM, DetailLevel.DEEP},
            metadata={
                "action_type": action.action_type.value,
                "source": action.source_location
            }
        )
    
    def _create_decision_node(self, action: SemanticAction) -> SFMNode:
        """Create a decision node from semantic action"""
        label = action.guard_condition or self._clean_action_label(action.effect)
        return self._create_node(
            SFMNodeType.DECISION,
            label,
            action.effect,
            detail_levels={DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP},
            metadata={
                "action_type": action.action_type.value,
                "condition": action.guard_condition,
                "source": action.source_location
            }
        )
    
    def _create_state_change_node(self, action: SemanticAction) -> SFMNode:
        """Create a state change node from semantic action"""
        return self._create_node(
            SFMNodeType.STATE_CHANGE,
            self._clean_action_label(action.effect),
            action.effect,
            detail_levels={DetailLevel.MEDIUM, DetailLevel.DEEP},
            metadata={
                "action_type": action.action_type.value,
                "source": action.source_location
            }
        )
    
    def _create_operation_node(self, action: SemanticAction) -> SFMNode:
        """Create an operation node from semantic action"""
        return self._create_node(
            SFMNodeType.OPERATION,
            self._clean_action_label(action.effect),
            action.effect,
            detail_levels={DetailLevel.DEEP},
            metadata={
                "action_type": action.action_type.value,
                "control_impact": action.control_impact,
                "state_impact": action.state_impact,
                "source": action.source_location
            }
        )
    
    def _clean_action_label(self, effect: str) -> str:
        """Clean up action effect string for use as node label"""
        # Remove prefixes like "Validate:", "Call function:", etc.
        import re
        label = re.sub(r'^(Validate|Check permission|Modify state|Call function|Decision point):\s*', '', effect)
        
        # Truncate if too long
        max_length = 50
        if len(label) > max_length:
            label = label[:max_length-3] + "..."
        
        return label


def save_sfm_to_file(sfm: ScenarioFlowModel, output_path: str):
    """Save SFM to a JSON file"""
    with open(output_path, 'w') as f:
        f.write(sfm.to_json())
    logger.info(f"Saved SFM to {output_path}")


def load_sfm_from_file(input_path: str) -> ScenarioFlowModel:
    """Load SFM from a JSON file"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    sfm = ScenarioFlowModel(
        scenario_name=data["scenarioName"],
        entry_function=data["entryFunction"],
        metadata=data.get("metadata", {})
    )
    
    # Reconstruct nodes
    for node_data in data["nodes"]:
        node = SFMNode(
            id=node_data["id"],
            node_type=SFMNodeType(node_data["nodeType"]),
            label=node_data["label"],
            description=node_data.get("description", ""),
            detail_levels={DetailLevel(dl) for dl in node_data.get("detailLevels", ["high", "medium", "deep"])},
            metadata=node_data.get("metadata", {})
        )
        sfm.add_node(node)
    
    # Reconstruct edges
    for edge_data in data["edges"]:
        edge = SFMEdge(
            from_node=edge_data["from"],
            to_node=edge_data["to"],
            condition=edge_data.get("condition"),
            edge_type=edge_data.get("edgeType", "normal"),
            detail_levels={DetailLevel(dl) for dl in edge_data.get("detailLevels", ["high", "medium", "deep"])}
        )
        sfm.add_edge(edge)
    
    logger.info(f"Loaded SFM from {input_path}")
    return sfm


def build_scenario_flow_model(function_summary: any) -> ScenarioFlowModel:
    """
    Convenience function to build a Scenario Flow Model from a function summary.
    
    Args:
        function_summary: Function summary (can be FunctionSummary wrapper or AggregatedSemantics)
    
    Returns:
        ScenarioFlowModel
    """
    from agent5.bottom_up_aggregator import FunctionSummary, AggregatedSemantics
    
    # Extract the actual aggregated semantics
    if hasattr(function_summary, 'aggregated'):
        # It's a FunctionSummary wrapper
        aggregated = function_summary.aggregated
        entry_function = aggregated.function_name
    elif isinstance(function_summary, AggregatedSemantics):
        # It's directly an AggregatedSemantics
        aggregated = function_summary
        entry_function = aggregated.function_name
    else:
        raise TypeError(f"Unsupported function_summary type: {type(function_summary)}")
    
    builder = SFMBuilder()
    sfm = builder.build_sfm(entry_function, aggregated)
    
    return sfm
