"""
Stage 5: Detail-Level Filtering

Apply rule-based filtering to the Scenario Flow Model based on detail level.
Filtering happens AFTER aggregation.

Detail Levels:
- HIGH: Business-level steps only
- MEDIUM (default): All decisions + validations + state changes
- DEEP: Expanded critical sub-operations affecting control/state

NEVER EXPAND:
- Logging
- Metrics
- Utility helpers
- Memory alloc wrappers
- Serialization helpers
"""

from typing import List, Set
from copy import deepcopy

from agent5.sfm_builder import ScenarioFlowModel, SFMNode, SFMEdge, DetailLevel
from agent5.logging_utils import get_logger

logger = get_logger(__name__)


class DetailLevelFilter:
    """
    Filters Scenario Flow Model based on requested detail level.
    This is Stage 5: Detail-Level Filtering.
    """
    
    def __init__(self):
        pass
    
    def filter_sfm(self, sfm: ScenarioFlowModel, detail_level: DetailLevel) -> ScenarioFlowModel:
        """
        Filter SFM to include only nodes and edges appropriate for the requested detail level.
        
        Args:
            sfm: Original Scenario Flow Model
            detail_level: Requested detail level (HIGH, MEDIUM, or DEEP)
            
        Returns:
            Filtered ScenarioFlowModel
        """
        logger.info(f"Filtering SFM for detail level: {detail_level.value}")
        
        # Create a deep copy to avoid modifying the original
        filtered_sfm = ScenarioFlowModel(
            scenario_name=sfm.scenario_name,
            entry_function=sfm.entry_function,
            metadata=sfm.metadata.copy()
        )
        
        # Filter nodes
        included_node_ids = set()
        for node in sfm.nodes:
            if self._should_include_node(node, detail_level):
                filtered_sfm.add_node(deepcopy(node))
                included_node_ids.add(node.id)
        
        # Filter edges and reconnect
        filtered_edges = []
        for edge in sfm.edges:
            if self._should_include_edge(edge, detail_level, included_node_ids):
                filtered_edges.append(deepcopy(edge))
        
        # Reconnect edges - if an edge's source or target is excluded, reconnect around it
        reconnected_edges = self._reconnect_edges(filtered_edges, included_node_ids, sfm)
        
        for edge in reconnected_edges:
            filtered_sfm.add_edge(edge)
        
        logger.info(f"Filtered SFM: {len(filtered_sfm.nodes)} nodes, {len(filtered_sfm.edges)} edges "
                   f"(original: {len(sfm.nodes)} nodes, {len(sfm.edges)} edges)")
        
        return filtered_sfm
    
    def _should_include_node(self, node: SFMNode, detail_level: DetailLevel) -> bool:
        """Determine if a node should be included at the given detail level"""
        # Always include start and end nodes
        if node.node_type.value in ["start", "end"]:
            return True
        
        # Check if node is tagged for this detail level
        if detail_level not in node.detail_levels:
            return False
        
        # Never include utility operations (logging, metrics, etc.)
        if self._is_utility_operation(node):
            return False
        
        return True
    
    def _should_include_edge(self, edge: SFMEdge, detail_level: DetailLevel, 
                            included_node_ids: Set[str]) -> bool:
        """Determine if an edge should be included"""
        # Check if both nodes are included
        if edge.from_node not in included_node_ids or edge.to_node not in included_node_ids:
            return False
        
        # Check if edge is tagged for this detail level
        if detail_level not in edge.detail_levels:
            return False
        
        return True
    
    def _reconnect_edges(self, edges: List[SFMEdge], included_node_ids: Set[str], 
                        original_sfm: ScenarioFlowModel) -> List[SFMEdge]:
        """
        Reconnect edges when intermediate nodes are excluded.
        Creates direct paths bypassing excluded nodes.
        """
        # Build adjacency map from original SFM
        adjacency = {}
        for edge in original_sfm.edges:
            if edge.from_node not in adjacency:
                adjacency[edge.from_node] = []
            adjacency[edge.from_node].append(edge)
        
        # For each excluded node, find paths around it
        reconnected = []
        for edge in edges:
            reconnected.append(edge)
        
        # Find nodes that need reconnection (have successors/predecessors excluded)
        for edge in original_sfm.edges:
            if edge.from_node in included_node_ids and edge.to_node not in included_node_ids:
                # Find the next included node reachable from edge.to_node
                next_included = self._find_next_included_node(edge.to_node, adjacency, included_node_ids)
                if next_included:
                    # Create a direct edge
                    new_edge = SFMEdge(
                        from_node=edge.from_node,
                        to_node=next_included,
                        condition=edge.condition,
                        edge_type=edge.edge_type,
                        detail_levels=edge.detail_levels
                    )
                    # Check if this edge doesn't already exist
                    if not any(e.from_node == new_edge.from_node and e.to_node == new_edge.to_node 
                             for e in reconnected):
                        reconnected.append(new_edge)
        
        return reconnected
    
    def _find_next_included_node(self, start_node: str, adjacency: dict, 
                                included_nodes: Set[str], visited: Set[str] = None) -> str:
        """Find the next included node reachable from start_node using BFS"""
        if visited is None:
            visited = set()
        
        if start_node in visited:
            return None
        
        visited.add(start_node)
        
        if start_node in included_nodes:
            return start_node
        
        # BFS to find next included node
        if start_node in adjacency:
            for edge in adjacency[start_node]:
                result = self._find_next_included_node(edge.to_node, adjacency, included_nodes, visited)
                if result:
                    return result
        
        return None
    
    def _is_utility_operation(self, node: SFMNode) -> bool:
        """Determine if a node represents a utility operation that should never be expanded"""
        label_lower = node.label.lower()
        description_lower = node.description.lower()
        
        # Keywords for utility operations
        utility_keywords = [
            'log', 'logging', 'logger',
            'metric', 'metrics', 'telemetry',
            'trace', 'debug', 'info',
            'malloc', 'free', 'alloc', 'dealloc',
            'serialize', 'deserialize', 'marshal', 'unmarshal',
            'format', 'print', 'cout', 'cerr',
            'helper', 'util', 'utility'
        ]
        
        for keyword in utility_keywords:
            if keyword in label_lower or keyword in description_lower:
                return True
        
        return False


def apply_detail_level(sfm: ScenarioFlowModel, detail_level: str) -> ScenarioFlowModel:
    """
    Apply detail level filtering to an SFM.
    
    Args:
        sfm: Original Scenario Flow Model
        detail_level: Detail level string ("high", "medium", or "deep")
        
    Returns:
        Filtered ScenarioFlowModel
    """
    # Convert string to enum
    detail_enum = DetailLevel.HIGH
    if detail_level.lower() == "medium":
        detail_enum = DetailLevel.MEDIUM
    elif detail_level.lower() == "deep":
        detail_enum = DetailLevel.DEEP
    
    filter_engine = DetailLevelFilter()
    return filter_engine.filter_sfm(sfm, detail_enum)
