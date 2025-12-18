"""
Stage 5: Detail-Level Filtering (RULE-BASED)

This module filters the Scenario Flow Model based on the requested detail level.
Filtering happens AFTER aggregation, not during it.

Detail levels:
- HIGH: Business-level steps only
- MEDIUM: All decisions + validations + state changes
- DEEP: Expanded critical sub-operations affecting control/state

NEVER EXPAND:
- Logging
- Metrics
- Utility helpers
- Memory alloc wrappers
- Serialization helpers
"""

import logging
from typing import Dict, List

from .sfm_constructor import ScenarioFlowModel, SFMNode, DetailLevel, SFMNodeType

logger = logging.getLogger(__name__)


class SFMFilter:
    """
    Filters a Scenario Flow Model based on detail level.
    This is a purely rule-based, deterministic process.
    """
    
    def __init__(self):
        logger.info("SFMFilter initialized")
    
    def filter(
        self,
        sfm: ScenarioFlowModel,
        detail_level: DetailLevel
    ) -> ScenarioFlowModel:
        """
        Filter the SFM to include only nodes appropriate for the detail level.
        
        Args:
            sfm: Original ScenarioFlowModel
            detail_level: Requested detail level
            
        Returns:
            Filtered ScenarioFlowModel
        """
        logger.info(f"Filtering SFM to {detail_level.value} detail level")
        
        # Determine which nodes to include
        included_nodes = self._filter_nodes(sfm, detail_level)
        
        # Rebuild edges to skip filtered nodes
        filtered_nodes = self._rebuild_edges(sfm, included_nodes)
        
        # Find new end nodes
        end_nodes = self._find_end_nodes(filtered_nodes)
        
        # Create filtered SFM
        filtered_sfm = ScenarioFlowModel(
            scenario_name=sfm.scenario_name,
            entry_function=sfm.entry_function,
            start_node_id=sfm.start_node_id,
            end_node_ids=end_nodes,
            nodes=filtered_nodes
        )
        
        logger.info(
            f"Filtered SFM: {len(sfm.nodes)} -> {len(filtered_nodes)} nodes "
            f"at {detail_level.value} level"
        )
        
        return filtered_sfm
    
    def _filter_nodes(
        self,
        sfm: ScenarioFlowModel,
        detail_level: DetailLevel
    ) -> List[str]:
        """
        Determine which nodes should be included at the given detail level.
        
        Returns:
            List of node IDs to include
        """
        included = []
        
        for node_id, node in sfm.nodes.items():
            if self._should_include_node(node, detail_level):
                included.append(node_id)
        
        return included
    
    def _should_include_node(
        self,
        node: SFMNode,
        detail_level: DetailLevel
    ) -> bool:
        """Determine if a node should be included at the given detail level"""
        
        # Always include START and END nodes
        if node.node_type in [SFMNodeType.START, SFMNodeType.END]:
            return True
        
        # Check minimum detail level
        if detail_level == DetailLevel.HIGH:
            return node.min_detail_level == DetailLevel.HIGH
        elif detail_level == DetailLevel.MEDIUM:
            return node.min_detail_level in [DetailLevel.HIGH, DetailLevel.MEDIUM]
        else:  # DEEP
            return True  # Include all nodes
    
    def _rebuild_edges(
        self,
        sfm: ScenarioFlowModel,
        included_nodes: List[str]
    ) -> Dict[str, SFMNode]:
        """
        Rebuild edges to connect included nodes, skipping filtered ones.
        
        Returns:
            Dictionary of filtered nodes with updated edges
        """
        included_set = set(included_nodes)
        filtered_nodes = {}
        
        # Copy included nodes
        for node_id in included_nodes:
            node = sfm.nodes[node_id]
            # Create a copy to avoid modifying original
            filtered_node = SFMNode(
                id=node.id,
                node_type=node.node_type,
                label=node.label,
                description=node.description,
                condition=node.condition,
                min_detail_level=node.min_detail_level,
                source_function=node.source_function,
                is_critical=node.is_critical,
                is_error_path=node.is_error_path
            )
            filtered_nodes[node_id] = filtered_node
        
        # Update edges to skip filtered nodes
        for node_id, node in filtered_nodes.items():
            original = sfm.nodes[node_id]
            
            # Update next_nodes
            node.next_nodes = self._find_next_included_nodes(
                original.next_nodes,
                included_set,
                sfm.nodes
            )
            
            # Update decision branches
            if original.true_branch:
                next_true = self._find_next_included_node(
                    original.true_branch,
                    included_set,
                    sfm.nodes
                )
                if next_true:
                    node.true_branch = next_true
                    # Also add to next_nodes for graph traversal
                    if next_true not in node.next_nodes:
                        node.next_nodes.append(next_true)
            
            if original.false_branch:
                next_false = self._find_next_included_node(
                    original.false_branch,
                    included_set,
                    sfm.nodes
                )
                if next_false:
                    node.false_branch = next_false
                    # Also add to next_nodes for graph traversal
                    if next_false not in node.next_nodes:
                        node.next_nodes.append(next_false)
        
        return filtered_nodes
    
    def _find_next_included_nodes(
        self,
        successors: List[str],
        included: set,
        all_nodes: Dict[str, SFMNode]
    ) -> List[str]:
        """Find the next included nodes, skipping filtered ones"""
        result = []
        
        for succ_id in successors:
            next_node = self._find_next_included_node(succ_id, included, all_nodes)
            if next_node and next_node not in result:
                result.append(next_node)
        
        return result
    
    def _find_next_included_node(
        self,
        start_id: str,
        included: set,
        all_nodes: Dict[str, SFMNode]
    ) -> Optional[str]:
        """
        Find the next included node by traversing the graph.
        Uses BFS to handle multiple paths.
        """
        if start_id in included:
            return start_id
        
        visited = set()
        queue = [start_id]
        
        while queue:
            node_id = queue.pop(0)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            if node_id in included:
                return node_id
            
            # Continue searching through successors
            node = all_nodes.get(node_id)
            if node:
                queue.extend(node.next_nodes)
                if node.true_branch:
                    queue.append(node.true_branch)
                if node.false_branch:
                    queue.append(node.false_branch)
        
        return None
    
    def _find_end_nodes(self, nodes: Dict[str, SFMNode]) -> List[str]:
        """Find all end nodes (nodes with no successors or END type)"""
        end_nodes = []
        
        for node_id, node in nodes.items():
            if node.node_type == SFMNodeType.END:
                end_nodes.append(node_id)
            elif (node.node_type == SFMNodeType.ERROR and 
                  len(node.next_nodes) == 0):
                end_nodes.append(node_id)
        
        return end_nodes


from typing import Optional

