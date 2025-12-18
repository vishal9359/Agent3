"""
Mermaid Generator - Stage 6

This module translates Scenario Flow Model (SFM) into Mermaid flowchart syntax.
The LLM acts as a STRICT TRANSLATOR ONLY - no logic changes, no depth changes, no inference.

This is a thin wrapper around the existing Mermaid translation logic.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sfm_constructor import ScenarioFlowModel, StepType

logger = logging.getLogger(__name__)


@dataclass
class MermaidNode:
    """A node in the Mermaid flowchart."""
    id: str
    label: str
    shape: str = "rectangle"  # rectangle, diamond, rounded, stadium


@dataclass
class MermaidEdge:
    """An edge in the Mermaid flowchart."""
    from_id: str
    to_id: str
    label: str = ""


@dataclass
class MermaidFlowchart:
    """Complete Mermaid flowchart representation."""
    nodes: List[MermaidNode] = field(default_factory=list)
    edges: List[MermaidEdge] = field(default_factory=list)
    title: str = ""
    
    def to_mermaid(self) -> str:
        """Convert to Mermaid syntax."""
        lines = ["flowchart TD"]
        
        if self.title:
            lines.append(f"    title[{self.title}]")
            lines.append("")
        
        # Add nodes
        for node in self.nodes:
            shape_open, shape_close = self._get_shape_delimiters(node.shape)
            lines.append(f"    {node.id}{shape_open}{node.label}{shape_close}")
        
        lines.append("")
        
        # Add edges
        for edge in self.edges:
            if edge.label:
                lines.append(f"    {edge.from_id} -->|{edge.label}| {edge.to_id}")
            else:
                lines.append(f"    {edge.from_id} --> {edge.to_id}")
        
        return "\n".join(lines)
    
    def _get_shape_delimiters(self, shape: str) -> tuple:
        """Get Mermaid shape delimiters."""
        shapes = {
            "rectangle": ("[", "]"),
            "rounded": ("(", ")"),
            "stadium": ("([", "])"),
            "diamond": ("{", "}"),
            "circle": ("((", "))"),
        }
        return shapes.get(shape, ("[", "]"))


class MermaidGenerator:
    """
    Generates Mermaid flowcharts from Scenario Flow Models.
    
    This is a deterministic translator - no logic inference.
    """

    def __init__(
        self,
        chat_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            chat_model: Optional chat model for LLM-assisted formatting
            ollama_base_url: Ollama server URL
        """
        self.chat_model = chat_model
        self.ollama_base_url = ollama_base_url
    
    def generate_from_sfm(self, sfm: ScenarioFlowModel) -> MermaidFlowchart:
        """
        Generate Mermaid flowchart from SFM.
        
        Args:
            sfm: Scenario Flow Model
        
        Returns:
            MermaidFlowchart
        """
        logger.info(f"Generating Mermaid flowchart for scenario: {sfm.scenario_name}")
        
        flowchart = MermaidFlowchart(title=sfm.scenario_name)
        
        # Convert steps to nodes
        for step_id, step in sfm.steps.items():
            node = MermaidNode(
                id=step_id,
                label=step.label,
                shape=self._step_type_to_shape(step.step_type),
            )
            flowchart.nodes.append(node)
        
        # Convert transitions to edges
        for step_id, step in sfm.steps.items():
            # Normal flow
            for next_id in step.next_steps:
                edge = MermaidEdge(from_id=step_id, to_id=next_id)
                flowchart.edges.append(edge)
            
            # Success path
            if step.on_success:
                edge = MermaidEdge(
                    from_id=step_id,
                    to_id=step.on_success,
                    label="Yes" if step.step_type == StepType.DECISION else "Success",
                )
                flowchart.edges.append(edge)
            
            # Failure path
            if step.on_fail:
                edge = MermaidEdge(
                    from_id=step_id,
                    to_id=step.on_fail,
                    label="No" if step.step_type == StepType.DECISION else "Fail",
                )
                flowchart.edges.append(edge)
        
        logger.info(f"  ✓ Generated {len(flowchart.nodes)} nodes and {len(flowchart.edges)} edges")
        
        return flowchart
    
    def _step_type_to_shape(self, step_type: StepType) -> str:
        """Map step type to Mermaid shape."""
        shape_map = {
            StepType.START: "stadium",
            StepType.END: "stadium",
            StepType.PROCESS: "rectangle",
            StepType.DECISION: "diamond",
            StepType.STATE_CHANGE: "rectangle",
            StepType.VALIDATION: "diamond",
            StepType.ERROR_EXIT: "rounded",
        }
        return shape_map.get(step_type, "rectangle")

