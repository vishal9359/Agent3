"""
Stage 6: Mermaid Translation (LLM STRICT TRANSLATOR)

This module translates the filtered Scenario Flow Model into Mermaid syntax.
The LLM is used STRICTLY as a translator, with NO logic changes allowed.

Rules:
- Input: Scenario Flow Model (JSON)
- Output: Mermaid only
- No logic changes
- No depth changes
- No inference
"""

import json
import logging
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .sfm_constructor import ScenarioFlowModel, SFMNode, SFMNodeType

logger = logging.getLogger(__name__)


class MermaidTranslator:
    """
    Translates SFM to Mermaid flowchart syntax.
    
    The LLM is used ONLY as a syntax translator.
    All logic and structure come from the SFM.
    """
    
    def __init__(self, model_name: str = "llama3.2:3b", use_llm: bool = True):
        """
        Initialize the Mermaid translator
        
        Args:
            model_name: Ollama model for LLM translation
            use_llm: If False, use deterministic rule-based translation
        """
        self.use_llm = use_llm
        
        if use_llm:
            self.llm = ChatOllama(
                model=model_name,
                temperature=0.0  # Deterministic
            )
            
            self.translation_prompt = ChatPromptTemplate.from_template(
                """You are a Mermaid flowchart syntax generator. Your ONLY task is to translate the provided Scenario Flow Model into valid Mermaid syntax.

DO NOT:
- Change the logic or structure
- Add or remove nodes
- Modify conditions or descriptions
- Infer anything not in the SFM

DO:
- Use proper Mermaid flowchart syntax
- Use appropriate node shapes for node types
- Create clear, readable labels
- Preserve all edges and branches

Scenario Flow Model (JSON):
{sfm_json}

Generate ONLY the Mermaid flowchart code. Start with "flowchart TD" and use proper Mermaid syntax.

Node types should use these shapes:
- START: ([Start])
- END: ([End])
- PROCESS: [Process]
- DECISION: {{Decision?}}
- SUBPROCESS: [[Subprocess]]
- ERROR: [/Error\\]

Use --> for edges and |label| for edge labels.
For decisions, use |Yes| and |No| labels on branches."""
            )
        
        logger.info(f"MermaidTranslator initialized (LLM: {use_llm})")
    
    def translate(self, sfm: ScenarioFlowModel) -> str:
        """
        Translate SFM to Mermaid flowchart.
        
        Args:
            sfm: Scenario Flow Model to translate
            
        Returns:
            Mermaid flowchart syntax as string
        """
        logger.info(f"Translating SFM '{sfm.scenario_name}' to Mermaid")
        
        if self.use_llm:
            return self._translate_with_llm(sfm)
        else:
            return self._translate_deterministic(sfm)
    
    def _translate_with_llm(self, sfm: ScenarioFlowModel) -> str:
        """Translate using LLM"""
        # Convert SFM to JSON
        sfm_json = json.dumps(sfm.to_dict(), indent=2)
        
        # Call LLM
        messages = self.translation_prompt.format_messages(sfm_json=sfm_json)
        
        try:
            response = self.llm.invoke(messages)
            mermaid_code = response.content.strip()
            
            # Validate it starts with flowchart declaration
            if not mermaid_code.startswith('flowchart'):
                mermaid_code = 'flowchart TD\n' + mermaid_code
            
            logger.info(f"LLM translation successful ({len(mermaid_code)} chars)")
            return mermaid_code
            
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            logger.info("Falling back to deterministic translation")
            return self._translate_deterministic(sfm)
    
    def _translate_deterministic(self, sfm: ScenarioFlowModel) -> str:
        """Deterministic rule-based translation"""
        lines = ['flowchart TD']
        
        # Track processed nodes to avoid duplicates
        processed = set()
        
        # Process nodes in order
        for node_id, node in sfm.nodes.items():
            if node_id in processed:
                continue
            
            # Generate node definition
            node_def = self._generate_node_definition(node)
            lines.append(f"    {node_def}")
            processed.add(node_id)
            
            # Generate edges
            edge_lines = self._generate_edges(node, sfm.nodes)
            lines.extend([f"    {line}" for line in edge_lines])
        
        mermaid_code = '\n'.join(lines)
        logger.info(f"Deterministic translation successful ({len(mermaid_code)} chars)")
        return mermaid_code
    
    def _generate_node_definition(self, node: SFMNode) -> str:
        """Generate Mermaid node definition"""
        node_id = self._sanitize_id(node.id)
        label = self._sanitize_label(node.label)
        
        # Choose shape based on node type
        if node.node_type == SFMNodeType.START:
            return f"{node_id}([{label}])"
        elif node.node_type == SFMNodeType.END:
            return f"{node_id}([{label}])"
        elif node.node_type == SFMNodeType.DECISION:
            return f"{node_id}{{{{{label}?}}}}"
        elif node.node_type == SFMNodeType.SUBPROCESS:
            return f"{node_id}[[{label}]]"
        elif node.node_type == SFMNodeType.ERROR:
            return f"{node_id}[/{label}\\]"
        else:  # PROCESS
            return f"{node_id}[{label}]"
    
    def _generate_edges(self, node: SFMNode, all_nodes: Dict[str, SFMNode]) -> List[str]:
        """Generate edges from a node"""
        edges = []
        node_id = self._sanitize_id(node.id)
        
        # Handle decision branches
        if node.node_type == SFMNodeType.DECISION:
            if node.true_branch:
                true_id = self._sanitize_id(node.true_branch)
                edges.append(f"{node_id} -->|Yes| {true_id}")
            
            if node.false_branch:
                false_id = self._sanitize_id(node.false_branch)
                edges.append(f"{node_id} -->|No| {false_id}")
        
        # Handle regular next nodes
        for next_id in node.next_nodes:
            # Skip if already handled by decision branches
            if next_id in [node.true_branch, node.false_branch]:
                continue
            
            next_id_clean = self._sanitize_id(next_id)
            edges.append(f"{node_id} --> {next_id_clean}")
        
        return edges
    
    def _sanitize_id(self, node_id: str) -> str:
        """Sanitize node ID for Mermaid"""
        # Replace invalid characters
        return node_id.replace('-', '_').replace(' ', '_')
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize label text for Mermaid"""
        # Escape special characters
        label = label.replace('"', "'")
        label = label.replace('[', '(')
        label = label.replace(']', ')')
        label = label.replace('{', '(')
        label = label.replace('}', ')')
        
        # Limit length
        if len(label) > 100:
            label = label[:97] + "..."
        
        return label
