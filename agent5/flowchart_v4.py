"""
Stage 6: Mermaid Translation (LLM STRICT TRANSLATOR)

Translate Scenario Flow Model to Mermaid flowchart syntax.

Input: Scenario Flow Model (filtered by detail level)
Output: Mermaid flowchart ONLY

Rules:
- No logic changes
- No depth changes
- No inference
- Strict translation ONLY
"""

from typing import Dict, List, Optional
import json

from agent5.sfm_builder import ScenarioFlowModel, SFMNode, SFMNodeType
from agent5.logging_utils import get_logger
from agent5.ollama_compat import get_ollama_llm

logger = get_logger(__name__)


class MermaidTranslator:
    """
    Translates Scenario Flow Model to Mermaid flowchart.
    This is Stage 6: Mermaid Translation.
    
    The LLM is used ONLY as a strict translator, not for logic inference.
    """
    
    def __init__(self, llm_model: str = "llama2:7b"):
        """
        Initialize the Mermaid translator.
        
        Args:
            llm_model: Name of the Ollama LLM model to use
        """
        self.llm = get_ollama_llm(llm_model)
    
    def translate_to_mermaid(self, sfm: ScenarioFlowModel, use_llm: bool = True) -> str:
        """
        Translate SFM to Mermaid flowchart.
        
        Args:
            sfm: Scenario Flow Model to translate
            use_llm: Whether to use LLM for translation (if False, use deterministic translation)
            
        Returns:
            Mermaid flowchart string
        """
        logger.info(f"Translating SFM to Mermaid (use_llm={use_llm})")
        
        # Validate SFM before translation
        if not self._validate_sfm(sfm):
            logger.error("SFM validation failed. Cannot generate flowchart.")
            raise ValueError("Invalid SFM: must have exactly one start node and at least one end node")
        
        if use_llm:
            try:
                mermaid = self._llm_translate(sfm)
                
                # Validate generated Mermaid
                if self._validate_mermaid(mermaid):
                    return mermaid
                else:
                    logger.warning("LLM-generated Mermaid is invalid. Falling back to deterministic.")
                    return self._deterministic_translate(sfm)
            except Exception as e:
                logger.error(f"LLM translation failed: {e}. Falling back to deterministic.")
                return self._deterministic_translate(sfm)
        else:
            return self._deterministic_translate(sfm)
    
    def _validate_sfm(self, sfm: ScenarioFlowModel) -> bool:
        """
        Validate SFM before translation.
        
        Rules:
        - Exactly one START node
        - At least one END node
        - All edges reference existing nodes
        """
        start_nodes = [n for n in sfm.nodes if n.node_type == SFMNodeType.START]
        end_nodes = [n for n in sfm.nodes if n.node_type == SFMNodeType.END]
        
        if len(start_nodes) != 1:
            logger.error(f"SFM must have exactly one START node, found {len(start_nodes)}")
            return False
        
        if len(end_nodes) < 1:
            logger.error("SFM must have at least one END node")
            return False
        
        # Check that all edges reference existing nodes
        node_ids = {n.id for n in sfm.nodes}
        for edge in sfm.edges:
            if edge.from_node not in node_ids:
                logger.error(f"Edge references non-existent node: {edge.from_node}")
                return False
            if edge.to_node not in node_ids:
                logger.error(f"Edge references non-existent node: {edge.to_node}")
                return False
        
        return True
    
    def _llm_translate(self, sfm: ScenarioFlowModel) -> str:
        """Use LLM to translate SFM to Mermaid"""
        prompt = self._build_translation_prompt(sfm)
        
        response = self.llm.invoke(prompt)
        mermaid = response.content if hasattr(response, 'content') else str(response)
        
        # Extract Mermaid code from response
        mermaid = self._extract_mermaid_code(mermaid)
        
        return mermaid
    
    def _build_translation_prompt(self, sfm: ScenarioFlowModel) -> str:
        """Build prompt for LLM translation"""
        prompt = f"""You are a Mermaid flowchart translator. Your ONLY task is to convert the provided Scenario Flow Model (SFM) into valid Mermaid flowchart syntax.

SCENARIO: {sfm.scenario_name}
ENTRY FUNCTION: {sfm.entry_function}

NODES:
"""
        for node in sfm.nodes:
            prompt += f"- {node.id} [{node.node_type.value}]: {node.label}\n"
        
        prompt += "\nEDGES:\n"
        for edge in sfm.edges:
            condition_str = f" ({edge.condition})" if edge.condition else ""
            prompt += f"- {edge.from_node} -> {edge.to_node}{condition_str} [type: {edge.edge_type}]\n"
        
        prompt += """
TASK:
Generate a valid Mermaid flowchart that represents this scenario flow.

RULES:
1. Use flowchart syntax: flowchart TD
2. Map node types to Mermaid shapes:
   - START: rounded box ([...])
   - END: rounded box ([...])
   - DECISION: diamond {...}
   - VALIDATION: trapezoid [.../]
   - STATE_CHANGE: rectangle [...]
   - OPERATION: rectangle [...]
   - ERROR_EXIT: rounded box ([...])
3. Use edge types for styling:
   - "true" branches: use -->|Yes| or -->|True|
   - "false" branches: use -->|No| or -->|False|
   - "error" edges: use -.->
4. Do NOT add any nodes or edges not in the SFM
5. Do NOT change the logic or structure
6. Output ONLY the Mermaid code, no explanations

MERMAID CODE:
```mermaid
"""
        
        return prompt
    
    def _extract_mermaid_code(self, response: str) -> str:
        """Extract Mermaid code from LLM response"""
        # Look for code between ```mermaid and ```
        import re
        
        match = re.search(r'```mermaid\s+(.*?)\s+```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for code between ``` and ```
        match = re.search(r'```\s+(.*?)\s+```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no code blocks found, return the whole response
        return response.strip()
    
    def _deterministic_translate(self, sfm: ScenarioFlowModel) -> str:
        """
        Deterministic translation from SFM to Mermaid.
        This is the fallback when LLM is not available or fails.
        """
        logger.info("Using deterministic Mermaid translation")
        
        mermaid_lines = ["flowchart TD"]
        
        # Generate node definitions
        for node in sfm.nodes:
            shape = self._get_mermaid_shape(node)
            mermaid_lines.append(f"    {node.id}{shape}")
        
        # Generate edges
        for edge in sfm.edges:
            arrow = self._get_mermaid_arrow(edge.edge_type)
            label = ""
            if edge.condition:
                label = f"|{edge.condition}|"
            elif edge.edge_type == "true":
                label = "|Yes|"
            elif edge.edge_type == "false":
                label = "|No|"
            
            mermaid_lines.append(f"    {edge.from_node} {arrow}{label} {edge.to_node}")
        
        return "\n".join(mermaid_lines)
    
    def _get_mermaid_shape(self, node: SFMNode) -> str:
        """Get Mermaid shape syntax for a node"""
        label = node.label.replace('"', "'")  # Escape quotes
        
        if node.node_type == SFMNodeType.START:
            return f"([{label}])"
        elif node.node_type == SFMNodeType.END:
            return f"([{label}])"
        elif node.node_type == SFMNodeType.DECISION:
            return f"{{{{{label}}}}}"
        elif node.node_type == SFMNodeType.VALIDATION:
            return f"[/{label}/]"
        elif node.node_type == SFMNodeType.STATE_CHANGE:
            return f"[{label}]"
        elif node.node_type == SFMNodeType.OPERATION:
            return f"[{label}]"
        elif node.node_type == SFMNodeType.ERROR_EXIT:
            return f"([{label}])"
        else:
            return f"[{label}]"
    
    def _get_mermaid_arrow(self, edge_type: str) -> str:
        """Get Mermaid arrow syntax for an edge type"""
        if edge_type == "error":
            return "-.->"]
        else:
            return "-->"
    
    def _validate_mermaid(self, mermaid: str) -> bool:
        """Validate that generated Mermaid is well-formed"""
        # Basic checks
        if not mermaid or len(mermaid) < 10:
            return False
        
        # Must start with flowchart directive
        if not mermaid.strip().startswith("flowchart"):
            return False
        
        # Must have at least one arrow
        if "-->" not in mermaid and "-.->": not in mermaid:
            return False
        
        return True


def generate_flowchart_from_sfm(sfm: ScenarioFlowModel, output_path: Optional[str] = None,
                               use_llm: bool = True, llm_model: str = "llama2:7b") -> str:
    """
    Generate Mermaid flowchart from Scenario Flow Model.
    
    Args:
        sfm: Scenario Flow Model to translate
        output_path: Optional path to save the Mermaid file
        use_llm: Whether to use LLM for translation
        llm_model: Name of the LLM model to use
        
    Returns:
        Mermaid flowchart string
    """
    translator = MermaidTranslator(llm_model=llm_model)
    mermaid = translator.translate_to_mermaid(sfm, use_llm=use_llm)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(mermaid)
        logger.info(f"Saved Mermaid flowchart to {output_path}")
    
    return mermaid

