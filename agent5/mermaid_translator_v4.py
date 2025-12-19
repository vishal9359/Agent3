"""
Stage 6: Mermaid Translation (LLM STRICT TRANSLATOR)

Input: Filtered Scenario Flow Model
Output: Mermaid flowchart ONLY

NO logic changes, NO depth changes, NO inference.
LLM is a STRICT translator from SFM to Mermaid syntax.
"""

from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from agent5.sfm_constructor import ScenarioFlowModel, SFMNodeType
from agent5.logging_utils import get_logger

logger = get_logger(__name__)


class MermaidTranslatorV4:
    """
    Strict translator from Scenario Flow Model to Mermaid flowchart.
    
    This is the FINAL stage. The LLM's ONLY job is to produce valid Mermaid
    syntax from the structured SFM. NO semantic changes allowed.
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=8192
        )
    
    def translate(self, filtered_sfm: ScenarioFlowModel) -> str:
        """Translate filtered SFM to Mermaid flowchart"""
        logger.info(f"Stage 6: Translating SFM to Mermaid (detail level: {filtered_sfm.detail_level.value})")
        
        # Build node mapping
        node_map = self._build_node_map(filtered_sfm)
        
        # Generate Mermaid syntax using deterministic rules + LLM for labels
        mermaid_lines = ["flowchart TD"]
        
        # Add nodes
        for node in filtered_sfm.nodes.values():
            mermaid_id = node_map[node.id]
            shape = self._get_mermaid_shape(node.node_type)
            label = self._sanitize_label(node.label)
            
            mermaid_lines.append(f"    {mermaid_id}{shape[0]}{label}{shape[1]}")
        
        # Add edges from nodes
        for node in filtered_sfm.nodes.values():
            from_id = node_map[node.id]
            
            # Add true/false branches for decision nodes
            if node.true_branch and node.true_branch in node_map:
                to_id = node_map[node.true_branch]
                condition = self._sanitize_label(node.condition or "Yes")
                mermaid_lines.append(f"    {from_id} -->|{condition}| {to_id}")
            
            if node.false_branch and node.false_branch in node_map:
                to_id = node_map[node.false_branch]
                condition = self._sanitize_label("No")
                mermaid_lines.append(f"    {from_id} -->|{condition}| {to_id}")
            
            # Add next nodes
            for next_id in node.next_nodes:
                if next_id in node_map:
                    to_id = node_map[next_id]
                    mermaid_lines.append(f"    {from_id} --> {to_id}")
        
        mermaid_code = "\n".join(mermaid_lines)
        
        # Optionally use LLM to improve label clarity (without changing semantics)
        if len(filtered_sfm.nodes) > 2:  # Skip for trivial diagrams
            mermaid_code = self._llm_polish_labels(mermaid_code, filtered_sfm)
        
        logger.info("Mermaid translation complete")
        return mermaid_code
    
    def _build_node_map(self, filtered_sfm: ScenarioFlowModel) -> Dict[str, str]:
        """Build mapping from node IDs to Mermaid IDs"""
        node_map = {}
        nodes_list = list(filtered_sfm.nodes.values())
        for i, node in enumerate(nodes_list):
            if node.node_type == SFMNodeType.START:
                node_map[node.id] = "START"
            elif node.node_type == SFMNodeType.END:
                node_map[node.id] = "END"
            else:
                node_map[node.id] = f"N{i}"
        return node_map
    
    def _get_mermaid_shape(self, node_type: SFMNodeType) -> tuple[str, str]:
        """Get Mermaid shape syntax for node type"""
        if node_type == SFMNodeType.START:
            return ("([", "])")
        elif node_type == SFMNodeType.END:
            return ("([", "])")
        elif node_type == SFMNodeType.DECISION:
            return ("{", "}")
        elif node_type == SFMNodeType.ERROR:
            return ("{{", "}}")
        elif node_type == SFMNodeType.PROCESS:
            return ("[", "]")
        elif node_type == SFMNodeType.SUBPROCESS:
            return ("[[", "]]")
        else:
            return ("[", "]")
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize label for Mermaid syntax"""
        # Remove or escape special characters
        sanitized = label.replace('"', "'")
        sanitized = sanitized.replace("\n", " ")
        sanitized = sanitized.replace("|", "/")
        # Truncate if too long
        if len(sanitized) > 100:
            sanitized = sanitized[:97] + "..."
        return sanitized
    
    def _llm_polish_labels(self, mermaid_code: str, filtered_sfm: ScenarioFlowModel) -> str:
        """Use LLM to polish labels for clarity (optional enhancement)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Mermaid flowchart syntax expert. Your ONLY task is to improve label clarity in the provided Mermaid code without changing:
- The structure
- The logic
- The node connections
- The node types

You may ONLY:
- Shorten overly verbose labels
- Make labels more concise
- Ensure proper Mermaid syntax

Output ONLY the improved Mermaid code, no explanations."""),
            ("human", """Scenario: {scenario_name}
Detail Level: {detail_level}

Current Mermaid code:
{mermaid_code}

Output improved Mermaid code:""")
        ])
        
        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "scenario_name": filtered_sfm.original_sfm.scenario_name,
                "detail_level": filtered_sfm.detail_level.value,
                "mermaid_code": mermaid_code
            })
            
            # Extract mermaid code from result
            content = result.content if hasattr(result, 'content') else str(result)
            
            # Validate that it starts with 'flowchart'
            if "flowchart" in content:
                # Extract just the mermaid code
                lines = content.split("\n")
                mermaid_lines = []
                in_code_block = False
                for line in lines:
                    if "```" in line:
                        in_code_block = not in_code_block
                        continue
                    if "flowchart" in line or in_code_block or line.strip().startswith("    "):
                        mermaid_lines.append(line)
                
                if mermaid_lines:
                    return "\n".join(mermaid_lines)
            
            # If LLM output is invalid, return original
            logger.warning("LLM polish produced invalid output, using original")
            return mermaid_code
        
        except Exception as e:
            logger.warning(f"LLM polish failed: {e}, using original")
            return mermaid_code



