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
            
            # Clean up LLM output
            mermaid_code = self._clean_llm_output(mermaid_code)
            
            logger.info(f"LLM translation successful ({len(mermaid_code)} chars)")
            return mermaid_code
            
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            logger.info("Falling back to deterministic translation")
            return self._translate_deterministic(sfm)
    
    def _clean_llm_output(self, mermaid_code: str) -> str:
        """Clean and validate LLM-generated Mermaid code"""
        lines = mermaid_code.split('\n')
        cleaned_lines = []
        in_code_block = False
        found_flowchart = False
        
        for line in lines:
            # Remove code block markers
            if '```' in line:
                if 'mermaid' in line.lower():
                    in_code_block = True
                elif in_code_block:
                    in_code_block = False
                continue
            
            # Skip empty lines at start
            if not cleaned_lines and not line.strip():
                continue
            
            # Check for flowchart declaration
            if 'flowchart' in line.lower() and 'td' in line.lower():
                if found_flowchart:
                    # Skip duplicate flowchart declarations
                    continue
                found_flowchart = True
                # Ensure proper format
                if not line.strip().startswith('flowchart'):
                    line = 'flowchart TD'
                cleaned_lines.append(line)
                continue
            
            # Add lines that look like Mermaid syntax
            line_stripped = line.strip()
            if (in_code_block or 
                found_flowchart or 
                line_stripped.startswith(('START', 'END', 'PROCESS', 'N', 'D', 'V', 'E')) or
                '-->' in line or
                '([[' in line or '[[' in line or '{{' in line or '[' in line):
                # Fix common syntax errors
                # Fix PROCESS node without brackets: "PROCESS text" -> "PROCESS[text]"
                if line_stripped.startswith('PROCESS ') and '[' not in line_stripped and '-->' not in line_stripped:
                    parts = line_stripped.split(' ', 1)
                    if len(parts) == 2:
                        # Preserve indentation
                        indent = len(line) - len(line.lstrip())
                        line_stripped = f"{parts[0]}[{parts[1]}]"
                        line = ' ' * indent + line_stripped
                
                # Fix any node ID followed by text without brackets
                # Pattern: "NODEID text" -> "NODEID[text]"
                if '-->' not in line_stripped and '[' not in line_stripped and '{' not in line_stripped:
                    # Check if it looks like a node definition (starts with alphanumeric, has space, then text)
                    import re
                    match = re.match(r'^([A-Z0-9_]+)\s+(.+)$', line_stripped)
                    if match:
                        node_id, label = match.groups()
                        # Only fix if it looks like a valid node ID (uppercase or starts with N)
                        if node_id.isupper() or node_id.startswith('N'):
                            indent = len(line) - len(line.lstrip())
                            line = ' ' * indent + f"{node_id}[{label}]"
                
                cleaned_lines.append(line)
        
        # If no flowchart declaration found, add it
        if not found_flowchart:
            cleaned_lines.insert(0, 'flowchart TD')
        
        result = '\n'.join(cleaned_lines)
        
        # Final validation: ensure it starts with flowchart
        if not result.strip().startswith('flowchart'):
            result = 'flowchart TD\n' + result
        
        return result
    
    def _translate_deterministic(self, sfm: ScenarioFlowModel) -> str:
        """Deterministic rule-based translation"""
        lines = ['flowchart TD']
        
        # Build node ID mapping to ensure valid Mermaid IDs
        node_id_map = {}
        node_counter = 0
        used_ids = set()
        
        # First pass: create valid node IDs
        for node_id, node in sfm.nodes.items():
            if node.node_type == SFMNodeType.START:
                mapped_id = 'START'
            elif node.node_type == SFMNodeType.END:
                mapped_id = 'END'
            else:
                sanitized = self._sanitize_id(node_id)
                # Ensure unique IDs
                while sanitized in used_ids:
                    sanitized = f'N{node_counter}'
                    node_counter += 1
                mapped_id = sanitized
            
            node_id_map[node_id] = mapped_id
            used_ids.add(mapped_id)
        
        # Track processed nodes to avoid duplicates
        processed = set()
        
        # Process nodes in order (start first, then others, then end)
        node_order = []
        for node_id, node in sfm.nodes.items():
            if node.node_type == SFMNodeType.START:
                node_order.insert(0, (node_id, node))
            elif node.node_type == SFMNodeType.END:
                node_order.append((node_id, node))
            else:
                node_order.append((node_id, node))
        
        for node_id, node in node_order:
            if node_id in processed:
                continue
            
            # Generate node definition with mapped ID
            mermaid_id = node_id_map[node_id]
            node_def = self._generate_node_definition(node, mermaid_id)
            lines.append(f"    {node_def}")
            processed.add(node_id)
        
        # Generate edges using mapped IDs
        for node_id, node in sfm.nodes.items():
            edge_lines = self._generate_edges(node, sfm.nodes, node_id_map)
            lines.extend([f"    {line}" for line in edge_lines])
        
        mermaid_code = '\n'.join(lines)
        logger.info(f"Deterministic translation successful ({len(mermaid_code)} chars)")
        return mermaid_code
    
    def _generate_node_definition(self, node: SFMNode, mermaid_id: str = None) -> str:
        """Generate Mermaid node definition"""
        if mermaid_id is None:
            node_id = self._sanitize_id(node.id)
        else:
            node_id = mermaid_id
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
    
    def _generate_edges(self, node: SFMNode, all_nodes: Dict[str, SFMNode], node_id_map: Dict[str, str] = None) -> List[str]:
        """Generate edges from a node"""
        edges = []
        
        # Get node ID from map if provided, otherwise sanitize
        if node_id_map:
            node_id = node_id_map.get(node.id, self._sanitize_id(node.id))
        else:
            node_id = self._sanitize_id(node.id)
        
        # Handle decision branches
        if node.node_type == SFMNodeType.DECISION:
            if node.true_branch:
                if node_id_map and node.true_branch in node_id_map:
                    true_id = node_id_map[node.true_branch]
                else:
                    true_id = self._sanitize_id(node.true_branch)
                edges.append(f"{node_id} -->|Yes| {true_id}")
            
            if node.false_branch:
                if node_id_map and node.false_branch in node_id_map:
                    false_id = node_id_map[node.false_branch]
                else:
                    false_id = self._sanitize_id(node.false_branch)
                edges.append(f"{node_id} -->|No| {false_id}")
        
        # Handle regular next nodes
        for next_id in node.next_nodes:
            # Skip if already handled by decision branches
            if next_id in [node.true_branch, node.false_branch]:
                continue
            
            if node_id_map and next_id in node_id_map:
                next_id_clean = node_id_map[next_id]
            else:
                next_id_clean = self._sanitize_id(next_id)
            edges.append(f"{node_id} --> {next_id_clean}")
        
        return edges
    
    def _sanitize_id(self, node_id: str) -> str:
        """Sanitize node ID for Mermaid"""
        # Replace invalid characters
        sanitized = node_id.replace('-', '_').replace(' ', '_').replace('::', '_')
        # Remove special characters that might cause issues
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'N' + sanitized
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'NODE'
        return sanitized
    
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


def translate_to_mermaid(
    sfm: 'ScenarioFlowModel',
    llm_model: str = "qwen2.5-coder:7b",
    llm_base_url: str = "http://localhost:11434",
    use_llm: bool = True
) -> str:
    """
    Convenience function to translate SFM to Mermaid flowchart code.
    
    Args:
        sfm: Scenario Flow Model to translate
        llm_model: Ollama model name (unused currently)
        llm_base_url: Ollama server URL (unused currently)
        use_llm: Whether to use LLM for translation
    
    Returns:
        Mermaid flowchart code as string
    """
    translator = MermaidTranslator(model_name=llm_model, use_llm=use_llm)
    return translator.translate(sfm)
