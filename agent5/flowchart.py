"""
Enhanced flowchart generation for C++ projects.

This module generates Mermaid flowcharts from Scenario Flow Models (SFM).
It implements the strict pipeline: AST → SFM → Mermaid

Key principles:
- SFM MUST exist before calling LLM
- LLM is TRANSLATOR ONLY (optional)
- Deterministic fallback always available
- Fail fast if SFM cannot be built
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agent5.config import SETTINGS
from agent5.ollama_compat import get_chat_ollama
from agent5.scenario_extractor import (
    ScenarioFlowModel,
    extract_scenario_from_function,
)


@dataclass(frozen=True)
class MermaidFlowchart:
    """Generated Mermaid flowchart with metadata."""
    
    mermaid: str
    node_count: int
    edge_count: int
    sfm: dict | None = None  # Original SFM for reference


def _sanitize_mermaid_label(label: str) -> str:
    """Make label safe for Mermaid syntax."""
    import re
    
    # Remove or escape problematic characters
    s = label.replace('"', "'")
    s = s.replace("[", "(")
    s = s.replace("]", ")")
    s = s.replace("{", "(")
    s = s.replace("}", ")")
    
    # Remove extra operators
    s = re.sub(r"[;:*&%^~`]", " ", s)
    s = " ".join(s.split())
    
    return s or "Step"


def _sfm_to_mermaid(sfm: ScenarioFlowModel) -> str:
    """
    Convert SFM to Mermaid flowchart (deterministic).
    
    This is the authoritative conversion that always works.
    """
    lines = ["flowchart TD"]
    
    # Remap start/end to avoid Mermaid keyword conflicts
    id_map = {"start": "startNode", "end": "endNode"}
    
    # Declare nodes
    for node in sfm.nodes:
        nid = id_map.get(node.id, node.id)
        label = _sanitize_mermaid_label(node.label)
        
        if node.node_type == "terminator":
            lines.append(f"  {nid}([{label}])")
        elif node.node_type == "decision":
            # Ensure label ends with ?
            if not label.endswith("?"):
                label += "?"
            lines.append(f"  {nid}{{{label}}}")
        elif node.node_type == "io":
            lines.append(f"  {nid}[/{label}/]")
        else:  # process
            lines.append(f"  {nid}[{label}]")
    
    # Declare edges
    for edge in sfm.edges:
        src = id_map.get(edge.src, edge.src)
        dst = id_map.get(edge.dst, edge.dst)
        
        if edge.label:
            label = _sanitize_mermaid_label(edge.label)
            lines.append(f"  {src} -- {label} --> {dst}")
        else:
            lines.append(f"  {src} --> {dst}")
    
    return "\n".join(lines) + "\n"


def _count_mermaid_elements(mermaid: str) -> tuple[int, int]:
    """Count nodes and edges in Mermaid code."""
    node_ids = set()
    edge_count = 0
    
    for line in mermaid.splitlines():
        t = line.strip()
        if not t or t.startswith("%%") or t.startswith("flowchart"):
            continue
        
        if "-->" in t:
            edge_count += 1
            # Extract node IDs from edge
            parts = t.split("-->")
            if len(parts) == 2:
                left = parts[0].split("--")[0].strip()
                right = parts[1].strip()
                # Extract ID before any brackets/braces
                for part in [left, right]:
                    nid = re.split(r"[\[\({]", part)[0].strip()
                    if nid:
                        node_ids.add(nid)
        else:
            # Node declaration
            nid = re.split(r"[\[\({]", t)[0].strip()
            if nid:
                node_ids.add(nid)
    
    return len(node_ids), edge_count


TRANSLATOR_SYSTEM_PROMPT = """You are a diagram translator.

You will receive a Scenario Flow Model (SFM) as JSON. The SFM is authoritative.

Task: Translate the SFM to Mermaid flowchart code ONLY.

Output rules (STRICT):
- Output ONLY Mermaid code, starting with: flowchart TD
- No markdown, no bullets, no headings, no explanations
- Use these shapes:
  - Terminator: id([Label])
  - Process:    id[Label]
  - Decision:   id{Label?}
  - I/O:        id[/Label/]
- Every node must be explicitly declared using one of the shapes above
- Edges may include labels using: -- LABEL --> syntax
- NEVER include extra text, explanations, or prose
"""


def _translate_sfm_with_llm(
    sfm: ScenarioFlowModel,
    chat_model: str,
    ollama_base_url: str,
) -> str | None:
    """
    Optionally translate SFM to Mermaid using LLM.
    
    The LLM is used ONLY as a translator. The SFM is authoritative.
    If LLM fails or produces invalid output, we fall back to deterministic conversion.
    
    Returns:
        Mermaid code or None if LLM fails
    """
    try:
        llm = get_chat_ollama(model=chat_model, base_url=ollama_base_url)
        
        sfm_json = json.dumps(sfm.to_dict(), indent=2)
        user_msg = f"Scenario Flow Model (JSON):\n\n{sfm_json}\n\nTranslate to Mermaid flowchart:"
        
        resp = llm.invoke([
            SystemMessage(content=TRANSLATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        
        content = getattr(resp, "content", str(resp))
        
        # Extract Mermaid code
        mermaid = _extract_mermaid(content)
        if mermaid:
            # Basic validation
            if "flowchart" in mermaid and "-->" in mermaid:
                return mermaid
        
        return None
    
    except Exception:
        return None


def _extract_mermaid(text: str) -> str | None:
    """Extract Mermaid code from LLM response."""
    # Remove markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    
    # Find flowchart start
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("flowchart"):
            start_idx = i
            break
    
    if start_idx is None:
        return None
    
    # Take everything from flowchart onward
    mermaid = "\n".join(lines[start_idx:]).strip()
    
    # Basic sanity check
    if not mermaid or "flowchart" not in mermaid:
        return None
    
    return mermaid + "\n"


def generate_scenario_flowchart_from_json_ast(
    ast_json_path: str | Path,
    function_name: str | None = None,
    *,
    max_steps: int = 30,
    detail_level: str = "medium",
    use_llm: bool = False,
    chat_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidFlowchart:
    """
    Generate a scenario-based flowchart from JSON AST file.
    
    Pipeline:
        1. Load JSON AST (generated by libclang)
        2. Build SFM from AST (deterministic, MUST succeed)
        3. Optionally translate with LLM (fallback to deterministic if fails)
        4. Return Mermaid flowchart
    
    Args:
        ast_json_path: Path to JSON AST file
        function_name: Entry function name (auto-detect if None)
        max_steps: Maximum steps in the scenario
        detail_level: Detail level (high|medium|deep)
        use_llm: Use LLM for translation (optional)
        chat_model: Chat model name
        ollama_base_url: Ollama base URL
        
    Returns:
        MermaidFlowchart
        
    Raises:
        RuntimeError: If SFM cannot be built (FAIL FAST)
    """
    from agent5.json_ast_sfm import build_sfm_from_json_file
    from agent5.scenario_extractor import DetailLevel
    
    # Convert string to enum
    detail_enum = DetailLevel.MEDIUM  # default
    if detail_level:
        detail_level_lower = detail_level.lower()
        if detail_level_lower == "high":
            detail_enum = DetailLevel.HIGH
        elif detail_level_lower == "deep":
            detail_enum = DetailLevel.DEEP
    
    # Step 1: Build SFM from JSON AST (REQUIRED, FAIL FAST)
    sfm = build_sfm_from_json_file(
        ast_json_path,
        function_name=function_name,
        max_steps=max_steps,
        detail_level=detail_enum,
    )
    
    # Step 2: Translate to Mermaid
    mermaid = None
    
    if use_llm:
        # Try LLM translation (optional, fallback if fails)
        mermaid = _translate_sfm_with_llm(
            sfm,
            chat_model=chat_model or SETTINGS.ollama_chat_model,
            ollama_base_url=ollama_base_url or SETTINGS.ollama_base_url,
        )
    
    # Fallback to deterministic conversion
    if not mermaid:
        mermaid = _sfm_to_mermaid(sfm)
    
    # Count elements
    nodes, edges = _count_mermaid_elements(mermaid)
    
    return MermaidFlowchart(
        mermaid=mermaid,
        node_count=nodes,
        edge_count=edges,
        sfm=sfm.to_dict(),
    )


def generate_scenario_flowchart(
    source_code: str,
    function_name: str | None = None,
    *,
    max_steps: int = 30,
    detail_level: str = "medium",
    use_llm: bool = False,
    chat_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidFlowchart:
    """
    Generate a scenario-based flowchart from C++ code.
    
    DEPRECATED: This function uses tree-sitter. Use generate_scenario_flowchart_from_json_ast instead.
    
    Pipeline:
        1. Extract SFM from code (deterministic, MUST succeed)
        2. Optionally translate with LLM (fallback to deterministic if fails)
        3. Return Mermaid flowchart
    
    Args:
        source_code: C++ source code
        function_name: Entry function name (auto-detect if None)
        max_steps: Maximum steps in the scenario
        detail_level: Detail level (high|medium|deep)
        use_llm: Use LLM for translation (optional)
        chat_model: Chat model name
        ollama_base_url: Ollama base URL
        
    Returns:
        MermaidFlowchart
        
    Raises:
        RuntimeError: If SFM cannot be built (FAIL FAST)
    """
    from agent5.scenario_extractor import DetailLevel
    
    # Convert string to enum
    detail_enum = DetailLevel.MEDIUM  # default
    if detail_level:
        detail_level_lower = detail_level.lower()
        if detail_level_lower == "high":
            detail_enum = DetailLevel.HIGH
        elif detail_level_lower == "deep":
            detail_enum = DetailLevel.DEEP
    
    # Step 1: Extract SFM (REQUIRED, FAIL FAST)
    sfm = extract_scenario_from_function(
        source_code,
        function_name=function_name,
        max_steps=max_steps,
        detail_level=detail_enum,
    )
    
    # Step 2: Translate to Mermaid
    mermaid = None
    
    if use_llm:
        # Try LLM translation (optional, fallback if fails)
        mermaid = _translate_sfm_with_llm(
            sfm,
            chat_model=chat_model or SETTINGS.ollama_chat_model,
            ollama_base_url=ollama_base_url or SETTINGS.ollama_base_url,
        )
    
    # Fallback to deterministic conversion
    if not mermaid:
        mermaid = _sfm_to_mermaid(sfm)
    
    # Count elements
    nodes, edges = _count_mermaid_elements(mermaid)
    
    return MermaidFlowchart(
        mermaid=mermaid,
        node_count=nodes,
        edge_count=edges,
        sfm=sfm.to_dict(),
    )


def generate_flowchart_from_file(
    file_path: Path,
    function_name: str | None = None,
    *,
    max_steps: int = 30,
    detail_level: str = "medium",
    use_llm: bool = False,
    chat_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidFlowchart:
    """
    Generate a flowchart from a C++ file.
    
    Args:
        file_path: Path to C++ file
        function_name: Entry function name (auto-detect if None)
        max_steps: Maximum steps
        detail_level: Detail level (high|medium|deep)
        use_llm: Use LLM for translation
        chat_model: Chat model name
        ollama_base_url: Ollama base URL
        
    Returns:
        MermaidFlowchart
        
    Raises:
        RuntimeError: If file cannot be read or SFM cannot be built
    """
    try:
        source_code = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Cannot read file {file_path}: {e}") from e
    
    if not source_code.strip():
        raise RuntimeError(f"File is empty: {file_path}")
    
    return generate_scenario_flowchart(
        source_code,
        function_name=function_name,
        max_steps=max_steps,
        detail_level=detail_level,
        use_llm=use_llm,
        chat_model=chat_model,
        ollama_base_url=ollama_base_url,
    )


def write_flowchart(
    output_path: Path,
    file_path: Path | None = None,
    function_name: str | None = None,
    *,
    ast_json_path: Path | None = None,
    project_path: Path | None = None,
    max_steps: int = 30,
    detail_level: str = "medium",
    use_llm: bool = False,
    chat_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidFlowchart:
    """
    Generate and write a flowchart to a file using JSON AST.
    
    This function now uses libclang-generated JSON AST instead of tree-sitter.
    
    Args:
        output_path: Path to output .mmd file
        file_path: Path to entry file (optional, for backward compatibility)
        function_name: Entry function name
        ast_json_path: Path to JSON AST file (if None, will look for ast_with_calls.json in project_path)
        project_path: Project root - used to locate AST JSON file
        max_steps: Maximum steps
        detail_level: Detail level (high|medium|deep)
        use_llm: Use LLM for translation
        chat_model: Chat model name
        ollama_base_url: Ollama base URL
        
    Returns:
        MermaidFlowchart
    """
    # Determine AST JSON path
    if ast_json_path is None:
        if project_path:
            ast_json_path = project_path / "ast_with_calls.json"
        elif file_path:
            ast_json_path = file_path.parent / "ast_with_calls.json"
        else:
            raise RuntimeError(
                "Either ast_json_path or project_path must be provided. "
                "Or ensure ast_with_calls.json exists in the project directory."
            )
    
    ast_json_path = Path(ast_json_path)
    if not ast_json_path.exists():
        raise FileNotFoundError(
            f"AST JSON file not found: {ast_json_path}. "
            "Please run libclang AST builder first to generate the JSON file."
        )
    
    # Generate flowchart from JSON AST
    flowchart = generate_scenario_flowchart_from_json_ast(
        ast_json_path,
        function_name=function_name,
        max_steps=max_steps,
        detail_level=detail_level,
        use_llm=use_llm,
        chat_model=chat_model,
        ollama_base_url=ollama_base_url,
    )
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(flowchart.mermaid, encoding="utf-8")
    
    # Also write SFM for debugging
    if flowchart.sfm:
        sfm_path = output_path.with_suffix(".sfm.json")
        sfm_path.write_text(
            json.dumps(flowchart.sfm, indent=2),
            encoding="utf-8",
        )
    
    return flowchart

