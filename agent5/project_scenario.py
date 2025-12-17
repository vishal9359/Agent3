"""
Project-level scenario understanding and flowchart generation.

This module enables understanding complete scenarios/operations across
an entire C++ project, not just single functions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from agent5.config import SETTINGS
from agent5.ollama_compat import get_chat_ollama
from agent5.scenario_extractor import ScenarioFlowModel, SFMBuilder, SFMNode, SFMEdge
from agent5.vectorstore import get_vectorstore


@dataclass
class ProjectScenario:
    """A complete project-level scenario."""
    
    scenario_name: str
    description: str
    entry_points: list[str]  # Functions/files where scenario starts
    key_steps: list[dict[str, Any]]  # High-level steps
    involved_files: list[str]  # All files involved
    sfm: ScenarioFlowModel | None = None


SCENARIO_UNDERSTANDING_SYSTEM_PROMPT = """You are a senior software architect analyzing C++ codebases.

Your task: Given a scenario/operation name and relevant code snippets, identify the HIGH-LEVEL flow of that scenario across the entire project.

Focus on:
- BUSINESS LOGIC flow (not implementation details)
- Key decision points (validation, error handling)
- State changes (create, update, delete operations)
- Cross-module interactions
- Success and failure paths

Exclude:
- Logging and metrics
- Low-level implementation details
- Utility functions
- Memory management details

Output format: JSON with this structure:
{
  "scenario_name": "...",
  "description": "Brief description of what this scenario does",
  "entry_points": ["function/file where it starts"],
  "steps": [
    {
      "id": 1,
      "type": "process|decision|io",
      "label": "High-level action",
      "description": "What happens",
      "files": ["file1.cpp", "file2.cpp"],
      "branches": {
        "success": 2,
        "failure": 5
      }
    }
  ]
}

Be concise and focus on the SCENARIO FLOW, not code details.
"""


def understand_project_scenario(
    *,
    collection: str,
    scenario_name: str,
    project_path: Path | None = None,
    k: int = 20,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
) -> ProjectScenario:
    """
    Understand a scenario across the entire project using RAG + LLM.
    
    Args:
        collection: Name of the indexed collection
        scenario_name: Scenario/operation name (e.g., "Create volume")
        project_path: Project root path
        k: Number of code chunks to retrieve
        chat_model: Chat model name
        embed_model: Embedding model name
        ollama_base_url: Ollama base URL
        
    Returns:
        ProjectScenario with understanding of the flow
        
    Raises:
        RuntimeError: If scenario cannot be understood
    """
    base_url = ollama_base_url or SETTINGS.ollama_base_url
    
    # Retrieve relevant code using RAG
    vs = get_vectorstore(collection, embed_model=embed_model, ollama_base_url=base_url)
    
    # Build comprehensive query
    queries = [
        scenario_name,
        f"{scenario_name} main function",
        f"{scenario_name} entry point",
        f"{scenario_name} flow",
        f"{scenario_name} implementation",
    ]
    
    docs: list[Document] = []
    for query in queries:
        try:
            docs.extend(vs.similarity_search(query, k=k // len(queries)))
        except Exception:
            pass
    
    # Deduplicate
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for d in docs:
        key = d.metadata.get("qualified_name", "") + d.metadata.get("relpath", "")
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    
    docs = unique_docs[:k]
    
    if not docs:
        raise RuntimeError(
            f"Cannot find any code related to scenario '{scenario_name}'. "
            f"Make sure the project is indexed with: python -m agent5 index --project_path <path> --collection {collection}"
        )
    
    # Format context for LLM
    context = _format_context_for_scenario(docs)
    
    # Ask LLM to understand the scenario
    llm = get_chat_ollama(model=chat_model or SETTINGS.ollama_chat_model, base_url=base_url)
    
    user_msg = f"""Scenario: {scenario_name}

Analyze the following code snippets and identify the HIGH-LEVEL flow of this scenario.

{context}

Output the scenario understanding as JSON following the specified format.
"""
    
    messages = [
        SystemMessage(content=SCENARIO_UNDERSTANDING_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]
    
    resp = llm.invoke(messages)
    content = getattr(resp, "content", str(resp))
    
    # Parse JSON response
    try:
        # Extract JSON from response (might have markdown code fences)
        json_str = _extract_json(content)
        data = json.loads(json_str)
        
        # Build ProjectScenario
        scenario = ProjectScenario(
            scenario_name=data.get("scenario_name", scenario_name),
            description=data.get("description", ""),
            entry_points=data.get("entry_points", []),
            key_steps=data.get("steps", []),
            involved_files=list(set(
                file
                for step in data.get("steps", [])
                for file in step.get("files", [])
            )),
        )
        
        return scenario
    
    except Exception as e:
        raise RuntimeError(
            f"Failed to understand scenario '{scenario_name}'. "
            f"LLM response parsing failed: {e}\n\n"
            f"LLM output:\n{content}"
        ) from e


def _format_context_for_scenario(docs: list[Document], max_chars: int = 40000) -> str:
    """Format documents as context for scenario understanding."""
    parts: list[str] = []
    total = 0
    
    for d in docs:
        src = d.metadata.get("relpath") or d.metadata.get("source") or "unknown"
        chunk_type = d.metadata.get("chunk_type", "code")
        name = d.metadata.get("qualified_name") or d.metadata.get("name", "")
        
        header = f"\n--- FILE: {src}"
        if name:
            header += f" | {chunk_type.upper()}: {name}"
        header += " ---\n"
        
        chunk = d.page_content
        
        # Truncate very long chunks
        if len(chunk) > 2000:
            chunk = chunk[:2000] + "\n... (truncated)"
        
        block = header + chunk + "\n"
        
        if total + len(block) > max_chars:
            break
        
        parts.append(block)
        total += len(block)
    
    return "".join(parts).strip()


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response (handles markdown code fences)."""
    text = text.strip()
    
    # Remove markdown code fences
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    
    # Find JSON object
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
    
    return text


def build_sfm_from_project_scenario(scenario: ProjectScenario, max_steps: int = 50) -> ScenarioFlowModel:
    """
    Build a Scenario Flow Model from a project-level scenario understanding.
    
    Args:
        scenario: ProjectScenario with high-level steps
        max_steps: Maximum steps in SFM
        
    Returns:
        ScenarioFlowModel
    """
    builder = SFMBuilder(max_steps=max_steps)
    
    # Track nodes by step ID
    node_map: dict[int, str] = {}
    
    # Process each step
    for step in scenario.key_steps:
        step_id = step.get("id")
        step_type = step.get("type", "process")
        label = step.get("label", f"Step {step_id}")
        
        # Create node
        if step_type == "decision":
            node_id = builder.add_decision(label)
        elif step_type == "io":
            node_id = builder.add_node(builder._new_id("io"), "io", label)
        else:  # process
            node_id = builder.add_process(label)
        
        if not node_id:
            break
        
        node_map[step_id] = node_id
    
    # Connect nodes based on flow
    prev_id = "start"
    for i, step in enumerate(scenario.key_steps):
        step_id = step.get("id")
        node_id = node_map.get(step_id)
        
        if not node_id:
            continue
        
        # Connect from previous
        if i == 0:
            builder.add_edge("start", node_id)
        else:
            builder.add_edge(prev_id, node_id)
        
        # Handle branches
        branches = step.get("branches", {})
        if branches:
            # Decision node with branches
            for branch_name, next_step_id in branches.items():
                next_node = node_map.get(next_step_id)
                if next_node:
                    builder.add_edge(node_id, next_node, branch_name.upper())
                elif branch_name.lower() in {"failure", "error", "fail"}:
                    builder.add_edge(node_id, "end", branch_name.upper())
            prev_id = node_id
        else:
            prev_id = node_id
    
    # Connect last node to end
    if prev_id != "start":
        builder.add_edge(prev_id, "end")
    
    return builder.build()


def generate_project_scenario_flowchart(
    *,
    collection: str,
    scenario_name: str,
    output_path: Path,
    project_path: Path | None = None,
    max_steps: int = 50,
    k: int = 20,
    chat_model: str | None = None,
    embed_model: str | None = None,
    ollama_base_url: str | None = None,
) -> tuple[ProjectScenario, str]:
    """
    Generate a flowchart for a project-level scenario.
    
    Args:
        collection: Indexed collection name
        scenario_name: Scenario/operation name
        output_path: Output .mmd file path
        project_path: Project root path
        max_steps: Maximum steps in flowchart
        k: Number of code chunks to retrieve
        chat_model: Chat model name
        embed_model: Embedding model name
        ollama_base_url: Ollama base URL
        
    Returns:
        Tuple of (ProjectScenario, mermaid_code)
    """
    # Understand the scenario using RAG + LLM
    scenario = understand_project_scenario(
        collection=collection,
        scenario_name=scenario_name,
        project_path=project_path,
        k=k,
        chat_model=chat_model,
        embed_model=embed_model,
        ollama_base_url=ollama_base_url,
    )
    
    # Build SFM from understanding
    sfm = build_sfm_from_project_scenario(scenario, max_steps=max_steps)
    scenario.sfm = sfm
    
    # Convert to Mermaid
    from agent5.flowchart import _sfm_to_mermaid
    mermaid = _sfm_to_mermaid(sfm)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mermaid, encoding="utf-8")
    
    # Also write scenario understanding
    understanding_path = output_path.with_suffix(".scenario.json")
    understanding_path.write_text(
        json.dumps({
            "scenario_name": scenario.scenario_name,
            "description": scenario.description,
            "entry_points": scenario.entry_points,
            "involved_files": scenario.involved_files,
            "steps": scenario.key_steps,
        }, indent=2),
        encoding="utf-8",
    )
    
    return scenario, mermaid

