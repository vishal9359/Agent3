# Version 4 Features: Bottom-up Semantic Aggregation

## Overview

Version 4 introduces a revolutionary **bottom-up semantic aggregation** approach to C++ project analysis, inspired by the DocAgent paper. This enables true project-level understanding and generates accurate, documentation-quality flowcharts for complex codebases.

## What's New

### 🚀 Bottom-up Semantic Aggregation

Unlike v3's top-down approach, v4 builds understanding from the ground up:

1. **Start from leaf functions** (functions that make no calls)
2. **Use LLM to extract semantics** (WHAT they do, not HOW)
3. **Progressively aggregate upward** through the call graph
4. **Each level builds on lower levels** until reaching the entry function

### 🏗️ Architecture Comparison

#### V3 (Top-down):
```
Entry Function → Follow calls → Expand (limited depth) → SFM
```

**Limitations:**
- Limited by max_depth parameter
- No semantic understanding (deterministic AST only)
- Cannot handle complex call graphs

#### V4 (Bottom-up):
```
All Functions → Call Graph → Leaf Summaries → Level-by-level Aggregation → Entry Summary → SFM
```

**Advantages:**
- Scalable (no depth limit)
- Semantic understanding (knows WHAT, not just HOW)
- Handles complex call graphs naturally

## How It Works

### Step 1: Build Call Graph

```python
call_graph = build_call_graph(project_path)
```

- Analyzes **all C++ files** in the project
- Extracts function definitions with metadata
- Resolves function calls across files, namespaces, classes
- Computes **levels** (distance from leaf functions)

### Step 2: Find Entry Function

```python
entry_node = find_entry_function(call_graph, function_name, file_path)
```

- Locates the entry function using name + optional file
- Handles overloads, templates, namespaces
- Provides helpful error messages if not found

### Step 3: Bottom-up Aggregation

```python
semantic_summaries = build_semantic_hierarchy(call_graph, entry_node, detail_level)
```

**For each level (starting from leaves):**

1. **Leaf functions** (level 0):
   - Extract source code
   - LLM analyzes: "What does this function DO?"
   - Output: Semantic summary (operations, decisions, state changes)

2. **Non-leaf functions** (level 1+):
   - Extract source code
   - Collect semantic summaries of all called functions
   - LLM aggregates: "How does this orchestrate its callees?"
   - Output: Higher-level semantic summary

3. **Repeat** until reaching the entry function

**Example:**

```
Level 0 (Leaves):
  - parseJSON() → "Parses JSON string into object"
  - validateUser() → "Checks if user credentials are valid"
  - saveToDatabase() → "Persists data to database"

Level 1:
  - processRequest(calls: parseJSON, validateUser, saveToDatabase)
    → "Handles HTTP request by validating user and saving data"

Level 2 (Entry):
  - main(calls: processRequest)
    → "Web server entry point that routes requests to handlers"
```

### Step 4: Convert to SFM

```python
sfm, metadata = extract_scenario_from_project(...)
```

- Translates semantic understanding into Scenario Flow Model
- Maintains **deterministic validation** (start/end nodes, valid edges)
- Generates rich metadata for debugging

### Step 5: Generate Flowchart

```python
mermaid = sfm_to_mermaid(sfm)
```

- Converts SFM to Mermaid flowchart
- Optionally uses LLM for translation
- Always has deterministic fallback

## Usage

### Enable V4 Mode

**Key:** Use `--project-path` to enable bottom-up aggregation!

```bash
python -m agent5 flowchart \
  --file src/main.cpp \
  --function handle_request \
  --project-path /path/to/project \
  --out flowchart.mmd \
  --detail-level medium
```

### Without `--project-path` (V3 mode)

```bash
python -m agent5 flowchart \
  --file src/main.cpp \
  --function handle_request \
  --out flowchart.mmd \
  --detail-level medium
```

This uses the **single-file deterministic approach** (v3) instead of bottom-up aggregation.

## Output Files

### V3 Mode
- `output.mmd` - Mermaid flowchart
- `output.sfm.json` - Scenario Flow Model (debug)

### V4 Mode
- `output.mmd` - Mermaid flowchart
- `output.sfm.json` - Scenario Flow Model (debug)
- `output.metadata.json` - **NEW**: Semantic summaries, call graph info, scenario description

### Example Metadata (v4)

```json
{
  "entry_function": "HTTPServer::handle_request",
  "entry_file": "/project/src/server.cpp",
  "call_graph_depth": 3,
  "reachable_functions": 15,
  "scenario_description": "## Scenario: HTTPServer::handle_request\n\n**Overview:** Handles incoming HTTP requests...",
  "semantic_summary": {
    "summary": "Processes HTTP requests with validation and routing",
    "key_operations": [
      "Parse request headers",
      "Validate authentication",
      "Route to handler",
      "Generate response"
    ],
    "decisions": [
      "Is authentication valid?",
      "Does route exist?",
      "Should cache response?"
    ],
    "state_changes": [
      "Update request counter",
      "Log access",
      "Save session"
    ]
  }
}
```

## Detail Levels

V4 respects detail levels throughout the pipeline:

### HIGH
- Only major business operations
- Minimal decision points
- Suitable for **architecture overview**

### MEDIUM (default)
- All validations, decisions, state changes
- Balanced detail
- Suitable for **documentation**

### DEEP
- Expanded critical sub-operations
- Internal validation steps
- Suitable for **debugging**

**Important:** Detail level affects **semantic aggregation**, not just SFM filtering!

## Advantages

### 1. True Cross-file Understanding
- Analyzes the **entire project**
- No depth limits
- Natural handling of complex call graphs

### 2. Semantic, Not Syntactic
- Understands **WHAT** the code does
- Collapses implementation details
- Produces high-level, readable flowcharts

### 3. Scalable
- Call graph levels provide natural boundaries
- LLM calls are cached per function
- Can handle large projects

### 4. Documentation Quality
- Suitable for **technical documentation**
- Explains business logic, not implementation
- Multiple detail levels for different audiences

## Limitations

1. **Requires LLM**: V4 needs a running Ollama instance
2. **Slower**: Bottom-up aggregation takes longer than v3 (trades speed for accuracy)
3. **LLM quality**: Output depends on the quality of the chat model
4. **Large projects**: Very large call graphs may require parameter tuning

## Technical Details

### Call Graph

```python
@dataclass
class FunctionInfo:
    name: str
    file_path: Path
    namespace: str | None
    class_name: str | None
    qualified_name: str  # namespace::class::function
    calls: list[str]
    called_by: list[str]
    is_leaf: bool
    level: int  # Distance from leaves
```

### Semantic Summary

```python
@dataclass
class SemanticSummary:
    function_name: str
    level: int
    summary: str
    key_operations: list[str]
    decisions: list[str]
    state_changes: list[str]
    metadata: dict | None
```

### LLM Prompts

V4 uses two specialized prompts:

1. **Leaf Function Prompt**:
   - "What does this function DO?"
   - Focus on semantic meaning
   - Output: JSON with operations, decisions, state changes

2. **Aggregation Prompt**:
   - "How does this orchestrate its callees?"
   - Combines own code + callee summaries
   - Output: Higher-level semantic summary

## Inspired By

This implementation is inspired by:

- **DocAgent** (Meta AI): Bottom-up docstring synthesis
- **Key adaptation**: Applied to flowchart generation instead of documentation

## When to Use V4

### Use V4 (bottom-up) when:
- ✅ You need **project-level understanding**
- ✅ The scenario spans **multiple files**
- ✅ You want **semantic, high-level** flowcharts
- ✅ You're creating **documentation**

### Use V3 (single-file) when:
- ✅ You're analyzing a **single function**
- ✅ You want **fast** results
- ✅ You don't have an **LLM available**
- ✅ You need **deterministic** output

## Example

### Input
```bash
python -m agent5 flowchart \
  --file src/server.cpp \
  --function handle_request \
  --project-path /path/to/webserver \
  --out server_flow.mmd \
  --detail-level medium
```

### Console Output
```
═══ Agent5: Generate Flowchart ═══

Mode: Entry-Point Scenario
Entry file: src/server.cpp
Entry function: handle_request
Analysis scope: /path/to/webserver (entire project)
Detail level: medium
Max steps: 30
Output: server_flow.mmd

Step 1: Building call graph across project...
Step 2: Bottom-up semantic aggregation (DocAgent-inspired)...
Step 3: Constructing Scenario Flow Model (SFM)...

✓ Flowchart generated successfully
  Nodes: 12
  Edges: 15
  Output: server_flow.mmd
  SFM (debug): server_flow.sfm.json
  Metadata (v4): server_flow.metadata.json
  (includes semantic summaries and call graph info)
```

### Generated Files
- `server_flow.mmd` - Visual flowchart
- `server_flow.sfm.json` - Structured flow model
- `server_flow.metadata.json` - Semantic understanding

## Summary

**Version 4 = Version 3 + Bottom-up Semantic Aggregation**

- ✅ Maintains all v3 guarantees (determinism, validation, fail-fast)
- ✅ Adds semantic understanding through LLM-assisted aggregation
- ✅ Enables true project-level flowchart generation
- ✅ Produces documentation-quality output

**Result:** Accurate, readable flowcharts for complex C++ projects! 🎉

