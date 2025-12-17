# Flowchart Generation Flow - Technical Documentation

This document explains the internal execution flow of how Agent5 generates flowchart diagrams from C++ code.

---

## Overview: Pipeline Architecture

```
CLI Arguments
    ↓
File Reading
    ↓
AST Parsing (tree-sitter)
    ↓
Entry Function Resolution
    ↓
Scenario Flow Extraction (deterministic, rule-based)
    ↓
Scenario Flow Model (SFM) - JSON structure
    ↓
SFM Validation (fail-fast gate)
    ↓
Mermaid Translation (LLM optional, deterministic fallback)
    ↓
Mermaid Flowchart Code (.mmd file)
```

---

## Step-by-Step Execution Flow

### 1. CLI Entry Point (`agent5/cli.py`)

**Function:** `main()`

**Flow:**
```
Parse arguments
    ↓
Identify mode: --scenario OR --file
    ↓
[Mode: --file]
    ↓
Call write_flowchart(
    output_path,
    file_path,           # Entry point locator
    function_name,       # Optional
    project_path,        # Analysis scope (optional)
    detail_level,        # high|medium|deep
    max_steps,
    use_llm
)
```

**Key Parameters:**
- `file_path`: Used ONLY to locate entry function (not analysis scope)
- `function_name`: Entry function name (auto-detect if None)
- `detail_level`: Controls extraction granularity (affects SFM structure)
- `max_steps`: Maximum steps before truncation

---

### 2. File Reading (`agent5/flowchart.py`)

**Function:** `write_flowchart()` → `generate_flowchart_from_file()`

**Flow:**
```
Read file content (UTF-8, ignore errors)
    ↓
Validate: file not empty
    ↓
Call generate_scenario_flowchart(source_code, ...)
```

---

### 3. Scenario Flowchart Generation (`agent5/flowchart.py`)

**Function:** `generate_scenario_flowchart()`

**Flow:**
```
Convert detail_level string → DetailLevel enum
    ↓
┌─────────────────────────────────────────┐
│ STEP 1: Extract SFM (REQUIRED, FAIL FAST) │
└─────────────────────────────────────────┘
    ↓
Call extract_scenario_from_function(
    source_code,
    function_name,
    max_steps,
    detail_level
)
    ↓
[If extraction fails → Raise RuntimeError]
    ↓
┌─────────────────────────────────────────────┐
│ STEP 2: Translate SFM to Mermaid             │
└─────────────────────────────────────────────┘
    ↓
If use_llm:
    Try _translate_sfm_with_llm()
    [If fails → fallback to deterministic]
    ↓
If no LLM or LLM failed:
    Call _sfm_to_mermaid() (deterministic)
    ↓
Count nodes and edges
    ↓
Return MermaidFlowchart(mermaid, node_count, edge_count, sfm)
```

**Critical Rule:** SFM MUST exist before any LLM is called. If SFM extraction fails, the process terminates immediately (fail-fast).

---

### 4. Scenario Flow Extraction (`agent5/scenario_extractor.py`)

**Function:** `extract_scenario_from_function()`

**Flow:**
```
Create tree-sitter Parser
    ↓
Set C++ language
    ↓
Parse source_code → AST (Abstract Syntax Tree)
    ↓
Get root node
    ↓
Find entry function node:
    Call _find_function(root, function_name)
    [Recursive search through AST]
    [If not found → Raise RuntimeError with available functions]
    ↓
Create SFMBuilder(max_steps, detail_level)
    ↓
Extract from function body:
    Call _extract_from_function_body(fn_node, builder)
    ↓
Build SFM:
    Call builder.build()
        - Creates ScenarioFlowModel
        - Validates: exactly 1 start, ≥1 end, all edges valid
        - If invalid → Raise RuntimeError
    ↓
Return ScenarioFlowModel
```

#### 4.1 Entry Function Resolution

**Function:** `_find_function()`

**Algorithm:**
```
Stack-based DFS traversal of AST
    ↓
Collect all function_definition nodes
    ↓
If function_name is None:
    Prioritize "main" function
    Else: return first function
    ↓
If function_name provided:
    Case-insensitive, partial matching
    Prioritize exact match → shortest partial match
    ↓
Return function node (or None)
```

---

### 5. Function Body Extraction (`agent5/scenario_extractor.py`)

**Function:** `_extract_from_function_body()`

**Flow:**
```
Get function body node
    ↓
Initialize frontier = ["start"]
    ↓
Process function body block:
    Call _process_block(body, builder, ["start"], None)
    ↓
Get updated frontier (execution points after body)
    ↓
Connect all frontier nodes → "end"
    ↓
[Extraction complete]
```

**Frontier Concept:**
- Represents current execution points (control flow positions)
- Starts with `["start"]`
- Updated after each statement/block
- All frontier nodes must eventually connect to `"end"`

---

### 6. Block Processing (`agent5/scenario_extractor.py`)

**Function:** `_process_block()`

**Flow:**
```
Get statements from block node
    ↓
current_frontier = incoming_frontier
    ↓
For each statement:
    current_frontier = _process_statement(
        statement,
        builder,
        current_frontier,
        branch_label (if first statement)
    )
    ↓
    If current_frontier is empty:
        Break (execution terminated)
    ↓
Return current_frontier
```

**Statement Processing:**
Each statement updates the frontier based on control flow:
- Sequential statements: frontier passes through
- Conditionals: frontier splits, then merges
- Returns: frontier becomes `["end"]`
- Loops: frontier cycles back

---

### 7. Statement Processing (`agent5/scenario_extractor.py`)

**Function:** `_process_statement()`

**Dispatch Logic:**
```
If step_count >= max_steps:
    Connect frontier → "end"
    Return []
    ↓
Switch statement type:
    ├─ if_statement      → _process_if()
    ├─ return_statement  → _process_return()
    ├─ throw_statement   → _process_throw()
    ├─ while_statement   → _process_loop()
    ├─ for_statement     → _process_loop()
    ├─ switch_statement  → _process_switch()
    ├─ expression_statement → _process_expression()
    └─ declaration       → _process_declaration()
```

#### 7.1 If Statement Processing

**Function:** `_process_if()`

**Flow:**
```
Extract condition text
    ↓
Add decision node: builder.add_decision(condition + "?")
    ↓
Connect frontier → decision node
    ↓
Process then branch:
    _process_block(consequence, builder, [decision], "YES")
    → then_frontier
    ↓
Process else branch (if exists):
    _process_block(alternative, builder, [decision], "NO")
    → else_frontier
    ↓
If branch has no nodes:
    Add "Proceed" process node
    Connect decision → process node with label
    ↓
Merge frontiers: then_frontier + else_frontier
    ↓
Return merged frontier
```

**Diagram:**
```
Frontier nodes
    ↓
Decision node (condition?)
    ├─ YES → then_frontier
    └─ NO  → else_frontier
        ↓
    Merged frontier
```

#### 7.2 Return Statement Processing

**Function:** `_process_return()`

**Flow:**
```
Extract return expression text
    ↓
Add process node: "Return [expression]"
    ↓
Connect frontier → process node
    ↓
Connect process node → "end"
    ↓
Return []  (execution terminates)
```

#### 7.3 Expression Statement Processing

**Function:** `_process_expression()`

**Flow:**
```
Extract function call (if exists)
    ↓
If no call: Return frontier unchanged
    ↓
Get callee function name
    ↓
Classify call:
    Call _classify_call(callee, detail_level)
    Returns: (include, semantic_label, category)
    ↓
If include == False:
    Return frontier unchanged (skip call)
    ↓
If include == True:
    Add process node: builder.add_process(semantic_label)
    Connect frontier → process node
    Return [process_node_id]
```

**Classification Logic:**
Based on `detail_level`:

| Detail Level | Includes | Excludes |
|--------------|----------|----------|
| **HIGH** | Major business (create, execute, handle, process)<br>Major state (create, delete, init, destroy) | Validations<br>Minor business<br>Critical sub-ops<br>State updates |
| **MEDIUM** | All business<br>All validations<br>All state changes | Critical sub-ops (get, fetch, read, load) |
| **DEEP** | Everything except utilities | Logging, metrics, utilities |

**Semantic Labeling:**
- Function name analyzed for verb patterns
- Extracted object from function name
- Collapsed to semantic action: "Create Volume", "Validate Input", etc.
- Function calls are NEVER expanded (always collapsed to one node)

#### 7.4 Declaration Processing

**Function:** `_process_declaration()`

**Flow:**
```
Extract declaration text
    ↓
Check if declaration should be included:
    Call _should_include_declaration(text, detail_level)
    ↓
If include:
    Add process node: "Declare [variable]"
    Connect frontier → process node
    Return [process_node_id]
    ↓
If exclude:
    Return frontier unchanged
```

**Inclusion Rules:**
- **HIGH**: Only configs/managers with initialization
- **MEDIUM**: Args, params, return values, errors
- **DEEP**: Also includes data structures, buffers, contexts

#### 7.5 Loop Processing

**Function:** `_process_loop()`

**Flow:**
```
Extract loop condition
    ↓
Add decision node: "Loop condition?"
    ↓
Connect frontier → decision node
    ↓
Process loop body:
    _process_block(body, builder, [decision], "YES")
    → body_frontier
    ↓
Connect body_frontier → decision node (loop back)
    ↓
Connect decision node → "NO" → next_frontier
    ↓
Return next_frontier
```

**Diagram:**
```
Frontier → Decision (condition?)
            ├─ YES → Body → (loop back)
            └─ NO  → next_frontier
```

---

### 8. SFM Building (`agent5/scenario_extractor.py`)

**Class:** `SFMBuilder`

**State:**
- `nodes`: List of SFMNode objects
- `edges`: List of SFMEdge objects
- `step_count`: Current step counter
- `max_steps`: Maximum allowed steps
- `detail_level`: Detail level enum

**Methods:**
- `add_process(label)`: Create process node
- `add_decision(label)`: Create decision node
- `add_io(label)`: Create I/O node
- `add_edge(src, dst, label)`: Create edge
- `build()`: Construct and validate ScenarioFlowModel

**Build Flow:**
```
Create ScenarioFlowModel(nodes, edges, metadata)
    ↓
Call sfm.validate()
    ↓
Validation checks:
    - Exactly 1 node with id="start"
    - ≥1 node with id="end"
    - All edges reference valid node IDs
    - No orphaned nodes (except start/end)
    ↓
If validation fails:
    Raise RuntimeError
    ↓
Return ScenarioFlowModel
```

---

### 9. Mermaid Translation (`agent5/flowchart.py`)

#### 9.1 Deterministic Translation

**Function:** `_sfm_to_mermaid()`

**Flow:**
```
Initialize: lines = ["flowchart TD"]
    ↓
Create ID remap: {"start": "startNode", "end": "endNode"}
    ↓
For each SFM node:
    nid = remap.get(node.id, node.id)
    label = sanitize_mermaid_label(node.label)
    ↓
    Switch node_type:
        ├─ "terminator" → nid([label])
        ├─ "decision"   → nid{label?}  (ensure ? suffix)
        ├─ "io"         → nid[/label/]
        └─ "process"    → nid[label]
    ↓
    Append node declaration line
    ↓
For each SFM edge:
    src = remap.get(edge.src, edge.src)
    dst = remap.get(edge.dst, edge.dst)
    ↓
    If edge.label:
        Append: src -- label --> dst
    Else:
        Append: src --> dst
    ↓
Return joined lines
```

**Mermaid Shape Mapping:**
- `terminator`: Start/End nodes → `id([Label])`
- `process`: Actions → `id[Label]`
- `decision`: Conditions → `id{Label?}`
- `io`: I/O operations → `id[/Label/]`

#### 9.2 LLM Translation (Optional)

**Function:** `_translate_sfm_with_llm()`

**Flow:**
```
Serialize SFM to JSON
    ↓
Create LLM prompt:
    SystemMessage: Translator instructions
    HumanMessage: SFM JSON + "Translate to Mermaid"
    ↓
Invoke LLM
    ↓
Extract Mermaid code from response:
    Call _extract_mermaid(response)
        - Remove markdown fences
        - Find "flowchart" start
        - Validate basic structure
    ↓
If valid Mermaid:
    Return mermaid code
    ↓
If invalid or missing:
    Return None (fallback to deterministic)
```

**LLM Role:** Translator ONLY. LLM does NOT infer logic or modify structure. SFM is authoritative.

---

### 10. File Output (`agent5/flowchart.py`)

**Function:** `write_flowchart()`

**Flow:**
```
Call generate_flowchart_from_file()
    → MermaidFlowchart
    ↓
Create output directory if needed
    ↓
Write mermaid code to .mmd file
    ↓
If SFM exists:
    Write SFM JSON to .sfm.json file (debug)
    ↓
Return MermaidFlowchart
```

---

## Data Flow: Scenario Flow Model (SFM)

### SFM Structure

```json
{
  "nodes": [
    {
      "id": "start",
      "type": "terminator",
      "label": "Start"
    },
    {
      "id": "node1",
      "type": "process",
      "label": "Parse arguments"
    },
    {
      "id": "node2",
      "type": "decision",
      "label": "Is valid?"
    },
    {
      "id": "end",
      "type": "terminator",
      "label": "End"
    }
  ],
  "edges": [
    {"src": "start", "dst": "node1"},
    {"src": "node1", "dst": "node2"},
    {"src": "node2", "dst": "node3", "label": "YES"},
    {"src": "node2", "dst": "end", "label": "NO"}
  ]
}
```

### Node Types

- **terminator**: Start/End nodes
- **process**: Action steps
- **decision**: Conditional branches
- **io**: Input/Output operations

### Edge Labels

- Edges may have labels (e.g., "YES", "NO", "Success")
- Labels used for decision branches and conditional flows

---

## Control Flow Handling

### Sequential Execution

```
Statement 1 → Statement 2 → Statement 3
    ↓            ↓            ↓
Frontier:    ["n1"]      ["n2"]      ["n3"]
```

### Conditional (If-Else)

```
Frontier: ["prev"]
    ↓
Decision node
    ├─ YES → Process block → ["then_node"]
    └─ NO  → Process block → ["else_node"]
    ↓
Merged frontier: ["then_node", "else_node"]
```

### Loops

```
Frontier: ["prev"]
    ↓
Decision: "condition?"
    ├─ YES → Loop body → (loop back to decision)
    └─ NO  → ["next"]
    ↓
Frontier: ["next"]
```

### Returns

```
Frontier: ["prev"]
    ↓
Return node
    ↓
Connect to "end"
    ↓
Frontier: [] (terminated)
```

---

## Detail Level Impact

### HIGH Level

**Filtering:**
- Only major business operations included
- Validations excluded
- Minor operations excluded
- Critical sub-operations excluded

**Result:** Minimal flowchart (5-7 nodes typical)

### MEDIUM Level (Default)

**Filtering:**
- All business operations included
- All validations included
- All state changes included
- Critical sub-operations excluded

**Result:** Complete documentation-quality flowchart (10-15 nodes typical)

### DEEP Level

**Filtering:**
- Everything included except utilities
- Critical sub-operations (get, fetch, read, load) included
- Internal validation steps shown
- Still excludes: logging, metrics, utilities

**Result:** Detailed flowchart (15-30+ nodes typical)

**Important:** Detail level affects SFM structure, not just labels. Different levels produce structurally different flowcharts.

---

## Error Handling

### Fail-Fast Philosophy

1. **Entry Function Not Found:**
   - Raises RuntimeError
   - Lists available functions
   - Process terminates

2. **SFM Validation Failure:**
   - Exactly 1 start node required
   - ≥1 end node required
   - All edges must reference valid nodes
   - No orphaned nodes allowed
   - Raises RuntimeError on failure

3. **LLM Translation Failure:**
   - Silently falls back to deterministic translation
   - No error raised (LLM is optional)

4. **File Read Errors:**
   - Raises RuntimeError with error details

---

## Key Design Principles

1. **Deterministic First:** Scenario extraction is rule-based, not LLM-based
2. **Fail Fast:** If SFM cannot be built, refuse to proceed
3. **Semantic Actions:** Function calls collapsed to single semantic steps
4. **Boundary Rules:** Include only scenario-relevant nodes
5. **Validated Models:** SFM must be valid before translation
6. **LLM as Translator:** LLM never infers logic, only translates SFM to Mermaid

---

## Summary: Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ 1. CLI: Parse arguments                                 │
│    - file_path (entry locator)                          │
│    - function_name (optional)                           │
│    - detail_level (high|medium|deep)                    │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Read C++ source file                                 │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Parse AST (tree-sitter)                              │
│    - Parse source_code → AST                            │
│    - Get root node                                      │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Find entry function                                  │
│    - Recursive DFS search                               │
│    - Case-insensitive, partial matching                 │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Extract scenario flow                                │
│    - Create SFMBuilder                                  │
│    - Process function body block                        │
│    - For each statement:                                │
│      * If → decision node, split branches               │
│      * Return → process node → end                      │
│      * Call → classify → add process node (if included) │
│      * Loop → decision node, cycle back                 │
│    - Connect frontier → end                             │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Build SFM                                            │
│    - Collect nodes and edges                            │
│    - Validate: 1 start, ≥1 end, valid edges            │
│    - If invalid → Raise RuntimeError                    │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 7. Translate to Mermaid                                 │
│    If use_llm:                                          │
│      Try LLM translation                                │
│      If fails → fallback                                │
│    Deterministic translation:                           │
│      - Map SFM nodes → Mermaid shapes                   │
│      - Map SFM edges → Mermaid arrows                   │
└─────────────────┬───────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────┐
│ 8. Write output                                         │
│    - Write .mmd file                                    │
│    - Write .sfm.json (debug)                            │
└─────────────────────────────────────────────────────────┘
```

---

**End of Document**

