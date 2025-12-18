# Agent5 V4 Architecture

## DocAgent-Inspired Bottom-Up Understanding Strategy

Version 4 introduces a revolutionary **6-stage pipeline** that builds semantic understanding from the bottom up, inspired by Meta's [DocAgent](https://github.com/facebookresearch/DocAgent) approach to code understanding.

---

## Design Philosophy

### Core Principles

1. **Bottom-Up Understanding** ✅  
   Build understanding from leaf functions upward through the call graph

2. **Scenario-Based Visualization** ✅  
   Present flowcharts at a high-level, scenario-driven abstraction  
   **NOT** function-call diagrams

3. **Strict Separation of Concerns** ✅  
   - **Deterministic Analysis**: AST, CFG, semantic extraction (NO LLM)
   - **LLM-Assisted Aggregation**: Semantic summarization only (NO logic inference)
   - **Strict Translation**: Mermaid generation (NO diagram logic)

4. **Single Source of Truth** ✅  
   Scenario Flow Model (SFM) is the authoritative representation  
   All diagrams are derived from SFM

### Critical Constraints

❌ **NEVER**: Function-call diagrams  
❌ **NEVER**: Recursive visual expansion  
❌ **NEVER**: LLM creativity in logic  
✅ **ALWAYS**: Bottom-up understanding  
✅ **ALWAYS**: Backtracking aggregation  
✅ **ALWAYS**: SFM is authoritative

---

## Pipeline Stages

### Stage 1: Full AST Construction (NO LLM)

**Purpose**: Parse entire C++ project deterministically

**Technology**:
- Primary: Clang AST (`clang -Xclang -ast-dump=json`)
- Fallback: Regex-based parsing (when Clang unavailable)

**Outputs**:
- Translation units (AST JSON per file)
- Control Flow Graphs (CFG) per function
- Call graph (function relationships)
- Leaf functions (functions that call no other functions)
- Entry point candidates

**Key Operations**:
```
C++ Project
    ↓
Find all .cpp/.h files
    ↓
Parse each file → AST JSON
    ↓
Extract functions → CFG per function
    ↓
Build call graph
    ↓
Identify: leaves, entry points, basic blocks
```

**Data Structures**:
```python
BasicBlock:
  - statements: List[str]
  - has_guard: bool
  - has_state_mutation: bool
  - has_error_exit: bool
  - successors/predecessors

FunctionCFG:
  - function_name: str
  - entry_block: str
  - exit_blocks: List[str]
  - basic_blocks: Dict[str, BasicBlock]
  - call_sites: List[Dict]

CallGraphNode:
  - function_name: str
  - callers: List[str]
  - callees: List[str]
  - is_leaf: bool
```

---

### Stage 2: Leaf-Level Semantic Extraction (BOTTOM LEVEL)

**Purpose**: Identify atomic semantic actions at deepest AST/CFG level

**Technology**: Rule-based classification (NO LLM)

**Semantic Action Types**:
1. **Validation**: Input checks, constraints
2. **Permission Check**: Authorization, access control
3. **State Mutation**: Variable/object modification
4. **Irreversible Side Effect**: Commits, deletions, sends
5. **Early Exit**: Returns, throws, aborts
6. **Computation**: Pure calculations
7. **Logging**: Debug/info/error logs *(excluded)*
8. **Metrics**: Counters, timers *(excluded)*
9. **Utility**: Helper functions *(excluded)*

**Classification Rules**:
```python
VALIDATION_KEYWORDS = {
    'validate', 'check', 'verify', 'ensure', 'assert',
    'is_valid', 'isEmpty', 'null', 'nullptr'
}

PERMISSION_KEYWORDS = {
    'auth', 'permission', 'allow', 'deny', 'access'
}

STATE_MUTATION_KEYWORDS = {
    'set', 'update', 'modify', 'delete', 'create'
}

IRREVERSIBLE_KEYWORDS = {
    'commit', 'finalize', 'publish', 'execute', 'persist'
}
```

**Output**:
```python
SemanticAction:
  - type: SemanticActionType
  - effect: str  # "Validate pointer is not null"
  - control_impact: bool
  - state_impact: bool
  - block_id: str
  - statements: List[str]
```

**Example**:
```cpp
if (!volume_id.empty()) {
  // Extracted semantic action:
  {
    "type": "validation",
    "effect": "Validate volume_id is not empty",
    "control_impact": true,
    "state_impact": false
  }
}
```

---

### Stage 3: Bottom-Up Backtracking & Semantic Aggregation (LLM-ASSISTED)

**Purpose**: Generate semantic summaries by moving up the call graph

**Technology**: LLM (Ollama) with strict prompts

**Process**:

1. **Start from Leaf Functions**
   - Leaf = functions that call no other functions
   - Generate local semantic summary from extracted actions

2. **Move Upward**
   - Process functions in reverse topological order
   - Each parent function receives child summaries

3. **Combine Child Summaries**
   - LLM aggregates child semantics into parent semantic description
   - Elide non-critical operations (logging, metrics)
   - Preserve control-flow and state semantics

4. **Continue to Entry**
   - Backtrack until reaching entry function
   - Entry summary = complete scenario understanding

**LLM Prompt Structure**:
```
You are analyzing C++ code to generate a semantic summary.

EXTRACTED SEMANTIC ACTIONS:
1. validation: Validate pointer is not null
2. state_mutation: Set volume state to "creating"

CALLED FUNCTIONS:
- create_volume_metadata: Creates volume metadata record
- allocate_storage: Allocates storage on backend

TASK: Generate structured semantic summary

CRITICAL RULES:
1. Based ONLY on provided semantic actions + child summaries
2. Do NOT infer logic not in AST
3. Summarize function calls, do NOT expand
4. Elide logging/metrics
5. Preserve control-flow and state semantics

OUTPUT (JSON):
{
  "summary": "...",
  "preconditions": [...],
  "postconditions": [...],
  "side_effects": [...],
  "control_flow": [...],
  "error_conditions": [...]
}
```

**Aggregation Rules**:
- ✅ Function calls are **SUMMARIZED**, not expanded
- ✅ Aggregation is **SEMANTIC**, not structural
- ❌ NO new logic introduced by LLM
- ❌ NO hallucination or creativity

**Output**:
```python
SemanticSummary:
  - function_name: str
  - summary: str  # High-level description
  - preconditions: List[str]
  - postconditions: List[str]
  - side_effects: List[str]
  - control_flow: List[str]
  - error_conditions: List[str]
  - child_summaries: Dict[str, str]
  - aggregation_level: int  # 0=leaf, increases upward
```

---

### Stage 4: Scenario Flow Model Construction (SINGLE SOURCE OF TRUTH)

**Purpose**: Convert aggregated semantics into deterministic SFM

**Technology**: Rule-based conversion (NO LLM)

**Scenario Flow Model (SFM)**:
```python
ScenarioStep:
  - step_id: str  # "S1", "S2", ...
  - step_type: StepType  # START, END, DECISION, ACTION, ...
  - label: str  # "Validate input"
  - description: str  # Full description
  - detail_levels: List[DetailLevel]  # [HIGH, MEDIUM, DEEP]
  - on_success: Optional[str]  # Next step ID
  - on_failure: Optional[str]  # Error step ID
  - metadata: Dict[str, Any]

ScenarioFlowModel:
  - scenario_name: str
  - entry_function: str
  - steps: Dict[str, ScenarioStep]
  - start_step: str
  - end_steps: List[str]
```

**Conversion Rules**:
- **Preconditions** → Validation steps (MEDIUM, DEEP)
- **Control flow** → Decision steps (HIGH if major, MEDIUM otherwise)
- **Side effects** → State change steps (DEEP if critical, MEDIUM otherwise)
- **Postconditions** → Action steps (HIGH, MEDIUM, DEEP)
- **Error conditions** → Error exit steps

**Validation**:
- ✅ Exactly 1 START step
- ✅ At least 1 END step
- ✅ All steps reachable from START
- ✅ All non-END steps have successors

---

### Stage 5: Detail-Level Filtering (RULE-BASED)

**Purpose**: Filter SFM based on desired granularity

**Technology**: Rule-based filtering (NO LLM)

**Detail Levels**:

#### HIGH: Business-Level Steps Only
- Only major business decisions
- Minimal detail
- Suitable for architecture overview

**Includes**:
- ✅ START/END
- ✅ Major decisions (business-critical)
- ✅ Final actions/results

**Excludes**:
- ❌ Validations
- ❌ State changes
- ❌ Internal sub-operations

#### MEDIUM (Default): Decisions + Validations + State Changes
- All decision points
- All validations
- All state-changing operations
- Suitable for documentation

**Includes**:
- ✅ START/END
- ✅ All decisions
- ✅ All validations
- ✅ All state changes
- ✅ Actions

**Excludes**:
- ❌ Internal sub-operations (unless critical)

#### DEEP: Expanded Critical Sub-Operations
- Everything in MEDIUM
- Plus: critical sub-operations that affect control flow or persistent state

**Includes**:
- ✅ Everything in MEDIUM
- ✅ Critical sub-operations
- ✅ Expanded validation logic
- ✅ Internal state transitions

**Never Expands**:
- ❌ Logging
- ❌ Metrics
- ❌ Utility helpers
- ❌ Memory allocation wrappers
- ❌ Serialization helpers

**Filtering Algorithm**:
1. Select steps matching detail level
2. Re-link steps to maintain flow (skip filtered steps)
3. Validate filtered SFM

---

### Stage 6: Mermaid Translation (LLM STRICT TRANSLATOR)

**Purpose**: Translate SFM to Mermaid flowchart syntax

**Technology**: 
- Primary: LLM-assisted (better labels/formatting)
- Fallback: Deterministic rule-based

**Critical Rules**:
- ✅ Input: SFM only (single source of truth)
- ✅ Output: Mermaid syntax only
- ❌ NO logic changes
- ❌ NO depth changes
- ❌ NO inference

**LLM Prompt**:
```
You are a strict translator: SFM → Mermaid.

SFM (JSON):
{ ... }

RULES:
1. Preserve EXACT structure
2. Do NOT add/remove steps
3. Do NOT change logic
4. Use appropriate Mermaid shapes:
   - START/END: Stadium ([" "])
   - DECISION/VALIDATION: Diamond {" "}
   - ACTION/STATE_CHANGE: Rectangle [" "]

OUTPUT: Mermaid code only
```

**Mermaid Node Shapes**:
- **START/END**: `([Label])`
- **DECISION**: `{Label}`
- **VALIDATION**: `{Label}`
- **ACTION**: `[Label]`
- **STATE_CHANGE**: `[Label]`
- **ERROR**: `((Label))`

**Validation**:
- Check all SFM step IDs are present in Mermaid
- Check no extra steps introduced
- Fallback to deterministic if validation fails

---

## Entry-Point Disambiguation

**CLI Parameters**:
```bash
--entry-function <function_name>
--entry-file <file_path>  # Optional but recommended
```

**Resolution Rules**:

1. **Both function + file provided**:
   - Strict resolution
   - Use exact match

2. **Function only**:
   - Search entire project
   - If exactly 1 match → use it
   - If multiple matches → **ERROR** (ask user to provide `--entry-file`)
   - If no matches → **ERROR** (list available functions)

3. **Neither provided**:
   - Auto-detect entry points using AST evidence:
     - Functions with no callers
     - Functions named `main`
     - Functions with specific patterns (e.g., `Handle*`, `Execute*`)

**Important**: 
- `--entry-file` is for **disambiguation only**
- `--project-path` defines the **analysis scope** (entire project)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    C++ Project Directory                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Full AST Construction (NO LLM)                    │
│  - Parse all .cpp/.h files with Clang                       │
│  - Build CFG per function                                   │
│  - Construct call graph                                     │
│  - Identify leaf functions & entry points                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   AST Parse Result   │
          │  - Translation Units │
          │  - Function CFGs     │
          │  - Call Graph        │
          │  - Leaf Functions    │
          └──────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Leaf-Level Semantic Extraction (NO LLM)           │
│  - Classify statements: validation, mutation, exit, etc.    │
│  - Extract semantic actions per basic block                 │
│  - Rule-based classification only                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Function Semantics  │
          │  (Leaf-Level)        │
          │  - Semantic Actions  │
          └──────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Bottom-Up Aggregation (LLM-ASSISTED)              │
│  - Start from leaf functions                                │
│  - Generate semantic summaries using LLM                    │
│  - Move upward in call graph                                │
│  - Combine child summaries into parent summaries            │
│  - Continue until entry function                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Aggregated Semantics │
          │  - Entry Summary     │
          │  - All Summaries     │
          └──────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Scenario Flow Model Construction (NO LLM)         │
│  - Convert semantics → SFM steps                            │
│  - Assign detail levels per step                            │
│  - Link steps (on_success, on_failure)                      │
│  - Validate SFM structure                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Scenario Flow Model  │
          │  (SFM)               │
          │  - All Steps         │
          │  - Detail Levels     │
          └──────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: Detail-Level Filtering (NO LLM)                   │
│  - Filter steps by detail level (high/medium/deep)          │
│  - Re-link filtered steps                                   │
│  - Validate filtered SFM                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Filtered SFM       │
          └──────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: Mermaid Translation (LLM STRICT TRANSLATOR)       │
│  - Translate SFM → Mermaid syntax                           │
│  - Choose node shapes based on step type                    │
│  - Validate structure preservation                          │
│  - Fallback to deterministic if LLM fails                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Mermaid Flowchart   │
          │      (.mmd file)     │
          └──────────────────────┘
```

---

## Usage Examples

### Basic Usage (V4 Pipeline)

```bash
python -m agent5 flowchart \
  --file path/to/file.cpp \
  --function HandleVolumeCreate \
  --project-path /path/to/project \
  --detail-level medium \
  --use_v4 \
  --out output.mmd
```

### Auto-Detect Entry Point

```bash
python -m agent5 flowchart \
  --file path/to/file.cpp \
  --project-path /path/to/project \
  --use_v4 \
  --out output.mmd
```

### High-Level Business Overview

```bash
python -m agent5 flowchart \
  --file path/to/file.cpp \
  --function ProcessRequest \
  --project-path /path/to/project \
  --detail-level high \
  --use_v4 \
  --out business_flow.mmd
```

### Deep Technical Analysis

```bash
python -m agent5 flowchart \
  --file path/to/file.cpp \
  --function AuthenticateUser \
  --project-path /path/to/project \
  --detail-level deep \
  --use_v4 \
  --out detailed_flow.mmd
```

---

## Comparison: V3 vs V4

| Aspect | V3 Pipeline | V4 Pipeline |
|--------|-------------|-------------|
| **Understanding** | Top-down (from entry) | Bottom-up (from leaves) |
| **AST Analysis** | Tree-sitter | Clang AST + CFG |
| **Semantic Extraction** | Single-pass | Leaf-level + aggregation |
| **LLM Role** | Optional translation | Semantic aggregation + translation |
| **Call Graph** | Not explicitly built | Explicit call graph analysis |
| **Aggregation** | None | Multi-level bottom-up |
| **SFM Detail Levels** | Assigned during extraction | Assigned during construction |
| **Reliability** | Good | Excellent (DocAgent-inspired) |
| **Complexity** | Moderate | High |
| **Best For** | Quick flowcharts | Documentation-quality flowcharts |

---

## Key Advantages of V4

1. **Deeper Understanding**  
   Bottom-up analysis provides more accurate semantic understanding

2. **Better Abstraction**  
   Semantic aggregation produces higher-quality scenario descriptions

3. **Explicit Call Graph**  
   Understands function relationships for better context

4. **Leaf-First Processing**  
   Ensures foundational understanding before aggregation

5. **LLM-Assisted Summarization**  
   Uses LLM where it excels (semantic summarization) while avoiding where it fails (logic inference)

6. **Rigorous Validation**  
   Multiple validation points ensure SFM correctness

7. **Deterministic Fallbacks**  
   Every LLM-assisted stage has deterministic fallback

---

## Technical Requirements

### Dependencies
```
clang (optional, for AST parsing)
Python 3.10+
langchain
ollama (for LLM)
chromadb (for RAG)
tree-sitter-cpp (fallback parsing)
```

### Recommended Hardware
- **RAM**: 8GB+ (16GB recommended for large projects)
- **CPU**: Multi-core recommended (parallel AST parsing)
- **Storage**: SSD recommended for large codebases

### LLM Requirements
- Ollama running locally or remotely
- Recommended models: `codellama:13b`, `deepseek-coder:6.7b`
- Minimum: 7B parameters for good semantic aggregation

---

## Limitations & Future Work

### Current Limitations
1. **Clang dependency**: Falls back to regex if unavailable
2. **Single entry point**: Cannot yet analyze multiple entry points simultaneously
3. **No inter-procedural analysis**: Limited to call graph traversal
4. **LLM latency**: Aggregation can be slow for large projects

### Future Enhancements
1. **Parallel aggregation**: Process independent call graph branches in parallel
2. **Caching**: Cache semantic summaries across runs
3. **Incremental analysis**: Only re-analyze changed functions
4. **Multi-language support**: Extend beyond C++
5. **Interactive refinement**: Allow user to refine SFM before diagram generation

---

## References

- [DocAgent (Meta Research)](https://github.com/facebookresearch/DocAgent) - Inspiration for bottom-up understanding
- [Clang AST Documentation](https://clang.llvm.org/docs/IntroductionToTheClangAST.html)
- [Control Flow Graph (Wikipedia)](https://en.wikipedia.org/wiki/Control-flow_graph)
- [Mermaid Flowchart Syntax](https://mermaid.js.org/syntax/flowchart.html)

---

**Version**: 4.0  
**Last Updated**: December 2025  
**Maintainer**: Agent5 Development Team

