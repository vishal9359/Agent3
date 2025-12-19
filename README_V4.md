# Agent5 V4: DocAgent-Inspired Bottom-Up Semantic Aggregation

**Version 4.0 - Major Architecture Overhaul**

Agent5 V4 represents a fundamental reimagining of how we analyze and document C++ code. Inspired by Facebook Research's DocAgent, V4 adopts a **bottom-up semantic aggregation** approach that provides deep, documentation-quality understanding of complex C++ projects while maintaining scenario-based, non-function-call flowcharts.

---

## 🎯 What's New in V4

### Core Design Principle

> **Understanding flows bottom-up, presentation remains scenario-based.**

V4 separates **understanding** (how we analyze code) from **presentation** (how we visualize it):

- **Understanding**: Bottom-up semantic aggregation from leaf functions to entry points
- **Presentation**: High-level scenario flowcharts with configurable detail levels

### Key Improvements

1. **Clang-Based AST + CFG Analysis** (Stage 1)
   - Full Control Flow Graphs for every function
   - Precise call graph extraction
   - Identification of leaf-level execution units, guards, state mutations, and error exits

2. **Leaf-Level Semantic Extraction** (Stage 2)
   - Deterministic, rule-based classification of atomic semantic actions
   - Types: validation, permission check, state mutation, side effect, early exit
   - NO LLM inference at this stage

3. **Bottom-Up Semantic Aggregation** (Stage 3)
   - LLM-assisted summarization starting from leaf functions
   - Backtracking upward through call graph
   - Child summaries aggregated into parent summaries
   - Non-critical operations elided automatically

4. **Scenario Flow Model (SFM)** (Stage 4)
   - Deterministic, JSON-based representation
   - Single source of truth for diagram generation
   - Explicit mapping to detail levels

5. **Rule-Based Filtering** (Stage 5)
   - Filter AFTER aggregation, not during
   - Three detail levels: HIGH, MEDIUM, DEEP
   - Consistent, predictable results

6. **Strict Mermaid Translation** (Stage 6)
   - LLM used ONLY as syntax translator
   - No logic inference allowed
   - Fallback to deterministic translation available

---

## 📋 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Full AST Construction (NO LLM)                    │
│  ├─ Parse entire C++ project with Clang                     │
│  ├─ Build CFG for each function                             │
│  ├─ Extract call graph relationships                        │
│  └─ Identify leaf functions, guards, state mutations        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Leaf-Level Semantic Extraction (RULE-BASED)       │
│  ├─ Classify atomic semantic actions                        │
│  ├─ Types: validation, permission, mutation, side effect    │
│  ├─ Extract control/state impact                            │
│  └─ NO LLM, NO hierarchy                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Bottom-Up Aggregation (LLM-ASSISTED)              │
│  ├─ Start from leaf functions                               │
│  ├─ Generate semantic summaries using LLM                   │
│  ├─ Move upward in call graph                               │
│  ├─ Combine child summaries into parent                     │
│  └─ Preserve control flow + state semantics                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Scenario Flow Model Construction                  │
│  ├─ Convert aggregated semantics to SFM (JSON)              │
│  ├─ One SFM per entry-point scenario                        │
│  ├─ Map nodes to detail levels (high/medium/deep)           │
│  └─ Single source of truth                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: Detail-Level Filtering (RULE-BASED)               │
│  ├─ Filter nodes based on --detail-level                    │
│  ├─ HIGH: Business-level steps only                         │
│  ├─ MEDIUM: + validations, decisions, state changes         │
│  └─ DEEP: + critical sub-operations                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: Mermaid Translation (LLM STRICT TRANSLATOR)       │
│  ├─ Input: Filtered SFM (JSON)                              │
│  ├─ Output: Mermaid flowchart syntax                        │
│  ├─ LLM translates ONLY, no logic changes                   │
│  └─ Fallback to deterministic translation available         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   📊 Mermaid Flowchart
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with required model
ollama pull llama3.2:3b
```

### Basic Usage

```bash
# Generate medium-detail flowchart
agent5-v4 flowchart \
    --project-path /path/to/cpp/project \
    --entry-function processRequest \
    --entry-file src/server.cpp \
    --detail-level medium \
    --output flowchart.mmd
```

### Project-Bounded Analysis (AST Scope)

All V4 analysis is **hard-bounded to your project root**:

- Only files **under the project root** are parsed for AST/CFG/semantics.
- System headers (`/usr/include`, `/usr/local/include`, toolchain headers) and
  third‑party/external code (`external/`, `third_party/`, `build/`, `out/`, `.cache/`, `bazel-*`) are **never expanded**.
- Imported symbols from libraries (e.g., `std::vector`, `std::sort`) may appear as types or call names,
  but their internal definitions are treated as **opaque** – no AST traversal, no semantics, no SFM nodes.
- Leaf detection and bottom‑up aggregation consider **only project-defined functions**; calls that cannot be
  resolved to project functions are treated as external actions and are not backtracked into.

The effective project boundary is:

- For the V4 CLI (`agent5-v4`): the `--project-path` / `--project-root` directory.
- For library usage: the `project_path` / `project_root` you pass into the pipeline/analyzers.

If the configured project root is invalid (does not exist or is not a directory), the tools **fail loudly**
instead of silently degrading, ensuring deterministic and predictable behavior.

### Usage Options

```bash
agent5-v4 flowchart --help
```

**Required:**
- `--project-path`: Root path of C++ project
- `--entry-function`: Name of entry function

**Optional:**
- `--entry-file`: File path to disambiguate entry function (recommended)
- `--detail-level`: `high`, `medium` (default), or `deep`
- `--scenario-name`: Custom name for the scenario
- `--output, -o`: Output file (default: stdout)
- `--output-dir`: Save intermediate outputs (SFM, summaries, etc.)
- `--model`: Ollama model (default: `llama3.2:3b`)
- `--no-llm-translator`: Use rule-based Mermaid translation

---

## 📊 Detail Levels Explained

### HIGH: Business-Level Overview

**Includes ONLY:**
- Major business operations
- Top-level subprocess calls

**Excludes:**
- Validations
- State changes
- Error handling
- All internal logic

**Use Case:** Architecture overview, executive summaries

**Example:**
```
Start → Process Request → Send Response → End
```

---

### MEDIUM: Documentation Quality (Default)

**Includes:**
- All business operations (from HIGH)
- Validations and preconditions
- Decision points
- State-changing operations
- Error handling

**Excludes:**
- Internal implementation details
- Utility helpers
- Logging/metrics

**Use Case:** Technical documentation, code reviews

**Example:**
```
Start → Validate Input → Check Permissions → 
Update Database → Send Notification → End
         ↓ (invalid)
      Reject Request
```

---

### DEEP: Implementation Detail

**Includes:**
- Everything from MEDIUM
- Critical sub-operations that affect control flow
- Data access operations (lookups, reads, loads)
- Error condition checks
- Postconditions

**Excludes (still):**
- Logging
- Metrics
- Trivial utilities
- Memory management wrappers

**Use Case:** Debugging, deep code analysis

**Example:**
```
Start → Parse Input → Validate Schema → 
Lookup User → Check Role → Verify Token →
Load Data → Transform → Save → Commit → End
```

---

## 🎯 Entry-Point Disambiguation

V4 provides flexible entry-point resolution:

### Both `--entry-function` and `--entry-file` (Recommended)

```bash
agent5-v4 flowchart \
    --project-path ./myproject \
    --entry-function handleRequest \
    --entry-file src/server.cpp
```

**Behavior:** Strict resolution. If function not in file, error.

### Only `--entry-function`

```bash
agent5-v4 flowchart \
    --project-path ./myproject \
    --entry-function handleRequest
```

**Behavior:**
- If exactly 1 match: Use it
- If 0 matches: Error with list of available functions
- If >1 matches: Error with disambiguation instructions

### Auto-Detection (Future)

Not yet implemented in V4. Requires heuristics for scenario entry points.

---

## 📁 Output Options

### Standard Output (Default)

```bash
agent5-v4 flowchart --project-path ./project --entry-function main
```

Prints Mermaid code to stdout.

### Save to File

```bash
agent5-v4 flowchart \
    --project-path ./project \
    --entry-function main \
    --output flowchart.mmd
```

### Save Intermediate Outputs

```bash
agent5-v4 flowchart \
    --project-path ./project \
    --entry-function main \
    --output-dir ./analysis \
    --output flowchart.mmd
```

Creates:
- `semantic_summaries.json`: All function semantic summaries
- `scenario_flow_model.json`: Full SFM before filtering
- `sfm_filtered_medium.json`: Filtered SFM at requested detail level
- `flowchart_medium.mmd`: Final Mermaid code

---

## ⚙️ Advanced Usage

### Analyze Entire Project

Generate semantic summaries for all functions (no flowchart):

```bash
agent5-v4 analyze \
    --project-path ./myproject \
    --output-dir ./analysis
```

Useful for:
- Understanding codebase structure
- Identifying critical functions
- Pre-computing summaries for multiple flowcharts

### Use Different LLM Model

```bash
agent5-v4 flowchart \
    --project-path ./project \
    --entry-function main \
    --model llama3.1:70b  # More powerful model
```

### Disable LLM Translator (Faster)

```bash
agent5-v4 flowchart \
    --project-path ./project \
    --entry-function main \
    --no-llm-translator  # Use deterministic rule-based translation
```

---

## 🔍 What Gets Expanded/Elided

### Always Included (All Levels)

- Start and End nodes

### HIGH Level Only

- Major business operations
- Top-level subprocess calls

### MEDIUM Level Adds

- Validations (input checks, schema validation)
- Permissions/authorization checks
- State mutations (database updates, object modifications)
- Decision points (branching logic)
- Error handling (try/catch, error returns)

### DEEP Level Adds

- Data access (lookups, reads, queries)
- Critical sub-operations (parsing, transformation)
- Error condition checks
- Postcondition validation

### NEVER Expanded (All Levels)

- Logging statements
- Metrics/monitoring calls
- Utility helpers (formatters, converters)
- Memory allocation wrappers
- Serialization/deserialization (unless critical)

---

## 🛠️ Troubleshooting

### Error: "No function named 'X' found"

**Cause:** Function doesn't exist or project not analyzed correctly.

**Solution:**
1. Check function name spelling
2. Ensure project path is correct
3. Verify C++ files use supported extensions (.cpp, .cc, .cxx, .c++)

### Error: "Ambiguous function name"

**Cause:** Multiple functions with same name exist.

**Solution:**
Use `--entry-file` to disambiguate:

```bash
agent5-v4 flowchart \
    --project-path ./project \
    --entry-function process \
    --entry-file src/handler.cpp  # Specify which file
```

### Clang Parse Errors

**Cause:** Project has complex build requirements.

**Solution:**
V4 uses basic C++17 parsing flags. For complex projects:
1. Ensure standard includes are available
2. Consider generating a `compile_commands.json` (future support)

### LLM Timeouts

**Cause:** Model is slow or unavailable.

**Solutions:**
1. Use smaller model: `--model llama3.2:3b`
2. Disable LLM translator: `--no-llm-translator`
3. Check Ollama is running: `ollama list`

---

## 📚 Examples

### Example 1: High-Level Architecture Overview

```bash
agent5-v4 flowchart \
    --project-path ./trading-system \
    --entry-function executeOrder \
    --entry-file src/engine.cpp \
    --detail-level high \
    --output architecture_overview.mmd
```

**Result:** 3-5 step business logic flowchart, perfect for presentations.

### Example 2: Medium-Detail Documentation

```bash
agent5-v4 flowchart \
    --project-path ./trading-system \
    --entry-function executeOrder \
    --entry-file src/engine.cpp \
    --detail-level medium \
    --output-dir ./docs/flowcharts \
    --output order_execution.mmd
```

**Result:** Complete documentation-quality flowchart with validations and state changes.

### Example 3: Deep Debugging Analysis

```bash
agent5-v4 flowchart \
    --project-path ./trading-system \
    --entry-function executeOrder \
    --entry-file src/engine.cpp \
    --detail-level deep \
    --output-dir ./debug \
    --output deep_analysis.mmd
```

**Result:** Detailed flowchart showing all critical operations, saved with intermediate outputs for inspection.

---

## 🔬 Technical Details

### Supported C++ Features

- Functions (free functions, member functions)
- Namespaces
- Classes and structs
- Control flow (if, while, for, switch)
- Function calls
- Return statements

### Limitations

- Template instantiation not fully supported
- Macro expansion limited
- Preprocessor conditionals may cause issues
- Requires relatively standard C++ code

### Performance

- Small projects (<100 files): ~30 seconds
- Medium projects (100-500 files): ~2-5 minutes
- Large projects (>500 files): May require batching

**Bottlenecks:**
- Clang AST parsing: Fast
- Semantic extraction: Fast
- LLM aggregation: Moderate (depends on model and number of functions)
- Mermaid translation: Fast

---

## 🤝 Comparison with V3

| Feature | V3 | V4 |
|---------|----|----|
| **Understanding** | Tree-sitter AST | Clang AST + CFG |
| **Analysis** | Top-down, LLM-heavy | Bottom-up, rule-based + LLM |
| **Call Graph** | Implicit | Explicit, analyzed |
| **Semantic Model** | None (direct to flowchart) | Hierarchical summaries |
| **Filtering** | During generation | After aggregation |
| **SFM** | JSON-based | JSON-based (enhanced) |
| **Detail Levels** | Often similar output | Structurally different |
| **LLM Role** | Analysis + translation | Summarization + translation only |
| **Cross-File** | Limited | Full project analysis |
| **Determinism** | Low (LLM-dependent) | High (rule-based stages) |

---

## 🎓 Concepts

### What is Bottom-Up Semantic Aggregation?

Instead of analyzing a function top-down (starting from entry, diving into calls), V4 starts at the **leaves** (functions that don't call others) and works **upward**:

1. Analyze leaf function: "This validates input"
2. Analyze parent: "This calls validator, then processes data"
3. Aggregate semantically: "Parent validates input, then processes data"

**Benefits:**
- Each function summarized once (cached)
- Deep understanding without deep expansion
- Natural abstraction hierarchy

### What is a Scenario Flow Model (SFM)?

A **deterministic, JSON-based representation** of a scenario's logic flow. Think of it as an intermediate representation (IR) for flowcharts.

**Why SFM?**
- Single source of truth
- Decouples analysis from presentation
- Enables validation and transformation
- Supports multiple output formats (Mermaid, PlantUML, etc.)

---

## 📖 References

- [DocAgent (Facebook Research)](https://github.com/facebookresearch/DocAgent) - Inspiration for bottom-up aggregation
- [Clang LibTooling](https://clang.llvm.org/docs/LibTooling.html) - AST and CFG analysis
- [Mermaid Flowcharts](https://mermaid.js.org/syntax/flowchart.html) - Output format

---

## 📝 License

[Insert License]

---

## 🐛 Known Issues

1. Template-heavy code may not parse correctly
2. Macro-heavy code may produce incomplete AST
3. Very large projects (>1000 functions) may require significant time
4. Complex C++ features (concepts, SFINAE) not fully supported

---

## 🚧 Roadmap

- [ ] Support for compile_commands.json
- [ ] Parallel processing for large projects
- [ ] Template instantiation analysis
- [ ] Multiple output formats (PlantUML, DOT)
- [ ] Interactive flowchart navigation
- [ ] Integration with documentation generators (Doxygen, Sphinx)

---

## 📞 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Agent5 V4** - Deep understanding, clear presentation.


