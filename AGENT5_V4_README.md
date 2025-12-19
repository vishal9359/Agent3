## Agent5 V4: DocAgent-Inspired Bottom-Up Semantic Aggregation

Welcome to Agent5 V4! This version implements a DocAgent-inspired pipeline for generating documentation-quality flowcharts from complex C++ projects.

---

### 🎯 Key Features

1. **Bottom-Up Understanding**: Builds semantic understanding from leaf functions upward
2. **Clang-Based AST Analysis**: Uses libclang for accurate C++ parsing
3. **Control Flow Graphs**: Extracts precise control flow, state mutations, and decision points
4. **Multi-Level Detail**: Generate flowcharts at `high`, `medium`, or `deep` detail levels
5. **Scenario-Based Output**: Produces human-readable scenario flows, NOT function-call diagrams

---

### 🏗️ Pipeline Architecture (6 Stages)

```
Stage 1: Full AST Construction (Clang)
   ↓
Stage 2: Leaf-Level Semantic Extraction
   ↓
Stage 3: Bottom-Up Semantic Aggregation (LLM-assisted)
   ↓
Stage 4: Scenario Flow Model Construction (SFM)
   ↓
Stage 5: Detail-Level Filtering
   ↓
Stage 6: Mermaid Translation (LLM strict translator)
```

---

### 📦 Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

**Critical dependency**: `libclang==18.1.1` (automatically installed)

---

### 🚀 Quick Start

#### Basic Usage

```bash
python -m agent5.cli_v4 \
  --project-path /path/to/cpp/project \
  --entry-function CreateVolume \
  --out flowchart.mmd
```

#### With Entry File Disambiguation

```bash
python -m agent5.cli_v4 \
  --project-path /path/to/cpp/project \
  --entry-function ProcessRequest \
  --entry-file src/api/handler.cpp \
  --out flowchart.mmd
```

#### Deep Detail Level

```bash
python -m agent5.cli_v4 \
  --project-path /path/to/cpp/project \
  --entry-function Login \
  --detail-level deep \
  --out flowchart.mmd
```

#### Debug Mode (Save Intermediate Artifacts)

```bash
python -m agent5.cli_v4 \
  --project-path /path/to/cpp/project \
  --entry-function Main \
  --detail-level medium \
  --debug \
  --out flowchart.mmd
```

This saves intermediate artifacts to `project_path/output/`:
- `call_graph.json` - Call graph of the project
- `cfgs.json` - Control flow graphs
- `semantics.json` - Leaf-level semantic actions
- `aggregated.json` - Aggregated semantics
- `sfm.json` - Raw Scenario Flow Model
- `sfm_medium.json` - Filtered SFM for detail level

---

### 🎚️ Detail Levels

| Level | Description | Includes |
|-------|-------------|----------|
| **high** | Business-level overview | Major business operations only |
| **medium** *(default)* | Documentation-quality | All validations, decisions, and state changes |
| **deep** | Implementation details | Expanded sub-operations affecting control/state |

**Never expanded** (at any level):
- Logging statements
- Metrics collection
- Utility helpers
- Memory allocators
- Serialization helpers

---

### 📋 Command-Line Options

```
Required Arguments:
  --project-path PATH       Path to C++ project (defines analysis scope)
  --out PATH               Output path for .mmd file

Entry Point Selection:
  --entry-function NAME    Entry function name (e.g., 'CreateVolume')
                          If omitted, auto-detects entry point
  --entry-file PATH        File containing entry function (for disambiguation)
                          NOTE: This does NOT limit analysis scope!

Flowchart Configuration:
  --detail-level LEVEL     Detail level: high, medium, deep (default: medium)

Clang Configuration:
  --include-paths PATHS    Comma-separated include paths for Clang
                          Example: '/usr/include,/usr/local/include'

LLM Configuration:
  --llm-model MODEL        Ollama LLM model (default: llama2:7b)
  --no-llm                Disable LLM (use deterministic fallbacks only)
  --ollama-base-url URL    Ollama base URL (default: http://localhost:11434)

Debugging:
  --debug                  Save intermediate artifacts to project_path/output/
```

---

### 🔍 Entry Point Resolution Rules

#### Case 1: Both --entry-function and --entry-file provided

```bash
--entry-function CreateVolume --entry-file src/volumes.cpp
```

**Result**: Strict resolution. Function must exist in the specified file.

#### Case 2: Only --entry-function provided

```bash
--entry-function ProcessRequest
```

**Result**:
- If **exactly one match** exists → use it
- If **multiple matches** exist → error with disambiguation instructions
- If **no match** exists → error

#### Case 3: Neither provided

```bash
--project-path /path/to/project
```

**Result**: Auto-detects entry point using:
1. Functions with no callers (entry points)
2. Functions named "main" or similar
3. First available entry point

---

### ⚠️ Critical Constraints

1. **Entry File ≠ Analysis Scope**
   - `--entry-file` is ONLY for disambiguating the entry function
   - `--project-path` ALWAYS defines the analysis scope
   - The agent analyzes the ENTIRE project, not just one file

2. **No Function-Call Diagrams**
   - The agent generates **scenario flows**, not call graphs
   - Function calls are *summarized*, not *expanded*

3. **No LLM Logic Inference**
   - LLM is used ONLY for:
     - Semantic summarization (Stage 3)
     - Mermaid translation (Stage 6)
   - LLM does NOT invent logic or control flow

4. **Scenario Flow Model is Authoritative**
   - SFM (Stage 4) is the single source of truth
   - All subsequent stages work from SFM

---

### 📝 Example Workflow

Let's say you have a C++ project for a storage system and want to document the "Create Volume" scenario:

```bash
# Step 1: Generate medium-detail flowchart
python -m agent5.cli_v4 \
  --project-path ~/projects/storage-system \
  --entry-function CreateVolume \
  --out docs/create_volume_medium.mmd

# Step 2: Generate high-level flowchart for architects
python -m agent5.cli_v4 \
  --project-path ~/projects/storage-system \
  --entry-function CreateVolume \
  --detail-level high \
  --out docs/create_volume_high.mmd

# Step 3: Generate deep flowchart for developers
python -m agent5.cli_v4 \
  --project-path ~/projects/storage-system \
  --entry-function CreateVolume \
  --detail-level deep \
  --debug \
  --out docs/create_volume_deep.mmd
```

Now you have:
- `create_volume_high.mmd` - For architecture presentations
- `create_volume_medium.mmd` - For documentation
- `create_volume_deep.mmd` - For code reviews
- `~/projects/storage-system/output/` - Debug artifacts

---

### 🐛 Troubleshooting

#### "Function 'X' not found in project"

**Solution**: Ensure the function exists and is not in a header-only library. Check spelling and case.

#### "Ambiguous function name 'X'"

**Solution**: Add `--entry-file` to specify which file contains your target function.

```bash
--entry-function Process --entry-file src/handler.cpp
```

#### "Failed to parse X"

**Solution**: Add include paths using `--include-paths`:

```bash
--include-paths '/usr/include,/usr/local/include,./external/libs'
```

#### LLM connection issues

**Solution**: Check Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

Or use `--ollama-base-url` to point to your Ollama instance.

Alternatively, disable LLM:

```bash
--no-llm
```

---

### 🔬 Understanding the Pipeline (Technical)

#### Stage 1: Full AST Construction

- Parses entire C++ project using Clang AST
- Builds call graph
- Identifies leaf functions and entry points
- **NO LLM inference**

#### Stage 2: Leaf-Level Semantic Extraction

- Extracts atomic semantic actions from each basic block:
  - Validations
  - Permission checks
  - State mutations
  - Irreversible side effects
  - Early exits
- **NO LLM inference**

#### Stage 3: Bottom-Up Semantic Aggregation

- Starts from leaf functions
- Uses LLM to generate semantic summaries based ONLY on extracted facts
- Moves upward in call graph, combining child summaries
- Preserves control flow and state semantics
- **LLM is strictly fact-based, NO creativity**

#### Stage 4: Scenario Flow Model Construction

- Converts aggregated semantics into a JSON-based SFM
- Maps actions to detail levels
- SFM is the **single source of truth**

#### Stage 5: Detail-Level Filtering

- Applies rule-based filtering to SFM
- Removes nodes/edges not relevant for requested detail level
- Reconnects edges around filtered nodes

#### Stage 6: Mermaid Translation

- Uses LLM as a **strict translator** from SFM to Mermaid
- No logic changes, no depth changes, no inference
- Deterministic fallback if LLM fails

---

### 📚 Further Reading

- [DocAgent Paper](https://github.com/facebookresearch/DocAgent) - Inspiration for bottom-up understanding
- [Clang AST Documentation](https://clang.llvm.org/docs/IntroductionToTheClangAST.html)
- [Mermaid Flowchart Syntax](https://mermaid.js.org/syntax/flowchart.html)

---

### 🤝 Contributing

Found a bug or have a feature request? Open an issue on GitHub!

---

### 📄 License

[Your License Here]

---

**Happy Documenting! 🚀**



