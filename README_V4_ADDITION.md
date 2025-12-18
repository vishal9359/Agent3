# Agent5 V4: DocAgent-Inspired Pipeline

**Add this section to the main README.md**

---

## 🎉 NEW in Version 4.0: DocAgent-Inspired Bottom-Up Semantic Aggregation

Agent5 V4 introduces a **production-ready**, **bottom-up semantic aggregation pipeline** inspired by DocAgent's docstring synthesis approach. This enables accurate, deep, documentation-quality flowcharts for complex C++ projects.

### Key V4 Features
- ✅ **Bottom-Up Understanding:** Build meaning from leaf nodes upward, never expand downward
- ✅ **Clang AST + CFG:** Full C++ parsing with control-flow graph extraction  
- ✅ **Six-Stage Pipeline:** Deterministic, fail-fast architecture
- ✅ **Structural Detail Levels:** `high` (3-5 steps), `medium` (10-20 steps), `deep` (30+ steps)
- ✅ **LLM Optional:** Use LLM for semantic aggregation or rely on deterministic rules
- ✅ **Debug Mode:** Export AST, semantics, and SFM for troubleshooting
- ✅ **Project-Wide Analysis:** True cross-file scenario tracking
- ✅ **Fail-Fast:** Clear errors instead of incorrect outputs

### Quick Comparison: V3 vs V4

| Feature | V3 | V4 |
|---------|----|----|
| Parser | tree-sitter | Clang AST + CFG |
| Analysis | Top-down expansion | Bottom-up aggregation |
| Scope | Single file (default) | Project-wide |
| Detail Control | Basic filtering | Structural levels |
| Production Ready | No | **Yes** |

---

## V4 Quick Start

### Installation

```bash
# Install system dependencies
# Ubuntu/Debian
sudo apt-get install clang llvm libclang-dev

# macOS
brew install llvm

# Install Agent5 V4
cd /path/to/Agent5
git checkout version4
pip install -e .
```

### Basic Usage

#### High-Level Overview (Business Logic Only)
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create_volume \
  --project-path /path/to/project \
  --detail-level high \
  --out flowcharts/overview.mmd
```

#### Medium Detail (Documentation Quality) ⭐ RECOMMENDED
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create_volume \
  --project-path /path/to/project \
  --detail-level medium \
  --out flowcharts/documented.mmd
```

#### Deep Detail (Full Expanded)
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create_volume \
  --project-path /path/to/project \
  --detail-level deep \
  --out flowcharts/detailed.mmd
```

#### With LLM-Assisted Semantic Aggregation
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create_volume \
  --project-path /path/to/project \
  --detail-level medium \
  --use_llm \
  --chat_model llama3.1 \
  --out flowcharts/llm_assisted.mmd
```

#### Debug Mode (Troubleshooting)
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create_volume \
  --project-path /path/to/project \
  --debug \
  --out flowcharts/debug.mmd
```

**Exported artifacts:**
- `debug_ast_context.json`: Full AST + CFG
- `debug_semantic_summaries.json`: Aggregated semantics
- `debug_sfm.json`: Scenario Flow Model

---

## V4 CLI Reference

### New Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--use-v4-pipeline` | Enable V4 DocAgent-inspired pipeline | `false` (uses V3) |
| `--function` | Entry function name (required in V4) | None |
| `--detail-level` | Structural detail level: `high`, `medium`, `deep` | `medium` |
| `--project-path` | Project root (defines analysis scope) | `file.parent` |
| `--include-paths` | Comma-separated Clang include directories | None |
| `--debug` | Export intermediate artifacts | `false` |
| `--use_llm` | Use LLM for semantic aggregation | `false` |

### Entry-Point Resolution

V4 uses a three-parameter system for precise entry-point resolution:

1. **`--file`**: Entry file (for disambiguation only, NOT scope limiter)
2. **`--function`**: Entry function name (required)
3. **`--project-path`**: Project root (defines analysis scope)

**Example:**
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \              # Where to find entry function
  --function create_volume \            # Which function to analyze
  --project-path /path/to/cinder \     # Scope: entire project
  --detail-level medium \
  --out flowcharts/create_volume.mmd
```

---

## V4 Detail Levels

### `high` - Business-Level Overview
- **Include:** Only major business operations
- **Exclude:** Validations, decisions, state changes
- **Output:** 3-5 steps
- **Use case:** Architecture overview, executive summary

**Example:**
```
Start → Create Volume → End
```

### `medium` - Documentation Quality ⭐ RECOMMENDED
- **Include:** Business logic + validations + decisions + state changes
- **Exclude:** Internal sub-operations, utilities
- **Output:** 10-20 steps
- **Use case:** Technical documentation, code reviews

**Example:**
```
Start
  → Validate volume_id
  → Check user permissions
  → Verify volume doesn't exist
  → Create volume in database
  → Update quota
  → Send notification
  → End
```

### `deep` - Expanded Critical Operations
- **Include:** Everything in medium + critical sub-operations + side effects
- **Exclude:** Logging, metrics, trivial utilities
- **Output:** 30+ steps
- **Use case:** Debugging, performance analysis, security audits

**Example:**
```
Start
  → Parse request parameters
  → Validate volume_id format
  → Check volume_id not null
  → Authenticate user token
  → Check user has create permission
  → Query database for existing volume
  → Decision: Volume exists?
    → Yes: Return error
    → No: Continue
  → Open database connection
  → Begin transaction
  → Insert volume record
  → Commit transaction
  → Close database connection
  → Calculate new quota usage
  → Update user quota
  → Send async notification to message queue
  → End
```

---

## V4 Architecture Overview

### Six-Stage Pipeline

```
Stage 1: Clang AST + CFG Construction (NO LLM)
         ↓
Stage 2: Leaf-Level Semantic Extraction (Deterministic)
         ↓
Stage 3: Bottom-Up Semantic Aggregation (LLM-Assisted)
         ↓
Stage 4: Scenario Flow Model Construction (Single Source of Truth)
         ↓
Stage 5: Detail-Level Filtering (Rule-Based)
         ↓
Stage 6: Mermaid Translation (Strict Translator)
```

**Key Principles:**
1. **Bottom-Up, Never Top-Down:** Build meaning from leaf nodes upward
2. **LLM as Tool, Not Oracle:** Optional semantic interpretation, always has fallback
3. **Fail Fast, Never Guess:** Refuse to proceed on ambiguity
4. **Scenario-Based, Never Function-Call-Based:** Flowcharts represent execution scenarios

📚 **Full Details:** See [`V4_ARCHITECTURE.md`](V4_ARCHITECTURE.md)

---

## Documentation

- **[V4_QUICKSTART.md](V4_QUICKSTART.md)**: Quick start guide with examples
- **[V4_ARCHITECTURE.md](V4_ARCHITECTURE.md)**: Complete technical architecture
- **[V4_CHANGELOG.md](V4_CHANGELOG.md)**: Version 4.0.0 changelog
- **[FLOWCHART_GENERATION_FLOW.md](FLOWCHART_GENERATION_FLOW.md)**: Internal execution flow

---

## Migration from V3

### Before (V3)
```bash
python -m agent5 flowchart \
  --file src/volume.cpp \
  --detail-level medium \
  --out flowcharts/volume.mmd
```

### After (V4)
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create_volume \
  --project-path /path/to/project \
  --detail-level medium \
  --out flowcharts/volume.mmd
```

**Changes:**
1. Add `--use-v4-pipeline` flag
2. Add `--function` parameter (required)
3. Add `--project-path` for project-wide analysis

**Backward Compatibility:** V3 pipeline still available (default without `--use-v4-pipeline`)

---

## Examples

### Example 1: OpenStack Cinder Volume Creation
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file cinder/volume/api.py \
  --function create \
  --project-path /opt/stack/cinder \
  --detail-level medium \
  --out flowcharts/volume_create.mmd
```

### Example 2: Custom Project with Include Paths
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/handler.cpp \
  --function handle_request \
  --project-path /home/user/myproject \
  --include-paths "/usr/include/boost,/opt/custom/include" \
  --detail-level deep \
  --debug \
  --out flowcharts/handler.mmd
```

### Example 3: LLM-Assisted with Ollama
```bash
# Start Ollama
ollama serve

# Generate flowchart
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/api.cpp \
  --function process \
  --project-path /path/to/project \
  --use_llm \
  --chat_model llama3.1 \
  --detail-level medium \
  --out flowcharts/api_process.mmd
```

---

## Troubleshooting

### "libclang not available"
```bash
# Install libclang
pip install libclang

# Install system clang
# Ubuntu: sudo apt-get install libclang-dev
# macOS: brew install llvm
```

### "Entry function not found"
```bash
# Use --debug to see available functions
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create_volume \
  --project-path /path/to/project \
  --debug \
  --out flowcharts/volume.mmd

# Check debug_ast_context.json for available functions
```

### "Ambiguous function name"
Use `--file` to disambiguate:
```bash
python -m agent5 flowchart \
  --use-v4-pipeline \
  --file src/volume.cpp \
  --function create \
  --project-path /path/to/project \
  --out flowcharts/volume.mmd
```

---

## License

[Your License Here]

---

**Version 4.0.0 marks a fundamental shift in how Agent5 understands and documents C++ code. By adopting bottom-up semantic aggregation, we now produce production-ready, documentation-quality flowcharts suitable for real-world use.**

