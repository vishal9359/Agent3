# Changelog - Version 4

## Agent5 V4: DocAgent-Inspired Bottom-Up Semantic Aggregation

**Release Date**: December 18, 2025

---

### 🎉 Major Changes

This version represents a **complete architectural redesign** of Agent5, implementing a DocAgent-inspired bottom-up semantic aggregation pipeline for generating documentation-quality flowcharts from complex C++ projects.

---

### 🏗️ New Architecture: 6-Stage Pipeline

#### Stage 1: Full AST Construction (Clang)
- **New Module**: `agent5/clang_ast_parser.py`
- Replaces tree-sitter with libclang for accurate C++ parsing
- Builds full Abstract Syntax Tree (AST) for all translation units
- Extracts call graph and identifies leaf functions
- Supports project-wide analysis
- **Key Classes**:
  - `ClangASTParser`: Main AST parser
  - `CallGraph`: Project call graph representation
  - `FunctionInfo`: Detailed function metadata

#### Stage 1b: Control Flow Graph Construction
- **New Module**: `agent5/cfg_builder.py`
- Builds Control Flow Graphs (CFG) for each function
- Identifies basic blocks, guard conditions, state mutations, error exits
- Analyzes semantic properties at block level
- **Key Classes**:
  - `CFGBuilder`: CFG construction engine
  - `ControlFlowGraph`: CFG representation
  - `BasicBlock`: Basic block with semantic properties

#### Stage 2: Leaf-Level Semantic Extraction
- **New Module**: `agent5/semantic_extractor.py`
- Extracts atomic semantic actions from CFG blocks
- Identifies: validations, permission checks, state mutations, side effects, early exits
- **NO LLM inference** - purely deterministic
- **Key Classes**:
  - `SemanticExtractor`: Semantic extraction engine
  - `SemanticAction`: Atomic semantic action
  - `FunctionSemantics`: Function-level semantic description

#### Stage 3: Bottom-Up Semantic Aggregation
- **New Module**: `agent5/aggregator.py`
- Implements DocAgent-inspired bottom-up backtracking
- Starts from leaf functions, aggregates upward
- Uses LLM for semantic summarization (strictly fact-based)
- Combines child summaries while preserving control flow and state semantics
- **Key Classes**:
  - `BottomUpAggregator`: Aggregation engine
  - `AggregatedSemantics`: Aggregated semantic understanding

#### Stage 4: Scenario Flow Model Construction
- **New Module**: `agent5/sfm_builder.py`
- Converts aggregated semantics into Scenario Flow Model (SFM)
- SFM is the **single source of truth** for flowchart generation
- Explicit mapping to detail levels
- JSON-based representation
- **Key Classes**:
  - `SFMBuilder`: SFM construction engine
  - `ScenarioFlowModel`: SFM representation
  - `SFMNode`, `SFMEdge`: SFM components

#### Stage 5: Detail-Level Filtering
- **New Module**: `agent5/detail_filter.py`
- Rule-based filtering of SFM by detail level
- Supports `high`, `medium`, `deep` detail levels
- Never expands utility operations (logging, metrics, etc.)
- Reconnects edges when intermediate nodes are filtered
- **Key Classes**:
  - `DetailLevelFilter`: Filtering engine
  - `DetailLevel`: Enum for detail levels

#### Stage 6: Mermaid Translation
- **New Module**: `agent5/flowchart_v4.py`
- LLM as **strict translator** from SFM to Mermaid
- No logic changes, no depth changes, no inference
- Deterministic fallback if LLM fails
- Validates SFM before translation
- **Key Classes**:
  - `MermaidTranslator`: Translation engine

---

### 🔧 Pipeline Orchestration

- **New Module**: `agent5/pipeline.py`
- Orchestrates all 6 stages
- Handles entry-point resolution
- Saves intermediate artifacts in debug mode
- **Key Classes**:
  - `FlowchartPipeline`: Main pipeline orchestrator

---

### 🖥️ New CLI: cli_v4.py

- **New Module**: `agent5/cli_v4.py`
- Dedicated CLI for V4 pipeline
- Simplified, focused interface
- Clear documentation in help text

**New Command Structure**:
```bash
python -m agent5.cli_v4 \
  --project-path /path/to/project \
  --entry-function FunctionName \
  --entry-file path/to/file.cpp \
  --detail-level medium \
  --out flowchart.mmd
```

**Key Arguments**:
- `--project-path` (required): Defines analysis scope (ENTIRE project)
- `--entry-function`: Entry point function name (auto-detects if omitted)
- `--entry-file`: For disambiguating entry function (NOT for limiting scope)
- `--detail-level`: `high`, `medium` (default), or `deep`
- `--include-paths`: Comma-separated include paths for Clang
- `--llm-model`: Ollama model to use (default: `llama2:7b`)
- `--no-llm`: Disable LLM (use deterministic fallbacks)
- `--debug`: Save intermediate artifacts to `project_path/output/`

---

### 📦 Dependencies

**New Dependency**:
- `libclang==18.1.1` - Python bindings for Clang

**Updated**:
- `tree-sitter` and `tree-sitter-cpp` marked as "legacy support" (retained for backwards compatibility)

---

### 🎯 Detail Levels (Now Fully Functional)

#### HIGH Level
- **Purpose**: Architecture overview for stakeholders
- **Includes**: Major business operations only
- **Excludes**: Implementation details, validations, internal decisions

#### MEDIUM Level (Default)
- **Purpose**: Documentation-quality flowcharts for developers
- **Includes**: All validations, decisions, state changes
- **Excludes**: Internal sub-operations

#### DEEP Level
- **Purpose**: Implementation details for code review
- **Includes**: All MEDIUM content + critical sub-operations affecting control/state
- **Excludes**: Logging, metrics, utility helpers (always excluded)

**Previously**: Detail levels produced nearly identical flowcharts
**Now**: Detail levels produce structurally distinct flowcharts with meaningful differences

---

### 🔍 Entry-Point Resolution (Clarified)

**Critical Rule**: `--entry-file` ≠ analysis scope

#### Resolution Logic

1. **Both `--entry-function` and `--entry-file` provided**:
   - Strict resolution
   - Function must exist in specified file
   - Analysis scope = entire `--project-path`

2. **Only `--entry-function` provided**:
   - Search across entire project
   - Error if ambiguous (multiple matches)
   - Error if not found

3. **Neither provided**:
   - Auto-detect using AST evidence:
     - Functions with no callers
     - Functions named "main"
     - First available entry point

**Previously**: `--entry-file` incorrectly limited analysis scope to single file
**Now**: `--project-path` ALWAYS defines the analysis scope

---

### 🚫 What We Don't Do (Critical Constraints)

1. **No Function-Call Diagrams**
   - V4 generates **scenario flows**, not call graphs
   - Function calls are *summarized*, not *expanded*

2. **No Recursive Visual Expansion**
   - Functions are collapsed into semantic actions
   - Control flow is preserved, not function boundaries

3. **No LLM Creativity**
   - LLM is used ONLY for:
     - Semantic summarization (Stage 3) - strictly fact-based
     - Mermaid translation (Stage 6) - strict translation only
   - LLM does NOT invent logic, control flow, or behavior

4. **Scenario Flow Model is Authoritative**
   - SFM (Stage 4) is the single source of truth
   - All subsequent stages work from SFM, never bypass it

---

### 🐛 Bug Fixes from V3

1. **Fixed**: Detail levels producing identical flowcharts
   - Implemented strict, rule-based filtering
   - Each detail level now produces structurally distinct output

2. **Fixed**: `--entry-file` limiting analysis to single file
   - Clarified: `--entry-file` is for disambiguation only
   - Analysis scope is ALWAYS defined by `--project-path`

3. **Fixed**: Function-call diagram generation instead of scenario flows
   - New pipeline ensures scenario-based flow from the ground up
   - Function calls are summarized, never expanded

4. **Fixed**: LLM inventing logic not present in code
   - Stage 2 (Semantic Extraction) is deterministic
   - Stage 3 (Aggregation) uses LLM strictly for summarization
   - All logic comes from AST/CFG analysis, not LLM

---

### 📊 Performance Characteristics

- **Stage 1 (AST Construction)**: O(n) where n = lines of code
- **Stage 2 (Semantic Extraction)**: O(b) where b = number of basic blocks
- **Stage 3 (Aggregation)**: O(f) where f = number of functions in call path
  - LLM calls: One per function in bottom-up order
- **Stage 4-6**: O(1) relative to project size (work on aggregated data)

**Memory Usage**:
- Full AST is kept in memory for analysis
- CFGs are built on-demand
- Intermediate artifacts can be saved to disk (`--debug` mode)

---

### 🧪 Testing Recommendations

1. **Smoke Test**: Run on a simple C++ project with known entry point
2. **Detail Level Test**: Generate `high`, `medium`, `deep` for same entry point, verify differences
3. **Ambiguous Entry Test**: Provide ambiguous function name without file, verify error handling
4. **Cross-File Test**: Use entry point that calls functions across multiple files
5. **No LLM Test**: Run with `--no-llm` to verify deterministic fallbacks work

---

### 📚 Documentation

**New Documents**:
- `AGENT5_V4_README.md`: Comprehensive V4 user guide
- `CHANGELOG_v4.md`: This file
- `FLOWCHART_GENERATION_FLOW.md`: Internal flow documentation (updated for V4)

---

### 🔄 Migration from V3

**Breaking Changes**:
- V3 CLI commands do not work with V4 pipeline
- Use `agent5.cli_v4` for V4 pipeline
- V3 is still available via original `agent5.cli`

**Recommended Migration Path**:
1. Test V4 pipeline with existing entry points
2. Compare outputs with V3 (V4 should be more accurate)
3. Migrate workflows to V4 CLI
4. Update documentation references

---

### 🎓 Learning Resources

For understanding the approach:
- [DocAgent (Facebook Research)](https://github.com/facebookresearch/DocAgent) - Bottom-up understanding strategy
- [Clang AST Introduction](https://clang.llvm.org/docs/IntroductionToTheClangAST.html)
- [Control Flow Analysis](https://en.wikipedia.org/wiki/Control-flow_graph)

---

### 🙏 Acknowledgments

- DocAgent team (Facebook Research) for the bottom-up understanding approach
- Clang/LLVM team for libclang
- Ollama project for open-source LLM infrastructure

---

### 🔮 Future Enhancements (Potential)

- **Caching**: Cache AST/CFG/semantics between runs
- **Incremental Analysis**: Only reanalyze changed functions
- **Multi-Entry Point**: Generate flowcharts for multiple entry points in one pass
- **Interactive Mode**: Web UI for exploring scenarios
- **Custom Rules**: User-defined semantic classification rules
- **Code Generation**: Reverse direction - generate code from SFM

---

### 📝 Known Limitations

1. **C++ Template Complexity**: Heavy template metaprogramming may not be fully captured
2. **Macro Expansion**: Macros are analyzed post-expansion; original macro logic is lost
3. **External Libraries**: Functions from external libraries are treated as black boxes
4. **Inline Assembly**: Inline assembly is not analyzed
5. **Virtual Calls**: Virtual function calls are analyzed conservatively

---

### 🤝 Contributing

Contributions welcome! Focus areas:
- Improving semantic classification rules
- Adding support for more C++ patterns
- Optimizing performance for very large projects
- Enhancing LLM prompts for better summarization

---

**Version**: 4.0.0  
**Build Date**: December 18, 2025  
**Codename**: "DocAgent"
