# Version 4 Quick Start Guide

## What is V4?

Version 4 introduces a **DocAgent-inspired bottom-up semantic aggregation** approach for generating documentation-quality flowcharts from complex C++ projects. Unlike V3's single-file analysis, V4 understands your entire project by building meaning from leaf functions upward through the call graph.

## Installation

### 1. Install libclang

V4 requires libclang for AST analysis.

**Ubuntu/Debian:**
```bash
sudo apt-get install libclang-14-dev
```

**macOS:**
```bash
brew install llvm
```

**Windows:**
Download and install LLVM from https://releases.llvm.org/

### 2. Install Agent5

```bash
cd agent5
pip install -r requirements.txt
pip install -e .
```

### 3. Verify Installation

```bash
python -c "import clang.cindex; print('libclang OK')"
```

## Basic Usage

### Generate a Medium-Detail Flowchart

```bash
python -m agent5 flowchart \
  --file src/my_file.cpp \
  --function MyFunction \
  --project-path /path/to/project \
  --detail-level medium \
  --out flowchart.mmd \
  --use_v4
```

### With LLM Enhancement

```bash
python -m agent5 flowchart \
  --file src/my_file.cpp \
  --function MyFunction \
  --project-path /path/to/project \
  --detail-level medium \
  --out flowchart.mmd \
  --use_v4 \
  --use_llm \
  --chat_model llama3.2:3b
```

## Detail Levels Explained

### High: Architecture Overview

**Use when:** You want a bird's-eye view of the scenario

**Includes:**
- Major business operations only
- Minimal decision points
- High-level state changes

**Example:**
```
Start → Create Resource → Validate → Save → End
```

### Medium: Documentation (Default)

**Use when:** You want comprehensive documentation

**Includes:**
- All validations
- All decision points
- All state-changing operations

**Example:**
```
Start → Validate Input → Check Permissions → 
Create Resource → Update State → Save to DB → End
```

### Deep: Detailed Analysis

**Use when:** You need to understand critical internal operations

**Includes:**
- Everything from Medium
- Critical sub-operations (lookups, reads, loads)
- Internal validation steps
- Control-flow-affecting operations

**Excludes (always):**
- Logging
- Metrics
- Utility helpers
- Memory allocation wrappers

**Example:**
```
Start → Parse Input → Validate Schema → Check Permissions →
Lookup Existing Resource → Lock Resource → 
Create New Resource → Update Index → Save to DB → 
Release Lock → End
```

## Understanding the Pipeline

V4 uses a 6-stage pipeline:

```
1. AST Construction (Clang)
   ↓
2. Leaf-Level Semantic Extraction (Rule-Based)
   ↓
3. Bottom-Up Aggregation (LLM-Assisted)
   ↓
4. Scenario Flow Model Construction
   ↓
5. Detail-Level Filtering
   ↓
6. Mermaid Translation
```

## Debug Mode

Enable debug mode to inspect intermediate artifacts:

```bash
python -m agent5 flowchart \
  --file src/my_file.cpp \
  --function MyFunction \
  --project-path /path/to/project \
  --out flowchart.mmd \
  --use_v4 \
  --debug
```

This generates:
- `flowchart.sfm.json` - Scenario Flow Model (single source of truth)

## Common Use Cases

### 1. Onboarding Documentation

```bash
# Generate high-level overview
python -m agent5 flowchart \
  --file src/main.cpp \
  --function main \
  --project-path /path/to/project \
  --detail-level high \
  --out docs/architecture.mmd \
  --use_v4
```

### 2. API Documentation

```bash
# Generate medium-detail API flow
python -m agent5 flowchart \
  --file src/api_handler.cpp \
  --function HandleRequest \
  --project-path /path/to/project \
  --detail-level medium \
  --out docs/api_flow.mmd \
  --use_v4 \
  --use_llm
```

### 3. Deep Debugging

```bash
# Generate deep analysis with debug artifacts
python -m agent5 flowchart \
  --file src/core.cpp \
  --function ProcessTransaction \
  --project-path /path/to/project \
  --detail-level deep \
  --out docs/transaction_flow.mmd \
  --use_v4 \
  --use_llm \
  --debug
```

## Entry Point Resolution

### Auto-Detection

If you don't specify `--function`, V4 will auto-detect entry points:

```bash
python -m agent5 flowchart \
  --file src/main.cpp \
  --project-path /path/to/project \
  --out flowchart.mmd \
  --use_v4
```

### Disambiguation

If multiple functions have the same name, use `--file` to disambiguate:

```bash
python -m agent5 flowchart \
  --file src/module_a.cpp \
  --function Process \
  --project-path /path/to/project \
  --out flowchart.mmd \
  --use_v4
```

## Custom Include Paths

For projects with custom include directories:

```bash
python -m agent5 flowchart \
  --file src/my_file.cpp \
  --function MyFunction \
  --project-path /path/to/project \
  --include_paths "/usr/include,/opt/custom/include,./libs" \
  --out flowchart.mmd \
  --use_v4
```

## Troubleshooting

### "libclang not available"

**Problem:** libclang not installed or not found

**Solution:**
1. Install libclang (see Installation)
2. Set CLANG_LIBRARY_PATH environment variable

```bash
export CLANG_LIBRARY_PATH=/usr/lib/llvm-14/lib
```

### "Function not found"

**Problem:** Entry function not detected in specified file

**Solution:**
1. Check function name spelling
2. Use `--debug` to see all available functions
3. Ensure function is defined (not just declared)

### "SFM validation failed"

**Problem:** Scenario Flow Model has structural issues

**Solution:**
1. Enable `--debug` to inspect SFM
2. Check for complex control flow (goto, exceptions)
3. Report issue with minimal reproduction case

## Next Steps

1. Read `CHANGELOG_v4.md` for complete feature list
2. Read `V4_ARCHITECTURE.md` for technical details
3. Experiment with different detail levels
4. Try with and without LLM to see the difference
5. Use `--debug` to understand how V4 processes your code

## Feedback

V4 is a major architectural change. Your feedback is valuable:
- What works well?
- What needs improvement?
- What use cases are not covered?

Please open issues on GitHub with examples.
