# Using libclang AST Builder for Flowchart Generation

This guide explains how to use the new libclang-based AST builder to generate Mermaid flowcharts.

## Overview

The workflow has been updated to use libclang for AST building instead of tree-sitter:

1. **Build AST** - Generate JSON AST file from C++ codebase using libclang
2. **Generate Flowchart** - Build SFM and generate Mermaid flowchart from JSON AST

## Prerequisites

### 1. Install libclang

The Python `clang` package requires the actual libclang library to be installed on your system.

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install libclang-dev
# For LLVM 18 specifically:
sudo apt-get install llvm-18-dev libclang-18-dev
```

#### macOS
```bash
brew install llvm
# Set environment variable:
export CLANG_LIB_PATH=/usr/local/opt/llvm/lib/libclang.dylib
```

#### Windows
Install LLVM from: https://github.com/llvm/llvm-project/releases
Set environment variable:
```powershell
$env:CLANG_LIB_PATH="C:\Program Files\LLVM\bin\libclang.dll"
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `clang` package will be installed from requirements.txt.

## Usage

### Step 1: Build AST from C++ Codebase

Generate a JSON AST file containing function information and call relationships:

```bash
python -m agent5 build-ast \
  --project_path /path/to/your/cpp/project \
  -o ast_with_calls.json \
  --compile-args -std=c++17 -I/path/to/include
```

**Options:**
- `--project_path`: Path to the C++ project root (required)
- `-o, --output`: Output JSON file path (default: `<project_path>/ast_with_calls.json`)
- `--compile-args`: Compilation arguments (default: `-std=c++17`)
  - Use this to specify include paths, defines, etc.
  - Example: `--compile-args -std=c++17 -I./include -DDEBUG`

**Output:**
The command generates a JSON file with the following structure for each function:
```json
{
  "uid": "functionName:file.cpp:42",
  "name": "functionName",
  "line_start": 42,
  "column_start": 1,
  "line_end": 50,
  "column_end": 1,
  "file_name": "/path/to/file.cpp",
  "module_name": "path.to.file",
  "description": "LLM-generated description",
  "flowchart": "LLM-generated flowchart",
  "callees": [{"uid": "calleeFunction:file.cpp:100"}],
  "callers": [{"uid": "callerFunction:file.cpp:20"}]
}
```

### Step 2: Generate Flowchart from JSON AST

Generate a Mermaid flowchart from the JSON AST:

```bash
python -m agent5 flowchart \
  --file source.cpp \
  --function main \
  --ast-json ast_with_calls.json \
  --out flowchart.mmd \
  --detail-level medium \
  --max-steps 30
```

**Options:**
- `--file`: Path to entry C++ file (optional, for reference)
- `--function`: Entry function name (optional, uses first function if not specified)
- `--ast-json`: Path to JSON AST file (if not provided, looks for `ast_with_calls.json` in project_path)
- `--out`: Output path for `.mmd` file (required)
- `--detail-level`: Detail level: `high`, `medium`, or `deep` (default: `medium`)
- `--max-steps`: Maximum steps in flowchart (default: 30)
- `--use-llm`: Use LLM for translation (optional)
- `--project-path`: Project root path (used to locate AST JSON if `--ast-json` not provided)

**Output:**
- `.mmd` file: Mermaid flowchart code
- `.sfm.json` file: Scenario Flow Model (for debugging)

## Complete Example

```bash
# 1. Build AST
python -m agent5 build-ast \
  --project_path ./my_cpp_project \
  -o ./my_cpp_project/ast_with_calls.json \
  --compile-args -std=c++17 -I./include

# 2. Generate flowchart
python -m agent5 flowchart \
  --file ./my_cpp_project/src/main.cpp \
  --function main \
  --ast-json ./my_cpp_project/ast_with_calls.json \
  --out ./flowchart.mmd \
  --detail-level medium
```

## Troubleshooting

### libclang Not Found

If you get an error about libclang not being found:

1. **Set CLANG_LIB_PATH environment variable:**
   ```bash
   export CLANG_LIB_PATH=/usr/lib/llvm-18/lib/libclang.so
   ```

2. **Or install the correct version:**
   - Check which LLVM version you have: `llvm-config --version`
   - Install corresponding libclang: `sudo apt-get install libclang-XX-dev`

### Compilation Errors

If libclang fails to parse files due to missing headers:

1. **Add include paths:**
   ```bash
   --compile-args -std=c++17 -I/usr/include/c++/11 -I/usr/include
   ```

2. **For system headers on Linux:**
   ```bash
   --compile-args -std=c++17 $(pkg-config --cflags --libs stdc++)
   ```

### Empty AST Output

If the AST JSON file is empty or missing functions:

1. Check that your files have valid C++ syntax
2. Ensure compilation arguments are correct
3. Check for parsing warnings in the output
4. Verify that functions are properly declared (not just headers)

## Differences from Tree-sitter Approach

The libclang approach provides:
- ✅ More accurate function call relationships
- ✅ Better handling of C++ language features
- ✅ LLM-generated descriptions and flowcharts per function
- ✅ Better cross-file analysis
- ❌ Requires system libclang installation
- ❌ Slower for very large codebases (generates LLM content)


