# Quick Start: libclang AST Flowchart Generation

## Quick Reference

### 1. Build AST JSON
```bash
python -m agent5 build-ast --project_path ./my_project -o ast.json
```

### 2. Generate Flowchart
```bash
python -m agent5 flowchart --function main --ast-json ast.json --out flow.mmd
```

## Prerequisites Check

1. **Install libclang** (system library, not Python package):
   - Linux: `sudo apt-get install libclang-dev`
   - macOS: `brew install llvm`
   - Windows: Download LLVM from GitHub releases

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running:**
   ```bash
   ollama list
   ```

## Complete Example

```bash
# Step 1: Build AST (this may take a few minutes - generates LLM descriptions)
python -m agent5 build-ast \
  --project_path ./examples \
  -o ast_with_calls.json \
  --compile-args -std=c++17

# Step 2: Generate flowchart
python -m agent5 flowchart \
  --function main \
  --ast-json ast_with_calls.json \
  --out flowchart.mmd \
  --detail-level medium

# View the result
cat flowchart.mmd
```

## Troubleshooting

**Error: "libclang.so not found"**
- Set environment variable: `export CLANG_LIB_PATH=/usr/lib/llvm-18/lib/libclang.so`
- Or install: `sudo apt-get install libclang-18-dev`

**Error: "Function not found in AST"**
- Check function name spelling (case-sensitive)
- Verify the function exists in the JSON file: `grep -i "functionName" ast_with_calls.json`

**Empty AST output**
- Check compilation arguments (may need include paths)
- Verify C++ files are parseable
- Check for parsing warnings in build-ast output

For more details, see `USAGE_LIBCLANG.md`.

