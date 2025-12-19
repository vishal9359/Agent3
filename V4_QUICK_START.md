# Version 4 Quick Start Guide

## What's New in v4?

**🎉 Cross-File Scenario Analysis is here!**

Version 4 implements the critical feature you requested: `--entry-file` now correctly serves as an entry-point locator only, while `--project-path` defines the full analysis scope.

## Quick Comparison

### Before (v3): Single-File Analysis

```bash
agent5 flowchart \
  --file src/manager.cpp \
  --function CreateVolume \
  --out flow.mmd
```

**Result:** Shallow flowchart limited to `manager.cpp` only

### After (v4): Cross-File Analysis

```bash
agent5 flowchart \
  --file src/manager.cpp \
  --function CreateVolume \
  --project-path /path/to/project \
  --out flow.mmd
```

**Result:** Deep flowchart following calls across `validator.cpp`, `storage.cpp`, `registry.cpp`, etc.

## Key Changes

### 1. New Module: `cross_file_scenario.py`
- Indexes all functions across the entire project
- Follows function calls based on detail level
- Recursively extracts and integrates SFMs from multiple files
- Limits depth to avoid infinite recursion (default: 3 levels)

### 2. Updated Flowchart Generation
- `--project-path` enables cross-file analysis
- Without `--project-path`: single-file analysis (legacy mode)
- Detail levels now control cross-file following depth

### 3. Enhanced Documentation
- Updated README with v4 features
- CHANGELOG_v4.md with technical details
- V4_IMPLEMENTATION_SUMMARY.md with comprehensive explanation

## Usage Examples

### Basic Cross-File Analysis

```bash
# Generate flowchart following calls across the entire project
agent5 flowchart \
  --file src/main.cpp \
  --function main \
  --project-path . \
  --detail-level medium \
  --out main_flow.mmd
```

### Deep Analysis

```bash
# Follow calls up to 3 levels deep with extensive detail
agent5 flowchart \
  --file src/core/processor.cpp \
  --function ProcessRequest \
  --project-path /path/to/project \
  --detail-level deep \
  --out processor_deep.mmd
```

### High-Level Overview

```bash
# Quick architecture overview with minimal cross-file following
agent5 flowchart \
  --file src/handler.cpp \
  --function HandleRequest \
  --project-path /path/to/project \
  --detail-level high \
  --out handler_overview.mmd
```

### Single-File Analysis (Legacy)

```bash
# Analyze only the specified file (no --project-path)
agent5 flowchart \
  --file src/utils/helper.cpp \
  --function HelperFunction \
  --out helper_flow.mmd
```

## How It Works

1. **Index Phase:**
   - Scans all `.cpp`, `.cc`, `.cxx`, `.c`, `.hpp`, `.h`, `.hxx` files
   - Parses with Tree-sitter
   - Creates function index: `name → (file, code)`

2. **Extraction Phase:**
   - Starts from entry function in entry file
   - Builds SFM (Scenario Flow Model)
   - Identifies function calls

3. **Expansion Phase:**
   - Looks up function definitions in index
   - Recursively extracts their SFMs
   - Integrates into main flowchart
   - Continues up to 3 levels deep

4. **Translation Phase:**
   - Converts unified SFM to Mermaid
   - Outputs flowchart spanning multiple files

## Detail Level Impact on Cross-File Analysis

### HIGH
- Only follows major business operations
- Minimal cross-file expansion
- Fast, suitable for architecture overview

### MEDIUM (Default)
- Follows validations, decisions, state changes
- Moderate cross-file expansion
- Balanced, suitable for documentation

### DEEP
- Follows all critical operations
- Extensive cross-file expansion
- Detailed, suitable for debugging

## Testing Your Changes

### Test 1: Verify Cross-File Analysis

```bash
# Index your project
agent5 index --project_path /path/to/project --collection test --clear

# Generate flowchart with cross-file analysis
agent5 flowchart \
  --file src/main.cpp \
  --function main \
  --project-path /path/to/project \
  --out cross_file.mmd

# Check the output:
# - Should include nodes from multiple files
# - Should show function call expansions
# - Should be much more detailed than v3
```

### Test 2: Compare Single-File vs Cross-File

```bash
# Single-file analysis (v3 behavior)
agent5 flowchart \
  --file src/main.cpp \
  --function main \
  --out single.mmd

# Cross-file analysis (v4 behavior)
agent5 flowchart \
  --file src/main.cpp \
  --function main \
  --project-path /path/to/project \
  --out cross.mmd

# Compare:
# - single.mmd: Shallow, only shows main.cpp
# - cross.mmd: Deep, shows calls across project
```

### Test 3: Verify Detail Level Differences

```bash
# Test all three levels
agent5 flowchart --file src/main.cpp --function main --project-path . --detail-level high --out high.mmd
agent5 flowchart --file src/main.cpp --function main --project-path . --detail-level medium --out medium.mmd
agent5 flowchart --file src/main.cpp --function main --project-path . --detail-level deep --out deep.mmd

# Verify:
# - high.mmd: Fewest nodes
# - medium.mmd: More nodes
# - deep.mmd: Most nodes
```

## Performance Expectations

### Small Project (50 files)
- Index time: ~1 second
- Flowchart generation: ~2-5 seconds

### Medium Project (500 files)
- Index time: ~5 seconds
- Flowchart generation: ~5-15 seconds

### Large Project (5000 files)
- Index time: ~30 seconds
- Flowchart generation: ~10-45 seconds

**Note:** Performance depends on:
- Number of files
- Size of files
- Detail level
- Recursion depth
- Number of function calls

## Known Limitations

1. **Heuristic Call Detection:** Uses pattern matching, not perfect
2. **No Namespace Disambiguation:** May merge same-named functions
3. **No Template Tracking:** Template specializations not distinguished
4. **Memory Overhead:** Function index stored in memory

See `V4_IMPLEMENTATION_SUMMARY.md` for complete details.

## Troubleshooting

### Issue: Flowchart is still shallow

**Cause:** Missing `--project-path` parameter

**Solution:**
```bash
# Add --project-path to enable cross-file analysis
agent5 flowchart --file src/main.cpp --function main --project-path . --out flow.mmd
```

### Issue: Generation is too slow

**Cause:** Detail level too deep or project too large

**Solution:**
```bash
# Use "high" detail level for faster generation
agent5 flowchart --file src/main.cpp --function main --project-path . --detail-level high --out flow.mmd
```

### Issue: Cannot find function definition

**Cause:** Function not in project or index build failed

**Solution:**
1. Check that function exists in the project
2. Verify all C++ files are in `project-path`
3. Check for parsing errors in logs

### Issue: Flowchart has duplicate nodes

**Cause:** Function called multiple times or circular dependencies

**Solution:** This is expected behavior. The current implementation prefixes nodes to show context.

## Migration from v3

**No action required!** All v3 commands work in v4.

**To enable new features:**
- Add `--project-path` to flowchart commands
- Optionally adjust `--detail-level` based on needs

## Next Steps

1. **Test on your C++ project:**
   ```bash
   agent5 flowchart --file your/entry.cpp --function YourFunction --project-path /your/project --out test.mmd
   ```

2. **Compare with v3 output** to verify improvements

3. **Adjust detail level** based on your documentation needs

4. **Report issues** if you encounter problems

5. **Experiment with different detail levels** to find what works best

## Getting Help

- Read `V4_IMPLEMENTATION_SUMMARY.md` for technical details
- Read `CHANGELOG_v4.md` for complete change list
- Check `README.md` for full documentation
- Open GitHub issues for bugs or feature requests

---

**Enjoy the new cross-file scenario analysis feature! 🚀**



