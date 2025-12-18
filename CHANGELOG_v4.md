# Changelog - Version 4

## Overview

Version 4 implements **cross-file scenario analysis**, fulfilling the critical requirement that `--entry-file` should only disambiguate the entry point, not limit the analysis scope.

## Major Changes

### 1. Cross-File Scenario Extraction

**New Module: `agent5/cross_file_scenario.py`**

Implements scenario flow extraction that follows function calls across multiple files in a C++ project.

**Key Features:**
- Starts from the entry function in the specified entry file
- Builds an index of all functions across the entire project
- Follows function calls based on detail level (high/medium/deep)
- Recursively extracts and integrates SFMs from called functions
- Limits recursion depth to avoid infinite loops (default: 3 levels)
- Properly handles visited functions to avoid duplicate analysis

**Strategy:**
1. Build a function index by scanning all C++ files in `project_path`
2. Extract SFM for the entry function
3. Identify function calls that should be expanded (based on detail level)
4. Search for function definitions across the project
5. Recursively extract their SFMs and integrate them
6. Continue up to `max_depth` levels (default: 3)

### 2. Updated Flowchart Generation

**Modified: `agent5/flowchart.py`**

- `write_flowchart()` now uses cross-file analysis when `project_path` is provided
- Falls back to single-file analysis when `project_path` is `None`
- Properly distinguishes between:
  - **`file_path`**: Locates the entry point function (entry-point locator)
  - **`project_path`**: Defines the analysis scope (analysis boundary)

**Behavior:**
- If `project_path` is provided: Full cross-file analysis
- If `project_path` is `None`: Single-file analysis (legacy mode)

### 3. Detail Level Impact on Cross-File Analysis

**Detail levels control which function calls are followed:**

- **HIGH**: Only top-level business operations
  - Minimal cross-file following
  - Suitable for architecture overview
  
- **MEDIUM** (default): Include validations, decisions, state changes
  - Moderate cross-file following
  - Suitable for documentation
  
- **DEEP**: Expand critical sub-operations affecting control flow
  - Extensive cross-file following
  - Suitable for detailed analysis
  - Still excludes logging, metrics, utility helpers

## Technical Details

### Function Index

The cross-file extractor builds a complete function index by:
1. Finding all C++ files in the project (`.cpp`, `.cc`, `.cxx`, `.c`, `.hpp`, `.h`, `.hxx`)
2. Parsing each file with Tree-sitter
3. Extracting all function definitions
4. Creating a mapping: `function_name → (file_path, source_code)`

### SFM Integration

When integrating a called function's SFM into the main SFM:
1. Nodes from the called function are prefixed with `FunctionName_`
2. Labels are prefixed to show context: `FunctionName: Step`
3. Edges are remapped to use the new node IDs
4. Start/End nodes of called functions are omitted
5. Call site node is connected to the first node of the called function

### Depth Limiting

To prevent infinite recursion:
- `max_depth` parameter limits how many levels deep we follow calls
- Default: 3 levels
- Functions are tracked in `_visited_functions` to avoid re-processing
- Each level of recursion processes calls from the previous level

## Breaking Changes

None. The changes are backward compatible:
- Single-file mode still works when `project_path` is not provided
- All existing CLI commands work as before
- New functionality is opt-in via `--project-path`

## Usage Examples

### Cross-File Analysis (NEW)

```bash
# Generate flowchart following calls across the entire project
agent5 flowchart \
  --file src/main.cpp \
  --function main \
  --project-path /path/to/project \
  --output flowchart.mmd \
  --detail-level medium

# Deep analysis with extensive cross-file following
agent5 flowchart \
  --file src/core/processor.cpp \
  --function ProcessRequest \
  --project-path /path/to/project \
  --output processor_flow.mmd \
  --detail-level deep
```

### Single-File Analysis (Legacy)

```bash
# Analyze only the specified file (no --project-path)
agent5 flowchart \
  --file src/utils/helper.cpp \
  --function HelperFunction \
  --output helper_flow.mmd
```

## Bug Fixes

### Critical Bug: Entry-File Limiting Scope

**Problem:**
- When `--entry-file` was provided, analysis was restricted to that file only
- This violated the principle that `entry-file` should only disambiguate the entry point
- Result: Incomplete, shallow flowcharts

**Solution:**
- `--project-path` now explicitly defines the analysis scope
- `--entry-file` is strictly used for locating the entry function
- Cross-file analysis follows execution flow across the entire project
- Single-file analysis is now opt-in (when `project_path` is not provided)

## Performance Considerations

### Function Index Building
- Scans all C++ files in the project once
- Parsing is done with Tree-sitter (fast)
- Index is built in memory (no persistence between runs)
- For large projects (1000+ files), indexing may take 5-10 seconds

### Recursion Depth
- Default `max_depth=3` is a balance between completeness and performance
- Each level can exponentially increase the number of functions analyzed
- Adjust based on project size and detail requirements:
  - Depth 1: Fast, good for quick overview
  - Depth 2-3: Moderate, good for documentation
  - Depth 4+: Slower, only for very detailed analysis

## Known Limitations

1. **Heuristic Function Call Detection**
   - Cross-file expansion identifies calls based on node labels
   - Uses pattern matching (not perfect)
   - May miss some function calls or identify false positives
   - Future enhancement: Direct AST call graph analysis

2. **No Template Instantiation Tracking**
   - Template functions are indexed by name only
   - Template specializations are not distinguished
   - May merge different template instantiations

3. **Namespace Disambiguation**
   - Functions with same name in different namespaces may conflict
   - Current implementation uses simple name matching
   - Future enhancement: Fully qualified name resolution

4. **SFM Integration Strategy**
   - Current approach prefixes and appends nodes
   - Not true inlining (keeps called function nodes separate)
   - Future enhancement: True SFM inlining with node merging

## Testing Recommendations

1. **Test on a small project first** to verify cross-file analysis works correctly
2. **Compare detail levels** (high vs medium vs deep) to see structural differences
3. **Check that cross-file analysis follows calls** across multiple files
4. **Verify performance** on your specific project size
5. **Adjust `max_depth`** if flowcharts are too shallow or too complex

## Migration from v3

No migration required. All v3 commands work in v4.

To enable cross-file analysis, simply add `--project-path` to your flowchart commands:

```bash
# Before (v3): Single-file analysis
agent5 flowchart --file src/main.cpp --function main --output flow.mmd

# After (v4): Cross-file analysis
agent5 flowchart --file src/main.cpp --function main --project-path . --output flow.mmd
```

## Next Steps

Potential future enhancements:
1. **Persist function index** to disk for faster subsequent runs
2. **True SFM inlining** instead of prefixing/appending
3. **Direct call graph extraction** from AST (more accurate than heuristics)
4. **Namespace-aware function resolution**
5. **Template instantiation tracking**
6. **Configurable `max_depth`** via CLI
7. **Visual indicators** in flowchart for cross-file boundaries

---

**Version:** 4.0.0  
**Date:** 2025-12-18  
**Status:** Implemented and ready for testing

