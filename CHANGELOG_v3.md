# Agent5 Version 3 Changelog

## Version 3.0.0 - Documentation-Quality Flowcharts

### 🎯 Major Features

#### 1. Detail-Level Support for Flowchart Generation

Added three detail levels to control flowchart granularity for different audiences:

**HIGH** - Executive/Architect View
- Only top-level business steps
- Minimal detail, maximum clarity
- Perfect for presentations and high-level documentation
- Example: "Create resource" → "Validate" → "Allocate" → "Register"

**MEDIUM** - Developer View (Default)
- Include validations, decisions, and state-changing operations
- Balanced detail for day-to-day development
- Shows business logic flow with key decision points
- Example: Adds input validation, error checks, state transitions

**DEEP** - Debug/Documentation View
- Expand critical sub-operations affecting control flow or persistent state
- Maximum detail for debugging and comprehensive documentation
- Includes data lookups, reads, and critical operations
- Example: Shows database queries, file reads, cache lookups

**Usage:**
```bash
python -m agent5 flowchart \
  --file src/handler.cpp \
  --out flow.mmd \
  --detail-level deep
```

**Rules Enforced:**
- ✅ NEVER expand logging, metrics, or utility helpers
- ✅ NEVER produce function-call-based diagrams
- ✅ Depth controlled by RULES, not LLM creativity
- ✅ Maintains fail-fast and deterministic extraction

#### Implementation Details

**Function Classification:**
Functions are now categorized and included based on detail level:

- **business**: Core business logic → Always included
- **validation**: Input/data validation → Included in medium+
- **state**: State-changing operations → Included in medium+
- **critical**: Critical sub-operations → Included in deep only
- **utility**: Utility functions → Never included

**Examples:**

| Function Call | Category | High | Medium | Deep |
|---------------|----------|------|--------|------|
| `createVolume()` | business | ✓ | ✓ | ✓ |
| `validateInput()` | validation | ✗ | ✓ | ✓ |
| `updateState()` | state | ✗ | ✓ | ✓ |
| `fetchData()` | critical | ✗ | ✗ | ✓ |
| `log()`, `printf()` | utility | ✗ | ✗ | ✗ |

### 🔧 API Changes

#### New Parameters

**scenario_extractor.py:**
```python
extract_scenario_from_function(
    source_code: str,
    function_name: str | None = None,
    *,
    max_steps: int = 30,
    detail_level: DetailLevel = DetailLevel.MEDIUM,  # NEW
) -> ScenarioFlowModel
```

**flowchart.py:**
```python
generate_scenario_flowchart(
    source_code: str,
    function_name: str | None = None,
    *,
    max_steps: int = 30,
    detail_level: str = "medium",  # NEW
    use_llm: bool = False,
    chat_model: str | None = None,
    ollama_base_url: str | None = None,
) -> MermaidFlowchart
```

**CLI:**
```bash
python -m agent5 flowchart \
  --file <file.cpp> \
  --out <output.mmd> \
  --detail-level {high|medium|deep}  # NEW
```

### 📊 Use Cases

#### Use Case 1: Executive Presentation
```bash
python -m agent5 flowchart \
  --file src/payment_processor.cpp \
  --out payment_overview.mmd \
  --detail-level high
```
Result: Clean, high-level flow showing only major business steps

#### Use Case 2: Developer Documentation
```bash
python -m agent5 flowchart \
  --file src/payment_processor.cpp \
  --out payment_flow.mmd \
  --detail-level medium
```
Result: Balanced view with validations and key decisions

#### Use Case 3: Debugging Session
```bash
python -m agent5 flowchart \
  --file src/payment_processor.cpp \
  --out payment_detailed.mmd \
  --detail-level deep
```
Result: Comprehensive flow including data operations

### 🔄 Migration from v2

**No Breaking Changes:**
- Default behavior unchanged (medium detail level)
- All existing commands work as before
- New `--detail-level` parameter is optional

**To upgrade:**
```bash
cd Agent5
git checkout version3
git pull origin version3
```

### 📝 Files Modified

- `agent5/scenario_extractor.py`: Added DetailLevel enum, enhanced classification
- `agent5/flowchart.py`: Added detail_level parameter support
- `agent5/cli.py`: Added --detail-level CLI option
- `README.md`: Updated with detail-level documentation

### 🎉 Benefits

1. **Flexible Documentation**: One codebase, multiple audience levels
2. **Better Presentations**: High-level flows for stakeholders
3. **Improved Debugging**: Deep flows show all operations
4. **Controlled Output**: Rules-based, not LLM-guessing
5. **Backward Compatible**: Existing workflows unchanged

### 📖 Documentation

See updated README.md for:
- Detail level examples
- Use case scenarios
- Best practices for each level
- API reference

---

## Upgrade Instructions

```bash
# Clone or pull latest
git checkout version3

# Install (no new dependencies)
pip install -r requirements.txt

# Try it out
python -m agent5 flowchart \
  --file your_file.cpp \
  --out test.mmd \
  --detail-level deep
```

**Happy documenting! 🚀**




