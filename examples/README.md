# Examples

This directory contains example C++ files for testing Agent5.

## simple_calculator.cpp

A simple command-line calculator that demonstrates:
- Argument parsing
- Input validation
- Control flow (if statements)
- Function calls collapsed to semantic actions
- Error handling with returns

### Test Commands

Generate flowchart:
```bash
python -m agent5 flowchart \
  --file examples/simple_calculator.cpp \
  --out calculator_flow.mmd \
  --function main
```

Expected flowchart should show:
1. Start
2. Parse arguments
3. Validate numbers (decision)
4. Validate operator (decision)
5. Calculate result
6. Handle errors
7. End

