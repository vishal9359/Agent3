# Installation Guide - Agent5 V4

This guide covers installation and setup for Agent5 V4.

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM recommended
- **Disk Space**: ~2GB (including dependencies and models)

### Required Software

1. **Python 3.8+**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **Ollama** (for local LLM inference)
   - Install from: https://ollama.ai
   - Verify installation:
     ```bash
     ollama --version
     ```

3. **libclang** (automatically installed via pip, but system libclang may be needed)
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get install libclang-dev
     ```
   - **macOS:**
     ```bash
     brew install llvm
     ```
   - **Windows:** Usually works with pip-installed libclang

---

## Installation Steps

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone <repository-url>
cd Agent5

# Checkout version4 branch
git checkout version4

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Option 2: Install with Setup Script

```bash
# Clone and navigate to the repository
git clone <repository-url>
cd Agent5
git checkout version4

# Install using setup_v4.py
pip install .
```

---

## Post-Installation Setup

### 1. Install Ollama Models

Agent5 V4 requires an LLM for semantic aggregation and Mermaid translation.

**Recommended Model (Default):**
```bash
ollama pull llama3.2:3b
```

**Alternative Models:**
```bash
# Smaller, faster (less accurate)
ollama pull llama3.2:1b

# Larger, more accurate (slower)
ollama pull llama3.1:8b
ollama pull llama3.1:70b  # Requires significant RAM
```

**Verify Model Installation:**
```bash
ollama list
```

### 2. Verify Installation

```bash
# Check V4 CLI is available
agent5-v4 --help

# Should display V4 command help
```

### 3. Test with Example Project

```bash
# Create a simple test file
mkdir -p test_project
cat > test_project/main.cpp << 'EOF'
#include <iostream>

bool validate(int x) {
    return x > 0;
}

int process(int x) {
    if (!validate(x)) {
        return -1;
    }
    return x * 2;
}

int main() {
    int result = process(5);
    std::cout << "Result: " << result << std::endl;
    return 0;
}
EOF

# Generate flowchart
agent5-v4 flowchart \
    --project-path test_project \
    --entry-function main \
    --entry-file main.cpp \
    --detail-level medium

# Should output a Mermaid flowchart to stdout
```

---

## Configuration

### Environment Variables

Agent5 V4 respects the following environment variables:

- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)
- `AGENT5_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

**Example:**
```bash
export OLLAMA_HOST=http://my-ollama-server:11434
export AGENT5_LOG_LEVEL=DEBUG
```

### Logging

Enable verbose logging with the `-v` or `--verbose` flag:

```bash
agent5-v4 --verbose flowchart \
    --project-path ./myproject \
    --entry-function main
```

---

## Troubleshooting

### Issue: "agent5-v4: command not found"

**Cause:** Entry point not installed or virtual environment not activated.

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Reinstall in development mode
pip install -e .
```

### Issue: "No module named 'clang'"

**Cause:** libclang not installed correctly.

**Solution:**
```bash
# Reinstall libclang
pip uninstall libclang
pip install libclang==18.1.1

# On Linux, ensure system libclang is available
sudo apt-get install libclang-dev
```

### Issue: "Ollama connection failed"

**Cause:** Ollama not running or wrong host.

**Solution:**
```bash
# Start Ollama
ollama serve

# Or specify custom host
export OLLAMA_HOST=http://localhost:11434
```

### Issue: "Model 'llama3.2:3b' not found"

**Cause:** Model not pulled.

**Solution:**
```bash
ollama pull llama3.2:3b
```

### Issue: Clang parse errors for complex C++ code

**Cause:** V4 uses basic C++17 parsing flags.

**Solution:**
- Ensure code uses standard C++ (avoid heavy macros/templates)
- Consider generating `compile_commands.json` (future feature)

---

## Upgrading from V3

If you have Agent5 V3 installed:

1. **Both versions can coexist:**
   - V3: `agent5` command
   - V4: `agent5-v4` command

2. **Update dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install V4:**
   ```bash
   pip install -e .
   ```

4. **No migration of data needed:**
   - V3 uses RAG database (ChromaDB)
   - V4 analyzes on-the-fly (no database)

---

## Development Setup

For contributors and developers:

```bash
# Clone and setup
git clone <repository-url>
cd Agent5
git checkout version4

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests (if available)
pytest tests/

# Check linting
flake8 agent5/

# Type checking
mypy agent5/
```

---

## Uninstallation

```bash
# Uninstall Agent5
pip uninstall agent5-v4

# Remove virtual environment
rm -rf venv

# Optionally remove Ollama models
ollama rm llama3.2:3b
```

---

## Next Steps

After successful installation:

1. Read the **README_v4.md** for usage instructions
2. Check **CHANGELOG_v4.md** for version details
3. Try the examples in **README_v4.md**
4. Explore intermediate outputs with `--output-dir`

---

## Support

For issues:
1. Check this installation guide
2. Review **README_v4.md** troubleshooting section
3. Open an issue on GitHub with:
   - Python version
   - OS and version
   - Full error message
   - Command used

---

**Happy Analyzing!** 🚀




