# Agent5 — Enhanced C++ Project Understanding & Flowchart Generation

Agent5 is an advanced AI agent that deeply understands C++ projects through **AST-aware analysis** and generates **accurate scenario-based flowcharts**. It addresses the limitations of previous approaches by implementing deterministic, rule-based scenario extraction before any LLM involvement.

## Key Features

### 🎯 Core Capabilities
- **AST-Aware Chunking**: Semantic code understanding through Abstract Syntax Tree analysis
- **Scenario Flow Model (SFM)**: Deterministic, rule-based flow extraction
- **RAG with Semantic Search**: Vector-based retrieval with AST-aware chunking
- **Accurate Flowcharts**: Generate Mermaid diagrams from C++ code scenarios
- **Open-Source Stack**: Uses only open-source models and frameworks

### 🏗️ Architecture

The agent implements a strict pipeline that ensures reliability:

```
C++ Code
  ↓
AST + CFG Analysis (Tree-sitter)
  ↓
Scenario Extraction (RULE-BASED, DETERMINISTIC)
  ↓
Scenario Flow Model (JSON) ← VALIDATION GATE
  ↓
LLM (TRANSLATOR ONLY, OPTIONAL)
  ↓
Mermaid Flowchart
```

### 🔑 Key Principles

1. **Deterministic First**: Scenario extraction is rule-based, not LLM-based
2. **Fail Fast**: If SFM cannot be built, the agent refuses to proceed (no guessing)
3. **Semantic Actions**: Function calls are collapsed into semantic steps (never auto-descend)
4. **Boundary Rules**: Include only scenario-relevant nodes (exclude logging, metrics, utilities)
5. **Validated Models**: SFM must have exactly 1 start, at least 1 end, and valid flow

## Requirements

- **Python 3.11+**
- **Ollama** installed and running (default: `http://localhost:11434`)
- Required Ollama models:
  - Chat model: `qwen3:8b` (or `qwen2.5-coder`, `llama3.1`, etc.)
  - Embedding model: `jina/jina-embeddings-v2-base-en`

## Installation

### 1. Clone and Setup

```bash
cd Agent5
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 2. Install Ollama Models

```bash
# Install chat model
ollama pull qwen3:8b

# Install embedding model
ollama pull jina/jina-embeddings-v2-base-en
```

## Usage

Agent5 provides three main commands: `index`, `ask`, and `flowchart`.

### 1. Index a C++ Project

Index your C++ project using AST-aware chunking:

```bash
python -m agent5 index \
  --project_path /path/to/your/cpp/project \
  --collection my_project \
  --clear
```

**Options:**
- `--project_path`: Path to the C++ project root (required)
- `--collection`: Name for the vector store collection (required)
- `--scope`: Optional path to limit indexing to a subdirectory
- `--clear`: Clear existing collection before indexing
- `--embed_model`: Override embedding model
- `--ollama_base_url`: Override Ollama URL

### 2. Ask Questions (RAG)

Ask questions about your indexed project:

```bash
python -m agent5 ask \
  --collection my_project \
  --question "Explain the startup flow of the system"
```

**With focus file** (for better context on specific files):

```bash
python -m agent5 ask \
  --collection my_project \
  --project_path /path/to/project \
  --focus src/main.cpp \
  --question "How does the main function initialize the system?"
```

**Options:**
- `--collection`: Collection name (required)
- `--question`: Your question (required)
- `--k`: Number of chunks to retrieve (default: 10)
- `--focus`: Focus on a specific file
- `--project_path`: Project root (for resolving focus path)
- `--chat_model`: Override chat model
- `--embed_model`: Override embedding model

### 3. Generate Flowcharts

Generate a scenario-based flowchart from C++ code:

```bash
python -m agent5 flowchart \
  --file /path/to/source.cpp \
  --out flowchart.mmd \
  --function main \
  --max_steps 30
```

**Auto-detect entry function:**

```bash
python -m agent5 flowchart \
  --file src/handler.cpp \
  --out handler_flow.mmd \
  --max_steps 40
```

**With LLM translation** (optional, has deterministic fallback):

```bash
python -m agent5 flowchart \
  --file src/handler.cpp \
  --out handler_flow.mmd \
  --use_llm \
  --chat_model qwen3:8b
```

**Options:**
- `--file`: Input C++ source file (required)
- `--out`: Output .mmd file path (required)
- `--function`: Entry function name (auto-detect if omitted)
- `--max_steps`: Maximum steps in flowchart (default: 30)
- `--use_llm`: Use LLM for Mermaid translation (optional)
- `--chat_model`: Chat model for LLM translation
- `--ollama_base_url`: Override Ollama URL

### Viewing Flowcharts

The generated `.mmd` files contain Mermaid flowchart code. View them using:

1. **Mermaid Live Editor**: https://mermaid.live/
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **GitHub/GitLab**: Mermaid diagrams render automatically in markdown files

## Configuration

Agent5 can be configured via environment variables:

```bash
# Ollama settings
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_CHAT_MODEL="qwen3:8b"
export OLLAMA_EMBED_MODEL="jina/jina-embeddings-v2-base-en"

# ChromaDB storage
export CHROMA_DIR=".chroma"

# AST chunking settings
export AGENT5_MIN_CHUNK_LINES="10"
export AGENT5_MAX_CHUNK_LINES="500"
export AGENT5_CHUNK_OVERLAP_LINES="20"
```

## What Makes Agent5 Better?

### Compared to Agent3

Agent5 improves upon Agent3 with:

1. **AST-Aware Chunking**: Instead of arbitrary text splitting, code is chunked by semantic units (functions, classes, namespaces) with preserved context

2. **Enhanced Scenario Extraction**: 
   - Stricter boundary rules (exclude noise)
   - Better semantic classification of function calls
   - Improved handling of complex control flow
   - Validated SFM before proceeding

3. **Fail-Fast Philosophy**: 
   - SFM MUST be valid before LLM is called
   - Clear error messages when extraction fails
   - No "best effort" guessing

4. **Better Metadata**: AST chunks include:
   - Qualified names (e.g., `Namespace::Class::method`)
   - Dependencies (functions/classes referenced)
   - Chunk type (function, class, namespace, header)
   - Start/end line numbers

### Key Innovations

**Scenario Boundary Rules:**
- ✅ Include: Argument parsing, validation, business decisions, state changes, returns
- ❌ Exclude: Logging, metrics, utility helpers, deep internal calls

**Semantic Action Collapse:**
- Instead of showing every function call, collapse into semantic actions:
  - `parseConfig()` → "Parse configuration"
  - `validateInput()` → "Validate input"
  - `createConnection()` → "Create connection"

**Strict Validation:**
- Exactly 1 start node required
- At least 1 end node required
- All edges must reference valid nodes
- All branches must terminate or rejoin

## Examples

### Example 1: Simple CLI Program

```cpp
// calculator.cpp
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage();
        return 1;
    }
    
    int a = parseNumber(argv[1]);
    int b = parseNumber(argv[2]);
    
    if (!isValid(a) || !isValid(b)) {
        logError("Invalid numbers");
        return 1;
    }
    
    int result = add(a, b);
    printResult(result);
    return 0;
}
```

**Generate flowchart:**

```bash
python -m agent5 flowchart --file calculator.cpp --out calc_flow.mmd
```

**Result:** A clean flowchart showing:
- Start
- Parse arguments (semantic action, not individual calls)
- Validate inputs (decision)
- Add numbers (semantic action)
- Print result
- End

### Example 2: RAG Query

```bash
python -m agent5 ask \
  --collection poseidonos \
  --question "How does the volume creation flow work?" \
  --focus src/volume/volume_creator.cpp
```

**Result:** Agent5 will:
1. Retrieve relevant semantic chunks (functions, classes)
2. Include the focus file in full
3. Use AST metadata to understand relationships
4. Provide a step-by-step explanation with file citations

## Architecture Details

### AST Chunking

Agent5 uses Tree-sitter to parse C++ code into an AST, then extracts semantic chunks:

1. **Headers**: Includes, macros, forward declarations
2. **Namespaces**: Namespace definitions with contents
3. **Classes/Structs**: Class definitions with methods
4. **Functions**: Individual function implementations

Each chunk includes:
- Full source code
- Qualified name (e.g., `MyNamespace::MyClass::myMethod`)
- Dependencies (other entities referenced)
- Line numbers
- Scope information

### Scenario Extraction

The scenario extractor uses a rule-based approach:

1. **Parse AST**: Use Tree-sitter to build the syntax tree
2. **Identify Entry**: Find the entry function (explicit or auto-detect)
3. **Walk CFG**: Traverse control flow deterministically
4. **Apply Rules**: Include/exclude nodes based on boundary rules
5. **Classify Calls**: Collapse function calls into semantic actions
6. **Build SFM**: Create validated Scenario Flow Model
7. **Validate**: Ensure SFM meets structural requirements

### SFM Structure

A Scenario Flow Model is a JSON structure:

```json
{
  "nodes": [
    {"id": "start", "type": "terminator", "label": "Start"},
    {"id": "p1", "type": "process", "label": "Parse arguments"},
    {"id": "d1", "type": "decision", "label": "Valid input?"},
    {"id": "p2", "type": "process", "label": "Calculate result"},
    {"id": "end", "type": "terminator", "label": "End"}
  ],
  "edges": [
    {"src": "start", "dst": "p1"},
    {"src": "p1", "dst": "d1"},
    {"src": "d1", "dst": "p2", "label": "YES"},
    {"src": "d1", "dst": "end", "label": "NO"},
    {"src": "p2", "dst": "end"}
  ]
}
```

## Limitations

- **Single-function focus**: Current scenario extraction works best on a single entry function
- **Complex macros**: Heavy macro usage may reduce accuracy (Tree-sitter limitation)
- **Templates**: Template-heavy code may need manual entry function specification
- **Cross-file tracing**: Multi-file scenarios require focus file specification

## Troubleshooting

### "Ollama model not found"
```bash
# Pull the required models
ollama pull qwen3:8b
ollama pull jina/jina-embeddings-v2-base-en
```

### "Cannot build SFM"
- Ensure the focus file contains the entry function
- Specify `--function` explicitly if auto-detection fails
- Check that the function has a body (not just declaration)

### "No documents found to index"
- Verify `--project_path` points to a directory with C++ files
- Check that files have proper extensions (`.cpp`, `.h`, `.hpp`, etc.)
- Use `--scope` to limit to a specific subdirectory if needed

## Development

### Project Structure

```
agent5/
├── __init__.py           # Package initialization
├── __main__.py           # Module entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration settings
├── ast_chunker.py       # AST-aware code chunking
├── scenario_extractor.py # Scenario Flow Model extraction
├── flowchart.py         # Mermaid flowchart generation
├── cpp_loader.py        # C++ project loader
├── rag_system.py        # RAG question-answering
├── indexer.py           # Vector store indexing
├── vectorstore.py       # ChromaDB integration
├── ollama_compat.py     # Ollama model interface
├── fs_utils.py          # Filesystem utilities
└── logging_utils.py     # Logging utilities
```

### Running Tests

```bash
# Test on a simple C++ file
python -m agent5 flowchart \
  --file examples/simple.cpp \
  --out test_flow.mmd

# Test indexing
python -m agent5 index \
  --project_path examples/sample_project \
  --collection test \
  --clear

# Test RAG
python -m agent5 ask \
  --collection test \
  --question "What does the main function do?"
```

## References

- **DocAgent**: Inspired by Facebook Research's approach to AST-based code understanding
- **Tree-sitter**: Used for robust C++ parsing
- **LangChain/LangGraph**: Framework for RAG and agent workflows
- **Mermaid**: Flowchart rendering format

## License

[Specify your license here]

## Contributing

Contributions welcome! Please:
1. Test on real C++ projects
2. Report issues with example code
3. Suggest improvements to scenario extraction rules
4. Add support for more C++ patterns

## Version History

### v2.0.0 (Agent5)
- Complete rewrite with AST-aware chunking
- Enhanced scenario extraction with SFM
- Fail-fast validation
- Improved semantic understanding
- Better error messages

### v1.0.0 (Agent3)
- Initial version with basic RAG and flowcharts
- Tree-sitter integration
- Simple text-based chunking

