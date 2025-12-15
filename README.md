# Agent3 — C++ Project RAG + Flowchart Agent (Ollama + LangGraph)

This agent:
- Indexes any **C++ project folder** into a **Chroma** vector DB (RAG).
- Answers questions about the project using **LangGraph + Ollama**.
- Generates **Mermaid flowchart diagrams** for the whole project or a scoped module path using **tree-sitter** static analysis (function + call edges).

## Requirements
- Python **3.11+**
- Ollama installed and running (defaults to `http://localhost:11434`)
- An Ollama **chat model** and **embedding model** pulled (defaults below).

## Quick start

### 1) Create venv + install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Index a C++ repo (example: Poseidonos)

```bash
python -m agent3 index --project_path /path/to/poseidonos --collection poseidonos
```

### 3) Ask questions (RAG)

```bash
python -m agent3 ask --collection poseidonos --question "What is the startup flow of the system?"
```

Ask with a focus file (higher accuracy for file-specific questions):

```bash
python -m agent3 ask \
  --collection poseidonos \
  --project_path /path/to/poseidonos \
  --focus src/cli/create_volume_command.cpp \
  --question "Explain the execution flow when the user runs the create volume CLI"
```

### 4) Generate flowchart (Mermaid)

Project-wide (may be large; use `--max_nodes` to keep it readable):

```bash
python -m agent3 flowchart --project_path /path/to/poseidonos --scope /path/to/poseidonos --out flowchart.mmd --max_nodes 120
```

Scoped module:

```bash
python -m agent3 flowchart --project_path /path/to/poseidonos --scope /path/to/poseidonos/src --out src_flow.mmd --max_nodes 120
```

Scenario-driven (execution) flowchart (whiteboard-style):

```bash
python -m agent3 flowchart \
  --project_path /path/to/poseidonos \
  --out create_volume.mmd \
  --scenario "Create a volume via CLI" \
  --collection poseidonos \
  --focus src/cli/create_volume_command.cpp \
  --model qwen3:8b
```

### View Mermaid diagrams
- Paste the `.mmd` output into any Mermaid viewer (e.g. Mermaid Live Editor) or render in VS Code with Mermaid extensions.

## Configuration (env vars)

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL` (default: `qwen3`)
- `OLLAMA_EMBED_MODEL` (default: `jina/jina-embeddings-v2-base-en`)
- `CHROMA_DIR` (default: `.chroma`)

You can also override per command:
- `agent3 ask --model qwen3:8b ...`

## Notes
- This is designed to work on **any** C++ project without repo-specific assumptions.
- Flowcharts are built from a **static call graph** (best-effort). Macros/templates can reduce static accuracy; the agent will still produce a consistent diagram from parsed call sites.


