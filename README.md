# LLM Orchestrator

A high-performance orchestration framework for managing multi-agent LLM workflows with fine-grained control over context, tooling, and resource allocation.

## Features

- **Context Management**: Intelligent context assembly with token budgeting
- **Tool Versioning**: Semantic versioning for tools with migration support
- **Cost Control**: Real-time cost monitoring and budget enforcement
- **Sandboxed Execution**: Secure, isolated execution environments
- **Structured Artifacts**: Type-safe data passing between components
- **REPL Interface**: Interactive debugging and control

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-orchestrator.git
   cd llm-orchestrator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```python
from orchestrator.context.assembler import ContextAssembler, ContextBudget

# Initialize the context assembler
assembler = ContextAssembler()

# Define your messages and budget
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

budget = ContextBudget(max_tokens_in=1000)
provider_caps = {"max_tokens": 4000, "max_input_tokens": 4000, "max_output_tokens": 1000}

# Assemble the context
context = await assembler.build(
    messages=messages,
    provider_caps=provider_caps,
    budget=budget
)
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
llm-orchestrator/
├── orchestrator/
│   ├── context/           # Context management and assembly
│   ├── models/            # Data models and schemas
│   ├── tools/             # Tool definitions and registry
│   ├── artifacts/         # Artifact storage and management
│   ├── sandbox/           # Sandbox execution environments
│   └── prompts/           # Versioned prompt templates
├── tests/                 # Test suite
├── .env.example          # Example environment variables
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Configuration

Copy `.env.example` to `.env` and update the values as needed:

```bash
cp .env.example .env
```

## License

MIT
# LLM-Orchestrator
