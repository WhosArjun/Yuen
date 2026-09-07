# Local Agent

Local Agent is a self-hosted AI coding assistant that runs entirely on local infrastructure. It uses [Ollama](https://ollama.com) to serve an open-source language model and exposes that model to a set of real tools — filesystem access, a terminal, Python execution, Git, memory, and web search — so it can plan and complete multi-step tasks autonomously.

No commercial API key (OpenAI, Anthropic, Gemini, or otherwise) is used or required at any point in this project.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Architecture](#architecture)
7. [Security Model](#security-model)
8. [Testing](#testing)
9. [Extending the Project](#extending-the-project)
10. [Troubleshooting](#troubleshooting)
11. [Known Limitations](#known-limitations)
12. [License](#license)

## Overview

Local Agent implements a standard agent loop: it sends the user's request and the current conversation state to a local model, allows the model to invoke tools as needed, executes those tools, and returns the results to the model for further reasoning. This continues until the model produces a final response or a configured step limit is reached.

Key design principles:

- **No external dependency on paid AI services.** The only network calls this project makes to an LLM are to a local Ollama instance.
- **Model-driven tool use.** Tool invocation is handled through Ollama's native function-calling interface; there is no hard-coded routing logic mapping user phrases to actions.
- **Workspace isolation.** All file and code operations are confined to a configurable workspace directory.
- **Explicit confirmation for destructive actions.** File deletion and shell commands matching known dangerous patterns require user approval before execution.

## Requirements

- Python 3.10 or later
- [Ollama](https://ollama.com), installed and running locally
- An Ollama-compatible model that supports tool calling (e.g., `llama3.1:8b`, `qwen2.5:7b`)
- Approximately 5–8 GB of free disk space for the model weights, depending on model choice

## Installation

### Windows

```powershell
# 1. Install Ollama
irm https://ollama.com/install.ps1 | iex

# 2. Download a model
ollama pull llama3.1:8b

# 3. Set up the project environment
cd local-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 4. Run the agent
python main.py
```

### macOS / Linux

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh      # Linux; use ollama.com/download on macOS

# 2. Download a model
ollama pull llama3.1:8b

# 3. Set up the project environment
cd local-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Run the agent
python main.py
```

If system memory is limited, a smaller model such as `qwen2.5:3b` can be substituted throughout.

## Usage

Once running, the agent presents an interactive prompt:

```
Model:      llama3.1:8b
Workspace:  ./workspace
Ollama:     http://localhost:11434
Max steps:  30
Confirm dangerous actions: True

You: Build me a Python calculator and test it.

Agent: I'll inspect the workspace first.
Agent: [tool: filesystem_list] {"path": "."}
Agent: The workspace is empty. I'll create calculator.py.
Agent: [tool: filesystem_write] {"path": "calculator.py", "content": "..."}
Agent: I'll run it to make sure it works.
Agent: [tool: python_run_file] {"path": "calculator.py"}
Agent: Done. I created and tested calculator.py — it supports +, -, *, / on two numbers.
```

### Session Commands

| Command | Description |
|---|---|
| `plan` | Displays the agent's current step-by-step task plan |
| `exit` / `quit` | Ends the session |
| `[y/N]` prompt | Approves or declines a flagged action |

All files created or modified by the agent are contained within the `workspace/` directory. The agent has no access to files outside this directory.

## Configuration

Configuration can be supplied via command-line arguments, environment variables, or a `config.json` file, in that order of precedence.

### Command-Line Arguments

```bash
python main.py --model qwen2.5:7b --workspace ./my_project --max-steps 50
python main.py --ollama-host http://localhost:11434
python main.py --config myconfig.json
python main.py --keep-alive 1h        # keep the model loaded in memory longer between messages
python main.py --temperature 0.1      # lower = more deterministic, factual answers
python main.py --no-confirm           # disables confirmation prompts; not recommended
```

### config.json

```json
{
  "model": "llama3.1:8b",
  "workspace": "./workspace",
  "max_steps": 30,
  "require_confirmation": true,
  "command_timeout": 60,
  "keep_alive": "30m",
  "temperature": 0.2
}
```

### Environment Variables

`MODEL`, `WORKSPACE`, `MAX_STEPS`, `REQUIRE_CONFIRMATION`, `OLLAMA_HOST`, `COMMAND_TIMEOUT`, `KEEP_ALIVE`, `TEMPERATURE`

### `keep_alive` and Response Speed

By default, Ollama unloads a model from memory 5 minutes after the last request, so if you pause and
come back, the next response has to reload the full model from disk — often the single biggest cause
of a "slow" response. This project sets `keep_alive` to `30m` by default, and passes it on every
request, so the model stays warm for the whole session. Set it higher (`1h`, or `-1` to never unload)
if you have the RAM/VRAM to spare, or lower if you want memory freed up between uses.

## Architecture

```
local-agent/
├── main.py                CLI entry point and startup health checks
├── config.py               Configuration loading (file, environment, CLI)
├── requirements.txt
│
├── agent/
│   ├── agent.py             Core agent loop
│   ├── llm.py               Ollama client and tool-call parsing
│   ├── tool_manager.py      Tool schema registry, dispatch, confirmation logic
│   ├── memory.py            SQLite-backed conversation history and stored facts
│   └── planner.py           Task plan storage and updates
│
├── tools/
│   ├── filesystem.py        Sandboxed file operations
│   ├── terminal.py          Shell command execution
│   ├── python_exec.py       Python code and script execution
│   ├── git.py               Local Git operations (no remote push)
│   ├── web.py               Key-free web search
│   └── sandbox.py           Shared path-resolution and workspace enforcement
│
├── tests/                   Automated test suite
└── workspace/               Default sandbox directory
```

### Agent Loop

1. The user's message is recorded to memory.
2. The agent constructs context consisting of a system prompt (including the active plan) and recent conversation history.
3. This context, along with all available tool schemas, is sent to the local model.
4. If the model requests a tool call, the call is validated, confirmed if required, and executed. The actual result — including any error output — is returned to the model as part of the conversation.
5. Steps 3–4 repeat until the model returns a plain-text response with no further tool calls, or the configured step limit is reached.

This feedback mechanism is also how the agent performs self-correction: if a tool call such as `python_run_code` returns an error, that error is visible to the model in the next step, and the model may revise its approach accordingly. No separate retry logic is implemented for this purpose.

## Security Model

| Control | Implementation |
|---|---|
| Workspace sandboxing | File paths are resolved and validated against the configured workspace root; requests outside this boundary are rejected. |
| Deletion confirmation | File deletion always requires explicit user approval. |
| Command confirmation | Shell commands matching a configurable list of dangerous patterns (e.g., `rm -rf`, `format`, `shutdown`, `git push`, `git reset --hard`) require approval before execution. |
| No remote Git operations | Push, pull, and clone operations are not implemented. Commits are local only. |
| Environment sanitization | Known sensitive environment variable names (API keys, tokens, credentials) are stripped from the environment used for command and code execution. |
| Execution timeouts | Both shell commands and Python execution are subject to configurable timeouts. |

**Limitation.** The terminal tool executes commands through a standard shell to support normal command syntax (pipes, redirection, etc.). While filesystem and Python execution tools enforce a hard sandbox boundary, the terminal tool's protection is limited to pattern matching and user confirmation; it does not prevent a confirmed command from referencing paths outside the workspace. This project is intended for single-user, local use and has not been hardened for multi-user or networked deployment.

## Testing

The test suite uses a mocked LLM and does not require a running Ollama instance.

```bash
pytest tests/ -v
```

| Test Module | Coverage |
|---|---|
| `test_filesystem.py` | File operations and workspace boundary enforcement |
| `test_terminal.py` | Command execution, dangerous-pattern detection, timeouts |
| `test_python_exec.py` | Output capture, exception handling, timeouts |
| `test_llm.py` | Ollama connectivity checks and tool-call parsing |
| `test_tool_manager.py` | Schema validation, dispatch, confirmation logic |
| `test_memory.py` | Conversation history, fact storage, plan persistence |
| `test_agent_loop.py` | End-to-end loop behavior, including a self-correction scenario and step-limit enforcement |

For a manual, end-to-end verification against a live model:

```bash
python main.py --model llama3.1:8b
```
```
You: Create a file called notes.txt with the text "hello" and read it back to me.
```

## Extending the Project

**Adding a tool.** Implement the function in `tools/`, register its schema in `TOOL_SCHEMAS` within `agent/tool_manager.py`, and add it to the dispatch table in `ToolManager`. No changes to the agent loop are required.

**Changing models.** Any Ollama model with tool-calling support can be used via the `--model` flag or `config.json`. No part of the codebase assumes a specific model.

**Adding an interface.** `Agent.run_turn(text) -> str` in `agent/agent.py` is interface-agnostic. `main.py` is a thin CLI wrapper around this method; alternative interfaces (web, desktop) can call it directly.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| "Could not reach a running Ollama server" | Start Ollama with `ollama serve`, or open the Ollama application. |
| "Model 'x' is not downloaded yet" | Run `ollama pull <model-name>` for the model specified in configuration. |
| First response is slow | Expected; the model must load into memory on first use. This can take 30 seconds to several minutes depending on hardware. |
| Tool calls are inconsistent or absent | Model-dependent behavior. Larger, tool-capable models (e.g., `llama3.1:8b` or later, `qwen2.5:7b`+) perform more reliably. |
| Web search returns an error | The search backend depends on a public HTML endpoint and requires network access; failures are reported explicitly rather than substituted with fabricated results. |
| `venv` creation fails with a permission error | The virtual environment likely already exists; activate it directly rather than recreating it. |

## Improving Accuracy

Local models have a training-data cutoff and no inherent awareness of the current date or of events
after that cutoff. Two things in this project address that directly:

- **Real current date/time**, read from the system clock, is injected into every system prompt, and
  is also available on demand via the `get_current_datetime` tool — the model is instructed to treat
  this as ground truth rather than guessing.
- **Mandatory web search for time-sensitive questions.** The `web_search` tool's description and the
  system prompt both explicitly instruct the model that it must search rather than answer from memory
  for anything involving current events, schedules, "next"/"latest" releases, prices, or who currently
  holds a role. If the model still answers from memory on this kind of question, it is a model-quality
  issue — larger or newer tool-calling models (`qwen2.5:7b`+, `qwen3.5:9b`) follow these instructions
  more reliably than smaller ones (`qwen2.5:3b`, `llama3.2:3b`).
- **Lower temperature** (`--temperature 0.1` or so) reduces creative variation in factual answers,
  at the cost of more repetitive phrasing.

If the model consistently ignores the instruction to search, that is not a bug in the agent loop —
it reflects the model's own reliability at following system-prompt instructions, which improves with
model size.

## Known Limitations

- Tool-calling reliability is dependent on the selected model.
- The web search implementation relies on scraping a public search page and may require maintenance if that page's structure changes.
- Terminal command sandboxing is best-effort and not a hard security boundary.
- This project is designed for local, single-user use and is not intended for multi-tenant or networked deployment.

## License

This project is provided as-is for local, personal, and educational use.
