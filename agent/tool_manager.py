"""
agent/tool_manager.py

Registers every tool as an OpenAI-style function-calling schema (which
Ollama's tool-calling models understand natively), and dispatches calls
made by the LLM to the real Python implementations in tools/.

The LLM decides which tool to call and with what arguments -- this
module does not guess intent. It only:
    1. validates that the requested tool exists and required args are present
    2. checks whether the call needs human confirmation (destructive ops)
    3. executes the tool
    4. returns a structured result back to the caller (agent.py), which
       feeds it back to the LLM as a "tool" message

Tools that always require confirmation (dangerous by nature), regardless
of arguments:
    - filesystem_delete_file
    - terminal_run_command (only when it matches a dangerous pattern --
      checked dynamically, not statically listed here)
    - git_commit is NOT dangerous (local only); there is no push tool at all
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from tools import filesystem, terminal, python_exec, git as git_tool, web as web_tool

# Tool names that ALWAYS require explicit user confirmation before running,
# no matter what arguments are passed.
ALWAYS_CONFIRM = {"filesystem_delete_file"}

# Tools that require confirmation *conditionally* (decided by the tool
# implementation itself, e.g. terminal matches a dangerous pattern). The
# tool manager surfaces `requires_confirmation: True` results from these
# back to the user as a confirmation prompt.
CONDITIONAL_CONFIRM = {"terminal_run_command"}


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "filesystem_list",
            "description": "List files and directories at a path inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the workspace. Use '.' for the root."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_read",
            "description": "Read the contents of a text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path to the file."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_write",
            "description": "Create or overwrite a text file inside the workspace with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "Full text content to write."},
                    "overwrite": {"type": "boolean", "description": "Allow overwriting an existing file. Default true."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_edit",
            "description": "Edit a file by replacing an exact snippet of text (`find`) with new text (`replace`). Fails if `find` is not present, so read the file first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "find": {"type": "string", "description": "Exact existing text to find."},
                    "replace": {"type": "string", "description": "Text to replace it with."},
                },
                "required": ["path", "find", "replace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_delete_file",
            "description": "Delete a file or empty directory inside the workspace. ALWAYS requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "confirmed": {"type": "boolean", "description": "Must be true; set only after the user has explicitly confirmed."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_mkdir",
            "description": "Create a directory (and parents) inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_search",
            "description": "Search for files by glob name pattern (e.g. '*.py'), optionally filtered by file content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py'."},
                    "content_query": {"type": "string", "description": "Optional substring the file content must contain."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_run_command",
            "description": "Run a shell command with the working directory set to the workspace. Dangerous commands require confirmed=true, set only after the user has approved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "confirmed": {"type": "boolean", "description": "Set true only if this command was already shown to and approved by the user."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_run_code",
            "description": "Execute a Python code snippet in a subprocess and capture stdout, stderr and exit code. Use this to test code you just wrote.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_run_file",
            "description": "Execute an existing Python file inside the workspace and capture its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Optional command-line arguments."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the working-tree git status of the workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show a git diff, optionally for a specific path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git commit history.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Number of commits, default 10."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stage files for commit (git add).",
            "parameters": {
                "type": "object",
                "properties": {"paths": {"type": "array", "items": {"type": "string"}, "description": "Paths to stage. Defaults to all ('.')."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Create a local git commit with a message. This is local only -- there is no tool to push to a remote.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "List branches, or create a new one if `name` is given.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_init",
            "description": "Initialize a git repository in the workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web for current, real-time information (free, key-free "
                "DuckDuckGo HTML search). Returns titles, URLs and snippets. "
                "MANDATORY for any question involving current events, news, sports schedules or "
                "results, 'next'/'upcoming'/'latest' releases or games, prices, or who currently "
                "holds a role or position -- your own training data has a knowledge cutoff and is "
                "NOT reliable for these. Always call this tool for such questions rather than "
                "answering from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_remember",
            "description": "Save a small, useful fact about the current project/task for later recall (do not store everything -- only what's worth remembering).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "category": {"type": "string", "description": "Optional grouping label."},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": "Recall a previously saved fact by key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search saved facts by keyword.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "planner_set_plan",
            "description": "Create or replace the current task plan with an ordered list of steps. Use for multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "planner_update_step",
            "description": "Update the status of a plan step by its 0-based index. status must be one of pending, in_progress, done, failed, skipped.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_index": {"type": "integer"},
                    "status": {"type": "string"},
                },
                "required": ["step_index", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "planner_get_plan",
            "description": "Get the current task plan and step statuses.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the real current date and time from the user's computer clock. Use this whenever you need to know today's date -- never guess it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolManager:
    def __init__(self, config, memory, planner, confirm_callback: Optional[Callable[[str, Dict], bool]] = None):
        """
        confirm_callback(tool_name, arguments) -> bool
            Called when a tool call needs human confirmation. Should
            return True to proceed, False to abort. Defaults to a CLI
            input() prompt if not provided.
        """
        self.config = config
        self.memory = memory
        self.planner = planner
        self.confirm_callback = confirm_callback or self._default_confirm

        self._dispatch_table: Dict[str, Callable[[Dict], Dict]] = {
            "filesystem_list": lambda a: filesystem.list_files(self.config.workspace, a.get("path", ".")),
            "filesystem_read": lambda a: filesystem.read_file(self.config.workspace, a["path"]),
            "filesystem_write": lambda a: filesystem.write_file(
                self.config.workspace, a["path"], a["content"], a.get("overwrite", True)
            ),
            "filesystem_edit": lambda a: filesystem.edit_file(
                self.config.workspace, a["path"], a["find"], a["replace"]
            ),
            "filesystem_delete_file": lambda a: filesystem.delete_file(
                self.config.workspace, a["path"], a.get("confirmed", False)
            ),
            "filesystem_mkdir": lambda a: filesystem.create_directory(self.config.workspace, a["path"]),
            "filesystem_search": lambda a: filesystem.search_files(
                self.config.workspace, a["pattern"], a.get("content_query")
            ),
            "terminal_run_command": lambda a: terminal.run_command(
                self.config.workspace,
                a["command"],
                timeout=self.config.command_timeout,
                dangerous_patterns=self.config.dangerous_patterns,
                confirmed=a.get("confirmed", False),
            ),
            "python_run_code": lambda a: python_exec.run_python_code(
                self.config.workspace, a["code"], timeout=self.config.command_timeout
            ),
            "python_run_file": lambda a: python_exec.run_python_file(
                self.config.workspace, a["path"], timeout=self.config.command_timeout, args=a.get("args")
            ),
            "git_status": lambda a: git_tool.status(self.config.workspace),
            "git_diff": lambda a: git_tool.diff(self.config.workspace, a.get("path")),
            "git_log": lambda a: git_tool.log(self.config.workspace, a.get("n", 10)),
            "git_add": lambda a: git_tool.add(self.config.workspace, a.get("paths")),
            "git_commit": lambda a: git_tool.commit(self.config.workspace, a["message"]),
            "git_branch": lambda a: git_tool.branch(self.config.workspace, a.get("name")),
            "git_init": lambda a: git_tool.init(self.config.workspace),
            "web_search": lambda a: web_tool.web_search(a["query"], a.get("max_results", 5)),
            "memory_remember": lambda a: self._memory_remember(a),
            "memory_recall": lambda a: self._memory_recall(a),
            "memory_search": lambda a: {"success": True, "matches": self.memory.search_facts(a["query"])},
            "planner_set_plan": lambda a: self.planner.set_plan(a["steps"]),
            "planner_update_step": lambda a: self.planner.update_step(a["step_index"], a["status"]),
            "planner_get_plan": lambda a: {"success": True, "plan": self.planner.get_plan_text()},
            "get_current_datetime": lambda a: {
                "success": True,
                "current_datetime": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
            },
        }

    def _memory_remember(self, a: Dict) -> Dict:
        self.memory.remember(a["key"], a["value"], a.get("category", "general"))
        return {"success": True, "key": a["key"]}

    def _memory_recall(self, a: Dict) -> Dict:
        value = self.memory.recall(a["key"])
        if value is None:
            return {"success": False, "error": f"No memory found for key '{a['key']}'."}
        return {"success": True, "key": a["key"], "value": value}

    @staticmethod
    def _default_confirm(tool_name: str, arguments: Dict) -> bool:
        print(f"\n[confirmation required] Tool: {tool_name}")
        for k, v in arguments.items():
            print(f"    {k}: {v}")
        answer = input("Proceed? [y/N]: ").strip().lower()
        return answer in ("y", "yes")

    def get_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self._dispatch_table:
            return {"success": False, "error": f"Unknown tool '{tool_name}'."}

        # Tools that always require confirmation, regardless of arguments.
        if tool_name in ALWAYS_CONFIRM and self.config.require_confirmation:
            if not arguments.get("confirmed"):
                approved = self.confirm_callback(tool_name, arguments)
                if not approved:
                    return {"success": False, "error": "User declined to confirm this action."}
                arguments = dict(arguments)
                arguments["confirmed"] = True

        try:
            result = self._dispatch_table[tool_name](arguments)
        except KeyError as e:
            return {"success": False, "error": f"Missing required argument: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Tool '{tool_name}' raised an exception: {e}"}

        # Conditional confirmation: the tool itself flagged that it needs
        # approval (e.g. terminal command matched a dangerous pattern).
        if (
            isinstance(result, dict)
            and result.get("requires_confirmation")
            and self.config.require_confirmation
            and not arguments.get("confirmed")
        ):
            approved = self.confirm_callback(tool_name, arguments)
            if approved:
                arguments = dict(arguments)
                arguments["confirmed"] = True
                try:
                    result = self._dispatch_table[tool_name](arguments)
                except Exception as e:
                    return {"success": False, "error": f"Tool '{tool_name}' raised an exception: {e}"}
            else:
                return {"success": False, "error": "User declined to confirm this action."}

        return result
