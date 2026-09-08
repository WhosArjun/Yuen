"""
agent/agent.py

The real agent loop:

    1. Receive user request.
    2. Send request + recent context to the local LLM, along with the
       full list of available tool schemas.
    3. The LLM decides (on its own -- no hard-coded if/else routing)
       whether it needs a tool, and which one.
    4. If it requests a tool call, validate + (if needed) confirm it.
    5. Execute the tool.
    6. Feed the tool result back to the LLM as a "tool" role message.
    7. The LLM analyzes the result and may call another tool (this is
       exactly how self-correction happens: a failed python_run_code
       result is visible to the model, which can then edit the file
       and try again) or produce a final text answer.
    8. Repeat until the LLM gives a plain-text answer, or MAX_STEPS is
       hit, and always report to the user what happened.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .llm import OllamaLLM, LLMResponse
from .memory import Memory
from .planner import Planner
from .tool_manager import ToolManager

SYSTEM_PROMPT_TEMPLATE = """You are a local, autonomous coding & task agent running entirely on the \
user's own machine via a local Ollama model. You have NO internet-based AI backend -- you ARE the \
local model. You have tools to inspect and modify a sandboxed workspace directory, run shell \
commands, execute Python, use git locally, search the web, remember facts, and track a step-by-step \
plan.

The real current date and time, read directly from the user's computer clock, is: {current_datetime}
Treat this as ground truth. Do not guess or state a different current date.

Workspace: {workspace}
Current plan:
{plan}

Guidelines:
- Decide for yourself when a tool call is needed. Do not claim you ran a command or wrote a file \
unless you actually called the corresponding tool.
- Your own training data has a knowledge cutoff and does NOT include anything that happened after \
that cutoff. You cannot know, from memory alone, anything about: current events, news, scores or \
schedules, "next" or "upcoming" or "latest" releases/games/versions, who currently holds a role or \
position, prices, or any other fact that changes over time. For ANY question like this, you MUST \
call web_search before answering -- do not answer from memory and do not guess a year or a plausible-\
sounding fact. If web_search fails or returns nothing useful, say so plainly instead of guessing.
- For multi-step tasks, call planner_set_plan first, and update step statuses with \
planner_update_step as you make progress.
- After writing code, use python_run_code or python_run_file to actually test it. If it fails, read \
the stderr/traceback, fix the code, and try again -- do not just report the failure to the user.
- Destructive or dangerous actions (deleting files, risky shell commands) require user confirmation; \
if a tool result says it needs confirmation, explain briefly what you want to do and why, and note \
that you're waiting for approval -- the user will be prompted directly by the system.
- Never invent tool output. Only report what a tool call actually returned.
- Keep the user informed with short status updates between tool calls (e.g. "Reading the file...", \
"Running the tests...").
- When the task is complete, give a clear, concise final summary of what you did.
"""


class MaxStepsExceeded(Exception):
    pass


class Agent:
    def __init__(self, config, session_id: str = "default", confirm_callback=None,
                 status_callback: Optional[callable] = None):
        self.config = config
        self.llm = OllamaLLM(model=config.model, host=config.ollama_host)
        self.memory = Memory(config.memory_db, session_id=session_id)
        self.planner = Planner(self.memory)
        self.tool_manager = ToolManager(config, self.memory, self.planner, confirm_callback=confirm_callback)
        # status_callback(message: str) -> None, used to stream progress to the UI
        self.status_callback = status_callback or (lambda msg: None)

    def _system_prompt(self) -> str:
        now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
        return SYSTEM_PROMPT_TEMPLATE.format(
            current_datetime=now,
            workspace=self.config.workspace,
            plan=self.planner.get_plan_text(),
        )

    def _build_messages(self) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.memory.get_recent_messages(limit=30))
        return messages

    def _requires_live_information(self, text: str) -> bool:
        lower = text.lower()
        if not lower.strip():
            return False

        live_markers = [
            "current", "currently", "latest", "newest", "news",
            "today", "tomorrow", "this week", "this month", "this year",
            "next year", "next year's", "upcoming", "next", "now",
            "schedule", "scores", "results", "winner", "prices",
            "who currently", "who is currently", "what is the current",
        ]
        if any(marker in lower for marker in live_markers):
            return True

        # Specific dynamic facts that should never be guessed from model memory.
        if re.search(r"\b(?:frc|first robotics|robotics competition)\b", lower):
            return True

        # Strong “time-sensitive fact” patterns: “next/this/latest/now + entity”
        if re.search(
            r"\b(?:next|latest|current|upcoming|this|today|tomorrow)\b.*\b(?:game|release|event|schedule|winner|price|result|version|roster|president|leader|team)\b",
            lower,
        ):
            return True

        return False

    def _prepare_live_information(self, user_input: str) -> Optional[Dict[str, Any]]:
        if not self._requires_live_information(user_input):
            return None

        base_query = user_input.strip()
        if len(base_query) > 180:
            base_query = base_query[:180].strip()

        self.status_callback("Checking current information before answering.")
        result = self.tool_manager.dispatch("web_search", {"query": base_query, "max_results": 5})

        self.memory.add_message("assistant", "I’m checking current information before answering.")
        self.memory.add_message("tool", f"[web_search result] {json.dumps(result, ensure_ascii=False)[:8000]}")

        if not result.get("success"):
            return {"success": False, "error": result.get("error", "Web search failed."), "query": base_query}
        return result

    def run_turn(self, user_input: str) -> str:
        """
        Runs one full user turn: may involve many tool calls internally,
        but returns a single final text response.
        """
        self.memory.add_message("user", user_input)
        live_check = self._prepare_live_information(user_input)
        if live_check is not None and not live_check.get("success"):
            final = (
                "I can’t answer that accurately because I could not verify it using current web information. "
                "For time-sensitive facts like upcoming FRC games, I need a successful web lookup before answering."
            )
            self.memory.add_message("assistant", final)
            return final

        for step in range(1, self.config.max_steps + 1):
            messages = self._build_messages()
            try:
                response: LLMResponse = self.llm.chat(
                    messages,
                    tools=self.tool_manager.get_schemas(),
                    temperature=self.config.temperature,
                    keep_alive=self.config.keep_alive,
                )
            except Exception as e:
                final = f"[error] Could not get a response from the local model: {e}"
                self.memory.add_message("assistant", final)
                return final

            if response.tool_calls:
                # Record the assistant's intent to call tools (as content, if any)
                if response.content:
                    self.status_callback(response.content)
                    self.memory.add_message("assistant", response.content)

                for call in response.tool_calls:
                    name = call.get("name")
                    args = call.get("arguments") or {}
                    self.status_callback(f"[tool: {name}] {_short_args(args)}")

                    result = self.tool_manager.dispatch(name, args)
                    result_text = json.dumps(result, ensure_ascii=False)[:8000]

                    # Feed the tool call + its result back into context so the
                    # model can see exactly what happened and self-correct.
                    self.memory.add_message(
                        "assistant",
                        f"[called tool {name} with args {json.dumps(args, ensure_ascii=False)}]",
                    )
                    self.memory.add_message("tool", f"[{name} result] {result_text}")

                # Loop back: let the model see the tool results and decide next step
                continue

            # No tool calls -> this is the final answer for this turn.
            final_text = response.content.strip() if response.content else "(no response)"
            self.memory.add_message("assistant", final_text)
            return final_text

        final = (
            f"[stopped] Reached the maximum of {self.config.max_steps} steps for this turn "
            "without a final answer. You can ask me to continue."
        )
        self.memory.add_message("assistant", final)
        return final


def _short_args(args: Dict[str, Any], max_len: int = 120) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."
