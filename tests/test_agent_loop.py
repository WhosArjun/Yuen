"""
End-to-end tests for agent/agent.py's tool-calling loop, using a scripted
fake LLM instead of a real Ollama server. This proves the agent loop
itself (not the model) correctly: dispatches tool calls, feeds results
back, supports multi-step self-correction, and respects MAX_STEPS.
"""

import tempfile
import shutil
from unittest.mock import patch, MagicMock

import pytest

from config import Config
from agent.llm import LLMResponse


class ScriptedLLM:
    """A fake LLM that returns a pre-scripted sequence of responses,
    one per .chat() call, so we can simulate multi-turn tool use."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, temperature=0.2, keep_alive="30m"):
        self.calls.append(messages)
        if not self.responses:
            return LLMResponse(content="(no more scripted responses)", tool_calls=[])
        return self.responses.pop(0)


@pytest.fixture
def workspace():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_agent(workspace, scripted_responses, max_steps=10):
    cfg = Config()
    cfg.workspace = workspace
    cfg.memory_db = f"{workspace}/mem.sqlite3"
    cfg.max_steps = max_steps
    cfg.require_confirmation = True
    cfg.ensure_dirs()

    with patch("agent.agent.OllamaLLM") as MockLLMClass:
        MockLLMClass.return_value = ScriptedLLM(scripted_responses)
        from agent.agent import Agent
        agent = Agent(cfg, session_id="test", confirm_callback=lambda n, a: True)
    return agent


def test_simple_tool_call_then_final_answer(workspace):
    responses = [
        LLMResponse(content="I'll write the file now.", tool_calls=[
            {"name": "filesystem_write", "arguments": {"path": "hello.py", "content": "print('hi')"}}
        ]),
        LLMResponse(content="Done, I created hello.py.", tool_calls=[]),
    ]
    agent = make_agent(workspace, responses)
    final = agent.run_turn("create a hello world script")
    assert "hello.py" in final or "Done" in final

    import os
    assert os.path.exists(f"{workspace}/hello.py")


def test_self_correction_loop(workspace):
    """
    Simulates: model writes buggy code -> runs it -> sees the error ->
    fixes it -> runs again -> succeeds -> gives final answer. This is the
    real self-correction path, driven entirely by tool results being fed
    back into context (no scripted if/else in the agent).
    """
    responses = [
        LLMResponse(content="Writing initial (buggy) script.", tool_calls=[
            {"name": "filesystem_write", "arguments": {"path": "calc.py", "content": "print(1/0)"}}
        ]),
        LLMResponse(content="Testing it.", tool_calls=[
            {"name": "python_run_file", "arguments": {"path": "calc.py"}}
        ]),
        LLMResponse(content="It failed with ZeroDivisionError, fixing.", tool_calls=[
            {"name": "filesystem_write", "arguments": {"path": "calc.py", "content": "print(1/1)"}}
        ]),
        LLMResponse(content="Testing again.", tool_calls=[
            {"name": "python_run_file", "arguments": {"path": "calc.py"}}
        ]),
        LLMResponse(content="Fixed and verified. calc.py now runs successfully.", tool_calls=[]),
    ]
    agent = make_agent(workspace, responses, max_steps=10)
    final = agent.run_turn("write a script that divides two numbers and test it")
    assert "fixed" in final.lower() or "success" in final.lower()


def test_max_steps_exceeded(workspace):
    # Always returns a tool call, never a final answer -> should hit max_steps
    responses = [
        LLMResponse(content="looping", tool_calls=[
            {"name": "git_status", "arguments": {}}
        ])
        for _ in range(20)
    ]
    agent = make_agent(workspace, responses, max_steps=3)
    final = agent.run_turn("do something forever")
    assert "maximum" in final.lower()


def test_dangerous_delete_declined(workspace):
    with open(f"{workspace}/keep.txt", "w") as f:
        f.write("data")

    responses = [
        LLMResponse(content="Deleting file.", tool_calls=[
            {"name": "filesystem_delete_file", "arguments": {"path": "keep.txt"}}
        ]),
        LLMResponse(content="Could not delete, user declined.", tool_calls=[]),
    ]

    cfg = Config()
    cfg.workspace = workspace
    cfg.memory_db = f"{workspace}/mem.sqlite3"
    cfg.require_confirmation = True
    cfg.ensure_dirs()

    with patch("agent.agent.OllamaLLM") as MockLLMClass:
        MockLLMClass.return_value = ScriptedLLM(responses)
        from agent.agent import Agent
        agent = Agent(cfg, session_id="test2", confirm_callback=lambda n, a: False)
        agent.run_turn("delete keep.txt")

    import os
    assert os.path.exists(f"{workspace}/keep.txt")


def test_time_sensitive_question_triggers_web_search_guard(workspace):
    responses = [
        LLMResponse(content="I checked the current information and the game is ...", tool_calls=[]),
    ]
    agent = make_agent(workspace, responses)

    with patch.object(agent.tool_manager, "dispatch", return_value={
        "success": True,
        "query": "2026 FIRST Robotics Competition game",
        "results": [{"title": "FRC 2026 game", "url": "https://example.com", "snippet": "The 2026 game is ..."}],
    }) as fake_dispatch:
        final = agent.run_turn("What is next year's FRC game?")

    assert "FRC" in final or "current" in final.lower() or "game" in final.lower()
    assert any(call.args[0] == "web_search" for call in fake_dispatch.call_args_list)


def test_time_sensitive_question_refuses_if_web_check_fails(workspace):
    responses = [
        LLMResponse(content="I know the answer from memory: it's game X.", tool_calls=[]),
    ]
    agent = make_agent(workspace, responses)

    with patch.object(agent.tool_manager, "dispatch", return_value={
        "success": False,
        "error": "No results returned (search backend may be unreachable, or blocked network access).",
        "query": "What is next year's FRC game?",
    }):
        final = agent.run_turn("What is next year's FRC game?")

    assert "cannot answer" in final.lower() or "verify" in final.lower() or "current information" in final.lower()
