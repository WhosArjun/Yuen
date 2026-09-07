"""
Tests for agent/llm.py. These use a mock instead of a real Ollama
server so the test suite can run in CI/sandboxes that don't have Ollama
installed. See test_agent_loop.py and README.md for how to do a real,
end-to-end manual smoke test against an actual local model.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.llm import OllamaLLM, LLMResponse


@pytest.fixture
def llm_with_mock_client():
    with patch("agent.llm.Client") as MockClient:
        instance = MockClient.return_value
        llm = OllamaLLM(model="llama3.1:8b", host="http://localhost:11434")
        llm.client = instance
        yield llm, instance


def test_check_server_running_true(llm_with_mock_client):
    llm, client = llm_with_mock_client
    client.list.return_value = {"models": []}
    assert llm.check_server_running() is True


def test_check_server_running_false(llm_with_mock_client):
    llm, client = llm_with_mock_client
    client.list.side_effect = ConnectionError("no server")
    assert llm.check_server_running() is False


def test_check_model_available_true(llm_with_mock_client):
    llm, client = llm_with_mock_client
    client.list.return_value = {"models": [{"model": "llama3.1:8b"}]}
    assert llm.check_model_available() is True


def test_check_model_available_false(llm_with_mock_client):
    llm, client = llm_with_mock_client
    client.list.return_value = {"models": [{"model": "mistral:7b"}]}
    assert llm.check_model_available() is False


def test_chat_parses_plain_text_response(llm_with_mock_client):
    llm, client = llm_with_mock_client
    client.chat.return_value = {"message": {"content": "hello!", "tool_calls": None}}
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert isinstance(resp, LLMResponse)
    assert resp.content == "hello!"
    assert resp.tool_calls == []


def test_chat_parses_tool_call_response(llm_with_mock_client):
    llm, client = llm_with_mock_client
    client.chat.return_value = {
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "filesystem_list", "arguments": {"path": "."}}}
            ],
        }
    }
    resp = llm.chat([{"role": "user", "content": "list files"}], tools=[{"type": "function"}])
    assert resp.tool_calls == [{"name": "filesystem_list", "arguments": {"path": "."}}]
