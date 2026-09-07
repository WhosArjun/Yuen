"""
agent/llm.py

Thin wrapper around a local Ollama server. This is the ONLY place in the
project that talks to the model. There is no OpenAI/Anthropic/Gemini
client anywhere -- this module only ever calls http://<ollama_host>.

Requires the `ollama` python package (pip install ollama), which is a
thin HTTP client for the locally-running `ollama serve` daemon. No API
key is used or accepted.

Uses Ollama's native tool-calling support (the `tools` parameter on
/api/chat), which is available for tool-capable models such as
llama3.1, qwen2.5, mistral-nemo, firefunction-v2, etc. If the selected
model does not support tool calling, Ollama will simply return a plain
text response with no tool_calls, and the agent loop treats that as a
final answer.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import ollama
    from ollama import Client
    OLLAMA_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - exercised only when dep missing
    ollama = None
    Client = None
    OLLAMA_IMPORT_ERROR = e


class OllamaNotInstalledError(RuntimeError):
    pass


class OllamaNotRunningError(RuntimeError):
    pass


class OllamaModelNotFoundError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[Dict[str, Any]]
    raw: Any = None


class OllamaLLM:
    """
    Wraps a local Ollama model. All calls are local HTTP calls to the
    Ollama daemon (default http://localhost:11434) -- never to a remote
    paid API.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        if ollama is None:
            raise OllamaNotInstalledError(
                "The 'ollama' python package is not installed.\n"
                "Install it with:  pip install ollama\n"
                f"(underlying import error: {OLLAMA_IMPORT_ERROR})"
            )
        self.model = model
        self.host = host
        self.client = Client(host=host)

    # ---------------------------------------------------------------
    # Health checks
    # ---------------------------------------------------------------

    def check_server_running(self) -> bool:
        """Returns True if the Ollama daemon is reachable."""
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def check_model_available(self) -> bool:
        """Returns True if self.model has been pulled locally."""
        try:
            resp = self.client.list()
        except Exception as e:
            raise OllamaNotRunningError(str(e)) from e

        models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
        names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if name is None:
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
            if name:
                names.append(name)
        # Ollama tags look like "llama3.1:8b"; also accept a bare match
        # against the part before ':' in case the user didn't specify a tag.
        target = self.model
        target_base = target.split(":")[0]
        return any(n == target or n.split(":")[0] == target_base for n in names)

    def ensure_ready(self):
        """Raises a clear, specific error if Ollama or the model isn't ready."""
        if not self.check_server_running():
            raise OllamaNotRunningError(
                "Could not reach the Ollama server at "
                f"{self.host}.\n"
                "Make sure Ollama is installed and running:\n"
                "  Windows/Mac: open the Ollama app, or run 'ollama serve'\n"
                "  Linux:       run 'ollama serve' in a terminal\n"
            )
        if not self.check_model_available():
            raise OllamaModelNotFoundError(
                f"Model '{self.model}' is not downloaded yet.\n"
                f"Pull it with:  ollama pull {self.model}\n"
            )

    # ---------------------------------------------------------------
    # Chat
    # ---------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        keep_alive: Optional[str] = "30m",
    ) -> LLMResponse:
        """
        Send a chat request to the local model.

        messages:   list of {"role": "system"|"user"|"assistant"|"tool", "content": str, ...}
        tools:      list of function-calling schemas (OpenAI-style), or None
        keep_alive: how long Ollama should keep the model loaded in memory after this
                    request (e.g. "30m", "1h", or "-1" for indefinitely). Without this,
                    Ollama's default is only 5 minutes, so the model gets unloaded and
                    has to be reloaded from disk on your next message -- this is one of
                    the biggest causes of a "slow" first response after any pause.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        if tools:
            kwargs["tools"] = tools
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive

        try:
            resp = self.client.chat(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Ollama chat request failed: {e}") from e

        message = resp.get("message", {}) if isinstance(resp, dict) else getattr(resp, "message", {})
        if not isinstance(message, dict):
            # ollama-python may return a pydantic-like object
            message = {
                "content": getattr(message, "content", ""),
                "tool_calls": getattr(message, "tool_calls", None),
            }

        content = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                tc = {
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", None),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", {}),
                    }
                }
            func = tc.get("function", {})
            tool_calls.append({
                "name": func.get("name"),
                "arguments": func.get("arguments") or {},
            })

        return LLMResponse(content=content, tool_calls=tool_calls, raw=resp)
