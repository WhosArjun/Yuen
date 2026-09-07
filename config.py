"""
config.py

Central configuration for the local agent. No API keys are used anywhere
in this project. Configuration can come from (in order of precedence,
highest first):

    1. Command-line arguments (parsed in main.py and passed in)
    2. Environment variables
    3. A local config.json file (optional, created by the user)
    4. Hard-coded defaults below

Nothing in this file requires network access or paid services.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

# Commands / patterns that are always treated as dangerous and require
# explicit user confirmation before the terminal tool will run them.
DEFAULT_DANGEROUS_PATTERNS = [
    "rm -rf", "rm -r", "rmdir", "del /f", "del /s", "format ",
    "mkfs", "dd if=", "dd of=", ":(){:|:&};:", "shutdown", "reboot",
    "> /dev/sda", "chmod -r 777", "chown -r", "git push", "git reset --hard",
    "truncate -s 0", "diskpart", "reg delete", "taskkill /f",
]


@dataclass
class Config:
    # --- Model settings ---
    model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    # --- Workspace / sandbox settings ---
    workspace: str = str(PROJECT_ROOT / "workspace")

    # --- Agent loop settings ---
    max_steps: int = 30
    require_confirmation: bool = True
    command_timeout: int = 60  # seconds, for terminal + python execution

    # --- Performance ---
    # How long Ollama keeps the model loaded in memory after a request.
    # Longer values avoid repeated reload delays during a session at the cost
    # of holding RAM/VRAM. Use "-1" to keep it loaded indefinitely, or "0" to
    # unload immediately after each response.
    keep_alive: str = "30m"
    temperature: float = 0.2

    # --- Memory ---
    memory_db: str = str(PROJECT_ROOT / ".agent" / "memory.sqlite3")

    # --- Safety ---
    dangerous_patterns: list = field(default_factory=lambda: list(DEFAULT_DANGEROUS_PATTERNS))
    blocked_env_vars: list = field(default_factory=lambda: [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID", "GOOGLE_API_KEY", "GITHUB_TOKEN", "SSH_AUTH_SOCK",
        "PASSWORD", "SECRET", "TOKEN", "PRIVATE_KEY",
    ])

    def ensure_dirs(self):
        Path(self.workspace).mkdir(parents=True, exist_ok=True)
        Path(self.memory_db).parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, config_path: str = None, overrides: dict = None) -> "Config":
        """
        Load configuration with precedence: overrides (e.g. CLI args) >
        environment variables > config.json > defaults.
        """
        data = {}

        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data.update(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                print(f"[config] Warning: could not read {path}: {e}")

        env_map = {
            "MODEL": "model",
            "OLLAMA_HOST": "ollama_host",
            "WORKSPACE": "workspace",
            "MAX_STEPS": "max_steps",
            "REQUIRE_CONFIRMATION": "require_confirmation",
            "COMMAND_TIMEOUT": "command_timeout",
            "KEEP_ALIVE": "keep_alive",
            "TEMPERATURE": "temperature",
        }
        for env_key, field_name in env_map.items():
            if env_key in os.environ:
                value = os.environ[env_key]
                data[field_name] = _coerce(field_name, value)

        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    data[k] = v

        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, _coerce(k, v))

        cfg.workspace = str(Path(cfg.workspace).resolve())
        cfg.ensure_dirs()
        return cfg


def _coerce(field_name, value):
    if field_name in ("max_steps", "command_timeout") and not isinstance(value, int):
        return int(value)
    if field_name == "temperature" and not isinstance(value, float):
        return float(value)
    if field_name == "require_confirmation" and not isinstance(value, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return value


def write_example_config(path: str = None):
    """Writes an example config.json the user can edit."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    example = Config().to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    return path


if __name__ == "__main__":
    p = write_example_config()
    print(f"Wrote example config to {p}")
