"""
tools/terminal.py

Executes shell commands with the working directory pinned to the
configured workspace. Dangerous commands (see config.dangerous_patterns)
are flagged as requiring confirmation by the tool_manager before this
module ever runs them.

Limitations (documented honestly, not hidden):
    - This runs a real shell (subprocess with shell=True) so the agent
      can use pipes/redirection like a normal terminal. That means a
      sufficiently adversarial command could still reference absolute
      paths outside the workspace (e.g. `cat /etc/passwd`). We reduce
      risk by: pinning cwd to the workspace, pattern-matching known
      destructive commands for mandatory confirmation, sanitizing the
      environment (see config.blocked_env_vars), and enforcing a
      timeout. This is best-effort sandboxing, not a hard security
      boundary -- do not run this agent as an untrusted-multi-user
      service.
"""

import os
import subprocess
from typing import Dict, List


def is_dangerous(command: str, dangerous_patterns: List[str]) -> bool:
    lowered = command.lower()
    return any(pat.lower() in lowered for pat in dangerous_patterns)


def _sanitized_env(blocked_env_vars: List[str]) -> Dict[str, str]:
    env = dict(os.environ)
    for var in blocked_env_vars:
        # Remove exact matches and anything containing the blocked substring
        # (case-insensitive), e.g. "TOKEN" blocks "GITHUB_TOKEN" too.
        for key in list(env.keys()):
            if var.upper() in key.upper():
                env.pop(key, None)
    return env


def run_command(
    workspace: str,
    command: str,
    timeout: int = 60,
    dangerous_patterns: List[str] = None,
    confirmed: bool = False,
) -> dict:
    dangerous_patterns = dangerous_patterns or []

    if is_dangerous(command, dangerous_patterns) and not confirmed:
        return {
            "success": False,
            "error": "This command matches a dangerous pattern and requires user confirmation "
                     "(confirmed=True) before it will run.",
            "command": command,
            "requires_confirmation": True,
        }

    env = _sanitized_env(["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY",
                           "AWS_ACCESS_KEY_ID", "GOOGLE_API_KEY", "GITHUB_TOKEN",
                           "PASSWORD", "SECRET", "TOKEN", "PRIVATE_KEY"])

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s.", "command": command}
    except OSError as e:
        return {"success": False, "error": f"Failed to execute command: {e}", "command": command}

    return {
        "success": proc.returncode == 0,
        "command": command,
        "stdout": proc.stdout[-20_000:],
        "stderr": proc.stderr[-20_000:],
        "exit_code": proc.returncode,
    }
