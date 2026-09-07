"""
tools/git.py

Wraps a small, explicit allowlist of git subcommands, all run with cwd
pinned to the workspace. `git push` is intentionally NOT implemented --
there is no code path in this file that can push to a remote. Commits
are local only.
"""

import subprocess
from typing import List, Optional

ALLOWED_SUBCOMMANDS = {"status", "diff", "log", "add", "commit", "branch", "init"}


def run_git(workspace: str, subcommand: str, args: Optional[List[str]] = None, timeout: int = 30) -> dict:
    args = args or []
    if subcommand not in ALLOWED_SUBCOMMANDS:
        return {
            "success": False,
            "error": f"git subcommand '{subcommand}' is not allowed. "
                     f"Allowed: {sorted(ALLOWED_SUBCOMMANDS)}. "
                     f"(Remote operations like push/pull/clone are not implemented for safety.)",
        }

    cmd = ["git", subcommand] + args
    try:
        proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"success": False, "error": "git is not installed or not on PATH."}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"git {subcommand} timed out after {timeout}s."}

    return {
        "success": proc.returncode == 0,
        "command": " ".join(cmd),
        "stdout": proc.stdout[-20_000:],
        "stderr": proc.stderr[-20_000:],
        "exit_code": proc.returncode,
    }


def status(workspace: str) -> dict:
    return run_git(workspace, "status", ["--short", "--branch"])


def diff(workspace: str, path: Optional[str] = None) -> dict:
    return run_git(workspace, "diff", [path] if path else [])


def log(workspace: str, n: int = 10) -> dict:
    return run_git(workspace, "log", [f"-{n}", "--oneline"])


def add(workspace: str, paths: List[str]) -> dict:
    return run_git(workspace, "add", paths or ["."])


def commit(workspace: str, message: str) -> dict:
    if not message or not message.strip():
        return {"success": False, "error": "Commit message must not be empty."}
    return run_git(workspace, "commit", ["-m", message])


def branch(workspace: str, name: Optional[str] = None) -> dict:
    return run_git(workspace, "branch", [name] if name else [])


def init(workspace: str) -> dict:
    return run_git(workspace, "init")
