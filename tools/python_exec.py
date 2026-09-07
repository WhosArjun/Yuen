"""
tools/python_exec.py

Executes Python code snippets or files in a subprocess (not in-process
exec/eval, to avoid corrupting the agent's own process and to allow a
hard timeout). stdout, stderr, exit code, and a formatted traceback (if
any) are captured and returned so the LLM can debug its own code
(self-correction loop in agent.py).

Code is written to a temp file inside workspace/.agent/tmp and executed
with `python <file>` from the workspace directory, using the same
python interpreter that is running this agent (sys.executable), so no
extra runtime dependency is required.
"""

import subprocess
import sys
import time
import uuid
from pathlib import Path


def run_python_code(workspace: str, code: str, timeout: int = 60) -> dict:
    tmp_dir = Path(workspace) / ".agent" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / f"snippet_{uuid.uuid4().hex}.py"

    try:
        script_path.write_text(code, encoding="utf-8")
    except OSError as e:
        return {"success": False, "error": f"Could not write temp script: {e}"}

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout}s."}
    except OSError as e:
        return {"success": False, "error": f"Failed to execute python: {e}"}
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

    duration = round(time.time() - start, 3)
    return {
        "success": proc.returncode == 0,
        "stdout": proc.stdout[-20_000:],
        "stderr": proc.stderr[-20_000:],
        "exit_code": proc.returncode,
        "duration_seconds": duration,
    }


def run_python_file(workspace: str, path: str, timeout: int = 60, args: list = None) -> dict:
    from .sandbox import resolve_in_workspace, PathEscapeError

    try:
        target = resolve_in_workspace(workspace, path)
    except PathEscapeError as e:
        return {"success": False, "error": str(e)}

    if not Path(target).is_file():
        return {"success": False, "error": f"File does not exist: {path}"}

    cmd = [sys.executable, target] + (args or [])
    try:
        proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout}s."}
    except OSError as e:
        return {"success": False, "error": f"Failed to execute python file: {e}"}

    return {
        "success": proc.returncode == 0,
        "stdout": proc.stdout[-20_000:],
        "stderr": proc.stderr[-20_000:],
        "exit_code": proc.returncode,
    }
