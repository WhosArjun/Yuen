"""
tools/sandbox.py

Shared helper for restricting filesystem/terminal/python/git tools to a
configured workspace directory. This is the core safety boundary for
the "no arbitrary destructive filesystem operations" requirement.

Path resolution is done with os.path.realpath so that '..' segments and
symlinks cannot be used to escape the workspace.
"""

import os
from pathlib import Path


class PathEscapeError(Exception):
    """Raised when a requested path would fall outside the workspace."""


def resolve_in_workspace(workspace: str, relative_path: str) -> str:
    """
    Resolve `relative_path` (which may be relative or absolute) against
    `workspace`, and raise PathEscapeError if the resolved path is not
    inside the workspace.
    """
    workspace_real = os.path.realpath(workspace)
    if relative_path in ("", ".", None):
        candidate = workspace_real
    elif os.path.isabs(relative_path):
        candidate = os.path.realpath(relative_path)
    else:
        candidate = os.path.realpath(os.path.join(workspace_real, relative_path))

    try:
        common = os.path.commonpath([workspace_real, candidate])
    except ValueError:
        # e.g. different drives on Windows
        raise PathEscapeError(
            f"Path '{relative_path}' resolves outside the workspace ({workspace_real})."
        )

    if common != workspace_real:
        raise PathEscapeError(
            f"Path '{relative_path}' resolves outside the workspace ({workspace_real})."
        )
    return candidate


def ensure_workspace_exists(workspace: str):
    Path(workspace).mkdir(parents=True, exist_ok=True)
