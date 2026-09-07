"""
tools/filesystem.py

File system operations restricted to the configured workspace directory.
All paths are resolved through tools.sandbox.resolve_in_workspace, which
raises PathEscapeError for anything outside the workspace.

Each public function returns a plain dict: {"success": bool, ...}.
Deletion always requires confirmation, which is enforced by the
tool_manager (see DANGEROUS_TOOLS in tool_manager.py), not here --
but this module also double-checks a `confirmed` flag as defense in depth.
"""

import fnmatch
import os
from pathlib import Path

from .sandbox import resolve_in_workspace, PathEscapeError, ensure_workspace_exists

MAX_READ_BYTES = 300_000  # avoid dumping huge files into the model context
MAX_LIST_ENTRIES = 500


def list_files(workspace: str, path: str = ".") -> dict:
    try:
        target = resolve_in_workspace(workspace, path)
    except PathEscapeError as e:
        return {"success": False, "error": str(e)}

    if not os.path.exists(target):
        return {"success": False, "error": f"Path does not exist: {path}"}
    if not os.path.isdir(target):
        return {"success": False, "error": f"Not a directory: {path}"}

    entries = []
    for entry in sorted(os.listdir(target))[:MAX_LIST_ENTRIES]:
        full = os.path.join(target, entry)
        entries.append({
            "name": entry,
            "type": "dir" if os.path.isdir(full) else "file",
            "size": os.path.getsize(full) if os.path.isfile(full) else None,
        })
    return {"success": True, "path": path, "entries": entries}


def read_file(workspace: str, path: str) -> dict:
    try:
        target = resolve_in_workspace(workspace, path)
    except PathEscapeError as e:
        return {"success": False, "error": str(e)}

    if not os.path.isfile(target):
        return {"success": False, "error": f"File does not exist: {path}"}

    size = os.path.getsize(target)
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_READ_BYTES)
    except OSError as e:
        return {"success": False, "error": f"Could not read file: {e}"}

    truncated = size > MAX_READ_BYTES
    return {"success": True, "path": path, "content": content, "truncated": truncated, "size": size}


def write_file(workspace: str, path: str, content: str, overwrite: bool = True) -> dict:
    try:
        target = resolve_in_workspace(workspace, path)
    except PathEscapeError as e:
        return {"success": False, "error": str(e)}

    if os.path.exists(target) and os.path.isdir(target):
        return {"success": False, "error": f"Cannot write: {path} is a directory."}
    if os.path.exists(target) and not overwrite:
        return {"success": False, "error": f"File already exists and overwrite=False: {path}"}

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return {"success": False, "error": f"Could not write file: {e}"}

    return {"success": True, "path": path, "bytes_written": len(content.encode("utf-8"))}


def edit_file(workspace: str, path: str, find: str, replace: str) -> dict:
    """Simple, reliable find-and-replace edit (first occurrence by default: all occurrences)."""
    result = read_file(workspace, path)
    if not result["success"]:
        return result
    original = result["content"]
    if find not in original:
        return {"success": False, "error": f"Text to find was not present in {path}."}
    updated = original.replace(find, replace)
    return write_file(workspace, path, updated, overwrite=True)


def delete_file(workspace: str, path: str, confirmed: bool = False) -> dict:
    if not confirmed:
        return {"success": False, "error": "Deletion requires confirmed=True (user must confirm first)."}
    try:
        target = resolve_in_workspace(workspace, path)
    except PathEscapeError as e:
        return {"success": False, "error": str(e)}

    if not os.path.exists(target):
        return {"success": False, "error": f"Path does not exist: {path}"}

    try:
        if os.path.isdir(target):
            os.rmdir(target)  # only remove empty dirs directly; refuse recursive delete here
        else:
            os.remove(target)
    except OSError as e:
        return {"success": False, "error": f"Could not delete: {e}"}
    return {"success": True, "path": path, "deleted": True}


def create_directory(workspace: str, path: str) -> dict:
    try:
        target = resolve_in_workspace(workspace, path)
    except PathEscapeError as e:
        return {"success": False, "error": str(e)}
    try:
        Path(target).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"success": False, "error": f"Could not create directory: {e}"}
    return {"success": True, "path": path}


def search_files(workspace: str, pattern: str, content_query: str = None) -> dict:
    """
    Search for files whose name matches a glob `pattern` (e.g. "*.py"),
    optionally filtered to those containing `content_query` as substring.
    """
    ensure_workspace_exists(workspace)
    workspace_real = os.path.realpath(workspace)
    matches = []
    for root, _dirs, files in os.walk(workspace_real):
        for fname in files:
            if not fnmatch.fnmatch(fname, pattern):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, workspace_real)
            if content_query:
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        if content_query not in f.read(MAX_READ_BYTES):
                            continue
                except OSError:
                    continue
            matches.append(rel)
            if len(matches) >= MAX_LIST_ENTRIES:
                break
    return {"success": True, "pattern": pattern, "matches": matches}
