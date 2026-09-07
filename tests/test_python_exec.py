import tempfile
import shutil
import pytest

from tools import python_exec, filesystem


@pytest.fixture
def workspace():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_run_python_code_success(workspace):
    result = python_exec.run_python_code(workspace, "print('hello from agent')", timeout=10)
    assert result["success"]
    assert "hello from agent" in result["stdout"]
    assert result["exit_code"] == 0


def test_run_python_code_captures_exception(workspace):
    result = python_exec.run_python_code(workspace, "raise ValueError('boom')", timeout=10)
    assert not result["success"]
    assert "ValueError" in result["stderr"]
    assert "boom" in result["stderr"]
    assert result["exit_code"] != 0


def test_run_python_file(workspace):
    filesystem.write_file(workspace, "script.py", "print(1 + 1)")
    result = python_exec.run_python_file(workspace, "script.py", timeout=10)
    assert result["success"]
    assert "2" in result["stdout"]


def test_run_python_file_missing(workspace):
    result = python_exec.run_python_file(workspace, "nope.py", timeout=10)
    assert not result["success"]


def test_run_python_code_timeout(workspace):
    result = python_exec.run_python_code(workspace, "import time; time.sleep(5)", timeout=1)
    assert not result["success"]
    assert "timed out" in result["error"].lower()
