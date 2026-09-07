import tempfile
import shutil
import sys
import pytest

from tools import terminal
from config import DEFAULT_DANGEROUS_PATTERNS


@pytest.fixture
def workspace():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_simple_command_runs(workspace):
    cmd = "echo hello" if sys.platform != "win32" else "echo hello"
    result = terminal.run_command(workspace, cmd, timeout=10, dangerous_patterns=[])
    assert result["success"]
    assert "hello" in result["stdout"].lower()


def test_dangerous_command_blocked_without_confirmation(workspace):
    result = terminal.run_command(
        workspace, "rm -rf /", timeout=10, dangerous_patterns=DEFAULT_DANGEROUS_PATTERNS, confirmed=False
    )
    assert not result["success"]
    assert result.get("requires_confirmation") is True


def test_dangerous_command_allowed_with_confirmation_flag(workspace):
    # We don't actually want to run rm -rf in tests; use a milder pattern
    # that's still on the dangerous list to prove the confirmed=True path works.
    result = terminal.run_command(
        workspace, "echo would-be-dangerous", timeout=10,
        dangerous_patterns=["would-be-dangerous"], confirmed=True,
    )
    assert result["success"]


def test_command_timeout(workspace):
    if sys.platform == "win32":
        pytest.skip("timeout test uses a POSIX sleep command")
    result = terminal.run_command(workspace, "sleep 5", timeout=1, dangerous_patterns=[])
    assert not result["success"]
    assert "timed out" in result["error"].lower()


def test_is_dangerous_detection():
    assert terminal.is_dangerous("sudo rm -rf /", DEFAULT_DANGEROUS_PATTERNS)
    assert not terminal.is_dangerous("ls -la", DEFAULT_DANGEROUS_PATTERNS)
