import os
import tempfile
import shutil
import pytest

from tools import filesystem
from tools.sandbox import PathEscapeError, resolve_in_workspace


@pytest.fixture
def workspace():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_write_and_read_file(workspace):
    result = filesystem.write_file(workspace, "hello.txt", "hello world")
    assert result["success"]
    read = filesystem.read_file(workspace, "hello.txt")
    assert read["success"]
    assert read["content"] == "hello world"


def test_list_files(workspace):
    filesystem.write_file(workspace, "a.txt", "a")
    filesystem.create_directory(workspace, "sub")
    listing = filesystem.list_files(workspace, ".")
    assert listing["success"]
    names = {e["name"] for e in listing["entries"]}
    assert "a.txt" in names
    assert "sub" in names


def test_edit_file(workspace):
    filesystem.write_file(workspace, "a.txt", "hello world")
    result = filesystem.edit_file(workspace, "a.txt", "world", "there")
    assert result["success"]
    read = filesystem.read_file(workspace, "a.txt")
    assert read["content"] == "hello there"


def test_edit_file_missing_text(workspace):
    filesystem.write_file(workspace, "a.txt", "hello world")
    result = filesystem.edit_file(workspace, "a.txt", "notpresent", "x")
    assert not result["success"]


def test_delete_requires_confirmation(workspace):
    filesystem.write_file(workspace, "a.txt", "hello")
    result = filesystem.delete_file(workspace, "a.txt", confirmed=False)
    assert not result["success"]
    assert os.path.exists(os.path.join(workspace, "a.txt"))

    result2 = filesystem.delete_file(workspace, "a.txt", confirmed=True)
    assert result2["success"]
    assert not os.path.exists(os.path.join(workspace, "a.txt"))


def test_path_escape_blocked(workspace):
    with pytest.raises(PathEscapeError):
        resolve_in_workspace(workspace, "../../etc/passwd")


def test_read_outside_workspace_blocked(workspace):
    result = filesystem.read_file(workspace, "../../etc/passwd")
    assert not result["success"]
    assert "outside" in result["error"].lower()


def test_search_files(workspace):
    filesystem.write_file(workspace, "one.py", "print('hi')")
    filesystem.write_file(workspace, "two.txt", "not python")
    result = filesystem.search_files(workspace, "*.py")
    assert result["success"]
    assert "one.py" in result["matches"]
    assert "two.txt" not in result["matches"]
