import tempfile
import shutil
import pytest

from config import Config
from agent.memory import Memory
from agent.planner import Planner
from agent.tool_manager import ToolManager, TOOL_SCHEMAS


@pytest.fixture
def env():
    d = tempfile.mkdtemp()
    cfg = Config()
    cfg.workspace = d
    cfg.memory_db = f"{d}/mem.sqlite3"
    cfg.ensure_dirs()
    mem = Memory(cfg.memory_db, session_id="test")
    planner = Planner(mem)
    yield cfg, mem, planner
    mem.close()
    shutil.rmtree(d, ignore_errors=True)


def test_schemas_are_well_formed():
    names = set()
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        assert fn["name"] not in names, f"duplicate tool name {fn['name']}"
        names.add(fn["name"])


def test_dispatch_unknown_tool(env):
    cfg, mem, planner = env
    tm = ToolManager(cfg, mem, planner, confirm_callback=lambda n, a: True)
    result = tm.dispatch("not_a_real_tool", {})
    assert not result["success"]


def test_dispatch_filesystem_write_and_read(env):
    cfg, mem, planner = env
    tm = ToolManager(cfg, mem, planner, confirm_callback=lambda n, a: True)
    write_result = tm.dispatch("filesystem_write", {"path": "x.txt", "content": "hi"})
    assert write_result["success"]
    read_result = tm.dispatch("filesystem_read", {"path": "x.txt"})
    assert read_result["success"]
    assert read_result["content"] == "hi"


def test_delete_requires_confirmation_and_respects_decline(env):
    cfg, mem, planner = env
    tm = ToolManager(cfg, mem, planner, confirm_callback=lambda n, a: False)
    tm.dispatch("filesystem_write", {"path": "x.txt", "content": "hi"})
    result = tm.dispatch("filesystem_delete_file", {"path": "x.txt"})
    assert not result["success"]
    assert "declined" in result["error"].lower()


def test_delete_proceeds_when_confirmed(env):
    cfg, mem, planner = env
    tm = ToolManager(cfg, mem, planner, confirm_callback=lambda n, a: True)
    tm.dispatch("filesystem_write", {"path": "x.txt", "content": "hi"})
    result = tm.dispatch("filesystem_delete_file", {"path": "x.txt"})
    assert result["success"]


def test_planner_tools(env):
    cfg, mem, planner = env
    tm = ToolManager(cfg, mem, planner, confirm_callback=lambda n, a: True)
    result = tm.dispatch("planner_set_plan", {"steps": ["step one", "step two"]})
    assert result["success"]
    plan = tm.dispatch("planner_get_plan", {})
    assert "step one" in plan["plan"]
    update = tm.dispatch("planner_update_step", {"step_index": 0, "status": "done"})
    assert update["success"]


def test_memory_tools(env):
    cfg, mem, planner = env
    tm = ToolManager(cfg, mem, planner, confirm_callback=lambda n, a: True)
    tm.dispatch("memory_remember", {"key": "framework", "value": "flask"})
    recall = tm.dispatch("memory_recall", {"key": "framework"})
    assert recall["success"]
    assert recall["value"] == "flask"
