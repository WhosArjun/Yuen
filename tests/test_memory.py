import tempfile
import shutil
import pytest

from agent.memory import Memory


@pytest.fixture
def memory():
    d = tempfile.mkdtemp()
    m = Memory(f"{d}/mem.sqlite3", session_id="test")
    yield m
    m.close()
    shutil.rmtree(d, ignore_errors=True)


def test_add_and_get_recent_messages(memory):
    memory.add_message("user", "hello")
    memory.add_message("assistant", "hi there")
    recent = memory.get_recent_messages(limit=10)
    assert recent[0]["role"] == "user"
    assert recent[1]["role"] == "assistant"


def test_remember_and_recall(memory):
    memory.remember("db", "postgres")
    assert memory.recall("db") == "postgres"


def test_remember_overwrite(memory):
    memory.remember("db", "postgres")
    memory.remember("db", "sqlite")
    assert memory.recall("db") == "sqlite"


def test_forget(memory):
    memory.remember("temp", "value")
    memory.forget("temp")
    assert memory.recall("temp") is None


def test_search_facts(memory):
    memory.remember("backend_framework", "flask")
    memory.remember("frontend_framework", "react")
    results = memory.search_facts("framework")
    assert "backend_framework" in results
    assert "frontend_framework" in results


def test_plan_storage(memory):
    memory.save_plan(["step 1", "step 2", "step 3"])
    plan = memory.get_plan()
    assert len(plan) == 3
    assert plan[0]["status"] == "pending"
    memory.update_step_status(0, "done")
    plan = memory.get_plan()
    assert plan[0]["status"] == "done"
