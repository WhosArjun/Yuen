"""
agent/planner.py

A small, real (not decorative) planning component. The plan is a list of
steps with statuses, persisted via Memory (SQLite) so it survives across
turns in a session. The LLM itself decides the plan content by calling
the `planner_set_plan` / `planner_update_step` tools (see tool_manager.py
and tools registered in agent.py) -- this module just stores and renders
it. Nothing here is hard-coded to a specific kind of task.
"""

from typing import Any, Dict, List

from .memory import Memory

VALID_STATUSES = {"pending", "in_progress", "done", "failed", "skipped"}


class Planner:
    def __init__(self, memory: Memory):
        self.memory = memory

    def set_plan(self, steps: List[str]) -> Dict[str, Any]:
        steps = [s.strip() for s in steps if s and s.strip()]
        if not steps:
            return {"success": False, "error": "No non-empty steps provided."}
        self.memory.save_plan(steps)
        return {"success": True, "plan": self.get_plan_text()}

    def update_step(self, step_index: int, status: str) -> Dict[str, Any]:
        if status not in VALID_STATUSES:
            return {"success": False, "error": f"Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}"}
        plan = self.memory.get_plan()
        if step_index < 0 or step_index >= len(plan):
            return {"success": False, "error": f"step_index {step_index} out of range (plan has {len(plan)} steps)."}
        self.memory.update_step_status(step_index, status)
        return {"success": True, "plan": self.get_plan_text()}

    def get_plan(self) -> List[Dict[str, Any]]:
        return self.memory.get_plan()

    def get_plan_text(self) -> str:
        plan = self.get_plan()
        if not plan:
            return "(no active plan)"
        symbols = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "done": "[x]",
            "failed": "[!]",
            "skipped": "[-]",
        }
        lines = []
        for step in plan:
            sym = symbols.get(step["status"], "[ ]")
            lines.append(f"{sym} {step['step_index'] + 1}. {step['description']}")
        return "\n".join(lines)

    def clear(self):
        self.memory.clear_plan()
