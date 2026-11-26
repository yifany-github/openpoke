"""Timezone awareness for execution agents, set_timezone tool, id-preserving truncation."""

import json

import pytest

from server.agents.execution_agent import agent as exec_agent
from server.agents.interaction_agent import tools
from server.services import timezone_store as tz_mod


@pytest.fixture
def tz_store(tmp_path, monkeypatch):
    store = tz_mod.TimezoneStore(tmp_path / "timezone.txt")
    monkeypatch.setattr(tz_mod, "_timezone_store", store)
    return store


def test_execution_prompt_states_current_time_and_timezone(tz_store, stores):
    tz_store.set_timezone("America/New_York")
    prompt = exec_agent.ExecutionAgent("Call mom reminder").build_system_prompt()
    line = [l for l in prompt.splitlines() if l.startswith("Current time:")][0]
    assert "America/New_York" in line
    assert "UTC-0500" in line or "UTC-0400" in line  # EST or EDT


def test_set_timezone_tool_persists_valid_zone(tz_store):
    result = tools.handle_tool_call("set_timezone", {"timezone": "Asia/Shanghai"})
    assert result.success
    assert tz_store.get_timezone() == "Asia/Shanghai"


def test_set_timezone_tool_rejects_unknown_zone(tz_store):
    result = tools.handle_tool_call("set_timezone", {"timezone": "New York"})
    assert not result.success and "IANA" in result.payload["hint"]
    assert tz_store.get_timezone() == "UTC"


def test_truncation_keeps_ids_that_fall_past_the_cut():
    payload = json.dumps({"messages": [{"id": f"msg-{i:03d}", "threadId": f"thr-{i:03d}",
                                        "from": f"person{i}@example.com", "snippet": "x" * 60}
                                       for i in range(12)]})
    out = exec_agent.truncate_keeping_ids(payload, 500)
    assert out.startswith(payload[:500])
    for i in range(12):
        assert f"msg-{i:03d}" in out and f"thr-{i:03d}" in out and f"person{i}@example.com" in out
    assert len(out) < len(payload)


def test_truncation_leaves_short_results_alone_and_handles_plain_text():
    assert exec_agent.truncate_keeping_ids("ok", 500) == "ok"
    text = "a" * 600 + " contact bob@example.com"
    assert exec_agent.truncate_keeping_ids(text, 500).endswith("[ids beyond cut: bob@example.com]")
