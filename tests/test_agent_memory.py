"""Execution-agent side: bounded history, rolling compaction, fallback."""

import asyncio

import pytest

from server.agents.execution_agent import runtime as exec_runtime
from server.agents.execution_agent.agent import ExecutionAgent
from server.services.execution import agent_memory
from server.services.execution.agent_memory import (
    compact,
    needs_compaction,
    parse_summary,
    render_history,
    split_runs,
)

AGENT = "Email to Alice"


def add_runs(stores, n, start=1):
    for i in range(start, start + n):
        stores.logs.record_request(AGENT, f"request {i}: email alice")
        stores.logs.record_action(AGENT, f"Calling gmail_create_draft with: draft-{i}")
        stores.logs.record_tool_response(AGENT, "gmail_create_draft", f'{{"draft_id": "r-{i:04d}"}}')
        stores.logs.record_agent_response(AGENT, f"Draft r-{i:04d} created")


def fake_summarizer(calls):
    """Extractive stand-in for the LLM: keeps every draft id it sees."""

    async def summarize(system_prompt, content):
        calls.append(content)
        ids = sorted(set(tok.strip('",{}:') for tok in content.split() if tok.strip('",{}:').startswith("r-")))
        return "DESCRIPTION: Drafts lunch emails to Alice\nMEMORY:\n- drafts: " + ", ".join(ids)

    return summarize


def test_split_runs_groups_by_request(stores):
    add_runs(stores, 3)
    runs = split_runs(list(stores.logs.iter_entries(AGENT)))
    assert len(runs) == 3
    assert all(run[0][0] == "agent_request" and len(run) == 4 for run in runs)


def test_history_is_unchanged_while_small(stores, settings):
    add_runs(stores, 2)
    history = render_history(AGENT)
    assert "## Recent runs (1-2)" in history
    assert "Memory summary" not in history
    assert history.count("<agent_request") == 2


def test_history_stays_bounded_without_compaction(stores, settings):
    # window=3, batch=2 -> at most 5 raw runs even if summarization never ran
    add_runs(stores, 40)
    history = render_history(AGENT)
    assert history.count("<agent_request") == 5
    assert "35 older run(s) omitted" in history
    assert "## Recent runs (36-40)" in history


def test_compaction_threshold(stores, settings):
    add_runs(stores, 4)
    assert not needs_compaction(AGENT)
    add_runs(stores, 1, start=5)
    assert needs_compaction(AGENT)


def test_compaction_folds_old_runs_and_keeps_window(stores, settings):
    add_runs(stores, 5)
    calls = []
    assert asyncio.run(compact(AGENT, fake_summarizer(calls)))

    state = stores.memory.load(AGENT)
    assert state.covered_runs == 2  # 5 runs - window 3
    assert "request 1" in calls[0] and "request 3" not in calls[0]

    history = render_history(AGENT)
    assert "## Memory summary (runs 1-2)" in history
    assert "r-0001" in history and "r-0002" in history  # facts survive via summary
    assert history.count("<agent_request") == 3
    assert stores.roster.get(AGENT) is None  # agent not registered -> no description write


def test_compaction_is_incremental_and_updates_description(stores, settings):
    stores.roster.add_agent(AGENT)
    add_runs(stores, 5)
    calls = []
    asyncio.run(compact(AGENT, fake_summarizer(calls)))
    add_runs(stores, 3, start=6)  # 8 runs, 2 covered -> 6 raw >= 5
    asyncio.run(compact(AGENT, fake_summarizer(calls)))

    state = stores.memory.load(AGENT)
    assert state.covered_runs == 5
    # Second call only sees the previous summary + runs 3..5, not the whole log again.
    assert "Existing memory:\nDESCRIPTION" in calls[1]
    assert "request 1:" not in calls[1] and "request 3:" in calls[1]
    assert stores.roster.get(AGENT).description == "Drafts lunch emails to Alice"


def test_prompt_size_is_flat_after_many_runs(stores, settings):
    calls = []
    sizes = []
    for n in range(1, 61):
        add_runs(stores, 1, start=n)
        if needs_compaction(AGENT):
            asyncio.run(compact(AGENT, fake_summarizer(calls)))
        sizes.append(len(ExecutionAgent(AGENT).build_system_prompt_with_history()))
    assert max(sizes[20:]) < sizes[9] * 1.5  # flat, not linear
    assert "r-0001" in ExecutionAgent(AGENT).build_system_prompt_with_history()


def test_parse_summary():
    assert parse_summary("DESCRIPTION: Owns X\nMEMORY:\n- a") == ("Owns X", "DESCRIPTION: Owns X\nMEMORY:\n- a")
    assert parse_summary("no header") == ("", "no header")


def test_failed_summarizer_leaves_state_untouched(stores, settings):
    add_runs(stores, 6)

    async def broken(system_prompt, content):
        raise RuntimeError("LLM down")

    async def run():
        task = agent_memory.schedule_compaction(AGENT, broken)
        assert task is not None
        await task

    asyncio.run(run())
    assert stores.memory.load(AGENT).covered_runs == 0
    assert agent_memory._in_flight == set()  # can retry next run
    assert render_history(AGENT).count("<agent_request") == 5


def test_runtime_records_run_refreshes_description_and_schedules_compaction(stores, settings, monkeypatch):
    """End-to-end through ExecutionAgentRuntime with a stubbed LLM."""
    stores.roster.add_agent(AGENT)
    stores.roster.update_description(AGENT, "Started for: invite Alice to lunch")
    add_runs(stores, 4)
    stores.logs.record_request(AGENT, "request 5: confirm")  # interaction agent logs the request

    async def fake_llm(**kwargs):
        return {"choices": [{"message": {"content": "Confirmed lunch with Alice, draft r-0005 sent"}}]}

    scheduled = []
    monkeypatch.setattr(exec_runtime, "request_chat_completion", fake_llm)
    monkeypatch.setattr(exec_runtime, "schedule_compaction", lambda name: scheduled.append(name))

    result = asyncio.run(exec_runtime.ExecutionAgentRuntime(AGENT).execute("request 5: confirm"))

    assert result.success
    assert scheduled == [AGENT]
    assert needs_compaction(AGENT)
    desc = stores.roster.get(AGENT).description
    assert desc.startswith("Started for: invite Alice to lunch | Latest: Confirmed lunch with Alice")
