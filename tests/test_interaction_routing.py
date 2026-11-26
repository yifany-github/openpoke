"""Interaction-agent side: roster rendering, name resolution, search_agents."""

import asyncio
from datetime import datetime, timezone

import pytest

from server.agents.interaction_agent import agent as ia_agent
from server.agents.interaction_agent import tools
from server.services.execution.roster import AgentRecord, agent_key

from .conftest import days_ago


@pytest.fixture
def dispatched(monkeypatch):
    """Capture execution dispatches instead of running agents."""
    calls = []

    async def fake_execute(agent_name, instructions, request_id=None):
        calls.append((agent_name, instructions))

    monkeypatch.setattr(tools._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute)
    return calls


def call_tool(name, **args):
    async def runner():
        result = tools.handle_tool_call(name, args)
        await asyncio.sleep(0)  # let fire-and-forget dispatch tasks run
        return result

    return asyncio.run(runner())


def seed(stores, n_filler=40):
    records = [
        AgentRecord(name="Email to Alice", key=agent_key("Email to Alice"),
                    description="Lunch invite to Alice for Friday",
                    created_at=days_ago(9).isoformat(), last_used_at=days_ago(9).isoformat(), run_count=2),
        AgentRecord(name="Weekly digest", key=agent_key("Weekly digest"),
                    description="Monday newsletter summary",
                    created_at=days_ago(60).isoformat(), last_used_at=days_ago(60).isoformat(), run_count=8),
    ]
    for i in range(n_filler):
        name = f"Filler task {i}"
        records.append(AgentRecord(name=name, key=agent_key(name), description=f"unrelated chore number {i}",
                                   created_at=days_ago(1).isoformat(), last_used_at=days_ago(1).isoformat()))
    stores.roster.bulk_load(records)


# ----------------------------------------------------------------------------- rendering

def test_render_shows_top_k_with_descriptions_and_more_marker(stores, settings):
    seed(stores)
    block = ia_agent._render_active_agents("can you check if alice replied about lunch")

    assert block.count("<agent ") == settings.agent_roster_prompt_limit
    assert 'name="Email to Alice"' in block
    assert "Lunch invite to Alice for Friday" in block
    assert '<more_agents count="37">' in block


def test_render_pins_agents_with_active_triggers(stores, settings):
    seed(stores)
    stores.triggers.create_trigger(
        agent_name="Weekly digest", payload="Summarize newsletters",
        recurrence_rule="RRULE:FREQ=WEEKLY;BYDAY=MO", start_time="2026-11-23T09:00:00",
    )
    block = ia_agent._render_active_agents("what's the weather")

    assert '<agent name="Weekly digest"' in block and 'pinned="true"' in block


def test_render_pins_agent_named_in_current_turn(stores, settings):
    seed(stores)
    block = ia_agent._render_active_agents("", pinned_text="[SUCCESS] Weekly digest: sent")
    assert '<agent name="Weekly digest"' in block


def test_prepare_message_uses_conversation_tail_as_query(stores, settings):
    seed(stores)
    transcript = "<user_message>can you invite Alice to lunch friday</user_message>"
    content = ia_agent.prepare_message_with_history("yes, go ahead", transcript)[0]["content"]
    assert 'name="Email to Alice"' in content


def test_small_roster_renders_everything(stores, settings):
    stores.roster.add_agent("Email to Alice")
    block = ia_agent._render_active_agents("hi")
    assert "Email to Alice" in block
    assert "more_agents" not in block


# ----------------------------------------------------------------------------- dispatch

def test_new_agent_is_created_and_dispatched(stores, settings, dispatched):
    result = call_tool("send_message_to_agent", agent_name="Flight to Tokyo", instructions="find flights")

    assert result.success and result.payload["new_agent_created"] is True
    assert dispatched == [("Flight to Tokyo", "find flights")]
    record = stores.roster.get("Flight to Tokyo")
    assert record.run_count == 1
    assert record.description.startswith("Started for: find flights")


def test_respelled_name_reuses_canonical_agent(stores, settings, dispatched):
    stores.roster.add_agent("Email to Alice")
    result = call_tool("send_message_to_agent", agent_name="email-to-alice", instructions="send the draft")

    assert result.success
    assert result.payload["new_agent_created"] is False
    assert result.payload["agent_name"] == "Email to Alice"
    assert result.payload["resolved_from"] == "email-to-alice"
    assert dispatched == [("Email to Alice", "send the draft")]
    assert stores.roster.get_agents() == ["Email to Alice"]


def test_similar_name_is_rejected_with_candidates(stores, settings, dispatched):
    stores.roster.add_agent("Email to Alice")
    result = call_tool("send_message_to_agent", agent_name="Reply to Alice re lunch", instructions="reply yes")

    assert not result.success
    assert result.payload["error"] == "similar_agents_exist"
    assert [a["agent_name"] for a in result.payload["similar_agents"]] == ["Email to Alice"]
    assert dispatched == []
    assert stores.roster.get_agents() == ["Email to Alice"]


def test_force_new_overrides_similarity_check(stores, settings, dispatched):
    stores.roster.add_agent("Email to Alice")
    result = call_tool("send_message_to_agent", agent_name="Alice contract", instructions="review",
                       force_new=True)

    assert result.success and result.payload["new_agent_created"] is True
    assert set(stores.roster.get_agents()) == {"Email to Alice", "Alice contract"}


def test_search_agents_reaches_agents_hidden_from_prompt(stores, settings):
    seed(stores)
    block = ia_agent._render_active_agents("filler chores")
    assert "Weekly digest" not in block  # old + irrelevant -> hidden

    result = call_tool("search_agents", query="newsletter digest")
    assert result.success
    assert result.payload["matches"][0]["agent_name"] == "Weekly digest"
    assert result.payload["total_agents"] == 42
