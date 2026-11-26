"""Interaction runtime end-to-end, turn serialization, per-agent run lock, shared manager."""

import asyncio
import json

import pytest

from server.agents.execution_agent import batch_manager as bm
from server.agents.execution_agent.runtime import ExecutionResult
from server.agents.interaction_agent import runtime as ia_runtime
from server.agents.interaction_agent import tools
from server.services import trigger_scheduler as ts
from server.services.conversation.log import ConversationLog
from server.services.conversation.summarization import working_memory_log as wml
from server.services.execution.agent_memory import split_runs


# ----------------------------------------------------------------------------- helpers

def tool_call(name, **args):
    return {"id": f"call_{name}_{len(json.dumps(args))}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def llm_reply(*calls, content=""):
    return {"choices": [{"message": {"content": content, "tool_calls": list(calls)}}]}


@pytest.fixture
def interaction(stores, settings, tmp_path, monkeypatch):
    """An InteractionAgentRuntime factory wired to temp logs; LLM and dispatch stubbed."""
    monkeypatch.setattr(settings, "conversation_summary_threshold", 0)
    monkeypatch.setattr(wml, "_working_memory_log", wml.WorkingMemoryLog(tmp_path / "wm.log"))
    conversation = ConversationLog(tmp_path / "conversation.log")
    dispatched = []

    async def fake_execute(agent_name, instructions, request_id=None):
        dispatched.append((agent_name, instructions))

    monkeypatch.setattr(tools._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute)
    monkeypatch.setattr(tools, "get_conversation_log", lambda: conversation)

    def make():
        rt = ia_runtime.InteractionAgentRuntime()
        rt.conversation_log = conversation
        return rt

    make.dispatched = dispatched
    make.conversation = conversation
    return make


# ----------------------------------------------------------------------------- B5

def test_model_recovers_from_similar_name_rejection(interaction, stores, monkeypatch):
    """LLM invents a near-duplicate name -> tool rejects -> LLM reuses the existing agent."""
    stores.roster.add_agent("Email to Alice")
    stores.roster.update_description("Email to Alice", "Lunch invite to Alice for Friday")
    seen_tool_results = []

    async def scripted_llm(*, messages, **kwargs):
        last = messages[-1]
        if last["role"] == "user":
            return llm_reply(tool_call("send_message_to_agent", agent_name="Reply to Alice re lunch",
                                       instructions="Tell Alice Friday works"))
        seen_tool_results.append(json.loads(last["content"]))
        if seen_tool_results[-1]["status"] == "error":
            similar = seen_tool_results[-1]["error"]["similar_agents"][0]["agent_name"]
            return llm_reply(tool_call("send_message_to_agent", agent_name=similar,
                                       instructions="Tell Alice Friday works"),
                             tool_call("send_message_to_user", message="On it"))
        return llm_reply(content="")

    monkeypatch.setattr(ia_runtime, "request_chat_completion", scripted_llm)

    async def run():
        result = await interaction().execute("tell alice friday works for lunch")
        await asyncio.sleep(0)  # let the fire-and-forget dispatch run
        return result

    result = asyncio.run(run())

    assert result.success and result.response == "On it"
    assert seen_tool_results[0]["error"]["error"] == "similar_agents_exist"
    assert interaction.dispatched == [("Email to Alice", "Tell Alice Friday works")]
    assert stores.roster.get_agents() == ["Email to Alice"]


# ----------------------------------------------------------------------------- C1

def test_interaction_turns_are_serialized(interaction, monkeypatch):
    """A second turn must see the first turn's reply instead of racing it."""
    prompts = []

    async def slow_llm(*, messages, **kwargs):
        last = messages[-1]
        if last["role"] == "user":
            prompts.append(last["content"])
            await asyncio.sleep(0.02)  # first turn is mid-flight when the second arrives
            turn = "first" if "first message" in last["content"].split("<new_")[1] else "second"
            return llm_reply(tool_call("send_message_to_user", message=f"reply to {turn}"))
        return llm_reply(content="")

    monkeypatch.setattr(ia_runtime, "request_chat_completion", slow_llm)

    async def run():
        rt = interaction()
        await asyncio.gather(rt.execute("first message"), rt.handle_agent_message("second message"))

    asyncio.run(run())

    history_of_second = prompts[1].split("</conversation_history>")[0]
    assert "reply to first" in history_of_second


# ----------------------------------------------------------------------------- B2

def test_same_agent_runs_are_serialized_and_logged_contiguously(stores, settings, monkeypatch):
    class FakeRuntime:
        def __init__(self, agent_name):
            self.name = agent_name

        async def execute(self, instructions):
            logs = stores.logs
            logs.record_action(self.name, f"start {instructions}")
            await asyncio.sleep(0.01)
            logs.record_agent_response(self.name, f"done {instructions}")
            return ExecutionResult(agent_name=self.name, success=True, response="ok")

    async def no_dispatch(self, payload):
        return None

    monkeypatch.setattr(bm, "ExecutionAgentRuntime", FakeRuntime)
    monkeypatch.setattr(bm.ExecutionBatchManager, "_dispatch_to_interaction_agent", no_dispatch)

    async def run():
        mgr = bm.ExecutionBatchManager()
        await asyncio.gather(mgr.execute_agent("Email to Alice", "one"),
                             mgr.execute_agent("email-to-alice", "two"))

    asyncio.run(run())

    runs = split_runs(list(stores.logs.iter_entries("Email to Alice")))
    assert [[tag for tag, _, _ in run] for run in runs] == [["agent_request", "agent_action", "agent_response"]] * 2
    assert runs[0][0][2] == "one" and runs[0][2][2] == "done one"
    assert runs[1][0][2] == "two" and runs[1][2][2] == "done two"


# ----------------------------------------------------------------------------- C3

def test_triggers_use_the_shared_batch_manager(stores, monkeypatch):
    assert tools._EXECUTION_BATCH_MANAGER is bm.get_execution_batch_manager()

    calls = []

    class FakeManager:
        async def execute_agent(self, agent_name, instructions):
            calls.append(agent_name)
            return ExecutionResult(agent_name=agent_name, success=True, response="ok")

    monkeypatch.setattr(ts, "get_execution_batch_manager", lambda: FakeManager())
    monkeypatch.setattr(ts, "get_trigger_service", lambda: stores.triggers)
    trigger = stores.triggers.create_trigger(agent_name="Weekly digest", payload="Summarize newsletters",
                                             start_time="2026-11-23T09:00:00")
    scheduler = ts.TriggerScheduler()
    asyncio.run(scheduler._execute_trigger(trigger))

    assert calls == ["Weekly digest"]
