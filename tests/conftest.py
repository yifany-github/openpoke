"""Shared fixtures: isolate every persistent store in a temp dir and stub the LLM."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from server.config import get_settings
from server.services import triggers as triggers_pkg
from server.services.execution import agent_memory, log_store, roster
from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Point roster, execution logs, agent memory and triggers at tmp_path."""
    exec_dir = tmp_path / "execution_agents"
    the_roster = roster.AgentRoster(exec_dir / "roster.json")
    logs = log_store.ExecutionAgentLogStore(exec_dir)
    memory = agent_memory.AgentMemoryStore(exec_dir)
    trigger_service = TriggerService(TriggerStore(tmp_path / "triggers.db"))

    monkeypatch.setattr(roster, "_agent_roster", the_roster)
    monkeypatch.setattr(log_store, "_execution_agent_logs", logs)
    monkeypatch.setattr(agent_memory, "_agent_memory", memory)
    monkeypatch.setattr(triggers_pkg, "_trigger_service", trigger_service)
    monkeypatch.setattr(agent_memory, "_in_flight", set())

    class Stores:
        pass

    s = Stores()
    s.dir = exec_dir
    s.roster = the_roster
    s.logs = logs
    s.memory = memory
    s.triggers = trigger_service
    return s


@pytest.fixture
def settings(monkeypatch):
    """Mutable settings with small, predictable limits."""
    cfg = get_settings()
    monkeypatch.setattr(cfg, "agent_roster_prompt_limit", 5)
    monkeypatch.setattr(cfg, "execution_history_window_runs", 3)
    monkeypatch.setattr(cfg, "execution_history_compaction_batch", 2)
    monkeypatch.setattr(cfg, "openrouter_api_key", "test-key")
    return cfg


def days_ago(n: float) -> datetime:
    return NOW - timedelta(days=n)
