import json

from server.services.execution.roster import AgentRoster, agent_key, name_similarity

from .conftest import NOW, days_ago


def test_agent_key_matches_log_slug():
    assert agent_key("Email to Alice") == agent_key("email-to-alice") == agent_key("Email  to  Alice!")
    assert agent_key("Email to Alice") != agent_key("Email to Bob")


def test_migrates_legacy_list_format_and_drops_slug_duplicates(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["Email to Alice", "email-to-alice", "Vercel Job Offer"]))

    roster = AgentRoster(path)

    assert roster.get_agents() == ["Email to Alice", "Vercel Job Offer"]
    saved = json.loads(path.read_text())
    assert saved["version"] == 2
    assert {a["key"] for a in saved["agents"]} == {"email-to-alice", "vercel-job-offer"}


def test_add_agent_is_idempotent_across_spellings(tmp_path):
    roster = AgentRoster(tmp_path / "roster.json")
    first = roster.add_agent("Email to Alice", now=NOW)
    second = roster.add_agent("email-to-alice", now=NOW)

    assert first == second
    assert roster.get_agents() == ["Email to Alice"]
    assert roster.get("EMAIL TO ALICE").name == "Email to Alice"


def test_touch_and_description_persist(tmp_path):
    path = tmp_path / "roster.json"
    roster = AgentRoster(path)
    roster.add_agent("Email to Alice", now=days_ago(3))
    roster.touch("email to alice", now=NOW)
    roster.update_description("Email to Alice", "  Lunch   plans with Alice  ")

    reloaded = AgentRoster(path)
    record = reloaded.get("Email to Alice")
    assert record.run_count == 1
    assert record.last_used_dt == NOW
    assert record.description == "Lunch plans with Alice"


def test_name_similarity_flags_same_thread_but_not_different_people():
    assert name_similarity("Email to Alice", "Reply to Alice re lunch") >= 0.6  # subset
    assert name_similarity("Vercel Job Offer", "Vercel offer") >= 0.6
    assert name_similarity("Flight to Tokyo", "Tokyo flights") >= 0.6  # plural folding
    assert name_similarity("Vercel Job Offer", "Stripe Job Offer") < 0.6
    assert name_similarity("Alice lunch", "Alice lunch plans") > name_similarity("Alice lunch", "Email to Alice")
    assert name_similarity("Alice lunch", "Alice contract") < 0.6
    assert name_similarity("Email to Alice", "Email to Bob") == 0.0
    assert name_similarity("Flight to Tokyo", "Tokyo Restaurant Search") < 0.6
    # Generic-only names carry no identity signal.
    assert name_similarity("Send email", "Search emails") == 0.0


def test_find_similar_excludes_exact_key_and_orders_by_score(tmp_path):
    roster = AgentRoster(tmp_path / "roster.json")
    for name in ["Email to Alice", "Alice contract", "Email to Bob", "Alice lunch plans"]:
        roster.add_agent(name, now=NOW)

    similar = [r.name for r in roster.find_similar("Alice lunch")]

    assert similar[0] == "Alice lunch plans"          # closest subset match
    assert "Email to Bob" not in similar
    assert "Email to Alice" in similar               # {alice} subset of {alice, lunch}
    assert "Alice contract" not in similar           # different topic for the same person
    # The exact-key agent is never reported as "similar" to itself.
    assert all(r.key != "email-to-alice" for r in roster.find_similar("email-to-alice"))


def test_triggers_follow_agent_identity_not_spelling(stores):
    """Legacy trigger saved under a respelled name must still belong to the canonical agent."""
    legacy = stores.triggers.create_trigger(
        agent_name="email-to-alice", payload="Remind me about lunch",
        start_time="2026-12-01T09:00:00",
    )

    canonical = "Email to Alice"
    assert [t.id for t in stores.triggers.list_triggers(agent_name=canonical)] == [legacy.id]
    updated = stores.triggers.update_trigger(legacy.id, agent_name=canonical, status="paused")
    assert updated is not None and updated.status == "paused"
    assert stores.triggers.list_triggers(agent_name="Email to Bob") == []


def test_concurrent_writers_do_not_lose_updates(tmp_path):
    """Two roster instances (e.g. two server workers) mutating the same file."""
    path = tmp_path / "roster.json"
    a, b = AgentRoster(path), AgentRoster(path)  # b's in-memory copy goes stale below
    a.add_agent("Email to Alice")
    b.add_agent("Flight to Tokyo")
    a.touch("Email to Alice")
    b.touch("Email to Alice")

    fresh = AgentRoster(path)
    assert fresh.get_agents() == ["Email to Alice", "Flight to Tokyo"]
    assert fresh.get("Email to Alice").run_count == 2


def test_corrupt_roster_is_backed_up_not_overwritten(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text('{"version": 2, "agents": [{"name": "Email to Ali')  # truncated write
    roster = AgentRoster(path)
    roster.add_agent("Flight to Tokyo")

    backups = list(tmp_path.glob("roster.json.corrupt-*"))
    assert len(backups) == 1 and "Email to Ali" in backups[0].read_text()
    assert AgentRoster(path).get_agents() == ["Flight to Tokyo"]


def test_wrong_shape_roster_does_not_crash_startup(tmp_path):
    for bad in ['{"agents": 5}', '{"agents": [{"name": "A", "run_count": "lots"}]}']:
        path = tmp_path / "roster.json"
        path.write_text(bad)
        roster = AgentRoster(path)  # used to raise at import time and stop the server
        assert roster.get_agents() == []
        assert list(tmp_path.glob("roster.json.corrupt-*"))
        for f in tmp_path.glob("roster.json.corrupt-*"):
            f.unlink()


def test_non_ascii_agent_names_are_matchable():
    from server.services.execution.roster import unicode_tokens

    assert "妈妈" in unicode_tokens("给妈妈发邮件")
    assert unicode_tokens("Café menu") == ["café", "menu"]
    assert name_similarity("给妈妈发邮件", "妈妈生日提醒") > 0
