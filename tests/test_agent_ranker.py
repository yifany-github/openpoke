from server.services.execution.agent_ranker import (
    mentioned_agent_keys,
    search_agents,
    select_agents,
    tokenize,
)
from server.services.execution.roster import AgentRecord, agent_key

from .conftest import NOW, days_ago


def rec(name, description="", age_days=1.0, runs=1):
    stamp = days_ago(age_days).isoformat()
    return AgentRecord(name=name, key=agent_key(name), description=description,
                       created_at=stamp, last_used_at=stamp, run_count=runs)


AGENTS = [
    rec("Email to Alice", "Lunch invite to Alice (alice@acme.com) for Friday", age_days=10),
    rec("Stripe contract", "Contract review thread with Priya at Stripe", age_days=20),
    rec("Flight to Tokyo", "Searching flights to Tokyo for December", age_days=0.1),
    rec("Weekly digest", "Monday 9am summary of unread newsletters", age_days=2),
    rec("Vercel Job Offer", "Offer negotiation with Vercel recruiter", age_days=30),
]


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("Did Priya reply to the emails?") == ["priya", "reply", "email"]


def test_relevance_beats_recency_when_query_matches():
    selection = select_agents(AGENTS, "any update from Priya on the contract?", k=2, now=NOW)
    assert selection.shown[0].name == "Stripe contract"


def test_recency_decides_when_query_has_no_signal():
    selection = select_agents(AGENTS, "ok send it", k=2, now=NOW)
    assert selection.shown[0].name == "Flight to Tokyo"


def test_top_k_and_hidden_count():
    selection = select_agents(AGENTS, "", k=2, now=NOW)
    assert len(selection.shown) == 2
    assert selection.hidden_count == 3
    assert selection.total == 5


def test_pinned_agents_always_shown_even_if_irrelevant_and_old():
    pinned = {agent_key("Vercel Job Offer")}
    selection = select_agents(AGENTS, "flights to Tokyo", k=1, pinned_keys=pinned, now=NOW)
    names = [r.name for r in selection.shown]
    assert names == ["Vercel Job Offer", "Flight to Tokyo"]  # pinned first, then top-1
    assert selection.pinned_keys == pinned


def test_search_returns_only_lexical_matches():
    names = [r.name for r in search_agents(AGENTS, "Alice", now=NOW)]
    assert names == ["Email to Alice"]
    assert search_agents(AGENTS, "zzz-nothing", now=NOW) == []


def test_mentioned_agent_keys_finds_batch_result_senders():
    text = "[SUCCESS] Email to Alice: draft created\n[FAILED] Weekly digest: timeout"
    assert mentioned_agent_keys(AGENTS, text) == {agent_key("Email to Alice"), agent_key("Weekly digest")}


def test_pinned_agents_are_capped():
    pinned = {r.key for r in AGENTS}  # everything has a trigger
    selection = select_agents(AGENTS, "contract", k=1, pinned_keys=pinned, max_pinned=2, now=NOW)
    assert len(selection.pinned_keys) == 2
    assert selection.shown[0].name == "Stripe contract"  # best-scoring pinned kept first
    assert len(selection.shown) == 3  # 2 pinned + top-1 of the overflow


def test_mentions_require_whole_words():
    agents = [rec("Bob"), rec("Email to Bob")]
    assert mentioned_agent_keys(agents, "Bobby says hi") == set()
    assert mentioned_agent_keys(agents, "[SUCCESS] Email to Bob: sent") == {agent_key("Bob"), agent_key("Email to Bob")}
