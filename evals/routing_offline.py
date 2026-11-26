"""Offline routing eval: does the agent the user is talking about still reach the prompt?

No LLM involved. We build a 300-agent registry with ground truth, then for each
follow-up message we render the interaction agent's turn with the *real*
``prepare_message_with_history`` and check whether the target agent is listed.

- Baseline lists every agent (recall is 100% by construction, at ~3.3k tokens, names only).
  So this eval cannot show that routing got *better*; it shows we cut tokens ~80% while
  keeping the right agent visible. Routing accuracy needs an LLM: see ``routing_llm.py``.
- Ablations (recency-only, BM25-only) justify the hybrid score.
- ``+search`` counts misses that the model could still reach via ``search_agents``,
  using the user's message as the search query.

Also evaluates duplicate-agent detection on hand-labeled name pairs.

    .venv/bin/python -m evals.routing_offline
"""

from __future__ import annotations

import json
import logging
import random
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import tiktoken

from server.agents.interaction_agent import agent as ia_agent
from server.config import get_settings
from server.services import triggers as triggers_pkg
from server.services.execution import agent_ranker
from server.services.execution import roster as roster_mod
from server.services.execution.roster import AgentRecord, AgentRoster, agent_key, name_similarity
from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore

ENC = tiktoken.get_encoding("cl100k_base")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
N_AGENTS = 300
N_QUERIES = 240
K_VALUES = [5, 10, 15, 20, 30]

PEOPLE = ["Alice", "Bob", "Sharanjeet", "Keith", "Maria", "Chen", "Priya", "Omar", "Lena", "Diego",
          "Yuki", "Fatima", "Tom", "Grace", "Ivan", "Nora", "Kwame", "Sofia", "Raj", "Hannah",
          "Mateo", "Aisha", "Jonas", "Mei", "Lucas", "Zara", "Felix", "Ines", "Kenji", "Olga"]
COMPANIES = ["Vercel", "Stripe", "Notion", "Linear", "Figma", "Ramp", "Plaid", "Brex", "Retool", "Airtable"]
# topic -> paraphrases a user might use instead of the word in the agent's name
TOPICS = {
    "lunch": ["lunch", "lunch plans", "meal"],
    "contract": ["contract", "agreement", "paperwork"],
    "invoice": ["invoice", "bill", "payment"],
    "intro": ["intro", "introduction", "connecting us"],
    "offer": ["offer", "comp package", "offer letter"],
    "feedback": ["feedback", "review notes", "comments"],
    "meeting": ["meeting", "call", "sync"],
    "demo": ["demo", "product walkthrough", "demo"],
    "renewal": ["renewal", "subscription", "renewal"],
    "referral": ["referral", "recommendation", "referral"],
    "interview": ["interview", "interview loop", "onsite"],
    "partnership": ["partnership", "collab", "partnership deal"],
}
NAME_TEMPLATES = ["Email to {p}", "{p} {t}", "{c} {t}", "{t} with {p}", "Reply to {p} re {t}",
                  "{c} {t} with {p}", "{p} at {c}"]
EXPLICIT_QUERIES = ["did {p} reply about the {ts}?", "any update from {p} on the {ts}?",
                    "can you nudge {p} about the {ts}", "what did {p} say re the {ts}",
                    "follow up with {p} at {c} about the {ts}"]
FOLLOWUP_MESSAGES = ["yes send it", "ok go ahead", "looks good, ship it", "change the time to 3pm then send"]


@dataclass
class Scenario:
    person: str
    company: str
    topic: str
    record: AgentRecord


@dataclass
class Query:
    kind: str
    message: str
    transcript: str
    target_key: str


def build_registry(rng: random.Random) -> List[Scenario]:
    combos = [(p, t) for p in PEOPLE for t in TOPICS]
    rng.shuffle(combos)
    scenarios, used = [], set()
    for person, topic in combos[:N_AGENTS]:
        company = rng.choice(COMPANIES)
        name = rng.choice(NAME_TEMPLATES).format(p=person, c=company, t=topic)
        if agent_key(name) in used:
            name = f"{person} {topic}"
        used.add(agent_key(name))
        age = rng.uniform(0, 60)
        stamp = (NOW - timedelta(days=age)).isoformat()
        description = (f"Started for: Email {person} ({person.lower()}@{company.lower()}.com) at {company} "
                       f"about the {topic} | Latest: draft sent, waiting on {person}'s reply")
        record = AgentRecord(name=name, key=agent_key(name), description=description,
                             created_at=stamp, last_used_at=stamp, run_count=rng.randint(1, 12))
        scenarios.append(Scenario(person, company, topic, record))
    return scenarios


VAGUE_QUERIES = ["any news on the {c} {ts}?", "where are we with {c} on the {ts}", "did the {c} {ts} go through?"]
PERSON_QUERIES = ["did {p} reply?", "anything back from {p}?", "ping {p} again"]
KINDS = ["explicit", "paraphrase", "vague", "person_only", "followup"]


def build_queries(rng: random.Random, scenarios: List[Scenario]) -> List[Query]:
    by_company_topic: dict[tuple[str, str], list[Scenario]] = {}
    latest_by_person: dict[str, Scenario] = {}
    for s in scenarios:
        by_company_topic.setdefault((s.company, s.topic), []).append(s)
        best = latest_by_person.get(s.person)
        if best is None or s.record.last_used_dt > best.record.last_used_dt:
            latest_by_person[s.person] = s
    unique_ct = [v[0] for v in by_company_topic.values() if len(v) == 1]

    queries = []
    neutral = "<user_message>what's on my calendar today?</user_message>\n<poke_reply>Just a 2pm call.</poke_reply>"
    for i in range(N_QUERIES):
        kind = KINDS[i % len(KINDS)]
        s = rng.choice(scenarios)
        if kind == "vague":
            # No person named: company + paraphrased topic, only where that pair is unique.
            s = rng.choice(unique_ct)
            message = rng.choice(VAGUE_QUERIES).format(c=s.company, ts=rng.choice(TOPICS[s.topic][1:]))
            queries.append(Query(kind, message, neutral, s.record.key))
            continue
        if kind == "person_only":
            # "did Priya reply?" -> her most recently active thread; recency must break the tie.
            s = latest_by_person[s.person]
            message = rng.choice(PERSON_QUERIES).format(p=s.person)
            queries.append(Query(kind, message, neutral, s.record.key))
            continue
        if kind == "followup":
            # Context-free reply; the agent's name is in the recent transcript tail.
            transcript = (f"<user_message>email {s.person} about the {s.topic}</user_message>\n"
                          f"<agent_message>[SUCCESS] {s.record.name}: Draft created for "
                          f"{s.person.lower()}@{s.company.lower()}.com</agent_message>\n"
                          f"<poke_reply>Here's the draft for {s.person}. Send it?</poke_reply>")
            message = rng.choice(FOLLOWUP_MESSAGES)
        else:
            topic_word = s.topic if kind == "explicit" else rng.choice(TOPICS[s.topic][1:])
            message = rng.choice(EXPLICIT_QUERIES).format(p=s.person, c=s.company, ts=topic_word)
            transcript = neutral
        queries.append(Query(kind, message, transcript, s.record.key))
    return queries


def shown_keys(block: str) -> List[str]:
    """Agent keys in the order they are listed (best first)."""
    return [agent_key(m) for m in re.findall(r'<agent name="([^"]+)"', block)]


def run_strategy(queries: List[Query], k: int, relevance: float, recency: float) -> dict:
    settings = get_settings()
    settings.agent_roster_prompt_limit = k
    agent_ranker.RELEVANCE_WEIGHT, agent_ranker.RECENCY_WEIGHT = relevance, recency
    hits, search_hits, tok, top3 = 0, 0, 0, 0
    by_kind: dict[str, list[int]] = {}
    roster = roster_mod.get_agent_roster()
    for q in queries:
        content = ia_agent.prepare_message_with_history(q.message, q.transcript)[0]["content"]
        block = content.split("<active_agents>")[1].split("</active_agents>")[0]
        tok += len(ENC.encode(block))
        listed = shown_keys(block)
        hit = q.target_key in listed
        hits += hit
        top3 += q.target_key in listed[:3]
        if not hit:
            found = agent_ranker.search_agents(roster.records(), q.message, limit=10, now=NOW)
            search_hits += any(r.key == q.target_key for r in found)
        by_kind.setdefault(q.kind, []).append(int(hit))
    n = len(queries)
    return {
        "k": k,
        "recall": round(hits / n, 3),
        "recall_with_search": round((hits + search_hits) / n, 3),
        "top3": round(top3 / n, 3),
        "avg_block_tokens": round(tok / n),
        **{f"recall_{kind}": round(sum(v) / len(v), 3) for kind, v in sorted(by_kind.items())},
    }


def baseline_tokens(scenarios: List[Scenario]) -> int:
    block = "\n".join(f'<agent name="{s.record.name}" />' for s in scenarios)
    return len(ENC.encode(block))


# ----------------------------------------------------------------------------- dedupe

SAME_THREAD = [
    ("Email to Alice", "email-to-alice"), ("Email to Alice", "Email  to  Alice!"),
    ("Email to Alice", "Reply to Alice"), ("Email to Alice", "Alice lunch"),
    ("Alice lunch", "Alice lunch plans"), ("Vercel Job Offer", "Vercel offer"),
    ("Vercel Job Offer", "Vercel job offer negotiation"), ("Stripe contract", "Stripe contract review"),
    ("Flight to Tokyo", "Tokyo flights"), ("Tokyo Restaurant Search", "Tokyo restaurants"),
    ("Weekly newsletter digest", "Newsletter digest"), ("Q3 Budget Analysis", "Q3 budget"),
    ("Email to Sharanjeet", "Follow up with Sharanjeet"), ("Keith intro", "Intro to Keith"),
    ("Dentist reminder", "Reminder: dentist appointment"), ("Rent payment reminder", "Rent reminder"),
    ("Email to Bob re invoice", "Bob invoice"), ("Linear onboarding", "Linear onboarding emails"),
    ("Priya interview loop", "Interview with Priya"), ("Ramp renewal", "Ramp subscription renewal"),
]
DIFFERENT_THREAD = [
    ("Email to Alice", "Email to Bob"), ("Alice lunch", "Alice contract"),
    ("Flight to Tokyo", "Flight to Lisbon"), ("Stripe contract", "Notion contract"),
    ("Vercel Job Offer", "Stripe Job Offer"), ("Weekly newsletter digest", "Weekly finance digest"),
    ("Keith intro", "Maria intro"), ("Dentist reminder", "Rent reminder"),
    ("Q3 Budget Analysis", "Q4 Budget Analysis"), ("Bob invoice", "Bob birthday gift"),
    ("Tokyo Restaurant Search", "Flight to Tokyo"), ("Priya interview loop", "Omar interview loop"),
    ("Linear onboarding", "Notion onboarding"), ("Ramp renewal", "Brex renewal"),
    # Ambiguous from names alone; flagging lets the model read descriptions and decide.
    ("Email to Alice", "Alice contract"), ("Email to Bob re invoice", "Bob birthday gift"),
]


def dedupe_eval() -> dict:
    def outcome(existing: str, new: str) -> str:
        if agent_key(existing) == agent_key(new):
            return "auto_reused"
        if name_similarity(existing, new) >= roster_mod.SIMILARITY_THRESHOLD:
            return "flagged"
        return "created"

    same = [outcome(a, b) for a, b in SAME_THREAD]
    diff = [outcome(a, b) for a, b in DIFFERENT_THREAD]
    caught = sum(o != "created" for o in same)
    false_flags = [(a, b) for (a, b), o in zip(DIFFERENT_THREAD, diff) if o != "created"]
    return {
        "same_thread_pairs": len(same),
        "baseline_duplicates_created": sum(a != b for a, b in SAME_THREAD),  # exact string match
        "baseline_slug_collisions": sum(a != b and agent_key(a) == agent_key(b) for a, b in SAME_THREAD),
        "after_auto_reused": same.count("auto_reused"),
        "after_flagged": same.count("flagged"),
        "after_duplicates_created": same.count("created"),
        "after_missed_pairs": [p for p, o in zip(SAME_THREAD, same) if o == "created"],
        "different_thread_pairs": len(diff),
        "after_false_flags": len(false_flags),
        "after_false_flag_pairs": false_flags,
        "catch_rate": round(caught / len(same), 3),
        "false_flag_rate": round(len(false_flags) / len(diff), 3),
    }


def _table(rows: List[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    logging.getLogger("openpoke.server").setLevel(logging.WARNING)
    rng = random.Random(7)
    scenarios = build_registry(rng)
    queries = build_queries(rng, scenarios)

    with tempfile.TemporaryDirectory() as d:
        roster_mod._agent_roster = AgentRoster(Path(d) / "roster.json")
        roster_mod._agent_roster.bulk_load([s.record for s in scenarios])
        triggers_pkg._trigger_service = TriggerService(TriggerStore(Path(d) / "triggers.db"))
        agent_ranker_now = agent_ranker.score_agents
        agent_ranker.score_agents = lambda records, query, now=None: agent_ranker_now(records, query, NOW)

        rows = [{"strategy": "hybrid (ours)", **run_strategy(queries, k, 0.7, 0.3)} for k in K_VALUES]
        ablations = [
            {"strategy": "recency only", **run_strategy(queries, 20, 0.0, 1.0)},
            {"strategy": "BM25 only", **run_strategy(queries, 20, 1.0, 0.0)},
            {"strategy": "hybrid (ours)", **run_strategy(queries, 20, 0.7, 0.3)},
        ]

    results = {
        "agents": len(scenarios),
        "queries": len(queries),
        "baseline": {"recall": 1.0, "block_tokens": baseline_tokens(scenarios), "descriptions": False},
        "k_sweep": rows,
        "ablation_k20": ablations,
        "dedupe": dedupe_eval(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "routing_offline.json").write_text(json.dumps(results, indent=2))

    print(f"# Offline routing eval ({len(scenarios)} agents, {len(queries)} queries)\n")
    print(f"Baseline: all {len(scenarios)} names listed, recall 1.0, {results['baseline']['block_tokens']} tokens, no descriptions\n")
    print("## K sweep\n")
    print(_table(rows))
    print("\n## Ablation at K=20\n")
    print(_table(ablations))
    print("\n## Duplicate-agent detection\n")
    print(json.dumps(results["dedupe"], indent=2))


if __name__ == "__main__":
    main()
