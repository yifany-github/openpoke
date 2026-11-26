"""A second, independently written routing workload ("fresh").

The dev workload in ``routing_offline.py`` was used while tuning the prompt, and the prompt's
examples once quoted its names. This workload shares **no** people, organisations, topics,
synonyms, agent-name templates or query templates with it or with the system prompt. It is
also closer to real use:
- each person belongs to one organisation, not a random one per thread;
- 20% of agents have no description at all, like agents migrated from the original roster;
- the remaining descriptions are written by the production code path (`agent_memory.describe`,
  the configured small model) from a synthetic first-run log, not by a template. They are
  cached in ``results/fresh_descriptions.json`` so runs are reproducible.

    .venv/bin/python -m evals.workload_fresh --generate-descriptions   # one-off, ~300 small-model calls (~$0.05)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

from evals.routing_offline import NOW, Query, Scenario
from server.services.execution.roster import AgentRecord, agent_key

DESCRIPTIONS_PATH = Path(__file__).resolve().parent / "results" / "fresh_descriptions.json"
N_AGENTS = 300
N_QUERIES = 240
NO_DESCRIPTION_FRACTION = 0.2

PEOPLE = ["Aurelio", "Bettina", "Cosmin", "Delphine", "Emeka", "Freya", "Gustavo", "Hiroshi", "Ingrid",
          "Jamal", "Katya", "Leopold", "Marisol", "Nikhil", "Oksana", "Pablo", "Rosalind", "Soren",
          "Thandiwe", "Ulrich", "Valentina", "Wesley", "Ximena", "Yusuf", "Zelda", "Anouk", "Benedikt",
          "Calla", "Dmitri", "Esperanza"]
ORGS = ["Hollis Legal", "Brightline Dental", "Cedarwood Properties", "Northwind Freight", "Juniper Bank",
        "Atlas Movers", "Quill Press", "Orbit Gym", "Maple Insurance", "Harbor Vet"]
TOPICS = {  # topic -> paraphrases a user might say instead
    "lease": ["rental agreement", "tenancy paperwork"],
    "appointment": ["booking", "visit"],
    "refund": ["reimbursement", "money back"],
    "quote": ["estimate", "pricing"],
    "claim": ["insurance claim", "case"],
    "membership": ["subscription", "plan"],
    "manuscript": ["draft chapters", "book draft"],
    "shipment": ["delivery", "package"],
    "statement": ["account statement", "bank letter"],
    "deposit": ["down payment", "security deposit"],
    "inspection": ["walkthrough", "site visit"],
    "invitation": ["invite", "RSVP"],
}
NAME_TEMPLATES = ["{t} - {p}", "{o} {t}", "{p} ({o})", "Follow-ups with {p}", "{t} for {p}",
                  "{o}: {t}", "Thread with {p} on {t}"]
INSTRUCTION_TEMPLATES = [  # what the interaction agent might write when spawning the agent
    "Email {p} at {o} ({email}) about the {t}.",
    "Reach out to {p} about the {t} and ask for an update.",
    "Write to {email} regarding the {t} with {o}.",
    "Draft a note to {p} from {o} following up on the {t}.",
]
EXPLICIT = ["has {p} gotten back to us on the {ts}?", "chase {p} on the {ts} please",
            "what's the latest from {p} regarding the {ts}", "remind {p} about the {ts}"]
VAGUE = ["is the {o} {ts} sorted yet?", "anything new from {o} about the {ts}?",
         "where did we land with {o} on the {ts}"]
PERSON_ONLY = ["heard back from {p}?", "nudge {p}", "any word from {p}?"]
FOLLOWUP = ["send it", "perfect, go ahead and send", "make it friendlier and send", "yep"]
KINDS = ["explicit", "paraphrase", "vague", "person_only", "followup"]


def _email(person: str, org: str) -> str:
    return f"{person.lower()}@{org.lower().replace(' ', '')}.com"


def build_scenarios(rng: random.Random) -> List[Scenario]:
    org_of = {p: rng.choice(ORGS) for p in PEOPLE}
    combos = [(p, t) for p in PEOPLE for t in TOPICS]
    rng.shuffle(combos)
    scenarios, used = [], set()
    for person, topic in combos[:N_AGENTS]:
        org = org_of[person]
        name = rng.choice(NAME_TEMPLATES).format(p=person, o=org, t=topic)
        if agent_key(name) in used:
            name = f"{topic} - {person}"
        used.add(agent_key(name))
        stamp = (NOW - timedelta(days=rng.uniform(0, 60))).isoformat()
        record = AgentRecord(name=name, key=agent_key(name), created_at=stamp, last_used_at=stamp,
                             run_count=rng.randint(1, 12))
        scenarios.append(Scenario(person, org, topic, record))
    return scenarios


def first_run_log(s: Scenario, rng: random.Random) -> tuple[str, str]:
    instruction = rng.choice(INSTRUCTION_TEMPLATES).format(p=s.person, o=s.company, t=s.topic,
                                                           email=_email(s.person, s.company))
    response = f"Draft created to {_email(s.person, s.company)} about the {s.topic}; waiting for the user to confirm."
    return instruction, response


def build_registry(rng: random.Random) -> List[Scenario]:
    """Scenarios with descriptions from the cache (LLM-written) and 20% left blank."""
    scenarios = build_scenarios(rng)
    cache: Dict[str, str] = json.loads(DESCRIPTIONS_PATH.read_text()) if DESCRIPTIONS_PATH.exists() else {}
    blank = set(rng.sample(range(len(scenarios)), int(len(scenarios) * NO_DESCRIPTION_FRACTION)))
    for i, s in enumerate(scenarios):
        s.record.description = "" if i in blank else cache.get(s.record.key, "")
    return scenarios


def build_queries(rng: random.Random, scenarios: List[Scenario]) -> List[Query]:
    by_org_topic: Dict[tuple, List[Scenario]] = {}
    latest: Dict[str, Scenario] = {}
    for s in scenarios:
        by_org_topic.setdefault((s.company, s.topic), []).append(s)
        if s.person not in latest or s.record.last_used_dt > latest[s.person].record.last_used_dt:
            latest[s.person] = s
    unique_ot = [v[0] for v in by_org_topic.values() if len(v) == 1]
    neutral = "<user_message>can you check my schedule for friday?</user_message>\n<poke_reply>Friday's open after 2.</poke_reply>"
    queries = []
    for i in range(N_QUERIES):
        kind = KINDS[i % len(KINDS)]
        s = rng.choice(scenarios)
        if kind == "vague":
            s = rng.choice(unique_ot)
            msg = rng.choice(VAGUE).format(o=s.company, ts=rng.choice(TOPICS[s.topic]))
            queries.append(Query(kind, msg, neutral, s.record.key))
        elif kind == "person_only":
            s = latest[s.person]
            queries.append(Query(kind, rng.choice(PERSON_ONLY).format(p=s.person), neutral, s.record.key))
        elif kind == "followup":
            transcript = (f"<user_message>write to {s.person} about the {s.topic}</user_message>\n"
                          f"<agent_message>[SUCCESS] {s.record.name}: Draft ready for "
                          f"{_email(s.person, s.company)}</agent_message>\n"
                          f"<poke_reply>Draft for {s.person} is ready. Send?</poke_reply>")
            queries.append(Query(kind, rng.choice(FOLLOWUP), transcript, s.record.key))
        else:
            ts = s.topic if kind == "explicit" else rng.choice(TOPICS[s.topic])
            queries.append(Query(kind, rng.choice(EXPLICIT).format(p=s.person, ts=ts), neutral, s.record.key))
    return queries


async def generate_descriptions(seed: int = 21) -> Dict[str, str]:
    """Run the production `describe` path once per agent on a synthetic first-run log."""
    from server.services.execution import agent_memory, log_store, roster

    rng = random.Random(seed)
    scenarios = build_scenarios(random.Random(seed))
    out: Dict[str, str] = {}
    sem = asyncio.Semaphore(8)
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        roster._agent_roster = roster.AgentRoster(base / "roster.json")
        log_store._execution_agent_logs = log_store.ExecutionAgentLogStore(base)
        agent_memory._agent_memory = agent_memory.AgentMemoryStore(base)
        logs = log_store._execution_agent_logs
        for s in scenarios:
            instruction, response = first_run_log(s, rng)
            roster._agent_roster.add_agent(s.record.name)
            logs.record_request(s.record.name, instruction)
            logs.record_agent_response(s.record.name, response)

        async def one(s: Scenario) -> None:
            async with sem:
                await agent_memory.describe(s.record.name)
                out[s.record.key] = agent_memory.get_agent_memory().load(s.record.name).description

        await asyncio.gather(*(one(s) for s in scenarios))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-descriptions", action="store_true")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()
    if args.generate_descriptions:
        descriptions = asyncio.run(generate_descriptions(args.seed))
        DESCRIPTIONS_PATH.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False))
        filled = sum(bool(v) for v in descriptions.values())
        print(f"wrote {filled}/{len(descriptions)} descriptions to {DESCRIPTIONS_PATH}")
        for k, v in list(descriptions.items())[:5]:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
