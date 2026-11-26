"""Prompt growth: how big do OpenPoke's prompts get as agents multiply (width) and as one
agent is reused (depth)?

Drives the *real* prompt-building code (no LLM, no network) and works against either
checkout, so the same script measures before and after:

    # after (this branch)
    .venv/bin/python -m evals.prompt_growth --label after

    # before (original code in a worktree of main)
    git worktree add ../openpoke-main main
    (cd ../openpoke-main && PYTHONPATH=. ../openpoke/.venv/bin/python ../openpoke/evals/prompt_growth.py --label baseline)

Tokens are counted with tiktoken cl100k_base as a proxy for Claude's tokenizer
(not public); expect +/-15% on absolute numbers. Growth shape is what matters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

import tiktoken

from server.agents.execution_agent import agent as exec_agent_mod
from server.agents.interaction_agent import agent as ia_agent_mod
from server.services.execution import log_store as log_store_mod
from server.services.execution import roster as roster_mod

try:  # only exists after the fix
    from server.services.execution import agent_memory as memory_mod
except ImportError:  # pragma: no cover - baseline checkout
    memory_mod = None

ENC = tiktoken.get_encoding("cl100k_base")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

WIDTHS = [0, 10, 50, 100, 300, 1000]
DEPTHS = [1, 5, 10, 25, 50, 100, 200]
TOOL_CALLS_PER_RUN = 3        # assumption: search + draft + send/trigger
LLM_ITERATIONS_PER_RUN = 4    # assumption: tool rounds + final answer (cap is 8)
CONTEXT_WINDOW = 200_000


def tokens(text: str) -> int:
    return len(ENC.encode(text))


# --------------------------------------------------------------------------- width

PEOPLE = ["Alice", "Bob", "Sharanjeet", "Keith", "Maria", "Chen", "Priya", "Omar",
          "Lena", "Diego", "Yuki", "Fatima", "Tom", "Grace", "Ivan", "Nora"]
COMPANIES = ["Vercel", "Stripe", "Notion", "Linear", "Figma", "Ramp", "Anthropic", "OpenAI"]
CITIES = ["Tokyo", "Lisbon", "NYC", "Berlin", "Austin", "Seoul"]
TOPICS = ["lunch", "contract", "invoice", "intro", "follow-up", "offer", "feedback", "meeting"]
TEMPLATES = [
    "Email to {p}", "{p} {t}", "Reply to {p} re {t}", "{c} Job Offer", "{c} {t}",
    "Flight to {city}", "{city} Restaurant Search", "Reminder: {t} with {p}",
    "Weekly {t} digest", "Search emails from {p}", "Forward {c} {t} to {p}",
]


def synthetic_agents(n: int, seed: int = 0) -> list[tuple[str, str]]:
    """(name, description) pairs; descriptions are ~1 line like the fix produces."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    while len(out) < n:
        p, c, city, t = rng.choice(PEOPLE), rng.choice(COMPANIES), rng.choice(CITIES), rng.choice(TOPICS)
        name = rng.choice(TEMPLATES).format(p=p, c=c, city=city, t=t)
        if name in seen:
            name = f"{name} {len(out)}"
        seen.add(name)
        desc = f"Handles the {t} thread with {p} ({p.lower()}@{c.lower()}.com) at {c}; last draft awaiting confirmation."
        out.append((name, desc))
    return out


def measure_width(tmp: Path) -> list[dict]:
    rows = []
    for n in WIDTHS:
        roster = roster_mod.AgentRoster(tmp / f"roster_{n}.json")
        roster_mod._agent_roster = roster
        for name, desc in synthetic_agents(n):
            roster.add_agent(name)
            if hasattr(roster, "update_description"):
                roster.update_description(name, desc)
        try:
            block = ia_agent_mod._render_active_agents("can you follow up with Priya about the Stripe contract")
        except TypeError:  # baseline signature takes no query
            block = ia_agent_mod._render_active_agents()
        rows.append({
            "agents": n,
            "roster_tokens": tokens(block),
            "agents_listed": block.count("<agent "),
            "system_prompt_tokens": tokens(ia_agent_mod.build_system_prompt()),
        })
    return rows


# --------------------------------------------------------------------------- depth

INSTRUCTION = ("Email Alice to ask if she's free for lunch tomorrow around noon near the office. "
               "Keep it short and friendly, and show me the draft before sending.")
TOOL_ARGS = json.dumps({"to": "alice@example.com", "subject": "Lunch tomorrow?",
                        "body": "Hey Alice, are you free for lunch tomorrow around noon? " * 4})
TOOL_RESULT = json.dumps({"messages": [{"id": f"18c{i:05d}", "threadId": f"18c{i:05d}",
                                        "from": "alice@example.com", "subject": "Re: lunch",
                                        "snippet": "Sounds good, see you then! " * 3}
                                       for i in range(10)]})
FINAL_RESPONSE = ("Draft created. To: alice@example.com | Subject: Lunch tomorrow? | Body: Hey Alice, "
                  "are you free for lunch tomorrow around noon? Awaiting user confirmation before "
                  "sending. Draft id r-123456789. Thread 18c00001 found from last week.")


async def worst_case_summarizer(system_prompt: str, content: str) -> str:
    """Stand-in for the LLM that always returns a summary at the hard size cap."""
    head = "DESCRIPTION: Lunch scheduling and email threads with Alice (alice@example.com)\nMEMORY:\n"
    filler = "- thread 18c00001, draft r-123456789, prefers noon near the office\n"
    return (head + filler * 200)[: memory_mod.MAX_SUMMARY_CHARS]


def measure_depth(tmp: Path, mode: str) -> list[dict]:
    store = log_store_mod.ExecutionAgentLogStore(tmp / f"exec_{mode}")
    log_store_mod._execution_agent_logs = store
    exec_agent_mod.get_execution_agent_logs = lambda: store  # baseline binds the name at import
    if memory_mod is not None:
        memory_mod._agent_memory = memory_mod.AgentMemoryStore(tmp / f"exec_{mode}")
    agent = exec_agent_mod.ExecutionAgent("Email to Alice")
    base_tokens = tokens(agent.build_system_prompt())

    rows = []
    runs_done = 0
    peak = 0
    for target in DEPTHS:
        while runs_done < target:
            store.record_request(agent.name, INSTRUCTION)
            for _ in range(TOOL_CALLS_PER_RUN):
                agent.record_tool_execution("gmail_search", TOOL_ARGS, TOOL_RESULT)
            agent.record_response(FINAL_RESPONSE)
            runs_done += 1
            # Peak = prompt the *next* run would see, measured before compaction catches up.
            peak = max(peak, tokens(agent.build_system_prompt_with_history()))
            if mode == "compaction" and memory_mod.needs_compaction(agent.name):
                asyncio.run(memory_mod.compact(agent.name, worst_case_summarizer))
        prompt_tokens = tokens(agent.build_system_prompt_with_history())
        rows.append({
            "runs": target,
            "system_prompt_tokens": prompt_tokens,
            "history_tokens": prompt_tokens - base_tokens,
            "peak_prompt_tokens_so_far": peak,
            "tokens_per_run_est": prompt_tokens * LLM_ITERATIONS_PER_RUN,
            "fits_context": prompt_tokens < CONTEXT_WINDOW,
        })
    return rows


def measure_slug_collision(tmp: Path) -> dict:
    roster = roster_mod.AgentRoster(tmp / "collision_roster.json")
    store = log_store_mod.ExecutionAgentLogStore(tmp / "collision_logs")
    names = ["Email to Alice", "email-to-alice", "Email  to  Alice!"]
    for name in names:
        roster.add_agent(name)
        store.record_request(name, f"request for {name!r}")
    return {"roster_entries": len(roster.get_agents()),
            "log_files": len(list((tmp / "collision_logs").glob("*.log"))), "names": names}


# --------------------------------------------------------------------------- main

def _table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(f"{r[k]:,}" if isinstance(r[k], int) and not isinstance(r[k], bool)
                                else str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    import logging

    logging.getLogger("openpoke.server").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="after" if memory_mod else "baseline")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        results = {
            "label": args.label,
            "tokenizer": "cl100k_base (proxy for Claude)",
            "assumptions": {"tool_calls_per_run": TOOL_CALLS_PER_RUN,
                            "llm_iterations_per_run": LLM_ITERATIONS_PER_RUN},
            "width": measure_width(tmp),
            "depth": measure_depth(tmp, "no_compaction"),
            "slug_collision": measure_slug_collision(tmp),
        }
        if memory_mod is not None:
            results["depth_with_compaction"] = measure_depth(tmp, "compaction")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"prompt_growth_{args.label}.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"# Prompt growth ({args.label})\n")
    print("## Width: interaction-agent roster block\n")
    print(_table(results["width"]))
    title = "no compaction" if memory_mod else "original code"
    print(f"\n## Depth: execution-agent system prompt ({title})\n")
    print(_table(results["depth"]))
    if "depth_with_compaction" in results:
        print("\n## Depth: with compaction (summary at its 4000-char worst case)\n")
        print(_table(results["depth_with_compaction"]))
    print("\n## Slug collision\n")
    print(json.dumps(results["slug_collision"], indent=2))
    print(f"\nSaved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
