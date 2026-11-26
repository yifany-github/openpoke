"""Live memory-compaction eval: does a bounded, summarized history still answer questions
about old runs as well as the full, unbounded history did?

One execution agent ("Alice threads") accumulates 20 runs. Facts are planted in early
runs (thread ids, emails, invoice numbers, dates, a preference, a trigger's status) and
then buried under routine runs. Compaction runs for real (``agent_memory.compact`` with
the configured summarizer model) exactly as it would in production: WINDOW=5, BATCH=5,
so by run 20 everything before run 16 exists only in the rolling summary.

Then the execution model answers questions whose answers sit in those old runs, once with
the bounded history (after) and once with the full log (before, the original behavior).

    .venv/bin/python -m evals.memory_compaction_llm --yes     # ~$0.15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

import tiktoken

from server.agents.execution_agent import agent as exec_agent_mod
from server.config import get_settings
from server.openrouter_client import request_chat_completion
from server.services.execution import agent_memory, log_store, roster

AGENT = "Alice threads"
ENC = tiktoken.get_encoding("cl100k_base")
RESULTS = Path(__file__).resolve().parent / "results" / "memory_compaction_llm.json"

# (instruction, tool calls [(tool, args, result)], final response)
FILLER = ("Check whether Alice replied on any open thread.",
          [("gmail_search", '{"query": "from:alice@acme.com newer_than:1d"}', '{"messages": []}')],
          "No new replies from Alice. Nothing pending changed.")
RUNS = {
    1: ("Email Alice (alice@acme.com) to ask about lunch next week.",
        [("gmail_create_draft", '{"to": "alice@acme.com", "subject": "Lunch next week?"}',
          '{"draft_id": "r-1001", "thread_id": "18c-lunch-001"}')],
        "Draft r-1001 created on thread 18c-lunch-001, awaiting user confirmation."),
    2: ("User confirmed; send the lunch draft. Note Alice said she prefers Thursdays.",
        [("gmail_execute_draft", '{"draft_id": "r-1001"}', '{"status": "sent"}')],
        "Sent draft r-1001. Alice prefers Thursdays for lunch."),
    3: ("Every Monday at 9am, check whether Alice replied.",
        [("createTrigger", '{"payload": "check Alice replies", "rrule": "FREQ=WEEKLY;BYDAY=MO;BYHOUR=9"}',
          '{"trigger_id": 41, "status": "active"}')],
        "Created trigger 41: weekly on Monday 09:00 to check Alice's replies."),
    4: ("Find the contract email Alice sent and who is cc'd.",
        [("gmail_search", '{"query": "from:alice@acme.com contract"}',
          '{"messages": [{"threadId": "18c-contract-207", "cc": "priya@stripe.com"}]}')],
        "Contract thread is 18c-contract-207; Priya (priya@stripe.com) is cc'd."),
    9: ("Alice says to loop in her assistant for scheduling.",
        [("gmail_search", '{"query": "from:alice@acme.com assistant"}',
          '{"messages": [{"snippet": "please cc my assistant Tom, tom@acme.com"}]}')],
        "Alice's assistant is Tom (tom@acme.com); cc him on scheduling emails."),
    12: ("Alice moved lunch; update the plan.",
         [("gmail_search", '{"query": "thread:18c-lunch-001"}',
           '{"messages": [{"snippet": "Can we do Dec 4 at Nopa instead?"}]}')],
         "Lunch moved to 2026-12-04 at Nopa (thread 18c-lunch-001)."),
    15: ("Alice forwarded an invoice; record the details.",
         [("gmail_search", '{"query": "from:alice@acme.com invoice"}',
           '{"messages": [{"snippet": "Invoice INV-3391 for $1,250 due 2026-12-15"}]}')],
         "Invoice INV-3391 for $1,250, due 2026-12-15."),
    18: ("Pause the Monday reply check for now.",
         [("updateTrigger", '{"trigger_id": 41, "status": "paused"}', '{"trigger_id": 41, "status": "paused"}')],
         "Trigger 41 paused."),
}
N_RUNS = 20
QUESTIONS = [  # (question, expected substrings: any one must appear)
    ("What is the Gmail thread id of the contract thread with Alice?", ["18c-contract-207"]),
    ("Who is cc'd on the contract thread? Give the email address.", ["priya@stripe.com"]),
    ("What is the email address of Alice's assistant?", ["tom@acme.com"]),
    ("Which day of the week does Alice prefer for lunch?", ["thursday"]),
    ("Where and on what date is the lunch with Alice now?", ["nopa"]),
    ("What is the invoice number Alice forwarded?", ["inv-3391"]),
    ("What is the current status of trigger 41?", ["paused"]),
    ("What thread id is the lunch conversation on?", ["18c-lunch-001"]),
]


async def ask(system_prompt: str, question: str, model: str) -> str:
    resp = await request_chat_completion(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Answer from your execution history only, in one short sentence. {question}"}],
        max_tokens=200,
    )
    return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


async def main_async() -> dict:
    settings = get_settings()
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        roster._agent_roster = roster.AgentRoster(base / "roster.json")
        log_store._execution_agent_logs = log_store.ExecutionAgentLogStore(base)
        agent_memory._agent_memory = agent_memory.AgentMemoryStore(base)
        logs = log_store._execution_agent_logs
        roster._agent_roster.add_agent(AGENT)
        agent = exec_agent_mod.ExecutionAgent(AGENT)

        compactions = 0
        for i in range(1, N_RUNS + 1):
            instruction, calls, response = RUNS.get(i, FILLER)
            logs.record_request(AGENT, instruction)
            for tool, args, result in calls:
                agent.record_tool_execution(tool, args, result)
            agent.record_response(response)
            if agent_memory.needs_compaction(AGENT):
                compactions += await agent_memory.compact(AGENT)  # real summarizer model

        bounded = agent.build_system_prompt_with_history()
        full = f"{agent.build_system_prompt()}\n\n# Execution History\n\n{logs.load_transcript(AGENT)}"
        state = agent_memory.get_agent_memory().load(AGENT)

        rows = []
        for question, expected in QUESTIONS:
            a_before, a_after = await asyncio.gather(ask(full, question, settings.execution_agent_model),
                                                     ask(bounded, question, settings.execution_agent_model))
            ok = lambda ans: any(e in ans.lower() for e in expected)
            rows.append({"question": question, "expected": expected,
                         "before_full_history": a_before, "before_correct": ok(a_before),
                         "after_bounded_history": a_after, "after_correct": ok(a_after)})

        return {
            "summarizer_model": settings.summarizer_model,
            "answer_model": settings.execution_agent_model,
            "runs": N_RUNS, "compactions": compactions, "covered_runs": state.covered_runs,
            "prompt_tokens_full": len(ENC.encode(full)), "prompt_tokens_bounded": len(ENC.encode(bounded)),
            "description": state.description, "summary": state.summary,
            "before_correct": sum(r["before_correct"] for r in rows),
            "after_correct": sum(r["after_correct"] for r in rows),
            "questions": len(rows), "rows": rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="confirm spending API credits (~$0.15)")
    args = parser.parse_args()
    logging.getLogger("openpoke.server").setLevel(logging.WARNING)
    if not get_settings().openrouter_api_key:
        sys.exit("Set OPENROUTER_API_KEY to run.")
    if not args.yes:
        sys.exit("Makes ~3 summarizer calls and ~16 answer calls (~$0.15). Re-run with --yes.")
    result = asyncio.run(main_async())
    RESULTS.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "summary")}, indent=2))
    for r in result["rows"]:
        print(f"- {r['question']}\n    before: {'OK ' if r['before_correct'] else 'MISS'} {r['before_full_history'][:110]!r}"
              f"\n    after:  {'OK ' if r['after_correct'] else 'MISS'} {r['after_bounded_history'][:110]!r}")
    print("\nSUMMARY:\n" + result["summary"])


if __name__ == "__main__":
    main()
