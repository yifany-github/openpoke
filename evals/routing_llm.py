"""Live routing eval: with a real LLM, does the interaction agent pick the right agent?

The offline eval shows the target agent stays *visible*. This one measures what actually
matters: given the prompt, does the model reuse the correct execution agent, create a
needless duplicate, or misroute?

Same 300-agent registry and query mix as ``routing_offline.py``, two conditions:

- **baseline**: original system prompt (read from git ``main``), all agent names, no
  descriptions, original tools. Any agent_name is accepted.
- **after**: this branch's prompt + top-K roster with descriptions + ``search_agents`` +
  the real ``send_message_to_agent`` name resolution (similar-name rejection included).

For each query the interaction loop runs (max 4 LLM calls) until the model dispatches to an
agent. Nothing is actually executed.

    OPENROUTER_API_KEY=... .venv/bin/python -m evals.routing_llm --n 30 --yes

Cost: about 30 queries x 2 conditions x 1-3 calls x ~5k input tokens, roughly $1-3 on Sonnet.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from evals.routing_offline import KINDS, NOW, Query, build_queries, build_registry
from server.agents.interaction_agent import agent as ia_agent
from server.agents.interaction_agent import tools as ia_tools
from server.config import get_settings
from server.openrouter_client import request_chat_completion
from server.services import triggers as triggers_pkg
from server.services.execution import agent_ranker
from server.services.execution import roster as roster_mod
from server.services.execution.roster import AgentRoster, agent_key
from server.services.triggers.service import TriggerService
from server.services.triggers.store import TriggerStore

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MAX_CALLS = 4
CALL_DELAY = 0.0  # seconds between LLM calls (--delay)
REPO_ROOT = Path(__file__).resolve().parent.parent


def baseline_system_prompt() -> str:
    return subprocess.check_output(
        ["git", "show", "main:server/agents/interaction_agent/system_prompt.md"], cwd=REPO_ROOT, text=True
    ).strip()


def baseline_tools() -> List[Dict[str, Any]]:
    schemas = copy.deepcopy([t for t in ia_tools.TOOL_SCHEMAS if t["function"]["name"] != "search_agents"])
    for t in schemas:
        if t["function"]["name"] == "send_message_to_agent":
            t["function"]["parameters"]["properties"].pop("force_new", None)
    return schemas


def baseline_content(q: Query, names: List[str]) -> str:
    roster_block = "\n".join(f'<agent name="{n}" />' for n in names)
    return (f"<conversation_history>\n{q.transcript}\n</conversation_history>\n\n"
            f"<active_agents>\n{roster_block}\n</active_agents>\n\n"
            f"<new_user_message>\n{q.message}\n</new_user_message>")


async def run_one(q: Query, condition: str, model: str, records, system_prompt: str, tools) -> Dict[str, Any]:
    roster = roster_mod.get_agent_roster()
    roster.bulk_load(copy.deepcopy(records))  # fresh registry per query
    known = {r.key for r in records}

    if condition == "baseline":
        messages = [{"role": "user", "content": baseline_content(q, [r.name for r in records])}]
    else:
        messages = ia_agent.prepare_message_with_history(q.message, q.transcript)

    out: Dict[str, Any] = {"kind": q.kind, "message": q.message, "target": q.target_key,
                           "chosen": None, "prompt_tokens": 0, "calls": 0, "rejections": 0, "searches": 0}
    for _ in range(MAX_CALLS):
        if CALL_DELAY:
            await asyncio.sleep(CALL_DELAY)  # stay under free-tier per-minute limits
        resp = await request_chat_completion(model=model, messages=messages, system=system_prompt, tools=tools)
        out["calls"] += 1
        out["prompt_tokens"] += (resp.get("usage") or {}).get("prompt_tokens", 0)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": tool_calls} if tool_calls else {})})
        if not tool_calls:
            if msg.get("content"):
                out.setdefault("said_to_user", []).append(msg["content"])
            break
        dispatched: Optional[str] = None
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "send_message_to_agent" and condition == "baseline":
                payload = {"status": "submitted", "agent_name": args.get("agent_name")}
                dispatched = dispatched or args.get("agent_name")
            elif name in {"send_message_to_agent", "search_agents"}:
                out["searches"] += name == "search_agents"
                result = ia_tools.handle_tool_call(name, args)  # real resolution logic
                payload = result.payload
                if name == "send_message_to_agent":
                    if result.success:
                        dispatched = dispatched or payload["agent_name"]
                    else:
                        out["rejections"] += 1
            else:
                if name == "send_message_to_user":
                    out.setdefault("said_to_user", []).append(args.get("message", ""))
                payload = {"status": "ok"}
            messages.append({"role": "tool", "tool_call_id": call.get("id") or name, "content": json.dumps(payload)})
        if dispatched:
            out["chosen"] = dispatched
            break

    chosen_key = agent_key(out["chosen"]) if out["chosen"] else None
    out["correct"] = chosen_key == q.target_key
    out["outcome"] = ("correct" if out["correct"] else "no_dispatch" if chosen_key is None
                      else "new_duplicate" if chosen_key not in known else "wrong_existing")
    return out


def summarize(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rates are over queries that got an answer; API failures are reported separately."""
    rows = [r for r in all_rows if r["outcome"] != "api_error"] or [{"outcome": "none", "correct": False,
                                                                     "kind": "", "prompt_tokens": 0, "calls": 0}]
    n = len(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    by_kind = {k: round(sum(r["correct"] for r in rows if r["kind"] == k) /
                        sum(r["kind"] == k for r in rows), 3) for k in KINDS if any(r["kind"] == k for r in rows)}
    return {
        "n": n,
        "api_errors": sum(r["outcome"] == "api_error" for r in all_rows),
        "accuracy": round(outcomes["correct"] / n, 3),
        "new_duplicate_rate": round(outcomes["new_duplicate"] / n, 3),
        "wrong_existing_rate": round(outcomes["wrong_existing"] / n, 3),
        "no_dispatch_rate": round(outcomes["no_dispatch"] / n, 3),
        "avg_prompt_tokens_per_query": round(sum(r["prompt_tokens"] for r in rows) / n),
        "avg_llm_calls": round(sum(r["calls"] for r in rows) / n, 2),
        "accuracy_by_kind": by_kind,
    }


CONDITIONS = {
    "baseline": "original prompt, all agent names, no descriptions, original tools",
    "after": "fixed system: top-K roster with descriptions + search_agents + name resolution",
    "full_roster": "ablation: as 'after' but ALL agents listed (no top-K) - isolates what top-K adds",
    "after_heuristic_desc": "ablation: as 'after' but descriptions are what the live system writes before "
                            "the first compaction ('Started for: <first instruction> | Latest: ...'), "
                            "without company or email",
}


def heuristic_records(scenarios):
    """Descriptions as the production heuristic would write them early in an agent's life:
    the first instruction plus the latest outcome. Deliberately omits company and email."""
    out = []
    for s in scenarios:
        r = copy.deepcopy(s.record)
        r.description = (f"Started for: Email {s.person} about the {s.topic}. | Latest: Draft created for "
                         f"{s.person}, waiting for the user to confirm before sending.")
        out.append(r)
    return out


async def main_async(args) -> None:
    settings = get_settings()
    model = args.model or settings.interaction_agent_model
    rng = random.Random(args.seed)
    if args.vocab == "fresh":
        from evals import workload_fresh as W

        rng = random.Random(21)  # descriptions in results/fresh_descriptions.json are keyed to this seed
        scenarios = W.build_registry(rng)
        queries = W.build_queries(rng, scenarios)
    else:
        scenarios = build_registry(rng)
        queries = build_queries(rng, scenarios)
    per_kind = max(1, args.n // len(KINDS))
    kinds = args.kinds.split(",") if args.kinds else KINDS
    sample = [q for k in kinds for q in [x for x in queries if x.kind == k][:per_kind]]
    records = [s.record for s in scenarios]

    real_score = agent_ranker.score_agents
    agent_ranker.score_agents = lambda r, query, now=None: real_score(r, query, NOW)
    with tempfile.TemporaryDirectory() as d:
        roster_mod._agent_roster = AgentRoster(Path(d) / "roster.json")
        triggers_pkg._trigger_service = TriggerService(TriggerStore(Path(d) / "triggers.db"))

        async def _noop(*a, **k):
            from types import SimpleNamespace

            return SimpleNamespace(success=True)

        ia_tools._EXECUTION_BATCH_MANAGER.execute_agent = _noop

        results: Dict[str, Any] = {"model": model, "agents": len(records)}
        results["seed"] = args.seed
        results["vocab"] = args.vocab
        default_k = settings.agent_roster_prompt_limit
        for condition in args.conditions.split(","):
            if condition not in CONDITIONS:
                sys.exit(f"unknown condition {condition!r}; choose from {', '.join(CONDITIONS)}")
            system_prompt = baseline_system_prompt() if condition == "baseline" else ia_agent.build_system_prompt()
            tools = baseline_tools() if condition == "baseline" else ia_tools.TOOL_SCHEMAS
            # Ablations: same prompt/tools/registry as "after", one factor changed.
            settings.agent_roster_prompt_limit = 10_000 if condition == "full_roster" else default_k
            cond_records = heuristic_records(scenarios) if condition == "after_heuristic_desc" else records
            rows = []
            for i, q in enumerate(sample, 1):
                try:
                    row = await run_one(q, condition, model, cond_records, system_prompt, tools)
                except Exception as exc:  # one flaky API call must not sink the whole run
                    row = {"kind": q.kind, "message": q.message, "target": q.target_key, "chosen": None,
                           "prompt_tokens": 0, "calls": 0, "correct": False, "outcome": "api_error",
                           "error": str(exc)[:300]}
                rows.append(row)
                print(f"[{condition} {i}/{len(sample)}] {row['outcome']:<14} {q.kind:<12} {q.message!r} -> {row['chosen']!r}",
                      file=sys.stderr)
            results[condition] = {"summary": summarize(rows), "rows": rows}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps({c: results[c]["summary"] for c in args.conditions.split(",")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="queries (split evenly across kinds)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--yes", action="store_true", help="confirm spending API credits")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between LLM calls (free tier: ~3.5)")
    parser.add_argument("--out", default="routing_llm.json", help="results filename under evals/results/")
    parser.add_argument("--seed", type=int, default=7, help="workload seed (7 = dev set, 11 = held-out)")
    parser.add_argument("--conditions", default="baseline,after")
    parser.add_argument("--kinds", default="", help="comma-separated subset of query kinds")
    parser.add_argument("--vocab", choices=["dev", "fresh"], default="dev",
                        help="dev = templates used while tuning (seed picks a sample); fresh = independent workload")
    args = parser.parse_args()
    global CALL_DELAY
    CALL_DELAY = args.delay
    logging.getLogger("openpoke.server").setLevel(logging.WARNING)

    if not (os.getenv("OPENROUTER_API_KEY") or get_settings().openrouter_api_key):
        sys.exit("Set OPENROUTER_API_KEY (or add it to .env) to run the live routing eval.")
    if not args.yes:
        sys.exit(f"This makes ~{args.n * 2 * 2} LLM calls (~$1-3). Re-run with --yes to proceed.")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
