# Fixing agent overload in OpenPoke

## TL;DR

OpenPoke's execution agents live forever, but nothing manages their lifecycle. Agents have
no description, never retire, and their memory is never compacted. Overload shows up in two
directions that the system prompt pushes at the same time:

- **Width (too many agents).** Every agent ever created is listed by name only in every
  interaction-agent turn. Routing gets harder as the roster grows: the model can't tell
  agents apart, so it creates near-duplicates, which makes routing harder still.
- **Depth (agents too fat).** Every run of an execution agent injects the agent's *entire*
  log into its system prompt. This is where the tokens actually explode: about 930 tokens per
  reuse with no cap. After 200 reuses one call is ~187k tokens: the whole context window of
  the model OpenPoke shipped with (Sonnet 4, 200k). Today's 1M-context models would still
  run it, at ~750k input tokens (~$1.50 on Sonnet 5.5) per task.

The fix:

| | Before | After |
|---|---|---|
| Roster block, 300 agents | 3,263 tok, names only | **928 tok** (top 20 + descriptions), −72% |
| Roster block, 1,000 agents | 11,230 tok | **942 tok**, −92% |
| Execution prompt after 50 reuses | 47,237 tok | **≤ 11,372 tok** (peak, worst-case summary) |
| Execution prompt after 200 reuses | 186,587 tok (≈ Sonnet 4's 200k window) | **≤ 11,372 tok** |
| Same-thread names creating a duplicate agent | 20 / 20 | **0 / 20** (2 auto-reused, 18 flagged) |
| Target agent visible to the router (offline, 300 agents) | 100% (by listing everything) | **97.5%** at K=20, using 61% fewer tokens |
| **Live routing, Sonnet 5.5, unseen workload: silent misroutes** (wrong agent + needless new agent) | 10 / 30 (33%) | **2 / 30 (7%)**, p = 0.01 |
| Live routing: correct agent chosen | 20 / 30 (67%) | **28 / 30 (93%)**, p = 0.01 |

Numbers come from `evals/` (see [How it was measured](#how-it-was-measured)). Token counts
use `cl100k_base` as a stand-in for Claude's tokenizer (±15%). What matters is the shape of
the curves, not the exact values.

---

## 1. Where the overload is in the code

| Location | Problem |
|---|---|
| `services/execution/roster.py` | Roster is a JSON `list[str]`. It is append-only, has no metadata, and can only be cleared wholesale. |
| `agents/interaction_agent/agent.py::_render_active_agents` | Emits `<agent name="…"/>` for **every** agent on **every** turn. |
| `agents/interaction_agent/tools.py::send_message_to_agent` | New-vs-reuse decided by **exact string** match on an LLM-invented name. |
| `services/execution/log_store.py::_slugify` | Log files are keyed by slug. "Email to Alice" and "email-to-alice" are 2 roster agents sharing **1 log file**, so their memories cross. |
| `agents/execution_agent/agent.py::build_system_prompt_with_history` | Injects the full log every run; `conversation_limit` defaults to `None`. |
| `agents/interaction_agent/system_prompt.md` | "split into as much concurrent calls as possible" (pushes width) plus "prefer existing agent… even if the name is unrelated" (pushes depth). |

Measured baseline (`evals/results/prompt_growth_baseline.json`, original code):

- Width: about 11 tokens per agent, linear. 300 agents is 1.5× the whole system prompt.
- Depth: about 930 tokens per run, linear and unbounded. 200 reuses is ~187k tokens per LLM call
  (about 750k per run at ~4 calls).

At realistic scale, width is not mainly a token problem: 3k tokens is affordable. Width is a
**routing-quality** problem. Depth is the **token** problem. Each needed its own fix.

## 2. Design

### 2a. Depth: bounded agent memory (`services/execution/agent_memory.py`)

The history an agent sees is now:

```
## Memory summary (runs 1..C)     <- rolling LLM summary, hard cap 4,000 chars
## Recent runs (C+1..N)           <- raw log, between WINDOW and WINDOW+BATCH-1 runs
```

- After each run, if the number of unsummarized runs reaches `WINDOW + BATCH` (5 + 5 by
  default), a **background** task folds the oldest ones into the summary and keeps the newest
  `WINDOW` runs raw. This mirrors the conversation summarizer the interaction agent already
  has, so the two memory layers are now symmetric. The fold is incremental (previous summary
  plus the new runs only) and single-flight per agent.
- The summary prompt requires keeping identifiers verbatim (emails, thread, draft and trigger
  ids, RRULEs). Agent memory is useless if "reply to that thread" loses the thread id.
- **Fallback.** If summarization fails or no LLM is configured, the prompt is still bounded:
  only the newest `WINDOW + BATCH` raw runs are shown, with an explicit
  `N older run(s) omitted` note. The bound never depends on the LLM.
- The summary's first line is `DESCRIPTION: …`. That one line becomes the agent's
  description in the registry (2b), so a single LLM call serves both fixes.

Alternatives considered:
- **Pure sliding window.** Cheapest, but it drops exactly the long-lived context that makes
  persistent agents worth having.
- **Retrieval over the agent's own log.** More machinery than one agent's history needs today.

### 2b. Width: a registry, a relevance-ranked roster, and search

**Registry** (`roster.py`). Each agent record now holds `name`, `key` (normalized identity,
identical to the log slug), `description`, `created_at`, `last_used_at` and `run_count`.
Saves are atomic (temp file + rename). The legacy list format migrates on load, and
slug-colliding duplicates collapse.

**What the interaction agent sees** (`agent_ranker.py`, `interaction_agent/agent.py`):

```xml
<agent name="Stripe contract" last_used="2026-11-18" runs="4">Contract review with Priya at Stripe…</agent>
…
<more_agents count="280">Older or less relevant agents are not listed. … call search_agents …</more_agents>
```

- **Pinned, always shown.** Agents with an active trigger (they wake up by themselves, so
  their results must stay routable) and agents named in the current turn (e.g. the sender of
  a batch result).
- **Top-K of the rest.** The score is `0.7 · BM25(name×2 + description) + 0.3 · recency`. The
  query is the current message plus the last ~1,500 characters of the conversation, so a bare
  "yes, send it" still finds the agent that just produced the draft. Recency decays with a
  3-day half-life.
- **`search_agents(query)` tool.** Reaches everything that isn't listed. The prompt tells the
  model to search before creating an agent for something that sounds like earlier work.
- **No explicit archive state.** Agents that fall out of the top-K are effectively archived,
  and they come back as soon as they become relevant or get used. That gives one less state
  machine to keep consistent with triggers and logs.

BM25 is about 40 lines of pure Python: deterministic, dependency-free, and easy to test. The
scorer is a pure function, so swapping in embeddings later is a local change (see
limitations).

**Choosing K.** I swept K ∈ {5, 10, 15, 20, 30} offline and picked the smallest K with recall
≥ 0.95, which was **K = 20**. It is configurable via `OPENPOKE_AGENT_ROSTER_LIMIT`.

**Duplicate prevention** (`send_message_to_agent`):

1. Same normalized key ("email-to-alice" for "Email to Alice"): reuse automatically under the
   canonical name. This also fixes the log-collision bug.
2. Similar content tokens (generic words like *email/reply/to* ignored, plurals folded, one
   name's tokens a subset of the other's or Jaccard ≥ 0.6): **reject** and return the
   candidates with their descriptions. The model then reuses one, or retries with
   `force_new=true`.
3. Otherwise: create the agent.

Similar names are rejected-with-candidates instead of auto-merged on purpose. "Alice lunch"
and "Alice contract" can be genuinely parallel work, and only the model (reading the
descriptions) can tell. A false flag costs one extra LLM round trip. A wrong merge would
corrupt an agent's memory.

**Prompt changes.** Parallelize only *independent* work. Name agents after the thread of work
they own. Explain `<active_agents>`, `<more_agents>`, `search_agents` and
`similar_agents_exist`.

## 3. How it was measured

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r server/requirements.txt pytest tiktoken
.venv/bin/python -m pytest                       # 49 unit/integration tests
.venv/bin/python -m evals.prompt_growth          # width/depth token curves (after)
.venv/bin/python -m evals.routing_offline        # routing recall, K sweep, ablations, dedupe
OPENROUTER_API_KEY=… .venv/bin/python -m evals.routing_llm --n 30 --yes --seed 11 --delay 2   # live LLM routing (~$0.5)
```

Baseline numbers come from running the *same* `prompt_growth.py` against an untouched
checkout of `main`:

```bash
git worktree add ../openpoke-main main
cd ../openpoke-main && PYTHONPATH=. ../openpoke/.venv/bin/python ../openpoke/evals/prompt_growth.py --label baseline
```

### Tests (`tests/`, 49 tests, no network)

All stores (roster, logs, memory, triggers) are redirected to `tmp_path`, and the LLM is
stubbed.
- `test_roster.py`: key normalization, legacy migration, idempotent add, persistence,
  similarity edge cases (subset vs. different person vs. generic-only names).
- `test_agent_ranker.py`: relevance beats recency, recency breaks ties, top-K and hidden
  count, pinned agents survive, search, batch-result mentions.
- `test_interaction_routing.py`: rendered roster (top-K, descriptions, `<more_agents>`,
  trigger pinning, transcript-tail query), respelled-name reuse, similar-name rejection,
  `force_new`, search reaching hidden agents.
- `test_agent_memory.py`: run splitting, bounded fallback without an LLM, compaction
  threshold, incremental folding, identifiers surviving compaction, flat prompt size over 60
  runs, failure handling (state untouched, retry allowed), and an end-to-end
  `ExecutionAgentRuntime` run with a stubbed LLM.
- `test_runtime_concurrency.py`: an end-to-end interaction turn where the scripted LLM invents
  a near-duplicate name, gets `similar_agents_exist`, and reuses the existing agent;
  interaction turns serialized; same-agent runs serialized with contiguous logs; triggers
  using the shared batch manager.
- `test_llm_client.py`: `max_tokens` always sent; 429/503 retried; 402 not retried;
  concurrency capped (via `httpx.MockTransport`).

The two lock tests and the `wait` test were checked *against* the fix: with the fix
removed, they fail.

**End-to-end smoke test** (real server, real Sonnet 5.5, no Gmail):
1. "remind me to call mom tomorrow at 6pm" created an agent named after the thread (`Call mom
   reminder`), gave it a description, and stored the trigger.
2. "I am in New York. actually make it 7pm" **reused** that agent (`run_count` 2) and moved
   the trigger to 00:00 UTC.

That run is what surfaced the `wait` and logging bugs in §5. After the fix, the replay logged
zero errors.

### Evaluators (`evals/`)

**`prompt_growth.py`** drives the real prompt builders, unmodified, on both checkouts.

| Reuses of one agent | Before | After, no LLM (fallback) | After, compaction (peak) |
|---|---|---|---|
| 10 | 10,077 | 10,117 | 10,117 |
| 50 | 47,237 | 10,130 | 11,372 |
| 100 | 93,687 | 10,130 | 11,372 |
| 200 | 186,587 | 10,130 | 11,372 |

**`routing_offline.py`** uses 300 synthetic agents with ground truth and 240 follow-up
messages across five kinds:
- *explicit*: names the person and topic;
- *paraphrase*: names the person, with a topic synonym;
- *vague*: company plus a topic synonym, no person;
- *person_only*: "did Priya reply?", where the target is her most recent thread;
- *followup*: "yes send it", where the agent appears only in the transcript tail.

Each message is rendered through the real `prepare_message_with_history`, and the eval
checks whether the target agent is listed.

| K | recall | top-3 | tokens | explicit | paraphrase | person_only | followup | vague |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.858 | 0.804 | 323 | 1.0 | 0.729 | 1.0 | 1.0 | 0.562 |
| 10 | 0.912 | 0.804 | 604 | 1.0 | 0.958 | 1.0 | 1.0 | 0.604 |
| 15 | 0.942 | 0.804 | 882 | 1.0 | 1.0 | 1.0 | 1.0 | 0.708 |
| **20** | **0.975** | **0.804** | **1,161** | 1.0 | 1.0 | 1.0 | 1.0 | 0.875 |
| 30 | 1.0 | 0.804 | 1,721 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Baseline: recall 1.0 at 2,988 tokens, names only.

Ablation at K=20:

| Scorer | recall | top-3 |
|---|---|---|
| recency only | 0.217 | 0.042 |
| BM25 only | 0.963 | 0.671 |
| hybrid (ours) | 0.975 | 0.804 |

Recency barely helps recall, but it is what ranks Priya's *latest* thread first.

Dedupe, on hand-labeled name pairs:
- 20 same-thread pairs. The baseline creates 20 duplicates, 2 of which also silently share a
  log file. After the fix: 0 duplicates (2 auto-reused, 18 flagged).
- 16 different-thread pairs. 1 false flag ("Email to Alice" vs "Alice contract"), which is
  ambiguous from names alone and resolved by the model reading the descriptions.

**`routing_llm.py`** is the eval that can show routing actually got *better*. The offline
eval can't: the baseline lists everything, so its recall is trivially 100%. This script sends
the same queries to the real interaction-agent model under both the baseline prompt
(original system prompt from `main`, all names, original tools) and the new one (real tool
handlers, including rejection and search). It reports:
- correct reuse,
- needless new agent,
- wrong existing agent,
- no dispatch,
- prompt tokens per query.

#### Live results (`anthropic/claude-sonnet-5.5`, 300 agents, 30 queries per condition)

Getting the eval to run at all surfaced two production bugs (§5): the retired default model
and the missing `max_tokens`.

Each prompt version was judged on a workload it had never seen:

| Workload (seed) | Used for |
|---|---|
| dev (7) | Iterating v1 → v2 |
| held-out (11) | Validating v2; its traces motivated v3 |
| fresh (13) | Final check of v3. Generated after v3 was frozen; no prompt changes since |

"Before" is always the original prompt from `main`, with all names and the original tools.
p-values are one-sided Fisher exact tests.

**Final prompt (v3) on the fresh set (seed 13)**, `evals/results/routing_llm_sonnet55_seed13_v3.json`:

| | Before | After (v3) |
|---|---|---|
| Correct agent | 66.7% (20/30) | **93.3% (28/30)** (p = 0.011) |
| **Silent misroute** (wrong existing agent + needless new agent) | 33.3% (10/30) | **6.7% (2/30)** (p = 0.011) |
| … wrong existing agent | 13.3% | 3.3% |
| … needless new agent | 20.0% | 3.3% |
| Asked the user to clarify, no dispatch | 0% | 0% |
| Input tokens per query (all LLM calls) | 9,138 | 8,274 (−9%) |

| explicit | paraphrase | vague | person_only | followup |
|---|---|---|---|---|
| 100 → 100% | 67 → 83% | 17 → **83%** | 50 → **100%** | 100 → 100% |

The two remaining misses are both synonym gaps, the limitation predicted offline:
- "subscription" for an agent about a "renewal": the model searched, found nothing, and
  created a new agent.
- One wrong pick on a vague query.

**Silent misroutes are the metric that matters.** A misroute sends the user's request into
the wrong thread, or forks a thread and loses its context, and nobody notices. A clarifying
question costs one message. Typical "before" failures:
- "did Ivan reply?" went to `Email to Ivan`, the wrong one of his threads;
- "did the Retool bill go through?" created a new `Retool bill status` agent.

The original roster shows only names, so the model can't see that Ivan has eight threads, or
which agent owns Retool's invoice.

**How the prompt evolved.**

*v1 (dev set, `…_dev_v1.json`).* Accuracy rose from 63% to 74%, and wrong-existing routes
fell from 23% to 4%. But needless new agents went *up*, from 13% to 19%. The traces showed
why: the model got `similar_agents_exist` and still used `force_new=true` to create agents
like "Fatima reply check". It treated "check for replies" as a new task.

*v2.* One rule: status questions belong to the agent that owns the work, because that agent
has the thread ids.
- Dev (`…_dev_v2.json`): needless new agents 6.7%, wrong routes 0%, clarifying questions 27%.
- Held-out (`…_heldout.json`):

  | | Before | After (v2) |
  |---|---|---|
  | Correct | 50.0% | 73.3% (p = 0.055) |
  | Silent misroute | 46.7% | 10.0% (p = 0.002) |
  | Clarify | 3.3% | 16.7% |

  Every clarifying question was on a person-only query. A re-run probe
  (`…_person_only_probe.json`) showed the model usually already knew the answer: "You've got
  a lot of Ivans going… If it's the latest thread, that's Ivan at Vercel about the lunch."

*v3.* When one candidate clearly stands out (most recent, or best description match), act on
it and say which one you picked. Ask only when none stands out.
- Person-only queries on seed 13 (`…_seed13_person_only.json`): 6/6 correct, 0 clarifying
  questions.
- The full seed-13 run above: no regression on the other kinds, 0% clarifying questions.

**Caveats.**
- n = 30 per condition and a single run per condition, so expect a few points of LLM
  variance. Both headline differences on seed 13 are significant at p ≈ 0.01.
- The synthetic workload is dense: each person has ~10 concurrent threads, which makes
  person-only questions genuinely ambiguous. The "most recent thread" label is one reasonable
  answer, not the only one.
- The eval stops at the first accepted dispatch. It only sees v3 announcing its pick
  ("Checking your latest thread with Jonas at Retool") when that arrives in the same
  response.
- Token savings per query are modest at 300 agents, because the system prompt and tool
  schemas dominate. They grow with roster size (see `prompt_growth`).

### Honest caveats

- The offline workload is synthetic, and its queries share vocabulary with the descriptions.
  That favors lexical retrieval. The *vague* bucket (87.5%) is the realistic weak spot.
- Simulating `search_agents` with the raw user message is pessimistic. A real model would
  write a better query ("Stripe contract").
- Depth assumes 3 tool calls per run, each hitting the 500-char result truncation. The slope
  depends on that assumption. The fact that growth was unbounded and is now bounded does not.

## 4. Limitations and next steps

1. **Synonyms.** BM25 misses "agreement" ≈ "contract". The next step is an embedding scorer
   behind the same `score_agents` interface, with BM25 kept as a fallback.
2. **Descriptions before the first compaction** are heuristic: "Started for: <first
   instruction> | Latest: <last result>". The LLM-written description appears from the first
   compaction on.
3. **Agent identity is still the name string.** Normalizing it to a slug key everywhere
   (roster, logs, memory, triggers) closes the collision bugs. A stable generated id plus a
   display name would let agents be renamed or merged later.
4. **Clarify vs. act.** Prompt v3 acts on a clear front-runner and names it, instead of
   asking "which Ivan?". It made no clarifying calls on the synthetic workloads. The right
   threshold between acting and asking is a product decision to A/B test with real users.

## 5. Other problems found in OpenPoke

Fixed in this branch, each with tests:

| Problem | Where | Fix |
|---|---|---|
| **The app could not run at all.** Every model was hard-coded to `anthropic/claude-sonnet-4`, which OpenRouter has retired, so every request returned 404 | `config.py` | Models are configurable per role (`OPENPOKE_*_MODEL`). Defaults: Sonnet 5.5 for the agents, Haiku 5.5 for the summarizers and email classifier (about 20× cheaper for background work) |
| **No `max_tokens`.** OpenRouter reserved the full 64k output budget on every call, so output cost was uncapped and requests failed with 402 on low-balance accounts | `openrouter_client/client.py` | `max_tokens` is always sent (`OPENPOKE_MAX_OUTPUT_TOKENS`, default 4096) |
| **Unbounded parallel LLM calls, no retry.** Fan-out × tool loops × background summarizers could all fire at once, and a single 429 failed the whole task | `openrouter_client/client.py` | Per-loop semaphore (`OPENPOKE_LLM_MAX_CONCURRENCY`, default 8); retry 408/429/5xx/network errors with exponential backoff; 4xx errors are not retried |
| **Interaction-agent turns race.** Four entry points (user message, batch result, trigger result, email watcher) each read the transcript, call the LLM and append, concurrently. Two turns could answer from the same stale transcript | `interaction_agent/runtime.py` | Turns are serialized (FIFO `asyncio.Lock`) |
| **Same agent run concurrently interleaves its log**, corrupting its history | `execution_agent/batch_manager.py` | Per-agent run lock; the request is logged inside it, so each run's entries are contiguous |
| **Trigger-fired runs never logged their request**, so the agent's history showed actions with no instruction | `trigger_scheduler.py`, `batch_manager.py` | Same change: every run logs its request in the batch manager |
| **Each trigger created its own `ExecutionBatchManager`**, bypassing batching | `trigger_scheduler.py` | One shared manager |
| **Memory cross-talk from slug collisions.** Names differing only in case or punctuation were separate agents sharing one log file | `roster.py`, `log_store.py` | The registry key is the slug |
| **Triggers keyed by raw spelling.** A trigger saved under "email-to-alice" was invisible to `listTriggers`/`updateTrigger` for "Email to Alice" | `triggers/store.py` | Queries compare `agent_key(agent_name)` via a SQLite UDF; one identity everywhere (roster, logs, memory, triggers) |
| **Non-atomic, lossy roster writes.** The file was truncated (`open('w')`) *before* locking, and writers saved stale in-memory copies over each other | `roster.py` | Read-modify-write under a lock file; temp file + `os.replace` |
| **Leftover template text in the execution prompt** (`[TO BE FILLED IN BY USER …]`) sent on every call | `execution_agent/system_prompt.md` | Replaced with guidance on reading the memory summary |
| **The prompt references a tool that doesn't exist** (`reacttomessage`) | `interaction_agent/system_prompt.md` | Removed |
| **`wait` never ended the turn.** Found in a live end-to-end run: the model called `wait` repeatedly until the 8-iteration limit raised, so turns "failed" after burning extra LLM calls | `interaction_agent/runtime.py` | `wait` ends the turn |
| **Error logs never said why.** Failures are logged as `logger.error(msg, extra={"error": …})`, but the formatter dropped `extra` fields | `logging_config.py` | Formatter appends `extra` fields |

Found but not fixed here, with the proposed fix:

| Problem | Where | Proposed fix |
|---|---|---|
| One global batch: a slow agent holds back every other agent's result (up to the 90 s timeout), and new dispatches join an old batch | `execution_agent/batch_manager.py` | Batch per interaction turn (pass the batch id from the dispatching turn); deliver each result as it lands, with a short debounce so near-simultaneous results still arrive together |
| Tool results truncated to 500 chars in agent memory, which can cut off the very ids memory needs | `execution_agent/agent.py::record_tool_execution` | Extract ids (message, thread, draft) structurally before truncating |
