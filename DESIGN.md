# Fixing agent overload in OpenPoke

## TL;DR

**Problem.** OpenPoke's execution agents live forever, but nothing manages them. They have no
description, never retire, and their memory is never compacted. Overload shows up in two
directions:

- **Width.** Every agent ever created is listed by name only in every interaction-agent
  turn. With hundreds of agents, the router can't tell "Follow-ups with Dmitri" from the
  agent that actually owns Dmitri's quote. It sends work to the wrong thread or forks a new
  one.
  *Measured:* on an independent 300-agent workload, the original system routed **50%** of
  follow-up requests to the wrong agent or a needless new one.
- **Depth.** Every run of an execution agent re-sends that agent's **entire** history,
  about 930 tokens more per reuse, with no cap. After 200 reuses one call is ~187k tokens.

**Fix.**
1. A registry with metadata. The router sees only pinned agents plus the top-20 relevant
   ones, each with a one-line LLM-written description. It can `search_agents` for the rest,
   and near-duplicate names are bounced back to the model.
2. Bounded agent memory: the last 5 runs stay raw, older runs fold into a rolling summary.

**Results** (Sonnet 5.5; details and methodology in [EVALUATION.md](EVALUATION.md)):

| | Before | After |
|---|---|---|
| Correct agent chosen, independent workload (40 paired queries) | 20/40 (50%) | **35/40 (88%)**, paired McNemar p < 0.001 |
| Silent misroute (wrong agent, or a needless new agent) | 20/40 | **5/40** |
| Router prompt tokens per request | 9,749 | 8,149 |
| Execution prompt after 200 reuses of one agent | 186,587 tok | **≤ 14,717 tok** |
| Facts from compacted runs still answerable (8 questions, real summarizer) | 8/8 (full log) | **8/8** with half the tokens |

Ablations on the same 40 queries (EVALUATION.md §4.2) show where the gain comes from:

| Router sees | Correct | Prompt tokens per request |
|---|---|---|
| names only (original) | 20/40 | 9,749 |
| **all** 300 agents, with descriptions and recency | 35/40 | 30,697 |
| top-20, with heuristic descriptions | 34/40 | 8,406 |
| **top-20, with LLM descriptions (this fork)** | **35/40** | **8,149** |

- **The accuracy comes from giving the router metadata**: a description and when the agent
  was last used. Every variant with metadata beats names-only (p ≤ 0.001).
- **Top-K is what keeps it affordable at no accuracy cost.** It matches the full roster
  exactly (paired 1 vs 1) at 1/3.8 of the tokens, and that ratio grows with the number of
  agents.
- **Description quality mattered less than I expected.** Heuristic descriptions score 34/40
  against 35/40 (n.s.). An earlier dev-workload ablation suggested a large gap; the
  independent workload didn't confirm it.

The main caveats (EVALUATION.md §6):
- single run per condition;
- synthetic workloads, written by me but kept independent of prompt tuning;
- the "person-only" label (most recent thread) matches a rule in the prompt. Excluding
  those queries, the result is 20/32 → 28/32.

Other problems found in OpenPoke, including a retired default model that stopped the app
from running at all, are in [OTHER_FINDINGS.md](OTHER_FINDINGS.md).

---

## 0. Relation to the upstream author's diagnosis

The OpenPoke [explainer](https://www.shloked.com/writing/openpoke) names this problem
itself: "Execution Agent Overload". It notes that the full roster goes to the interaction
agent on every request, and that "after days of use, you might have hundreds of specialized
agents". It sketches four remedies. Here is how this fork maps to them:

| Author's suggestion | What this fork does | Evidence |
|---|---|---|
| Semantic search over agent descriptions | Agents get descriptions: an LLM writes one after the first run, and compaction refreshes it. The router sees the top-20 by BM25 over name + description, and the `search_agents` tool reaches the rest. **Lexical, not embedding-based, on purpose**: deterministic and testable, with its weak spot measured (synonyms account for most of the remaining failures). Embeddings are the next step. | Correct routing 50% → 88% on an independent workload |
| Archive dormant agents | A **soft archive**: agents that drop out of the top-K leave the prompt and come back when relevant. No separate state has to stay consistent with triggers, logs and memory. | Top-K matches the full roster's accuracy (35 vs 35 of 40) at 1/3.8 of the tokens |
| Cluster related agents | No clustering of existing agents. **Near-duplicates are stopped at creation**: a respelled name reuses the agent; a similar name is bounced back with the candidates and their descriptions. Merging existing clusters is future work and needs stable ids. | 20/20 same-thread name pairs caught (2 auto-reused, 18 sent back to the model) |
| Hot cache of recently active agents | A recency term in the ranking (3-day half-life), plus pinning of agents with active triggers or named in the current turn | "Person-only" requests: 0/8 → 7/8 |
| *(not in the post)* | **Depth**: each agent's own history grew without bound. It is now bounded by a rolling summary. | 186k → ≤ 14.7k tokens after 200 reuses; 8/8 old facts still answerable |

## 1. Where the overload is in the code

| Location | Problem |
|---|---|
| `services/execution/roster.py` | The roster is a JSON `list[str]`: append-only, no metadata, clearable only wholesale. |
| `agents/interaction_agent/agent.py::_render_active_agents` | Emits `<agent name="…"/>` for **every** agent on **every** turn. |
| `agents/interaction_agent/tools.py::send_message_to_agent` | New-vs-reuse is decided by **exact string match** on an LLM-invented name. |
| `services/execution/log_store.py::_slugify` | Logs are keyed by slug, so "Email to Alice" and "email-to-alice" are two agents sharing one log file. |
| `agents/execution_agent/agent.py::build_system_prompt_with_history` | Injects the full log every run (`conversation_limit` defaults to `None`). |
| `agents/interaction_agent/system_prompt.md` | Says to "split into as much concurrent calls as possible", which pushes width, and to "prefer existing agent… even if the name is unrelated", which pushes depth. |

Measured on the original code (`evals/prompt_growth.py`):
- **Width** costs about 11 tokens per agent. At 300 agents that's ~3.3k tokens: affordable,
  so **width is mainly a routing-quality problem**.
- **Depth** grows without bound, so **depth is the token problem**.

## 2. Design

### Width: registry, relevance-ranked roster, search, dedupe

- **Registry** (`roster.py`). Each agent has a `name`, a normalized `key` (the same slug used
  for logs, memory and triggers), a `description`, `created_at`, `last_used_at` and
  `run_count`. Writes are locked read-modify-write with atomic replace. The legacy list
  format migrates on load.
- **Descriptions.** The heuristic "Started for: <first instruction> | Latest: <result>" is
  written immediately. After the first run, one small-model call (`agent_memory.describe`,
  Haiku) replaces it with one line covering the people, organisation, email and topic.
  Compactions refresh it later. The LLM description's measured benefit over the heuristic is
  small (35 vs 34 of 40); it costs one cheap call per agent.
- **What the router sees** (`agent_ranker.py`):
  - pinned agents first: those with an active trigger, or named in the current turn;
  - then the top-K by `0.7·BM25(name×2 + description) + 0.3·recency` (3-day half-life).
    The query is the current message plus the conversation tail, so "yes, send it" still
    finds the agent that drafted it;
  - then `<more_agents count=N>` with a pointer to `search_agents`.

  There is no explicit archive state: agents that fall out of the top-K are effectively
  archived and return when relevant.
- **Dedupe.** A respelled name ("email-to-alice") reuses the existing agent automatically.
  A *similar* name ("Reply to Alice re lunch" vs "Email to Alice") is **rejected with the
  candidates and their descriptions**, so the model reuses one or passes `force_new=true`.
  Auto-merging is deliberately avoided: "Alice lunch" and "Alice contract" may be parallel
  work. A false flag costs one LLM round trip; a wrong merge corrupts memory.
- **Why BM25.** It is deterministic, dependency-free and testable, and its weak spot
  (synonyms, below) is measured. The scorer is a pure function, so embeddings can replace it
  locally.
- **Why K = 20.** It was the smallest K with ≥95% offline recall on the dev workload. On the
  independent workload, recall at K = 20 is 91%, which is part of why `search_agents`
  exists.

### Depth: bounded agent memory (`agent_memory.py`)

```
## Memory summary (runs 1..C)   <- rolling summary, hard cap 4,000 chars
## Recent runs (C+1..N)         <- raw, between 5 and 9 runs
```

- When 10 runs are unsummarized, a background task folds all but the newest 5 into the
  summary. The fold is incremental and runs once at a time per agent. This mirrors the
  conversation summarizer the interaction agent already had.
- The summary prompt requires identifiers verbatim: emails, thread, draft and trigger ids,
  RRULEs.
- **The bound never depends on the LLM.** If summarization fails, only the newest 10 raw runs
  are shown, with an "N older runs omitted" note.
- Rejected alternatives:
  - a pure sliding window, which drops the long-lived context that makes persistent agents
    worth having;
  - retrieval over one agent's log, which is more machinery than needed.

### Prompt changes

- Parallelize only independent work.
- Name agents after the thread of work they own.
- Status questions belong to the agent that owns the work.
- When one candidate clearly stands out, act on it and say which one; otherwise ask.

The examples added to the prompt use names that appear in neither evaluation workload. Upstream's own tool example ("Vercel Job Offer", "Email to Sharanjeet") overlaps the dev workload, which is one reason the headline numbers come from the independent workload.

## 3. Testing and evaluation (summary)

- **67 unit and integration tests** run with no network; any real API call fails the test.
  Every bug fix has a test that was checked to **fail without the fix**.
- **Evaluators** in `evals/`:
  - prompt growth, before and after;
  - offline routing recall (K sweep, ablations, dedupe) on a dev and an independent workload;
  - live routing with paired McNemar tests and ablations;
  - a live memory-compaction QA test.
- **Live end-to-end runs** against the real server:
  - a reminder created and then rescheduled, with the agent reused;
  - a trigger firing on schedule;
  - two simultaneous messages;
  - clearing history.
- **Two rounds of independent review** of the diff found 11 issues, all fixed. One was a
  change of mine that made things worse, and I reverted it.

Full details: [EVALUATION.md](EVALUATION.md).

## 4. Limitations and next steps

1. **Synonyms.** BM25 misses "pricing" ≈ "quote" and "walkthrough" ≈ "inspection". These
   are 3 of the 5 remaining live failures. Next: an embedding scorer behind `score_agents`,
   with BM25 kept as a fallback.
2. **Agents without descriptions** (legacy, or before the first-run description lands) are
   findable by name only. That explains most of the 9% offline recall gap.
3. **Act vs. ask.** The prompt now acts on a clear front-runner and names it. The right
   threshold is a product question for an A/B test with real users.
4. **Lifecycle.** Logs are never trimmed (the prompt is bounded; disk is not), and
   identity is still the name slug. A generated id would allow renames and merges.
5. **Single process.** Turn serialization and per-agent locks live in memory. Multiple
   workers would need shared locks.
