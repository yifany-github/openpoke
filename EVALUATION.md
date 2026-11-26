# Evaluation

How the agent-overload fix was tested, and how I checked that it actually improved things.
The summary is in [DESIGN.md](DESIGN.md). Raw outputs are in `evals/results/`.

## 1. Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r server/requirements.txt pytest tiktoken
.venv/bin/python -m pytest                                  # 67 tests, network blocked
.venv/bin/python -m evals.prompt_growth                     # §2
.venv/bin/python -m evals.routing_offline [--vocab fresh]   # §3
# live (needs OPENROUTER_API_KEY), §4–5:
.venv/bin/python -m evals.routing_llm --vocab fresh --n 40 --yes --delay 2 \
    --conditions baseline,after,full_roster,after_heuristic_desc
.venv/bin/python -m evals.analyze_live evals/results/routing_llm_sonnet55_fresh.json
.venv/bin/python -m evals.memory_compaction_llm --yes
```

The "before" numbers for prompt growth come from running the same script against a checkout
of the original commit (`git worktree add ../openpoke-main 5b5f635`).

Token counts from `prompt_growth` and `routing_offline` use `tiktoken` cl100k as a proxy
(±15%). Live token counts are the API's own `prompt_tokens`.

## 2. Prompt growth (deterministic, no LLM)

Drives the real prompt builders on both checkouts.

| Agents in roster | Before: roster block | After: roster block (top-20 + descriptions) |
|---|---|---|
| 10 | 95 | 435 |
| 300 | 3,263 | 928 |
| 1,000 | 11,230 | 942 |

| Reuses of one agent | Before | After, no LLM (fallback) | After, with compaction (peak) |
|---|---|---|---|
| 10 | 10,077 | 13,462 | 13,462 |
| 50 | 47,237 | 13,475 | 14,717 |
| 200 | 186,587 | 13,475 | 14,717 |

The "after" logs are about 35% larger per run than "before". This simulation's tool results
are full of ids, and the fork now keeps the ids that used to be cut off at the 500-char
truncation (OTHER_FINDINGS #17). The prompt also states the current time and timezone (#16).
Both cost a fixed amount per run, and the bound still holds.

Two things to read from these tables:
- **The depth bound holds by construction.** A real summarizer passes the stub summary in
  this script; §5 tests the summary's quality.
- **Descriptions cost tokens at small roster sizes.** The roster block is larger below about
  40 agents.

## 3. Offline routing (deterministic)

**Question:** with most agents hidden, is the agent the user means still in the router's
prompt?

The baseline lists everything, so its recall is 100% by construction. This eval can only
show that hiding agents loses little. Whether routing improves needs §4.

**Workloads.** Both have 300 agents and 240 follow-up messages across five kinds:
- *explicit*: person and topic;
- *paraphrase*: person and a topic synonym;
- *vague*: organisation and a synonym, no person;
- *person_only*: label = that person's most recent thread;
- *followup*: "send it", with the agent only in the transcript tail.

| | dev (`routing_offline.py`) | fresh (`workload_fresh.py`) |
|---|---|---|
| Used while tuning K, weights and prompt | yes | **no** |
| Shares names, organisations, topics or templates with the other workload or the prompt | – | **no** |
| Descriptions | template from the scenario fields | **written by the production `describe()` (Haiku)** from a first-run log; **20% left blank** |
| People ↔ organisations | random per thread | fixed per person |

Recall of the target agent in the rendered roster (pinned + top-K):

| K | dev recall | fresh recall | fresh tokens |
|---|---|---|---|
| 5 | 0.858 | 0.767 | 285 |
| 10 | 0.912 | 0.867 | 547 |
| **20** | **0.975** | **0.912** | **1,048** |
| 30 | 1.0 | 0.946 | 1,574 |

The fresh workload is the honest number: **about 9% of targets are not in the top-20**,
mostly agents with blank or organisation-less descriptions on *vague* queries (0.67). The
model then has to use `search_agents`.

Ablation of the scorer at K=20 (fresh workload):

| Scorer | recall | target ranked in top-3 |
|---|---|---|
| recency only | 0.17 | 0.02 |
| BM25 only | 0.92 | 0.59 |
| BM25 + recency | 0.91 | **0.67** |

**Duplicate-name detection** used hand-labelled pairs:
- 20 same-thread pairs: 2 are auto-reused (respellings) and 18 are flagged for the model to
  decide. The baseline treats any string difference as a new agent, so its "20/20
  duplicates" holds by definition.
- 16 different-thread pairs: 1 was falsely flagged ("Email to Alice" vs "Alice contract"),
  which costs one round trip.

## 4. Live routing (Sonnet 5.5)

`evals/routing_llm.py` gives the interaction agent each query and runs its tool loop until
it dispatches, for up to 4 LLM calls. Nothing is executed.

**Conditions**, all over the same queries:

| Condition | What the model sees |
|---|---|
| `baseline` | the original system prompt from `main`, all agent names, no descriptions, the original tools |
| `after` | this branch: top-20 roster with descriptions and recency, `search_agents`, name resolution and rejection |
| `full_roster` | like `after`, but **all** agents listed, with no top-K. Isolates what top-K adds |
| `after_heuristic_desc` | like `after`, but descriptions are the pre-LLM heuristic, with no organisation or email. Isolates description quality |

**Outcomes:**
- *correct*;
- *silent misroute*: a wrong existing agent, or a new agent when one existed;
- *clarify*: the model asked the user and dispatched nothing;
- *API error*: excluded pairwise.

Comparisons use an **exact two-sided McNemar** test on paired queries.

### 4.1 Headline: independent workload (fresh, n = 40)

`evals/results/routing_llm_sonnet55_fresh.json`:

| | baseline | after |
|---|---|---|
| Correct | 20/40 (50%) | **35/40 (88%)** |
| Silent misroute | 20/40 | 5/40 |
| Clarify | 0 | 0 |
| Prompt tokens per query | 9,749 | 8,149 |

Paired, 16 queries are right only after the fix and 1 only before it (p = 0.0003).

| Kind | explicit | paraphrase | vague | person_only | followup |
|---|---|---|---|---|---|
| baseline | 5/8 | 2/8 | 5/8 | 0/8 | 8/8 |
| after | 8/8 | 5/8 | 7/8 | 7/8 | 8/8 |

**Why the baseline fails.** Most of its errors go to generic names such as "Follow-ups with
Dmitri", "Katya (Maple Insurance)" or "Email to …". Names alone can't say which of a
person's threads owns the quote. It also never sees recency.

**Remaining failures after the fix:**
- 3 synonym misses ("pricing" for an agent about a "quote", "walkthrough" for "inspection",
  "plan" for "membership"); the model created a new agent;
- 1 vague wrong pick;
- 1 person-only wrong pick.

**Excluding the person-only kind**, whose label matches a prompt rule (see §6), the result is
20/32 → 28/32.

### 4.2 Ablations (fresh workload, same 40 queries)

`…_fresh.json` (baseline, after) and `…_fresh_ablations.json`:

| Condition | correct | silent misroute | prompt tokens / query | by kind: explicit / paraphrase / vague / person_only / followup |
|---|---|---|---|---|
| baseline | 20/40 | 20 | 9,749 | 5 / 2 / 5 / 0 / 8 (of 8) |
| full_roster | 35/40 | 5 | **30,697** | 8 / 5 / 7 / 7 / 8 |
| after_heuristic_desc | 34/40 | 6 | 8,406 | 8 / 5 / 5 / 8 / 8 |
| after | **35/40** | 5 | **8,149** | 8 / 5 / 7 / 7 / 8 |

Paired (exact McNemar, two-sided):

| A vs B | only A right | only B right | p |
|---|---|---|---|
| after vs baseline | 16 | 1 | 0.0003 |
| full_roster vs baseline | 17 | 2 | 0.001 |
| after_heuristic_desc vs baseline | 16 | 2 | 0.001 |
| after vs full_roster | 1 | 1 | 1.0 |
| after vs after_heuristic_desc | 3 | 2 | 1.0 |

**Reading:**
- **What fixes routing is the metadata**: descriptions plus recency. Every condition that has
  it beats names-only by about 15 queries.
- **Top-K costs nothing in accuracy.** It is identical to listing everything, at 1/3.8 the
  tokens.
- **The LLM description adds little over the heuristic one** on this workload (35 vs 34).

The remaining failures are mostly synonym misses: "plan" for "membership", "case" for
"claim", "pricing" for "quote", "walkthrough" for "inspection". There are also a few vague and
person-only wrong picks. Only two queries fail in every metadata condition ("remind Hiroshi
about the plan" and "where did we land with Brightline Dental on the down payment"). The
synonym misses are the case for an embedding scorer.

#### Earlier ablation (dev workload, seed 13, n = 30, prompt before de-leaking)

`…_seed13_v3.json` and `…_seed13_ablations.json`:

| Condition | answered | correct | prompt tokens / query |
|---|---|---|---|
| baseline | 30 | 20 (67%) | 9,138 |
| full_roster | 22* | 19 (86%) | **37,258** |
| after_heuristic_desc | 30 | 23 (77%) | 9,210 |
| after | 30 | 28 (93%) | 8,274 |

\*8 requests failed because the API key hit its spending cap. That is itself what listing
every agent costs.

Paired results:
- **`after` vs `full_roster`:** 21 of 22 pairs agree; 1 query is right only after the fix
  (p = 1.0).
- **`after` vs `after_heuristic_desc`:** 5 queries are right only for `after`, 0 the other
  way (p = 0.06). All 5 are vague queries (5/6 → 1/6), which need the organisation in the
  description.

This run had pointed the same way on top-K. It suggested a large description-quality gap,
which the independent workload did not reproduce. The gap came from the dev workload's
organisation-only queries, where the organisation appeared only in the rich descriptions.
It is kept here as history, and as a reminder of why the headline comes from the
independent workload.

### 4.3 How the prompt evolved, and what was wrong along the way

- **v1 (dev, seed 7).** Wrong-agent routes fell from 23% to 4%, but needless new agents
  rose from 13% to 19%. The model got `similar_agents_exist` and still used `force_new` to
  create "Fatima reply check".
- **v2.** Added "status questions belong to the agent that owns the work". On the held-out
  sample (seed 11), misroutes went 14/30 → 3/30, but 17% of answers were clarifying
  questions. A probe showed the model usually knew the latest thread and asked anyway.
- **v3.** Added "act on a clear front-runner and say which one".
- **What was wrong with this process**, flagged by an independent methodology review:
  - **the prompt examples I added quoted names from the dev workload** (Fatima, Ivan,
    Stripe, Vercel);
  - seeds 7, 11 and 13 are samples of the *same* templates, so they are not truly held out;
  - I had used an unpaired test (Fisher) and reported the same comparison twice.

  Fixed: the examples now use names absent from every workload, the fresh workload (§3) was
  built independently, and comparisons are paired. **§4.1 is the result to trust; the
  dev-seed results are history.**

## 5. Memory compaction quality (live)

`evals/memory_compaction_llm.py`:
- One agent accumulates 20 runs.
- Facts are planted in early runs: thread ids, a cc'd email, an assistant's email, a
  preference, a rescheduled date, an invoice number, a trigger's status. Routine runs bury
  them.
- Compaction runs for real (Haiku) three times, so everything before run 16 exists only in
  the summary.
- Sonnet then answers 8 questions whose answers are in those runs.

| History given to the agent | correct | prompt tokens |
|---|---|---|
| Full log (original behaviour) | 8/8 | 3,682 |
| Bounded: summary + last 5 runs | **8/8** | **1,847** |

This was run before the prompt gained its ~35-token current-time line. The tool results in
this scenario are under 500 chars, so id-preserving truncation doesn't apply.

The summary kept every identifier and marked unconfirmed items as such ("2026-12-04 at Nopa —
proposed, not yet confirmed"). This is one scenario with 8 questions. It shows the mechanism
works, not a failure rate.

## 6. Caveats

- **One run per condition.** Temperature is left at the provider default. The baseline
  swings 50–67% across workloads.
- **Synthetic workloads.** Both were written by me. The fresh one is independent of the
  tuning, but it isn't real usage. A sample of real conversations is the missing piece.
- **The person-only label is "the most recent thread"**, and v3's prompt rule is to act on
  the clear front-runner. Part of that kind's gain is the prompt agreeing with the label.
  §4.1 also reports the result without it.
- **Not measured:** latency; the cost of the extra small-model calls (one description per
  new agent, one summary per 5 runs); real Gmail flows (no Composio account was used).
- **Single process.** The locks are in-process.

## 7. Other verification

- **Tests.** 67, with network calls blocked. Every bug fix has a test that was checked to fail
  without the fix. The suite passes on Python 3.10 and 3.11 from a fresh clone.
- **Live end-to-end runs** against the real server (Sonnet 5.5, no Gmail):
  - a reminder created, then rescheduled; the same agent was reused and the trigger moved;
  - a trigger firing on schedule and its result delivered;
  - two messages sent at the same instant, both answered;
  - clearing history, after which logs, memory, roster and triggers were empty.

  These runs surfaced the `wait` loop, the swallowed error logs and the timezone issue
  (see [OTHER_FINDINGS.md](OTHER_FINDINGS.md)).
- **Reproducibility.** The offline evals reproduce byte-for-byte from a fresh clone. Every
  number in DESIGN.md was cross-checked against `evals/results/`.
- **Independent review.** Two rounds of review saw only the diff and found 11 issues, all
  fixed with tests. One of them was a change of mine that hurt (trigger runs sharing the
  user's batch), which I reverted. A third review audited this methodology and led to §4.3's
  fixes.
