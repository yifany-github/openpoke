# Other problems found in OpenPoke

Separate from the agent-overload fix ([DESIGN.md](DESIGN.md)), ordered by severity.

Sources:
- reading the code;
- running the app end to end against a real model;
- two rounds of independent review of the diff.

"Mine" marks problems introduced by this fork's own changes and later caught and fixed.
Every fix has a test that fails without it.

## Fixed

| # | Severity | Problem | Where | Fix |
|---|---|---|---|---|
| 1 | **Blocker** | **The app could not run at all.** Every model was hard-coded to `anthropic/claude-sonnet-4`, which OpenRouter has retired, so every LLM request returned 404. | `config.py` | Models are configurable per role (`OPENPOKE_*_MODEL`). Defaults: Sonnet 5.5 for the agents; Haiku 5.5 for summarizers and the email classifier, about 20× cheaper for background work. |
| 2 | High | **No `max_tokens`.** OpenRouter reserved the full 64k output budget per call, so output cost was uncapped, and low-balance accounts got 402s. | `openrouter_client/client.py` | Always send `max_tokens` (`OPENPOKE_MAX_OUTPUT_TOKENS`, 4096). |
| 3 | High | **Interaction-agent turns race.** User messages, batch results, trigger results and the email watcher each read the transcript, call the LLM and append concurrently. Two turns could answer from the same stale transcript. | `interaction_agent/runtime.py` | Turns are serialized (FIFO lock) and time-limited (`OPENPOKE_TURN_TIMEOUT`, 180 s). A timed-out user turn still replies to the user. |
| 4 | High | **`wait` never ended a turn.** Found in a live run: the model called `wait` until the 8-iteration limit raised, so turns ended in errors after burning extra LLM calls. | `interaction_agent/runtime.py` | A response that is only `wait` ends the turn silently. Other calls in the same response are still seen by the model. |
| 5 | High | **Memory cross-talk from slug collisions.** "Email to Alice" and "email-to-alice" were two agents sharing one log file. | `roster.py`, `log_store.py` | One normalized key everywhere: roster, logs, memory and triggers. |
| 6 | Medium | **Triggers keyed by raw spelling.** A trigger saved under one spelling was invisible to `listTriggers`/`updateTrigger` under another. | `triggers/store.py` | Match on `agent_key(agent_name)` via a SQLite UDF. |
| 7 | Medium | **Same agent run concurrently interleaved its log**, and trigger-fired runs never logged their request. | `execution_agent/batch_manager.py` | Per-agent run lock (its wait counts toward the 90 s timeout). The request is logged inside the lock. |
| 8 | Medium | **Unbounded parallel LLM calls, no retry.** One 429 failed a whole task. | `openrouter_client/client.py` | Concurrency cap (`OPENPOKE_LLM_MAX_CONCURRENCY`, 8). Retry 408/429/5xx/network errors with backoff, honoring `Retry-After`. Configurable timeout. |
| 9 | Medium | **Lossy, crash-prone roster writes.** The file was truncated *before* locking, writers saved stale copies over each other, and a malformed `roster.json` crashed server startup. | `roster.py` | Read-modify-write under a lock file and atomic replace. Unreadable or wrong-shape files are moved to `roster.json.corrupt-<ts>`. |
| 10 | Medium | **Error logs never said why.** Failures are logged with `extra={"error": …}`, which the formatter dropped. | `logging_config.py` | The formatter appends `extra` fields. |
| 11 | Low | **Non-ASCII agent names were unsearchable.** The tokenizer only matched `[a-z0-9]`, so "给妈妈发邮件" had no tokens and vanished once it left the top-K. | `roster.py`, `agent_ranker.py` | Unicode tokens; CJK runs split into character bigrams. |
| 12 | Low | Leftover template text `[TO BE FILLED IN BY USER …]` was sent to the model on every execution call, and the prompt referenced a `reacttomessage` tool that doesn't exist. | system prompts | Removed. |
| 13 | Mine | A memory compaction finishing after "clear history" could resurrect old memory. | `agent_memory.py`, `routes/chat.py` | Re-check the log before saving. Clearing history runs on the event loop. |
| 14 | Mine | Sharing one batch manager between triggers and user requests made user results wait on trigger runs. | `trigger_scheduler.py` | **Reverted.** A separate batch per trigger (upstream's design) is right. |
| 15 | Mine | Atomic roster writes left the file at mode 0600. | `roster.py` | `chmod 0644`. |

## Found, not fixed (with proposed fix)

| Severity | Problem | Where | Proposed fix |
|---|---|---|---|
| **Security** | **An OpenRouter API key is in the public git history.** `b5695fa` (2025-09-17) committed a server access log whose URLs carry `api_key=sk-or-v1-…`. `f715ae0` deleted the file, but every clone and fork still contains it. | upstream history | Revoke the key. Optionally purge it with `git filter-repo`. Never accept credentials in query strings. The current server reads keys from `.env`, and upstream gitignored `*.log` in `6fa16ac`. |
| Medium | **One global batch.** A slow execution agent holds back every other agent's result dispatched in the same window, up to the 90 s timeout. | `execution_agent/batch_manager.py` | Batch per interaction turn (pass the turn's batch id). Deliver each result as it lands, with a short debounce. |
| Medium | **A timezone given in chat is ignored.** The timezone only comes from the browser. In a live run, "remind me in 1 minute" after "I'm in New York" was scheduled 5 hours late. | `services/timezone_store.py` | A `set_timezone` tool for the interaction agent. Stamp the current time *with its zone* into execution instructions. |
| Low | Tool results are truncated to 500 chars in agent memory, which can cut off the ids that memory needs. | `execution_agent/agent.py` | Extract ids (message, thread, draft) structurally before truncating. |
| Low | Execution logs are never trimmed and are re-parsed several times per run. The prompt is bounded; disk and CPU are not. | `log_store.py` | Archive runs already covered by the summary; cache parsed runs. |
| Ops | All locks are per process. Several server workers would break turn serialization and per-agent run locks. | runtimes | Single worker (as shipped), or move locks and queues to Redis or SQLite. |
