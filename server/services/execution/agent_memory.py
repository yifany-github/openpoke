"""Bounded long-term memory for persistent execution agents.

Before: every run of an execution agent injected the agent's *entire* log into its system
prompt. Each reuse added ~1k tokens with no cap, so heavily used agents got slower, more
expensive and eventually could not fit in the context window at all.

Now an agent's history is rendered as:

    <memory summary covering runs 1..C>     (rolling LLM summary, size-capped)
    <raw log of runs C+1..N>                (between WINDOW and WINDOW+BATCH-1 runs)

When the number of raw (unsummarized) runs reaches WINDOW + BATCH, a background
compaction folds the oldest ones into the summary, keeping the newest WINDOW runs raw.
This mirrors the conversation summarizer the interaction agent already has.

If compaction fails or no LLM is available, the prompt is still bounded: only the newest
WINDOW + BATCH raw runs are rendered, with a note that older runs were omitted.

The summary's first line is a one-sentence DESCRIPTION, which also feeds the agent
registry, so the interaction agent can tell agents apart when routing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from typing import Awaitable, Callable, List, Optional, Tuple

from ...config import get_settings
from ...logging_config import logger
from .log_store import ExecutionAgentLogStore, get_execution_agent_logs
from .roster import agent_key, get_agent_roster

Entry = Tuple[str, str, str]  # (tag, timestamp, payload)
Summarizer = Callable[[str, str], Awaitable[str]]  # (system_prompt, user_content) -> text

MAX_SUMMARY_CHARS = 4000
_DESCRIPTION_PREFIX = "DESCRIPTION:"


@dataclass
class MemoryState:
    summary: str = ""
    description: str = ""
    covered_runs: int = 0  # runs 1..covered_runs are represented by ``summary``


class AgentMemoryStore:
    """Persists one small JSON memory file per agent next to its log."""

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir

    def _path(self, agent_name: str) -> Path:
        return self._base_dir / f"{agent_key(agent_name)}.memory.json"

    def load(self, agent_name: str) -> MemoryState:
        path = self._path(agent_name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MemoryState(
                summary=str(data.get("summary") or ""),
                description=str(data.get("description") or ""),
                covered_runs=int(data.get("covered_runs") or 0),
            )
        except FileNotFoundError:
            return MemoryState()
        except Exception as exc:
            logger.warning(f"Failed to read agent memory for {agent_name}: {exc}")
            return MemoryState()

    def save(self, agent_name: str, state: MemoryState) -> None:
        path = self._path(agent_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            logger.error(f"Failed to write agent memory for {agent_name}: {exc}")

    def clear_all(self) -> None:
        for path in self._base_dir.glob("*.memory.json"):
            try:
                path.unlink()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to delete {path}: {exc}")


# ---------------------------------------------------------------------------- helpers

def split_runs(entries: List[Entry]) -> List[List[Entry]]:
    """Group log entries into runs; each run starts at an ``agent_request`` entry."""
    runs: List[List[Entry]] = []
    for entry in entries:
        if entry[0] == "agent_request" or not runs:
            runs.append([])
        runs[-1].append(entry)
    return runs


def _settings_window() -> Tuple[int, int]:
    settings = get_settings()
    window = max(settings.execution_history_window_runs, 1)
    batch = max(settings.execution_history_compaction_batch, 1)
    return window, batch


def render_history(agent_name: str) -> str:
    """Bounded history block for the agent's system prompt ('' if no history)."""
    log_store = get_execution_agent_logs()
    state = get_agent_memory().load(agent_name)
    runs = split_runs(list(log_store.iter_entries(agent_name)))
    window, batch = _settings_window()

    covered = min(state.covered_runs, len(runs))
    raw_runs = runs[covered:]
    omitted = 0
    cap = window + batch
    if len(raw_runs) > cap:  # compaction is behind or failing: stay bounded anyway
        omitted = len(raw_runs) - cap
        raw_runs = raw_runs[-cap:]

    parts: List[str] = []
    if state.summary:
        parts.append(f"## Memory summary (runs 1-{covered})\n{state.summary}")
    if omitted:
        parts.append(f"<!-- {omitted} older run(s) omitted: not yet summarized -->")
    if raw_runs:
        first = len(runs) - len(raw_runs) + 1
        body = ExecutionAgentLogStore.render_entries([e for run in raw_runs for e in run])
        parts.append(f"## Recent runs ({first}-{len(runs)})\n{body}")
    return "\n\n".join(parts)


def needs_compaction(agent_name: str) -> bool:
    state = get_agent_memory().load(agent_name)
    total = len(split_runs(list(get_execution_agent_logs().iter_entries(agent_name))))
    window, batch = _settings_window()
    return total - state.covered_runs >= window + batch


# ---------------------------------------------------------------------------- compaction

_SUMMARY_SYSTEM_PROMPT = dedent(
    """
    You maintain the long-term memory of ONE execution agent inside a personal assistant.
    The agent will read your memory before every future task, instead of its old logs.

    Output exactly this format:
    DESCRIPTION: <one line, max 25 words: what this agent owns - people, threads, recurring duties>
    MEMORY:
    - <bullet>

    Rules:
    1. Keep every identifier verbatim: email addresses, thread/message/draft ids, trigger ids,
       schedules (RRULEs), dates, amounts, links.
    2. Keep people, decisions, preferences, and the status of anything pending.
    3. Merge the existing memory with the new runs; drop chatter and completed, obsolete items.
    4. At most 250 words. Do not invent facts.
    """
).strip()


def build_compaction_prompt(agent_name: str, previous_summary: str, runs: List[List[Entry]]) -> str:
    logs = ExecutionAgentLogStore.render_entries([e for run in runs for e in run])
    return (
        f"Agent name: {agent_name}\n\n"
        f"Existing memory:\n{previous_summary.strip() or 'None'}\n\n"
        f"New runs to merge:\n{logs}"
    )


def parse_summary(text: str) -> Tuple[str, str]:
    """Split LLM output into (description, summary)."""
    text = (text or "").strip()
    description = ""
    lines = text.splitlines()
    if lines and lines[0].upper().startswith(_DESCRIPTION_PREFIX):
        description = lines[0][len(_DESCRIPTION_PREFIX):].strip()
    return description, text[:MAX_SUMMARY_CHARS]


async def _openrouter_summarizer(system_prompt: str, content: str) -> str:
    from ...openrouter_client import request_chat_completion

    settings = get_settings()
    response = await request_chat_completion(
        model=settings.summarizer_model,
        messages=[{"role": "user", "content": content}],
        system=system_prompt,
        api_key=settings.openrouter_api_key,
    )
    choices = response.get("choices") or []
    content_out = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not content_out:
        raise RuntimeError("summarizer returned no content")
    return content_out


async def compact(agent_name: str, summarizer: Optional[Summarizer] = None) -> bool:
    """Fold all but the newest WINDOW runs into the rolling summary. Returns True if it ran."""
    memory = get_agent_memory()
    state = memory.load(agent_name)
    runs = split_runs(list(get_execution_agent_logs().iter_entries(agent_name)))
    window, batch = _settings_window()
    if len(runs) - state.covered_runs < window + batch:
        return False

    target = len(runs) - window
    to_fold = runs[state.covered_runs:target]
    fingerprint = runs[0][0]  # first entry of the log: (tag, timestamp, payload)
    prompt = build_compaction_prompt(agent_name, state.summary, to_fold)
    text = await (summarizer or _openrouter_summarizer)(_SUMMARY_SYSTEM_PROMPT, prompt)
    description, summary = parse_summary(text)

    # The LLM call can take minutes. If history was cleared (DELETE /chat/history) or the
    # log otherwise replaced meanwhile, saving would resurrect the old summary and mark
    # new runs as already covered. Only save if the log still starts the same way.
    current = split_runs(list(get_execution_agent_logs().iter_entries(agent_name)))
    if not current or current[0][0] != fingerprint or len(current) < target:
        logger.warning(f"[{agent_name}] log changed during compaction; discarding stale summary")
        return False

    memory.save(
        agent_name,
        MemoryState(summary=summary, description=description or state.description, covered_runs=target),
    )
    if description:
        get_agent_roster().update_description(agent_name, description)
    logger.info(f"[{agent_name}] memory compacted through run {target}")
    return True


_in_flight: set[str] = set()
_tasks: set[asyncio.Task] = set()  # strong refs so pending compactions aren't garbage-collected


def schedule_compaction(agent_name: str, summarizer: Optional[Summarizer] = None) -> Optional[asyncio.Task]:
    """Start a background compaction if one is due and none is running for this agent."""
    key = agent_key(agent_name)
    if key in _in_flight or not needs_compaction(agent_name):
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    async def _run() -> None:
        try:
            await compact(agent_name, summarizer)
        except Exception as exc:
            # Bounded fallback in render_history keeps prompts small even if this keeps failing.
            logger.warning(f"[{agent_name}] memory compaction failed: {exc}")
        finally:
            _in_flight.discard(key)

    _in_flight.add(key)
    task = loop.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def refresh_description(agent_name: str, latest_response: str) -> None:
    """Cheap description refresh after every run (no LLM call).

    Prefers the LLM-written description from the last compaction; otherwise keeps the
    original purpose and appends the latest outcome.
    """
    roster = get_agent_roster()
    record = roster.get(agent_name)
    if record is None:
        return
    state = get_agent_memory().load(agent_name)
    if state.description:
        roster.update_description(agent_name, state.description)
        return
    purpose = record.description.split(" | Latest:")[0]
    latest = " ".join((latest_response or "").split())[:140]
    roster.update_description(agent_name, f"{purpose} | Latest: {latest}" if latest else purpose)


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_agent_memory = AgentMemoryStore(_DATA_DIR / "execution_agents")


def get_agent_memory() -> AgentMemoryStore:
    return _agent_memory


__all__ = [
    "AgentMemoryStore",
    "MemoryState",
    "compact",
    "get_agent_memory",
    "needs_compaction",
    "parse_summary",
    "refresh_description",
    "render_history",
    "schedule_compaction",
    "split_runs",
]
