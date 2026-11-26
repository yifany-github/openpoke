"""Execution-agent registry.

Originally this was a flat JSON list of agent names that only ever grew. Every name was
injected into every interaction-agent turn, with no way to tell agents apart beyond the
name an LLM happened to pick. The registry now stores per-agent metadata so the
interaction agent can be shown a *relevant subset* of agents (see ``agent_ranker``):

- ``key``: normalized identity. "Email to Alice" and "email-to-alice" are the same agent
  (they already shared a log file on disk, see ``log_store._slugify``).
- ``description``: one line describing what the agent owns, refreshed from its memory.
- ``created_at`` / ``last_used_at`` / ``run_count``: recency and activity signals.

The on-disk format is versioned; the legacy ``["name", ...]`` format is migrated on load.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ...logging_config import logger

ROSTER_VERSION = 2

# Words that describe *how* an agent works rather than *what* it owns. Ignored when
# deciding whether two agent names refer to the same thread of work.
_GENERIC_NAME_TOKENS = {
    "a", "an", "and", "about", "agent", "draft", "email", "emails", "for", "forward",
    "from", "in", "of", "on", "re", "reply", "search", "send", "the", "to", "with",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def agent_key(name: str) -> str:
    """Normalized agent identity. Must match the log-store filename slug."""
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in (name or "").strip()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "agent"


def _fold(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def name_tokens(name: str) -> set[str]:
    """Content tokens of an agent name (generic verbs/stopwords removed, plurals folded)."""
    return {_fold(tok) for tok in _TOKEN_RE.findall((name or "").lower()) if tok not in _GENERIC_NAME_TOKENS}


SIMILARITY_THRESHOLD = 0.6


def name_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two agent names, based on content tokens.

    Jaccard overlap of content tokens, lifted into [0.6, 1] when one name's tokens are a
    subset of the other's ("Email to Alice" vs "Reply to Alice re lunch"): a more specific
    name for the same person/topic is always worth flagging, but closer matches rank higher.
    Non-subset names need >= 60% overlap, so "Vercel Job Offer" vs "Stripe Job Offer" (50%)
    is not flagged.
    """
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    if ta <= tb or tb <= ta:
        return SIMILARITY_THRESHOLD + (1 - SIMILARITY_THRESHOLD) * jaccard
    return jaccard


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentRecord:
    name: str
    key: str
    description: str = ""
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None
    run_count: int = 0

    @property
    def last_used_dt(self) -> Optional[datetime]:
        stamp = self.last_used_at or self.created_at
        if not stamp:
            return None
        try:
            parsed = datetime.fromisoformat(stamp)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class AgentRoster:
    """Persistent registry of execution agents with metadata."""

    def __init__(self, roster_path: Path):
        self._roster_path = roster_path
        self._agents: List[AgentRecord] = []
        self.load()

    # ------------------------------------------------------------------ persistence

    def _read(self) -> Tuple[List[AgentRecord], bool]:
        """Parse the file. Returns (records, needs_rewrite) without touching self."""
        if not self._roster_path.exists():
            return [], True
        try:
            data = json.loads(self._roster_path.read_text(encoding="utf-8") or "[]")
        except Exception as exc:
            # Don't let the next write silently replace an unreadable file with a
            # one-agent roster: move it aside so it can be recovered by hand.
            backup = self._roster_path.with_name(
                f"{self._roster_path.name}.corrupt-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
            )
            try:
                os.replace(self._roster_path, backup)
                logger.error(f"roster.json unreadable ({exc}); moved to {backup.name}")
            except OSError as move_exc:  # pragma: no cover - defensive
                logger.error(f"roster.json unreadable ({exc}); could not back it up: {move_exc}")
            return [], True

        if isinstance(data, list):  # legacy v1: ["name", ...]
            migrated: List[AgentRecord] = []
            seen: set[str] = set()
            for raw in data:
                name = str(raw)
                key = agent_key(name)
                if key in seen:  # v1 allowed slug-colliding duplicates
                    continue
                seen.add(key)
                migrated.append(AgentRecord(name=name, key=key))
            return migrated, True

        agents = data.get("agents", []) if isinstance(data, dict) else []
        records: List[AgentRecord] = []
        for item in agents:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            records.append(
                AgentRecord(
                    name=str(item["name"]),
                    key=str(item.get("key") or agent_key(item["name"])),
                    description=str(item.get("description") or ""),
                    created_at=item.get("created_at"),
                    last_used_at=item.get("last_used_at"),
                    run_count=int(item.get("run_count") or 0),
                )
            )
        return records, False

    def load(self) -> None:
        """Load the registry, migrating the legacy list-of-names format."""
        self._agents, needs_rewrite = self._read()
        if needs_rewrite:
            self.save()

    @contextmanager
    def _locked(self):
        """Exclusive cross-process lock on a sidecar lock file."""
        self._roster_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._roster_path.with_suffix(".lock"), "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write(self) -> None:
        """Atomic write (temp file + rename). Caller must hold the lock."""
        payload = {"version": ROSTER_VERSION, "agents": [asdict(a) for a in self._agents]}
        fd, tmp = tempfile.mkstemp(dir=self._roster_path.parent, prefix=".roster.")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; keep the permissions a plain open() gave
        os.replace(tmp, self._roster_path)

    def save(self) -> None:
        """Persist the in-memory registry as-is."""
        try:
            with self._locked():
                self._write()
        except Exception as exc:
            logger.warning(f"Failed to save roster.json: {exc}")

    @contextmanager
    def _mutation(self):
        """Read-modify-write under the lock, starting from what is on disk.

        Another process (or another roster instance) may have written since we last
        loaded; mutating a stale in-memory copy and saving it would drop their changes.
        """
        with self._locked():
            self._agents, _ = self._read()
            yield
            self._write()

    # ------------------------------------------------------------------ queries

    def get_agents(self) -> list[str]:
        """Canonical names of all agents (insertion order)."""
        return [a.name for a in self._agents]

    def records(self) -> List[AgentRecord]:
        return list(self._agents)

    def get(self, name: str) -> Optional[AgentRecord]:
        """Look an agent up by any spelling that normalizes to the same key."""
        key = agent_key(name)
        for record in self._agents:
            if record.key == key:
                return record
        return None

    def find_similar(self, name: str, threshold: float = SIMILARITY_THRESHOLD, limit: int = 5) -> List[AgentRecord]:
        """Existing agents whose names look like the same thread of work as ``name``."""
        scored = [
            (name_similarity(name, record.name), record)
            for record in self._agents
            if record.key != agent_key(name)
        ]
        scored = [item for item in scored if item[0] >= threshold]
        scored.sort(key=lambda item: (item[0], item[1].last_used_dt or datetime.min.replace(tzinfo=timezone.utc)),
                    reverse=True)
        return [record for _, record in scored[:limit]]

    # ------------------------------------------------------------------ mutations

    def add_agent(self, agent_name: str, *, now: Optional[datetime] = None) -> AgentRecord:
        """Register an agent, returning the existing record if the key is already known."""
        with self._mutation():
            existing = self.get(agent_name)
            if existing is not None:
                return existing
            stamp = _iso(now or _now())
            record = AgentRecord(name=agent_name, key=agent_key(agent_name), created_at=stamp, last_used_at=stamp)
            self._agents.append(record)
            return record

    def touch(self, agent_name: str, *, now: Optional[datetime] = None) -> None:
        """Record that an agent was dispatched work."""
        with self._mutation():
            record = self.get(agent_name)
            if record is not None:
                record.last_used_at = _iso(now or _now())
                record.run_count += 1

    def update_description(self, agent_name: str, description: str) -> None:
        cleaned = " ".join((description or "").split())[:280]
        if not cleaned:
            return
        with self._mutation():
            record = self.get(agent_name)
            if record is not None:
                record.description = cleaned

    def bulk_load(self, records: Iterable[AgentRecord]) -> None:
        """Replace all records (used by tests and evals)."""
        self._agents = list(records)
        self.save()

    def clear(self) -> None:
        """Clear the agent roster."""
        self._agents = []
        try:
            if self._roster_path.exists():
                self._roster_path.unlink()
            logger.info("Cleared agent roster")
        except Exception as exc:
            logger.warning(f"Failed to clear roster.json: {exc}")


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_agent_roster = AgentRoster(_ROSTER_PATH)


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
