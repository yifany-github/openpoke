"""Pick which execution agents the interaction agent should see on a given turn.

The interaction agent used to receive every agent name ever created. Instead we show:

1. **Pinned** agents, always: agents with active triggers (they will wake up on their own
   and their results must be routable) and agents explicitly named in the current turn
   (e.g. the agent whose result is being reported).
2. The **top-K** remaining agents by a blend of lexical relevance (BM25 over name +
   description, queried with the current turn and the recent conversation tail) and
   recency (exponential decay on ``last_used_at``).

Everything else is summarized as a count, and the interaction agent can reach it through
the ``search_agents`` tool. Agents that drift out of the top-K are effectively archived,
without a separate archive state to keep consistent.

Pure functions, no I/O: easy to unit-test and to drive from offline evals.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Set

from .roster import AgentRecord, unicode_tokens

_STOPWORDS = {
    "a", "an", "and", "are", "about", "at", "be", "can", "did", "do", "for", "from", "get",
    "has", "have", "he", "her", "him", "his", "i", "if", "in", "is", "it", "me", "my", "of",
    "on", "or", "re", "she", "so", "that", "the", "their", "them", "they", "this", "to",
    "up", "was", "we", "what", "when", "with", "you", "your",
}

DEFAULT_TOP_K = 20
DEFAULT_MAX_PINNED = 10
RELEVANCE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3
RECENCY_HALF_LIFE_DAYS = 3.0


def tokenize(text: str) -> List[str]:
    tokens = []
    for tok in unicode_tokens(text):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural folding: "emails" -> "email"
        tokens.append(tok)
    return tokens


def _document(record: AgentRecord) -> List[str]:
    # Name tokens are counted twice: names are short and deliberately chosen.
    return tokenize(record.name) * 2 + tokenize(record.description)


def bm25_scores(records: Sequence[AgentRecord], query: str, k1: float = 1.2, b: float = 0.75) -> List[float]:
    docs = [_document(r) for r in records]
    q_terms = set(tokenize(query))
    if not docs or not q_terms:
        return [0.0] * len(docs)
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df = Counter(term for d in docs for term in set(d))
    scores: List[float] = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(doc) / avg_len))
        scores.append(score)
    return scores


def recency_scores(records: Sequence[AgentRecord], now: datetime) -> List[float]:
    out = []
    for record in records:
        last = record.last_used_dt
        if last is None:
            out.append(0.0)
            continue
        age_days = max((now - last).total_seconds() / 86400.0, 0.0)
        out.append(0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))
    return out


def score_agents(records: Sequence[AgentRecord], query: str, now: Optional[datetime] = None) -> List[float]:
    now = now or datetime.now(timezone.utc)
    rel = bm25_scores(records, query)
    max_rel = max(rel, default=0.0)
    rel_norm = [r / max_rel for r in rel] if max_rel > 0 else rel
    rec = recency_scores(records, now)
    return [RELEVANCE_WEIGHT * a + RECENCY_WEIGHT * b for a, b in zip(rel_norm, rec)]


@dataclass
class AgentSelection:
    shown: List[AgentRecord]
    pinned_keys: Set[str]
    hidden_count: int
    total: int


def select_agents(
    records: Sequence[AgentRecord],
    query: str,
    *,
    k: int = DEFAULT_TOP_K,
    pinned_keys: Iterable[str] = (),
    max_pinned: int = DEFAULT_MAX_PINNED,
    now: Optional[datetime] = None,
) -> AgentSelection:
    """Return pinned agents plus the top-``k`` others, best first.

    Pinned agents are capped at ``max_pinned`` (best-scoring first) so that, e.g., dozens
    of recurring reminders cannot reintroduce the overload; overflow competes normally.
    """
    pinned = set(pinned_keys)
    scores = score_agents(records, query, now)
    ranked = sorted(zip(scores, range(len(records))), key=lambda item: item[0], reverse=True)

    pinned_records = [records[i] for _, i in ranked if records[i].key in pinned][: max(max_pinned, 0)]
    pinned_shown = {r.key for r in pinned_records}
    others = [records[i] for _, i in ranked if records[i].key not in pinned_shown]
    shown = pinned_records + others[: max(k, 0)]
    return AgentSelection(
        shown=shown,
        pinned_keys=pinned_shown,
        hidden_count=len(records) - len(shown),
        total=len(records),
    )


def search_agents(
    records: Sequence[AgentRecord], query: str, *, limit: int = 10, now: Optional[datetime] = None
) -> List[AgentRecord]:
    """Relevance-first search used by the ``search_agents`` tool."""
    rel = bm25_scores(records, query)
    scores = score_agents(records, query, now)
    ranked = sorted(range(len(records)), key=lambda i: (rel[i] > 0, scores[i]), reverse=True)
    return [records[i] for i in ranked[:limit] if rel[i] > 0]


def mentioned_agent_keys(records: Sequence[AgentRecord], text: str) -> Set[str]:
    """Keys of agents whose exact name appears in ``text`` as whole words (e.g. a batch
    result line "[SUCCESS] Email to Bob: ..."). "Bob" must not match inside "Bobby"."""
    lowered = (text or "").lower()
    keys = set()
    for r in records:
        if r.name and re.search(rf"(?<![a-z0-9]){re.escape(r.name.lower())}(?![a-z0-9])", lowered):
            keys.add(r.key)
    return keys


__all__ = [
    "AgentSelection",
    "DEFAULT_TOP_K",
    "bm25_scores",
    "mentioned_agent_keys",
    "score_agents",
    "search_agents",
    "select_agents",
    "tokenize",
]
