"""Paired analysis of live routing results.

All conditions answer the *same* queries, so comparisons use an exact two-sided McNemar test
on the discordant pairs, not an independent-samples test. Queries where either condition hit
an API error are dropped from that comparison, and the remaining n is reported.

    .venv/bin/python -m evals.analyze_live evals/results/routing_llm_sonnet55_fresh.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from math import comb
from typing import Dict, List

MISROUTE = {"wrong_existing", "new_duplicate"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _key(row: Dict) -> str:
    return f"{row['kind']}|{row['message']}|{row['target']}"


def analyze(results: Dict) -> str:
    conds = [c for c in ("baseline", "full_roster", "after_heuristic_desc", "after") if c in results]
    rows = {c: {_key(r): r for r in results[c]["rows"]} for c in conds}
    lines = ["| condition | n answered | correct | silent misroute | asked to clarify | API errors | prompt tokens / query |",
             "|---|---|---|---|---|---|---|"]
    for c in conds:
        rs = list(rows[c].values())
        ok = [r for r in rs if r["outcome"] != "api_error"]
        n = len(ok)
        pct = lambda x: f"{x}/{n} ({100 * x / n:.0f}%)" if n else "-"
        tokens = round(sum(r["prompt_tokens"] for r in ok) / n) if n else 0
        lines.append(f"| {c} | {n} | {pct(sum(r['correct'] for r in ok))} | "
                     f"{pct(sum(r['outcome'] in MISROUTE for r in ok))} | "
                     f"{pct(sum(r['outcome'] == 'no_dispatch' for r in ok))} | {len(rs) - n} | {tokens:,} |")

    lines += ["", "Paired comparisons (correct agent chosen; exact two-sided McNemar):", "",
              "| A vs B | pairs | only A right | only B right | p |", "|---|---|---|---|---|"]
    for a, b in combinations(conds, 2):
        keys = [k for k in rows[a] if k in rows[b]
                and rows[a][k]["outcome"] != "api_error" and rows[b][k]["outcome"] != "api_error"]
        only_a = sum(rows[a][k]["correct"] and not rows[b][k]["correct"] for k in keys)
        only_b = sum(rows[b][k]["correct"] and not rows[a][k]["correct"] for k in keys)
        lines.append(f"| {a} vs {b} | {len(keys)} | {only_a} | {only_b} | {mcnemar_exact(only_a, only_b):.3f} |")

    kinds: List[str] = sorted({r["kind"] for c in conds for r in rows[c].values()})
    lines += ["", "Accuracy by query kind:", "", "| condition | " + " | ".join(kinds) + " |",
              "|---|" + "---|" * len(kinds)]
    for c in conds:
        ok = [r for r in rows[c].values() if r["outcome"] != "api_error"]
        cells = []
        for k in kinds:
            rk = [r for r in ok if r["kind"] == k]
            cells.append(f"{sum(r['correct'] for r in rk)}/{len(rk)}" if rk else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    for path in sys.argv[1:]:
        print(f"## {path}\n")
        print(analyze(json.load(open(path))))
        print()


if __name__ == "__main__":
    main()
