"""Micro-benchmark: session search with and without the native primitives.

Builds a synthetic indexed corpus in a temporary directory (no gateway, no real
data), then measures, alternating the two variants round by round:

* single-thread search latency;
* per-query latency while 4 threads search at once;
* how late a thread that only sleeps 1 ms wakes up while those 4 threads search
  (a direct reading of GIL starvation).

Usage::

    python native/kc_native/bench_search.py [--sessions 300] [--rounds 4] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

from kiro_crew import history_search
from kiro_crew.history import ConversationLog

_WORDS = (
    "gateway session search index memory latency deploy pipeline review merge "
    "request timeout contention hypothesis snippet redaction transcript worker "
    "conductor sqlite trigram fold needle score ranking recency budget python rust"
).split()
_CJK = ["\u5185\u5b58\u6cc4\u6f0f", "\u91cd\u542f\u4e4b\u540e", "\u5360\u7528\u56de\u843d"]


def _text(rng: random.Random, words: int) -> str:
    out = []
    for _ in range(words):
        r = rng.random()
        if r < 0.03:
            out.append(rng.choice(_CJK))
        elif r < 0.06:
            out.append(str(rng.randint(1, 99999)))
        elif r < 0.07:
            out.append(f"#{rng.randint(1, 5000)}")
        else:
            out.append(rng.choice(_WORDS) + rng.choice(("", "", "s", "ed", "ing", "_v2")))
    return " ".join(out)


def build_corpus(root: Path, sessions: int, seed: int) -> ConversationLog:
    rng = random.Random(seed)
    log = ConversationLog(base_dir=root)
    for i in range(sessions):
        key = f"bench-{i:04d}"
        # Long-tailed sizes: most sessions are small, a few are megabytes.
        messages = int(rng.paretovariate(1.2) * 8)
        for _ in range(min(messages, 400)):
            log.append(key, rng.choice(("user", "assistant")), _text(rng, rng.randint(20, 600)))
    history_search._INDEX_QUIET_WINDOW_SECS = 0.0
    while log._catalog_projection.backfill_index(budget_secs=120).get("remaining"):
        pass
    return log


def make_queries(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        shape = rng.randrange(5)
        if shape == 0:
            out.append(rng.choice(_WORDS))
        elif shape == 1:
            out.append(" ".join(rng.sample(_WORDS, 2)))
        elif shape == 2:
            out.append(rng.choice(_WORDS)[:3])
        elif shape == 3:
            out.append(rng.choice(_CJK)[:2])
        else:
            out.append(f"#{rng.randint(1, 5000)}")
    return out


def _loadavg() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return -1.0


def _pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def _search_all(log: ConversationLog, queries: list[str], sink: list[float]) -> None:
    for q in queries:
        t = time.perf_counter()
        log.search_sessions(q, limit=50)
        sink.append((time.perf_counter() - t) * 1000)


def one_round(log: ConversationLog, native: object, queries: list[str]) -> dict:
    history_search._NATIVE_SEARCH = native
    single: list[float] = []
    _search_all(log, queries, single)

    parallel: list[float] = []
    lateness: list[float] = []
    stop = threading.Event()

    def sleeper() -> None:
        while not stop.is_set():
            t = time.perf_counter()
            time.sleep(0.001)
            lateness.append((time.perf_counter() - t) * 1000 - 1.0)

    sl = threading.Thread(target=sleeper)
    sl.start()
    workers = [
        threading.Thread(target=_search_all, args=(log, queries[i::4], parallel)) for i in range(4)
    ]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    wall = time.perf_counter() - t0
    stop.set()
    sl.join()
    return {
        "single_p50_ms": statistics.median(single),
        "single_p95_ms": _pct(single, 95),
        "four_p50_ms": statistics.median(parallel),
        "four_p95_ms": _pct(parallel, 95),
        "four_wall_s": wall,
        "sleeper_late_p50_ms": statistics.median(lateness),
        "sleeper_late_p99_ms": _pct(lateness, 99),
        "sleeper_samples": len(lateness),
        "loadavg_1m": _loadavg(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--queries", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import kc_native  # type: ignore[import-not-found]

    if history_search._load_native_search() is not kc_native:
        print("kc_native is not accepted by this tree's fingerprint", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="kc-search-bench-") as tmp:
        log = build_corpus(Path(tmp), args.sessions, seed=7)
        corpus_bytes = sum(p.stat().st_size for p in Path(tmp).glob("*.jsonl"))
        queries = make_queries(seed=11, count=args.queries)
        _search_all(log, queries, [])  # warm the OS page cache and the SQLite pages
        results: dict[str, list[dict]] = {"python": [], "native": []}
        for r in range(args.rounds):
            order = [("python", None), ("native", kc_native)]
            if r % 2:
                order.reverse()
            for name, mod in order:
                results[name].append(one_round(log, mod, queries))
        log._catalog_projection._index.close()
    summary = {
        name: {k: statistics.median(r[k] for r in rounds) for k in rounds[0]}
        for name, rounds in results.items()
    }
    report = {
        "sessions": args.sessions,
        "corpus_mb": round(corpus_bytes / 1e6, 1),
        "queries": args.queries,
        "rounds": args.rounds,
        "median_of_rounds": summary,
        "rounds_raw": results,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(json.dumps({"corpus_mb": report["corpus_mb"], **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
