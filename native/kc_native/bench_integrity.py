"""Cold logical-cache misses vs immediate warm repeats; never drops OS caches.

Uses bench_search's unchanged seed-7 corpus and seed-11 queries. Each cold query
gets a new ConversationLog (empty metadata/fold/snippet caches and a new SQLite
connection); its warm repeat reuses that exact object. The indexed path decodes
blobs on every request: it has no snippet-result or inflated-blob cache.
Run with --library pointing at an isolated, freshly built libkc_native shared
library. Nothing is installed and no live data or gateway is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import tempfile
import time
from pathlib import Path

import bench_search as bench


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--queries", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("kc_native", args.library)
    assert spec is not None and spec.loader is not None
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    assert native.SEARCH_FINGERPRINT == bench.history_search._native_search_fingerprint()
    queries = bench.make_queries(11, args.queries)
    rows = []
    with tempfile.TemporaryDirectory(prefix="kc-search-integrity-") as tmp:
        root = Path(tmp)
        seed = bench.build_corpus(root, args.sessions, 7)
        seed._catalog_projection._index.close()
        corpus_bytes = sum(p.stat().st_size for p in root.glob("*.jsonl"))
        for round_id in range(args.rounds):
            # Ten-query blocks limit load drift; both orderings occur each round.
            for start in range(0, len(queries), 10):
                arms = [("python", None), ("native", native)]
                if (round_id + start // 10) % 2:
                    arms.reverse()
                for arm, module in arms:
                    bench.history_search._NATIVE_SEARCH = module
                    for qi in range(start, min(start + 10, len(queries))):
                        log = bench.ConversationLog(base_dir=root)
                        try:
                            cold_result = None
                            for phase in ("cold", "warm"):
                                load_before = bench._loadavg()
                                t = time.perf_counter()
                                result = log.search_sessions(queries[qi], limit=50)
                                elapsed = (time.perf_counter() - t) * 1000
                                identity = [(r["key"], r.get("snippet", "")) for r in result]
                                if phase == "cold":
                                    cold_result = identity
                                else:
                                    assert identity == cold_result
                                rows.append(
                                    dict(
                                        round=round_id,
                                        query=qi,
                                        arm=arm,
                                        phase=phase,
                                        ms=elapsed,
                                        load_before=load_before,
                                        load_after=bench._loadavg(),
                                    )
                                )
                        finally:
                            if log._catalog_projection._index is not None:
                                log._catalog_projection._index.close()
            print(f"round {round_id + 1}/{args.rounds}", flush=True)
    summary = {}
    for arm in ("python", "native"):
        for phase in ("cold", "warm"):
            per_round = []
            for r in range(args.rounds):
                vals = [
                    x["ms"] for x in rows if (x["arm"], x["phase"], x["round"]) == (arm, phase, r)
                ]
                per_round.append(dict(p50_ms=statistics.median(vals), p95_ms=bench._pct(vals, 95)))
            summary[f"{arm}_{phase}"] = per_round
    report = dict(
        sessions=args.sessions,
        corpus_bytes=corpus_bytes,
        queries=queries,
        library=str(args.library.resolve()),
        summary=summary,
        rows=rows,
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
