"""Linux-only uninstrumented RSS probe; synthetic blobs, isolated child per arm."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def rss() -> dict[str, int]:
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith(("VmRSS:", "VmHWM:"))
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    library = ap.add_mutually_exclusive_group(required=True)
    library.add_argument("--library", type=Path)
    library.add_argument(
        "--library-stdin",
        action="store_true",
        help="Read a JSON library path from stdin (child only)",
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", choices=["python", "native"])
    ap.add_argument("--blob", type=Path)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.library_stdin:
        if not args.child:
            ap.error("--library-stdin requires --child; use --library for the parent")
        args.library = Path(json.load(sys.stdin))
    if args.child:
        from kiro_crew.history_index import decode_raw_texts
        from kiro_crew.history_search import snippet_from_texts

        spec = importlib.util.spec_from_file_location("kc_native", args.library)
        assert spec is not None and spec.loader is not None
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        blob = args.blob.read_bytes()
        barrier = threading.Barrier(args.threads)

        def run() -> None:
            barrier.wait(timeout=60)
            for _ in range(20):
                if args.child == "native":
                    result = native.snippet_from_blob(blob, ["alpha"], 20, 80, 200)
                else:
                    result = snippet_from_texts(decode_raw_texts(blob), ["alpha"], 20, 80, 200)
                assert result == "alpha"

        gc.collect()
        before = rss()
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(run) for _ in range(args.threads)]
            for future in futures:
                future.result(timeout=120)
        print(json.dumps(dict(before=before, after=rss())))
        return
    rows = []
    with tempfile.TemporaryDirectory(prefix="kc-snippet-memory-") as tmp:
        for count in [900, 1100, 4000]:
            rng = random.Random(23)
            texts = ["alpha"] + [rng.randbytes(500).hex() for _ in range(count)]
            raw = json.dumps(texts).encode()
            blob = Path(tmp) / "blob.z"
            blob.write_bytes(zlib.compress(raw))
            for round_id in range(6):
                for threads in [1, 4]:
                    arms = ["python", "native"] if round_id % 2 == 0 else ["native", "python"]
                    for arm in arms:
                        cmd = [
                            sys.executable,
                            __file__,
                            "--child",
                            arm,
                            "--threads",
                            str(threads),
                            "--blob",
                            str(blob),
                            "--library-stdin",
                        ]
                        result = subprocess.run(
                            cmd,
                            input=json.dumps(str(args.library)),
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            check=True,
                            timeout=180,
                        )
                        rows.append(
                            dict(
                                round=round_id,
                                arm=arm,
                                threads=threads,
                                raw_bytes=len(raw),
                                compressed_bytes=blob.stat().st_size,
                                **json.loads(result.stdout),
                            )
                        )
    assert args.out is not None
    args.out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
