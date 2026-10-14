# kc_native

Optional PyO3 fast paths for Kiro Crew. When the module is importable, each is used by default; when it is missing, out of date, or turned off, the pure-Python code runs. Both give the same answer for every input.

* `might_contain_credential`: the pre-filter in `kiro_crew.security.redaction`. Used when `GATE_FINGERPRINT` matches the Python pre-filter; `KIROCREW_NATIVE_REDACTION=0` turns it off.
* `count_needles` and `snippet_from_blob`: the counting and snippet scans of session search (`kiro_crew.history_search`). Python still parses the query and computes every score; these only do the scans, with the GIL released on large inputs. Used when `SEARCH_FINGERPRINT` matches `history_search._native_search_fingerprint()`; `KIROCREW_NATIVE_SEARCH=0` turns it off.

Strings are read from the str object's own storage and never converted to UTF-8, because CPython keeps that UTF-8 copy inside the string until the string dies. That is also why the module is built for one CPython version (no abi3): the limited API cannot reach a string's storage.

## Unicode tables

Search needs `str.isdigit`, `str.isspace` and `str.casefold` exactly as the interpreter defines them. `src/search_tables.rs` is generated from the interpreter the module is built for:

```bash
python native/kc_native/gen_search_tables.py > native/kc_native/src/search_tables.rs
```

## After editing the mirrored Python

`SEARCH_FINGERPRINT` in `src/search.rs` hashes the code of `count_needle`, `snippet_from_texts` and `decode_raw_texts` (comments and docstrings excluded) plus the Unicode version. Editing any of them turns the native search path off until the Rust side is brought in line and the constant updated to:

```bash
python -c "from kiro_crew import history_search as h; print(h._native_search_fingerprint())"
```

`test/test_native_search_fallback.py` fails while the constant is stale.

## Build

Into the gateway's own venv:

```bash
cd native/kc_native
VIRTUAL_ENV=/path/to/venv PATH=/path/to/venv/bin:$PATH \
  pip install maturin==1.15.0 && maturin develop --release --locked
```

For a wheel: `maturin build --release --locked`, then `pip install target/wheels/kc_native-*.whl`.

## Tests

```bash
cargo test --release --locked                                       # in native/kc_native
.venv/bin/python -m pytest test/test_native_redaction_gate.py test/test_native_search_parity.py   # need the module built
.venv/bin/python -m pytest test/test_native_redaction_fallback.py test/test_native_search_fallback.py  # run without it
```

## Benchmark

```bash
.venv/bin/python native/kc_native/bench_search.py --sessions 300 --rounds 6
```

Builds a synthetic indexed corpus in a temporary directory and reports single-thread and 4-thread search latency and the lateness of a thread sleeping 1 ms, alternating native and Python rounds.
