"""Session search uses the native primitives only when they match, else Python.

These cases run without ``kc_native`` built. ``test_native_search_parity.py``
covers the built module.
"""

from __future__ import annotations

import re
import runpy
import sys
import types
import unicodedata
from pathlib import Path

import pytest

from kiro_crew import history_search
from kiro_crew.history import ConversationLog

_NATIVE_DIR = Path(__file__).resolve().parents[1] / "native" / "kc_native"
_TABLES = (_NATIVE_DIR / "src" / "search_tables.rs").read_text(encoding="utf-8")
_TABLES_UNIDATA = re.search(r'UNIDATA_VERSION: &str = "([^"]+)"', _TABLES).group(1)  # type: ignore[union-attr]

#: The committed tables and fingerprint describe one Unicode version; an
#: interpreter with another one correctly gets the Python path instead.
_same_unicode = pytest.mark.skipif(
    unicodedata.unidata_version != _TABLES_UNIDATA,
    reason=f"tables are for Unicode {_TABLES_UNIDATA}",
)


def _fake(fingerprint: str) -> types.ModuleType:
    fake = types.ModuleType("kc_native")
    fake.SEARCH_FINGERPRINT = fingerprint
    fake.count_needles = lambda text, specs: [0] * len(specs)
    fake.snippet_from_blob = lambda *args: None
    return fake


def test_a_matching_module_is_used_by_default(monkeypatch):
    monkeypatch.delenv("KIROCREW_NATIVE_SEARCH", raising=False)
    fake = _fake(history_search._native_search_fingerprint())
    monkeypatch.setitem(sys.modules, "kc_native", fake)
    assert history_search._load_native_search() is fake


def test_missing_module_keeps_python(monkeypatch):
    monkeypatch.delenv("KIROCREW_NATIVE_SEARCH", raising=False)
    monkeypatch.setitem(sys.modules, "kc_native", None)  # import raises ImportError
    assert history_search._load_native_search() is None


def test_stale_module_keeps_python(monkeypatch, caplog):
    monkeypatch.delenv("KIROCREW_NATIVE_SEARCH", raising=False)
    monkeypatch.setitem(sys.modules, "kc_native", _fake("not-this-build"))
    with caplog.at_level("DEBUG", logger=history_search.__name__):
        assert history_search._load_native_search() is None
    assert "fingerprint does not match" in caplog.text


def test_module_without_search_keeps_python(monkeypatch):
    """A build of the crate from before the search primitives existed."""
    monkeypatch.delenv("KIROCREW_NATIVE_SEARCH", raising=False)
    monkeypatch.setitem(sys.modules, "kc_native", types.ModuleType("kc_native"))
    assert history_search._load_native_search() is None


def test_env_zero_turns_it_off(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "kc_native", _fake(history_search._native_search_fingerprint())
    )
    monkeypatch.setenv("KIROCREW_NATIVE_SEARCH", "0")
    assert history_search._load_native_search() is None


@_same_unicode
def test_rust_constant_matches_the_python_code():
    """Editing the mirrored Python code without updating the crate fails here,
    in every job, rather than only silently disabling the native path."""
    source = (_NATIVE_DIR / "src" / "search.rs").read_text(encoding="utf-8")
    match = re.search(r'pub const SEARCH_FINGERPRINT: &str = "([0-9a-f]{16})";', source)
    assert match is not None
    assert match.group(1) == history_search._native_search_fingerprint()


@_same_unicode
def test_committed_tables_match_the_generator():
    generator = runpy.run_path(str(_NATIVE_DIR / "gen_search_tables.py"))
    assert _TABLES == generator["render"]()


def test_fingerprint_ignores_comments_and_docstrings_but_not_code():
    def a(x):
        """Doc."""
        return x + 1  # note

    def b(x):
        # other note

        return x + 1

    def c(x):
        return x + 2

    assert history_search._code_tokens(a) == history_search._code_tokens(b)
    assert history_search._code_tokens(a) != history_search._code_tokens(c)


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(history_search, "_INDEX_QUIET_WINDOW_SECS", 0.0)
    log = ConversationLog(base_dir=tmp_path)
    log.append("s-a", "user", "we talked about ack contention and three hypotheses")
    log.append("s-a", "assistant", "the second hypothesis held")
    log.append("s-b", "user", "deployment notes, contention free")
    log._catalog_projection.backfill_index(budget_secs=30)
    try:
        yield log
    finally:
        index = log._catalog_projection._index
        if index is not None:
            index.close()


def test_snippet_falls_back_when_the_blob_does_not_decode(log, monkeypatch):
    monkeypatch.setattr(history_search, "_NATIVE_SEARCH", None)
    want = log.search_sessions("contention hypotheses")
    seen: list[bytes] = []

    def refuse(blob, *args):
        seen.append(blob)
        return None

    monkeypatch.setattr(
        history_search,
        "_NATIVE_SEARCH",
        types.SimpleNamespace(snippet_from_blob=refuse, count_needles=_python_count_needles),
    )
    assert log.search_sessions("contention hypotheses") == want
    assert seen, "the native snippet path was not tried"


def _python_count_needles(text, specs):
    """``count_needles`` with Python counting, for driving the native call sites."""
    out = []
    for spellings, digit_bounded, must_hit in specs:
        needle = history_search.SearchNeedle(
            spellings[0], 1.0, must_hit, tuple(spellings[1:]), digit_bounded
        )
        out.append(history_search.count_needle(needle, text))
    if any(n == 0 and spec[2] for n, spec in zip(out, specs)):
        return None
    return out


def test_native_call_sites_rank_like_python(log, monkeypatch):
    """The native wiring (specs, phrase count, disqualification) ranks exactly
    as the Python loop when the primitive answers like Python."""
    queries = ["contention", "contention hypotheses", "hyp", "deployment", "zzz", "ack held"]
    monkeypatch.setattr(history_search, "_NATIVE_SEARCH", None)
    want = [log.search_sessions(q) for q in queries]
    monkeypatch.setattr(
        history_search,
        "_NATIVE_SEARCH",
        types.SimpleNamespace(
            count_needles=_python_count_needles, snippet_from_blob=lambda *a: None
        ),
    )
    assert [log.search_sessions(q) for q in queries] == want


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError, AttributeError])
@pytest.mark.parametrize("primitive", ["count_needles", "snippet_from_blob"])
def test_native_errors_keep_python_results(log, monkeypatch, error, primitive):
    monkeypatch.setattr(history_search, "_NATIVE_SEARCH", None)
    query = "contention hypotheses"
    want = log.search_sessions(query)
    calls = []

    def fail(*args):
        calls.append(primitive)
        raise error("native scan failed")

    native = types.SimpleNamespace(
        count_needles=_python_count_needles, snippet_from_blob=lambda *args: None
    )
    setattr(native, primitive, fail)
    monkeypatch.setattr(history_search, "_NATIVE_SEARCH", native)
    assert log.search_sessions(query) == want
    assert calls, "the failing primitive was not reached"


@pytest.mark.parametrize("error", [OSError, RuntimeError])
def test_native_import_errors_keep_python(monkeypatch, error):
    import builtins

    original_import = builtins.__import__

    def import_native(name, *args, **kwargs):
        if name == "kc_native":
            raise error("native module initialization failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.delenv("KIROCREW_NATIVE_SEARCH", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_native)
    assert history_search._load_native_search() is None
