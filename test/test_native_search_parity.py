"""The native search primitives give exactly the Python answers.

Needs ``kc_native`` built (``native/kc_native``); every case skips without it, so
the CI job that builds the module imports it first, where a skip would not read
as a pass. ``test_native_search_fallback.py`` covers the paths without it.

Three levels, each compared native-on against native-off over the same corpus:
the counting primitive, the snippet primitive, and ``search_sessions`` end to end
over at least 400 queries. Any difference fails.
"""

from __future__ import annotations

import json
import random
import types
import zlib

import pytest

from kiro_crew import history_search
from kiro_crew._sqlite_compat import fts5_available
from kiro_crew.history import ConversationLog
from kiro_crew.history_index import decode_raw_texts, fold_texts

kc_native = pytest.importorskip("kc_native")

pytestmark = pytest.mark.skipif(not fts5_available(), reason="SQLite built without FTS5")

# Non-ASCII is written as escapes: the repository keeps source ASCII.
_WORDS = [
    "contention",
    "hypothesis",
    "deploy",
    "pipeline",
    "gateway",
    "timeout",
    "Memory",
    "leak",
    "redaction",
    "snippet",
    "Search",
    "index",
    "rust",
    "python",
    "pull",
    "request",
    "merge",
    "issue",
    "review",
    "STRASSE",
    "stra\u00dfe",
    "\u0130stanbul",
    "\u03a3\u03af\u03c3\u03c5\u03c6\u03bf\u03c2",
    "\u01c5ungla",
    "\ufb03ce",
    "caf\u00e9",
    "na\u00efve",
    "\u0645\u0631\u062d\u0628\u0627",
    "\U0001f680launch",
    "\U00010400\U00010428",
]
_CJK = [
    "\u5185\u5b58\u6cc4\u6f0f",
    "\u91cd\u542f\u4e4b\u540e",
    "\u5360\u7528\u56de\u843d",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30b9\u30c8",
    "\ud55c\uad6d\uc5b4",
]
_DIGITS = [
    "4411",
    "44110",
    "1544110293",
    "\u0664\u0664\u0661\u0661",
    "\uff14\uff14\uff11\uff11",
    "\u00b2\u00b3",
    "12",
    "5",
    "50051",
    # Non-ASCII digits touching a reference: the digit boundary must see them.
    "4411\u00b2",
    "\u06604411",
    "#4411\uff11",
    "pull/4411\u0663",
    "\u00b9#77",
]
_SPACES = [" ", "  ", "\t", "\n", "\x1c", "\u3000", "\u2028", "\xa0", "\x85"]
_FORGE = [
    "#4411",
    "pr 4411",
    "https://github.com/o/r/pull/4411",
    "owner/repo2#4411",
    "!12",
    "merge request 12",
    "issue 77",
    "#77",
]


def _sentence(rng: random.Random, n: int) -> str:
    parts = []
    for _ in range(n):
        pool = rng.choice((_WORDS, _WORDS, _CJK, _DIGITS, _FORGE))
        parts.append(rng.choice(pool))
        parts.append(rng.choice(_SPACES))
    if rng.random() < 0.1:
        parts.append("\x00nul")
    return "".join(parts)


def _seed(log: ConversationLog, rng: random.Random, sessions: int, prefix: str) -> None:
    for i in range(sessions):
        key = f"{prefix}-{i:02d}"
        for _ in range(rng.randint(1, 6)):
            log.append(key, rng.choice(("user", "assistant")), _sentence(rng, rng.randint(1, 30)))
    # Large enough to take the GIL-released branch of both primitives.
    log.append(f"{prefix}-big", "user", _sentence(rng, 40) * 900)


def _queries(rng: random.Random, count: int) -> list[str]:
    pools = (_WORDS, _CJK, _DIGITS, _FORGE)
    out: list[str] = []
    while len(out) < count:
        shape = rng.randrange(6)
        if shape == 0:
            q = rng.choice(rng.choice(pools))
        elif shape == 1:
            q = " ".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(2, 4)))
        elif shape == 2:
            w = rng.choice(_WORDS)
            q = w[: rng.randint(1, max(1, len(w)))]
        elif shape == 3:
            c = rng.choice(_CJK)
            start = rng.randrange(len(c))
            q = c[start : start + rng.randint(1, 3)]
        elif shape == 4:
            q = rng.choice(_WORDS).upper() + " " + rng.choice(_DIGITS)
        else:
            q = rng.choice(_FORGE) + " " + rng.choice(_WORDS)
        out.append(q)
    return out


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(history_search, "_INDEX_QUIET_WINDOW_SECS", 0.0)
    rng = random.Random(20261013)
    log = ConversationLog(base_dir=tmp_path)
    _seed(log, rng, 30, "indexed")
    log._catalog_projection.backfill_index(budget_secs=60)
    # Written after the backfill, so search reads these from the transcripts.
    _seed(log, rng, 8, "scanned")
    try:
        yield log, rng
    finally:
        index = log._catalog_projection._index
        if index is not None:
            index.close()


def test_a_fresh_build_is_accepted():
    """The module built from this tree must pass the loader's fingerprint check."""
    assert kc_native.SEARCH_FINGERPRINT == history_search._native_search_fingerprint()
    assert history_search._load_native_search() is kc_native


def _session_texts(log: ConversationLog) -> list[list[str]]:
    return [list(log._iter_message_texts(m["key"])) for m in log.list_sessions()]


def test_counts_match_python(corpus, monkeypatch):
    log, rng = corpus
    folded_docs = [fold_texts(t) for t in _session_texts(log)] + ["", "\ud800 4411 \udfff"]
    compared = 0
    for query in _queries(rng, 400):
        needles, phrase, _ = history_search.parse_search_query(query)
        if not needles:
            continue
        for folded in folded_docs:
            titles = [rng.choice((0, 0, 1)) for _ in needles]
            monkeypatch.setattr(history_search, "_NATIVE_SEARCH", None)
            want = history_search._content_counts(needles, folded, titles, None)
            if want is not None:
                want = (want[0], folded.count(phrase) if folded else 0)
            monkeypatch.setattr(history_search, "_NATIVE_SEARCH", kc_native)
            got = history_search._content_counts(needles, folded, titles, phrase)
            if not folded and got is not None:
                got = (got[0], 0)
            assert got == want, (query, folded[:80])
            compared += 1
    assert compared >= 400 * 10


def test_snippets_match_python(corpus):
    log, rng = corpus
    index = log._catalog_projection._index
    blobs = []
    for meta in log.list_sessions():
        st = log._path(meta["key"]).stat()
        blob = index.raw_blob(meta["key"], st)
        if blob is not None:
            blobs.append(blob)
    assert len(blobs) >= 30
    lead, trail, cap = log._SNIPPET_LEAD, log._SNIPPET_TRAIL, log._SNIPPET_MAX
    for query in _queries(rng, 400):
        needles = history_search.snippet_needles(query)
        if not needles:
            continue
        for blob in blobs:
            want = history_search.snippet_from_texts(
                decode_raw_texts(blob) or [], needles, lead, trail, cap
            )
            assert kc_native.snippet_from_blob(blob, needles, lead, trail, cap) == want, query
    assert kc_native.snippet_from_blob(b"not zlib", ["x"], lead, trail, cap) is None


def test_search_sessions_matches_python(corpus, monkeypatch):
    log, rng = corpus
    queries = _queries(rng, 450)
    monkeypatch.setattr(history_search, "_NATIVE_SEARCH", None)
    want = [log.search_sessions(q, limit=1000) for q in queries]
    calls = {"count_needles": 0, "snippet_from_blob": 0}

    def _counted(name):
        def call(*args):
            calls[name] += 1
            return getattr(kc_native, name)(*args)

        return call

    spy = types.SimpleNamespace(**{name: _counted(name) for name in calls})
    monkeypatch.setattr(history_search, "_NATIVE_SEARCH", spy)
    got = [log.search_sessions(q, limit=1000) for q in queries]
    diffs = [q for q, a, b in zip(queries, want, got) if a != b]
    assert diffs == []
    assert sum(len(rows) for rows in want) > 1000
    assert sum(1 for rows in want for r in rows if r.get("snippet")) > 500
    # Both primitives actually served this run.
    assert calls["count_needles"] > 1000 and calls["snippet_from_blob"] > 500


def _damaged_blob(kind):
    # Stored blocks put both mutations at deterministic uncompressed positions.
    raw = json.dumps(["alpha stable context", "tail " * 50000]).encode()
    blob = bytearray(zlib.compress(raw, level=0))
    if kind == "hit_bit":
        blob[blob.index(b"alpha stable context") + 5] ^= 1
    elif kind == "tail_bit":
        blob[blob.rindex(b"tail ") + 4] ^= 1
    elif kind == "checksum":
        blob[-1] ^= 1
    elif kind == "truncated_checksum":
        del blob[-1:]
    elif kind == "missing_checksum":
        del blob[-4:]
    elif kind == "truncated_payload":
        del blob[-100:]
    elif kind == "invalid_json":
        blob = zlib.compress(raw[:-1] + b",]", level=0)
    elif kind == "trailing_json":
        blob = zlib.compress(raw + b" false", level=0)
    elif kind == "invalid_utf8":
        blob = zlib.compress(raw[:-2] + b'\xff"]', level=0)
    else:
        raise AssertionError(kind)
    return bytes(blob)


_DAMAGE = (
    "hit_bit",
    "tail_bit",
    "checksum",
    "truncated_checksum",
    "missing_checksum",
    "truncated_payload",
    "invalid_json",
    "trailing_json",
    "invalid_utf8",
)


@pytest.mark.parametrize("kind", _DAMAGE)
def test_damaged_blob_never_returns_a_partial_snippet(kind):
    blob = _damaged_blob(kind)
    assert decode_raw_texts(blob) is None
    assert kc_native.snippet_from_blob(blob, ["alpha"], 20, 80, 200) is None


@pytest.mark.parametrize("needle", ["alpha", "absent"])
def test_every_truncation_is_rejected(needle):
    blob = zlib.compress(b'["alpha"]', level=0)
    for end in range(len(blob)):
        truncated = blob[:end]
        assert decode_raw_texts(truncated) is None
        assert kc_native.snippet_from_blob(truncated, [needle], 20, 80, 200) is None, end


@pytest.mark.parametrize("tail", ["1", "null", "{}", "[]", '"\\ud800"'])
def test_unsupported_tail_defers_to_python(tail):
    blob = zlib.compress(('["alpha", ' + tail + "]").encode())
    # Python accepts these lists; native cannot prove string-scan equivalence.
    assert decode_raw_texts(blob) is not None
    assert kc_native.snippet_from_blob(blob, ["alpha"], 20, 80, 200) is None


@pytest.mark.parametrize("suffix", [b"", b"ignored", zlib.compress(b"not JSON")])
@pytest.mark.parametrize("needles", [["beta", "alpha"], ["absent"], [""]])
def test_valid_blob_keeps_first_hit_and_python_stream_boundary(suffix, needles):
    texts = ["alpha stable context", "beta later", "tail " * 50000]
    blob = zlib.compress(json.dumps(texts).encode() + b" \r\n\t", level=0) + suffix
    assert decode_raw_texts(blob) == texts
    want = history_search.snippet_from_texts(texts, needles, 20, 80, 200)
    assert kc_native.snippet_from_blob(blob, needles, 20, 80, 200) == want


@pytest.mark.parametrize("kind", _DAMAGE)
def test_search_rereads_intact_transcript_for_damaged_index(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(history_search, "_INDEX_QUIET_WINDOW_SECS", 0.0)
    log = ConversationLog(base_dir=tmp_path)
    log.append("s-a", "user", "alpha stable context")
    log.append("s-a", "assistant", "tail " * 50000)
    projection = log._catalog_projection
    try:
        projection.backfill_index(budget_secs=60)
        index = projection._index
        assert index is not None and index.available
        blob = _damaged_blob(kind)
        conn = index._ensure_open()
        updated = conn.execute(
            "UPDATE session_entry SET raw_texts=? WHERE session_key=?", (blob, "s-a")
        )
        assert updated.rowcount == 1
        assert projection._snippet_blob("s-a") == blob
        reads = []
        native_results = []
        original = log._iter_message_texts

        def read(key):
            reads.append(key)
            return original(key)

        def snippet(*args):
            result = kc_native.snippet_from_blob(*args)
            native_results.append(result)
            return result

        monkeypatch.setattr(log, "_iter_message_texts", read)
        monkeypatch.setattr(
            history_search,
            "_NATIVE_SEARCH",
            types.SimpleNamespace(count_needles=kc_native.count_needles, snippet_from_blob=snippet),
        )
        got = log.search_sessions("alpha")
        assert native_results == [None]
        assert reads == ["s-a"]
        assert len(got) == 1 and got[0]["snippet"] == "alpha stable context"
        monkeypatch.setattr(history_search, "_NATIVE_SEARCH", None)
        assert got == log.search_sessions("alpha")
    finally:
        if projection._index is not None:
            projection._index.close()


@pytest.mark.parametrize("size", [(1 << 20) - 1, 1 << 20, (1 << 20) + 1, 2 << 20])
@pytest.mark.parametrize("damage", [None, "checksum", "truncated", "json", "utf8", "surrogate"])
def test_bounded_snippet_decode_boundary(size, damage):
    raw = b'["alpha"]' + b" " * (size - 9)
    if damage == "json":
        raw += b"false"
    elif damage == "utf8":
        raw += b"\xff"
    elif damage == "surrogate":
        raw = b'["alpha","\\ud800"]' + b" " * size
    blob = bytearray(zlib.compress(raw))
    if damage == "checksum":
        blob[-1] ^= 1
    elif damage == "truncated":
        del blob[-1:]
    blob = bytes(blob)
    texts = decode_raw_texts(blob)
    for needle in ["alpha", "absent"]:
        want = (
            history_search.snippet_from_texts(texts, [needle], 20, 80, 200)
            if texts is not None and damage != "surrogate"
            else None
        )
        assert kc_native.snippet_from_blob(blob, [needle], 20, 80, 200) == want
