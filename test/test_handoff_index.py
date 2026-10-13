"""The handed-over index: what it answers, and what it answers when it cannot read.

``handed_over`` exists so a start can tell "this session was handed to another session"
from "it never was" without reading any log, and so that the party whose grant a handoff
REVOKED cannot be the one who clears the record of it. Both properties rest on the shape
of its failures rather than on its happy path, so the failures are what this file is
about:

* an absent store is DECIDED and empty -- no handoff was recorded where nothing was ever
  written, and a first run must not refuse every slot;
* an index that EXISTS and cannot be established is UNDECIDED, and the caller refuses
  the manager-dependent verbs until a read succeeds, because a short read of this file
  is indistinguishable from "no handoff" and that is the one direction that hands a
  revoked grant back;
* an undecided answer is never cached and never rewritten, so one transient fault costs
  the calls made while it stood instead of the life of the process, and cannot turn an
  unreadable index into a permanently wrong one;
* a write that cannot land REFUSES, because a grant the index does not carry is one the
  next start will not read;
* a READER never publishes, because a read that stored its answer could put an older
  set back over a newer published one and so lose a committed handoff;
* and the read happens ONCE PER PASS, off the loop, with the answer passed down as a
  value -- so no slot's construction reads a file and no loop-affine function consults
  this module at all.

Each refusal is driven through a real file on disk wherever a real file can produce it
-- a directory where the index belongs, bytes that are not UTF-8, a key past the bound --
rather than by patching the module's own answers.
"""

from __future__ import annotations

import pytest

from kiro_crew.crew_log import handed_over
from kiro_crew.crew_log.errors import CrewLogError
from kiro_crew.crew_log.store import KIND_SESSION, crew_log_root
from kiro_crew.validation import MAX_SHORT_STRING


@pytest.fixture(autouse=True)
def _store(tmp_path, _floor_monkeypatch):
    """Bind the process to a store of this test's own, and forget what was published.

    The published snapshot is process-global by design (it spares the writer a re-read
    of a file it just wrote), so a test that did not clear it would answer from the
    previous test's store.

    Patched through ``_floor_monkeypatch`` and NOT the shared ``monkeypatch``, which is
    the rule for an AUTOUSE fixture: ``monkeypatch.undo()`` reverts every record on the
    instance it is called on, and several tests here call it mid-way to put the real
    ``_read`` back. On the shared instance that would also drop this ``KIROCREW_HOME``
    pin, and everything after it would resolve the OPERATOR's real data home -- where
    this file writes index files, makes directories and plants unreadable bytes.
    """
    _floor_monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "crew-home"))
    handed_over.reset_caches()
    yield
    handed_over.reset_caches()


def _index_path():
    return crew_log_root(KIND_SESSION) / handed_over.INDEX_DIR / handed_over.INDEX_NAME


def _write_index(raw: bytes | str) -> None:
    """Put *raw* at the index path, creating the leaf directory the writer would."""
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    path.write_bytes(raw)


# ── decided ──────────────────────────────────────────────────────────────────


def test_an_absent_store_is_decided_and_empty():
    """The first run of a gateway. Undecided here would refuse every slot it restores."""
    assert handed_over.handed_over_slots() == (frozenset(), True)


def test_a_recorded_slot_is_answered_and_the_answer_survives_a_fresh_read():
    """The happy path, read twice: once from the writer's own cache and once from disk."""
    assert handed_over.record_handed_over("chat-worker") is True
    assert handed_over.handed_over_slots() == (frozenset({"chat-worker"}), True)

    handed_over.reset_caches()
    assert handed_over.handed_over_slots() == (frozenset({"chat-worker"}), True)
    assert _index_path().read_text(encoding="utf-8") == "chat-worker\n"


def test_recording_the_same_slot_twice_writes_nothing():
    """The fact is "handed over at least once", which is what keeps the file
    proportional to handed-over SLOTS rather than to handoffs."""
    assert handed_over.record_handed_over("chat-worker") is True
    before = _index_path().stat().st_mtime_ns

    assert handed_over.record_handed_over("chat-worker") is True, "already recorded is true"
    assert _index_path().stat().st_mtime_ns == before, "and nothing was rewritten"


def test_each_distinct_slot_is_recorded_once_and_sorted():
    """Sorted so the file is comparable between runs, and a diff of it reads."""
    for key in ("chat-c", "chat-a", "chat-b", "chat-a"):
        assert handed_over.record_handed_over(key) is True

    assert _index_path().read_text(encoding="utf-8") == "chat-a\nchat-b\nchat-c\n"


# ── a key this index will not carry ──────────────────────────────────────────


@pytest.mark.parametrize("key", ["", None, "x" * (MAX_SHORT_STRING + 1)])
def test_a_key_the_index_cannot_carry_is_refused_before_any_write(key):
    """Refused rather than written, because the bound the READ applies has to be one
    the writer cannot put the file past -- otherwise this writer makes the index
    undecided for every later reader, including itself."""
    assert handed_over.record_handed_over(key) is False
    assert not _index_path().exists(), "nothing was written"


# ── undecided: the index exists and cannot be established ───────────────────


def test_an_unreadable_index_is_undecided_and_is_not_cached():
    """A directory where the file belongs: an OSError from the read itself.

    UNDECIDED, so the caller refuses; NOT cached, so the next call reads again. Caching
    it would make one transient fault permanent while every refusal it produces keeps
    telling the caller to retry in a moment.
    """
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()

    assert handed_over.handed_over_slots() == (frozenset(), False)
    assert handed_over._snapshot is None, "an undecided answer is not cached"

    # And it clears by itself once the fault does.
    path.rmdir()
    _write_index("chat-worker\n")
    assert handed_over.handed_over_slots() == (frozenset({"chat-worker"}), True)


def test_an_unforeseen_failure_is_undecided_rather_than_raised(monkeypatch):
    """The read never raises, so no caller needs a wrapper deciding what a raise means.

    A wrapper is where that decision gets made differently by the second caller to need
    it, and a raise lands in the middle of a build that has no answer for it either. So
    anything the read does not model answers undecided -- the same answer as a file it
    could not establish, because to a caller "the read did not happen" and "the read did
    not decide" are one fact and both must refuse.
    """

    def _unforeseen():
        raise RuntimeError("the data home could not be resolved at all")

    monkeypatch.setattr(handed_over, "_read", _unforeseen)

    assert handed_over.handed_over_slots() == handed_over.UNDECIDED
    assert handed_over._snapshot is None, "and nothing was published from it"


def test_a_refused_store_root_is_undecided():
    """The root is not the real one -- a linked or wrong kind directory. Nothing can be
    established about a store this reader will not open, including that it is empty."""
    root = crew_log_root(KIND_SESSION)
    root.parent.mkdir(parents=True, exist_ok=True)
    (root.parent / "elsewhere").mkdir()
    root.symlink_to(root.parent / "elsewhere")

    assert handed_over.handed_over_slots() == (frozenset(), False)


def test_an_index_past_the_byte_bound_is_undecided_before_it_is_parsed():
    """The ceiling is applied to the BYTES, so a file grown by something other than this
    writer cannot be read into memory first and judged after."""
    _write_index(b"x" * (handed_over.MAX_INDEX_BYTES + 1))

    assert handed_over.handed_over_slots() == (frozenset(), False)


def test_an_index_that_is_not_utf8_is_undecided():
    """Undecided and not empty: bytes that do not decode say nothing about whether a
    handoff was recorded."""
    _write_index(b"chat-worker\n\xff\xfe\n")

    assert handed_over.handed_over_slots() == (frozenset(), False)


def test_an_over_long_slot_key_in_the_file_is_undecided():
    """One line past the per-key bound makes the whole file undecided. Dropping the long
    line instead would answer from a file this reader knows it does not understand."""
    _write_index("chat-worker\n" + "x" * (MAX_SHORT_STRING + 1) + "\n")

    assert handed_over.handed_over_slots() == (frozenset(), False)


def test_an_index_past_the_slot_bound_is_undecided():
    """Past the count bound the read refuses too, which costs a handoff rather than
    answering from a file whose size this reader will not accept on the next start."""
    _write_index("".join(f"chat-{n}\n" for n in range(handed_over.MAX_INDEXED_SLOTS + 1)))

    assert handed_over.handed_over_slots() == (frozenset(), False)


# ── a write that cannot land refuses the handoff ─────────────────────────────


def test_an_undecided_index_is_never_rewritten():
    """The write is refused, and the file is left exactly as it was.

    Rewriting it with what this process believes would DROP every slot the unreadable
    file holds -- one unreadable index turned into a permanently wrong one, which is
    worse than the fault and does not clear when it does.
    """
    _write_index(b"\xff\xfe\n")
    raw = _index_path().read_bytes()

    assert handed_over.record_handed_over("chat-worker") is False
    assert _index_path().read_bytes() == raw, "the unreadable file is untouched"
    assert handed_over._snapshot is None, "and the undecided read is still not cached"


def test_a_full_index_refuses_a_new_slot_and_still_answers_the_ones_it_holds():
    """At the bound the handoff is refused, not silently dropped from the index, and a
    slot already recorded is still answered -- the bound is on growth."""
    at_bound = [f"chat-{n}" for n in range(handed_over.MAX_INDEXED_SLOTS)]
    _write_index("".join(f"{key}\n" for key in at_bound))

    assert handed_over.record_handed_over("chat-one-too-many") is False
    assert handed_over.record_handed_over(at_bound[0]) is True, "already recorded"


def test_a_write_that_fails_refuses_the_handoff(monkeypatch):
    """A grant the index does not carry is one the next start will not read, so the
    recipient would hold a reach that vanishes at the next restart while the giver
    silently gets it back. Refusing costs the handoff instead."""

    def _boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(handed_over, "atomic_write", _boom)

    assert handed_over.record_handed_over("chat-worker") is False
    assert handed_over.handed_over_slots() == (frozenset(), True), "and nothing was recorded"


def test_a_refused_path_on_the_write_side_refuses_the_handoff(monkeypatch):
    """The root check runs on the WRITE too, and its refusal is a ``CrewLogError``
    rather than an ``OSError`` -- caught as the same answer, because the handoff is
    equally unrecordable either way."""
    real = handed_over.checked_crew_log_root
    calls: list[str] = []

    def _refuse_the_write(kind: str):
        calls.append(kind)
        if len(calls) > 1:
            raise CrewLogError("the kind root is not the real one", code="root_refused")
        return real(kind)

    monkeypatch.setattr(handed_over, "checked_crew_log_root", _refuse_the_write)

    assert handed_over.record_handed_over("chat-worker") is False


# ── the cache is a cache of DECIDED answers only ─────────────────────────────


def test_a_reader_never_publishes_what_it_read(monkeypatch):
    """The finding, driven: a reader that stored its answer could DROP a handoff.

    The sequence is ordinary and the loss is permanent. A reader starts, reads the file
    as it stands; a handoff lands meanwhile and publishes the newer set; the reader then
    stores the OLDER set it is holding. From that moment this process believes the newer
    slot was never handed over -- and the next handoff rewrites the whole file from that
    belief, so the committed handoff leaves the index on disk too. The creator whose
    grant the log recorded going away has it back.

    So a read publishes nothing. Driven by letting one read overlap a write, which is
    the race, rather than by asserting on the attribute alone.
    """
    assert handed_over.record_handed_over("chat-early") is True
    handed_over.reset_caches()

    real_read = handed_over._read
    published: list[object] = []

    def _read_then_let_a_handoff_land():
        answer = real_read()
        # Put the real reader back BEFORE the overlapping write: the write takes the
        # writer's lock and reads through it, so leaving this hook installed would
        # re-enter it under a lock it already holds.
        monkeypatch.setattr(handed_over, "_read", real_read)
        # The handoff the reader did not see, landing while it holds its answer.
        assert handed_over.record_handed_over("chat-late") is True
        published.append(handed_over._snapshot)
        return answer

    monkeypatch.setattr(handed_over, "_read", _read_then_let_a_handoff_land)
    stale = handed_over.handed_over_slots()

    assert stale.slots == {"chat-early"}, "the reader answers from what it read"
    assert published and published[0].slots == {"chat-early", "chat-late"}
    assert handed_over._snapshot == published[0], "and the reader did not put its own back"

    assert handed_over.handed_over_slots().names("chat-late"), "both handoffs stand"


def test_a_read_with_nothing_published_does_not_start_publishing(monkeypatch):
    """A process that has recorded nothing keeps reading the file, and that is the cost.

    It is a small bounded file, and a hydrate takes ONE read for its whole pass and
    passes the answer down, so the price of never publishing from a read is one read per
    pass. What it buys is that the race above cannot exist.
    """
    _write_index("chat-worker\n")
    reads: list[int] = []
    real_read = handed_over._read

    def _counted():
        reads.append(1)
        return real_read()

    monkeypatch.setattr(handed_over, "_read", _counted)
    for _ in range(3):
        assert handed_over.handed_over_slots().names("chat-worker")

    assert len(reads) == 3, "a reader does not install a cache for the next one"
    assert handed_over._snapshot is None, "nothing was published"


def test_the_writer_publishes_so_it_need_not_re_read(monkeypatch):
    """The one publish there is. After a write this process knows the file's contents,
    so the next writer compares against memory instead of reading again."""
    assert handed_over.record_handed_over("chat-a") is True
    assert handed_over._snapshot == handed_over.HandoffIndex(frozenset({"chat-a"}), True)

    def _never(*args, **kwargs):
        raise AssertionError("the writer re-read a file it had just published")

    monkeypatch.setattr(handed_over, "_read", _never)
    assert handed_over.record_handed_over("chat-b") is True
    assert handed_over._snapshot.slots == {"chat-a", "chat-b"}


def test_the_writer_publishes_a_whole_snapshot_and_a_reader_takes_no_lock():
    """A lock-free reader sees the previous complete answer or the new one, never a
    half-built set, which is what lets the loop-affine hydrate read this without waiting
    on a writer that holds its lock across an fsync.
    """
    assert handed_over.record_handed_over("chat-a") is True
    first = handed_over.handed_over_slots()
    assert handed_over.record_handed_over("chat-b") is True
    second = handed_over.handed_over_slots()

    assert first == (frozenset({"chat-a"}), True), "the earlier snapshot was not mutated"
    assert second == (frozenset({"chat-a", "chat-b"}), True)
    assert isinstance(second[0], frozenset)


def test_a_reader_holds_no_lock_while_a_writer_does():
    """The property stated as a measurement rather than as a comment: the read answers
    while the write lock is held by somebody else. Sharing one lock here would park the
    event loop for the length of a handoff's disk sync."""
    assert handed_over.record_handed_over("chat-worker") is True

    with handed_over._write_lock:
        assert handed_over.handed_over_slots() == (frozenset({"chat-worker"}), True)


# ── read once, off the loop, and pass the answer down ───────────────────────


def test_the_codec_reads_no_file_to_answer_whether_a_slot_was_handed_over(tmp_path, monkeypatch):
    """The apply half takes the index as a VALUE, and a read there is a bug.

    ``metadata_codec.apply`` is loop-affine: it builds the slot, and slot construction
    broadcasts through ``Queue.put_nowait``. A file read inside it is therefore a
    blocking read on the event loop, once per restored tab -- and when the index cannot
    be read, once per tab forever, because an undecided answer is deliberately not
    kept. Both halves of that cost go away when the answer is a parameter.

    Driven by making the module's own read raise if anything calls it during the apply.
    """
    from chat_test_helpers import _make_state

    from kiro_crew.dashboard.slot_persistence import metadata_codec as codec

    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-built")

    def _never():
        raise AssertionError("the codec read the index instead of taking it as a value")

    monkeypatch.setattr(handed_over, "handed_over_slots", _never)
    monkeypatch.setattr(handed_over, "_read", _never)

    codec.apply(
        state,
        slot,
        {"created_by": "chat-lead"},
        codec.Restore(name="chat-built"),
        handoff_index=handed_over.HandoffIndex(frozenset({"chat-built"}), True),
    )

    assert slot._manager_unresolved is True, "and the value it was handed decided it"


def test_an_undecided_value_hydrates_every_slot_unresolved():
    """What a caller hands in when ITS read failed, and what that has to mean.

    The empty set is the unsafe answer here: it says nothing was ever handed over,
    which serves every slot from its creator -- including the one whose grant the log
    recorded going away. So undecided means unresolved for every key, whether or not the
    key is in the (empty) set.
    """
    assert handed_over.UNDECIDED.decided is False
    assert handed_over.UNDECIDED.names("chat-anything") is False
    assert handed_over.UNDECIDED.hydrates_unresolved("chat-anything") is True
    assert handed_over.UNDECIDED.hydrates_unresolved("") is True

    decided = handed_over.HandoffIndex(frozenset({"chat-handed"}), True)
    assert decided.hydrates_unresolved("chat-handed") is True
    assert decided.hydrates_unresolved("chat-never") is False


#: Every function in ``src`` that builds a slot from a metadata line, with the keyword
#: each takes the index through. Loop-affine, all of them: they reach
#: ``get_or_create_slot``, which broadcasts through ``Queue.put_nowait``.
_SLOT_BUILDERS = {
    "apply": "handoff_index",
    "_rehydrate_slot_from_history": "_prefetched_handoff_index",
    "_apply_recent_session": "handoff_index",
    "_apply_restored_open_slot": "handoff_index",
    "_hydrate_slot_from_history": "handoff_index",
    "_materialise_slot_from_history": "handoff_index",
}

#: The named reads. Each is reached from a worker thread or from a synchronous caller,
#: never from the loop, and each exists so the lookup it makes is not written inside a
#: build.
_INDEX_READS = {"_read_handoff_index", "handed_over_slots"}


def _source_of(module) -> "tuple[str, str]":
    """``(label, source)`` for *module*, rooted at the module rather than the CWD.

    A relative path here would resolve against whatever directory the runner started
    in, which is not the repository for every runner.
    """
    import pathlib

    path = pathlib.Path(module.__file__)
    return (f"{path.parent.name}/{path.name}", path.read_text(encoding="utf-8"))


def _build_modules() -> list:
    """Every module in ``src`` that holds a slot build. Imported, not path-globbed."""
    from kiro_crew.dashboard import chat_persistence, session_transfer
    from kiro_crew.dashboard.chat_api import resume
    from kiro_crew.dashboard.slot_persistence import metadata_codec

    return [metadata_codec, chat_persistence, resume, session_transfer]


def _owning_function(tree) -> "dict[int, str]":
    import ast

    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                owner[getattr(inner, "lineno", -1)] = node.name
    return owner


def test_the_codec_is_the_one_build_that_cannot_read_the_index_at_all():
    """``apply`` has no fallback, by design, so it cannot start to have one.

    Everything above it may hold a fallback for a synchronous caller; the codec may
    not, because it is the one build that runs for EVERY slot of every pass. A read
    there is the per-slot, on-loop read this whole shape exists to remove, and a
    default would be how a new caller got one by saying nothing.
    """
    import ast
    import inspect

    from kiro_crew.dashboard.slot_persistence import metadata_codec as codec

    #: Required, not defaulted: both possible defaults are wrong, so there is none.
    signature = inspect.signature(codec.apply)
    assert signature.parameters["handoff_index"].default is inspect.Parameter.empty

    _, source = _source_of(codec)
    owner = _owning_function(ast.parse(source))
    reads = [
        f"apply:{node.lineno}"
        for node in ast.walk(ast.parse(source))
        if owner.get(getattr(node, "lineno", -1)) == "apply"
        and (
            (
                isinstance(node, ast.Call)
                and (getattr(node.func, "id", "") or getattr(node.func, "attr", "")) in _INDEX_READS
            )
            or (
                isinstance(node, (ast.Import, ast.ImportFrom))
                and "handed_over" in ((getattr(node, "module", "") or ""))
            )
        )
    ]
    assert reads == [], f"the codec reached the index itself at {reads}"


def test_every_slot_build_in_src_is_handed_the_index_rather_than_reading_it():
    """Counted, because this is the third round of findings from one lazy read.

    Each previous attempt looked local and correct at the site that made it. The rule
    that holds instead is structural and about CALL SITES: anything that builds a slot
    from a line is given the answer by its caller, so the fallback a build keeps for a
    synchronous caller is never the path a driver takes. An async driver that forgot the
    keyword would fall back to reading -- on the loop, once per slot -- and look fine.

    So every call to a slot build inside ``src`` names the index. Zero omissions, not a
    list of approved ones.
    """
    import ast

    missing: list[str] = []
    for module in _build_modules():
        where, source = _source_of(module)
        tree = ast.parse(source)
        owner = _owning_function(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            keyword = _SLOT_BUILDERS.get(called)
            if keyword is None:
                continue
            at = owner.get(node.lineno, "<module>")
            if at == called:
                # A build's own recursion or its docstring-level self-reference.
                continue
            if not any(kw.arg == keyword for kw in node.keywords):
                missing.append(f"{where}::{at}:{node.lineno} -> {called}()")

    assert missing == [], (
        "these calls build a slot without naming the handed-over index: "
        f"{missing}. The callee then falls back to reading it, which on a driver is a "
        f"file read per slot on the event loop. Read it once (one of {sorted(_INDEX_READS)}) "
        "in the caller's own worker-thread step and pass it through."
    )
