"""A handoff whose commit refuses puts the parent back, and the window settles alone.

The two halves of a handoff are the tree edge and the manager, and this file is about
what happens when only the first of them lands. ``_reparent_target`` appends the edge and
waits for it to be on disk BEFORE the commit runs, which is the right order -- a field
written first and an append that then failed would leave the gateway acting on a move no
record carries. The cost of that order is this file's subject: between the two there is a
window where the edge is durable and the manager has not moved.

A refusal inside that window is not a tidy no-op if it is left alone. The target hangs
under the recipient while the giver is still its manager, so the recipient cannot drive
it and the giver's own ``session_release`` answers ``not_parent`` -- the move is neither
done nor undoable by either session, and only an operator editing state could separate
them.

So the refusal is compensated: the previous parent is restored, durably, before the
caller is told. The log is append-only, so "restored" is another edge stated on top and
the fold prefers the newest -- both moves stay readable.

The second half of the file is the cancellation rule. The window runs in its own task
behind a shield, because a request that goes away must not take a half-finished handoff
with it: the entry is already queued and the commit may already have published. Cancelling
the caller cancels the WAIT and not the work.
"""

from __future__ import annotations

import asyncio
import functools

import pytest
from chat_test_helpers import _make_state
from test_session_control_tree import (  # noqa: F401 -- imported FIXTURES apply here
    _enabled,
    _hold,
    _key,
    _live,
    _run,
    _slot,
    _tree_on,
)

from kiro_crew.crew_log import emit, handed_over
from kiro_crew.dashboard import session_control as sc
from kiro_crew.testing.wait import async_wait_until, await_until

#: The REAL marker writer, captured at import -- before the autouse fixture below stubs
#: ``sc._record_handed_over`` for every test -- so the tests that are about the writer
#: itself exercise it rather than the stand-in.
_REAL_RECORD_HANDED_OVER = sc._record_handed_over


@pytest.fixture(autouse=True)
def _saves(_floor_monkeypatch):
    """Report every slot save as landed and record which slots were saved.

    The same floor the handoff suite uses, for the same reason: what these tests are
    about is the ORDER of the durable writes, so the cache write is stubbed and observed.
    """
    saved: list[str] = []

    async def _save(state, slot, **kwargs):
        saved.append(slot.key)
        return True

    async def _mark(state, slot):
        # The INDEX entry is the commit's second durable write, after the log entry.
        # Recorded in the same list so a test can read the order of both -- and it
        # writes the real index, because that index is the authority every resolve
        # consults and a stub that skipped it would make every slot read as
        # never-handed-over.
        saved.append(f"index:{slot.key}")
        return bool(handed_over.record_handed_over(slot.key))

    _floor_monkeypatch.setattr(sc, "save_slot_off_loop", _save)
    _floor_monkeypatch.setattr(sc, "_record_handed_over", _mark)
    return saved


@pytest.fixture(autouse=True)
def _own_handoff_index(tmp_path, _floor_monkeypatch):
    """An index of this suite's own, and a cache that starts empty.

    The handed-over index is the AUTHORITY for whether a slot was ever handed over, so
    a suite sharing one with the operator's real store would read their handoffs and
    write its own into them. Autouse, because every test here resolves a manager.
    """
    _floor_monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "crew-home"))
    handed_over.reset_caches()
    yield
    handed_over.reset_caches()


def _handed_over(slot):
    """Record *slot* in the index, as a committed handoff does. Answers the slot."""
    assert handed_over.record_handed_over(slot.key), "the suite's own index must accept this"
    return slot


def _team(state):
    """A lead and a conductor, plus a worker the lead created. The dispatch shape."""
    owner = _live(state, _slot(state, "chat-owner-tab"), "sid-owner")
    lead = _live(state, _slot(state, "chat-lead"), "sid-lead")
    conductor = _live(state, _slot(state, "chat-conductor"), "sid-conductor")
    worker = _live(state, _slot(state, "chat-worker"), "sid-worker")
    lead._created_by = owner.key
    conductor._created_by = lead.key
    worker._created_by = lead.key
    return lead, conductor, worker


def _entries(monkeypatch, *, manager_lands: bool = True) -> list[dict]:
    """Record every tree and manager entry the verb emits, IN ORDER.

    All three emitters, which is the point: the compensation is itself an entry, so a
    recorder that watched only the manager write could not tell a restored parent from an
    abandoned one. The order is what the assertions read -- the edge, then the manager,
    then (only on a refusal) the edge that puts the parent back.

    Each stub settles its own append, because the verb waits on it. ``manager_lands``
    stands in for a manager entry the writer gave up on, which is the second way a commit
    refuses after the edge is already durable.
    """
    calls: list[dict] = []

    def _record(op: str, sid: str, kw: dict, landed: bool) -> None:
        settle = kw.pop("on_settled", None)
        calls.append({"op": op, "sid": sid, **kw})
        if settle is not None:
            settle(landed)

    monkeypatch.setattr(
        emit, "on_session_adopted", lambda sid, **kw: _record("adopt", sid, kw, True)
    )
    monkeypatch.setattr(
        emit, "on_session_released", lambda sid, **kw: _record("release", sid, kw, True)
    )
    monkeypatch.setattr(
        emit, "on_manager_set", lambda sid, **kw: _record("manager", sid, kw, manager_lands)
    )
    return calls


def _refuse_recipient_once(state, conductor, calls: list[dict]) -> None:
    """Close the recipient from the EDGE's settle callback.

    That callback is the suspension between the last gate and the commit, which is where
    a recipient really can go: its own routes take its slot lock and not this verb's. It
    is the cheapest honest way to reach the commit's refusal with the edge already down.
    """
    original = emit.on_session_adopted

    def _close_then_settle(sid, **kw):
        if not calls:
            state._slots.pop(conductor.key, None)
        original(sid, **kw)

    emit.on_session_adopted = _close_then_settle


def _refuse_recipient_after_the_manager_append(state, conductor, calls: list[dict]) -> None:
    """Close the recipient from the MANAGER entry's settle callback.

    A later suspension than :func:`_refuse_recipient_once` reaches, and the one the
    check before the append cannot cover: the manager entry is on disk by the time this
    runs, so the only thing standing between a replaced recipient and the reach is a
    check made AFTER the append. The recipient's own routes take its slot lock and not
    this verb's, which is what makes this window real rather than contrived.
    """
    original = emit.on_manager_set

    def _close_then_settle(sid, **kw):
        if not any(call["op"] == "manager" for call in calls):
            state._slots.pop(conductor.key, None)
        original(sid, **kw)

    emit.on_manager_set = _close_then_settle


def _undecided_manager_append(monkeypatch) -> None:
    """Make the handoff's own manager append answer UNDECIDED, once.

    Stubbed at :func:`~kiro_crew.crew_log.emit.awaiting_settlement` rather than by
    withholding a settle for the real bound, because the bound is seconds and what is
    under test here is not how long the wait is. The entry is still QUEUED through the
    recorder, with nothing settling it, which is exactly the state the undecided answer
    describes: an append that may land at any time afterwards.

    Only the first call is answered this way. The correction that follows goes through
    the real wait, so what the assertions read is the correction's own outcome.
    """
    real = emit.awaiting_settlement
    answered: list[str] = []

    async def _settlement(emit_one, *, what, **kwargs):
        if not answered:
            answered.append(what)
            emit_one(lambda _wrote: None)
            return None
        return await real(emit_one, what=what, **kwargs)

    monkeypatch.setattr(emit, "awaiting_settlement", _settlement)


# -- a refused commit puts the parent back ------------------------------------


def test_a_refused_commit_restores_the_previous_parent(tmp_path, monkeypatch):
    """The finding itself. The edge is durable, the commit refuses, and the parent the
    target had is put back by a further entry -- so the tree and the manager agree again
    and the giver's own release still works.

    Without the compensation the last entry on disk would name the conductor as the
    parent while the lead was still the manager, and neither session could undo it.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _refuse_recipient_once(state, conductor, calls)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced"
    assert [call["op"] for call in calls] == ["adopt", "adopt"], "the edge, then the undo"
    undo = calls[-1]
    assert undo["slot"] == worker.key
    assert undo["parent_slot"] == lead.key, "back under the parent it had"
    assert undo["previous_parent_slot"] == conductor.key, "and it says what it came from"
    assert worker._manager == "", "the manager never moved, so nothing to take back"


def test_a_refused_commit_releases_a_target_that_had_no_parent(tmp_path, monkeypatch):
    """Putting a root target back is a RELEASE, and that is not a detail.

    An adoption naming an empty parent is not an edge at all and the emitter refuses to
    write one, so a compensation that only ever adopted would have left the root case
    silently uncompensated -- the one case where the undo matters most, because there is
    no earlier parent to notice the target is missing.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    # No ``_hold``: the worker is a root, which is what a dispatched session is until
    # something adopts it.
    _refuse_recipient_once(state, conductor, calls)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced"
    assert [call["op"] for call in calls] == ["adopt", "release"], "the edge, then the undo"
    undo = calls[-1]
    assert undo["slot"] == worker.key
    assert undo["previous_parent_slot"] == conductor.key, "released from where it was put"
    assert worker._manager == ""


def test_a_manager_entry_that_never_landed_also_restores_the_parent(tmp_path, monkeypatch):
    """The second way the commit refuses after the edge is down: the manager entry is the
    commit point, so a writer that gave up on it means the handoff did not happen -- and
    the edge that was ordered before it has to come back out."""
    calls = _entries(monkeypatch, manager_lands=False)
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target="chat-worker", to="chat-conductor"
            )
        )

    assert exc.value.code == "handoff_not_recorded"
    assert [call["op"] for call in calls] == ["adopt", "manager", "adopt"]
    assert calls[-1]["parent_slot"] == lead.key
    assert worker._manager == "", "nothing is published on the strength of a dropped append"


def test_a_committed_handoff_leaves_the_parent_where_it_moved(tmp_path, monkeypatch, _saves):
    """Control. With nothing refusing there is NO undo entry -- a compensation that fired
    on the success path would hand every session straight back and the verb would do
    nothing at all."""
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    result = _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )

    assert [call["op"] for call in calls] == ["adopt", "manager"], "no undo on success"
    assert result["manager"] == conductor.key
    assert result["previous_manager"] == lead.key
    assert worker._manager == conductor.key
    assert _saves == [f"index:{worker.key}"], "two entries and one index write, no slot save"


def test_a_compensation_that_cannot_be_written_still_reports_the_original_refusal(
    tmp_path, monkeypatch
):
    """The undo is bounded and non-raising. A caller that asked for a handoff must be told
    why the HANDOFF failed, not why the undo did -- the undo's own failure is an operator
    problem and goes to the log, because no retry by this caller fixes it."""
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _refuse_recipient_once(state, conductor, calls)

    def _undo_explodes(sid, **kw):
        if calls:
            raise OSError("the compensating append could not be queued")
        raise AssertionError("the first edge must still be emitted")

    original = emit.on_session_adopted

    def _dispatch(sid, **kw):
        if calls:
            _undo_explodes(sid, **kw)
        original(sid, **kw)

    monkeypatch.setattr(emit, "on_session_adopted", _dispatch)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced", "the original refusal, not the undo's"
    assert worker._manager == ""


# -- the commit window settles independently of the caller --------------------


def test_the_slot_stays_fenced_across_the_commit(tmp_path, monkeypatch):
    """No other verb may decide about the slot inside the window.

    The append's own fence clears when the append settles, which is exactly when the
    window OPENS -- so without a hold of its own the slot would be unfenced for the one
    stretch where its recorded parent and its manager disagree. Read from inside the
    commit, which is the only place the question can be asked.
    """
    _entries(monkeypatch)
    fenced: list[tuple[str, bool]] = []
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    original = emit.on_manager_set

    def _held(key: str) -> bool:
        pending = sc._tree_pending_slots.get(key)
        return pending is not None and not pending.done()

    def _check_then_emit(sid, **kw):
        # Observed at the commit point, which is the moment that matters: the edge is on
        # disk, the manager has not moved yet, and the two disagree until this lands.
        fenced.append(("target", _held(worker.key)))
        fenced.append(("recipient", _held(conductor.key)))
        original(sid, **kw)

    monkeypatch.setattr(emit, "on_manager_set", _check_then_emit)

    _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )

    assert fenced == [("target", True), ("recipient", True)], "BOTH endpoints, at the commit"
    assert worker.key not in sc._tree_pending_slots, "and released when the window closes"
    assert conductor.key not in sc._tree_pending_slots


def test_a_cancelled_caller_does_not_abandon_the_commit(tmp_path, monkeypatch):
    """The cancellation rule. The caller goes away mid-commit -- a disconnect, a tool
    timeout -- and the window finishes anyway: the entry is queued and the manager is
    published, so stopping halfway is what would leave the two halves apart.

    Driven at the manager entry, which is the commit point: the settle is withheld, the
    caller is cancelled while waiting on it, and only then does the entry land.
    """
    pushes: list[int] = []
    settles: list = []
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    monkeypatch.setattr(state, "push_slots_update", lambda: pushes.append(1))

    monkeypatch.setattr(
        emit,
        "on_session_adopted",
        lambda sid, **kw: (kw.get("on_settled") or (lambda _ok: None))(True),
    )

    def _withhold(sid, **kw):
        settles.append(kw.get("on_settled"))

    monkeypatch.setattr(emit, "on_manager_set", _withhold)

    async def _drive() -> None:
        call = asyncio.ensure_future(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
        # Until the commit has reached the withheld entry, which is where the caller is
        # parked when it is cancelled. Waited on the STATE the verb publishes rather than
        # a count of zero-length sleeps: the verb hops a thread more than once on the way
        # here (the gate's warm, the id resolution), and a thread hop needs wall time, so
        # a fixed number of ``sleep(0)`` passes runs out on a loaded machine and the test
        # fails for the one reason it is not about. The helper's bound is half the test's
        # timeout and only names a hang.
        await async_wait_until(
            lambda: bool(settles), describe=lambda: "the commit never reached the manager entry"
        )
        window = next(iter(sc._commit_windows))

        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert not window.done(), "the cancellation must not have taken the window with it"

        settles[0](True)
        await await_until(window, what="the commit window the cancelled caller left behind")

    _run(_drive())

    assert worker._manager == conductor.key, "the window published after the caller left"
    assert pushes == [1], "and the sidebar work it owed still ran"
    assert not sc._commit_windows, "the window is not held after it finishes"
    assert worker.key not in sc._tree_pending_slots, "and the slot is unfenced"
    assert conductor.key not in sc._tree_pending_slots


# -- a handoff refused AFTER its own entry landed is taken back on the log ----


def test_a_recipient_replaced_after_the_append_is_refused(tmp_path, monkeypatch):
    """The check that decides runs AFTER the manager entry, not before it.

    The append suspends, and the recipient can be closed, rebound or reopened inside that
    suspension by its own routes, which take its slot lock and not this verb's. A verdict
    formed before the append and acted on after it would move the reach to whatever sits
    at the key by then.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _refuse_recipient_after_the_manager_append(state, conductor, calls)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced"
    assert worker._manager == "", "nothing is published to a recipient that changed"


def test_a_handoff_refused_after_its_entry_landed_is_taken_back_on_the_log(tmp_path, monkeypatch):
    """The finding. The entry is the AUTHORITY, so a refusal that only declines to
    publish the field leaves the log still granting the reach -- and the next start
    publishes it, the gateway's refusal having lived in memory alone.

    The correction is another entry on top, because the log is append-only and the fold
    takes the newest statement per slot. This target had no manager before, so the
    correction is the revert: a correction naming the creator would instead record a
    handoff TO the creator, leaving the field set where it was empty.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _refuse_recipient_after_the_manager_append(state, conductor, calls)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced"
    assert [call["op"] for call in calls] == [
        "adopt",
        "manager",
        "manager",
        "adopt",
    ], "the edge, the manager entry, the manager taken back, then the edge put back"
    taken_back = calls[2]
    assert taken_back["slot"] == worker.key
    assert taken_back["manager"] == "", "no manager is named"
    assert taken_back["to_creator"] is True, "and the revert is what says so"
    assert taken_back["previous"] == conductor.key, "it records the handoff it undoes"


def test_the_correction_names_the_manager_the_handoff_replaced(tmp_path, monkeypatch):
    """A target already handed over once reverts to the manager it HAD, not to its
    creator. The two are different sessions, so a correction that always reverted to the
    creator would quietly take the earlier handoff away as well -- an undo of something
    nobody refused.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    # Already handed over once, so the FIRST manager is the session that may hand it on
    # again: ``handoff`` is one of the verbs a handoff widens, and the creator's own
    # reach went with the first transfer. The recipient is authorized on its own terms
    # too, so the giver has to be able to name it -- hence the conductor is this
    # session's own.
    first = _live(state, _slot(state, "chat-first-manager"), "sid-first-manager")
    first._created_by = lead.key
    conductor._created_by = first.key
    worker._manager = first.key
    worker._manager_tab = first._tab_id
    _refuse_recipient_after_the_manager_append(state, conductor, calls)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(first), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced"
    taken_back = calls[2]
    assert taken_back["manager"] == first.key, "back to the manager it had"
    assert taken_back["to_creator"] is False, "which is not a revert to the creator"
    assert worker._manager == first.key, "and the field never moved off it"
    # The OCCUPANCY goes back with the key. A correction that restored the name alone
    # would leave the record a bare name, and the next start would compare it against
    # whoever answers to that name then -- refusing the manager that is still live,
    # which is the whole reason the occupancy is on the entry.
    assert taken_back["manager_tab"] == first._tab_id


def test_an_undecided_manager_append_is_corrected_rather_than_abandoned(tmp_path, monkeypatch):
    """An append still QUEUED when the wait gives up is the case a two-valued answer
    cannot handle. It is not a refusal: it may land at any time afterwards, and it would
    then be the newest statement on the log -- so a start would publish a manager this
    verb refused.

    The correction is queued behind it rather than skipped. The writer is FIFO per unit,
    so whichever order the two land in, the newest statement is the correction.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _undecided_manager_append(monkeypatch)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "handoff_not_recorded"
    assert [call["op"] for call in calls] == ["adopt", "manager", "manager", "adopt"]
    assert calls[2]["to_creator"] is True, "the queued handoff is taken back"
    assert calls[2]["previous"] == conductor.key
    assert worker._manager == "", "and nothing is published on an undecided append"


# -- the settlement answer the correction is decided from ---------------------


def test_a_settled_append_answers_true_and_a_dropped_one_answers_false(tmp_path):
    """The two decided answers, which need no correction between them: one landed and one
    never will."""

    async def _both() -> tuple[bool | None, bool | None]:
        landed = await emit.awaiting_settlement(lambda s: s(True), what="a landed append")
        dropped = await emit.awaiting_settlement(lambda s: s(False), what="a dropped append")
        return landed, dropped

    assert _run(_both()) == (True, False)


def test_an_append_that_never_settles_answers_undecided(tmp_path):
    """The third answer, and the only one that obliges a caller to correct the log.

    Kept apart from False because the two differ in exactly what matters to a caller that
    has to undo: a dropped append left nothing behind, and this one may still land.
    """

    async def _never() -> bool | None:
        return await emit.awaiting_settlement(
            lambda _settle: None, what="an append nothing settles", timeout=0.05
        )

    assert _run(_never()) is None


def test_the_two_valued_wait_reads_undecided_as_not_recorded(tmp_path):
    """``awaiting_commit`` is the same wait for callers with nothing to take back, and it
    folds the two ways of not being on disk together -- which is safe only because those
    callers publish on the append rather than undo on its refusal."""

    async def _folded() -> bool:
        return await emit.awaiting_commit(
            lambda _settle: None, what="an append nothing settles", timeout=0.05
        )

    assert _run(_folded()) is False


# -- BOTH endpoints are fenced while the tree append is in flight -------------


def _withheld_tree_append(monkeypatch) -> "tuple[list[dict], list]":
    """Record the emitted entries and WITHHOLD the FIRST tree append's settle.

    Leaves the first move parked exactly where a slow writer parks it: the edge handed
    over, nothing on disk yet, and the projection still showing the tree as it was. That
    is the state every decision made meanwhile is made against, so it is the state the
    fence and the cycle check have to cover.

    Only the first, so a LATER move can still complete inside that window. A helper that
    withheld every append would make every second move fail for the one reason these
    tests are not about, and a guard that refused everything while anything was pending
    would pass just as well as one that reads the edges.
    """
    calls: list[dict] = []
    settles: list = []

    def _record(op: str, sid: str, kw: dict, landed: bool | None) -> None:
        settle = kw.pop("on_settled", None)
        calls.append({"op": op, "sid": sid, **kw})
        if landed is None:
            settles.append(settle)
        elif settle is not None:
            settle(landed)

    def _adopt(sid, **kw):
        # The first is held; every one after it settles, so the window stays open around
        # moves that are meant to be decided inside it.
        _record("adopt", sid, kw, None if not settles else True)

    monkeypatch.setattr(emit, "on_session_adopted", _adopt)
    monkeypatch.setattr(
        emit, "on_session_released", lambda sid, **kw: _record("release", sid, kw, True)
    )
    monkeypatch.setattr(emit, "on_manager_set", lambda sid, **kw: _record("manager", sid, kw, True))
    return calls, settles


def test_the_reverse_handoff_is_refused_while_the_first_append_is_in_flight(tmp_path, monkeypatch):
    """The cycle a target-only fence admits. A handoff's commit window outlives the
    request, so the edge can still be queued when the next verb is decided -- and the
    REVERSE handoff names the other endpoint as its target, so a fence on this one never
    sees it. ``is_ancestor`` then reads a tree the pending edge has not reached, admits
    the opposite move, and both entries land: a loop in an append-only log that no fold
    can present and no verb can undo.

    Fencing both endpoints refuses it instead, and ``tree_write_pending`` is the honest
    code -- the earlier write may still land, so the caller is told to read the tree and
    try again rather than that anything is wrong.
    """
    calls, settles = _withheld_tree_append(monkeypatch)
    monkeypatch.setattr(sc, "_TREE_APPEND_TIMEOUT", 0.05)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    async def _drive() -> None:
        first = asyncio.ensure_future(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
        # The caller's own wait runs out while the append is still queued, which is the
        # only way to reach the next verb with the window still running.
        with pytest.raises(sc.SessionControlError) as pending:
            await first
        assert pending.value.code == "tree_write_pending"
        assert settles, "the edge was handed to the writer"

        with pytest.raises(sc.SessionControlError) as reverse:
            await sc.handoff_target(
                state, caller_session_key=_key(lead), target=conductor.key, to=worker.key
            )
        assert (
            reverse.value.code == "tree_write_pending"
        ), "the recipient of the in-flight handoff is fenced too"
        assert [call["op"] for call in calls] == ["adopt"], "the reverse appended nothing"

        settles[0](True)
        for window in list(sc._commit_windows):
            await await_until(asyncio.shield(window), what="a commit window")

    _run(_drive())

    assert conductor.key not in sc._tree_pending_slots, "the parent fence is released"
    assert worker.key not in sc._tree_pending_slots


def test_a_handoff_the_caller_stopped_waiting_for_is_still_audited_and_broadcast(
    tmp_path, monkeypatch, _saves
):
    """The reach moves in the window, so the record of it has to be written there too.

    A caller whose wait runs out raises through ``_audit_denials``, which writes a
    ``denied`` line for that timeout. The window then commits and the manager really
    does move -- so an ``allowed`` line written in the caller's frame is never written
    at all, and the SEL is left saying the opposite of what happened. The sidebar is the
    same problem in a different place: a broadcast tied to the caller's frame never
    tells anyone the tree changed.
    """
    calls, settles = _withheld_tree_append(monkeypatch)
    monkeypatch.setattr(sc, "_TREE_APPEND_TIMEOUT", 0.05)
    audits: list[dict] = []
    monkeypatch.setattr(sc, "_audit", lambda **kw: audits.append(kw))
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    pushes: list[int] = []
    monkeypatch.setattr(state, "push_slots_update", lambda: pushes.append(1))

    async def _drive() -> None:
        with pytest.raises(sc.SessionControlError) as pending:
            await sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        assert pending.value.code == "tree_write_pending"
        assert [entry["outcome"] for entry in audits] == ["denied"], "the timeout is denied"
        assert not pushes, "and nothing is broadcast yet"

        settles[0](True)
        for window in list(sc._commit_windows):
            await await_until(asyncio.shield(window), what="a commit window")

    _run(_drive())

    assert worker._manager == conductor.key, "the window published after the caller left"
    outcomes = [entry["outcome"] for entry in audits]
    assert outcomes == ["denied", "allowed"], "and the SEL says the handoff happened"
    allowed = audits[-1]
    assert allowed["slot_key"] == worker.key
    assert allowed["operation"] == "handoff"
    assert pushes == [1], "the sidebar is told exactly once"


# -- the verb's own pre-lock refusals are audited -----------------------------


def test_a_handoff_that_changes_nothing_is_audited(tmp_path, monkeypatch):
    """``already_manager`` is the verb's OWN refusal and it is raised before the lock, so
    nothing on the success path records it. A denied attempt to move control of another
    session is exactly what the SEL is for -- a boundary with no trail is the one most
    worth a trail."""
    audits: list[dict] = []
    monkeypatch.setattr(sc, "_audit", lambda **kw: audits.append(kw))
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=worker.key
            )
        )

    assert exc.value.code == "already_manager"
    assert audits == [
        {
            "caller_session_key": _key(lead),
            "operation": "handoff",
            "slot_key": worker.key,
            "outcome": "denied",
            "detail": {"code": "already_manager"},
        }
    ]


def test_a_recipient_on_another_memory_store_is_audited(tmp_path, monkeypatch):
    """The recipient-side refusals are resolved before the lock as well, and they are the
    ones worth seeing most: each is an attempt to put a session under a principal that
    may not have it. Audited against the TARGET, which is the session whose control was
    being moved; the recipient is named in the refusal's own message."""
    audits: list[dict] = []
    monkeypatch.setattr(sc, "_audit", lambda **kw: audits.append(kw))
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    conductor.memory_store = "member-raymond"

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_other_store"
    assert [entry["outcome"] for entry in audits] == ["denied"]
    assert audits[0]["slot_key"] == worker.key, "attributed to the session being moved"
    assert audits[0]["detail"] == {"code": "recipient_other_store"}


# -- the cycle check sees QUEUED edges, not only settled ones -----------------


def _four_node_tree(state):
    """Four sessions and the two settled edges from the finding: ``D -> A``, ``B -> C``.

    All four are opened by the owner tab, so none of them is ownership-fenced and any of
    them may hand any other over -- which is what lets the sequence be driven at all. The
    shape is the point: the loop the fourth move closes runs through every one of them, so
    no two moves share an endpoint and a per-endpoint fence can never see it coming.
    """
    owner = _live(state, _slot(state, "chat-owner-tab"), "sid-owner")
    nodes = {}
    for name in ("a", "b", "c", "d"):
        slot = _live(state, _slot(state, f"chat-{name}"), f"sid-{name}")
        slot._created_by = owner.key
        nodes[name] = slot
    _hold(state, nodes["d"].key, nodes["a"].key, sid="sid-d")
    _hold(state, nodes["b"].key, nodes["c"].key, sid="sid-b")
    return owner, nodes


def test_a_queued_edge_is_part_of_the_cycle_check(tmp_path, monkeypatch):
    """The four-node loop a per-endpoint fence cannot refuse.

    ``D -> A`` and ``B -> C`` are settled. ``A -> B`` is handed to the writer and stays
    queued, so the projection does not carry it. A handoff of ``C`` to ``D`` then names
    neither ``A`` nor ``B``: both endpoint fences pass, and an ancestor check against the
    settled tree alone sees no loop. Let it through and the log holds
    ``A -> B -> C -> D -> A``, four valid append-only entries the fold can only mark.

    The fix is to check against the settled tree PLUS every queued edge, so the move is
    refused ``would_cycle`` on the view that includes the one edge still in flight.
    """
    calls, settles = _withheld_tree_append(monkeypatch)
    monkeypatch.setattr(sc, "_TREE_APPEND_TIMEOUT", 0.05)
    state = _make_state(tmp_path)
    owner, nodes = _four_node_tree(state)

    async def _drive() -> None:
        with pytest.raises(sc.SessionControlError) as pending:
            await sc.handoff_target(
                state,
                caller_session_key=_key(owner),
                target=nodes["a"].key,
                to=nodes["b"].key,
            )
        assert pending.value.code == "tree_write_pending", "A -> B is queued, not settled"
        assert sc._tree_pending_edges == {nodes["a"].key: nodes["b"].key}

        with pytest.raises(sc.SessionControlError) as closing:
            await sc.handoff_target(
                state,
                caller_session_key=_key(owner),
                target=nodes["c"].key,
                to=nodes["d"].key,
            )
        assert closing.value.code == "would_cycle", "the queued edge closes the loop"
        assert [call["op"] for call in calls] == ["adopt"], "the fourth move wrote nothing"

        settles[0](True)
        for window in list(sc._commit_windows):
            await await_until(asyncio.shield(window), what="a commit window")

    _run(_drive())

    assert sc._tree_pending_edges == {}, "the queued edge is forgotten once it settles"


def test_the_same_move_is_allowed_once_the_queued_edge_settles_elsewhere(tmp_path, monkeypatch):
    """The guard refuses a LOOP, not a coincidence of timing.

    Same four sessions and the same fourth move, with ``A -> B`` never queued: nothing is
    in flight, the settled tree holds no path from ``C`` up to ``D``, and the handoff goes
    through. Without this the previous test would pass just as well against a guard that
    refused every move while any append was pending.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    owner, nodes = _four_node_tree(state)
    assert sc._tree_pending_edges == {}

    result = _run(
        sc.handoff_target(
            state, caller_session_key=_key(owner), target=nodes["c"].key, to=nodes["d"].key
        )
    )

    assert result["manager"] == nodes["d"].key
    assert [call["op"] for call in calls] == ["adopt", "manager"], "it really moved"
    assert sc._tree_pending_edges == {}


def test_a_queued_edge_that_closes_no_loop_does_not_refuse(tmp_path, monkeypatch):
    """A pending edge is not a veto. ``A -> B`` is queued and a handoff of ``D`` to ``C``
    is decided in that window: the combined view carries the queued edge and still shows
    no path from ``D`` up to ``C``, so the move is allowed.

    This is the half that keeps the guard honest about what it refuses -- an
    overlay that made every concurrent move fail would satisfy the loop case and be
    useless.
    """
    calls, settles = _withheld_tree_append(monkeypatch)
    monkeypatch.setattr(sc, "_TREE_APPEND_TIMEOUT", 0.05)
    state = _make_state(tmp_path)
    owner, nodes = _four_node_tree(state)

    async def _drive() -> None:
        with pytest.raises(sc.SessionControlError):
            await sc.handoff_target(
                state,
                caller_session_key=_key(owner),
                target=nodes["a"].key,
                to=nodes["b"].key,
            )
        assert sc._tree_pending_edges == {nodes["a"].key: nodes["b"].key}

        # D is under A; putting it under C adds D -> C, and C's own chain is C (B -> C is
        # the other direction), so nothing closes.
        moved = await sc.handoff_target(
            state, caller_session_key=_key(owner), target=nodes["d"].key, to=nodes["c"].key
        )
        assert moved["manager"] == nodes["c"].key
        # Asserted HERE, while the first edge is still queued: this is the move that had
        # to be allowed inside the window, and once the first one settles its own commit
        # adds a second manager entry that says nothing about this.
        assert [call["op"] for call in calls].count("manager") == 1, "it committed"
        assert nodes["d"]._manager == nodes["c"].key

        settles[0](True)
        for window in list(sc._commit_windows):
            await await_until(asyncio.shield(window), what="a commit window")

    _run(_drive())

    assert sc._tree_pending_edges == {}, "both edges are forgotten once they settle"


def test_a_pending_set_that_cannot_be_read_refuses(tmp_path, monkeypatch):
    """FAIL CLOSED on the view itself. A cycle check is only as good as the tree it ran
    against, so "no loop" concluded from a view that could not be built is a confident
    wrong answer rather than a missing one -- the same rule this whole PR applies to an
    unreadable log.
    """
    _entries(monkeypatch)
    state = _make_state(tmp_path)
    owner, nodes = _four_node_tree(state)

    def _unreadable(_nodes):
        raise RuntimeError("the queued edges could not be folded in")

    monkeypatch.setattr(sc, "_tree_with_pending_edges", _unreadable)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(owner), target=nodes["c"].key, to=nodes["d"].key
            )
        )

    assert exc.value.code == "tree_not_ready"
    assert exc.value.status == 409, "retryable: the next attempt may read it"


def test_no_reachable_sequence_writes_a_cycle_to_the_log(tmp_path, monkeypatch):
    """The property the three cases above are instances of, asserted over the whole
    four-node shape: every ORDERED pair of moves is attempted, and whatever the verb
    allows, the edges that reach the log never form a loop.

    Driven against the real fold rather than a hand-checked list, because the thing worth
    pinning is not which pairs refuse but that no pair persists a cycle -- an append-only
    log cannot take one back.
    """

    def _loops(edges: dict[str, str]) -> bool:
        """Whether following ``child -> parent`` from any node ever revisits a node.

        Written out rather than taken from ``is_ancestor``, which answers a different
        question and answers it False for ``candidate == of`` by contract -- so asking it
        whether a slot is its own ancestor can never report a loop.
        """
        for start in edges:
            seen: set[str] = set()
            cursor = start
            while cursor in edges:
                if cursor in seen:
                    return True
                seen.add(cursor)
                cursor = edges[cursor]
        return False

    pairs = [("a", "b"), ("c", "d"), ("d", "c"), ("b", "a"), ("a", "c"), ("b", "d")]
    for first, second in [(x, y) for x in pairs for y in pairs if x != y]:
        calls, settles = _withheld_tree_append(monkeypatch)
        monkeypatch.setattr(sc, "_TREE_APPEND_TIMEOUT", 0.05)
        state = _make_state(tmp_path)
        owner, nodes = _four_node_tree(state)
        edges = {nodes["d"].key: nodes["a"].key, nodes["b"].key: nodes["c"].key}

        async def _drive() -> None:
            for child, parent in (first, second):
                try:
                    await sc.handoff_target(
                        state,
                        caller_session_key=_key(owner),
                        target=nodes[child].key,
                        to=nodes[parent].key,
                    )
                except sc.SessionControlError:
                    # A refusal the CALLER saw says nothing about the log: an append
                    # handed over before the refusal still lands. What reached the log is
                    # read off the emissions below, which is the whole shape of this bug.
                    pass
            for settle in settles:
                if settle is not None:
                    settle(True)
            for window in list(sc._commit_windows):
                try:
                    await await_until(asyncio.shield(window), what="a commit window")
                except sc.SessionControlError:
                    pass

        _run(_drive())

        # Folded from what was EMITTED, in order, the way the tree folds it: an adoption
        # sets the parent and a release clears it, newest entry winning. This is the tree
        # a cold start would rebuild, including every compensation.
        for call in calls:
            if call["op"] == "adopt":
                edges[call["slot"]] = call["parent_slot"]
            elif call["op"] == "release":
                edges.pop(call["slot"], None)

        assert not _loops(edges), f"{first} then {second} persisted a loop: {edges}"
        sc._tree_pending_edges.clear()
        sc._commit_windows.clear()


# -- a rollback is a forward write and is checked like one --------------------


def _loop_waiting_for_the_rollback(state):
    """The tree a compensation of ``A`` meets after the window ran: ``A -> B``,
    ``R -> A``, ``P -> R``.

    No cycle in it -- each edge was checked and each was fine -- and yet putting ``A``
    back under ``P``, the parent it had when the verb started, closes
    ``A -> P -> R -> A``. That is the shape the rollback has to refuse, and the reason it
    cannot be read off the tree the verb started from: the previous parent moved BELOW
    the target while the window was running.
    """
    owner = _live(state, _slot(state, "chat-owner-tab"), "sid-owner")
    nodes = {}
    for name in ("a", "p", "r", "b"):
        slot = _live(state, _slot(state, f"chat-{name}"), f"sid-{name}")
        slot._created_by = owner.key
        nodes[name] = slot
    _hold(state, nodes["a"].key, nodes["b"].key, at=300, sid="sid-a")
    _hold(state, nodes["r"].key, nodes["a"].key, sid="sid-r")
    _hold(state, nodes["p"].key, nodes["r"].key, at=200, sid="sid-p")
    return owner, nodes


def test_a_rollback_that_would_cycle_is_recognised(tmp_path, monkeypatch):
    """The rollback is checked against the same settled-plus-queued view the forward path
    uses. Asserted on the tree above, where no single edge is a loop and the restoration
    is the one that would complete it."""
    state = _make_state(tmp_path)
    _owner, nodes = _loop_waiting_for_the_rollback(state)

    assert sc._restoring_would_cycle(nodes["a"].key, nodes["p"].key) is True
    # The other direction, so the check is not simply answering True: B is above A, and
    # putting A back under B closes nothing.
    assert sc._restoring_would_cycle(nodes["a"].key, nodes["b"].key) is False


def test_a_rollback_that_cannot_be_checked_is_treated_as_cyclic(tmp_path, monkeypatch):
    """FAIL CLOSED, and for a compensation that means answering True. A view this cannot
    build is a view that cannot clear the edge, and the alternative -- releasing to the
    root -- loses nesting rather than persisting a loop. That is the opposite direction
    from the forward path, which refuses the operation, because a compensation has no
    refusal available: it IS the undo.
    """
    state = _make_state(tmp_path)
    _owner, nodes = _loop_waiting_for_the_rollback(state)

    def _unreadable(_nodes):
        raise RuntimeError("the queued edges could not be folded in")

    monkeypatch.setattr(sc, "_tree_with_pending_edges", _unreadable)
    assert sc._restoring_would_cycle(nodes["a"].key, nodes["b"].key) is True


def test_the_compensation_releases_instead_of_restoring_into_a_loop(tmp_path, monkeypatch):
    """The whole rollback, driven through the helper the window calls: the target is
    RELEASED to the root rather than adopted back into the cycle.

    Without this the compensation is the forward path missing its guard -- it would state
    ``A -> P`` on an append-only log, and four valid entries later the fold has a loop it
    can only mark while no verb undoes it.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    _owner, nodes = _loop_waiting_for_the_rollback(state)

    landed = _run(
        sc._restore_parent(
            target_sid="sid-a",
            slot_key=nodes["a"].key,
            parent_key=nodes["b"].key,
            parent_sid="sid-b",
            previous_parent=nodes["p"].key,
            previous_parent_sid="sid-p",
        )
    )

    assert landed is True
    assert [call["op"] for call in calls] == ["release"], "released, not adopted back"
    assert calls[0]["slot"] == nodes["a"].key
    assert calls[0]["previous_parent_slot"] == nodes["b"].key, "it says where it came from"


def test_a_rollback_with_no_loop_still_restores_the_parent(tmp_path, monkeypatch):
    """The guard refuses a loop, not a restoration. With the previous parent still ABOVE
    the target, the compensation adopts it back exactly as before -- otherwise every
    refused move would quietly flatten its target to the root."""
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    _owner, nodes = _loop_waiting_for_the_rollback(state)

    landed = _run(
        sc._restore_parent(
            target_sid="sid-a",
            slot_key=nodes["a"].key,
            parent_key=nodes["r"].key,
            parent_sid="sid-r",
            previous_parent=nodes["b"].key,
            previous_parent_sid="sid-b",
        )
    )

    assert landed is True
    assert [call["op"] for call in calls] == ["adopt"]
    assert calls[0]["parent_slot"] == nodes["b"].key, "back under the parent it had"


# -- the commit admission is the admission a restart re-applies ---------------


def test_a_recipient_on_another_store_than_the_target_is_refused(tmp_path, monkeypatch):
    """The sharpest of the store rules: the one a RESTART will apply.

    The commit compared GIVER against RECIPIENT, and the rebuild compares RECIPIENT
    against the session it manages. Three sessions make those disagree -- an
    owner-rooted lead on the default store dispatches a member-store worker and a
    default-store conductor -- so a handoff is accepted now and silently revoked at the
    next start, which is worse than refusing it up front. Both comparisons are made at
    the verb, so the answer the caller gets is the answer the next boot will give.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    lead.memory_store = ""
    conductor.memory_store = ""
    worker.memory_store = "member-raymond"

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_other_store"
    assert "different memory stores" in exc.value.message
    assert calls == [], "no grant is appended for a handoff that would not survive"
    assert worker._manager == ""


def test_the_stores_are_re_read_immediately_before_publication(tmp_path, monkeypatch):
    """The append suspends, so neither reading survives from the gate. A target rebound
    inside that window is refused at the publication check rather than published on a
    verdict formed before the change."""
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    original = emit.on_manager_set

    def _rebind_then_settle(sid, **kw):
        if not any(call["op"] == "manager" for call in calls):
            worker.memory_store = "member-raymond"
        original(sid, **kw)

    emit.on_manager_set = _rebind_then_settle

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced"
    assert worker._manager == "", "nothing is published across a store that moved"
    assert [call["op"] for call in calls] == [
        "adopt",
        "manager",
        "manager",
        "adopt",
    ], "the grant is taken back on the log and the parent put back"


# -- the manager reconstruction is off the boot path and gated ----------------


def test_a_queued_rollback_is_announced_to_the_next_mover(tmp_path, monkeypatch):
    """The mirror image of the rollback's own cycle guard. Checking the restoration
    against the queued edges while not PUBLISHING it among them leaves the loop reachable
    from the other side: the next mover's cycle check reads a tree without the edge this
    compensation is about to write, admits a move that closes against it, and both land.

    So the rollback takes a settlement-backed waiter like any forward append -- it fences
    the slot and publishes ``slot -> previous parent`` -- and both survive a wait that
    runs out, because an append still queued is exactly the one a later decision must see.
    """
    settles: list = []

    def _withhold(sid, **kw):
        settles.append(kw.get("on_settled"))

    monkeypatch.setattr(emit, "on_session_adopted", _withhold)
    monkeypatch.setattr(emit, "COMMIT_WAIT_SECONDS", 0.05)
    state = _make_state(tmp_path)
    _owner, nodes = _loop_waiting_for_the_rollback(state)

    async def _drive() -> None:
        landed = await sc._restore_parent(
            target_sid="sid-a",
            slot_key=nodes["a"].key,
            parent_key=nodes["r"].key,
            parent_sid="sid-r",
            previous_parent=nodes["b"].key,
            previous_parent_sid="sid-b",
        )
        assert landed is False, "nothing is on disk while the append is still queued"
        assert sc._tree_pending_edges == {
            nodes["a"].key: nodes["b"].key
        }, "the rollback edge is published for the next mover's cycle check"
        pending = sc._tree_pending_slots.get(nodes["a"].key)
        assert pending is not None and not pending.done(), "and the slot stays fenced"

        settles[0](True)
        await asyncio.sleep(0)

    _run(_drive())

    assert sc._tree_pending_edges == {}, "both clear when the writer settles, not before"
    assert nodes["a"].key not in sc._tree_pending_slots


def test_a_queued_rollback_blocks_the_move_that_would_close_on_it(tmp_path, monkeypatch):
    """What the announcement buys, end to end: while the restoration of ``A`` under ``B``
    is queued, a handoff that would put ``B`` under ``A`` is refused ``would_cycle``
    rather than decided against a tree missing that edge."""
    settles: list = []

    def _withhold(sid, **kw):
        settles.append(kw.get("on_settled"))

    monkeypatch.setattr(emit, "on_session_adopted", _withhold)
    monkeypatch.setattr(emit, "COMMIT_WAIT_SECONDS", 0.05)
    state = _make_state(tmp_path)
    owner, nodes = _loop_waiting_for_the_rollback(state)

    async def _drive() -> None:
        assert (
            await sc._restore_parent(
                target_sid="sid-a",
                slot_key=nodes["a"].key,
                parent_key=nodes["r"].key,
                parent_sid="sid-r",
                previous_parent=nodes["b"].key,
                previous_parent_sid="sid-b",
            )
            is False
        )
        assert sc._tree_pending_edges == {nodes["a"].key: nodes["b"].key}

        with pytest.raises(sc.SessionControlError) as exc:
            await sc.handoff_target(
                state,
                caller_session_key=_key(owner),
                target=nodes["b"].key,
                to=nodes["a"].key,
            )
        # Either refusal is the guard doing its job: the slot fence catches the move
        # whose endpoint is already in flight, and the cycle check catches it against
        # the announced edge. Both are fail-closed and retryable.
        assert exc.value.code in ("would_cycle", "tree_write_pending"), exc.value.code

        settles[0](True)
        await asyncio.sleep(0)

    _run(_drive())


# -- a fenced recipient may not be handed an unfenced session -----------------


def test_an_unfenced_target_handed_to_a_fenced_recipient_is_refused(tmp_path, monkeypatch):
    """The deputy the fence exists to prevent, assembled out of two correct admissions.

    The fence decides what a caller may reach, and for the verbs a handoff widens it
    reads the MANAGER. So a fenced, agent-created worker given one of the person's own
    tabs drives a session the fence does not apply to: through it the worker reaches every
    session in the workspace and reads the results back, while each half of how it got
    there was admissible on its own.

    Refused here rather than by making a managed slot count as fenced, which would change
    the TARGET's fence status as a side effect of being managed -- a person's own tab
    would silently lose its own reach for having been handed to someone.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    owner = _live(state, _slot(state, "chat-owner-tab"), "sid-owner")
    # The person's OWN second tab: no creator, so the fence does not apply to it.
    own_tab = _live(state, _slot(state, "chat-own-tab-2"), "sid-own-2")
    # An agent-made worker: fenced, reaches only what it created.
    worker = _live(state, _slot(state, "chat-worker"), "sid-worker")
    worker._created_by = owner.key
    assert not sc._caller_is_ownership_fenced(state, own_tab.key), "the target is unfenced"
    assert sc._caller_is_ownership_fenced(state, worker.key), "and the recipient is fenced"

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(owner), target=own_tab.key, to=worker.key
            )
        )

    assert exc.value.code == "deputy_refused"
    assert calls == [], "nothing is appended for a handoff that would build a deputy"
    assert own_tab._manager == ""


def test_the_two_handoffs_the_feature_is_for_are_still_allowed(tmp_path, monkeypatch):
    """The refusal costs the feature nothing. Fenced to fenced is the dispatch shape the
    verb exists for -- a lead handing agent-created workers to an agent-created conductor
    -- and unfenced to unfenced is a person moving their own tabs around. Only the mixed
    direction builds a deputy, and only it is refused.
    """
    state = _make_state(tmp_path)
    owner = _live(state, _slot(state, "chat-owner-tab"), "sid-owner")
    lead = _live(state, _slot(state, "chat-lead"), "sid-lead")
    conductor = _live(state, _slot(state, "chat-conductor"), "sid-conductor")
    worker = _live(state, _slot(state, "chat-worker"), "sid-worker")
    lead._created_by = owner.key
    conductor._created_by = lead.key
    worker._created_by = lead.key
    second_tab = _live(state, _slot(state, "chat-own-tab-2"), "sid-own-2")

    # Fenced to fenced: the dispatch shape.
    _entries(monkeypatch)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    assert sc._caller_is_ownership_fenced(state, worker.key)
    assert sc._caller_is_ownership_fenced(state, conductor.key)
    moved = _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )
    assert moved["manager"] == conductor.key

    # Unfenced to unfenced: the person's own tabs.
    _entries(monkeypatch)
    assert not sc._caller_is_ownership_fenced(state, second_tab.key)
    assert not sc._caller_is_ownership_fenced(state, owner.key)
    third = _live(state, _slot(state, "chat-own-tab-3"), "sid-own-3")
    assert not sc._caller_is_ownership_fenced(state, third.key)
    moved = _run(
        sc.handoff_target(
            state, caller_session_key=_key(owner), target=third.key, to=second_tab.key
        )
    )
    assert moved["manager"] == second_tab.key


# -- every path that BUILDS a slot resolves its manager -----------------------


def test_the_resolver_is_what_every_build_path_calls(tmp_path, monkeypatch):
    """One helper, and the boot reconstruction is one of its callers rather than a second
    implementation. Asserted by driving the pass with the resolver stubbed: if the pass
    had its own copy of the read, the stub would not be reached and the count would not
    be the stub's.
    """
    state = _make_state(tmp_path)
    _team(state)
    seen: list[str] = []

    async def _resolver(state_, slot, **kwargs):
        seen.append(slot.key)
        return False

    monkeypatch.setattr(sc, "resolve_slot_manager", _resolver)
    assert _run(sc.rebuild_managers_from_log(state)) == 0
    assert sorted(seen) == sorted(state._slots), "every slot went through the one helper"


def test_a_resumed_handed_session_comes_up_under_its_recorded_manager(tmp_path, monkeypatch):
    """A slot built AFTER boot gets its manager read too, because the slot's own line
    carries none and the boot pass is long over.

    Published without it, the field is empty -- which reads as "the creator holds it" --
    so the handoff silently reverts: the giver gets send, read, stop and close back on a
    session the log records as driven by someone else, and the recorded manager is refused
    ``not_creator``. A grant handed back to someone recorded as not having it.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    # Handed over in an earlier life, so its line carries the marker and the
    # log is the authority for it -- which is what this test is about.
    _handed_over(worker)
    worker._manager = ""

    # What the log says about the rebuilt slot, resolved through the one helper the
    # resume path calls.
    monkeypatch.setattr(sc, "_manager_still_admissible", lambda *_a, **_k: None)

    def _sync_reader(key, _tab):
        return (conductor.key, True) if key == worker.key else ("", True)

    moved = _run(
        sc.resolve_slot_manager(
            state,
            worker,
            slot_manager=_sync_reader,
            # The occupancy rides the SAME seam as the manager read, so an injected
            # reader cannot leave the two disagreeing about the same log.
            slot_manager_tab=lambda _key, _tab: conductor._tab_id,
        )
    )

    assert moved is True
    assert worker._manager == conductor.key, "the recorded manager is in control"
    assert sc.manager_of(worker) == conductor.key
    # And the giver is refused on the verbs a handoff moves.
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "not_creator", "the giver does not get the reach back"


# -- UNRESOLVED is its own state, and a slot in it is refused -----------------


def test_an_unresolved_slot_refuses_the_verbs_that_read_the_manager(tmp_path, monkeypatch):
    """An undecided read establishes nothing, so the slot is denied rather than answered
    from its creator.

    Empty means "the creator holds it", which is a claim a failed read cannot make: the
    entry moving the reach away may be sitting in exactly the part the read missed. So
    the two states are kept apart, and the slot that has no established answer refuses
    the verbs whose authorization depends on one.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    worker._manager_unresolved = True

    for operation in ("send", "read", "summary", "stop", "close", "adopt", "release", "handoff"):
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state, caller_session_key=_key(lead), target=worker.key, operation=operation
            )
        assert exc.value.code == "manager_unresolved", operation
        assert exc.value.status == 409, "retryable: a later read may succeed"

    # A verb OUTSIDE the widened set does not read the manager, so it is unaffected.
    assert (
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="rename"
        )
        is worker
    )

    # And an authoritative read clears it.
    def _decided(_key, _tab):
        return ("", True)

    assert _run(sc.resolve_slot_manager(state, worker, slot_manager=_decided)) is (False)
    assert worker._manager_unresolved is False
    assert (
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
        is worker
    )


def test_a_slot_hydrated_from_a_line_is_unresolved_until_the_log_is_read(tmp_path, monkeypatch):
    """DEFAULT-DENY, which is what makes a forgotten build path fail closed.

    Every path that builds a slot from a metadata line goes through the codec, and the
    codec marks the slot unresolved. Only the resolver clears it. So a path added later
    that publishes a slot without consulting the log does not silently revert a handoff
    -- it produces a slot that refuses the manager-dependent verbs, which is visible.

    This is the closest expressible form of "a new path cannot publish a servable slot
    without the resolver": it is asserted on the hydrating chokepoint rather than on a
    roster of call sites, so a path nobody has written yet is covered too.

    Scoped by the handed-over INDEX: default-deny applies to a slot the index names,
    because that is the only slot a forgotten read could get wrong. A slot it does not
    name is decided as creator-held at the same chokepoint, and the LINE says nothing
    about it either way -- which is the point, since the line is writable by the session
    whose grant a handoff revoked.
    """
    from kiro_crew.dashboard.slot_persistence import metadata_codec as codec

    state = _make_state(tmp_path)
    built = state.get_or_create_slot("chat-rehydrated")
    assert built._manager_unresolved is False, "a freshly minted slot needs no read"
    _handed_over(built)

    codec.apply(
        state,
        built,
        {"created_by": "chat-lead"},
        codec.Restore(name="chat-rehydrated"),
        # The index as a hydrate's own pass read it, handed in as a value: the codec
        # takes no read of its own, so what a slot comes up as is decided by the answer
        # its caller held.
        handoff_index=handed_over.handed_over_slots(),
    )

    assert built._manager_unresolved is True, "an INDEXED slot hydrates unresolved"
    assert built._manager == "", "and nothing on the line offers a manager to adopt"

    plain = state.get_or_create_slot("chat-never-handed")
    codec.apply(
        state,
        plain,
        {"created_by": "chat-lead"},
        codec.Restore(name="chat-never-handed"),
        handoff_index=handed_over.handed_over_slots(),
    )
    assert plain._manager_unresolved is False, "an unindexed slot is decided at hydrate"


def test_targeted_rehydration_resolves_the_manager_of_the_slot_it_builds(tmp_path, monkeypatch):
    """The THIRD path that builds a slot after boot, beside revive and resume.

    ``rehydrate_slot_from_history_async`` is how anything outside the persistence module
    rebuilds one slot from disk -- reached from ``api_send_message`` when a cron
    notification wakes an unpinned worker the startup budget skipped. It restores
    ``_created_by`` from the line and the line carries no manager, so without the read
    the handoff silently reverts: the giver gets send, read, stop and close back on a
    session the log records as driven by someone else.

    Asserted on the CALL, which is the same assertion the revive path carries. The codec
    marks every hydrated slot unresolved, so a path that skips this read produces a slot
    that refuses rather than one that quietly reverts -- but refusing forever is not the
    outcome either, which is why the call is pinned too.
    """
    from unittest.mock import MagicMock

    from kiro_crew.dashboard import chat_persistence

    state = _make_state(tmp_path)
    built = _live(state, _slot(state, "chat-woken-worker"), "sid-woken")
    built._created_by = "chat-lead"
    # RETRACTED from the table: the function short-circuits on a slot that is already
    # published, and the case under test is the one where it has to build it.
    state._slots.pop(built.key, None)

    monkeypatch.setattr(
        chat_persistence,
        "_prefetch_rehydrate_inputs",
        lambda *a, **k: ({"created_by": "chat-lead"}, True, [{}], {}, None, None, False),
    )
    monkeypatch.setattr(chat_persistence, "slot_closed_since", lambda *a, **k: False)
    monkeypatch.setattr(chat_persistence, "_deletion_during_read", lambda *a, **k: None)
    monkeypatch.setattr(chat_persistence, "_rehydrate_slot_from_history", lambda *a, **k: built)
    monkeypatch.setattr(chat_persistence, "_recover_mcp_app_claims", lambda *a, **k: None)

    resolved: list[str] = []

    async def _traced(state_, slot, **kwargs):
        resolved.append(slot.key)
        return False

    monkeypatch.setattr(sc, "resolve_slot_manager", _traced)

    log = MagicMock()
    state.conversation_log = log

    async def _drive() -> None:
        await chat_persistence.rehydrate_slot_from_history_async(state, built.key)

    _run(_drive())

    assert resolved == [built.key], "the slot it built went through the manager resolver"


# -- the target-side refusals run on the RESOLVED slot, not on the caller's string ----


@pytest.mark.parametrize("naming", ["key", "title", "stem"])
def test_the_deputy_refusal_runs_whatever_the_target_is_named_by(tmp_path, monkeypatch, naming):
    """``_resolve_slot`` accepts a slot key, a transcript stem and a title, so a refusal
    written against the caller's STRING runs on one of those forms and silently skips the
    other two.

    That is the shape this change has had to close eight times: a guard that is correct
    where it runs, on a path that reaches the decision without running it. So the two
    refusals that compare the recipient against the session being handed take the RESOLVED
    slot, from the gate that already holds it -- one copy, every naming form.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    owner = _live(state, _slot(state, "chat-owner-tab"), "sid-owner")
    own_tab = _live(state, _slot(state, "chat-own-tab-2"), "sid-own-2")
    own_tab.title = "My Own Notes"
    own_tab._titled = True
    worker = _live(state, _slot(state, "chat-worker"), "sid-worker")
    worker._created_by = owner.key
    assert not sc._caller_is_ownership_fenced(state, own_tab.key)
    assert sc._caller_is_ownership_fenced(state, worker.key)

    named = {
        "key": own_tab.key,
        "title": "My Own Notes",
        # What ``list_sessions`` reports, which the tool description tells callers to pass.
        "stem": f"dashboard_{own_tab.key}",
    }[naming]
    assert sc._resolve_slot(state, named) is own_tab, "the harness names the same slot"

    with pytest.raises(sc.SessionControlError) as exc:
        _run(sc.handoff_target(state, caller_session_key=_key(owner), target=named, to=worker.key))

    assert exc.value.code == "deputy_refused", naming
    assert calls == [], "nothing is appended for a handoff that would build a deputy"
    assert own_tab._manager == ""


@pytest.mark.parametrize("naming", ["key", "title", "stem"])
def test_the_store_comparison_runs_whatever_the_target_is_named_by(tmp_path, monkeypatch, naming):
    """The same hole, in the refusal that keeps a handoff from being accepted now and
    revoked at the next start. It compares the recipient against the session being handed,
    so it needs the resolved slot for the same reason."""
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    worker.title = "Worker Notes"
    worker._titled = True
    lead.memory_store = ""
    conductor.memory_store = ""
    worker.memory_store = "member-raymond"

    named = {
        "key": worker.key,
        "title": "Worker Notes",
        "stem": f"dashboard_{worker.key}",
    }[naming]

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(state, caller_session_key=_key(lead), target=named, to=conductor.key)
        )

    assert exc.value.code == "recipient_other_store", naming
    assert calls == [], "no grant is appended for a handoff that would not survive"


def test_the_manager_read_is_not_the_last_awaiting_act_on_the_resume_path(tmp_path):
    """ORDERING, asserted on the source because that is what the contract is about.

    ``resume_slot_from_history`` depends on the containment hook being the LAST awaiting
    act: the store-backed boundaries (channel link, Slack binding, outbound mirror) are
    read inside it, so an await placed after it can let a binding recorded during that
    await reach a published slot with no pass left to see it. The manager read is an
    off-loop crew-log read, so it is an awaiting act and belongs above the hook.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source = (repo_root / "src/kiro_crew/dashboard/chat_api/resume.py").read_text(encoding="utf-8")
    resolve_at = source.index("await _session_control.resolve_slot_manager(state, slot)")
    hook_at = source.index("refusal = await containment(slot)\n", resolve_at)
    final_at = source.index("refusal = final_check(slot)", resolve_at)
    publish_at = source.index("state._slots[slot.key] = slot", resolve_at)

    assert resolve_at < hook_at, "the manager read runs BEFORE the last containment pass"
    assert hook_at < final_at < publish_at, "and the hook stays the last awaiting act"


# ── The refusal lives on the READ, not on an entry predicate ─────────────────
#
# The refusal owed when who manages a slot has not been read belongs on the READ
# (``require_manager_of``), not on a condition at the top of ``authorize_target``. A
# condition there has to name which operations consult the manager, and such a
# predicate only APPROXIMATES the read: it is too wide for a caller whose answer the
# manager cannot change, and too narrow for the three sites that reach the value
# without passing the gate. What this block pins is that the guard sits where the
# value is read, so neither error is reachable.


def test_an_unfenced_caller_is_served_while_the_managers_are_unread(tmp_path):
    """The availability property, and the First Principles clearance condition verbatim.

    A person's own tab carries no ``_created_by`` and no channel link, so it is not
    ownership-fenced and ``authorize_target``'s fence is skipped before any manager is
    read. Its answer therefore cannot depend on the manager -- and a refusal it cannot
    act on is a gateway that stops serving its owner after every restart, for a feature
    that install may never use.

    Both ways the value can be unread are covered, because they were two separate
    conditions and would have had to be fixed twice.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    owner = state.get_slot("chat-owner-tab")
    assert not sc._caller_is_ownership_fenced(state, owner.key), "the premise of this test"
    _hold(state, worker.key, lead.key, sid="sid-worker")

    # A MARKED slot whose log has not been read: the one state left that refuses the
    # manager-dependent verbs, and an unfenced caller still never reaches that read.
    _handed_over(worker)
    worker._manager_unresolved = True
    for operation in ("send", "read", "summary", "stop", "close"):
        assert (
            sc.authorize_target(
                state, caller_session_key=_key(owner), target=worker.key, operation=operation
            )
            is worker
        ), operation


def test_the_handoff_verb_still_refuses_an_unfenced_caller_on_an_unread_manager(tmp_path):
    """The site the proposed subtraction would have admitted a guess at.

    Gating the refusal on ``ownership_fenced`` alone is not safe, because ``handoff``
    reads the manager itself and not through the fence: its gate asks whether the
    recipient ALREADY manages the slot, and its commit records the manager it replaces.
    An unfenced caller is exactly the caller that subtraction would have let through,
    and a handoff recorded against a guessed previous manager is a correction that
    later names the wrong session.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    owner = state.get_slot("chat-owner-tab")
    assert not sc._caller_is_ownership_fenced(state, owner.key)
    worker._manager_unresolved = True

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(owner), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "manager_unresolved"
    assert exc.value.status == 409


def test_a_broadcast_refuses_rather_than_name_a_guessed_audience(tmp_path):
    """The second site outside the fence, and the one with no per-target gate behind it.

    ``broadcast_audience`` filters EVERY slot by who manages it, for any caller, fenced
    or not. Answered from an unread field it silently reaches a session handed away and
    silently misses one handed over. So an unreadable manager has to change the answer
    -- and the direction is EXCLUDE, not refuse, because enumerating is not authorizing:
    excluding narrows reach, while the fence's refusal protects a grant. Refusing the
    whole audience would make one unreadable slot a process-lifetime outage for every
    caller's broadcasts and rosters, including unfenced owners and other workspaces.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    assert sc.broadcast_audience(state, lead.key) == [conductor.key, worker.key]

    # A slot the CONDUCTOR holds by handoff, so the conductor is neither its creator nor
    # a session the field names once the read fails.
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.broadcast_audience(state, conductor.key) == [worker.key]
    worker._manager = ""
    worker._manager_unresolved = True
    assert sc.broadcast_audience(state, conductor.key) == [], "narrowed, not 409"

    # A caller that holds it whatever the read says still gets it, so a read fault
    # cannot hide a caller's own session from its own roster.
    assert worker._created_by == lead.key
    assert sc.broadcast_audience(state, lead.key) == [conductor.key, worker.key]


def test_every_authorization_read_of_the_manager_goes_through_the_chokepoint(tmp_path):
    """The enumeration, mechanized: no src site authorizes off the raw read.

    Nine findings on this line had one shape -- a guard correct where it runs, on a path
    that reaches the decision without running it -- and each was closed by adding the
    guard to one more path. This test closes the SHAPE instead: ``manager_of`` is the
    raw read and may only be called by the rebuild (which is what makes the value
    known) and by :func:`require_manager_of` (which refuses first). Any other caller is
    a path that authorizes off a value nobody checked, and this fails rather than
    waiting for a reviewer to find it.
    """
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    # Encoding PINNED: the default is the platform's, which is cp1252 on Windows, and
    # this module carries non-ASCII bytes -- so an unpinned read decodes there and not
    # here, and the contract this test pins would go unchecked on half the matrix.
    source = (repo_root / "src/kiro_crew/dashboard/session_control.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    #: The two functions allowed to read the manager without refusing first, each for a
    #: stated reason. ``require_manager_of`` IS the refusal, and nothing else may read
    #: the raw value for an authorization answer.
    permitted = {"require_manager_of"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in permitted or node.name == "manager_of":
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "manager_of":
                offenders.append(f"{node.name}:{inner.lineno}")

    assert not offenders, (
        "these functions read manager_of() directly instead of require_manager_of(), so "
        f"they authorize off a value that may never have been read: {offenders}. Either "
        "call require_manager_of(state, slot), or add the function to `permitted` above "
        "with the reason it may read the raw value."
    )


#: The attributes that together say who manages a slot. One fact, so one writer.
#:
#: ``_manager_revision`` is one of them, not bookkeeping beside them. It is how a
#: caller that suspended learns the fact was stated while it was away, so a site that
#: could bump it without stating anything -- or state something without bumping it --
#: would hand a stale read a revision check it passes.
_MANAGER_ATTRIBUTES = frozenset(
    {
        "_manager",
        "_manager_tab",
        "_manager_unresolved",
        "_manager_grant_inadmissible",
        "_manager_revision",
    }
)

#: Where each of them MAY be assigned, by ``file::function``, with the reason.
#:
#: ``publish_manager`` is the funnel itself. ``_ChatSlot.__init__`` is the object being
#: built rather than a transition on a built one, and it writes every one. The adaptive
#: controller's ``_manager`` is an unrelated attribute on an unrelated class -- an
#: ``ExecActuator`` handle, pre-existing on main -- and is listed because the NAME
#: collides, not the concept.
_PERMITTED_MANAGER_WRITERS = frozenset(
    {
        "src/kiro_crew/dashboard/session_control.py::publish_manager",
        "src/kiro_crew/dashboard/state.py::__init__",
        "src/kiro_crew/adaptive/controller.py::__init__",
    }
)


@functools.lru_cache(maxsize=1)
def _src_sources() -> "tuple[tuple[str, str], ...]":
    """Every module under ``src/kiro_crew`` as ``(posix relative path, source text)``.

    Read once for the whole file: the contract tests below each scan the package, and
    re-reading it per test is the difference between seconds and most of a minute on
    every shard that picks this file up.

    Encoding PINNED, as in the chokepoint test above: the platform default is cp1252 on
    Windows, so an unpinned read would decode there and not here, and the contracts
    these trees carry would go unchecked on half the matrix.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    return tuple(
        (path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted((root / "src" / "kiro_crew").rglob("*.py"))
    )


#: ``relative path -> AST``, so the trees two tests share are parsed once and the
#: line-ownership cache below stays valid across them.
_TREES: "dict[str, object]" = {}


def _trees_mentioning(needle: str) -> "list[tuple[str, object]]":
    """The modules whose text contains *needle*, parsed.

    A substring pre-filter, not the check: what decides is always the AST walk the
    caller does. Parsing and walking the whole package per test costs more than the
    rest of this file put together, and the names these contracts are about appear in
    a handful of modules.
    """
    import ast

    out: "list[tuple[str, object]]" = []
    for where, text in _src_sources():
        if needle not in text:
            continue
        tree = _TREES.get(where)
        if tree is None:
            tree = _TREES[where] = ast.parse(text)
        out.append((where, tree))
    return out


#: ``id(tree) -> {lineno: innermost function name}``, built on demand and kept, because
#: three contract tests ask it of the same trees and the trees themselves are cached.
_ENCLOSING: "dict[int, dict[int, str]]" = {}


def _enclosing_functions(tree) -> "dict[int, str]":
    """Which function OWNS each line of *tree*, innermost winning.

    A comparison written inside a closure belongs to the closure, not to the function
    that happens to contain it: attributing it to the outer one would make a permitted
    inner function unpermittable by name.
    """
    import ast

    cached = _ENCLOSING.get(id(tree))
    if cached is not None:
        return cached
    owner: "dict[int, str]" = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            # Innermost wins because ``ast.walk`` reaches an enclosing function first
            # only by breadth; the nested def's own pass overwrites what it covers.
            owner[getattr(inner, "lineno", -1)] = node.name
    _ENCLOSING[id(tree)] = owner
    return owner


def _manager_attribute_writes() -> "list[str]":
    """Every ``<obj>.<manager attribute> = ...`` in src, as ``file::function:lineno``."""
    import ast

    writes: list[str] = []
    for where, tree in _trees_mentioning("_manager"):
        found: "list[int]" = []
        for node in ast.walk(tree):
            targets = (
                list(node.targets)
                if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
            )
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in _MANAGER_ATTRIBUTES:
                    found.append(node.lineno)
        if not found:
            # Named-function attribution is the expensive half, so it is asked only of
            # the handful of modules that write one of the four at all.
            continue
        owner = _enclosing_functions(tree)
        writes.extend(f"{where}::{owner.get(line, '<module>')}:{line}" for line in found)
    return writes


def test_the_four_manager_attributes_have_exactly_one_writer(tmp_path):
    """The ruling, mechanized: a fifth assignment site fails this test.

    Twenty-one sites across the resolve, the verb, the compensation and the hydrate
    could write four attributes that are ONE fact, and the same ordering defect was
    found at four of them in turn -- flags cleared, a read awaited, the key written
    after. Each point-fix was right and each left the other points, because the
    invariant is not statable at a site. It is statable here: the only transition on a
    built slot goes through ``publish_manager``, which writes all four from arguments
    with nothing suspending between them.

    This fails on a NEW site rather than on the defect the site would carry, which is
    the difference between catching the fifth instance and catching the shape.
    """
    offenders = [
        write
        for write in _manager_attribute_writes()
        if write.rsplit(":", 1)[0] not in _PERMITTED_MANAGER_WRITERS
    ]
    assert not offenders, (
        "these sites assign one of the four manager attributes directly: "
        f"{offenders}. The four are one statement about who may drive a session, and a "
        "site that writes some of them leaves the slot describing a state that never "
        "existed. Call session_control.publish_manager(slot, manager=..., "
        "manager_tab=..., unresolved=..., inadmissible=...) instead."
    )


def test_the_one_manager_writer_cannot_suspend(tmp_path):
    """``publish_manager`` holds no ``await``, and this fails if one is added.

    The funnel is only worth having if the four attributes move together, and an
    ``await`` between two of them reopens exactly the window that was found four times:
    a slot reading as resolved with no key published yet, which on a marked slot reads
    as "the creator holds it" and hands a revoked giver its reach back. A caller that
    needs an awaited value -- the occupancy read off the log -- awaits it BEFORE calling
    and passes the result in, so the suspension happens while the slot still holds its
    previous whole statement.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(sc.publish_manager)))
    assert not isinstance(tree.body[0], ast.AsyncFunctionDef), "the one writer is sync"
    suspensions = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom))
    ]
    assert not suspensions, (
        "publish_manager suspends, at these lines of its own source: "
        f"{suspensions}. Every reader that arrives during that suspension sees half of "
        "one statement and half of another. Await the value in the CALLER and pass it."
    )


def test_only_one_predicate_answers_whether_a_caller_may_drive_a_slot(tmp_path):
    """The read side's counterpart, and the same shape as the write side's.

    "May this caller drive this slot" is answered by comparing a manager read against a
    caller's key, and every site that makes that comparison has to apply the conditions
    that can go stale while a record stands -- the occupancy of the recorded session and
    the store the two sides share. Those live in ``_manager_record_refusal``, which only
    ``require_manager_of`` reaches, which only ``_managed_by_other`` turns into an
    answer. A second site making the comparison is a second place to forget a condition,
    and the drain re-asserting the admission is the reader that found that out.
    """
    import ast

    #: ``_managed_by_other`` IS the predicate. ``_gate`` is the handoff's own
    #: ``already_manager`` refusal, which asks whether the RECIPIENT is already the
    #: manager rather than whether a caller may drive the slot, and reads ``for_write``
    #: so the refusal it offers stays reachable.
    permitted = {"_managed_by_other", "_gate"}
    offenders: list[str] = []
    for where, tree in _trees_mentioning("manager_of("):
        if where != "src/kiro_crew/dashboard/session_control.py":
            continue
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for side in [node.left, *node.comparators]:
                reader = getattr(getattr(side, "func", None), "id", "")
                if reader not in {"manager_of", "require_manager_of"}:
                    continue
                at = owner.get(node.lineno, "<module>")
                if at not in permitted:
                    offenders.append(f"{at}:{node.lineno}")

    assert not offenders, (
        "these functions compare a manager read against a key themselves instead of "
        f"calling _managed_by_other: {offenders}. Each one is a place the next "
        "staleness condition has to be remembered. Call _managed_by_other(state, slot, "
        "caller_key), or add the function to `permitted` with the reason it asks a "
        "different question."
    )


def test_only_one_site_writes_a_manager_entry(tmp_path):
    """One assembly for the ``session/manager_set`` entry, for the occupancy's sake.

    ``manager_tab`` is optional at the emitter and has to be: the entry forms that name
    no manager carry no occupancy. Optional means omitting it is silent at the write and
    expensive at the read -- the restore brings back a bare name and the compare then
    refuses the live manager. The handoff's correction omitted it, which is the second
    finding of that shape on this line, so the two writers go through one function that
    names every field.
    """
    import ast

    permitted = {"src/kiro_crew/dashboard/session_control.py::_emit_manager_record"}
    offenders: list[str] = []
    for where, tree in _trees_mentioning("on_manager_set"):
        if where == "src/kiro_crew/crew_log/emit.py":
            continue  # the emitter itself, which is what is being funnelled to
        found = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "on_manager_set"
        ]
        if not found:
            continue
        owner = _enclosing_functions(tree)
        offenders.extend(
            f"{where}::{owner.get(line, '<module>')}:{line}"
            for line in found
            if f"{where}::{owner.get(line, '<module>')}" not in permitted
        )

    assert not offenders, (
        "these sites call on_manager_set directly instead of going through "
        f"_emit_manager_record: {offenders}. Each one can omit the occupancy that "
        "qualifies the key it records, and the emitter cannot refuse it."
    )


def test_a_correction_that_does_not_land_leaves_the_slot_unresolved(tmp_path, monkeypatch):
    """A dropped rollback must not leave the giver being served a reach it will lose.

    The grant's settlement times out (still queued, outcome unknown) and the corrective
    append is then REJECTED -- the correlated pair, because a writer backed up enough to
    time out the first is the same writer at its ceiling for the second. The log may then
    end with the refused grant, which leaves this process unable to say who manages the
    slot.

    Answering the giver anyway is a grant handed out here and taken away at the next
    start, which is the direction the whole verb exists to avoid. So the slot goes
    UNRESOLVED, which is default-deny, and the caller is told that plainly with its own
    code instead of being told the giver still holds it.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    real_settlement = emit.awaiting_settlement
    real_commit = emit.awaiting_commit

    async def _manager_entry_stays_queued(emit_call, *, what=""):
        # ONLY the manager entry is left QUEUED -- outcome unknown, which is the one
        # settlement value that can still land after this verb refuses. The tree append
        # the verb makes first keeps its real (stubbed-and-settled) path, because
        # stalling that one refuses `tree_write_pending` before the commit is reached.
        if what.startswith("the handoff of"):
            emit_call(lambda _landed: None)
            return None
        return await real_settlement(emit_call, what=what)

    async def _correction_is_rejected(emit_call, *, what=""):
        # And the correction behind it is REJECTED. The correlated pair: a writer backed
        # up enough to time out the grant is the same writer at its ceiling here.
        if "back to" in what:
            emit_call(lambda _landed: None)
            return False
        return await real_commit(emit_call, what=what)

    monkeypatch.setattr(emit, "awaiting_settlement", _manager_entry_stays_queued)
    monkeypatch.setattr(emit, "awaiting_commit", _correction_is_rejected)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "handoff_unresolved", "not 'the giver still manages it'"
    assert exc.value.status == 409
    assert worker._manager_unresolved is True, "the slot answers no one until it is re-read"
    assert worker._manager == "", "and nothing was published"
    assert [call["op"] for call in calls].count("manager") == 2, "the grant, then the correction"

    # And the refusal is now what every reader of the value gets, including the giver
    # that was being served before.
    with pytest.raises(sc.SessionControlError) as served:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert served.value.code == "manager_unresolved"


def test_a_refusal_arms_the_read_that_clears_it(tmp_path, monkeypatch):
    """ "Retry in a moment" has to be a true instruction.

    ``resolve_slot_manager`` is called only by BUILD paths -- the boot pass, a revive, a
    resume from history -- and none of them runs again for a slot already open. So a
    slot that went unresolved stayed unresolved until the process restarted, while every
    refusal told the caller to retry. One bad unit anywhere in the store is enough to
    get there, because the listing behind the read is store-wide.

    Arming the re-read from the refusal itself is what makes the window self-clearing,
    and it is armed from the REFUSAL so there is no path that denies without scheduling
    the read that clears it.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    worker._manager_unresolved = True
    reread: list[str] = []

    async def _resolve(_state, slot, **kwargs):
        reread.append(slot.key)
        slot._manager_unresolved = False
        return False

    monkeypatch.setattr(sc, "resolve_slot_manager", _resolve)

    async def _drive() -> None:
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state, caller_session_key=_key(lead), target=worker.key, operation="send"
            )
        assert exc.value.code == "manager_unresolved"
        await async_wait_until(
            lambda: reread == [worker.key], describe=lambda: "the re-read was never armed"
        )
        # Cleared by the read, so the retry the refusal asked for now works.
        assert (
            sc.authorize_target(
                state, caller_session_key=_key(lead), target=worker.key, operation="send"
            )
            is worker
        )

    _run(_drive())
    assert not sc._manager_reresolve_inflight, "the slot is not left marked in-flight"


def test_the_armed_read_is_scheduled_once_for_a_slot_refused_many_times(tmp_path, monkeypatch):
    """A slot refused on every verb must not queue one store read per refusal.

    The refusal is raised from a sync chokepoint that a busy gateway can reach many
    times before the first read finishes, and the read lists the store. Without the
    in-flight set a fleet retrying in a loop would turn one unreadable slot into a
    store-read storm.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    worker._manager_unresolved = True
    release = asyncio.Event()
    started: list[str] = []

    async def _resolve(_state, slot, **kwargs):
        started.append(slot.key)
        await release.wait()
        slot._manager_unresolved = False
        return False

    monkeypatch.setattr(sc, "resolve_slot_manager", _resolve)

    async def _drive() -> None:
        for _ in range(5):
            with pytest.raises(sc.SessionControlError):
                sc.authorize_target(
                    state, caller_session_key=_key(lead), target=worker.key, operation="send"
                )
        await async_wait_until(lambda: started == [worker.key])
        assert started == [worker.key], "five refusals, one read"
        release.set()
        await async_wait_until(lambda: not sc._manager_reresolve_inflight)

    _run(_drive())


# ── Refusals name a session the way a person recognises it ───────────────────


def test_a_refusal_names_the_session_by_title_not_by_slot_key(tmp_path):
    """``chat-1760244`` identifies a session to the gateway and to nobody else.

    Every refusal this verb hands back is read by somebody -- in a tab, in an agent's
    transcript. A refusal that names only the key makes the reader go and look up which
    session it meant before they can act on what they were told, and the reader is
    usually the one person who already knows the session by its title.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    worker.title = "crawl the backlog"
    conductor.title = "release conductor"
    # A PRE-LOCK refusal, so the assertion is about the wording and not about which of
    # the verb's writes got furthest.
    conductor.memory_store = "another-principal"

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "recipient_other_store"
    assert "release conductor" in exc.value.message
    assert conductor.key not in exc.value.message, "the key is not what a person reads"


def test_an_untitled_session_still_falls_back_to_its_key(tmp_path):
    """The key is the FALLBACK and is never dropped.

    A session whose title is empty, blank or still being generated must still be
    identified. Naming it ``''`` would be worse than naming the key: the reader then has
    nothing at all, and a blank name reads as a bug rather than as a missing title.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)

    assert sc.name_for_person(state, worker) == worker.key, "no title: the key"
    worker.title = "   "
    assert sc.name_for_person(state, worker) == worker.key, "blank title: still the key"
    worker.title = "named at last"
    assert sc.name_for_person(state, worker) == "named at last"

    # A bare key is accepted too, because the verb holds the recipient and the replaced
    # manager as keys alone.
    assert sc.name_for_person(state, worker.key) == "named at last"
    assert sc.name_for_person(state, "chat-never-opened") == "chat-never-opened", (
        "a key naming no live slot answers the key -- there is nothing better to say "
        "about a session this gateway is not holding"
    )


def test_a_long_title_is_cut_rather_than_run_through_a_refusal(tmp_path):
    """A title is author-written and unbounded; a refusal is one line somebody reads."""
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    worker.title = "x" * 400

    named = sc.name_for_person(state, worker)
    assert len(named) <= sc._NAME_FOR_PERSON_MAX
    assert named.endswith("\u2026"), "cut with an ellipsis, so the reader knows it was cut"
    assert named.startswith("x"), "and the front is kept, which is the identifying part"


def test_a_restored_prompt_is_not_dropped_because_the_manager_is_not_read_yet(tmp_path):
    """The user's own typed prompts must survive a restart, unread manager or not.

    The restore path STRIPS the containment stamp deliberately (it names a write target
    and the metadata line is editable), so every restored entry is compared against the
    all-False fail-closed baseline. Every hydrated slot starts `_manager_unresolved`, and
    the process-wide flag stands while the rebuild runs -- so a boolean counted against
    that baseline makes the first drain after a restart delete every restored prompt with
    "Queued message dropped".

    Those prompts were typed by a person into the session's own surface. Their admission
    never rested on who manages the slot, so an unread manager is not a constraint they
    gained. `manager_unknown` is therefore compared only when the entry RECORDED it, the
    way `workspace` is, and never against the unmarked baseline.
    """
    state = _make_state(tmp_path)
    _lead, _conductor, worker = _team(state)
    worker._manager_unresolved = True
    now = sc.containment_snapshot(state, worker, on_probe_failure=True)
    assert now["manager_unknown"] is True, "the premise: the drain sees it held"

    # A RESTORED entry: no stamp at all, which is what the sanitizer hands back.
    assert sc.newly_held_constraints(now, {}) == [], "an unmarked entry is not dropped"
    assert sc.newly_held_constraints(now, None) == [], "nor is a malformed one"

    # A STAMPED entry admitted while the manager was readable IS dropped: its admission
    # did rest on the value, and it cannot be re-checked now.
    admitted = dict(now)
    admitted["manager_unknown"] = False
    assert sc.newly_held_constraints(
        now, {sc.QUEUED_CONTAINMENT_META_KEY: admitted}, manager_fence_applied=True
    ) == ["manager_unknown"]

    # And one admitted while it was ALREADY unknown is not a change -- it was knowingly
    # admitted under the constraint, so dropping it would refuse designed behaviour.
    assert (
        sc.newly_held_constraints(
            now, {sc.QUEUED_CONTAINMENT_META_KEY: dict(now)}, manager_fence_applied=True
        )
        == []
    )


# ── The caller half of the admission, re-asserted at the drain ───────────────


def test_a_queued_send_is_dropped_once_its_sender_no_longer_manages_the_target(tmp_path):
    """A handoff revokes the giver's reach, including over what it already queued.

    The target-side containment snapshot cannot see this: nothing about the worker
    changed. What changed is WHO manages it, and the entry was admitted on the relation
    between that value and the sender -- a relation that needed no re-check while the
    fence read ``_created_by``, because a slot's creator is written at birth and never
    moves. A handoff moves it, which is the point of the verb.

    Read through the same predicate ``authorize_target`` admitted the entry with, so the
    drain gains no second manager read of its own.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    stamp = sc.send_origin_meta(state, lead.key, manager_fenced=True)

    # Admitted while the lead held the worker.
    assert sc.sender_lost_its_reach(state, worker, stamp) is False

    # Handed over. Nothing about the worker's containment changed.
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.sender_lost_its_reach(state, worker, stamp) is True
    assert (
        sc.newly_held_constraints(
            sc.containment_snapshot(state, worker, on_probe_failure=True), stamp
        )
        == []
    ), "the snapshot alone sees no change, which is why this check exists"

    # The new manager's own queued delivery is untouched.
    assert (
        sc.sender_lost_its_reach(
            state, worker, sc.send_origin_meta(state, conductor.key, manager_fenced=True)
        )
        is False
    )


def test_a_human_typed_prompt_is_never_dropped_for_a_sender_relation(tmp_path):
    """There is no sender to re-compare, and inventing one deletes user speech.

    A person typing into a session's own surface is admitted by the authenticated
    composer, which never reads who manages the slot. The absence of the stamp is what
    marks that population -- and it marks restored entries too, whose stamp the restore
    path strips on purpose, so a delivery that outlives a restart is not dropped for a
    relation this process has no way to attribute.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id

    assert sc.sender_lost_its_reach(state, worker, {}) is False, "no stamp, no drop"
    assert sc.sender_lost_its_reach(state, worker, None) is False, "nor a malformed one"
    assert (
        sc.sender_lost_its_reach(state, worker, {"queued_containment": {"linked": False}}) is False
    ), "a containment stamp alone is not a sender"


def test_an_unread_manager_drops_under_its_own_reason_not_the_senders(tmp_path):
    """The wording must not claim the sender lost a reach this cannot verify it had.

    ``sender_lost_its_reach`` reads through ``require_manager_of``, so an unread manager
    raises rather than answering. That is still a drop -- the relation cannot be
    re-checked -- but it is the ``manager_unknown`` drop, and the notice says the value
    could not be read rather than asserting a transfer nobody recorded.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    worker._manager_unresolved = True
    stamp = sc.send_origin_meta(state, lead.key, manager_fenced=True)

    with pytest.raises(sc.SessionControlError) as exc:
        sc.sender_lost_its_reach(state, worker, stamp)
    assert exc.value.code == "manager_unresolved"

    notice = sc.describe_containment_change(["manager_unknown"])
    assert "could no longer be read" in notice
    assert "no longer manages" not in notice, "it must not assert a transfer"


# ── Three outcomes, and only two of them mean the creator holds it ───────────
#
# The invariant: ``_manager`` carries only a statement the log actually makes, and
# EMPTY is a statement -- "the creator holds this session". The log makes it in exactly
# two ways: it records no handoff at all, or it records an explicit ``to_creator``
# transfer back. A recorded manager that cannot be published makes neither.


def test_an_authoritative_no_handoff_gives_the_session_to_its_creator(tmp_path):
    """Outcome 1. An entry-less log is a FACT about the slot, not a failed read."""
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    worker._manager_unresolved = True

    _run(sc.resolve_slot_manager(state, worker, slot_manager=lambda _k, _t: ("", True)))

    assert worker._manager == "", "no handoff recorded, so the creator holds it"
    assert worker._manager_unresolved is False, "and the slot is servable"
    assert worker._manager_grant_inadmissible is False
    assert sc.require_manager_of(state, worker) == lead.key


def test_an_explicit_transfer_back_gives_the_session_to_its_creator(tmp_path):
    """Outcome 2. A decided revert reads as empty, which is what ``to_creator`` records.

    The entry form exists so a correction can say "nobody manages this but its creator"
    rather than recording a handoff TO the creator, which would leave the field set
    where it should be empty.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id

    _run(sc.resolve_slot_manager(state, worker, slot_manager=lambda _k, _t: ("", True)))

    assert worker._manager == "", "the transfer back is published as empty"
    assert worker._manager_unresolved is False
    assert worker._manager_grant_inadmissible is False
    assert sc.require_manager_of(state, worker) == lead.key


def test_a_recorded_manager_that_cannot_hold_it_denies_everyone(tmp_path):
    """Outcome 3, and the fix. NOT the recipient, and NOT the giver either.

    The log records a handoff to a conductor that has since closed. Handing the session
    back to the lead would be this gateway inventing the one statement the log never
    made -- a transfer back -- and handing a revoked grant to the session that gave it
    away. That is the direction the whole verb exists to prevent, so the answer is
    nobody until a fresh handoff records who holds it.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    # Handed over in an earlier life, so its line carries the marker and the
    # log is the authority for it -- which is what this test is about.
    _handed_over(worker)
    # The recorded manager is gone, which is what a restart after it closed looks like.
    state._slots.pop(conductor.key, None)

    moved = _run(
        sc.resolve_slot_manager(state, worker, slot_manager=lambda _k, _t: (conductor.key, True))
    )

    assert moved is True, "the field carries what the log records"
    assert worker._manager == conductor.key, "which is the inadmissible grant itself"
    assert worker._manager_unresolved is False, "the read succeeded; it is not unread"
    assert worker._manager_grant_inadmissible is True, "it is read and unusable"

    # The GIVER is refused, which is the whole finding.
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "manager_grant_inadmissible"
    assert exc.value.status == 409

    # A PERSON is not stranded: an owner's own tab is unfenced, so its reach never came
    # from this field and it can still act on the session.
    owner = state.get_slot("chat-owner-tab")
    assert not sc._caller_is_ownership_fenced(state, owner.key)
    for operation in ("send", "stop", "close", "release", "adopt"):
        assert (
            sc.authorize_target(
                state, caller_session_key=_key(owner), target=worker.key, operation=operation
            )
            is worker
        ), operation


def test_an_inadmissible_grant_does_not_arm_a_re_read(tmp_path, monkeypatch):
    """A read that already succeeded will keep succeeding with the same answer.

    The unread refusal arms a re-read because that is what makes "retry in a moment"
    true. This one must not: the store listing would be spent once per refusal to learn
    what the gateway already knows, and a busy fleet retrying in a loop would turn one
    closed conductor into a store-read storm.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    worker._manager_grant_inadmissible = True
    armed: list[str] = []

    async def _resolve(_state, slot, **kwargs):
        armed.append(slot.key)
        return False

    monkeypatch.setattr(sc, "resolve_slot_manager", _resolve)

    async def _drive() -> None:
        for _ in range(3):
            with pytest.raises(sc.SessionControlError) as exc:
                sc.authorize_target(
                    state, caller_session_key=_key(lead), target=worker.key, operation="send"
                )
            assert exc.value.code == "manager_grant_inadmissible"
        await asyncio.sleep(0)
        assert armed == [], "no re-read is scheduled for a decided answer"

    _run(_drive())


def test_a_fresh_handoff_brings_a_denied_slot_back(tmp_path, monkeypatch):
    """The way out that the refusal names, so the denial is not a dead end."""
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    worker._manager = "chat-gone"
    worker._manager_grant_inadmissible = True
    # An unfenced owner is who can still drive the recovery.
    owner = state.get_slot("chat-owner-tab")

    _run(
        sc.handoff_target(
            state, caller_session_key=_key(owner), target=worker.key, to=conductor.key
        )
    )

    assert worker._manager == conductor.key
    assert worker._manager_grant_inadmissible is False, "the statement the log was missing"
    assert [call["op"] for call in calls].count("manager") == 1
    assert sc.require_manager_of(state, worker) == conductor.key


def test_an_unhanded_restored_slot_is_served_by_its_creator_without_reading_the_log(
    tmp_path, monkeypatch
):
    """Design's clearance condition, as the condition it states.

    "A restored slot with no handoff record is served by its creator without reading the
    crew log." The handed-over marker on the slot's line is what makes that decidable:
    a line without it hydrates as decided-creator, so nothing waits on history for the
    sessions that were never handed over -- nearly all of them. Asserted through the
    codec, which is the path every restore takes, and with the log reader rigged to fail
    the test if it is called at all.
    """
    from kiro_crew.crew_log import session_tree
    from kiro_crew.dashboard.slot_persistence import metadata_codec

    def _must_not_read(_key):
        raise AssertionError("an unmarked slot read the crew log")

    monkeypatch.setattr(session_tree, "slot_manager", _must_not_read)
    monkeypatch.setattr(session_tree, "slot_manager_tab", _must_not_read)

    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    worker._manager_unresolved = True  # whatever a previous life left on the object
    metadata_codec.apply(
        state,
        worker,
        {"created_by": lead.key},
        metadata_codec.Resume(request_title="", history_key="dashboard:h"),
        handoff_index=handed_over.handed_over_slots(),
    )

    assert worker.key not in handed_over.handed_over_slots()[0]
    assert worker._manager_unresolved is False, "decided at hydrate, from the line"
    assert sc.require_manager_of(state, worker) == lead.key, "the creator holds it"
    for operation in ("send", "read", "summary", "stop", "close", "adopt", "release"):
        assert (
            sc.authorize_target(
                state, caller_session_key=_key(lead), target=worker.key, operation=operation
            )
            is worker
        ), operation
    # And the boot pass decides it the same way, still without a read.
    _run(sc.rebuild_managers_from_log(state))
    assert worker._manager == "" and worker._manager_unresolved is False


def test_a_marked_restored_slot_waits_on_its_own_log(tmp_path):
    """The other half: the index is what makes the log authoritative for a slot.

    A slot the index names comes up UNRESOLVED, so the fenced giver is refused until
    that slot's own read lands -- per slot, with no process-wide window.
    """
    from kiro_crew.dashboard.slot_persistence import metadata_codec

    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    _handed_over(worker)
    metadata_codec.apply(
        state,
        worker,
        {"created_by": lead.key},
        metadata_codec.Resume(request_title="", history_key="dashboard:h"),
        handoff_index=handed_over.handed_over_slots(),
    )
    assert worker.key in handed_over.handed_over_slots()[0]
    assert worker._manager_unresolved is True
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "manager_unresolved"


# ── The drain re-checks only the population the fence applied to ─────────────
#
# Re-assert at the drain exactly the predicate that admitted the entry, INCLUDING
# whether it applied. An unfenced caller's admission never reads the manager, because
# the fence short-circuits before the read -- so neither "the manager moved" nor "the
# manager cannot be read" is a constraint that admission gained. Getting this wrong
# deletes a message somebody had already been told was accepted, which is why the
# fail-closed direction here is NOT dropping.


def test_an_unfenced_senders_queued_message_is_never_dropped_for_the_manager(tmp_path):
    """A person's own tab sending into a session it did not create.

    ``authorize_target`` skipped the fence entirely, so the manager was never consulted
    and the caller is still authorized at delivery. Re-comparing the relation anyway
    drops the message and tells the sender its session does not manage a session it never
    managed and never needed to.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    owner = state.get_slot("chat-owner-tab")
    assert not sc._caller_is_ownership_fenced(state, owner.key), "the premise"

    # Stamped the way the verb stamps it for an unfenced caller.
    stamp = sc.send_origin_meta(state, owner.key, manager_fenced=False)
    assert sc.send_origin_manager_fenced(stamp) is False

    # Nothing was handed over, and it is still not dropped.
    assert sc.sender_lost_its_reach(state, worker, stamp) is False
    # Nor after a handoff to someone else.
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.sender_lost_its_reach(state, worker, stamp) is False
    # Nor while the manager cannot be read at all.
    worker._manager_unresolved = True
    assert sc.sender_lost_its_reach(state, worker, stamp) is False


def test_an_unfenced_senders_entry_is_not_dropped_for_an_unreadable_manager(tmp_path):
    """The same population, through the containment snapshot rather than the sender.

    ``manager_unknown`` has to carry the identical gate. Without it the boot window
    deletes every stamped entry from every unfenced caller, which is the same defect
    reached by the other of the two drain checks.
    """
    state = _make_state(tmp_path)
    _lead, _conductor, worker = _team(state)
    worker._manager_unresolved = True
    now = sc.containment_snapshot(state, worker, on_probe_failure=True)
    assert now["manager_unknown"] is True, "the premise: the drain sees it held"

    admitted = dict(now)
    admitted["manager_unknown"] = False
    entry = {sc.QUEUED_CONTAINMENT_META_KEY: admitted}

    assert (
        sc.newly_held_constraints(now, entry, manager_fence_applied=False) == []
    ), "an admission the fence never touched did not read the manager"
    assert sc.newly_held_constraints(now, entry, manager_fence_applied=True) == [
        "manager_unknown"
    ], "and a fenced one still is"


def test_a_stamp_without_the_verdict_is_read_as_unfenced(tmp_path):
    """Fail-closed here means NOT dropping, which is the opposite of the usual reading.

    An entry written before the verdict was recorded, or with a malformed stamp, carries
    no evidence that the manager was ever read for it. Treating that as "fenced" would
    drop messages on the strength of a predicate nobody can show was applied, and the
    thing being destroyed is text somebody was already told had been accepted.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id

    assert sc.send_origin_manager_fenced({}) is False
    assert sc.send_origin_manager_fenced(None) is False
    assert sc.send_origin_manager_fenced({sc.SEND_ORIGIN_META_KEY: "not-a-dict"}) is False
    # The shape a stamp from before this field had: slot and tab, no verdict.
    legacy = {sc.SEND_ORIGIN_META_KEY: {"slot": "chat-lead", "tab": "tab-1"}}
    assert sc.send_origin_manager_fenced(legacy) is False
    assert sc.sender_lost_its_reach(state, worker, legacy) is False


def test_one_unreadable_slot_does_not_take_out_every_callers_roster(tmp_path):
    """The availability half, and why an audience excludes where a fence refuses.

    ``_manager_grant_inadmissible`` is cleared only by a fresh handoff, so a slot whose
    recorded manager closed before a restart carries it for the life of the process. If
    the audience scan refused on it, every default broadcast and every roster read would
    answer 409 for every caller -- unfenced owners included, and callers in other
    workspaces, none of whom have anything to do with that slot.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    other = _live(state, _slot(state, "chat-other"), "sid-other")
    other._created_by = conductor.key

    # The denied slot, as a restart after its recorded manager closed leaves it.
    worker._manager = "chat-gone"
    worker._manager_grant_inadmissible = True

    # Every caller still gets an answer, and the denied slot is simply absent from the
    # audiences of callers that cannot show they hold it.
    assert sc.broadcast_audience(state, conductor.key) == [other.key]
    owner = state.get_slot("chat-owner-tab")
    # The owner tab created the lead, so the lead is its audience -- and the denied slot
    # is simply not in it, rather than taking the whole answer down with a 409.
    assert sc.broadcast_audience(state, owner.key) == [lead.key]
    # Its creator still sees it, because that verdict does not depend on the read.
    assert worker.key in sc.broadcast_audience(state, lead.key)


def test_no_other_mover_can_slip_between_the_tree_verdict_and_the_commit_fence(tmp_path):
    """The window the target fence must not have, driven rather than argued about.

    ``_tree_append_verdict`` removes the target from ``_tree_pending_slots`` the moment
    the append's outcome is known, and the commit re-holds it. If anything can be
    scheduled between those two, a racing reparent reads an unfenced slot, passes the
    pending check and commits while the manager append is still in flight -- leaving the
    tree naming one parent and the manager naming another, which is the state the
    compensation exists to prevent and cannot repair once both landed.

    Asserted by watching the fence from the settle callback, which is the one place that
    runs at exactly the moment the verdict is published.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    #: Whether the target was EVER observed unfenced by a would-be racing mover, sampled
    #: from the settle callback and from a loop hop immediately after it.
    unfenced_seen: list[str] = []

    def _sample(when: str) -> None:
        if worker.key not in sc._tree_pending_slots:
            unfenced_seen.append(when)

    settles: list = []

    def _record(op, sid, kw, landed):
        settle = kw.pop("on_settled", None)
        if settle is not None:
            if op == "adopt":
                # Published at the instant the verdict becomes known, which is where the
                # unfence happens.
                settle(landed)
                _sample("at the adopt settle")
            else:
                settle(landed)
        settles.append(op)

    import unittest.mock as _mock

    with (
        _mock.patch.object(
            emit, "on_session_adopted", lambda sid, **kw: _record("adopt", sid, kw, True)
        ),
        _mock.patch.object(
            emit, "on_session_released", lambda sid, **kw: _record("release", sid, kw, True)
        ),
        _mock.patch.object(
            emit, "on_manager_set", lambda sid, **kw: _record("manager", sid, kw, True)
        ),
    ):

        async def _drive() -> None:
            call = asyncio.ensure_future(
                sc.handoff_target(
                    state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
                )
            )
            # Hop the loop repeatedly for the whole life of the verb: every hop is a
            # point a racing mover could have been scheduled, so sampling on each one
            # covers the gap if there is one.
            for _ in range(40):
                if call.done():
                    break
                # ONLY the interval that matters: after the tree verdict is published
                # and before the manager append is handed over. Outside it the slot is
                # legitimately unfenced -- before the verb registers the move, and after
                # the whole thing is done -- so sampling there says nothing.
                if "adopt" in settles and "manager" not in settles:
                    _sample("on a loop hop inside the window")
                await asyncio.sleep(0)
            await call

        _run(_drive())

    assert "manager" in settles, "the verb reached its commit, so the window was exercised"
    # The slot is unfenced once the verb is DONE, which is correct and not the race.
    assert worker.key not in sc._tree_pending_slots
    assert unfenced_seen == [], (
        "the target was observable as unfenced mid-verb at: "
        f"{unfenced_seen} -- another mover could have committed there"
    )


def test_the_deciding_cycle_check_has_no_await_before_the_append(tmp_path):
    """Asserted on source ORDER, because the race it prevents cannot be built from here.

    A cycle verdict is a statement about the tree when it was read. The early check runs
    before the id resolutions, the gate's warm and the re-authorization, all of which
    suspend -- and a detached commit window belonging to an earlier verb can compensate
    inside that gap, restoring the edge the check was asked about. The verdict is then
    false when the append lands, and the log holds two edges that each passed alone with
    a loop between them.

    What makes that impossible is an ordering property: the DECIDING check is re-read
    from the tree and nothing awaits between it and the append. Driving the race from a
    test means landing a write inside a specific scheduler gap, and the attempt lands
    before the early check instead -- a test that passes for the wrong reason is worse
    than none, so the property itself is pinned, the way the tree fence's no-suspension
    window is.
    """
    import pathlib as _pathlib

    repo_root = _pathlib.Path(__file__).resolve().parents[1]
    source = (repo_root / "src/kiro_crew/dashboard/session_control.py").read_text(encoding="utf-8")
    deciding = source.index("if is_ancestor(slot.key, parent_key, _view_now):")
    append = source.index("settled, landed = _tree_append_waiter(slot.key)", deciding)
    between = source[deciding:append]

    assert "await" not in between, (
        "something awaits between the deciding cycle check and the append: "
        f"{[ln.strip() for ln in between.splitlines() if 'await' in ln]}. The tree can "
        "move in that gap, which is exactly what this check is placed late to prevent."
    )
    # And the re-read is a FRESH one, not the early view carried down.
    assert "_slot_tree_parent(slot.key)" in source[:deciding].rsplit("# THE CYCLE CHECK", 1)[-1], (
        "the deciding check must re-read the tree; reusing the early view would make it "
        "a second look at the same stale answer"
    )


# ── A manager key that was retaken cannot be identified ─────────────────────
#
# A slot key is a NAME, and a closed session's name can be retaken. The authorization
# compare is key to key, so without an occupancy a new occupant of the manager's key
# reads as the session that was handed the worker. This is the DEFAULT path: no switch,
# only a manager that closed and a later session that took its name.


def test_a_new_occupant_of_the_managers_key_does_not_inherit_its_reach(tmp_path):
    """The finding itself, and it needs no feature flag to reach.

    The conductor is handed the worker, closes, and a different session opens under the
    same key. A key-to-key compare cannot tell the two apart, so the newcomer would be
    answered as the worker's manager and reach send, read, stop, close, adopt, release
    and handoff on a session nobody gave it.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.require_manager_of(state, worker) == conductor.key, "the premise"
    assert not sc._managed_by_other(state, worker, conductor.key, conductor._tab_id)

    # The conductor closes and a DIFFERENT session takes its key. A fresh slot mints a
    # fresh tab identity, which is the whole of what distinguishes the two.
    recorded_tab = conductor._tab_id
    state._slots.pop(conductor.key, None)
    newcomer = state.get_or_create_slot(conductor.key)
    assert newcomer is not conductor, "a different session object"
    assert newcomer.key == conductor.key, "wearing the same name"
    assert newcomer._tab_id and newcomer._tab_id != recorded_tab, "and its own identity"
    # Agent-created, like the conductor it replaced -- so it IS ownership-fenced and the
    # fence does read the manager for it. An unfenced newcomer would be served without
    # the manager being consulted at all, which is a different (and correct) story.
    newcomer._created_by = lead.key

    with pytest.raises(sc.SessionControlError) as exc:
        sc.require_manager_of(state, worker)
    assert exc.value.code == "manager_session_changed"

    # And through the gate, which is the reach that matters: the newcomer is refused
    # every verb a handoff widens.
    for operation in ("send", "read", "summary", "stop", "close", "adopt", "release"):
        with pytest.raises(sc.SessionControlError) as denied:
            sc.authorize_target(
                state,
                caller_session_key=_key(newcomer),
                target=worker.key,
                operation=operation,
            )
        assert denied.value.code == "manager_session_changed", operation

    # LOST REACH, never a widened boundary: the giver does not get it back either.
    with pytest.raises(sc.SessionControlError) as giver:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert giver.value.code == "manager_session_changed"


def test_an_unverifiable_occupancy_refuses_the_same_way(tmp_path):
    """No live session under the key, and a record with no occupancy at all.

    Both say the same thing -- the key may have been retaken and nothing here can show
    it was not -- so both refuse. The second is what an entry written before the field
    existed looks like, and treating it as "any occupant will do" is the hole wearing a
    back-compatible hat.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)

    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    state._slots.pop(conductor.key, None)
    with pytest.raises(sc.SessionControlError) as gone:
        sc.require_manager_of(state, worker)
    assert gone.value.code == "manager_session_changed", "no occupant to compare"

    state2 = _make_state(tmp_path / "second")
    _lead2, conductor2, worker2 = _team(state2)
    worker2._manager = conductor2.key
    worker2._manager_tab = ""  # an entry from before the field existed
    with pytest.raises(sc.SessionControlError) as legacy:
        sc.require_manager_of(state2, worker2)
    assert legacy.value.code == "manager_session_changed"


def test_the_handoff_that_fixes_a_retaken_key_is_not_itself_refused(tmp_path, monkeypatch):
    """The way out the refusal names has to be reachable.

    ``for_write`` lets the handoff read a manager it is about to replace, for the same
    reason an inadmissible grant is readable for a write: refusing it here would make
    the denial permanent and the instruction a lie.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    worker._manager = "chat-vanished"
    worker._manager_tab = "a-tab-nobody-holds"
    owner = state.get_slot("chat-owner-tab")

    _run(
        sc.handoff_target(
            state, caller_session_key=_key(owner), target=worker.key, to=conductor.key
        )
    )

    assert worker._manager == conductor.key
    assert worker._manager_tab == conductor._tab_id, "the new occupancy is published too"
    assert [call["op"] for call in calls].count("manager") == 1
    assert sc.require_manager_of(state, worker) == conductor.key


def test_a_closed_and_resumed_manager_is_the_same_session_and_still_resolves(tmp_path):
    """The regression the key-re-occupancy rule causes, pinned so it cannot arrive here.

    "Has this key been retaken" is the wrong question. The right one is "is the session
    now holding this key the SAME session the edge was created for", and a close followed
    by a resume answers yes: the resume reads the slot's own persisted line, and
    ``_read_tab_id`` keeps the ``tab_id`` it finds there rather than minting one. So the
    identity survives, the edge resolves, and the handoff stands.

    Refusing here would break a plain resume of the very session that was handed the
    worker -- lost reach for no ambiguity at all.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.require_manager_of(state, worker) == conductor.key, "the premise"

    # CLOSED and RESUMED: a fresh slot object under the same key, hydrated from the same
    # persisted line, which is what carries the identity across.
    persisted_tab = conductor._tab_id
    state._slots.pop(conductor.key, None)
    resumed = state.get_or_create_slot(conductor.key)
    assert resumed is not conductor, "a new object, as a resume builds"
    resumed._tab_id = persisted_tab  # what _read_tab_id does with the line's tab_id
    resumed._created_by = _lead.key

    assert sc.require_manager_of(state, worker) == conductor.key, "same session, resolved"
    assert not sc._managed_by_other(
        state, worker, resumed.key, resumed._tab_id
    ), "and it still reaches"
    for operation in ("send", "read", "summary", "stop", "close", "adopt", "release"):
        assert (
            sc.authorize_target(
                state,
                caller_session_key=_key(resumed),
                target=worker.key,
                operation=operation,
            )
            is worker
        ), operation


def test_the_identity_is_the_persisted_one_not_a_fresh_mint(tmp_path):
    """Why the comparison can be made at all without inventing storage.

    ``_tab_id`` is persisted (``metadata_codec`` writes ``tab_id`` and ``_read_tab_id``
    keeps it), so the identity a manager edge needs already exists on both sides: the
    handoff records it on its own ``session/manager_set`` entry and the slot carries it.
    A session built WITHOUT a line -- a genuinely new occupant of the name -- mints a
    different one, which is the whole of what distinguishes the two cases.
    """
    state = _make_state(tmp_path)
    _lead, conductor, _worker = _team(state)
    minted = conductor._tab_id
    assert minted, "a slot carries an identity"

    state._slots.pop(conductor.key, None)
    newcomer = state.get_or_create_slot(conductor.key)
    assert newcomer._tab_id and newcomer._tab_id != minted, (
        "a new session under the same name mints its own identity, which is what makes "
        "'different session' decidable without any new state"
    )


# ── The handed-over index: written after the entry, never in place of it ──────


def test_no_marker_is_written_for_a_handoff_whose_entry_did_not_land(tmp_path, monkeypatch):
    """A marker must never name a handoff the log does not hold.

    The marker makes the log authoritative for the slot at the next start. Written for an
    entry that was refused, or is still queued with its outcome unknown, it would point a
    restore at a log whose newest statement is not a committed handoff.
    """
    calls = _entries(monkeypatch, manager_lands=False)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    marked: list[str] = []

    async def _mark(_state, slot):
        marked.append(slot.key)
        return True

    monkeypatch.setattr(sc, "_record_handed_over", _mark)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "handoff_not_recorded"
    assert "manager" in [call["op"] for call in calls], "the entry was attempted"
    assert marked == [], "and no marker followed an entry that did not land"
    assert worker.key not in handed_over.handed_over_slots()[0]
    assert worker._manager == ""


def test_a_marker_that_cannot_be_written_refuses_the_handoff(tmp_path, monkeypatch):
    """Publishing without the marker hands the recipient a reach that vanishes at restart.

    An unmarked slot's log is never read at a start, so a grant published here would be
    gone after the next restart while the giver silently got its reach back -- exactly
    the revoked-grant-restored shape this verb exists to prevent. So the handoff is
    refused, nothing is published, and the grant is taken back on the log.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    async def _cannot_mark(_state, _slot):
        return False

    monkeypatch.setattr(sc, "_record_handed_over", _cannot_mark)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "handoff_not_recorded"
    assert worker._manager == "", "nothing was published"
    assert worker.key not in handed_over.handed_over_slots()[0]
    assert [call["op"] for call in calls].count("manager") == 2, "the grant, then taken back"
    assert sc.require_manager_of(state, worker) == lead.key, "the giver still holds it"


def test_the_marker_is_written_before_the_manager_is_published(tmp_path, monkeypatch):
    """ORDER: entry, then index, then publish. Nothing reaches disk ahead of its record."""
    order: list[str] = []
    real_entries = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    async def _mark(_state, slot):
        order.append(f"index (entries so far: {[c['op'] for c in real_entries]})")
        assert slot._manager == "", "the manager is not published before the index entry"
        assert handed_over.record_handed_over(slot.key), "what the real writer does"
        return True

    monkeypatch.setattr(sc, "_record_handed_over", _mark)
    _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )
    assert order == ["index (entries so far: ['adopt', 'manager'])"]
    assert worker._manager == conductor.key
    assert worker.key in handed_over.handed_over_slots()[0]


def test_nothing_on_the_slots_line_says_it_was_handed_over(tmp_path, monkeypatch):
    """The line is the transcript's, and its writers include the revoked party.

    A metadata key here would be cheap and writable by the one session with a motive:
    clear it and the slot comes up never-handed-over, which is served by its creator
    without reading anything. The fact belongs in the fenced crew log instead, and what
    is pinned here is that the line carries NOTHING about it -- no key of any spelling.
    """
    from kiro_crew.history import SLOT_OWNED_META_KEYS

    _entries(monkeypatch)
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    _hold(state, worker.key, _lead.key, sid="sid-worker")
    key = sc.slot_history_key(worker)
    state.conversation_log.update_metadata(key, {"title": "exists"})

    _run(
        sc.handoff_target(
            state, caller_session_key=_key(_lead), target=worker.key, to=conductor.key
        )
    )

    assert worker._manager == conductor.key, "the handoff committed"
    assert worker.key in handed_over.handed_over_slots()[0], "and the index says so"
    line = state.conversation_log.get_metadata(key) or {}
    assert not [name for name in line if "handed" in name or "manager" in name], (
        f"the line carries something about who manages this session: {sorted(line)}. "
        "Every key here is writable by the session whose grant a handoff revokes."
    )
    assert not [name for name in SLOT_OWNED_META_KEYS if "handed" in name]


def test_the_real_index_writer_records_the_slot(tmp_path):
    """Through the real index, so the stubs above are standing in for this."""
    state = _make_state(tmp_path)
    _lead, _conductor, worker = _team(state)

    assert _run(_REAL_RECORD_HANDED_OVER(state, worker)) is True
    assert worker.key in handed_over.handed_over_slots()[0]
    # Keyed by SLOT, so a second handoff of the same session writes nothing new.
    assert _run(_REAL_RECORD_HANDED_OVER(state, worker)) is True
    assert handed_over.handed_over_slots() == (frozenset({worker.key}), True)


def test_the_real_index_writer_refuses_a_slot_with_no_key(tmp_path):
    """A record nothing can attribute is one no reader can use, so it is not written."""
    state = _make_state(tmp_path)
    _lead, _conductor, worker = _team(state)
    worker.key = ""

    assert _run(_REAL_RECORD_HANDED_OVER(state, worker)) is False
    assert handed_over.handed_over_slots() == (frozenset(), True)


# ── The funnel's behaviour, not just its shape ───────────────────────────────


def test_a_resolve_is_never_observable_as_resolved_with_no_manager(tmp_path):
    """The defect the write-side funnel removes, driven rather than argued.

    The occupancy read is store IO, so the resolve suspends between learning who the
    log names and publishing it. With the flags cleared before that suspension, a
    request arriving inside it saw a slot that reads as resolved with an empty
    ``_manager`` -- and on a MARKED slot empty means "the creator holds it", so the
    giver that handed the session away was served the reach the log took from it.

    Sampled from inside the occupancy reader, which is the exact instant, and the
    sample is the authorization answer rather than the field: what matters is not which
    attribute held what, but whether the giver could be told it manages the session.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _handed_over(worker)
    # What the hydrate leaves on a marked slot: the log has an answer and this process
    # has not read it. That is the WHOLE statement the slot holds across the suspension.
    worker._manager_unresolved = True
    _hold(state, worker.key, lead.key, sid="sid-worker")
    samples: list[str] = []

    def _occupancy(_key: str, _tab: str) -> str:
        # Inside the suspension: the manager has been read from the log and nothing is
        # published yet. Whatever an arriving request would be told, it is told here.
        try:
            samples.append(sc.require_manager_of(state, worker) or "<creator>")
        except sc.SessionControlError as exc:
            samples.append(f"refused:{exc.code}")
        return conductor._tab_id

    moved = _run(
        sc.resolve_slot_manager(
            state,
            worker,
            slot_manager=lambda _k, _t: (conductor.key, True),
            slot_manager_tab=_occupancy,
        )
    )

    assert moved is True and worker._manager == conductor.key
    assert samples == ["refused:manager_unresolved"], (
        "a request arriving inside the occupancy read must be refused, not answered "
        f"from an empty manager: {samples}"
    )


def test_turning_emission_off_does_not_hand_a_marked_slot_back_to_its_creator(
    tmp_path, monkeypatch
):
    """Emission and readability are different facts, and only the second one decides.

    Reading ``emit.enabled()`` and treating a disabled emitter as the DECIDED answer
    "no handoff is recorded" is the inference this forbids. Switching emission off stops
    new entries and leaves every entry already on disk where it is, so on a slot whose
    line carries the marker that inference is contradicted by the slot itself -- and it
    hands the giver back a reach the log took away, on an ordinary operator config
    change.

    The reader is the one that answers now. It already separates "the store was never
    written" from "the store is there and could not be read", which are the decided and
    undecided cases, so there is no flag left to get this wrong.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _handed_over(worker)
    worker._manager_unresolved = True
    _hold(state, worker.key, lead.key, sid="sid-worker")
    # The operator's config change: emission off, every entry already on disk still
    # there. Reading this state as "no handoff is recorded" is what the test forbids.
    monkeypatch.setattr(emit, "enabled", lambda: False)
    asked: list[str] = []

    def _reader(key: str, _tab: str) -> "tuple[str, bool]":
        asked.append(key)
        return (conductor.key, True)

    _run(
        sc.resolve_slot_manager(
            state,
            worker,
            slot_manager=_reader,
            slot_manager_tab=lambda _k, _t: conductor._tab_id,
        )
    )

    assert asked == [worker.key], "the log is read for a marked slot either way"
    assert worker._manager == conductor.key, "and what it records is what is published"
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "not_creator", "the giver does not get the reach back"


def test_a_marked_slot_whose_log_cannot_be_read_stays_unresolved(tmp_path):
    """The undecided half of the same seam: refused, not answered from the creator."""
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    _handed_over(worker)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    assert _run(
        sc.resolve_slot_manager(state, worker, slot_manager=lambda _k, _t: ("", False))
    ) is (False)
    assert worker._manager_unresolved is True
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "manager_unresolved"


def test_a_recipient_replaced_during_the_marker_write_is_refused(tmp_path, monkeypatch):
    """The marker write is a SUSPENSION too, so the deciding check is after it.

    It takes the transcript's cross-process lock off the loop, which is the same kind of
    window as the append -- and the recipient's own routes take the recipient's slot lock
    and not this verb's. A recipient rebound to another principal's store inside that
    window, with the verdict formed before it, is handed a worker from a store it may not
    read. So the check that decides is the last one before the publish, with nothing
    suspending after it.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")

    async def _mark_then_rebind(_state, slot):
        # Durable, and the recipient moves to another principal's store inside it. The
        # earlier check passed; only a check after this one can see this.
        conductor.memory_store = "another-principals-store"
        return True

    monkeypatch.setattr(sc, "_record_handed_over", _mark_then_rebind)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "recipient_replaced"
    assert worker._manager == "", "nothing is published to a recipient that moved"
    assert [call["op"] for call in calls].count("manager") == 2, "and the grant is taken back"


def test_a_manager_rebound_to_another_store_stops_being_able_to_drive_the_slot(tmp_path):
    """The read side's staleness case, which the write side cannot cover.

    Store equality is checked when the grant is admitted and when a record is applied,
    and both are checks on the state at that moment. The manager's own routes can rebind
    its store afterwards, under the manager's slot lock and not the handed session's. A
    grant admitted while the stores agreed then keeps answering after they stop
    agreeing, and the next restart is what notices: the rebuild re-validates on exactly
    this comparison and drops the grant, so the reach is one the process holds and the
    log does not.

    Asked in the one predicate every reader passes through, so the drain re-asserting
    the admission gets it without a second manager read of its own.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.require_manager_of(state, worker) == conductor.key, "premise: it can drive it"

    conductor.memory_store = "another-principals-store"

    with pytest.raises(sc.SessionControlError) as exc:
        sc.require_manager_of(state, worker)
    assert exc.value.code == "manager_grant_inadmissible"
    assert "another principal" in str(exc.value)
    # Readable for the WRITE that fixes it, like every other refusal of a standing
    # record: the way out the message names is a fresh handoff, and that verb has to be
    # able to say what it is replacing.
    assert sc.require_manager_of(state, worker, for_write=True) == conductor.key


def test_the_drain_re_asserts_the_store_through_the_same_predicate(tmp_path):
    """The reader that found the shape, covered by the funnel rather than by a check.

    A queued cross-session delivery is re-authorized at the drain against the predicate
    that admitted it. Putting store equality inside that predicate is what makes the
    drain pick it up with no code of its own -- which is the property being pinned, not
    the drop itself.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    stamp = sc.send_origin_meta(state, conductor.key, manager_fenced=True)
    assert sc.sender_lost_its_reach(state, worker, stamp) is False, "premise: it still holds it"

    conductor.memory_store = "another-principals-store"

    with pytest.raises(sc.SessionControlError) as exc:
        sc.sender_lost_its_reach(state, worker, stamp)
    assert exc.value.code == "manager_grant_inadmissible", (
        "the drain raises rather than answering, and the caller HOLDS the entry: a "
        "grant that cannot be confirmed is not a reason to destroy queued text"
    )


# ── Atomic is not ORDERED: a read that lost the race publishes nothing ───────


def test_a_resolve_that_lost_to_a_handoff_does_not_publish_its_stale_answer(tmp_path, monkeypatch):
    """The defect the funnel EXPOSED rather than caused, and the next layer down.

    Writing the four attributes together stops a reader seeing half of two statements.
    It says nothing about a statement that was already old when it arrived. A resolve
    reads the crew log off the loop and then publishes; a handoff can commit inside
    that window -- its entry lands, its marker lands, its manager is published -- and
    the older read then publishes the manager the handoff just revoked. Four atomic
    writes of a stale fact are still a lost update.

    Driven at the real instant: the handoff runs from inside the resolve's occupancy
    read, which is the resolve's own suspension, so the ordering is forced rather than
    hoped for.
    """
    _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    # The log still records the LEAD as the manager, which is what the stale read sees.
    _handed_over(worker)
    worker._manager_unresolved = True
    child = _live(state, _slot(state, "chat-child"), "sid-child")
    child._created_by = conductor.key

    def _occupancy(_key: str, _tab: str) -> str:
        # Inside the resolve's suspension: a real handoff commits here, publishing a
        # manager this read has never heard of.
        if worker._manager != conductor.key:
            _handed_over(worker)
            publish = sc.publish_manager
            publish(
                state.get_slot(worker.key), manager=conductor.key, manager_tab=conductor._tab_id
            )
        return lead._tab_id

    moved = _run(
        sc.resolve_slot_manager(
            state,
            worker,
            slot_manager=lambda _k, _t: (lead.key, True),
            slot_manager_tab=_occupancy,
        )
    )

    assert moved is False, "the read that lost the race reports nothing moved"
    assert worker._manager == conductor.key, (
        "the newer statement stands; the stale read must not publish the manager the "
        f"handoff revoked, and it published {worker._manager!r}"
    )


def test_the_revision_is_what_discards_it_and_an_authority_always_wins(tmp_path):
    """The two halves of the guard, stated directly on the one writer.

    A caller that suspends gives the revision back and is refused if the fact moved.
    A caller that IS the authority -- the handoff publishing a grant it has already
    made durable, a refusal forcing the slot unresolved -- passes nothing and always
    wins, which is what puts the discard on the right side of the race.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)

    before = sc.manager_revision(worker)
    assert sc.publish_manager(worker, manager=conductor.key, manager_tab=conductor._tab_id) is True
    assert sc.manager_revision(worker) == before + 1, "stating the fact counts"

    # A stale publication, named as such, is refused and changes nothing.
    assert (
        sc.publish_manager(worker, manager="chat-stale", manager_tab="x", expect_revision=before)
        is False
    )
    assert worker._manager == conductor.key
    assert sc.manager_revision(worker) == before + 1, "a discard does not count either"

    # The same publication with the CURRENT revision lands.
    assert (
        sc.publish_manager(
            worker,
            manager="chat-stale",
            manager_tab="x",
            expect_revision=sc.manager_revision(worker),
        )
        is True
    )
    assert worker._manager == "chat-stale"

    # And an unconditional writer never loses.
    assert sc.publish_manager(worker, manager="", manager_tab="", unresolved=True) is True
    assert worker._manager_unresolved is True


# ── A key is not an identity, this time on the PERSISTED record ──────────────


def test_a_grant_does_not_follow_a_reused_name_into_another_session(tmp_path):
    """The finding: identity has to be on the RECORD, not only on the comparison.

    Hand W over, close it, and open a new session under the same name. The line is the
    name's line, so the marker is carried onto the replacement; the log is the name's
    store, so the old unit's grant is still in it, and the replacement's own unit holds
    no manager entry at all -- which is what made the walk fall through to the older
    one. The restart then hands a fenced manager a session nobody gave it.

    ``_manager_record_refusal`` cannot reach this. It compares the identity of the
    MANAGER when the value is read, and here the value was written for a different
    TARGET and nothing recorded which. So the entry names the target's incarnation and
    a read matches against the asking session's own.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    handed = worker._tab_id
    assert handed, "premise: the handed session can say which incarnation it is"

    def _log(_key: str, _tab: str) -> "tuple[str, bool]":
        # One entry, written for the session that WAS handed over.
        return (conductor.key, True) if _tab == handed else ("", True)

    # The replacement: same name, its own identity, the marker carried onto its line.
    replacement = state.get_or_create_slot(worker.key)
    replacement._tab_id = "a-new-incarnation"
    _handed_over(replacement)
    replacement._manager_unresolved = True

    _run(
        sc.resolve_slot_manager(
            state,
            replacement,
            slot_manager=_log,
            slot_manager_tab=lambda _k, _t: conductor._tab_id,
        )
    )

    assert replacement._manager == "", (
        "the replacement inherits nothing: the grant names the session that was handed "
        "over, and this is not it"
    )
    assert replacement._manager_unresolved is False, "and that is DECIDED, not unread"
    assert sc.manager_of(replacement) == replacement._created_by


def test_a_resume_of_the_same_session_keeps_its_grant(tmp_path):
    """The other direction, which the fix must not break.

    A resume is the SAME session: ``_tab_id`` is persisted and ``_read_tab_id`` keeps
    the one it finds, so the identity crosses the restart and the record it qualifies
    still applies. Refusing here would lose a handoff to an ordinary resume, which is
    the cost of using key re-occupancy as the test instead of identity.
    """
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    handed = worker._tab_id
    _handed_over(worker)
    worker._manager_unresolved = True

    _run(
        sc.resolve_slot_manager(
            state,
            worker,
            slot_manager=lambda _k, tab: ((conductor.key, True) if tab == handed else ("", True)),
            slot_manager_tab=lambda _k, _t: conductor._tab_id,
        )
    )

    assert worker._manager == conductor.key, "the same session keeps what it was handed"
    assert sc.require_manager_of(state, worker) == conductor.key


def test_a_slot_with_no_identity_stays_unresolved_rather_than_creator_held(tmp_path):
    """UNDETERMINABLE is not "no handoff". A session that cannot say which incarnation
    it is cannot be matched against a record, and answering "the creator holds it"
    there would hand a marked slot back on the strength of a missing identity."""
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _handed_over(worker)
    worker._tab_id = ""

    from kiro_crew.crew_log.session_tree import slot_manager as real_reader

    assert real_reader(worker.key, "") == ("", False), "the reader refuses to answer"

    _run(sc.resolve_slot_manager(state, worker, slot_manager=real_reader))

    assert worker._manager_unresolved is True
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "manager_unresolved"


def test_a_handoff_of_a_session_with_no_identity_is_refused(tmp_path, monkeypatch):
    """Refused rather than recorded weakly. A record nothing can attribute is one a
    later occupant of the name reads as its own, which is the defect itself."""
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    worker._tab_id = ""

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code == "handoff_not_recorded"
    assert [call["op"] for call in calls].count("manager") == 0, "nothing was written"
    assert worker._manager == ""


def test_an_entry_that_cannot_name_its_target_session_is_not_a_valid_entry(tmp_path):
    """The floor under the verb, at the level that actually holds it.

    The target's identity is a REQUIRED field of ``session/manager_set``, so a grant
    nothing can attribute is not a shape the log accepts at all -- not merely one the
    emitter declines to assemble. That is the difference between a rule two writers
    have to remember and a rule the store enforces on every writer there will ever be.
    """
    from kiro_crew.crew_log.entry_types import validate_data
    from kiro_crew.crew_log.errors import CrewLogError

    whole = {
        "slot": "chat-worker",
        "slot_tab": "a1b2c3d4e5f6",
        "manager": "chat-conductor",
    }
    validate_data("session", "session/manager_set", whole)

    with pytest.raises(CrewLogError) as exc:
        validate_data(
            "session", "session/manager_set", {k: v for k, v in whole.items() if k != "slot_tab"}
        )
    assert "slot_tab" in str(exc.value) and "required" in str(exc.value)

    # The revert carries it too: a transfer back is about one session as much as the
    # transfer was, and a correction applied to the wrong incarnation is the same hole
    # in the other direction.
    with pytest.raises(CrewLogError):
        validate_data(
            "session",
            "session/manager_set",
            {"slot": "chat-worker", "manager": "", "to_creator": True},
        )


def test_a_successful_correction_restates_the_manager_so_a_stale_read_is_discarded(
    tmp_path, monkeypatch
):
    """A correction moves the LOG. Saying nothing about it leaves an in-flight read armed.

    The refusal path that works is the dangerous one. A resolve reads the log and sees
    the handoff's entry, which did land; the handoff is then refused and its entry
    corrected, and the correction touches the log alone -- so the resolve's revision
    snapshot is still current and its publish passes the check, granting the recipient
    exactly the reach the refusal withheld. The failing correction path was already
    covered, because it publishes UNRESOLVED and that advances the revision by itself.

    So the working path restates the restored manager. The value is the one the slot
    already holds, which is why this reads as a no-op and is not one: advancing the
    revision is the whole effect.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    # Already handed over once, so there is a published manager for the correction to
    # restore -- the case where the slot's statement is not just "the creator holds it".
    first = _live(state, _slot(state, "chat-first-manager"), "sid-first-manager")
    first._created_by = lead.key
    conductor._created_by = first.key
    _handed_over(worker)
    publish = sc.publish_manager
    publish(worker, manager=first.key, manager_tab=first._tab_id)

    #: What a resolve that read the log BEFORE the correction is holding, and the
    #: revision it would hand back.
    stale_revision = sc.manager_revision(worker)

    # The marker write fails, so the handoff is refused after its entry landed, and the
    # correction below is the one that succeeds.
    async def _marker_fails(_state, _slot_):
        return False

    monkeypatch.setattr(sc, "_record_handed_over", _marker_fails)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(first), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "handoff_not_recorded"
    assert [call["op"] for call in calls].count("manager") == 2, "the entry and its undo"
    assert calls[2]["manager"] == first.key, "the correction restores the manager it had"

    # The slot still says what it said -- and the revision has MOVED, which is what a
    # read that started before the correction has to run into.
    assert worker._manager == first.key
    assert sc.manager_revision(worker) > stale_revision, (
        "a successful correction has to state something, or the read it raced with "
        "lands on top of it"
    )
    assert (
        sc.publish_manager(
            worker,
            manager=conductor.key,
            manager_tab=conductor._tab_id,
            expect_revision=stale_revision,
        )
        is False
    ), "the refused recipient is exactly what the stale read would publish"
    assert worker._manager == first.key, "and it does not get in"
    assert sc.require_manager_of(state, worker) == first.key


def test_a_rebuild_that_read_the_entry_of_a_refused_handoff_publishes_nothing(
    tmp_path, monkeypatch
):
    """The same race driven end to end, with the resolve doing the reading.

    The boot rebuild is a background task nobody awaits, so it genuinely overlaps a
    handoff. Here it reads the log from inside its own occupancy suspension, at the
    instant the handoff has appended its entry and is about to be refused.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _handed_over(worker)

    async def _marker_fails(_state, _slot_):
        return False

    monkeypatch.setattr(sc, "_record_handed_over", _marker_fails)

    # The rebuild's snapshot, taken before the verb runs at all.
    revision = sc.manager_revision(worker)

    with pytest.raises(sc.SessionControlError):
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    # What the overlapping rebuild would do with what it read: the entry named the
    # conductor, and it landed, so this is a truthful read of a superseded log.
    assert [call["op"] for call in calls].count("manager") == 2
    published = sc.publish_manager(
        worker,
        manager=conductor.key,
        manager_tab=conductor._tab_id,
        expect_revision=revision,
    )

    assert published is False, "a read of the superseded entry publishes nothing"
    assert worker._manager == "", "the creator still holds it, as the refusal says"
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(conductor), target=worker.key, operation="send"
        )
    assert exc.value.code == "not_creator", "the refused recipient is not authorized"


def test_only_the_record_maker_and_the_comparison_read_the_live_occupant(tmp_path):
    """ "Who holds this key now" and "who the record names" are different values.

    They are the same string in the ordinary case, which is what makes mixing them up
    quiet. A path that APPLIES a record has to use the identity the record carries; the
    only code entitled to read the live occupant is the one that MAKES the record, plus
    the comparison that exists to ask whether the two still agree.

    Reading the live occupant anywhere else writes a stranger's identity into the slot.
    It happened on the inadmissible restore: a fenced replacement sitting on a retired
    manager's key got its own tab published, a failed recovery handoff copied that into
    its correction, and the log then named the replacement -- control nobody handed over,
    one store rebind away. That is the mirror of putting identity on the record: a path
    that refuses must never rewrite what it refuses.

    What the rule is ABOUT is reading the occupant of SOMEONE ELSE's key. A caller that
    reads the occupant of its own key reads ITSELF, which is not the confusion this
    guards -- the two values it keeps apart are both about the manager. Admitted by the
    argument it passes, not by a name added to the list, so the distinction stays
    checkable instead of becoming an exemption that grows.
    """
    import ast

    #: ``_commit`` MAKES the record, so the live occupant is the value it records.
    #: ``_manager_record_refusal`` is the comparison itself -- reading the occupant is
    #: the whole question it asks. ``_gate`` is ``_commit``'s own other half: it reads
    #: the RECEIVER's occupant, which is the value the commit a few lines later
    #: records, and the two halves of one verb must not read it differently.
    permitted = {"_commit", "_gate", "_manager_record_refusal"}
    #: The asker's own key. A read of this is the caller identifying itself to the one
    #: comparison, and it cannot name a stranger: the caller is the occupant.
    own_key = {"caller_key"}
    offenders: list[str] = []
    for where, tree in _trees_mentioning("_tab_identity("):
        if where != "src/kiro_crew/dashboard/session_control.py":
            continue
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", "") != "_tab_identity":
                continue
            at = owner.get(node.lineno, "<module>")
            key_arg = node.args[1] if len(node.args) > 1 else None
            if isinstance(key_arg, ast.Name) and key_arg.id in own_key:
                continue
            if at not in permitted:
                offenders.append(f"{at}:{node.lineno}")

    assert not offenders, (
        "these sites read the LIVE occupant of a key instead of the occupancy the "
        f"record carries: {offenders}. A path that applies a record must publish the "
        "identity that record holds; only the handoff may read the live occupant, "
        "because that is where the record is made -- or an asker reading its own key, "
        "which is itself."
    )


def test_a_refusal_raised_before_authorization_names_nothing_about_the_target(tmp_path):
    """A fail-closed path must not become the way to read what it protects.

    These refusals come from the READ (``require_manager_of``), and the read runs BEFORE
    the ownership fence has decided the caller may reach this slot -- which is the whole
    reason the guard sits there, and also what makes the wording a disclosure surface. A
    fenced member that merely NAMES a stranger's worker gets one of these back, so a body
    carrying the target's title answers "what is that session called" and one carrying
    the manager's key answers "who is driving it". Being refused becomes the way to ask.

    Driven for every shape that can be refused this way, against a target the caller is
    not authorized for at all. The CODE is checked to be specific in the same breath: the
    caller still has to know whether to retry, and the fix must not have flattened that.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    #: What the caller must not learn, each planted as a value a leak would echo.
    worker.title = "Q3 revenue teardown"
    worker._titled = True
    _hold(state, worker.key, lead.key, sid="sid-worker")
    assert sc._caller_is_ownership_fenced(state, conductor.key), "premise: the asker is fenced"

    shapes = {
        "manager_unresolved": lambda: setattr(worker, "_manager_unresolved", True),
        "manager_grant_inadmissible": lambda: setattr(worker, "_manager_grant_inadmissible", True),
        "manager_session_changed": lambda: (
            setattr(worker, "_manager", conductor.key),
            setattr(worker, "_manager_tab", "the-session-that-was-given-it"),
        ),
    }
    secrets = {
        "the target's title": worker.title,
        "the target's key": worker.key,
        "the manager's key": conductor.key,
    }

    for expected_code, arrange in shapes.items():
        worker._manager_unresolved = False
        worker._manager_grant_inadmissible = False
        worker._manager = ""
        worker._manager_tab = ""
        arrange()
        with pytest.raises(sc.SessionControlError) as exc:
            # The asker holds nothing here: it did not create the worker and was never
            # handed it. It names the key and gets a refusal, which is the whole attack.
            sc.authorize_target(
                state,
                caller_session_key=_key(conductor),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == expected_code, "the code stays specific"
        body = exc.value.message
        leaked = [what for what, value in secrets.items() if value and value in body]
        assert leaked == [], (
            f"the {expected_code} refusal body leaks {leaked}: {body!r}. It is raised "
            "before the fence authorizes the caller for this target, so the wording is "
            "readable by anyone who can name a key. Keep the code specific and the "
            "prose generic."
        )


def test_no_manager_key_is_compared_without_the_identity_beside_it():
    """A manager KEY comparison never stands alone. Counted, because remembering failed.

    A slot key is a NAME, and the sessions answering to one name over a gateway's life
    are many: a close and a resume put a new session object under the same key, and the
    explicitly-named keys (``cron-{id}``, ``workflow-{run_id}``, a channel's own) are
    deterministic, so the name is handed on deliberately. "The manager's key equals the
    asker's key" therefore answers a question about names while the caller asked one
    about sessions, and the gap between the two is a grant a replacement inherits from
    the session that was given it and then lost it.

    This defect has come back FOURTEEN times on this line, each time as a new copy of
    the comparison rather than a new idea about it, which is why the fix is structural
    and not a review habit: there is one function that compares an identity against the
    record (:func:`manager_record_names`) and one predicate built on it
    (:func:`_managed_by_other`), and a key comparison is allowed only where one of them
    is also asked. The count this asserts is ZERO sites standing alone -- not a list of
    approved ones, because an approved-site list is what the fifteenth instance would be
    added to.

    What this does NOT cover is ``_created_by``, which is compared by key across the
    module and carries no recorded identity anywhere. That is sound for a different
    reason, not an oversight: it is written once at birth and never written again, so no
    grant moves through it and there is no revoked predecessor for a replacement to
    inherit from.
    """
    import ast

    reads = {"manager_of", "require_manager_of"}
    funnels = {"manager_record_names", "_managed_by_other"}

    def _reads_the_manager(node: ast.AST) -> bool:
        """Whether *node* evaluates who manages a slot, however it is spelled."""
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name in reads:
                return True
            # ``str(getattr(slot, "_manager", "") or "")`` and its variants.
            return any(_reads_the_manager(arg) for arg in node.args)
        if isinstance(node, ast.Attribute):
            return node.attr == "_manager"
        if isinstance(node, ast.Constant):
            return node.value == "_manager"
        if isinstance(node, ast.BoolOp):
            return any(_reads_the_manager(value) for value in node.values)
        return False

    alone: list[str] = []
    for where, tree in _trees_mentioning("_manager"):
        owner = _enclosing_functions(tree)
        asks_identity = {
            owner.get(node.lineno, "<module>")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", "") or getattr(node.func, "attr", "")) in funnels
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                continue
            sides = [node.left, *node.comparators]
            if not any(_reads_the_manager(side) for side in sides):
                continue
            # ``== ""`` asks WHETHER a manager is recorded, not WHICH session it is.
            if any(isinstance(side, ast.Constant) and side.value == "" for side in sides):
                continue
            at = owner.get(node.lineno, "<module>")
            if at not in asks_identity:
                alone.append(f"{where}::{at}:{node.lineno}")

    assert alone == [], (
        "these sites compare a manager KEY with no identity asked anywhere in the same "
        f"function: {alone}. A key names a slot and not a session, so the answer also "
        "fits a later occupant of that name. Ask through manager_record_names, or "
        "through _managed_by_other which is built on it -- do not write the comparison "
        "a second time, because the second copy IS this defect."
    )


def test_an_inadmissible_restore_publishes_the_recorded_occupancy(tmp_path, monkeypatch):
    """The finding, driven. A replacement on the retired manager's key must not be
    stamped into the slot -- and from there onto the log by the next failed handoff."""
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _handed_over(worker)
    worker._manager_unresolved = True
    recorded_tab = "the-session-that-was-handed-it"

    # The conductor's key is now held by a REPLACEMENT with its own identity, on another
    # store -- which is what makes the recorded grant inadmissible.
    conductor._tab_id = "a-fenced-replacement"
    conductor.memory_store = "another-principals-store"

    _run(
        sc.resolve_slot_manager(
            state,
            worker,
            slot_manager=lambda _k, _t: (conductor.key, True),
            slot_manager_tab=lambda _k, _t: recorded_tab,
        )
    )

    assert worker._manager == conductor.key, "the field carries what the log records"
    assert worker._manager_grant_inadmissible is True, "and that record cannot authorize"
    assert worker._manager_tab == recorded_tab, (
        "the occupancy is the RECORDED one; publishing the live occupant "
        f"({conductor._tab_id!r}) is what a later correction would write to the log"
    )
    # The giver is refused either way, which is the outcome this branch exists for.
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "manager_grant_inadmissible"


def test_a_correction_after_an_inadmissible_restore_records_no_stranger(tmp_path, monkeypatch):
    """The second half of the chain: what the compensation copies onto the log.

    The correction restores the occupancy the slot holds, so publishing a stranger's
    identity in the step above is what would persist it. With the recorded value
    published, the correction can only ever write an identity the log already had.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    _handed_over(worker)
    worker._manager_unresolved = True
    recorded_tab = "the-session-that-was-handed-it"
    conductor._tab_id = "a-fenced-replacement"
    conductor.memory_store = "another-principals-store"

    _run(
        sc.resolve_slot_manager(
            state,
            worker,
            slot_manager=lambda _k, _t: (conductor.key, True),
            slot_manager_tab=lambda _k, _t: recorded_tab,
        )
    )

    # The recovery handoff the refusal tells the caller to make, and it FAILS at its
    # marker write -- so its correction restores what the slot said.
    owner = state.get_slot("chat-owner-tab")
    fresh = _live(state, _slot(state, "chat-fresh-manager"), "sid-fresh")
    fresh._created_by = owner.key

    async def _marker_fails(_state, _slot_):
        return False

    monkeypatch.setattr(sc, "_record_handed_over", _marker_fails)

    with pytest.raises(sc.SessionControlError):
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(owner), target=worker.key, to=fresh.key
            )
        )

    corrections = [call for call in calls if call["op"] == "manager"]
    assert len(corrections) == 2, "the entry and its undo"
    assert corrections[1]["manager_tab"] == recorded_tab, (
        "the correction writes the occupancy the log already held; "
        f"{conductor._tab_id!r} must never reach it"
    )
    assert corrections[1]["manager_tab"] != conductor._tab_id


#: What each ``send_origin_meta`` call site is allowed to pass for
#: ``manager_fenced``, by ``function:argument source``, with the reason.
#:
#: ``manager_fence_applied`` is the RESOLVED verdict: ``send_target`` derives it once,
#: under its own name, so no closure can read the possibly-``None`` argument instead.
#: ``False`` is a literal claim that the fence did not apply, and only a verb whose
#: gate never reads the manager may make it.
_PERMITTED_FENCE_SOURCES = {
    "manager_fence_applied",
    "False",
}

#: The stamping functions whose gate authorizes without reading the manager, so their
#: stamp must say the fence did NOT apply, mapped to every operation they serve.
#: Checked against ``HANDOFF_WIDENED_OPERATIONS`` below rather than trusted, because
#: that set is what decides it -- one helper serves both ``retry`` and ``continue``, so
#: widening EITHER has to fail here.
_UNWIDENED_STAMPERS = {"_continue_turn_of_target": ("retry", "continue")}


def test_every_send_origin_stamp_records_a_resolved_verdict(tmp_path):
    """The stamp says whether the MANAGER fence applied. Every site must know.

    This is the third time one shape has come back on this line, and the first two
    times it was fixed site by site. The stamp licenses the drain to re-compare the
    sender against the manager, so a site that passes its own fence verdict instead of
    the resolved one claims a predicate that was never applied -- and the drain then
    drops an accepted delivery as ``sender_no_longer_manages`` after the API answered
    ok. The caller is told its message went; nothing went.

    Two sources are legitimate and nothing else is: the resolved verdict
    (``manager_fence_applied``, derived once) and the literal ``False`` from a verb
    whose gate never reads the manager. A raw ``caller_fenced`` is neither -- it is the
    CALLER's fence status, which is a different question from whether the fence applied
    to this admission.

    An enumeration, so a stamp added later is covered without anyone remembering: a new
    site passing anything else fails here.
    """
    import ast

    #: Pinned against the real set, so "retry does not read the manager" is read off
    #: the code rather than asserted in a comment that can go stale.
    for verb, operations in _UNWIDENED_STAMPERS.items():
        widened = [op for op in operations if op in sc.HANDOFF_WIDENED_OPERATIONS]
        assert not widened, (
            f"{verb} stamps manager_fenced=False because none of {list(operations)} is "
            f"widened by a handoff, so its gate never reads the manager. {widened} now "
            "is, and the stamp has to change with it."
        )

    found: list[tuple[str, str]] = []
    offenders: list[str] = []
    for where, tree in _trees_mentioning("send_origin_meta("):
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", "") != "send_origin_meta":
                continue
            at = owner.get(node.lineno, "<module>")
            if at == "send_origin_meta":
                continue  # its own definition
            arg = next((kw.value for kw in node.keywords if kw.arg == "manager_fenced"), None)
            source = (
                getattr(arg, "id", None)
                or (repr(arg.value) if isinstance(arg, ast.Constant) else None)
                or ast.dump(arg)
                if arg is not None
                else "<absent>"
            )
            found.append((f"{at}:{node.lineno}", source))
            if source not in _PERMITTED_FENCE_SOURCES:
                offenders.append(f"{where}::{at}:{node.lineno} passes {source}")

    assert found, "the enumeration found no stamp sites, so it is checking nothing"
    assert not offenders, (
        "these stamp sites pass something other than a resolved verdict: "
        f"{offenders}. Pass manager_fence_applied (the verdict derived once for this "
        "admission), or the literal False if this verb's gate never reads the manager. "
        "A raw caller_fenced is the CALLER's fence status, which is a different "
        "question, and stamping it lets the drain drop an accepted delivery against a "
        "predicate that never applied to it."
    )

    #: And the unwidened verbs specifically: theirs must be the literal, not the
    #: resolved verdict, because for them the answer is known without resolving
    #: anything -- the gate did not read the manager.
    for verb in _UNWIDENED_STAMPERS:
        sources = {source for site, source in found if site.startswith(f"{verb}:")}
        assert sources == {"False"}, (
            f"{verb} must stamp manager_fenced=False, because its gate authorizes on "
            f"_created_by alone and never reads the manager; it passes {sources}"
        )


def test_a_retried_turn_is_not_dropped_at_the_drain_for_its_sender(tmp_path):
    """The behaviour behind the stamp, at the reader that found it.

    A retry's admission never read the manager, so the drain has no manager relation to
    re-assert for it. With the stamp claiming otherwise, a handoff of the target between
    the retry and the drain deletes the recovery continuation -- and the caller was told
    the retry ran.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)

    # What a retry stamps: the fence did NOT apply to this admission.
    stamp = sc.send_origin_meta(state, lead.key, manager_fenced=False)
    assert sc.send_origin_sender(stamp) == lead.key, "the sender is still recorded"
    assert sc.send_origin_manager_fenced(stamp) is False

    # The target is handed to someone else, which moves exactly the relation the drain
    # would re-compare.
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.require_manager_of(state, worker) == conductor.key

    assert sc.sender_lost_its_reach(state, worker, stamp) is False, (
        "nothing is dropped: the retry was admitted on _created_by, which has not "
        "moved, so there is no manager relation to have lost"
    )

    # The same stamp with the claim it had before drops it, which is the defect.
    claimed = sc.send_origin_meta(state, lead.key, manager_fenced=True)
    assert sc.sender_lost_its_reach(state, worker, claimed) is True


def test_nothing_outside_the_crew_log_records_that_a_slot_was_handed_over(tmp_path):
    """One copy of the fact, in the store its subject cannot write.

    Two copies of this fact -- one durable, one in memory -- leave a gap a refusal can
    land in: the slot ends up denied on an object that does not say its log is worth
    reading. One copy makes that gap unconstructible, so what is worth pinning is that
    it STAYS one copy, in the crew log, and that no slot attribute or metadata key
    grows back to hold a second.

    A second copy on the slot's own line is the specific regression: the line is the
    transcript's, and the party whose grant a handoff revoked can write it, so a field
    there decides the whole authorization for anyone who reads it.
    """
    import ast

    forbidden = {"_manager_handed_over", "manager_handed_over", "handed_over_marker"}
    offenders: list[str] = []
    for where, tree in _trees_mentioning("handed_over"):
        if where == "src/kiro_crew/crew_log/handed_over.py":
            continue  # the index module itself, which IS the one copy
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in forbidden:
                offenders.append(f"{where}:{node.lineno} the string {node.value!r}")
            targets = (
                list(node.targets)
                if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
            )
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in forbidden:
                    at = owner.get(node.lineno, "<module>")
                    offenders.append(f"{where}::{at}:{node.lineno}")

    assert not offenders, (
        "these sites keep a second copy of 'was this slot handed over': "
        f"{offenders}. The crew log's handed-over index is the only store for it, "
        "because every other place a slot's own writers can reach -- its metadata line "
        "above all -- is one the session whose grant a handoff revoked can clear."
    )


def test_a_refusal_after_the_index_entry_leaves_the_slot_reading_its_log(tmp_path, monkeypatch):
    """The bypass, driven. The index entry is on disk, so the slot must read the log.

    The recipient changes during the index write and the compensation then fails. The
    refusal is correct and the slot goes unresolved -- but the index also has to carry
    it, because that entry landed and the grant may still be on the log. Without it the
    next authorization takes the unindexed creator-held fast path and never reads the
    record at all.
    """
    calls = _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    assert worker.key not in handed_over.handed_over_slots()[0], "premise: never handed over"

    async def _mark_then_replace_recipient(_state, slot_):
        # The durable index entry LANDS, and the recipient moves inside that same
        # window. Writing the real index is the point: what the slot must come out of
        # this with is "my log has to be read".
        assert handed_over.record_handed_over(slot_.key)
        conductor.memory_store = "another-principals-store"
        return True

    monkeypatch.setattr(sc, "_record_handed_over", _mark_then_replace_recipient)

    # And the correction cannot be made durable either, which is the pairing that
    # reaches `_take_back`'s unresolved branch.
    async def _no_commit(_emit, *, what="", **_kw):
        return False

    monkeypatch.setattr(emit, "awaiting_commit", _no_commit)

    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )

    assert exc.value.code in {"recipient_replaced", "handoff_unresolved"}
    assert worker._manager == "", "nothing was published to the replaced recipient"
    assert worker.key in handed_over.handed_over_slots()[0], (
        "the index entry landed, so this slot has to READ ITS LOG -- an unindexed slot "
        "is decided without reading anything, which hands the creator the reach back "
        "over a grant that may still be recorded"
    )
    assert worker._manager_unresolved is True, "and it is refused until that read lands"
    with pytest.raises(sc.SessionControlError) as denied:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert denied.value.code == "manager_unresolved"
    # ONE entry on the log and no correction behind it, which is the state that makes
    # the marker load-bearing: the grant naming the replaced recipient is what a start
    # would read, and only a marked slot reads anything.
    assert [call["op"] for call in calls].count("manager") == 1


def test_an_unreadable_index_refuses_the_manager_verbs_and_nothing_else(tmp_path, monkeypatch):
    """THE COST, stated as a test, because the difference is the whole argument.

    Keeping the fact off the slot's line costs an availability dependency: every
    manager-dependent answer needs the index readable, where an unindexed slot needs
    nothing else at all. That is deliberate, and what makes it the right trade is the
    BLAST RADIUS -- so both halves are pinned here, not just the refusal.

    The half a reader fears is the second one. "Fail closed on an unreadable index"
    sounds like "a corrupt file bricks the gateway", and if it did, the trade would be
    wrong: an availability cliff is worse than the reach it protects. It does not. A
    slot still hydrates, the sidebar still lists it, and every verb that does not read
    the manager still works. Only the verbs a handoff widens refuse, and they refuse
    with a retryable code, so repairing or removing the index clears it with no restart.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    assert sc._caller_is_ownership_fenced(state, lead.key), "premise: the giver is fenced"

    # An index that EXISTS and cannot be established. Not an absent one: absent is the
    # decided answer "no handoff was ever recorded", which is every first run.
    path = handed_over._index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe not valid utf 8")
    handed_over.reset_caches()
    assert handed_over.handed_over_slots() == (frozenset(), False), "premise: UNDECIDED"

    # HALF ONE: the gateway still builds slots. A hydrate does not raise, and the slot
    # it produces is a normal, listable session.
    from kiro_crew.dashboard.slot_persistence import metadata_codec as codec

    built = state.get_or_create_slot("chat-rebuilt-while-broken")
    codec.apply(
        state,
        built,
        {"created_by": lead.key, "title": "still here"},
        codec.Restore(name="chat-rebuilt-while-broken"),
        # The undecided answer its pass read, which is what a real hydrate would hand
        # in while the index is broken.
        handoff_index=handed_over.handed_over_slots(),
    )
    assert built.title == "still here", "the rest of the line still hydrates"
    assert state.get_slot(built.key) is built, "and the slot is in the session map"

    # HALF TWO: the verbs a handoff widens refuse, retryably.
    for operation in sorted(sc.HANDOFF_WIDENED_OPERATIONS):
        _run(sc.resolve_slot_manager(state, worker))
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state, caller_session_key=_key(lead), target=worker.key, operation=operation
            )
        assert exc.value.code == "manager_unresolved", operation
        assert exc.value.status == 409, "retryable: repairing the index clears it"

    # HALF THREE: every other verb is untouched. These never read the manager, so a
    # broken index is not their business.
    for operation in ("rename", "set_model", "retry", "end_wait"):
        assert (
            sc.authorize_target(
                state, caller_session_key=_key(lead), target=worker.key, operation=operation
            )
            is worker
        ), operation

    # And removing the bad file restores everything, with no restart: absence is the
    # decided empty answer, so the slot resolves to its creator again.
    path.unlink()
    handed_over.reset_caches()
    _run(sc.resolve_slot_manager(state, worker))
    assert worker._manager_unresolved is False
    assert (
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
        is worker
    )


def test_every_publication_the_resolve_derives_from_a_read_is_revision_guarded(tmp_path):
    """The enumeration, because "every read-derived publication" is a set, not a site.

    ``resolve_slot_manager`` publishes from five branches and every one of them is an
    answer computed from something read off the loop -- the index, then the log, then
    the occupancy. A publication that does not hand its revision back is one a handoff
    committing during that read can be overwritten BY, and the cheapest of them to miss
    are the two earliest: they publish a CLEARED statement, so a stale "never handed
    over" wipes a fresh manager and its flags rather than merely being stale.

    The verb's own publish and the compensation's are deliberately NOT here: they are
    the authority, not a reading of it, and they must win against any read in flight.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(sc.resolve_slot_manager))
    tree = ast.parse(source)
    unguarded = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "publish_manager"
        and not any(kw.arg == "expect_revision" for kw in node.keywords)
    ]
    assert not unguarded, (
        "these publications in resolve_slot_manager do not hand back the revision they "
        f"were derived from, at lines {unguarded} of its own source. Each is an answer "
        "computed off the loop, so a handoff that commits during that read is silently "
        "overwritten -- and for a branch that publishes an empty manager, overwritten "
        "means the revoked creator is served again."
    )

    #: And the revision is taken before the FIRST suspension, not before the log read.
    #: The index read is itself off the loop, so a capture after it leaves the two
    #: earliest branches reading a fact that may already have moved.
    body = ast.parse(source).body[0]
    assert isinstance(body, ast.AsyncFunctionDef)
    capture = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", "") == "revision" for target in node.targets)
    ]
    awaits = [node.lineno for node in ast.walk(body) if isinstance(node, ast.Await)]
    assert len(capture) == 1, capture
    assert capture[0] < min(awaits), (
        f"the revision is captured at line {capture[0]} but this function already "
        f"suspends at line {min(awaits)}. It has to be read before the first one."
    )


def test_a_handoff_during_the_index_read_is_not_wiped_by_the_older_verdict(tmp_path, monkeypatch):
    """The finding, driven at the instant it happens.

    A creator's FIRST handoff of a slot is the dangerous case, because before it the
    index does not name that slot -- so a startup resolve reading the index gets
    "never handed over", which is a verdict it publishes by CLEARING the manager and
    both refusal flags. Commit the handoff inside that read and the cleared statement
    lands on top of it: the slot reads as creator-held, and the creator is exactly the
    party the handoff revoked.
    """
    _entries(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    assert worker.key not in handed_over.handed_over_slots()[0], "premise: first handoff"

    real_was_handed_over = sc._was_handed_over

    def _read_then_a_handoff_commits(slot):
        answer = real_was_handed_over(slot)
        # Inside the resolve's own suspension: the handoff commits, publishing the
        # conductor and recording the slot in the index.
        if slot.key == worker.key and not worker._manager:
            _run(
                sc.handoff_target(
                    state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
                )
            )
        return answer

    monkeypatch.setattr(sc, "_was_handed_over", _read_then_a_handoff_commits)

    moved = _run(sc.resolve_slot_manager(state, worker))

    assert moved is False, "the stale read publishes nothing and says so"
    assert worker._manager == conductor.key, (
        "the handoff's own statement stands; an older 'never handed over' verdict must "
        f"not clear it, and it published {worker._manager!r}"
    )
    assert sc.require_manager_of(state, worker) == conductor.key
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "not_creator", "the revoked creator does not get it back"
