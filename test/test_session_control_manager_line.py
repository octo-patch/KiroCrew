"""Skip-level reach along the manager line, and where a closed manager's reports go.

``_manager`` holds a session's DIRECT manager and falls back to ``_created_by`` when
nothing has been handed over. Two things read the line above that first hop:

* the ownership fence in ``authorize_target``, for the operations a handoff widens
  only, so a caller that manages the target's manager (or manages that one's manager)
  reaches the target as well -- the skip level a lead needs to reach the worker its
  conductor runs. ``create``, ``fork`` and the memory delegation gate keep reading
  ``_created_by`` alone and are untouched by the walk;
* ``revert_manager_line``, run on the committed side of a close, which sends a closing
  manager's reports up to the manager above it so a finished conductor does not strand
  the workers it was running.

Every edge of the walk fails CLOSED -- an empty hop, a cycle, a hop whose slot is gone,
a line past the depth bound -- so a line that cannot be stated denies reach.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from chat_test_helpers import _make_state
from test_session_handoff import WIDENED as PR1_WIDENED
from test_session_handoff import _team as pr1_team

from kiro_crew.dashboard import chat_handlers
from kiro_crew.dashboard import session_control as sc
from kiro_crew.dashboard.chat_utils import slot_history_key


@pytest.fixture(autouse=True)
def _enabled(_floor_monkeypatch):
    """Session control AND the manager line on for every test here.

    The line is OFF in shipped config, so the tests that exercise it have to turn it
    on; ``TestTheLineOffIsTheFenceWithoutIt`` turns it back off to pin that default.

    ``_floor_monkeypatch`` rather than the shared ``monkeypatch``: the shared one is
    undone as a whole, so a test calling ``monkeypatch.undo()`` would lift this
    autouse patch with its own and read the real config switch.
    """
    _floor_monkeypatch.setattr(sc, "session_control_enabled", lambda: True)
    _floor_monkeypatch.setattr(sc, "manager_chain_enabled", lambda: True)
    # The occupancy and departure records are MODULE-level mutable state. Cleared
    # around every test in this file, because a record one test wrote would otherwise
    # resolve a hop another test deliberately left unresolvable -- which is a passing
    # test proving nothing.
    seen = sc._chain_switch_seen[0]
    _forget_everything()

    async def _save(state, slot, **kwargs):
        """Report every report-metadata save as landed.

        The repair's ORDERING is what these tests are about -- staged, saved, published --
        so the disk write is stubbed and the landing is the thing varied. The failure
        half overrides this per test.
        """
        return True

    _floor_monkeypatch.setattr(sc, "save_slot_off_loop", _save)
    yield
    # AND cleared again on the way out, with the switch put back where it was found.
    # Clearing only on entry protects this file from everything before it and nothing
    # after it from this file: a record left behind marks the next file's fresh slot
    # under a reused key as having taken that name from somebody, and the walk then
    # reads an ambiguous hop in a test that never created one.
    _forget_everything()
    sc._chain_switch_seen[0] = seen


def _forget_everything():
    """Both halves of the in-memory record, which is what a RESTART leaves behind.

    The occupancy table has to go with the departure table: it is written when a key is
    taken, so clearing only the departures would leave an occupancy no live process
    could have, and the walk would read the close window rather than the gap a fresh
    process sees.
    """
    sc._key_occupancy.clear()
    sc._departed_managers.clear()
    # The eviction shadow and its counter too. They are what tell a key the CAP dropped
    # from one never seen, so a mark left by an earlier test makes a later test's gap
    # read as an eviction -- which is a passing or failing test decided by ordering.
    sc._recently_evicted_keys.clear()
    sc._key_occupancy_evictions[0] = 0
    # The WEAK OCCUPANT MAP too, which is the one piece of this state that a previous
    # test can leave behind without leaving a record behind. It holds slot OBJECTS, so
    # an entry lingers until the garbage collector takes the object -- and while it
    # lingers, that object's own edges count as naming their keys, which is what decides
    # whether the cap may reclaim a record. Left alone, a test about what the cap drops
    # answers differently depending on when a previous file's objects were collected.
    # A fresh process has no occupants; every slot a test opens registers itself again.
    sc._live_occupants.clear()
    # And the boot-window flag. These tests describe a gateway whose closed hops HAVE
    # been read, which is every state except the first moments of a start; the one test
    # about that window sets it back to False itself.
    sc._departures_rebuilt[0] = True
    # The bookkeeping is GATED on the switch: with the line off nothing records,
    # because nothing reads what a record holds. These tests run with it on, so the
    # sync writers must see what an async reader would have published.
    sc._chain_switch_seen[0] = True


def _publish_manager(state, slot, manager_key):
    """Publish *slot*'s manager through the base PR's own writer.

    The base PR has ONE writer for the four fields that say who manages a session, and
    it records the OCCUPANCY of the manager's key beside the key itself -- a test that
    assigns ``_manager`` alone is not describing a published handoff but one whose
    occupancy is unknown, which refuses before this PR's walk is reached at all.
    """
    above = state.get_slot(manager_key)
    sc.publish_manager(
        slot,
        manager=manager_key,
        manager_tab=str(getattr(above, "_tab_id", "") or "") if above is not None else "",
    )
    return slot


def _churn_closed(state, count, *, start=0, prefix="chat-churn"):
    """Open AND CLOSE *count* tabs, which is the only churn the cap reclaims.

    The cap drops a departed hop's own settled line. It never drops a live session's
    occupancy -- that session could then not record its own departure, because the
    writer checks its generation against the key's -- so filler that stays open
    reclaims nothing and a test that wants an eviction has to close what it opens.
    """
    for index in range(start, start + count):
        key = f"{prefix}-{index}"
        slot = _running_as(state.get_or_create_slot(key), f"tab-{prefix}-{index}")
        state._slots.pop(key, None)
        sc.note_manager_departed(state, key, slot)


def _present_for_a_writer(state, start_key, wanted_key):
    """The WRITER's polarity on one walk: present unless the line was read to its end.

    The loop guard refuses on ``found or unread`` -- a line it could not finish leaves a
    loop possible, and a write would persist one. The fence reads ``found`` alone and is
    safe to be wrong. Two readings of one walk, spelled here so a test says which caller
    it means.
    """
    line = sc._walk_manager_line(state, start_key, wanted_key)
    return line.found or line.unread


def _hop_whose_record_the_cap_took(state, hop_key, *, creator):
    """Close *hop_key* and let the cap reclaim its record, leaving the gap behind.

    The cap can only take a record no open session's line REACHES: the keep rule
    follows the same chain the walk follows, up through the records that are kept, so
    while anything open can climb to this key its record is pinned. That is the point
    of the rule -- a dropped ancestor is what lets the next session to open under its
    name inherit the sessions the original was running.

    So the order here is a RESTORE's order: the hop closes, the cap takes its record
    while the gateway holds nothing that names it, and the sessions naming it are
    rebuilt afterwards. Those are the caller's to create; this leaves the gap.
    """
    hop = _running_as(state.get_or_create_slot(hop_key), f"tab-{hop_key}")
    hop._created_by = creator
    _depart(state, hop_key)
    _churn_closed(state, sc._KEY_OCCUPANCY_MAX)
    assert sc._key_occupancy.get(hop_key) is None, "the premise: the cap took the record"
    return hop


def _recorded_occupancy(state, tabs, manager_key):
    """What a ``session/manager_set`` entry records as the manager's OCCUPANCY.

    The identity of the session that was handed the slot, which the log keeps whether or
    not that session is still open -- so a closed manager has one too, and a chained
    departure still reads. Live slot first, then the persisted line the test describes,
    then the default that line would carry.
    """
    if not manager_key:
        return ""
    live = state.get_slot(manager_key)
    if live is not None:
        return str(getattr(live, "_tab_id", "") or "")
    history_key = f"dashboard:{manager_key}"
    return tabs.get(history_key, f"tab-{history_key}")


def _log_reader(dotted, answer):
    """A stub BOUND TO THE REAL SIGNATURE of the crew-log reader it stands in for.

    A one-argument stub for a two-argument reader is why a whole round of this feature
    shipped broken: production raised ``TypeError`` on every call, a wide handler read
    that as "nothing to restore", and these tests stayed green because the stub happily
    answered a call the real function could never accept. So the stub asks the real
    callable whether the call it just received is one production could make, and raises
    if it is not.
    """
    import importlib
    import inspect

    module_path, _, name = dotted.rpartition(".")
    signature = inspect.signature(getattr(importlib.import_module(module_path), name))

    def _stub(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return answer(*args, **kwargs)

    return _stub


def _entry(key):
    """*key*'s CURRENT occupant's whole entry, or None.

    The same two steps the walk takes -- resolve the occupancy, then match the pair
    exactly -- because a lookup by key alone is the thing the pair key exists to make
    impossible.
    """
    return sc._departed_managers.get((key, sc._key_occupancy.get(key)))


def _unidentifiable(key):
    """Whether *key*'s record names no occupant, so no edge above can match it.

    The record carries the departed occupant's own ``_tab_id``. ``""`` is the refusing
    value: an edge naming this key cannot be shown to have meant this occupant, so the
    walk stops rather than resolving through it.
    """
    entry = _entry(key)
    return entry is not None and not entry.tab


def _recorded(key):
    """What *key*'s current occupant reported to, or None when there is no entry.

    ``None`` is "no entry for whoever holds this key now" and ``""`` is "reported to
    nobody". Whether the entry is settled is :func:`_pending`'s question, because the
    two are different facts and a test that means one should not read the other.
    """
    entry = _entry(key)
    return None if entry is None else entry.manager


def _pending(key):
    """Whether *key*'s current occupant is inside its close, with its line unsettled."""
    entry = _entry(key)
    return entry is not None and entry.pending


def _depart(state, key):
    """Take *key* out of the live table the way every departure path does.

    Pops the slot and records what it reported to -- one mapping write, which is the
    whole of what a departure now does to the reporting line. The walk reads THROUGH
    this record; nothing rewrites any session's own manager.
    """
    slot = state.get_slot(key)
    state._slots.pop(key, None)
    sc.note_manager_departed(state, key, slot)
    return slot


def _outside(state, slot, caller_key, caller_tab=None):
    """``_outside_manager_line``, which the fence reaches only with the line ON.

    Returns the OUTSIDE verdict only; :func:`_refusal_cause` reads the reason beside it.

    *caller_tab* defaults to the caller's REAL identity, read the way the fence reads
    it, so a test that does not care about identity gets the production answer rather
    than a blank claim. A test about identity passes the claim it means.
    """
    tab = sc._tab_identity(state, caller_key) if caller_tab is None else caller_tab
    return sc._outside_manager_line(state, slot, caller_key, tab)[0]


def _refusal_cause(state, slot, caller_key, caller_tab=None):
    """Why the walk could not finish, which the refusal names: see ``_LineRead.cause``."""
    tab = sc._tab_identity(state, caller_key) if caller_tab is None else caller_tab
    return sc._outside_manager_line(state, slot, caller_key, tab)[1]


def _line(state, *keys: str):
    """Open ``keys`` as a reporting line, each one managed by the one before it.

    Returns the slots in the order given, so ``_line(state, "lead", "mid", "worker")``
    leaves ``worker`` managed by ``mid`` and ``mid`` managed by ``lead``.
    """
    slots = []
    for index, key in enumerate(keys):
        slot = state.get_or_create_slot(key)
        if index:
            slot._created_by = keys[index - 1]
        slots.append(slot)
    return slots


class TestManagerLineWalk:
    """``_walk_manager_line``: who is on the line above a session."""

    def test_the_direct_manager_is_on_the_line(self, tmp_path):
        state = _make_state(tmp_path)
        _, worker = _line(state, "chat-mid", "chat-worker")
        assert sc._walk_manager_line(state, sc.manager_of(worker), "chat-mid").found

    def test_the_manager_two_levels_up_is_on_the_line(self, tmp_path):
        state = _make_state(tmp_path)
        _, _, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        assert sc._walk_manager_line(state, sc.manager_of(worker), "chat-lead").found

    def test_a_handed_over_hop_is_followed_like_a_created_one(self, tmp_path):
        """``manager_of`` reads ``_manager`` first, so a handoff redirects the walk."""
        state = _make_state(tmp_path)
        _, mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        _running_as(state.get_or_create_slot("chat-other"), "tab-other")
        # Through the real writer: a published handoff records WHICH occupant of the
        # manager's key was handed the slot, and an edge that names none is refused.
        _publish_manager(state, mid, "chat-other")
        assert sc._walk_manager_line(state, sc.manager_of(worker), "chat-other").found
        assert not sc._walk_manager_line(state, sc.manager_of(worker), "chat-lead").found

    def test_a_session_off_the_line_is_not_on_it(self, tmp_path):
        state = _make_state(tmp_path)
        _, _, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        state.get_or_create_slot("chat-stranger")
        assert not sc._walk_manager_line(state, sc.manager_of(worker), "chat-stranger").found

    def test_a_report_is_not_on_its_own_managers_line(self, tmp_path):
        """The line runs one way: a worker never reaches the lead above it."""
        state = _make_state(tmp_path)
        _, mid, _ = _line(state, "chat-lead", "chat-mid", "chat-worker")
        assert not sc._walk_manager_line(state, sc.manager_of(mid), "chat-worker").found

    def test_an_empty_caller_key_is_on_no_line(self, tmp_path):
        state = _make_state(tmp_path)
        _, worker = _line(state, "chat-mid", "chat-worker")
        assert not sc._walk_manager_line(state, sc.manager_of(worker), "").found

    def test_a_gap_in_the_line_fails_closed(self, tmp_path):
        """A middle manager whose slot is gone stops the walk short of the lead."""
        state = _make_state(tmp_path)
        _line(state, "chat-lead", "chat-mid", "chat-worker")
        worker = state.get_slot("chat-worker")
        state._slots.pop("chat-mid", None)
        assert not sc._walk_manager_line(state, sc.manager_of(worker), "chat-lead").found

    def test_a_cycle_fails_closed_and_terminates(self, tmp_path):
        state = _make_state(tmp_path)
        first, second = _line(state, "chat-a", "chat-b")
        first._manager = "chat-b"
        second._manager = "chat-a"
        assert not sc._walk_manager_line(state, sc.manager_of(first), "chat-stranger").found
        assert not sc._walk_manager_line(state, sc.manager_of(second), "chat-stranger").found

    def test_a_line_past_the_depth_bound_fails_closed(self, tmp_path):
        state = _make_state(tmp_path)
        keys = [f"chat-{index}" for index in range(sc.LINEAGE_WALK_MAX_DEPTH + 3)]
        _line(state, *keys)
        bottom = state.get_slot(keys[-1])
        # Inside the bound the root is reachable; the root of a longer line is not.
        assert sc._walk_manager_line(state, sc.manager_of(bottom), keys[2]).found
        assert not sc._walk_manager_line(state, sc.manager_of(bottom), keys[0]).found


class TestOutsideManagerLine:
    """``_outside_manager_line``: the fence's own reading of the walk."""

    def test_the_direct_manager_is_inside(self, tmp_path):
        state = _make_state(tmp_path)
        _, worker = _line(state, "chat-mid", "chat-worker")
        assert not _outside(state, worker, "chat-mid")

    def test_the_skip_level_is_inside(self, tmp_path):
        state = _make_state(tmp_path)
        _, _, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        assert not _outside(state, worker, "chat-lead")

    def test_a_stranger_is_outside(self, tmp_path):
        state = _make_state(tmp_path)
        _, worker = _line(state, "chat-mid", "chat-worker")
        state.get_or_create_slot("chat-stranger")
        assert _outside(state, worker, "chat-stranger")

    def test_a_slot_managed_by_nobody_is_outside_every_line(self, tmp_path):
        """Fail-closed at the edge ``_created_by_other`` fails closed at."""
        state = _make_state(tmp_path)
        orphan = state.get_or_create_slot("chat-orphan")
        assert _outside(state, orphan, "chat-anyone")

    def test_a_self_managing_slot_is_read_as_a_cycle(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("chat-loop")
        _publish_manager(state, slot, "chat-loop")
        assert _outside(state, slot, "chat-stranger")


class TestTheFenceReadsTheLineForWidenedOperationsOnly:
    """``authorize_target`` wiring: which verbs the walk decides.

    The caller is agent-created, so it IS ownership-fenced -- the population the
    walk exists for. ``_created_by`` carries the line here, because a gateway where
    nothing has been handed over is the one a lead and conductor run on.
    """

    def _line(self, tmp_path):
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"  # agent-created, therefore fenced
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = "chat-lead"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-mid"
        return state, lead, worker

    @pytest.mark.parametrize("operation", sorted(sc.HANDOFF_WIDENED_OPERATIONS))
    def test_a_widened_operation_admits_the_skip_level(self, tmp_path, operation):
        state, lead, worker = self._line(tmp_path)
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation=operation,
            )
            is worker
        )

    @pytest.mark.parametrize("operation", ["create", "fork", "rename", "set_model", "retry"])
    def test_an_operation_the_handoff_does_not_widen_still_reads_the_creator(
        self, tmp_path, operation
    ):
        """Minting and the per-slot settings keep the pre-line answer, at every depth."""
        state, lead, worker = self._line(tmp_path)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation=operation,
            )
        assert exc.value.code == "not_creator"

    def test_a_session_off_the_line_is_still_refused_on_a_widened_operation(self, tmp_path):
        state, _, worker = self._line(tmp_path)
        stranger = state.get_or_create_slot("chat-stranger")
        stranger._created_by = "chat-owner"
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(stranger),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"

    def test_a_report_still_cannot_reach_up_the_line(self, tmp_path):
        """Upward send stays refused: reports do not command their manager."""
        state, lead, worker = self._line(tmp_path)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(worker),
                target=lead.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"

    def test_an_unfenced_caller_is_unaffected(self, tmp_path):
        """A person's own tab was never fenced, so the walk decides nothing for it."""
        state, _, worker = self._line(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(owner),
                target=worker.key,
                operation="create",
            )
            is worker
        )

    def test_a_closed_middle_manager_leaves_the_worker_reachable(self, tmp_path):
        """The close-revert and the fence together: a finished conductor strands nobody."""
        state, lead, worker = self._line(tmp_path)
        _depart(state, "chat-mid")
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
            is worker
        )

    def test_a_closed_middle_manager_without_the_revert_fails_closed(self, tmp_path):
        """The gap the revert repairs, so the test above is measuring the repair."""
        state, lead, worker = self._line(tmp_path)
        state._slots.pop("chat-mid", None)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_forgotten"


class TestAHandoffCannotCloseTheLineIntoALoop:
    """The writer-side guard the line needs, because the walk fails closed on a loop.

    A loop in the reporting line is not a half-broken state: every walk over it hits
    the cycle edge, so every session in the loop loses its manager at once. It is the
    handoff that can write one -- handing a session DOWN to one of its own reports --
    so the refusal belongs at the handoff.
    """

    def test_a_found_loop_says_so_rather_than_naming_an_unread_line(self, tmp_path):
        """The other half of the shared code: the message names the receiver's report."""
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = "chat-lead"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-mid"

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(lead),
                    target=mid.key,
                    to=worker.key,
                )
            )
        assert exc.value.code == "would_cycle"

    def test_handing_a_session_to_its_own_report_is_refused(self, tmp_path):
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = "chat-lead"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-mid"

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(lead),
                    target=mid.key,
                    to=worker.key,
                )
            )
        assert exc.value.code == "would_cycle"

    def test_handing_a_session_sideways_is_not_refused_as_a_loop(self, tmp_path):
        """The guard must not swallow the ordinary case it sits next to."""
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = "chat-lead"
        peer = state.get_or_create_slot("chat-peer")
        peer._created_by = "chat-lead"

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(lead),
                    target=mid.key,
                    to=peer.key,
                )
            )
        # Past the loop guard, refused further in because this gateway records no
        # session tree for the edge to move in.
        assert exc.value.code != "would_cycle"


class TestWhatAGiverKeepsAfterAHandoff:
    """The other half of ``test_session_handoff``'s "the reach moves" pin.

    A giver that still manages the RECEIVER keeps reaching what it handed over, by the
    line and not by the creation record -- it drives the receiver, so it can have the
    session driven either way. A giver that handed the session off its own line keeps
    nothing, which is the case pinned in that file.
    """

    def _handed(self, tmp_path):
        state = _make_state(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = owner.key
        conductor = state.get_or_create_slot("chat-conductor")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = lead.key
        _publish_manager(state, worker, conductor.key)
        return state, lead, conductor, worker

    def test_a_lead_still_reaches_a_worker_it_handed_to_its_own_conductor(self, tmp_path):
        state, lead, _conductor, worker = self._handed(tmp_path)
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
            is worker
        )

    def test_the_reach_is_the_line_and_not_the_creation_record(self, tmp_path):
        """Drop the lead off the conductor's line and the same lead is refused."""
        state, lead, conductor, worker = self._handed(tmp_path)
        _publish_manager(state, conductor, "chat-owner")
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"
        assert worker._created_by == lead.key

    def test_the_conductor_loses_the_worker_when_the_lead_hands_it_on(self, tmp_path):
        """A sideways handoff out of the line takes the middle manager's reach with it."""
        state, _lead, conductor, worker = self._handed(tmp_path)
        peer = state.get_or_create_slot("chat-peer")
        peer._created_by = "chat-owner"
        _publish_manager(state, worker, peer.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(conductor),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"


class TestWhatAnUnreadableLineCountsAs:
    """``unread_is_present``: one walk, two polarities, because two callers differ.

    A line the walk cannot finish must answer "no" to both questions asked of it, and
    those are opposite booleans. The fence asks "may this caller act" and needs False.
    The loop guard asks "is it safe to write this edge" and needs True. The flag is
    which way the walk's own ignorance points.
    """

    def test_an_exhausted_line_is_read_to_the_end(self, tmp_path):
        """Reaching an empty hop ends the line, so both polarities agree."""
        state = _make_state(tmp_path)
        _, worker = _line(state, "chat-mid", "chat-worker")
        start = sc.manager_of(worker)
        assert not sc._walk_manager_line(state, start, "chat-stranger").found
        assert not _present_for_a_writer(state, start, "chat-stranger")

    def test_a_found_key_is_found_under_either_polarity(self, tmp_path):
        state = _make_state(tmp_path)
        _, _, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        start = sc.manager_of(worker)
        assert sc._walk_manager_line(state, start, "chat-lead").found
        assert _present_for_a_writer(state, start, "chat-lead")

    def test_an_empty_wanted_key_is_on_no_line(self, tmp_path):
        state = _make_state(tmp_path)
        _, worker = _line(state, "chat-mid", "chat-worker")
        assert not _present_for_a_writer(state, sc.manager_of(worker), "")

    def test_a_gap_counts_as_read_because_that_line_is_already_broken(self, tmp_path):
        """Deliberate: a person closing their own tab leaves this gap in every tree.

        The gap is a key with NOTHING on record -- no occupancy, so no entry to read
        under one. That is what a key from before a restart looks like, and what a
        handoff below it must not be refused for.
        """
        state = _make_state(tmp_path)
        _line(state, "chat-lead", "chat-mid", "chat-worker")
        worker = state.get_slot("chat-worker")
        state._slots.pop("chat-mid", None)
        _forget_everything()
        start = sc.manager_of(worker)
        assert not sc._walk_manager_line(state, start, "chat-lead").found
        assert not _present_for_a_writer(state, start, "chat-lead")

    def test_a_pending_vacancy_is_unread_rather_than_a_line_that_ended(self, tmp_path):
        """The gap's near neighbour, and the OPPOSITE answer for a writer.

        A key popped by a close in flight has an occupancy whose manager is not
        published yet. Readers are denied either way; the writer must refuse HERE and
        may proceed over a gap, and that difference is what keeps a handoff from
        closing a cycle under a manager whose own edge is about to land.
        """
        state = _make_state(tmp_path)
        _line(state, "chat-lead", "chat-mid", "chat-worker")
        worker = state.get_slot("chat-worker")
        mid = state.get_slot("chat-mid")
        state._slots.pop("chat-mid", None)
        sc.note_key_vacated("chat-mid", mid)  # the pop's own record

        assert _pending("chat-mid")
        start = sc.manager_of(worker)
        assert not sc._walk_manager_line(state, start, "chat-lead").found
        assert _present_for_a_writer(state, start, "chat-lead")

    def test_a_closed_owner_tab_does_not_freeze_a_whole_tree(self, tmp_path):
        """The regression the gap decision avoids: handoffs under a closed tab still work."""
        state = _make_state(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        lead, mid, peer = _line(state, "chat-lead", "chat-mid", "chat-peer")
        lead._created_by = owner.key
        peer._created_by = lead.key
        _depart(state, "chat-owner")
        assert sc.manager_of(lead) == "chat-owner"  # the dangling hop
        assert not _present_for_a_writer(state, peer.key, mid.key)

    def test_a_cycle_splits_the_two_polarities(self, tmp_path):
        state = _make_state(tmp_path)
        first, second = _line(state, "chat-a", "chat-b")
        first._manager = "chat-b"
        second._manager = "chat-a"
        start = sc.manager_of(first)
        assert not sc._walk_manager_line(state, start, "chat-stranger").found
        assert _present_for_a_writer(state, start, "chat-stranger")

    def test_running_out_of_depth_splits_the_two_polarities(self, tmp_path):
        state = _make_state(tmp_path)
        keys = [f"chat-{index}" for index in range(sc.LINEAGE_WALK_MAX_DEPTH + 3)]
        _line(state, *keys)
        start = sc.manager_of(state.get_slot(keys[-1]))
        assert not sc._walk_manager_line(state, start, keys[0]).found
        assert _present_for_a_writer(state, start, keys[0])

    def test_the_fence_refuses_on_an_unread_line(self, tmp_path):
        """The reader's polarity through the fence: unread is outside, which fails closed."""
        state = _make_state(tmp_path)
        keys = [f"chat-{index}" for index in range(sc.LINEAGE_WALK_MAX_DEPTH + 3)]
        _line(state, *keys)
        assert _outside(state, state.get_slot(keys[-1]), keys[0])


class TestAHandoffRefusesALineItCouldNotRead:
    """The writer's polarity, reached through the verb.

    A line deeper than the bound reads as "the target is not above the receiver" to the
    fence, which is the same answer a genuinely unrelated receiver gives. Acting on that
    writes the very loop the guard exists to refuse, and the loop then fails closed for
    every session in it -- so the writer asks the same walk with the opposite polarity.

    One code, ``would_cycle``, for a found loop and an unread line alike, since what a
    caller does about either is name a different receiver.
    """

    def test_a_line_past_the_depth_bound_refuses_the_handoff(self, tmp_path):
        state = _make_state(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        keys = [f"chat-{index}" for index in range(sc.LINEAGE_WALK_MAX_DEPTH + 3)]
        _line(state, *keys)
        state.get_slot(keys[0])._created_by = owner.key
        top, bottom = state.get_slot(keys[0]), state.get_slot(keys[-1])

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(owner),
                    target=top.key,
                    to=bottom.key,
                )
            )
        assert exc.value.code == "would_cycle"

    def test_a_cycle_above_the_receiver_refuses_the_handoff(self, tmp_path):
        state = _make_state(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        lead, _mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = owner.key
        first, second = _line(state, "chat-a", "chat-b")
        first._manager = "chat-b"
        second._manager = "chat-a"

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(owner),
                    target=worker.key,
                    to=second.key,
                )
            )
        assert exc.value.code == "would_cycle"
        assert worker._manager == ""

    def test_a_gap_above_the_receiver_does_not_refuse_the_handoff(self, tmp_path):
        """The other side of the gap decision, at the writer this time."""
        state = _make_state(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        lead, _mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = owner.key
        orphan = state.get_or_create_slot("chat-orphan")
        # A CREATION edge to a name nobody holds, which is the gap this test is about.
        # Set as a bare ``_manager`` it would be a published handoff whose occupancy is
        # unverifiable, and that refuses for a different reason entirely.
        orphan._created_by = "chat-gone"

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(owner),
                    target=worker.key,
                    to=orphan.key,
                )
            )
        assert exc.value.code != "would_cycle"


class TestTheLineOffIsTheFenceWithoutIt:
    """With ``agent.session_manager_chain`` off, this feature answers nothing of its own.

    The shipped default, and the reason the default ships no behaviour change: the fence
    admits the DIRECT manager and nobody else, a departure hands nothing up, and a
    handoff is refused only for the reasons it was refused before. So these are the
    equivalence tests -- each one states what the gateway answers WITHOUT the feature,
    and the switch being off has to produce that same answer.
    """

    @pytest.fixture(autouse=True)
    def _line_off(self, _floor_monkeypatch):
        _floor_monkeypatch.setattr(sc, "manager_chain_enabled", lambda: False)

    def _team(self, tmp_path):
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = "chat-lead"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-mid"
        return state, lead, mid, worker

    def test_the_direct_manager_is_still_admitted(self, tmp_path):
        """The reach the fence grants on its own has to survive the switch being off."""
        state, _lead, mid, worker = self._team(tmp_path)
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(mid),
                target=worker.key,
                operation="send",
            )
            is worker
        )

    @pytest.mark.parametrize("operation", PR1_WIDENED)
    def test_pr1s_own_team_and_verb_list_answer_what_pr1_expects(self, tmp_path, operation):
        """Driven from the BASE PR's own fixture and verb tuple, not a restatement here.

        ``_team`` and ``WIDENED`` are imported from the base PR's test module, so the
        equivalence is measured against that PR's own idea of the tree and of which
        verbs a handoff widens. A verb added to that tuple, or a change to how that
        fixture builds the tree, fails here rather than silently going unchecked.

        The base PR's expectation for this shape is ``not_creator`` on every widened
        verb -- the conductor created none of the workers -- and with the line off it
        has to stay exactly that.
        """
        state = _make_state(tmp_path)
        _lead, conductor, worker = pr1_team(state)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(conductor),
                target=worker.key,
                operation=operation,
            )
        assert exc.value.code == "not_creator"

    def test_the_verb_list_this_file_gates_matches_the_base_prs(self):
        """A verb widened on one side and not the other would make the pin vacuous."""
        assert set(PR1_WIDENED) == set(sc.HANDOFF_WIDENED_OPERATIONS)

    @pytest.mark.parametrize("operation", sorted(sc.HANDOFF_WIDENED_OPERATIONS))
    def test_the_skip_level_is_refused_on_every_widened_operation(self, tmp_path, operation):
        """What the fence answered before the line existed: creator or holder, one level."""
        state, lead, _mid, worker = self._team(tmp_path)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation=operation,
            )
        assert exc.value.code == "not_creator"

    def test_the_fence_matches_the_direct_manager_predicate_exactly(self, tmp_path):
        """Equivalence through the REAL fence, over every caller/target pair in the tree.

        Asserted against ``_managed_by_other`` -- the predicate PR1 ships -- rather than
        against a restatement of what it does, so this cannot agree with a wrong idea of
        the old behaviour. Driven through ``authorize_target`` on a widened operation, so
        what is measured is the path a verb actually takes.
        """
        state, lead, mid, worker = self._team(tmp_path)
        for target in (lead, mid, worker):
            for caller in (lead, mid, worker):
                if caller.key == target.key:
                    continue  # self-target is refused by its own rule, before the fence
                refused = None
                try:
                    sc.authorize_target(
                        state,
                        caller_session_key=slot_history_key(caller),
                        target=target.key,
                        operation="send",
                    )
                except sc.SessionControlError as exc:
                    refused = exc.code
                assert (refused == "not_creator") == sc._managed_by_other(
                    state, target, caller.key, sc._tab_identity(state, caller.key)
                ), (
                    target.key,
                    caller.key,
                    refused,
                )

    def test_a_handed_over_session_still_reaches_its_new_manager(self, tmp_path):
        """A handoff keeps working with the line off -- that reach is PR1's, not this PR's."""
        state, _lead, mid, worker = self._team(tmp_path)
        _publish_manager(state, worker, mid.key)
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(mid),
                target=worker.key,
                operation="send",
            )
            is worker
        )

    def test_a_departure_hands_nothing_up(self, tmp_path):
        state, _lead, mid, worker = self._team(tmp_path)
        state._slots.pop(mid.key, None)

        assert worker._manager == ""
        assert worker._created_by == "chat-mid"
        assert not worker._dirty

    def test_a_close_hands_nothing_up(self, tmp_path, monkeypatch):
        """Through the close path, so the switch is read where the repair is called."""
        state, _lead, mid, worker = self._team(tmp_path)

        async def _archived(state_, slot_, name_, *, pre_pop_check=None):
            state_._slots.pop(name_, None)

        monkeypatch.setattr(chat_handlers, "_close_slot", _archived)
        asyncio.run(chat_handlers.close_slot(state, mid, mid.key))

        assert worker._manager == ""

    def test_handing_a_session_to_its_own_report_is_not_refused_as_a_loop(self, tmp_path):
        """The loop guard is gated with the reach it protects.

        With nothing walking the line a loop breaks nothing, and refusing anyway would
        reject a handoff the gateway accepts today.
        """
        state, lead, mid, worker = self._team(tmp_path)
        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(lead),
                    target=mid.key,
                    to=worker.key,
                )
            )
        assert exc.value.code != "would_cycle"


class TestTheWalkResolvesADepartedHopAtReadTime:
    """A departed manager is read THROUGH, and nothing rewrote a field when it left.

    This is the whole of what a departure does now: it records its own manager, once,
    synchronously, with the slot in hand. No session's own manager is rewritten, nothing
    is saved, and no loop visits other slots -- so there is no second writer to exclude,
    no report whose manager could change between a check and a write, and no promoted
    manager whose identity could go stale across a save.

    Held in memory rather than read back from the departed session's persisted metadata.
    That read is disk I/O and the walk is called from the ownership fence, which runs
    inside the close's no-suspension window; a metadata read there is the blocking-call
    class the fence is built to avoid.
    """

    def test_the_top_manager_still_reaches_the_worker_through_a_closed_middle(self, tmp_path):
        """The case the feature exists for, with no field rewritten anywhere."""
        state = _make_state(tmp_path)
        lead, mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = "chat-owner"

        _depart(state, "chat-mid")

        # Nothing was written to the worker: its line is read through the departure.
        assert worker._manager == ""
        assert worker._created_by == "chat-mid"
        assert not _outside(state, worker, lead.key)
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
            is worker
        )

    def test_two_departed_hops_in_a_row_are_both_read_through(self, tmp_path):
        state = _make_state(tmp_path)
        top, upper, mid, worker = _line(state, "chat-top", "chat-upper", "chat-mid", "chat-worker")
        top._created_by = "chat-owner"

        _depart(state, "chat-mid")
        _depart(state, "chat-upper")

        assert not _outside(state, worker, top.key)

    def test_an_unreadable_departure_refuses_rather_than_skipping(self, tmp_path):
        """No record for the departed hop: the walk STOPS, it does not step over it."""
        state = _make_state(tmp_path)
        lead, _mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = "chat-owner"

        _depart(state, "chat-mid")
        _forget_everything()  # the record is gone, as an eviction leaves it

        assert _outside(state, worker, lead.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"

    def test_a_departure_with_nobody_above_records_nobody(self, tmp_path):
        """A person's own unattributed tab has no hop to resolve to.

        An entry is still written, naming nobody, rather than left absent, because an
        ABSENT entry means something else: an occupant still inside its own close.
        """
        state = _make_state(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        assert sc.note_manager_departed(state, owner.key, owner) == ""
        assert _recorded(owner.key) == ""
        assert not sc._walk_manager_line(state, owner.key, "chat-anyone").found

    def test_the_record_is_bounded_and_evicts_oldest_first(self, tmp_path):
        state = _make_state(tmp_path)
        for index in range(sc._KEY_OCCUPANCY_MAX + 5):
            slot = state.get_or_create_slot(f"chat-gone-{index}")
            slot._created_by = "chat-lead"
            _depart(state, slot.key)
        assert len(sc._departed_managers) == sc._KEY_OCCUPANCY_MAX
        assert _recorded("chat-gone-0") is None
        assert _recorded(f"chat-gone-{sc._KEY_OCCUPANCY_MAX + 4}") == "chat-lead"

    def test_the_switch_off_reads_through_nothing(self, tmp_path, _floor_monkeypatch):
        """Off, the fence never walks, so a departed hop changes no answer at all."""
        _floor_monkeypatch.setattr(sc, "manager_chain_enabled", lambda: False)
        state = _make_state(tmp_path)
        lead, _mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = "chat-owner"

        _depart(state, "chat-mid")

        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"


class TestWhatTheDepartureRecordPromisesAndWhatItDoesNot:
    """The three conditions the in-memory record is accepted under.

    Each is a property of the record itself rather than of one scenario, so each gets
    its own test: it does not survive a restart and says so by refusing, it cannot grow
    without limit, and it can never name a manager the disk does not agree with.
    """

    def test_a_restarted_gateway_refuses_until_the_rebuild_has_run(self, tmp_path):
        """Condition 1, as it stands BEFORE the boot pass: no record, so no reach.

        Refused, not stepped over: nothing above the closed hop is admitted. What this
        pins is the WINDOW, not the end state -- the line comes back when the boot
        rebuild reads it off the log (``TestTheHandUpSurvivesARestart``), and until then
        a walk meeting the closed hop answers what a gateway without this feature
        answers. The loss denies and grants nobody anything, which is why the window is
        safe to have.
        """
        state = _make_state(tmp_path)
        lead, mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = "chat-owner"
        _depart(state, "chat-mid")
        assert not _outside(state, worker, lead.key)  # reachable before the restart

        # What a restart leaves: the slots rehydrate, the in-memory record does not.
        _forget_everything()

        assert _outside(state, worker, lead.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"

    def test_a_restart_does_not_widen_anything_either(self, tmp_path):
        """The other half of condition 1: losing the record never ADMITS a caller."""
        state = _make_state(tmp_path)
        lead, mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = "chat-owner"
        stranger = state.get_or_create_slot("chat-stranger")
        stranger._created_by = "chat-owner"
        _depart(state, "chat-mid")
        _forget_everything()

        for caller in (lead, stranger):
            assert _outside(state, worker, caller.key), caller.key

    def test_the_record_cannot_grow_without_limit(self, tmp_path):
        """Condition 2, by count: oldest departures are dropped first."""
        state = _make_state(tmp_path)
        for index in range(sc._KEY_OCCUPANCY_MAX + 20):
            slot = state.get_or_create_slot(f"chat-left-{index}")
            slot._created_by = "chat-lead"
            _depart(state, slot.key)

        assert len(sc._departed_managers) == sc._KEY_OCCUPANCY_MAX
        assert _recorded("chat-left-0") is None
        assert _recorded(f"chat-left-{sc._KEY_OCCUPANCY_MAX + 19}") == "chat-lead"

    def test_the_size_bound_is_not_the_walk_depth_bound(self):
        """Two different bounds, two numbers, so a change to one cannot silently cap the
        other: the walk limits how far ONE line is followed, this limits how many lines
        can have a departed hop at once."""
        assert sc._KEY_OCCUPANCY_MAX != sc.LINEAGE_WALK_MAX_DEPTH

    def test_a_dropping_rule_based_on_live_managers_would_break_a_chained_departure(self, tmp_path):
        """Why condition 2 is bounded by COUNT and not by liveness.

        With a line's middle AND upper hops both departed, no LIVE session's manager
        names the upper one -- only the departed middle one did. A rule that dropped
        entries nothing live names would delete exactly the hop the walk still needs,
        and the top manager would lose reach to the bottom session. This states that,
        so the tighter-looking rule is not introduced later as a simplification.
        """
        state = _make_state(tmp_path)
        top, _upper, _mid, worker = _line(
            state, "chat-top", "chat-upper", "chat-mid", "chat-worker"
        )
        top._created_by = "chat-owner"
        _depart(state, "chat-mid")
        _depart(state, "chat-upper")
        assert not _outside(state, worker, top.key)

        named_by_a_live_slot = {sc.manager_of(slot) for slot in state._slots.values()}
        assert "chat-upper" not in named_by_a_live_slot
        del sc._departed_managers[("chat-upper", sc._key_occupancy["chat-upper"])]

        assert _outside(state, worker, top.key)

    def test_the_recorded_value_is_the_one_the_disk_carries(self, tmp_path):
        """Condition 3. The record reads the PERSISTED manager and nothing else.

        ``manager_of`` reads ``_manager`` -- the field the metadata line carries -- and
        falls back to ``_created_by``. There is no second, in-flight manager field on a
        slot for it to pick up, so the record cannot name a manager the disk disagrees
        with. The assertion is on the SLOT's own shape rather than on a scenario,
        because that is what makes the guarantee structural: reintroduce a pending
        manager field and this fails, which forces whoever does it to confirm that the
        record still reads the committed one.
        """
        state = _make_state(tmp_path)
        lead, mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = "chat-owner"

        # ``_manager_unresolved`` is the base PR's "no decided manager has been read
        # from the log yet" FLAG, not a staged manager value: ``manager_of`` does not
        # read it, so the record still cannot name a manager the log disagrees with.
        # What it does mean is that an unresolved slot's creator is not a decided
        # answer, so the departure path refuses to record one at all -- pinned by
        # ``test_a_departure_before_the_manager_was_read_records_nothing``. Any OTHER
        # manager-named field reaching a slot has to come through here and say why.
        #
        # ``_manager_grant_inadmissible`` is the base PR's other FLAG: the log was read
        # and the manager it names may not be published. A bool, never a key, so
        # ``manager_of`` cannot pick it up and the record still names the committed
        # manager -- and the slots it describes are DENIED rather than resolved to a
        # creator, which is the direction a departure already refuses to record.
        pending_fields = [
            name
            for name in type(mid).__slots__
            if "manager" in name
            and name
            not in (
                "_manager",
                "_manager_unresolved",
                "_manager_grant_inadmissible",
                # Shape fields the handoff machinery carries, none a pending-manager
                # VALUE ``manager_of`` could read: the occupancy identity, the published
                # marker, and the revision counter. Listed so this grep pins the absence
                # of a second manager-VALUE field rather than tripping on these.
                "_manager_tab",
                "_manager_handed_over",
                "_manager_revision",
            )
        ]
        assert pending_fields == [], pending_fields

        # The committed fields, as a handoff leaves them: the key AND the occupancy
        # it was published to, which is what the base PR's one writer records.
        _publish_manager(state, mid, "chat-lead")
        _depart(state, "chat-mid")

        assert _recorded("chat-mid") == "chat-lead"
        assert not _outside(state, worker, lead.key)

    def test_the_committed_field_wins_over_the_creator(self, tmp_path):
        """A handed-over session records its MANAGER, not whoever opened it."""
        state = _make_state(tmp_path)
        giver = state.get_or_create_slot("chat-giver")
        holder = state.get_or_create_slot("chat-holder")
        for slot in (giver, holder):
            slot._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = giver.key
        _publish_manager(state, mid, holder.key)  # handed over before it departed
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-mid"

        _depart(state, "chat-mid")

        assert _recorded("chat-mid") == "chat-holder"
        assert not _outside(state, worker, holder.key)
        assert _outside(state, worker, giver.key)

    def test_the_record_is_written_only_at_departure(self, tmp_path):
        """Condition 3's other half: nothing else adds to it.

        A handoff, a close that does not remove the slot, and a plain reach check all
        leave it alone -- it is written by the departure paths and by nothing else.
        """
        state = _make_state(tmp_path)
        lead, mid, worker = _line(state, "chat-lead", "chat-mid", "chat-worker")
        lead._created_by = "chat-owner"

        _outside(state, worker, lead.key)
        worker._manager = "chat-lead"
        sc.manager_of(worker)

        assert not sc._departed_managers


class TestAPublishedEdgeToAClosedManagerRefusesAndIsNotReadFromTheRecord:
    """THE KNOWN LIMIT OF THIS FEATURE, pinned so it cannot be widened by accident.

    A session's own edge is read through the base PR's chokepoint, which compares the
    recorded manager against the session ANSWERING to that key now -- and a key with no
    live occupant answers with nobody, so the read refuses. The walk never starts: the
    hop below the closed manager is the target itself.

    The departure record is sitting right there, naming the manager above, and reading
    the identity out of IT instead is the one widening this feature does not make. The
    reason is what the record is: a statement about a session that no longer exists.
    Nobody can be asked to confirm it and no live occupant can contradict it, so an
    authorization resting on it rests on an input that can only be trusted, never
    checked -- which is the opposite of what the whole chokepoint is for.

    The refusal is a DELAY and not a dead end, and the last test here is that half: an
    edge pointing at a closed session is a revert that has not happened, and handing the
    session over again is what clears it. The fix belongs to the write, not to the read.
    """

    @pytest.fixture(autouse=True)
    def _zeroed(self):
        _forget_everything()
        yield
        _forget_everything()

    @staticmethod
    def _handed_then_closed(state):
        """A worker handed to a conductor, published, and then the conductor closes."""
        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        conductor = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-somebody-else"
        _publish_manager(state, worker, conductor.key)
        assert not _outside(state, worker, conductor.key), "the premise: the recipient reaches it"
        assert not _outside(state, worker, lead.key), "and the lead above it does too"
        _depart(state, "chat-conductor")
        return lead, conductor, worker

    def test_the_read_refuses_with_the_base_prs_own_code(self, tmp_path):
        state = _make_state(tmp_path)
        _lead, _conductor, worker = self._handed_then_closed(state)

        with pytest.raises(sc.SessionControlError) as exc:
            sc.require_manager_of(state, worker)
        assert exc.value.code == "manager_session_changed", exc.value.code

    def test_the_record_exists_and_is_still_not_what_answers(self, tmp_path):
        """Not a missing record. A record deliberately not read for this question.

        Without this the pin above would also pass on a gateway where the departure was
        never recorded at all, which is a different situation with the same refusal.
        """
        state = _make_state(tmp_path)
        lead, conductor, worker = self._handed_then_closed(state)
        held = sc._key_occupancy[conductor.key]

        assert sc._departed_managers[(conductor.key, held)].manager == lead.key
        assert sc._departed_managers[(conductor.key, held)].tab == "tab-c"
        with pytest.raises(sc.SessionControlError):
            sc.require_manager_of(state, worker)

    def test_nobody_reaches_the_worker_and_the_refusal_is_not_silent(self, tmp_path):
        """Refused for the lead, the closed conductor's own key, and a stranger alike.

        What must NOT happen is the fence answering "not your session" quietly, or worse
        admitting somebody: the read raises, so every caller gets the read's own refusal.
        """
        state = _make_state(tmp_path)
        lead, conductor, worker = self._handed_then_closed(state)
        stranger = state.get_or_create_slot("chat-stranger")
        stranger._created_by = "chat-owner"

        for caller in (lead, stranger):
            with pytest.raises(sc.SessionControlError) as exc:
                sc.authorize_target(
                    state,
                    caller_session_key=slot_history_key(caller),
                    target=worker.key,
                    operation="send",
                )
            assert exc.value.code == "manager_session_changed", (caller.key, exc.value.code)
        assert conductor.key not in state._slots, "and the conductor really is gone"

    def test_handing_the_worker_over_again_clears_it(self, tmp_path):
        """The way out, which is the write and not the read.

        The refusal's own message says to hand the session over again, and this is that
        instruction being true: one republished edge and the worker is reachable, with
        the closed hop no longer on its line at all.
        """
        state = _make_state(tmp_path)
        lead, _conductor, worker = self._handed_then_closed(state)

        _publish_manager(state, worker, lead.key)

        assert sc.require_manager_of(state, worker) == lead.key
        assert not _outside(state, worker, lead.key)
        assert (
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
            is worker
        )


class TestAReusedSlotNameDoesNotInheritAnOlderLine:
    """A slot name is reusable, so the record must be about whoever held it LAST.

    Opening a tab under a key a departed session used is ordinary -- the create route
    lets the caller choose the name -- so the record for that key can still name the
    previous occupant's manager. If a later departure under the same key returns without
    clearing it, the walk resolves the NEW departure through the OLD line, and the
    earlier session's manager gains authority over sessions created under the
    replacement that it never had while the replacement was live.
    """

    def test_an_unmanaged_departure_clears_the_previous_occupants_record(self, tmp_path):
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        first = state.get_or_create_slot("chat-conductor")
        first._created_by = lead.key

        # The lead's conductor departs: the record names the lead.
        _depart(state, "chat-conductor")
        assert _recorded("chat-conductor") == "chat-lead"

        # A person opens a tab under that same key. Unattributed: nobody manages it.
        replacement = state.get_or_create_slot("chat-conductor")
        assert sc.manager_of(replacement) == ""
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        # While the replacement is LIVE the lead is already refused: the walk reads the
        # live slot, which nobody manages.
        assert _outside(state, worker, lead.key)

        # And it must STAY refused once the replacement departs.
        _depart(state, "chat-conductor")

        assert _recorded("chat-conductor") == ""
        assert _outside(state, worker, lead.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_ambiguous"

    def test_a_self_managed_departure_also_clears_it(self, tmp_path):
        """The other early return out of the record, same reasoning."""
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        first = state.get_or_create_slot("chat-conductor")
        first._created_by = lead.key
        _depart(state, "chat-conductor")

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._manager = "chat-conductor"  # names itself, so there is no hop above
        _depart(state, "chat-conductor")

        assert _recorded("chat-conductor") == ""

    def test_a_managed_replacement_records_its_own_manager_not_the_older_one(self, tmp_path):
        """The ordinary case the clearing must not break: the newer line wins."""
        state = _make_state(tmp_path)
        old_lead = state.get_or_create_slot("chat-old-lead")
        new_lead = state.get_or_create_slot("chat-new-lead")
        for slot in (old_lead, new_lead):
            slot._created_by = "chat-owner"
        first = state.get_or_create_slot("chat-conductor")
        first._created_by = old_lead.key
        _depart(state, "chat-conductor")

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = new_lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        _depart(state, "chat-conductor")

        assert _recorded("chat-conductor") == "chat-new-lead"
        # NEITHER lead reaches it, and the NEW one losing reach is the tombstone's
        # price rather than a miss: the worker's edge names the KEY, the key has had two
        # occupants, and nothing in the edge says which it meant -- so resolving to the
        # newer one is as much a guess as resolving to the older. Reach is lost, no
        # boundary moves.
        assert _outside(state, worker, new_lead.key)
        assert _outside(state, worker, old_lead.key)


#: Deadline for every barrier the close-window tests wait on. Each await is bounded, so
#: a close that fails before signalling fails the test that is waiting, naming the
#: barrier, instead of hanging until the runner kills the worker it runs on. Generous
#: because nothing here does real work while it waits: a passing run never reaches it.
_BARRIER_TIMEOUT = 10.0


class TestALateCloseCannotSpeakForAReusedKey:
    """A close is not instantaneous, and the record has to survive that.

    Every departure path takes the slot out of the live table FIRST and then keeps
    awaiting -- a task cancel with its own timeout, then a save -- so the key is free
    for a window before the departure is recorded. A replacement minted in that window
    can open sessions of its own and depart in turn, and the original close then
    arrives LAST. Without a fence it overwrites the replacement's record, and the walk
    resolves the replacement's reports through the ORIGINAL session's manager: a
    manager with reach over sessions it never managed, which nothing takes back before
    the entry is evicted by count.

    The fence is the occupant's GENERATION, carried on the entry, so an older
    occupant's close is refused rather than believed.
    """

    @staticmethod
    def _lead_and_conductor(state):
        """A fenced lead, and the conductor it opened, ready to depart."""
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        conductor = state.get_or_create_slot("chat-conductor")
        conductor._created_by = lead.key
        return lead, conductor

    @staticmethod
    def _held_at_the_save(monkeypatch):
        """Hold the REAL close open at the archival save that follows its pop.

        Only the awaited save is swapped, so the pop, the vacancy recorded at it and
        the departure recorded on the committed side are all the production ones. A
        stub standing in for the whole close would pop without either, and would then
        pass whatever the fix was.

        Returns the two events: one the close sets on arrival, one that releases it.
        Both waits on them are bounded by the caller.
        """
        reached = asyncio.Event()
        release = asyncio.Event()

        async def _held(_state, _slot, **_kwargs):
            reached.set()
            await asyncio.wait_for(release.wait(), timeout=_BARRIER_TIMEOUT)
            return True

        monkeypatch.setattr(chat_handlers, "save_slot_off_loop", _held)
        return reached, release

    @staticmethod
    async def _in_the_window(state, slot, reached, release, during):
        """Run the real close, call *during* while it sits in its post-pop window.

        Every await is bounded, and the close task is cancelled and collected in
        cleanup, so a close that raises before arriving fails this test at the barrier
        rather than leaving a task for the next one.
        """
        closing = asyncio.ensure_future(chat_handlers.close_slot(state, slot, slot.key))
        try:
            await asyncio.wait_for(reached.wait(), timeout=_BARRIER_TIMEOUT)
            assert state.get_slot(slot.key) is None, "the close has not popped the key yet"
            during()
            release.set()
            await asyncio.wait_for(closing, timeout=_BARRIER_TIMEOUT)
        finally:
            if not closing.done():
                closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)

    def test_a_close_finishing_after_a_replacement_records_nothing(self, tmp_path, monkeypatch):
        """The finding, through the real close path with the window held open.

        The close pops the slot and then suspends, exactly where the task cancel and
        the save sit. A tab opens under the freed key, opens a worker and closes. Only
        then does the original close return and reach its record.
        """
        state = _make_state(tmp_path)
        lead, original = self._lead_and_conductor(state)
        reached, release = self._held_at_the_save(monkeypatch)
        opened: dict[str, object] = {}

        def _reuse_the_key():
            # A person reopens a tab under the freed key. Unattributed: nobody
            # manages it, and the lead has nothing to do with it.
            state.get_or_create_slot("chat-conductor")
            worker = state.get_or_create_slot("chat-worker")
            worker._created_by = "chat-conductor"
            _depart(state, "chat-conductor")
            opened["worker"] = worker

        asyncio.run(self._in_the_window(state, original, reached, release, _reuse_the_key))
        worker = opened["worker"]

        # The replacement's record stands: nobody above it, so the hop is unresolvable.
        assert _recorded("chat-conductor") == ""
        assert _outside(state, worker, lead.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_ambiguous"

    def test_the_previous_occupants_manager_is_refused_while_the_key_closes(
        self, tmp_path, monkeypatch
    ):
        """The other half of the window: the key is out of the table and the entry on
        it is still the PREVIOUS occupant's, because this one's manager is published
        only once its close commits.

        Reuse the key under a different lead, open a worker, and close it. Across the
        pop-to-commit window the first lead must not reach that worker -- it never
        managed the session that opened it.
        """
        state = _make_state(tmp_path)
        first_lead, first = self._lead_and_conductor(state)
        second_lead = state.get_or_create_slot("chat-second-lead")
        second_lead._created_by = "chat-owner"

        _depart(state, "chat-conductor")
        assert _recorded("chat-conductor") == "chat-lead"

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = second_lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        reached, release = self._held_at_the_save(monkeypatch)
        reached_in_the_window: dict[str, bool] = {}

        def _ask_who_reaches_the_worker():
            for lead in (first_lead, second_lead):
                reached_in_the_window[lead.key] = not _outside(state, worker, lead.key)

        asyncio.run(
            self._in_the_window(state, replacement, reached, release, _ask_who_reaches_the_worker)
        )

        # In the window the hop resolves to nobody, so BOTH are outside. What that
        # costs is the closing session's own line, for the length of its close.
        assert reached_in_the_window == {"chat-lead": False, "chat-second-lead": False}
        # Once the close commits, the record names the line the key actually had -- and
        # the hop is STILL refused, because this key has had two occupants and the
        # worker's edge names only the key. The record being right is not the same as
        # the edge being unambiguous.
        assert _recorded("chat-conductor") == "chat-second-lead"
        assert _unidentifiable("chat-conductor")
        assert _outside(state, worker, second_lead.key)
        assert _outside(state, worker, first_lead.key)
        assert first._created_by == "chat-lead"  # nothing was written to the first slot

    def test_a_late_close_does_not_overwrite_a_managed_replacements_record(self, tmp_path):
        """The same window, with the replacement handed to someone else.

        The replacement's own manager keeps the key, and the original's manager reaches
        nothing -- the fence refuses the write rather than picking between the two.
        """
        state = _make_state(tmp_path)
        old_lead, original = self._lead_and_conductor(state)
        new_lead = state.get_or_create_slot("chat-new-lead")
        new_lead._created_by = "chat-owner"

        state._slots.pop("chat-conductor", None)  # the pop the close does first

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = new_lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        _depart(state, "chat-conductor")

        # Only NOW does the original close reach its record.
        assert sc.note_manager_departed(state, "chat-conductor", original) == ""

        assert _recorded("chat-conductor") == "chat-new-lead"
        # NEITHER lead reaches it, and the NEW one losing reach is the tombstone's
        # price rather than a miss: the worker's edge names the KEY, the key has had two
        # occupants, and nothing in the edge says which it meant -- so resolving to the
        # newer one is as much a guess as resolving to the older. Reach is lost, no
        # boundary moves.
        assert _outside(state, worker, new_lead.key)
        assert _outside(state, worker, old_lead.key)

    def test_the_generation_decides_and_not_the_arrival_order(self, tmp_path):
        """Reversing the arrival order changes nothing, which is the whole point.

        In order, the replacement's record is the answer because it is written last.
        Out of order it is still the answer, because the answer is whichever entry
        belongs to the occupancy the key HAS -- read off the key, not off the sequence
        the two closes happened to finish in.
        """
        state = _make_state(tmp_path)
        _lead, original = self._lead_and_conductor(state)
        state._slots.pop("chat-conductor", None)
        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = "chat-new-lead"
        state._slots.pop("chat-conductor", None)

        for order in (
            (original, replacement),  # in order: the replacement records last
            (replacement, original),  # out of order: the replacement records first
        ):
            sc._departed_managers.clear()
            for slot in order:
                sc.note_manager_departed(state, "chat-conductor", slot)
            assert _recorded("chat-conductor") == "chat-new-lead"

    def test_the_same_occupant_departing_twice_still_records(self, tmp_path):
        """An equal generation is not an older one: a path that records twice is fine."""
        state = _make_state(tmp_path)
        _lead, conductor = self._lead_and_conductor(state)
        state._slots.pop("chat-conductor", None)

        assert sc.note_manager_departed(state, "chat-conductor", conductor) == "chat-lead"
        assert sc.note_manager_departed(state, "chat-conductor", conductor) == "chat-lead"
        assert _recorded("chat-conductor") == "chat-lead"

    def test_each_slot_object_is_a_later_occupant_than_the_one_before(self, tmp_path):
        """What the fence rests on: construction order, read off the slot.

        Two slot objects share a key only when the first has left the live table or is
        on its way out, because ``get_or_create_slot`` hands back the sitting occupant
        rather than building a second one. So construction order IS occupancy order,
        and a later object always carries the higher number.
        """
        state = _make_state(tmp_path)
        first = state.get_or_create_slot("chat-conductor")
        state._slots.pop("chat-conductor", None)
        second = state.get_or_create_slot("chat-conductor")

        assert sc._occupancy_generation_of(second) > sc._occupancy_generation_of(first)
        # The sitting occupant is handed back, never rebuilt, so its number is stable.
        assert state.get_or_create_slot("chat-conductor") is second

    def test_an_object_with_no_generation_records_as_it_always_did(self, tmp_path):
        """The fence is never PROVEN against an object that cannot say which occupant
        it is, so such a departure records normally rather than being dropped.

        Every real slot carries the field from its constructor, so this is the
        direction for test doubles and any future slot-like object, not for a slot.
        """
        state = _make_state(tmp_path)
        double = SimpleNamespace(key="chat-conductor", _manager="", _created_by="chat-lead")

        assert sc._occupancy_generation_of(double) == 0
        assert sc.note_manager_departed(state, "chat-conductor", double) == "chat-lead"
        assert _recorded("chat-conductor") == "chat-lead"

    def test_the_generation_is_never_persisted(self, tmp_path):
        """It compares two objects alive in ONE process, so the disk must not carry it.

        A restart begins the counter again at 1 while the departure record starts
        empty, so nothing is compared across processes. Persisting the number would put
        an old process's counter beside a new one's and make the fence answer nonsense.
        """
        from kiro_crew.dashboard.slot_persistence.metadata_codec import FIELDS

        persisted = {field.attr for field in FIELDS}
        assert "_occupancy_generation" not in persisted
        # Nor is the manager, as of the base PR: the crew log is the only place a
        # manager lives and the slot line carries none. So BOTH halves of what the
        # record reads are process-local, and the record is written from the published
        # attribute rather than from anything the disk could disagree with.
        assert "_manager" not in persisted


class TestAHandoffCannotWriteACycleUnderAClosingManager:
    """A pending vacancy must not read as the top of a line to the handoff guard.

    The shape: R reports to M, M reports to T. M closes, pops its key and awaits its
    archival. Inside that window the owner hands T to R. The guard walks up from R to
    see whether T sits above it: it reaches M, whose key is popped and whose manager is
    not published yet. Read as a line that ENDED, the walk says T is not above R and
    the handoff is allowed -- and when M's close commits and records T, the line is
    T -> R -> M -> T. Every manager above M is cut off, every walk over the loop fails
    closed, and the cycle is persisted to the manager log.

    The guard is the only check against this, so the answer has to come from the walk:
    a pending vacancy is a line the walk COULD NOT READ, and the writer's polarity
    turns that into a refusal.
    """

    @staticmethod
    def _tree(tmp_path):
        """``owner`` over ``top`` over ``middle`` over ``report``.

        ``owner`` is a person's own tab -- UNATTRIBUTED, so the ownership fence does
        not apply to it at all. That is deliberate and it is the finding's own
        scenario: with the fence out of the way the handoff loop guard is the only
        check standing between this call and a persisted cycle, which is what makes
        the guard's answer the whole of the defence. A fenced caller would be refused
        earlier, by the fence walking the same unreadable hop, and would hide the
        guard behind it.
        """
        state = _make_state(tmp_path)
        owner = state.get_or_create_slot("chat-owner")
        top = state.get_or_create_slot("chat-top")
        top._created_by = owner.key
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = top.key
        report = state.get_or_create_slot("chat-report")
        report._created_by = middle.key
        return state, owner, top, middle, report

    def test_a_handoff_under_a_closing_manager_is_refused_while_its_line_is_unknown(self, tmp_path):
        state, owner, top, middle, report = self._tree(tmp_path)

        # The middle manager's close: popped, its line not yet published.
        state._slots.pop("chat-middle", None)
        sc.note_key_vacated("chat-middle", middle)
        assert _pending("chat-middle")

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(owner),
                    target=top.key,
                    to=report.key,
                )
            )
        # NOT ``would_cycle``: nothing proved a loop here. The hop above the receiver
        # belongs to a session inside its close, so the honest answer is that it is not
        # known yet, and the caller is told to retry rather than told its handoff is
        # impossible. Both refuse; only this one is retryable.
        assert exc.value.code == "manager_line_pending", exc.value.code
        assert top._manager == ""
        # So the close commits into a line with no loop in it.
        assert sc.note_manager_departed(state, "chat-middle", middle) == "chat-top"
        assert not _pending("chat-middle")
        assert sc._walk_manager_line(state, sc.manager_of(report), top.key).found

    def test_a_proven_loop_is_still_told_apart_from_one_that_is_merely_unknown(self, tmp_path):
        """The other code, so the two refusals cannot quietly become one.

        Nothing is closing here: the receiver really does report up to the target, and
        a retry would answer the same way forever.
        """
        state, owner, top, middle, report = self._tree(tmp_path)

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(owner),
                    target=top.key,
                    to=report.key,
                )
            )
        assert exc.value.code == "would_cycle", exc.value.code

    def test_a_rolled_back_close_leaves_the_pre_close_chain_governing(self, tmp_path):
        """The refusal is self-clearing in BOTH directions the close can end.

        A close that fails puts the tab back in the live table. The walk then reads the
        manager off the live slot again and never consults the record, so the chain that
        governs is the one that governed before the close started -- and the handoff is
        refused as the proven loop it always was, not as something unknown.
        """
        state, owner, top, middle, report = self._tree(tmp_path)

        state._slots.pop("chat-middle", None)
        sc.note_key_vacated("chat-middle", middle)
        assert _pending("chat-middle")

        # The close fails and the slot goes back.
        state._slots["chat-middle"] = middle
        assert _pending("chat-middle")  # the record is still there, and now unread

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(owner),
                    target=top.key,
                    to=report.key,
                )
            )
        assert exc.value.code == "would_cycle", exc.value.code

    def test_no_cycle_reaches_the_manager_log_across_the_close_window(self, tmp_path):
        """(d). The refusal happens BEFORE anything is appended, not after.

        A cycle that reached the log would survive a restart, because the rebuild takes
        the newest entry per slot and the metadata line no longer carries a manager at
        all. So the test watches the emitter rather than the field: the handoff must be
        turned away without a single ``session/manager_set`` append.
        """
        from kiro_crew.crew_log import emit

        state, owner, top, middle, report = self._tree(tmp_path)
        state._slots.pop("chat-middle", None)
        sc.note_key_vacated("chat-middle", middle)

        appended = []

        def _watch(sid, **kwargs):
            appended.append((sid, kwargs))

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "on_manager_set", _watch)
            with pytest.raises(sc.SessionControlError) as exc:
                asyncio.run(
                    sc.handoff_target(
                        state,
                        caller_session_key=slot_history_key(owner),
                        target=top.key,
                        to=report.key,
                    )
                )

        assert exc.value.code == "manager_line_pending", exc.value.code
        assert appended == []

    def test_the_same_handoff_is_allowed_once_the_close_has_committed_to_nobody(self, tmp_path):
        """The case the refusal must not swallow: a line that really did end.

        A committed departure reporting to nobody leaves no hop above it, now or
        later, so an edge written below it cannot become a cycle. If this test ever
        fails together with the one above, the two states have been collapsed again.
        """
        state, owner, top, middle, report = self._tree(tmp_path)
        middle._created_by = ""  # reported to nobody, so no hop above it ever
        middle._manager = ""

        _depart(state, "chat-middle")
        assert _recorded("chat-middle") == ""
        assert not _pending("chat-middle")

        with pytest.raises(sc.SessionControlError) as exc:
            asyncio.run(
                sc.handoff_target(
                    state,
                    caller_session_key=slot_history_key(owner),
                    target=top.key,
                    to=report.key,
                )
            )
        # Past the loop guard, refused further in because this gateway records no
        # session tree for the edge to move in -- the same place the ordinary sideways
        # handoff lands. What matters here is that the guard did not claim a cycle.
        assert exc.value.code != "would_cycle"


class TestAHandoffThatCommitsAfterItsTargetDepartedCorrectsTheRecord:
    """A commit window outlives its request, so the slot can leave the table inside it.

    The owner hands an idle conductor C from P to R. While the manager entry is still
    being written, the idle sweep archives C: that departure records the COMMITTED
    manager, which is still P, because R is not published until the entry lands. The
    commit then publishes R onto a slot that is no longer in the live table.

    Left there, the walk resolves C's hop through the entry -- naming P, the manager the
    handoff was taking it away from -- so P keeps reach over C's workers and R, who was
    given them, is refused. Nothing corrects it until the entry is evicted or the
    gateway restarts. So the commit rewrites the entry for its own occupancy.
    """

    @staticmethod
    def _line(tmp_path):
        """``P`` over conductor ``C`` over worker ``W``, plus an unrelated ``R``."""
        state = _make_state(tmp_path)
        previous = state.get_or_create_slot("chat-previous")
        previous._created_by = "chat-owner"
        receiver = state.get_or_create_slot("chat-receiver")
        receiver._created_by = "chat-owner"
        conductor = state.get_or_create_slot("chat-conductor")
        conductor._created_by = previous.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        return state, previous, receiver, conductor, worker

    def test_the_new_manager_reaches_the_workers_and_the_old_one_does_not(self, tmp_path):
        state, previous, receiver, conductor, worker = self._line(tmp_path)

        # The archive lands first, recording the manager committed at that moment.
        _depart(state, "chat-conductor")
        assert _recorded("chat-conductor") == "chat-previous"

        # Then the handoff's own commit publishes the new manager on the departed slot.
        conductor._manager = receiver.key
        sc._record_departure(
            "chat-conductor",
            sc._occupancy_generation_of(conductor),
            sc.manager_of(conductor),
            pending=False,
        )

        assert _recorded("chat-conductor") == "chat-receiver"
        assert not _outside(state, worker, receiver.key)
        assert _outside(state, worker, previous.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(previous),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"

    def test_a_correction_for_an_occupancy_the_key_has_left_is_refused(self, tmp_path):
        """The same guard every other writer goes through: no key is resurrected.

        If the key was taken by a new occupant while the commit was in flight, the
        correction belongs to an occupancy nobody can read, so it must not be written.
        """
        state, _previous, receiver, conductor, _worker = self._line(tmp_path)
        _depart(state, "chat-conductor")
        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = "chat-someone-else"

        conductor._manager = receiver.key
        wrote = sc._record_departure(
            "chat-conductor",
            sc._occupancy_generation_of(conductor),
            sc.manager_of(conductor),
            pending=False,
        )

        assert wrote is False
        # The replacement has not departed, so the only entry under its occupancy is
        # the tombstone the retake left: it names nobody and marks the key as reused.
        assert _recorded("chat-conductor") == ""
        assert _unidentifiable("chat-conductor")

    def test_a_live_slot_is_not_the_case_the_correction_is_for(self, tmp_path):
        """The ordinary handoff, where the correction in the commit must not fire.

        Its condition is that the slot has LEFT the table. A slot still sitting in it is
        answered from the table, so an entry written for it would be a record nothing
        reads and one more thing to keep in step. Asserted on the condition itself,
        because that is what the commit branches on.
        """
        state, _previous, receiver, conductor, worker = self._line(tmp_path)
        _publish_manager(state, conductor, receiver.key)

        assert state.get_slot(conductor.key) is conductor  # the condition is False
        assert _recorded("chat-conductor") is None
        # And the live slot is what answers, so the new manager already has the reach.
        assert not _outside(state, worker, receiver.key)

    def test_the_correction_carries_the_receivers_identity_not_a_stale_one(self, tmp_path):
        """GPT 6.1 / Opus 5.5 finding: the correction must restate manager_tab too.

        The commit publishes the receiver onto the slot and, when the slot has left the
        table, rewrites the departure entry. Written with manager_tab left at _KEEP, the
        entry would pair the newly published manager with the PREVIOUS edge's identity --
        the pop's pending ``""`` or a creation edge's ``None`` -- so a later different
        occupant of this key would resolve the line through the entry, or the identity
        check on the receiver's key would be skipped. The correction must carry
        _edge_identity(slot), which the slot's just-published _manager_tab holds.
        """
        state, _previous, receiver, conductor, worker = self._line(tmp_path)

        # The slot has left the table, so the pop wrote a PENDING entry naming nobody --
        # the stale identity the correction must not keep.
        sc.note_key_vacated(conductor.key, conductor)
        state._slots.pop(conductor.key, None)

        # The commit publishes the receiver on the slot (manager AND its tab), then
        # corrects the entry with the identity it just published.
        sc.publish_manager(
            conductor, manager=receiver.key, manager_tab=sc._tab_identity(state, receiver.key)
        )
        wrote = sc._record_departure(
            conductor.key,
            sc._occupancy_generation_of(conductor),
            receiver.key,
            pending=False,
            manager_tab=sc._edge_identity(conductor),
        )

        assert wrote is True
        entry = _entry("chat-conductor")
        assert entry.manager == "chat-receiver"
        assert entry.manager_tab == sc._tab_identity(state, receiver.key), (
            "the corrected entry names the identity the handoff published, not the "
            "stale one the pop left -- so a later occupant cannot resolve through it"
        )
        # The receiver reaches the worker through the corrected hop; the previous
        # manager does not.
        assert not _outside(state, worker, receiver.key)
        assert _outside(state, worker, _previous.key)


class TestAnEntryAnswersForOneOccupantAndNoOther:
    """The three legs that close the eviction shape, each pinned on its own.

    The shape: an old close suspends after popping its key, a same-key replacement
    opens a worker and departs, the bound then evicts the replacement's entry, and the
    old close finally records. Keyed by SLOT KEY that last write is the answer for the
    replacement's worker, and the old manager reaches a session it never managed.

    Three things refuse it, and the first is the one that does not depend on any writer
    having been fenced correctly.
    """

    def test_an_older_generations_entry_never_answers_for_the_current_occupant(self, tmp_path):
        """Leg (a). The entry exists, under the same key, and is still not read.

        Planted by hand rather than raced into place, because the property under test
        is the LOOKUP: an entry filed under another generation is not reachable for the
        occupant holding the key, whatever put it there.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        first = state.get_or_create_slot("chat-conductor")
        first._created_by = lead.key
        stale_generation = sc._occupancy_generation_of(first)
        state._slots.pop("chat-conductor", None)

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = "chat-someone-else"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        _depart(state, "chat-conductor")

        # The old occupant's entry, planted under its own generation.
        sc._departed_managers[("chat-conductor", stale_generation)] = lead.key
        assert sc._key_occupancy["chat-conductor"] != stale_generation

        assert _outside(state, worker, lead.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_ambiguous"

    def test_an_absent_exact_entry_refuses_instead_of_falling_back(self, tmp_path):
        """Leg (b). Eviction drops the exact entry; the walk must not reach for another.

        The eviction is performed directly on the table, which is what the bound does
        to it, and the older occupant's entry is left in place as the thing a fallback
        would find.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        first = state.get_or_create_slot("chat-conductor")
        first._created_by = lead.key
        stale_generation = sc._occupancy_generation_of(first)
        state._slots.pop("chat-conductor", None)

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = "chat-someone-else"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        _depart(state, "chat-conductor")

        held = sc._key_occupancy["chat-conductor"]
        del sc._departed_managers[("chat-conductor", held)]  # what eviction leaves
        sc._departed_managers[("chat-conductor", stale_generation)] = lead.key

        assert _recorded("chat-conductor") is None
        assert _outside(state, worker, lead.key)
        # And the occupancy is still what refuses, so no later write can be read for it.
        assert sc._key_occupancy["chat-conductor"] == held

    def test_a_close_whose_generation_lost_the_key_writes_nothing(self, tmp_path):
        """Leg (c). The write-side refusal, so the stale entry is never created.

        The old close arrives last, after the replacement has taken the key, departed
        and had its entry evicted -- the exact ordering the finding needs. It must come
        away having written nothing at all.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        original = state.get_or_create_slot("chat-conductor")
        original._created_by = lead.key
        state._slots.pop("chat-conductor", None)  # the pop the close does first

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = "chat-someone-else"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        _depart(state, "chat-conductor")
        held = sc._key_occupancy["chat-conductor"]
        del sc._departed_managers[("chat-conductor", held)]  # the eviction

        # Only NOW does the original close reach its record.
        assert sc.note_manager_departed(state, "chat-conductor", original) == ""

        assert not [key for key, _ in sc._departed_managers if key == "chat-conductor"]
        assert _outside(state, worker, lead.key)

    def test_the_bound_is_unchanged_because_it_is_not_what_closes_this(self):
        """Raising the bound would narrow the window and leave the shape intact.

        Stated as a test so that a later change which "fixes" this by making the cache
        bigger has to come through here and say so.
        """
        assert sc._KEY_OCCUPANCY_MAX == 512


class TestTheRecordBoundsWhatItRetainsAndNotOnlyHowMany:
    """A COUNT bound does not bound memory if each entry can hold any length.

    The manager a departure records comes from ``manager_of``, which reads
    ``_created_by`` -- restored from history metadata with no length check of its own.
    So a resumed header naming a megabyte-long creator would be retained here, 512
    times over, after the slot itself is gone and whether or not anyone has the chain
    switched on.

    The value is a SLOT KEY, so the slot-key bound is the one that applies to it.
    """

    def test_a_manager_past_the_slot_key_bound_is_not_retained(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("chat-conductor")
        slot._created_by = "chat-" + "x" * sc._SLOT_KEY_MAX_CHARS
        state._slots.pop("chat-conductor", None)

        assert sc.note_manager_departed(state, "chat-conductor", slot) == ""
        assert _recorded("chat-conductor") is None
        # Nothing of the oversized string is held anywhere in the table.
        retained = [m.manager for m in sc._departed_managers.values()]
        assert not [m for m in retained if len(m) > sc._SLOT_KEY_MAX_CHARS]

    def test_the_hop_is_left_unresolved_rather_than_truncated(self, tmp_path):
        """Truncating would invent a different key -- either nobody, or someone else.

        So the hop has no entry and the walk refuses at it, which costs reach across
        one departed hop and grants nothing.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        oversized = "chat-" + "x" * sc._SLOT_KEY_MAX_CHARS
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = oversized
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = mid.key
        _depart(state, "chat-mid")

        assert _recorded("chat-mid") is None
        assert _outside(state, worker, oversized)
        assert _outside(state, worker, lead.key)

    def test_a_manager_at_the_bound_is_still_recorded(self, tmp_path):
        """The bound is a limit, not a ban: a legal slot key is retained as before."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("chat-conductor")
        slot._created_by = "c" * sc._SLOT_KEY_MAX_CHARS
        state._slots.pop("chat-conductor", None)

        assert sc.note_manager_departed(state, "chat-conductor", slot) == slot._created_by
        assert _recorded("chat-conductor") == slot._created_by


class TestAnEvictionIsCountedAndNamedRatherThanSilent:
    """The bound is reached by ORDINARY use, so what it costs has to be observable.

    Every slot construction takes a key, so a long-running gateway passes the cap with
    nothing unusual happening. The reach it drops is reach across one departed hop, and
    the refusal that produces is the same ``not_creator`` an ownership failure produces
    -- so a lead that has quietly lost its closed conductor's workers looks exactly like
    a caller that never had them. Counting the overflow and naming it is what separates
    a retention limit from an ownership decision.
    """

    @pytest.fixture(autouse=True)
    def _zeroed(self):
        sc._key_occupancy_evictions[0] = 0
        sc._recently_evicted_keys.clear()
        yield
        sc._key_occupancy_evictions[0] = 0
        sc._recently_evicted_keys.clear()

    @staticmethod
    def _churn(state, count):
        """Open AND CLOSE *count* tabs, which is the churn the cap reclaims.

        Only a departed hop's own line is reclaimable: the cap never drops a live
        session's occupancy, because that session could then not record its own
        departure, and never a reuse tombstone, because that is the one record saying a
        key changed hands. So a test about the cap has to close what it opens.
        """
        for index in range(count):
            key = f"chat-take-{index}"
            slot = _running_as(state.get_or_create_slot(key), f"tab-{index}")
            state._slots.pop(key, None)
            sc.note_manager_departed(state, key, slot)

    def test_passing_the_cap_counts_every_key_it_drops(self, tmp_path):
        state = _make_state(tmp_path)
        self._churn(state, sc._KEY_OCCUPANCY_MAX + 7)

        assert sc._key_occupancy_evictions[0] == 7
        assert len(sc._key_occupancy) == sc._KEY_OCCUPANCY_MAX

    def test_the_dropped_keys_are_remembered_so_a_refusal_can_name_the_cause(self, tmp_path):
        state = _make_state(tmp_path)
        self._churn(state, sc._KEY_OCCUPANCY_MAX + 3)

        assert "chat-take-0" in sc._recently_evicted_keys
        assert "chat-take-0" not in sc._key_occupancy
        # And the shadow is bounded by the same number as the table it shadows.
        assert len(sc._recently_evicted_keys) <= sc._KEY_OCCUPANCY_MAX

    def test_nothing_is_counted_when_the_cap_is_not_reached(self, tmp_path):
        state = _make_state(tmp_path)
        for index in range(5):
            state.get_or_create_slot(f"chat-take-{index}")

        assert sc._key_occupancy_evictions[0] == 0
        assert not sc._recently_evicted_keys

    def test_an_evicted_hop_refuses_and_says_it_was_retention(self, tmp_path, caplog):
        """The denial names the cause, which is the rule's own last requirement."""
        import logging

        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        _hop_whose_record_the_cap_took(state, "chat-upper", creator=lead.key)
        # The session that hop created arrives AFTER its record is gone, which is the
        # order a restore has and the only order in which the cap can have taken it.
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-upper"
        assert "chat-upper" in sc._recently_evicted_keys

        with caplog.at_level(logging.INFO, logger=sc.logger.name):
            assert _outside(state, worker, lead.key), "and refused after it"
        named = [r for r in caplog.records if "retention limit" in r.getMessage()]
        assert named, "the refusal must say it was retention, not ownership"
        assert "chat-upper" in named[0].getMessage()


class TestAReusedKeyMakesItsEdgesAmbiguousAndTheWalkRefuses:
    """The reuse case, visible where deleting the earlier entry used to erase it.

    A manager edge names a slot KEY. Once a key has had two occupants, an edge naming it
    no longer says which one it meant, and the walk has no way to find out: the edge
    carries a key and nothing else. Dropping the earlier occupant's entry stopped the
    OLD manager being inherited, but it also removed the only evidence the key had been
    reused -- so the walk resolved an ambiguous edge confidently through whoever held the
    key now.

    The entry for a new occupancy is therefore born TOMBSTONED, under the pair the walk
    looks up, and the mark survives that occupant's own departure. A marked hop is
    refused.
    """

    def test_the_original_workers_old_manager_is_refused_through_the_tombstone(self, tmp_path):
        """The finding's sequence: C opens W and closes, C is reused, handed on, closes.

        Without the tombstone the walk follows the replacement's departure and hands the
        replacement's manager reach over the ORIGINAL C's worker -- a session it never
        managed.
        """
        state = _make_state(tmp_path)
        first_lead = state.get_or_create_slot("chat-first-lead")
        first_lead._created_by = "chat-owner"
        unrelated = state.get_or_create_slot("chat-unrelated")
        unrelated._created_by = "chat-owner"

        conductor = state.get_or_create_slot("chat-conductor")
        conductor._created_by = first_lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key  # the edge names the KEY "chat-conductor"
        _depart(state, "chat-conductor")
        assert not _outside(state, worker, first_lead.key), "reachable before the reuse"

        # The key is reused and handed to an unrelated session, which then closes.
        replacement = state.get_or_create_slot("chat-conductor")
        replacement._manager = unrelated.key
        _depart(state, "chat-conductor")

        assert _unidentifiable("chat-conductor"), "the change of hands is on record"
        assert _outside(state, worker, unrelated.key), "the replacement's manager"
        assert _outside(state, worker, first_lead.key), "and the original's, both refused"
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(unrelated),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_ambiguous"

    def test_a_new_occupant_still_cannot_inherit_the_old_manager(self, tmp_path):
        """The purpose the deletion served, still served: no inheritance either way.

        A tombstone is not a manager, so nothing resolves through it. The replacement's
        own reports do not answer to the manager the earlier occupant had.
        """
        state = _make_state(tmp_path)
        old_lead = state.get_or_create_slot("chat-old-lead")
        old_lead._created_by = "chat-owner"
        first = state.get_or_create_slot("chat-conductor")
        first._created_by = old_lead.key
        _depart(state, "chat-conductor")
        assert _recorded("chat-conductor") == "chat-old-lead"

        replacement = state.get_or_create_slot("chat-conductor")
        assert _recorded("chat-conductor") == "", "the tombstone names nobody"
        assert sc.manager_of(replacement) == ""
        later_worker = state.get_or_create_slot("chat-later-worker")
        later_worker._created_by = replacement.key
        _depart(state, "chat-conductor")

        assert _outside(state, later_worker, old_lead.key)

    def test_the_mark_survives_the_new_occupants_own_departure(self, tmp_path):
        """The ambiguity belongs to the KEY, not to one occupant's close.

        If recording a manager cleared the mark, the hop would become resolvable again
        the moment the replacement closed -- which is exactly the sequence above.
        """
        state = _make_state(tmp_path)
        first = state.get_or_create_slot("chat-conductor")
        first._created_by = "chat-first-lead"
        _depart(state, "chat-conductor")

        replacement = state.get_or_create_slot("chat-conductor")
        replacement._created_by = "chat-second-lead"
        assert _unidentifiable("chat-conductor")
        _depart(state, "chat-conductor")

        entry = _entry("chat-conductor")
        assert entry.manager == "chat-second-lead", "the record is still written"
        assert _unidentifiable("chat-conductor"), "and no edge above can match it"

    def test_a_key_never_reused_is_untouched(self, tmp_path):
        """The ordinary case the tombstone must not reach: one occupant, reach kept.

        This is the feature's own headline case, so it is pinned next to the refusal it
        sits beside: a closed conductor that nobody replaced still hands its workers up.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        conductor = state.get_or_create_slot("chat-conductor")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        _depart(state, "chat-conductor")

        assert not _unidentifiable("chat-conductor")
        assert not _outside(state, worker, lead.key)

    def test_the_entry_count_does_not_grow_with_the_tombstone(self, tmp_path):
        """One entry per key-generation, as before: a retake replaces, never appends."""
        state = _make_state(tmp_path)
        for _ in range(6):
            state.get_or_create_slot("chat-conductor")
            _depart(state, "chat-conductor")

        for_this_key = [k for k in sc._departed_managers if k[0] == "chat-conductor"]
        assert len(for_this_key) == 1
        assert sc._KEY_OCCUPANCY_MAX == 512


class TestADepartureWaitsForTheManagerToBeRead:
    """The base PR's default-deny, carried into the record and not only into the verbs.

    ``_manager_unresolved`` says no decided manager has been read from the crew log for
    a slot yet. An empty ``_manager`` means "the creator holds it" only AFTER that read;
    before it, the creator is not an answer. So a slot that departs while unresolved
    records nothing, and a line through it refuses -- rather than the record turning
    "not known yet" into a decided answer that outlives the refusal.
    """

    def test_a_departure_before_the_manager_was_read_records_nothing(self, tmp_path):
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = lead.key
        mid._manager_unresolved = True  # hydrated from a metadata line, not yet read
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = mid.key

        assert _depart(state, "chat-mid") is mid
        assert _recorded("chat-mid") is None
        assert _outside(state, worker, lead.key)

    def test_the_same_departure_records_once_the_manager_has_been_read(self, tmp_path):
        """The flag cleared is the decided state, and then the record is written."""
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = lead.key
        mid._manager_unresolved = False
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = mid.key

        _depart(state, "chat-mid")
        assert _recorded("chat-mid") == lead.key
        assert not _outside(state, worker, lead.key)


class TestALiveHopWhoseManagerIsUnreadIsNotFollowed:
    """The live half of the unresolved-manager rule, which the verb's own check misses.

    ``authorize_target`` refuses a TARGET carrying ``_manager_unresolved``. It says
    nothing about a hop the walk passes THROUGH: the target can be perfectly resolved
    while its manager two levels up is not. Following that hop reads ``manager_of``
    down to ``_created_by`` -- the creator -- and a giver whose reach a recorded handoff
    already moved away is admitted over everything below it.
    """

    def test_the_creator_is_not_admitted_through_an_unread_live_hop(self, tmp_path):
        state = _make_state(tmp_path)
        creator = state.get_or_create_slot("chat-creator")
        creator._created_by = "chat-owner"
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = creator.key  # the log says it was handed to someone else
        middle._manager = ""
        middle._manager_unresolved = True  # and that read has not landed
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = middle.key

        assert _outside(state, worker, creator.key)
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(creator),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator"

    def test_the_same_line_is_followed_once_the_read_has_landed(self, tmp_path):
        """Cleared, the hop is ordinary again -- so this refuses reach, not the feature."""
        state = _make_state(tmp_path)
        creator = state.get_or_create_slot("chat-creator")
        creator._created_by = "chat-owner"
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = creator.key
        middle._manager_unresolved = False
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = middle.key

        assert not _outside(state, worker, creator.key)

    def test_an_unread_hop_refuses_a_handoff_as_retryable(self, tmp_path):
        """The writer's side: the guard must not read this line either."""
        state = _make_state(tmp_path)
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = "chat-creator"
        middle._manager_unresolved = True
        report = state.get_or_create_slot("chat-report")
        report._created_by = middle.key

        result = sc._walk_manager_line(state, report.key, "chat-anyone")
        assert result.unread and result.waits


class TestAnEvictedRecordIsUnreadForAWriterAndAGapIsNot:
    """Two keys with no occupancy on record, and the writer must tell them apart.

    A key the CAP dropped had a readable line a moment ago. A writer concluding
    "nothing sits above this hop" from a discarded record is how the loop guard writes
    an edge that becomes a cycle when the evicted session reopens under its old creator.

    A key never seen in this process is a different fact -- a tab closed before a
    restart -- and refusing that one would freeze every handoff underneath it. So the
    reader is denied by both and only the writer separates them.
    """

    def test_an_evicted_key_is_unread_for_the_writer(self, tmp_path):
        state = _make_state(tmp_path)
        # The hop the cap can take is one nothing open can climb to, so its record goes
        # before the session naming it is back -- a restore's order.
        _hop_whose_record_the_cap_took(state, "chat-upper", creator="chat-top")
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-upper"
        assert "chat-upper" in sc._recently_evicted_keys

        result = sc._walk_manager_line(state, sc.manager_of(worker), "chat-top")
        assert not result.found
        assert result.unread, "a writer must not read this as a line that ended"
        assert not result.waits, "and waiting does not bring an evicted record back"

    def test_a_key_never_seen_stays_read_to_the_end(self, tmp_path):
        """The gap, unchanged: a handoff below a pre-restart closed tab still works."""
        state = _make_state(tmp_path)
        _line(state, "chat-top", "chat-mid", "chat-worker")
        worker = state.get_slot("chat-worker")
        state._slots.pop("chat-mid", None)
        _forget_everything()

        result = sc._walk_manager_line(state, sc.manager_of(worker), "chat-top")
        assert not result.found
        assert not result.unread, "the gap is read to the end for both polarities"


class TestARetentionDenialReadsDifferentlyFromAnOwnershipOne:
    """The bound has to be OBSERVABLE, not merely enforced.

    The finding's sequence: close a conductor, cycle enough distinct tabs that the cap
    drops its record, and a lead that was authorized a moment ago starts being refused.
    Answered as an ownership failure, that refusal sends whoever is debugging it to look
    for a permission problem that does not exist -- the symptom appears far from the
    cause. So a retention denial names itself and the eviction count moves.

    A reused key is a third situation and reads differently again: forgotten means the
    record is gone, ambiguous means the record cannot say which occupant an edge meant.
    """

    @pytest.fixture(autouse=True)
    def _zeroed(self):
        _forget_everything()
        yield
        _forget_everything()

    @staticmethod
    def _closed_conductor(state):
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        conductor = state.get_or_create_slot("chat-conductor")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        _depart(state, "chat-conductor")
        return lead, worker

    @staticmethod
    def _a_hop_the_cap_forgot(state):
        """A line whose upper hop closed and had its record reclaimed by the cap.

        The eviction tests need this and the others do not. A record is kept while any
        open session's line can still climb to its key, so the eviction has to happen
        while nothing names it and the session below has to arrive afterwards -- which
        is a restore's order, and the only order that leaves this gap.
        """
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        _hop_whose_record_the_cap_took(state, "chat-upper", creator=lead.key)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-upper"
        return lead, worker

    def test_the_denial_names_retention_and_the_count_moved(self, tmp_path):
        state = _make_state(tmp_path)
        # The control first, on a line whose record is intact: the same shape resolves,
        # so what the assertions below are about is the eviction and not the shape.
        intact_lead, intact_worker = self._closed_conductor(state)
        assert not _outside(state, intact_worker, intact_lead.key), "the record resolves"
        _forget_everything()

        lead, worker = self._a_hop_the_cap_forgot(state)

        assert sc._key_occupancy_evictions[0] > 0, "the cap counted what it dropped"
        assert _refusal_cause(state, worker, lead.key) == "forgotten"
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_forgotten", exc.value.code
        assert "retention limit" in str(exc.value)

    def test_an_ownership_refusal_still_reads_as_one(self, tmp_path):
        """The control: a caller that was never on the line gets the ownership answer.

        Without this the new code could swallow every refusal and the distinction would
        be worthless in the other direction.
        """
        state = _make_state(tmp_path)
        _lead, worker = self._closed_conductor(state)
        stranger = state.get_or_create_slot("chat-stranger")
        stranger._created_by = "chat-owner"

        assert _refusal_cause(state, worker, stranger.key) == ""
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(stranger),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "not_creator", exc.value.code

    def test_forgotten_and_ambiguous_do_not_collapse_into_one_message(self, tmp_path):
        """Two refusals, two reasons: the record is GONE, or it cannot say which."""
        state = _make_state(tmp_path)
        lead, worker = self._closed_conductor(state)

        # Reused: the key is taken again AND that occupant departs, so the walk reaches
        # the record rather than a live slot and finds it tombstoned. While the
        # replacement is live the walk reads the live slot, which is a different case.
        state.get_or_create_slot("chat-conductor")
        _depart(state, "chat-conductor")
        assert _refusal_cause(state, worker, lead.key) == "reused"
        with pytest.raises(sc.SessionControlError) as reused:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )
        assert reused.value.code == "manager_line_ambiguous"
        assert "taken by a different session" in str(reused.value)
        assert "retention limit" not in str(reused.value)


class TestTheHandUpSurvivesARestart:
    """The condition the in-memory-only record failed, and it is the feature's own case.

    A conductor closes; its workers name that closed conductor as their manager. Falling
    back to the creation chain strands them, which is the bug this feature exists to fix.
    So a record that dies with the process undoes the feature on the next restart -- the
    scenario it fails is precisely the scenario it is for.

    Nothing new is written to make it durable. A handoff already publishes
    ``session/manager_set`` only once the entry is on disk and the entry outlives the
    close, so what a departed session reported to is already recorded. The boot pass
    reads it back for the keys a live slot's manager edge names, which is the one input
    the base PR's own rebuild already iterates.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    @staticmethod
    def _after_a_restart(state, *, says, decided=True, units=1, tabs=None, manager_tabs=None):
        """Run the boot pass with the log answering *says* for a departed key.

        *tabs* overrides the identity a key's persisted line names, and *manager_tabs*
        the occupancy its own manager entry recorded. Both default to a line that names
        the key -- which is what a real line does, and what the pass needs to ask the
        log about the session that ran under the name.
        """
        from kiro_crew.crew_log import emit

        tabs = tabs or {}
        manager_tabs = manager_tabs or {}
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: (
                    {"tab_id": tabs.get(history_key, f"tab-{history_key}")},
                    True,
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager_tab",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager_tab",
                    # A real ``session/manager_set`` entry records WHICH occupant of
                    # the manager's key was handed the session, so the default here is
                    # the live manager's own identity. Left blank, every restored edge
                    # is an unverifiable handoff and refuses -- which is correct, and
                    # not what these tests are about.
                    lambda key, tab: manager_tabs.get(
                        key, _recorded_occupancy(state, tabs, says.get(key, ("", False))[0])
                    ),
                ),
            )
            # UNDECIDED for anything *says* does not name, so the stub speaks only for
            # the keys a test is about. A decided-empty default would have the pass
            # record "departed, reported to nobody" for every unrelated key it climbs
            # to, which is correct behaviour and would make these counts say nothing.
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager",
                    lambda key, tab: says.get(key, ("", False)),
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.store.session_units_for_slot",
                _log_reader(
                    "kiro_crew.crew_log.store.session_units_for_slot",
                    lambda slot, strict=False: tuple(f"unit-{i}" for i in range(units)),
                ),
            )
            return asyncio.run(sc._rebuild_departures_from_log(state))

    def test_the_lead_reaches_the_workers_again_after_a_restart(self, tmp_path):
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"  # its manager closed before the restart

        # Nothing in memory: this is a fresh process with the slots rehydrated.
        assert _recorded("chat-conductor") is None
        assert _outside(state, worker, lead.key), "stranded before the pass"

        brought_back = self._after_a_restart(state, says={"chat-conductor": ("chat-lead", True)})

        assert brought_back == 1
        assert _recorded("chat-conductor") == "chat-lead"
        assert not _outside(state, worker, lead.key), "and reachable after it"

    def test_an_undecided_read_leaves_the_hop_refused(self, tmp_path):
        """A read that may have missed an entry must not be written down as a fact."""
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        brought_back = self._after_a_restart(state, says={"chat-conductor": ("chat-lead", False)})

        assert brought_back == 0
        assert _recorded("chat-conductor") is None
        assert _outside(state, worker, lead.key)

    def test_a_restart_does_not_call_a_reset_session_a_reused_key(self, tmp_path):
        """Unit COUNT is not reuse, and treating it as reuse cost the ordinary case.

        A slot gets a new crew-log unit on a reset, an agent or model switch and a
        provider swap, while its identity does not change. Counting units therefore
        tombstoned every closed conductor that had ever been reset, and a lead came back
        from a restart refused over its own workers. What a restart needs is the
        identity, and the persisted line carries it; a later DIFFERENT occupant of the
        key is still caught when it takes the key.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        self._after_a_restart(state, says={"chat-conductor": ("chat-lead", True)}, units=2)

        assert not _unidentifiable("chat-conductor"), "a reset is not a change of hands"
        assert not _outside(state, worker, lead.key), "so the lead reaches its workers"

    def test_a_rebuilt_key_can_still_be_resumed_by_the_session_that_ran_it(self, tmp_path):
        """A restored entry has to NAME its occupant, or the resume is a stranger.

        The pass files an entry for a key whose session has closed. If that entry says
        nothing about who ran under the key, the resume that takes the key next reads
        "an occupant nobody can identify" and is refused for the life of the process --
        which is the regression this correction exists to remove, arriving one restart
        later instead. The identity comes from the key's own persisted line, which is
        the same space the live check compares in; a crew-log unit id is not, and
        recording one made every restored hop read as a different session.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        brought_back = self._after_a_restart(
            state,
            says={"chat-conductor": ("chat-lead", True)},
            tabs={"dashboard:chat-conductor": "tab-conductor"},
        )

        assert brought_back == 1
        assert _entry("chat-conductor").occupant == "tab-conductor"
        assert not _outside(state, worker, lead.key), "restored, as the other tests pin"

        # And the person reopens that conductor's tab, which keeps its own identity.
        resumed = _running_as(state.get_or_create_slot("chat-conductor"), "tab-conductor")
        resumed._created_by = "chat-lead"

        assert not _outside(state, worker, lead.key), "its own key, its own identity"

    def test_the_pass_climbs_more_than_one_closed_hop(self, tmp_path):
        """Two closed hops in a row, which is the case a liveness rule would miss."""
        state = _make_state(tmp_path)
        top = state.get_or_create_slot("chat-top")
        top._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-mid"

        brought_back = self._after_a_restart(
            state,
            says={
                "chat-mid": ("chat-upper", True),
                "chat-upper": ("chat-top", True),
            },
        )

        assert brought_back == 2
        assert not _outside(state, worker, top.key)

    def test_the_check_itself_reads_only_memory(self, tmp_path):
        """Rule: durability is the write and the boot read, never the permission check.

        The walk is called from inside the close path's no-suspension window, so it must
        not touch the log. Asserted by making any log read raise and then walking.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        self._after_a_restart(state, says={"chat-conductor": ("chat-lead", True)})

        def _explode(*_a, **_kw):
            raise AssertionError("the permission check must not read the crew log")

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr("kiro_crew.crew_log.session_tree.slot_manager", _explode)
            assert not _outside(state, worker, lead.key)

    def test_the_boot_pass_is_wired_into_the_managers_rebuild(self, tmp_path):
        """The WIRING, not the pass: one rebuild, one gate, one place it is driven from.

        Calling the departure pass directly proves it works and proves nothing about
        whether anything calls it. This drives ``rebuild_managers_from_log`` -- the task
        the restore path already creates inside the armed window -- and asserts the line
        above a closed manager came back from it.
        """
        from kiro_crew.crew_log import emit

        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        assert _outside(state, worker, lead.key)

        # DECIDED-empty for the live slots, because this drives the real rebuild and an
        # undecided read is what leaves a slot ``_manager_unresolved`` -- the base PR's
        # default-deny, which then refuses the walk for a reason that has nothing to do
        # with what this test is about. Only the closed key carries a manager.
        says = {
            "chat-conductor": ("chat-lead", True),
            "chat-lead": ("", True),
            "chat-worker": ("", True),
        }
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager",
                    lambda key, tab: says.get(key, ("", False)),
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.store.session_units_for_slot",
                _log_reader(
                    "kiro_crew.crew_log.store.session_units_for_slot",
                    lambda slot, strict=False: ("u",),
                ),
            )
            # Every real line names the session that ran under it, and the pass needs
            # that name to ask the log about the right one.
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: ({"tab_id": "tab-conductor"}, True),
            )
            # And the entry's OCCUPANCY, which a real ``session/manager_set`` carries:
            # a restored handoff that names no occupant is unverifiable and refuses.
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager_tab",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager_tab",
                    lambda key, tab: lead._tab_id,
                ),
            )
            asyncio.run(sc.rebuild_managers_from_log(state))

        assert _recorded("chat-conductor") == "chat-lead"
        assert not _outside(state, worker, lead.key)


class TestAReusedCallerKeyIsNotTheSessionItNames:
    """The match point: the key a caller is RECOGNISED by is itself a reusable name.

    Every other refusal here is about a hop the walk passes through. This one is about
    the hop it stops at. A deleted lead's key taken by a new session would otherwise be
    accepted as that lead -- the walk hands over the whole line below it on the strength
    of a string -- and that holds whether the new occupant is live or has departed in
    turn, because reuse is a property of the NAME.
    """

    def test_a_new_occupant_of_an_ancestors_key_does_not_inherit_the_line(self, tmp_path):
        """The SKIP-LEVEL reach, which is the reach the walk decides and the chain adds.

        The reused key has to be an ANCESTOR rather than the direct manager. A direct
        manager is answered by ``_managed_by_other``, a string compare that never
        reaches the walk and that runs with the switch off -- so a new occupant of a
        key being accepted as its previous occupant's DIRECT reports is the base's own
        one-level behaviour, not something this walk can refuse. What the chain adds is
        everything above that level, and that is what this refuses.
        """
        state = _make_state(tmp_path)
        top = state.get_or_create_slot("chat-top")
        top._created_by = "chat-owner"
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = top.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = middle.key
        assert not _outside(state, worker, top.key), "the real top reaches it"

        # The top closes and someone else opens a tab under its key.
        _depart(state, "chat-top")
        impostor = state.get_or_create_slot("chat-top")
        impostor._created_by = "chat-someone-else"

        assert sc._key_is_ambiguous(state, "chat-top")
        assert _outside(state, worker, "chat-top"), "and the new occupant does not"
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(impostor),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_ambiguous", exc.value.code

    def test_it_holds_while_the_new_occupant_is_still_live(self, tmp_path):
        """Reuse is a property of the name, not of whether its holder is open."""
        state = _make_state(tmp_path)
        first = state.get_or_create_slot("chat-mid")
        first._created_by = "chat-top"
        _depart(state, "chat-mid")
        live_replacement = state.get_or_create_slot("chat-mid")

        assert state.get_slot("chat-mid") is live_replacement
        assert sc._key_is_ambiguous(state, "chat-mid")
        assert sc._walk_manager_line(state, "chat-mid", "chat-mid").cause == "reused"

    def test_a_key_with_one_occupant_still_matches_immediately(self, tmp_path):
        """The ordinary case: this must cost nothing to a key nobody reused."""
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = lead.key

        assert not sc._key_is_ambiguous(state, "chat-lead")
        assert sc._walk_manager_line(state, "chat-lead", "chat-lead").found
        assert not _outside(state, worker, lead.key)

    def test_a_hop_mid_handoff_is_still_matched_on(self, tmp_path):
        """Deliberately NOT widened: an unsettled handoff is not an identity problem.

        Moving the whole hop-state block above the match would refuse a caller whose own
        key merely has a write in flight. Its identity is not in doubt -- only its
        manager is -- so the match stands and that question is asked further up.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        queued: asyncio.Future[bool] = (
            asyncio.get_event_loop_policy().new_event_loop().create_future()
        )
        sc._tree_pending_slots["chat-lead"] = queued
        try:
            assert sc._hop_handoff_unsettled("chat-lead")
            assert sc._walk_manager_line(state, "chat-lead", "chat-lead").found
        finally:
            sc._tree_pending_slots.pop("chat-lead", None)


def _ancestors_including_self(managers, start):
    """*start* and every key above it per *managers*, in walk order, cycles cut.

    The reporting line stated independently of the walk, so a sweep can say what the
    walk OUGHT to answer rather than only that it answers consistently.
    """
    chain = []
    key = start
    while key and key not in chain:
        chain.append(key)
        key = managers.get(key, "")
    return chain


class TestMovingTheReuseCheckChangedNothingForKeysNobodyRetook:
    """The safety condition on putting ``_key_is_ambiguous`` ahead of the identity match.

    Moving a refusing check EARLIER is safe only if the keys it does not refuse come out
    the way they did behind it -- same answer, through the same path. One example passing
    is not that claim: the branch order the move changed is reached differently for every
    pair of (where the walk starts, whom it is looking for), and the pairs that would
    break are the ones nobody thinks to write down. So the claim is made over a whole
    tree's input space at once, and the thing it is compared against is the same walk
    with the check REMOVED, which is the walk as it was before the move.

    The tree deliberately mixes the hop kinds the move runs through: live hops, branches
    that do not reach each other, a top with nobody above it, a departed hop with one
    committed occupant that the walk reads THROUGH, an unmanaged session, the empty key
    and a key this process has never seen.
    """

    MANAGERS = {
        "chat-owner": "",
        "chat-lead": "chat-owner",
        "chat-cond-a": "chat-lead",
        "chat-cond-b": "chat-lead",
        "chat-worker-a": "chat-cond-a",
        "chat-worker-b": "chat-cond-b",
        "chat-gone": "chat-lead",
        "chat-under-gone": "chat-gone",
        "chat-loner": "",
    }
    #: Everything the sweep starts from or asks about: every key in the tree, plus the
    #: two inputs that are not keys of it at all.
    PROBES = tuple(MANAGERS) + ("", "chat-never-seen")

    def _tree(self, tmp_path):
        """The tree above, with ``chat-gone`` departed and nobody's key retaken."""
        state = _make_state(tmp_path)
        for key, manager in self.MANAGERS.items():
            state.get_or_create_slot(key)._created_by = manager
        _depart(state, "chat-gone")
        assert _recorded("chat-gone") == "chat-lead", "departed, committed, one occupant"
        assert not any(sc._key_is_ambiguous(state, key) for key in self.PROBES)
        return state

    def _sweep(self, state):
        """Every ``(start, wanted)`` pair the tree admits, answered by the one walk."""
        return {
            (start, wanted): sc._walk_manager_line(state, start, wanted)
            for start in self.PROBES
            for wanted in self.PROBES
        }

    def test_the_whole_tree_answers_its_reporting_line_and_stops_nowhere(self, tmp_path):
        """What the answers ARE, stated without the walk: ancestor-or-self, read to the end.

        Pinned first because the equivalence below would also hold if both halves were
        broken the same way.
        """
        state = self._tree(tmp_path)

        for (start, wanted), read in self._sweep(state).items():
            expected = bool(wanted) and wanted in _ancestors_including_self(self.MANAGERS, start)
            assert read == sc._LineRead(expected, False), (start, wanted)

    def test_every_pair_answers_as_it_did_with_the_check_behind_the_match(self, tmp_path):
        """The equivalence itself: removing the moved check changes no answer here.

        ``_key_is_ambiguous`` forced False is the walk without the check at all, which
        for an unretaken key is also the walk with the check in its old position -- the
        old position was only ever reached by keys this one refuses.
        """
        state = self._tree(tmp_path)
        with_check = self._sweep(state)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(sc, "_key_is_ambiguous", lambda state, key: False)
            without_check = self._sweep(state)

        assert with_check == without_check

    def test_the_sweep_does_part_once_a_key_is_retaken(self, tmp_path):
        """And the equivalence is not vacuous: the sweep separates the two walks.

        The same comparison over the same tree, with one middle key retaken. Every pair
        whose line reaches that key before it finds what it was looking for now refuses
        and names reuse; every other pair still answers exactly as the walk without the
        check does. So the test above holds because nobody retook a key, not because the
        sweep cannot tell the difference.
        """
        state = self._tree(tmp_path)
        retaken = "chat-cond-a"
        _depart(state, retaken)
        # The replacement reports to the same manager, so an answer that differs cannot
        # be put down to the tree having changed shape.
        state.get_or_create_slot(retaken)._created_by = self.MANAGERS[retaken]
        assert sc._key_is_ambiguous(state, retaken)

        with_check = self._sweep(state)
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(sc, "_key_is_ambiguous", lambda state, key: False)
            without_check = self._sweep(state)

        parted = set()
        for (start, wanted), read in with_check.items():
            chain = _ancestors_including_self(self.MANAGERS, start)
            # An empty *wanted* is answered before the walk takes a step -- no session
            # holds that key, so it is on no line and no hop is ever looked at.
            if (
                wanted
                and retaken in chain
                and (wanted not in chain or chain.index(retaken) <= chain.index(wanted))
            ):
                assert read == sc._LineRead(False, True, "reused"), (start, wanted)
                parted.add((start, wanted))
            else:
                assert read == without_check[(start, wanted)], (start, wanted)
        assert parted, "the retake has to change something or this proves nothing"


def _running_as(slot, tab_id):
    """Give *slot* the tab identity the base PR compares managers on.

    ``_tab_id`` is the identity, not an ACP session id: it is persisted, so it is the
    same after a restart, and a reopened tab keeps its own while a different session
    under the same name gets another. A test that means "this is the same session as
    before" has to say it that way.
    """
    slot._tab_id = tab_id
    return slot


class TestTheSameSessionResumingItsOwnKeyIsNotAStranger:
    """A key is a NAME. The thing an edge has to be about is the session holding it.

    The first version of this refusal marked a key ambiguous once it had been taken
    twice, which is a proxy for the real question and is wrong in the direction a person
    meets most often: closing a tab and reopening it is two takes of one name by ONE
    session, and refusing that refuses every handoff to a session anybody has reopened.

    So the comparison is on the base PR's own identity, the persisted ``_tab_id``: the
    edge records which occupant of the manager's key it was published to, and the walk
    compares that against who answers to the key now. One identity, one rule, and the
    answer outlives the process -- a mechanism of this PR's own would resolve every edge
    confidently again after the next restart.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def _handed_above_the_conductor(self, tmp_path):
        """worker -> conductor -> lead, with the CONDUCTOR handed to the lead.

        The handed edge sits one level ABOVE the caller's target on purpose. A handoff
        is what records an identity, and the base PR asks about the record of the slot a
        verb was called on -- nobody asks it about a slot that is merely a STEP in
        somebody else's line, which is the hop this walk follows. So the hop whose key
        gets retaken here is the conductor's own manager, reached only by walking.
        """
        state = _make_state(tmp_path)
        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        conductor = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")
        conductor._created_by = "chat-owner"
        _publish_manager(state, conductor, lead.key)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        assert not _outside(state, worker, lead.key), "the premise: the lead reaches it"
        return state, lead, worker

    def test_a_closed_session_that_resumes_under_its_own_tab_is_still_itself(self, tmp_path):
        """THE REGRESSION. A reopened tab is the case a person hits most often.

        Nothing about this line changed: the same session is back under the same key,
        and the edge naming that key was published to it. Refusing here would deny the
        lead its own worker because the lead's own tab was closed and opened again.
        """
        state, lead, worker = self._handed_above_the_conductor(tmp_path)
        _depart(state, "chat-lead")
        assert _recorded("chat-lead") == "chat-owner"

        # The resume: the key is taken again, by the session that persisted that tab.
        resumed = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        resumed._created_by = "chat-owner"

        assert not sc._key_is_ambiguous(state, "chat-lead", "tab-lead")
        assert not _outside(state, worker, lead.key), "so the lead still reaches the worker"
        sc.authorize_target(
            state,
            caller_session_key=slot_history_key(lead),
            target=worker.key,
            operation="send",
        )

    def test_a_different_session_under_the_same_key_is_refused(self, tmp_path):
        """The case the refusal is actually for, with the reason code unchanged."""
        state, lead, worker = self._handed_above_the_conductor(tmp_path)
        _depart(state, "chat-lead")

        # Somebody else opens a tab under the closed lead's name.
        stranger = _running_as(state.get_or_create_slot("chat-lead"), "tab-other")
        stranger._created_by = "chat-someone-else"

        assert sc._key_is_ambiguous(state, "chat-lead", "tab-lead")
        assert _outside(state, worker, lead.key)
        assert _refusal_cause(state, worker, lead.key) == "reused"

    def test_a_stranger_stays_one_however_often_it_reopens(self, tmp_path):
        """What the edge means does not drift with the last take.

        Asked as "whom did the current occupant displace", a stranger is refused on the
        take that displaced the original and then displaces ITSELF on the next one --
        ids match, hop resolves, the original's whole line comes with it. The edge
        records which session it was published to, so no take can change the answer.
        """
        state, lead, worker = self._handed_above_the_conductor(tmp_path)
        _depart(state, "chat-lead")
        _running_as(state.get_or_create_slot("chat-lead"), "tab-other")
        assert _outside(state, worker, lead.key), "refused on the first take"

        _depart(state, "chat-lead")
        again = _running_as(state.get_or_create_slot("chat-lead"), "tab-other")
        again._created_by = "chat-elsewhere"

        assert sc._key_is_ambiguous(state, "chat-lead", "tab-lead")
        assert _outside(state, worker, lead.key), "and still refused after its own resume"

    def test_an_occupant_with_no_identity_is_refused_rather_than_assumed(self, tmp_path):
        """Two sessions with no identity are not thereby the same session.

        A slot carrying no tab answers ``""``, and so does a record of an occupant that
        departed without one. Comparing those as equal is how an unknown resolves to a
        definitive answer, so neither side's blank counts as a match.
        """
        state, lead, worker = self._handed_above_the_conductor(tmp_path)
        _depart(state, "chat-lead")

        nameless = _running_as(state.get_or_create_slot("chat-lead"), "")
        nameless._created_by = "chat-owner"

        assert sc._key_is_ambiguous(state, "chat-lead", "tab-lead")
        assert _outside(state, worker, lead.key)

    def test_the_cap_cannot_turn_a_reused_key_into_a_grant(self, tmp_path):
        """What the cap costs is reach. It may never buy anybody reach.

        The departure record is capped and the cap is reached by ordinary slot churn.
        The identity a hop is judged on is not in that table: the edge carries it and
        the live occupant answers it, so a dropped record cannot read as "this key is
        still the session the edge meant".
        """
        state, lead, worker = self._handed_above_the_conductor(tmp_path)
        _depart(state, "chat-lead")
        stranger = _running_as(state.get_or_create_slot("chat-lead"), "tab-other")
        stranger._created_by = "chat-someone-else"
        assert _outside(state, worker, lead.key), "refused before the churn"

        # Churn means sessions that OPENED AND CLOSED: the cap only drops a departed
        # hop's own line, so a key with no record of its own is not what it reclaims.
        for index in range(sc._KEY_OCCUPANCY_MAX + 1):
            key = f"chat-churn-{index}"
            sc.note_key_taken(key, 10_000 + index)
            sc._record_departure(key, 10_000 + index, "", pending=False, tab=f"t-{index}")

        assert _unidentifiable("chat-lead"), "the reuse evidence is not reclaimable"
        assert sc._key_is_ambiguous(state, "chat-lead", "tab-lead"), "the edge did not"
        assert _outside(state, worker, lead.key)


class TestAnInadmissibleGrantIsNotFollowedThroughAHopEither:
    """The base PR refuses a grant it read and may not publish. So must the walk.

    ``_manager_grant_inadmissible`` is set when the log WAS read and the manager it
    names cannot hold the session -- closed, or running under another principal's memory
    store. The verbs refuse on that slot. But a slot is also a HOP in somebody else's
    line, and a refusal that fires on the subject says nothing about the step: the walk
    reading ``manager_of`` through such a hop admits exactly the manager the admission
    rejected, over everything below it.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def _line_with_a_refused_grant(self, tmp_path):
        state = _make_state(tmp_path)
        rebound = _running_as(state.get_or_create_slot("chat-rebound"), "tab-rebound")
        rebound._created_by = "chat-owner"
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = "chat-lead"
        _publish_manager(state, middle, rebound.key)  # the grant the log recorded
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = middle.key
        assert not _outside(state, worker, rebound.key), "followed while it is admissible"
        middle._manager_grant_inadmissible = True
        return state, rebound, middle, worker

    def test_the_walk_stops_at_a_hop_whose_grant_was_refused(self, tmp_path):
        state, rebound, _middle, worker = self._line_with_a_refused_grant(tmp_path)

        assert _outside(state, worker, rebound.key)
        assert _refusal_cause(state, worker, rebound.key) == "inadmissible"
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(rebound),
                target=worker.key,
                operation="send",
            )
        assert exc.value.code == "manager_line_inadmissible", exc.value.code

    def test_the_close_does_not_publish_the_refused_grant_as_a_settled_edge(self, tmp_path):
        """Otherwise the close hands over what the live walk had just declined.

        An entry outlives the slot that carried the refusal, so recording this manager
        would make the hop resolvable the moment it stops being live -- the boundary
        would hold right up to the close and then open.
        """
        state, rebound, _middle, worker = self._line_with_a_refused_grant(tmp_path)

        assert sc.note_manager_departed(state, "chat-middle", state.get_slot("chat-middle")) == ""
        state._slots.pop("chat-middle", None)

        assert _recorded("chat-middle") is None, "nothing was recorded for it"
        assert _outside(state, worker, rebound.key), "and the line still refuses"


class TestAForgottenFactIsNotAFactInTheCallersFavour:
    """The one rule behind three different findings: eviction, close and restart each
    destroy a record, and none of them may turn into a permissive answer.

    Every one of those was reported separately -- a cap that drops an occupancy, a close
    that cannot settle its own entry, a restart with no readable line -- and every one
    had the same shape: something the gateway USED to know is gone, the code reads the
    absence as the harmless case, and the harmless case is the one that grants reach.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_a_dropped_record_is_a_gap_and_a_gap_refuses_a_writer(self, tmp_path):
        """The cap is reached by ordinary churn, so this is the common path, not an edge.

        What the cap destroys is the departed hop's own line. The IDENTITY of a hop is
        not in that table -- the edge carries it -- so a dropped record cannot read as
        "this key is still the session the edge meant". What it must not read as either
        is "nothing sits above this hop": a writer that concludes that from a discarded
        record writes an edge which becomes a cycle when the key reopens.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        # The cap takes the hop while nothing open can climb to it, and the session it
        # created is restored afterwards -- the only order that leaves a dropped record
        # on a line a live session walks.
        _hop_whose_record_the_cap_took(state, "chat-upper", creator=lead.key)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-upper"
        assert _outside(state, worker, lead.key), "a reader is refused"
        assert _present_for_a_writer(
            state, "chat-upper", "chat-anything"
        ), "and a writer reads the gap as a line it could not finish"

    def test_a_close_that_records_nothing_leaves_no_entry_promising_to_settle(self, tmp_path):
        """A pending entry nothing will ever overwrite tells callers to retry forever.

        The pop writes PENDING so the close window refuses on an entry of its own. Two
        commits record nothing at all -- an unread manager, and a grant that is not
        admissible -- and the entry they leave is permanent: the hop keeps answering
        ``manager_line_pending``, whose message promises the write is about to land. The
        hop must still refuse; it must stop claiming it is about to clear.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = mid.key
        mid._manager_unresolved = True  # closed before its manager was ever read

        sc.note_key_vacated("chat-mid", mid)
        assert _pending("chat-mid"), "the pop's own entry"
        state._slots.pop("chat-mid", None)
        assert sc.note_manager_departed(state, "chat-mid", mid) == ""

        assert _entry("chat-mid") is None, "nothing left claiming a write is coming"
        assert _outside(state, worker, lead.key), "the hop still refuses"
        assert _refusal_cause(state, worker, lead.key) == "forgotten", "and does not say retry"

    def test_the_refusal_does_not_blame_a_retention_limit_that_was_not_reached(self, tmp_path):
        """One walk answer, several causes, and only one of them is the cap.

        A record never written and a record a close removed read the same way as a
        record the cap dropped. Telling the first two that "the line passed through more
        than 512 other sessions" sends whoever is debugging the denial after churn that
        never happened -- so the retention limit is named when it is a fact and the
        refusal says the record is unusable when that is all it knows.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-mid"
        mid = state.get_or_create_slot("chat-mid")
        mid._created_by = lead.key
        mid._manager_unresolved = True
        sc.note_key_vacated("chat-mid", mid)
        state._slots.pop("chat-mid", None)
        sc.note_manager_departed(state, "chat-mid", mid)
        assert sc._key_occupancy_evictions[0] == 0, "nothing was ever dropped by the cap"

        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state,
                caller_session_key=slot_history_key(lead),
                target=worker.key,
                operation="send",
            )

        assert exc.value.code == "manager_line_forgotten", exc.value.code
        assert "other sessions since it closed" not in str(exc.value)
        assert "no usable record" in str(exc.value)


class TestARestoredEdgeIsRevalidatedAgainstThePrincipalThatRecordedIt:
    """The boot pass published grants after checking liveness and reuse and nothing else.

    A handoff's own admission requires the two sessions to run under ONE principal, and
    it applies that immediately before it appends. The entry outlives the process: in
    between, the recipient can be rebound to another member's store. So the log stays
    the authority on what was recorded, and this decides whether the record may still
    take effect -- the same split the base PR made for a live slot's manager, missing on
    the path that restores a CLOSED one's.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    @staticmethod
    def _restart(state, *, says, persisted, units=1):
        """Run the boot pass with the log and the metadata line both stubbed."""
        from kiro_crew.crew_log import emit

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager",
                    lambda key, tab: says.get(key, ("", False)),
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.store.session_units_for_slot",
                _log_reader(
                    "kiro_crew.crew_log.store.session_units_for_slot",
                    lambda slot, strict=False: tuple(f"sid-{slot}-{i}" for i in range(units)),
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager_tab",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager_tab",
                    # The OCCUPANCY a real manager entry carries. Without it every
                    # restored handoff is unverifiable and refuses, which is correct and
                    # is not what these tests are about.
                    lambda key, tab: _recorded_occupancy(state, {}, says.get(key, ("", False))[0]),
                ),
            )
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: persisted.get(
                    history_key, ({"tab_id": f"tab-{history_key}"}, True)
                ),
            )
            return asyncio.run(sc._rebuild_departures_from_log(state))

    def test_a_recipient_rebound_to_another_store_is_not_republished(self, tmp_path):
        state = _make_state(tmp_path)
        recipient = state.get_or_create_slot("chat-recipient")
        recipient._created_by = "chat-owner"
        recipient.memory_store = "store-b"  # rebound while the gateway was down
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        brought_back = self._restart(
            state,
            says={"chat-conductor": ("chat-recipient", True)},
            persisted={
                "dashboard:chat-conductor": (
                    {"memory_store": "store-a", "tab_id": "tab-conductor"},
                    True,
                )
            },
        )

        assert brought_back == 0
        assert _recorded("chat-conductor") is None
        assert _outside(state, worker, recipient.key)

    def test_the_same_edge_is_restored_when_the_principal_still_agrees(self, tmp_path):
        """The other half, or the check would just be a blanket refusal."""
        state = _make_state(tmp_path)
        recipient = state.get_or_create_slot("chat-recipient")
        recipient._created_by = "chat-owner"
        recipient.memory_store = "store-a"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        brought_back = self._restart(
            state,
            says={"chat-conductor": ("chat-recipient", True)},
            persisted={
                "dashboard:chat-conductor": (
                    {"memory_store": "store-a", "tab_id": "tab-conductor"},
                    True,
                )
            },
        )

        assert brought_back == 1
        assert not _outside(state, worker, recipient.key)

    def test_a_decided_empty_answer_means_the_creator_and_not_nobody(self, tmp_path):
        """``("", True)`` is a STATEMENT the log makes: no handoff, so the creator holds it.

        Recorded as "reports to nobody" the walk stops at that hop, and the most
        ordinary shape there is -- lead opens conductor, conductor opens worker, nobody
        hands anything over, conductor closes -- loses the lead its reach at every
        restart, which is what this pass exists to prevent.
        """
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        brought_back = self._restart(
            state,
            says={"chat-conductor": ("", True)},
            persisted={
                "dashboard:chat-conductor": (
                    {"created_by": "chat-lead", "tab_id": "tab-conductor"},
                    True,
                )
            },
        )

        assert brought_back == 1
        assert _recorded("chat-conductor") == "chat-lead", "the creator, from its own line"
        assert not _outside(state, worker, lead.key)

    def test_a_recipient_that_comes_back_on_another_store_mid_read_is_refused(self, tmp_path):
        """The verdict is only worth what it is worth AT THE WRITE.

        The admission reads the closed session's principal off the disk, and that read
        suspends. A recipient reopened inside that window -- under another store, which
        is the case the admission exists to refuse -- was not in the live table when the
        question was asked, so the answer describes a session that no longer exists. So
        every volatile part of it is asked again with no suspension before the write.
        """
        from kiro_crew.crew_log import emit

        state = _make_state(tmp_path)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        reopened: list = []

        def _recipient_returns_during_the_last_read(history_key):
            """Reopen the recipient from inside the read that judges IT.

            Deliberately this read and not an earlier one: while the recipient is
            closed, the admission reads its principal off the disk and that answer is
            correct when it is given. The recipient comes back DURING that read, so only
            a question asked after the last await can see it -- which is the whole
            finding.
            """
            if history_key == "dashboard:chat-recipient" and not reopened:
                back = state.get_or_create_slot("chat-recipient")
                back._created_by = "chat-owner"
                back.memory_store = "store-b"  # another principal entirely
                reopened.append(back)
            return ({"memory_store": "store-a", "tab_id": "tab-conductor"}, True)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager",
                    lambda key, tab: (
                        ("chat-recipient", True) if key == "chat-conductor" else ("", False)
                    ),
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.store.session_units_for_slot",
                _log_reader(
                    "kiro_crew.crew_log.store.session_units_for_slot",
                    lambda slot, strict=False: ("sid-conductor",),
                ),
            )
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                _recipient_returns_during_the_last_read,
            )
            brought_back = asyncio.run(sc._rebuild_departures_from_log(state))

        assert reopened, "the premise: it came back inside the window"
        assert brought_back == 0
        assert _recorded("chat-conductor") is None
        assert _outside(state, worker, "chat-recipient")

    def test_an_unreadable_metadata_line_records_nothing_at_all(self, tmp_path):
        """Every one of those four questions is answered by that read, so none is guessed."""
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        brought_back = self._restart(
            state,
            says={"chat-conductor": ("chat-lead", True)},
            persisted={"dashboard:chat-conductor": ({}, False)},
        )

        assert brought_back == 0
        assert _recorded("chat-conductor") is None
        assert _outside(state, worker, lead.key)


class TestNoLineAnswerDeletesAnAcceptedMessage:
    """Refuse an authorization. Never destroy words somebody was told were accepted.

    The drain re-asks the admission's own question, and two of its answers are not
    decisions about the sender at all: a line mid-close, and a line whose record the cap
    dropped. Both arrived as "the sender lost its reach" and both DELETED the message.
    The first was found, fixed, and the second came back the next round under a
    different cause -- so the rule is stated once here over the whole set.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def _lead_queued_on_a_grandchild(self, tmp_path):
        state = _make_state(tmp_path)
        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        conductor = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        meta = sc.send_origin_meta(state, lead.key, manager_fenced=True)
        assert not sc.sender_lost_its_reach(state, worker, meta), "admitted"
        return state, worker, meta

    def test_a_line_mid_close_holds_the_message(self, tmp_path):
        state, worker, meta = self._lead_queued_on_a_grandchild(tmp_path)
        conductor = state.get_slot("chat-conductor")
        state._slots.pop("chat-conductor", None)
        sc.note_key_vacated("chat-conductor", conductor)  # popped, not yet committed

        with pytest.raises(sc.SessionControlError) as exc:
            sc.sender_lost_its_reach(state, worker, meta)
        assert exc.value.code == "manager_line_pending", exc.value.code

    def test_the_sweep_actually_holds_it_instead_of_deleting_it(self, tmp_path):
        """THE RAISE IS NOT THE HOLD, and this is the half that was missing.

        Every test above asserts the refusal. The sweep is what has to act on it, and it
        compared the code against the ONE refusal that waited before this feature added
        a second: the entry went to the drop list, the person was told their accepted
        message had been dropped, and the 409 saying it would be retried was the only
        trace of the hold. So the sweep is driven here, and what is asserted is the
        entry still being in the queue and marked for the merge run to break at.
        """
        from kiro_crew.dashboard.chat_turn.steer_queue import _drop_stale_admissions

        state, worker, meta = self._lead_queued_on_a_grandchild(tmp_path)
        worker._queue.append({"id": "q1", "text": "run this", "meta": meta})
        conductor = state.get_slot("chat-conductor")
        state._slots.pop("chat-conductor", None)
        sc.note_key_vacated("chat-conductor", conductor)  # popped, not yet committed

        _drop_stale_admissions(state, worker)

        assert [q["id"] for q in worker._queue] == ["q1"], "the accepted text is still there"
        assert worker._queue[0].get("_held_for_manager") is True, "and held, not merged"

    def test_the_hold_clears_once_the_close_commits(self, tmp_path):
        """The other half of a hold: it has to END, or the queue is shut for good.

        The refusal promises a retry, so the next sweep after the close commits has to
        deliver -- here the conductor's departure lands, the line reads to the lead
        again, and the mark the previous pass left is cleared by the pass that decided it.
        """
        from kiro_crew.dashboard.chat_turn.steer_queue import _drop_stale_admissions

        state, worker, meta = self._lead_queued_on_a_grandchild(tmp_path)
        worker._queue.append({"id": "q1", "text": "run this", "meta": meta})
        conductor = state.get_slot("chat-conductor")
        state._slots.pop("chat-conductor", None)
        sc.note_key_vacated("chat-conductor", conductor)
        _drop_stale_admissions(state, worker)
        assert worker._queue[0].get("_held_for_manager") is True, "the premise: held"

        sc.note_manager_departed(state, "chat-conductor", conductor)  # the close commits
        _drop_stale_admissions(state, worker)

        assert [q["id"] for q in worker._queue] == ["q1"]
        assert not worker._queue[0].get("_held_for_manager"), "deliverable again"

    def test_a_record_the_cap_dropped_is_answered_one_level_not_held_forever(self, tmp_path):
        """``forgotten`` never clears, so WAITING on it is its own failure.

        A hold is not free: the drain stops starting turns on that session while the
        entry sits at the head, so holding on an answer nothing re-arms wedges the queue
        -- the person's own messages included -- until a restart. So the question is
        answered by the reach a gateway has without this feature at all: the one-level
        relation, which is a fact about this process rather than about a record that
        aged out.
        """
        state, worker, meta = self._lead_queued_on_a_grandchild(tmp_path)
        _depart(state, "chat-conductor")
        sc._departed_managers.clear()  # the cap dropped it
        sc._key_occupancy_evictions[0] = 1

        # No raise, so nothing is held; and the lead is not the DIRECT manager, so this
        # is the ordinary answer it would have got before the line existed.
        assert sc.sender_lost_its_reach(state, worker, meta)

    def test_a_replacement_on_the_senders_key_does_not_inherit_its_delivery(self, tmp_path):
        """The LAST hop was the one place the line was read off a string.

        Every hop the walk climbs through is judged against the edge that named it. The
        asker itself is named by the key it matched, and a key is a name a later session
        can answer to -- so a new occupant of a closed lead's key, having been handed
        that lead's old descendants again, presents a chain every other check agrees
        with. The entry waiting on the worker was stamped by the session that was
        granted the reach, and the replacement is not that session.

        The one-level fence already compares this, which is why the walk has to: a
        re-check that admits what the direct-manager answer refuses is not a re-check.
        """
        state, worker, meta = self._lead_queued_on_a_grandchild(tmp_path)
        assert sc.send_origin_sender_tab(meta) == "tab-lead", "the premise: the stamp names it"
        conductor = state.get_slot("chat-conductor")

        # ACROSS A RESTART, which is what leaves no departure record to catch this. The
        # entry comes back from disk with its provenance; the in-memory tables do not
        # come back at all, so the key-relative check has nothing to say about
        # "chat-lead" and the retake mark is not there to be read either.
        state._slots.pop("chat-lead", None)
        _forget_everything()
        replacement = _running_as(state.get_or_create_slot("chat-lead"), "tab-not-the-lead")
        replacement._created_by = "chat-owner"
        # And the replacement is handed the conductor, so the edge above the worker
        # names the session answering to that key NOW: the chain is consistent with
        # itself at every hop, and the only thing that disagrees is the stamp.
        _publish_manager(state, conductor, replacement.key)
        assert not sc._key_is_ambiguous(state, "chat-lead"), "the key's own record is gone"
        assert not sc._edge_is_ambiguous(
            state, "chat-lead", sc._edge_identity(conductor)
        ), "and the edge names the occupant"

        assert sc.sender_lost_its_reach(state, worker, meta)
        # And the replacement's OWN sends are admitted: the refusal is about the claim
        # the entry carries, not about the key being suspect.
        its_own = sc.send_origin_meta(state, replacement.key, manager_fenced=True)
        assert not sc.sender_lost_its_reach(state, worker, its_own)

    def test_a_stamp_from_before_the_identity_field_reads_key_only(self, tmp_path):
        """An empty claim is key-only, exactly as the one-level predicate reads one.

        A provenance written before that field existed carries no identity, and refusing
        it would delete messages whose senders were authorized to send them and still
        are -- for a field the sender never had the chance to write. Nothing is granted
        that a forged stamp could not already claim by naming the recorded identity.
        """
        state, worker, meta = self._lead_queued_on_a_grandchild(tmp_path)
        meta[sc.ADMISSION_PROVENANCE_META_KEY].pop("tab")

        assert not sc.sender_lost_its_reach(state, worker, meta)

    def test_a_direct_manager_keeps_its_delivery_through_a_dropped_record(self, tmp_path):
        """The other half: the one-level answer still ADMITS whom it always admitted."""
        state = _make_state(tmp_path)
        conductor = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")
        conductor._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        meta = sc.send_origin_meta(state, conductor.key, manager_fenced=True)
        sc._key_occupancy_evictions[0] = 1

        assert not sc.sender_lost_its_reach(state, worker, meta)


class TestARetakeOfAnEvictedKeyIsNotARetakeOfAFreeOne:
    """Taking a name finds no occupancy in two situations that mean opposite things.

    The name is new, or the record that would have said it has been held before is
    exactly what the cap reclaimed. Read as the first, every edge still naming it
    resolves through the new occupant, which hands a deleted session's descendants to
    whoever opened a tab under its name.

    Two rules answer it, and the first is the one that matters: a record is not
    reclaimable while an open session's LINE can still climb to its key, so the evidence
    survives exactly as long as anything can ask for it. The eviction shadow is the
    backstop for a key nobody references any more.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_the_cap_cannot_reclaim_an_ancestor_a_kept_record_names(self, tmp_path):
        """The keep rule follows the whole chain, not the keys a live slot names itself.

        The finding's sequence. A creates B and B creates W; A is deleted and B closes,
        and B's record is kept because the open W names it -- while A, named only by B's
        record, is not. Ordinary churn then takes A out of the occupancy table and out
        of the eviction shadow after it, and a key absent from BOTH reads as a name
        nobody ever held: the next session to open under A's name is marked as having
        displaced nobody, the creation edge above B resolves through it, and it inherits
        the worker A's session was running -- read, stop and close included.

        So the chain the cap protects is the chain the walk follows.
        """
        state = _make_state(tmp_path)
        top = _running_as(state.get_or_create_slot("chat-top"), "tab-top")
        top._created_by = "chat-owner"
        middle = _running_as(state.get_or_create_slot("chat-middle"), "tab-middle")
        middle._created_by = top.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = middle.key
        assert not _outside(state, worker, top.key), "the premise: the real top reaches it"

        _depart(state, "chat-top")  # deleted
        _depart(state, "chat-middle")  # and its conductor closes
        _churn_closed(state, sc._KEY_OCCUPANCY_MAX + 4)

        assert sc._key_occupancy.get("chat-middle") is not None, "the live worker names it"
        assert sc._key_occupancy.get("chat-top") is not None, "and that record names the top"
        # So the retake is still known to be one, and the replacement is refused.
        impostor = _running_as(state.get_or_create_slot("chat-top"), "tab-impostor")
        impostor._created_by = "chat-cron"

        assert impostor._displaced_tab == "tab-top", "taken from the session it names"
        assert _outside(state, worker, "chat-top")

    def test_the_cap_cannot_reclaim_a_key_a_live_edge_still_names(self, tmp_path):
        state = _make_state(tmp_path)
        top = _running_as(state.get_or_create_slot("chat-top"), "tab-top")
        top._created_by = "chat-owner"
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = top.key  # the edge that still names chat-top
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = middle.key
        assert not _outside(state, worker, top.key), "the premise: the real top reaches it"

        _depart(state, "chat-top")
        _churn_closed(state, sc._KEY_OCCUPANCY_MAX + 4)

        assert sc._key_occupancy.get("chat-top") is not None, "still referenced, so kept"
        cron = _running_as(state.get_or_create_slot("chat-top"), "tab-cron")
        cron._created_by = "chat-cron"

        assert cron._displaced_tab == "tab-top", "taken from the session it names"
        assert _outside(state, worker, "chat-top")

    def test_a_key_nobody_names_any_more_is_still_caught_by_the_shadow(self, tmp_path):
        """The backstop, for a record the cap was allowed to reclaim.

        Nothing open names this key, so its record goes -- and the take that follows
        would read the name as free. The eviction shadow is what says otherwise.
        """
        state = _make_state(tmp_path)
        gone = _running_as(state.get_or_create_slot("chat-gone"), "tab-gone")
        gone._created_by = "chat-owner"
        _depart(state, "chat-gone")
        _churn_closed(state, sc._KEY_OCCUPANCY_MAX + 4)

        assert "chat-gone" not in sc._keys_a_live_edge_names(), "the premise: nobody names it"
        assert sc._key_occupancy.get("chat-gone") is None, "nobody named it, so it went"
        assert "chat-gone" in sc._recently_evicted_keys, "and the shadow remembers"

        retaken = _running_as(state.get_or_create_slot("chat-gone"), "tab-new")

        assert retaken._displaced_tab == "", "took a name the cap had dropped"

    def test_a_brand_new_name_still_costs_nothing(self, tmp_path):
        """The shadow is asked only about a key with no occupancy, so this is unaffected."""
        state = _make_state(tmp_path)
        _churn_closed(state, sc._KEY_OCCUPANCY_MAX + 4)
        lead = _running_as(state.get_or_create_slot("chat-fresh-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        mid = state.get_or_create_slot("chat-fresh-mid")
        mid._created_by = lead.key
        worker = state.get_or_create_slot("chat-fresh-worker")
        worker._created_by = mid.key

        assert mid._displaced_tab is None
        assert not _outside(state, worker, lead.key)


class TestTheCapStillBoundsBothTablesWhenKeysAreReused:
    """A bound has to bound. Reuse evidence used to sit here and outlive every cap.

    The evidence that a key changed hands cannot be dropped -- dropped, it reads as
    "this key is pristine" and the next occupant of a deleted session's name inherits
    its descendants. Kept in the capped table, one entry per reused key, it made the
    cap stop bounding anything: close, resume and close enough distinct sessions and
    both tables grow past it for the life of the process.

    So the evidence lives on the live occupant, where what bounds it is the number of
    sessions that can be open at once, and the capped table is left holding only the
    lines a cap may reclaim.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_many_reuse_cycles_leave_both_tables_inside_the_cap(self, tmp_path):
        state = _make_state(tmp_path)
        for index in range(sc._KEY_OCCUPANCY_MAX + 8):
            key = f"chat-cycle-{index}"
            # Open, close, reopen the same name, close again: one full reuse cycle,
            # which is what used to leave a permanent entry behind.
            first = _running_as(state.get_or_create_slot(key), f"tab-{index}-a")
            state._slots.pop(key, None)
            sc.note_manager_departed(state, key, first)
            second = _running_as(state.get_or_create_slot(key), f"tab-{index}-b")
            state._slots.pop(key, None)
            sc.note_manager_departed(state, key, second)

        assert len(sc._key_occupancy) <= sc._KEY_OCCUPANCY_MAX
        assert len(sc._departed_managers) <= sc._KEY_OCCUPANCY_MAX

    def test_a_live_replacement_is_still_refused_after_that_churn(self, tmp_path):
        """And the protection the evidence buys survives the same churn."""
        state = _make_state(tmp_path)
        top = _running_as(state.get_or_create_slot("chat-top"), "tab-top")
        top._created_by = "chat-owner"
        middle = state.get_or_create_slot("chat-middle")
        middle._created_by = top.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = middle.key
        _depart(state, "chat-top")
        impostor = _running_as(state.get_or_create_slot("chat-top"), "tab-impostor")
        impostor._created_by = "chat-someone-else"

        for index in range(sc._KEY_OCCUPANCY_MAX + 8):
            key = f"chat-cycle-{index}"
            slot = _running_as(state.get_or_create_slot(key), f"tab-{index}")
            state._slots.pop(key, None)
            sc.note_manager_departed(state, key, slot)

        assert impostor._displaced_tab == "tab-top", "the mark is on the occupant"
        assert _outside(state, worker, "chat-top")


class TestWhatTheLineOffStillRecordsAndWhatItDoesNot:
    """Work nobody reads should not run -- EXCEPT the one fact nobody can rebuild.

    Every reader of the record is behind the switch, so with it off a record per close
    and a disk read per closed hop at boot are work whose output has no consumer. Those
    are skipped, and skipping them costs nothing that lasts: a departure is a statement
    the crew log already carries, so the boot pass brings back what the switch missed.

    The OCCUPANCY of a key is the exception, and it is recorded whatever the switch
    says. A take is the only moment the generation is knowable and nothing can
    reconstruct it afterwards -- and the moment keys are taken in BULK is the restore at
    startup, which runs before any async reader has published the switch. Skipped
    there, every restored session closes with no occupancy to key a departure on, the
    write refuses because the key's occupancy is not its own, and the walk reads that
    hop as the end of a line: after every restart a lead loses reach over a closed
    restored conductor's workers, which is the case the feature exists for.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_the_occupancy_is_recorded_and_the_departure_is_not(self, tmp_path):
        state = _make_state(tmp_path)
        sc._chain_switch_seen[0] = None  # the shipped default: never read, so off

        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        mid = _running_as(state.get_or_create_slot("chat-mid"), "tab-mid")
        mid._created_by = lead.key
        _depart(state, "chat-mid")

        assert sc._key_occupancy["chat-mid"], "the take is recorded: nothing rebuilds it"
        assert sc._departed_managers == {}, "the close is not: the log carries it"

    def test_a_hop_that_closed_while_the_line_was_off_is_unread_and_not_ended(self, tmp_path):
        """Turning the switch on must not read the skipped bookkeeping as an answer.

        The departure writers are gated, so a session that closes while the line is off
        files nothing. Flip the switch and the walk meets that hop with no record -- and
        if that read as a line that ENDED, the handoff's loop guard would conclude
        nothing sits above it and write an edge the log turns into a cycle.

        What keeps the two apart is the one write that is NOT gated: the key's occupancy
        is on record from the take, so the walk finds an occupancy with no record under
        it, which is "no usable record of that line" rather than "nothing above".
        """
        state = _make_state(tmp_path)
        sc._chain_switch_seen[0] = None  # off: the shipped default

        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        conductor = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        _depart(state, "chat-conductor")
        assert _recorded("chat-conductor") is None, "the premise: the close filed nothing"

        sc.note_chain_switch(True)  # an operator turns the line on

        line = sc._walk_manager_line(state, "chat-conductor", lead.key, claimed="tab-lead")
        assert not line.found, "a reader is refused"
        assert line.unread, "and a writer does not read it as a line that ended"
        assert line.cause == "forgotten", line.cause
        assert _outside(state, worker, lead.key)

    def test_a_key_taken_before_the_switch_was_read_still_records_its_departure(self, tmp_path):
        """The restart case, which is what the ungated take is for.

        The restore builds its slots and only then does a reader publish the switch, so
        every one of those keys is taken while the answer is still unknown. The close
        that follows has to be filable, or the hop it leaves reads as the end of a line
        and the lead above it is refused -- for a conductor that did exactly what a
        conductor does.
        """
        state = _make_state(tmp_path)
        sc._chain_switch_seen[0] = None  # before any async reader has published it

        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        conductor = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key

        sc.note_chain_switch(True)  # the rebuild task reads it and says so
        _depart(state, "chat-conductor")

        assert _recorded("chat-conductor") == lead.key, "the departure was filable"
        assert not _outside(state, worker, lead.key), "so the lead keeps its reach"

    def test_the_boot_pass_reads_no_disk_with_the_line_off(self, tmp_path):
        """It is a read per closed hop at every boot, and nothing reads it back."""
        from kiro_crew.crew_log import emit

        state = _make_state(tmp_path)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        reads: list = []

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(sc, "manager_chain_enabled", lambda: False)
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: (reads.append(history_key), ({}, True))[1],
            )
            brought_back = asyncio.run(sc._rebuild_departures_from_log(state))

        assert brought_back == 0
        assert reads == [], "no line was read"


class TestAClosedHopTheBootPassHasNotReachedIsUnread:
    """A gap means "no line" only once the pass that fills gaps has run.

    Before it, the record is not ABSENT -- it is not read yet, and the two answers part
    for the writer. A reader denied reach loses nothing it will not get back. The loop
    guard concludes "nothing sits above this hop" and writes an edge that becomes a
    PERSISTED cycle the moment the pass restores that hop's real manager, and one cycle
    takes the line away from every session in it.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_the_writer_refuses_a_gap_while_the_restore_is_armed(self, tmp_path):
        state = _make_state(tmp_path)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-closed-hop"

        sc._departures_rebuilt[0] = False  # what a fresh process looks like

        line = sc._walk_manager_line(state, "chat-closed-hop", "chat-anything")

        assert line.unread, "not a line that ended: a line not read yet"
        assert line.cause == "pending", line.cause
        assert line.waits, "and the pass settles either way, so a retry answers"

    def test_the_same_gap_reads_to_the_end_once_the_pass_has_run(self, tmp_path):
        """Afterwards a gap really is a line that ended, which must keep working.

        A key this process never saw -- a tab closed before a restart -- is exactly this
        shape, and refusing it would freeze every handoff underneath it.
        """
        state = _make_state(tmp_path)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-closed-hop"
        sc._departures_rebuilt[0] = True  # the pass has run

        line = sc._walk_manager_line(state, "chat-closed-hop", "chat-anything")

        assert not line.unread
        assert not _present_for_a_writer(state, "chat-closed-hop", "chat-anything")


class TestTheCapNeverDropsALiveSessionsOccupancy:
    """A cap may cost reach across a CLOSED hop. It may not break a running session.

    Evicting in take order alone drops the occupancy of sessions that are still open,
    and a live session whose occupancy is gone cannot record its own departure: the
    writer checks its generation against the key's and finds another. So a conductor
    that stays open through ordinary churn stops being readable the moment it closes.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_a_long_lived_session_still_records_its_own_departure(self, tmp_path):
        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        conductor = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key

        # Ordinary churn: every one of these closes, so every one is evictable.
        for index in range(sc._KEY_OCCUPANCY_MAX + 20):
            key = f"chat-churn-{index}"
            sc.note_key_taken(key, 90_000 + index)
            sc._record_departure(key, 90_000 + index, "", pending=False)

        assert sc._key_occupancy.get("chat-conductor") == conductor._occupancy_generation
        _depart(state, "chat-conductor")
        assert _recorded("chat-conductor") == "chat-lead", "its close was recorded"
        assert not _outside(state, worker, lead.key)


class TestADepartedEdgeCarriesThePrincipalItWasAdmittedUnder:
    """The live edge is refused once the two sides stop agreeing on a store. So is this.

    A departed edge grants exactly the reach the live one grants, so it has to answer
    the same question -- and it can only answer it from what the close wrote down. The
    field existed for a round while no writer filled it, which made every comparison
    ``""`` against ``""``: a manager since rebound to another principal kept its reach
    over everything below a closed hop, and two sides that agreed on a NON-default store
    were refused instead.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def _handed_then_closed(self, tmp_path, *, worker_store, lead_store):
        state = _make_state(tmp_path)
        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        lead.memory_store = lead_store
        mid = _running_as(state.get_or_create_slot("chat-mid"), "tab-mid")
        mid._created_by = "chat-owner"
        mid.memory_store = worker_store
        _publish_manager(state, mid, lead.key)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = mid.key
        worker.memory_store = worker_store
        assert not _outside(state, worker, lead.key), "the premise: the line reads"
        _depart(state, "chat-mid")
        return state, lead, worker

    def test_a_manager_rebound_after_the_close_loses_the_closed_hop(self, tmp_path):
        state, lead, worker = self._handed_then_closed(
            tmp_path, worker_store="store-x", lead_store="store-x"
        )
        assert not _outside(state, worker, lead.key), "same principal, so it still reads"

        state.get_slot("chat-lead").memory_store = ""  # rebound to the default store

        assert _outside(state, worker, lead.key)
        assert _refusal_cause(state, worker, lead.key) == "inadmissible"

    def test_two_sides_on_the_same_non_default_store_still_read(self, tmp_path):
        """The reverse of the same bug: a blank record refused what it should allow."""
        state, lead, worker = self._handed_then_closed(
            tmp_path, worker_store="store-x", lead_store="store-x"
        )

        assert _entry("chat-mid").store == "store-x", "the close wrote it down"
        assert not _outside(state, worker, lead.key)

    def test_a_handed_edge_with_no_recorded_occupancy_refuses_once_it_closes(self, tmp_path):
        """A hop refused while live must not become readable BY closing.

        A restored slot can carry a manager key with no occupancy -- the tab read came
        back empty or raised. The live edge is refused for exactly that reason. Recorded
        as a creation edge, the closed hop skipped both the identity and the store check,
        so the close was what opened it.
        """
        state = _make_state(tmp_path)
        lead = _running_as(state.get_or_create_slot("chat-lead"), "tab-lead")
        lead._created_by = "chat-owner"
        mid = _running_as(state.get_or_create_slot("chat-mid"), "tab-mid")
        mid._created_by = "chat-owner"
        sc.publish_manager(mid, manager=lead.key, manager_tab="")  # occupancy unread
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = mid.key
        assert _outside(state, worker, lead.key), "refused while the hop is live"

        _depart(state, "chat-mid")

        assert _entry("chat-mid").manager_tab == "", "not None, which would read as created"
        assert _outside(state, worker, lead.key), "and still refused once it is closed"


class TestTheDrainReChecksWhatTheAdmissionChecked:
    """A re-check may refuse what the admission refused. Never what it allowed.

    A queued delivery's sender is re-compared at the drain, because a handoff can move
    the relation that admitted it while it waits. That re-check read the DIRECT manager,
    which was the whole of the fence before this feature -- and is narrower than the
    fence now. So a lead that reaches its grandchild worker through its conductor is
    admitted, told the message is queued, and then has the message deleted at the drain
    with the reporting line never having changed: an accepted send, silently discarded.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def _lead_queues_on_a_grandchild(self, tmp_path):
        state = _make_state(tmp_path)
        lead = _running_as(state.get_or_create_slot("chat-lead"), "sid-lead")
        lead._created_by = "chat-owner"
        conductor = state.get_or_create_slot("chat-conductor")
        conductor._created_by = lead.key
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = conductor.key
        # The entry as the admitted send stamps it.
        # ``manager_fenced``, because an UNFENCED sender's admission never read the
        # manager at all and the base PR declines to re-check one -- which is a
        # different question from this one, and it answers first.
        meta = sc.send_origin_meta(state, lead.key, manager_fenced=True)
        assert sc.send_origin_manager_fenced(meta), "the premise: a fenced sender"
        assert not _outside(state, worker, lead.key), "the admission: the lead reaches it"
        return state, worker, meta

    def test_a_skip_level_sender_is_not_dropped_at_the_drain(self, tmp_path):
        state, worker, meta = self._lead_queues_on_a_grandchild(tmp_path)

        assert sc._managed_by_other(
            state, worker, "chat-lead", sc._tab_identity(state, "chat-lead")
        ), "the narrow read refuses"
        assert not sc.sender_lost_its_reach(state, worker, meta), "the admission's read does not"

    def test_the_narrow_read_is_still_what_answers_with_the_line_off(self, tmp_path):
        """Off is the base PR's behaviour here as everywhere else in this diff.

        Said by the ENTRY, because that is the only way it can be said: an admission
        made with the line off stamps it off, and the drain re-asserts the predicate
        that admitted the entry rather than the one in force now.
        """
        state, worker, _meta = self._lead_queues_on_a_grandchild(tmp_path)
        admitted_off = sc.send_origin_meta(state, "chat-lead", manager_fenced=True, chain=False)

        assert not sc.send_origin_chain_applied(admitted_off)
        assert sc.sender_lost_its_reach(state, worker, admitted_off)

    def test_the_entry_carries_the_policy_that_admitted_it(self, tmp_path):
        """The drain asks the ENTRY, so it neither reads config nor waits.

        Both other ways were wrong at this point: reading the switch inline parses the
        config file on a cache miss and stalls the loop, and awaiting it off-loop put a
        suspension in front of the dequeue, where a concurrent direct send starts a turn
        the drain then overwrites. The entry knows what admitted it, and that is also
        the more correct question -- an operator flipping the switch while a message
        waits does not change what admitted it.
        """
        state, worker, meta = self._lead_queues_on_a_grandchild(tmp_path)

        def _never(*_a, **_k):
            raise AssertionError("the drain must not read config")

        assert sc.send_origin_chain_applied(meta), "the admission stamped the switch"
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(sc, "manager_chain_enabled", _never)
            assert not sc.sender_lost_its_reach(state, worker, meta)

    def test_an_entry_stamped_before_the_switch_existed_reads_as_one_level(self, tmp_path):
        """The default is the base PR's own answer, which never drops more than it did."""
        state, worker, _meta = self._lead_queues_on_a_grandchild(tmp_path)
        old_stamp = {sc.ADMISSION_PROVENANCE_META_KEY: {"sender": "chat-lead", "fenced": True}}

        assert not sc.send_origin_chain_applied(old_stamp)
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(sc, "manager_chain_enabled", lambda: True)
            # The one-level predicate: the lead is not the worker's DIRECT manager.
            assert sc.sender_lost_its_reach(state, worker, old_stamp)

    def test_the_sweep_itself_reads_no_config(self, tmp_path):
        """Driven through the SWEEP, which is the only production caller.

        Every other test here calls the re-check directly, so all of them would pass
        with the sweep reading the switch itself -- on the gateway's own loop, where a
        cache miss parses and validates the config file and stalls every task on it.
        There is no parameter to hand down any more and there was never a caller with
        one resolved: the entry says what admitted it, so the sweep asks nothing.
        """
        from kiro_crew.dashboard.chat_turn.steer_queue import _drop_stale_admissions

        state, worker, meta = self._lead_queues_on_a_grandchild(tmp_path)
        worker._queue.append({"id": "q1", "text": "run this", "meta": meta})

        def _never(*_args, **_kw):
            raise AssertionError("the drain must not read config")

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(sc, "manager_chain_enabled", _never)
            _drop_stale_admissions(state, worker)

        assert len(worker._queue) == 1, "and the message a sender was promised survives"

    def test_a_sender_that_really_lost_the_line_is_still_dropped(self, tmp_path):
        """The case the re-check exists for, which the widening must not lose."""
        state, worker, meta = self._lead_queues_on_a_grandchild(tmp_path)
        # The conductor is handed to somebody else, so the lead is off the line.
        state.get_slot("chat-conductor")._manager = "chat-elsewhere"

        assert sc.sender_lost_its_reach(state, worker, meta)


class TestTheBootPassNeverTakesAKeySomebodyIsHolding:
    """The rebuild reads from disk, and a person can resume a key while it reads.

    Its liveness test runs before two ``asyncio.to_thread`` reads, and it runs as a
    background task on a gateway that is already serving. A resume landing inside that
    window holds the key by the time the pass comes back -- and claiming an occupancy
    then does not merely add a stale entry: it moves the key's occupancy off the live
    occupant, which silently disables every departure that occupant later records (each
    checks its own generation against the key's) and replaces the mark its construction
    wrote with an entry naming whatever manager the LOG remembers.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_a_resume_inside_the_read_window_keeps_its_key(self, tmp_path):
        from kiro_crew.crew_log import emit

        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"
        resumed: list = []

        def _resume_during_the_read(slot_key, strict=False):
            """The person's resume, landing between the liveness test and the write."""
            key = slot_key
            if key == "chat-conductor" and not resumed:
                slot = _running_as(state.get_or_create_slot(key), "tab-c")
                slot._created_by = "chat-lead"
                resumed.append(slot)
            return ("unit-0",)

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager",
                    lambda key, tab: (
                        ("chat-stale-manager", True) if key == "chat-conductor" else ("", False)
                    ),
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.store.session_units_for_slot",
                _log_reader(
                    "kiro_crew.crew_log.store.session_units_for_slot", _resume_during_the_read
                ),
            )
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: ({"tab_id": "tab-conductor"}, True),
            )
            brought_back = asyncio.run(sc._rebuild_departures_from_log(state))

        assert resumed, "the premise: the resume landed inside the window"
        live = resumed[0]
        assert brought_back == 0, "nothing was recorded for a key somebody holds"
        assert sc._key_occupancy.get("chat-conductor") == live._occupancy_generation
        assert _recorded("chat-conductor") is None, "no entry naming the log's manager"
        # And the live occupant can still record its own departure, which a claimed
        # occupancy would have silenced.
        _depart(state, "chat-conductor")
        assert _recorded("chat-conductor") == "chat-lead"
        assert not _outside(state, worker, lead.key)


class TestAHopTheBootPassCouldNotReadIsNotALineThatEnded:
    """An undecided read and a decided-empty one left the SAME state behind.

    The pass files an entry for a hop it resolved, and a decided empty answer -- this
    session reported to nobody -- is an entry too, naming nobody. A hop whose metadata
    would not open was left with neither, and once the pass has finished a key with no
    entry at all is what the walk reads as a line that ENDED.

    A reader loses nothing: both answers refuse reach. The handoff's LOOP GUARD is the
    reason the two must not read alike. Told that nothing sits above the hop it permits
    an edge, and the hop's real manager -- which the log does hold and this process
    could not read -- closes a cycle, which the commit then persists.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    @staticmethod
    def _pass_over_a_closed_hop(state, *, readable):
        """Run the rebuild over one closed hop whose metadata read answers *readable*."""
        from kiro_crew.crew_log import emit

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager", lambda key, tab: ("", True)
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.store.session_units_for_slot",
                _log_reader(
                    "kiro_crew.crew_log.store.session_units_for_slot",
                    lambda slot_key, strict=False: ("unit-0",),
                ),
            )
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: ({"tab_id": "tab-conductor"}, readable),
            )
            return asyncio.run(sc._rebuild_departures_from_log(state))

    def test_an_unreadable_hop_is_unread_for_a_writer(self, tmp_path):
        state = _make_state(tmp_path)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        assert self._pass_over_a_closed_hop(state, readable=False) == 0

        assert _recorded("chat-conductor") is None, "nothing is filed from a read that failed"
        line = sc._walk_manager_line(state, "chat-conductor", "chat-anything", claimed="")
        assert not line.found, "a reader is refused, as it was before"
        assert line.unread, "and a writer must not read this as a line that ended"
        assert line.cause == "forgotten", line.cause
        assert not line.waits, "nothing re-arms it, so the refusal promises no retry"

    def test_a_decided_empty_answer_is_still_a_line_that_ended(self, tmp_path):
        """The control. A hop the pass READ and found nothing above is an entry.

        Without this the fix above would read as "refuse every closed hop", which costs
        the handoff below a finished conductor for no reason at all.
        """
        state = _make_state(tmp_path)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        assert self._pass_over_a_closed_hop(state, readable=True) == 1

        assert _recorded("chat-conductor") == "", "a decided empty answer is written down"
        line = sc._walk_manager_line(state, "chat-conductor", "chat-anything", claimed="")
        assert not line.found
        assert not line.unread, "both polarities read this to the end"

    def test_the_mark_never_takes_a_key_a_live_session_holds(self, tmp_path):
        """The same window the rest of the pass guards: a resume during the read."""
        state = _make_state(tmp_path)
        live = _running_as(state.get_or_create_slot("chat-conductor"), "tab-c")

        assert not sc.note_hop_left_unreadable(state, "chat-conductor")
        assert sc._key_occupancy.get("chat-conductor") == live._occupancy_generation
        # And a key whose occupancy is already on record keeps the generation it has.
        sc._key_occupancy["chat-closed"] = 4242
        assert not sc.note_hop_left_unreadable(state, "chat-closed")
        assert sc._key_occupancy["chat-closed"] == 4242


class TestAnUnreadableReuseCheckRefusesRatherThanResolving:
    """The boot pass must not read a FAILED units list as "this key had one occupant".

    The rule: nothing whose absence weakens a boundary may be concluded from a read
    that did not land. Filing a rebuilt entry without a tombstone because the unit list
    would not open means the walk resolves an edge naming that key confidently, which
    grants reach. Counting the unreadable case as reused costs reach instead.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_an_unreadable_unit_list_comes_back_tombstoned(self, tmp_path):
        from kiro_crew.crew_log import emit

        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        def _unreadable(_key, **_kw):
            raise OSError("the unit list could not be read")

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: ({"tab_id": "tab-conductor"}, True),
            )
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager",
                    lambda key, tab: (
                        ("chat-lead", True) if key == "chat-conductor" else ("", False)
                    ),
                ),
            )
            patched.setattr("kiro_crew.crew_log.store.session_units_for_slot", _unreadable)
            asyncio.run(sc._rebuild_departures_from_log(state))

        assert _unidentifiable("chat-conductor"), "unreadable counts as ambiguous"
        assert _outside(state, worker, lead.key), "so the hop is refused, not resolved"

    def test_a_readable_single_unit_still_restores_the_reach(self, tmp_path):
        """The control: the conservative answer must not swallow the ordinary one."""
        from kiro_crew.crew_log import emit

        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-conductor"

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(emit, "enabled", lambda: True)
            patched.setattr(
                state.conversation_log,
                "get_metadata_status",
                lambda history_key: ({"tab_id": "tab-conductor"}, True),
            )
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager",
                    lambda key, tab: (
                        ("chat-lead", True) if key == "chat-conductor" else ("", False)
                    ),
                ),
            )
            patched.setattr(
                "kiro_crew.crew_log.store.session_units_for_slot",
                _log_reader(
                    "kiro_crew.crew_log.store.session_units_for_slot",
                    lambda slot, strict=False: ("u",),
                ),
            )
            # The occupancy a real manager entry carries. Without it the restored
            # handoff is unverifiable and refuses -- correct, and not this test's
            # subject, which is what an unreadable UNIT list counts as.
            patched.setattr(
                "kiro_crew.crew_log.session_tree.slot_manager_tab",
                _log_reader(
                    "kiro_crew.crew_log.session_tree.slot_manager_tab",
                    lambda key, tab: lead._tab_id,
                ),
            )
            asyncio.run(sc._rebuild_departures_from_log(state))

        assert not _unidentifiable("chat-conductor")
        assert not _outside(state, worker, lead.key)


class TestTheGapAnswerDoesNotRestOnAnEvictableTable:
    """The audit's own finding: the shadow that names an eviction is itself evictable.

    A key the cap dropped must read UNREAD, not as a line that ended. That answer was
    conditioned on finding the key in ``_recently_evicted_keys`` -- but that table is
    bounded by the same number as the one it shadows, so a key can age out of BOTH, and
    the walk would then read a forgotten record as "nothing sits above this hop". A
    boundary that moves when a cache forgets is the thing the rule forbids.

    So the condition is the eviction COUNTER, which nothing evicts. Once this process
    has dropped anything it cannot prove a given gap is not one of the drops, so it
    refuses. Before the first eviction a gap still reads to the end, which is the case
    that must keep working.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        _forget_everything()
        yield
        _forget_everything()

    def test_a_gap_refuses_once_anything_has_been_evicted_even_if_the_shadow_forgot(self, tmp_path):
        state = _make_state(tmp_path)
        _line(state, "chat-top", "chat-mid", "chat-worker")
        worker = state.get_slot("chat-worker")
        state._slots.pop("chat-mid", None)

        # What the two tables look like after the shadow has also aged this key out:
        # no occupancy, no shadow entry, but the process HAS evicted something.
        sc._key_occupancy.pop("chat-mid", None)
        sc._recently_evicted_keys.clear()
        sc._key_occupancy_evictions[0] = 1

        result = sc._walk_manager_line(state, sc.manager_of(worker), "chat-top")
        assert not result.found
        assert result.unread, "a forgotten record must not read as a line that ended"
        assert result.cause == "forgotten"

    def test_a_gap_on_a_process_that_has_evicted_nothing_still_reads_to_the_end(self, tmp_path):
        """The case that must not refuse: no eviction has happened, so no key was lost.

        A tab closed before a restart is this shape, and refusing it would freeze every
        handoff underneath it.
        """
        state = _make_state(tmp_path)
        _line(state, "chat-top", "chat-mid", "chat-worker")
        worker = state.get_slot("chat-worker")
        state._slots.pop("chat-mid", None)
        _forget_everything()

        assert sc._key_occupancy_evictions[0] == 0
        result = sc._walk_manager_line(state, sc.manager_of(worker), "chat-top")
        assert not result.found
        assert not result.unread, "both polarities read this to the end"

    def test_the_shadow_still_names_the_key_when_it_knows_it(self, tmp_path, caplog):
        """The shadow keeps its one job: saying WHICH key, for the log line."""
        import logging

        state = _make_state(tmp_path)
        lead = state.get_or_create_slot("chat-lead")
        lead._created_by = "chat-owner"
        _hop_whose_record_the_cap_took(state, "chat-upper", creator=lead.key)
        worker = state.get_or_create_slot("chat-worker")
        worker._created_by = "chat-upper"
        assert "chat-upper" in sc._recently_evicted_keys

        with caplog.at_level(logging.INFO, logger=sc.logger.name):
            assert _outside(state, worker, lead.key)
        named = [r for r in caplog.records if "retention limit" in r.getMessage()]
        assert named and "chat-upper" in named[0].getMessage()


class TestTheWalkHasNoBranchAheadOfItsOwnChecks:
    """The enumeration, mechanized, because ten findings all had the same shape.

    Each of those was a guard correct where it ran, on a path that reached a decision
    without running it, and each was closed by adding the guard to one more branch. This
    closes the SHAPE: it reads the walk's own source and asserts the positional rule the
    docstring states, so a branch inserted ahead of a check fails here rather than
    waiting for the eleventh reviewer.
    """

    @staticmethod
    def _walk_source():
        """The walk's CODE, with its docstring removed.

        The docstring quotes the branches by name on purpose -- the ordering rule is
        stated there so the next change has a list to check itself against -- and a pin
        that scans it would read those quotations as the branches themselves and answer
        about prose. So the prose is dropped and the pins below are about code only.
        """
        import inspect

        lines = inspect.getsource(sc._walk_manager_line).splitlines()
        opened = next(i for i, line in enumerate(lines) if line.strip().startswith('"""'))
        closed = next(i for i, line in enumerate(lines[opened + 1 :], opened + 1) if '"""' in line)
        return lines[closed + 1 :]

    def test_the_identity_check_precedes_every_use_of_the_identity(self):
        """``_key_is_ambiguous`` before the identity match and the visited test."""
        lines = self._walk_source()
        check = next(i for i, l in enumerate(lines) if "_key_is_ambiguous(state, key)" in l)
        match = next(i for i, l in enumerate(lines) if "key == wanted_key" in l)
        visited = next(i for i, l in enumerate(lines) if "key in seen" in l)
        assert check < match, "the identity match is an identity claim and came tenth"
        assert check < visited

    def test_every_live_hop_manager_read_goes_through_the_chokepoint(self):
        """The checks are no longer the walk's to remember, and that is the point.

        This walk once asked four questions of a live hop itself -- was the manager ever
        read, is the recorded grant admissible, is the session answering to that key the
        one the record names, do the two still agree on a store -- each added after a
        finding, each a copy of a question ``require_manager_of`` already asks. So the
        pin is no longer "the checks come first": it is that the walk reads a live hop's
        manager ONLY through the function that refuses first, and holds no raw read of
        its own to get the order wrong in.
        """
        lines = self._walk_source()
        unsettled = next(i for i, l in enumerate(lines) if "_hop_handoff_unsettled(key)" in l)
        split = next(i for i, l in enumerate(lines) if "above = state.get_slot(key)" in l)
        assert unsettled < split, "asked of every hop, before the live/departed branch"

        reads = [i for i, l in enumerate(lines) if "require_manager_of(state, above)" in l]
        assert reads, "the walk must still read a live hop's manager somewhere"
        raw = [
            line.strip()
            for line in lines
            if ("manager_of(above)" in line and "require_manager_of(state, above)" not in line)
            or 'getattr(above, "_manager"' in line
        ]
        assert raw == [], raw

    def test_every_production_walk_passes_the_askers_claim(self):
        """``claimed`` has a default, and no production caller may rely on it.

        The default exists because one of the two callers has no asker at all -- the
        handoff's loop guard asks whether a KEY is above another, not whether somebody
        is -- and a required argument there would be a blank passed to satisfy a
        signature. What must not happen is an authorization site inheriting the
        unchecked answer by omission, so this reads the module's own source and requires
        every call in it to say what the asker claims.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(sc))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_walk_manager_line"
        ]
        assert len(calls) == 2, [node.lineno for node in calls]
        missing = [
            node.lineno
            for node in calls
            if "claimed" not in {keyword.arg for keyword in node.keywords}
        ]
        assert missing == [], missing

    def test_only_the_empty_hop_sits_ahead_of_the_identity_check(self):
        """The third early return, and the only one allowed ahead of the checks.

        It is allowed because there is no hop to ask about. Any OTHER return reaching
        the front of the loop body is the eleventh variant and fails here.
        """
        lines = self._walk_source()
        body = next(i for i, l in enumerate(lines) if "for _ in range(LINEAGE_WALK_MAX_DEPTH)" in l)
        check = next(i for i, l in enumerate(lines) if "_key_is_ambiguous(state, key)" in l)
        ahead = [
            line.strip()
            for line in lines[body + 1 : check]
            if line.strip().startswith(("if ", "return ", "elif "))
        ]
        assert ahead == ["if not key:", "return _LineRead(False, False)"], ahead
