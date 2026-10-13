"""``session_handoff``: moving WHO MAY DRIVE a session, and nothing else with it.

The ownership fence admits the session that CREATED a target. That is the whole reason
a lead cannot put the workers it opened under the conductor it opened to run them: the
conductor created none of them, so every send, read, stop, close and adopt it makes
comes back ``not_creator``. ``_manager`` is the holder the fence reads instead, and
this verb is its only writer.

Two properties carry the design and each has its own section below.

* With nothing handed over, the fence decides what it always decided. ``_manager`` is
  empty on every slot until the verb writes one, and empty means "the creator holds
  it", so the widened read and the old read return the same value.
* A handoff widens the verbs a manager needs in order to RUN its reports, and not the
  verbs that mint a session. ``create`` and ``fork`` stay on ``_created_by``, which is
  never rewritten, so a manager that was GIVEN a worker cannot turn it into a deputy
  bound to the giver's memory store.

The tree-side harness is imported from ``test_session_control_tree`` rather than
restated: these tests drive the same verb body (``_reparent_target``) through the same
projection and the same emitter stub, and a second copy of that setup would drift from
the one the adoption tests exercise. The imported fixtures are autouse there and stay
autouse here, which is what seeds the projection and the session map for this module.
"""

from __future__ import annotations

import asyncio

import pytest
from chat_test_helpers import _make_state
from test_session_control_tree import (  # noqa: F401 -- imported FIXTURES apply here
    _emitted,
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
from kiro_crew.dashboard.slot_persistence import metadata_codec as codec
from kiro_crew.history import SLOT_OWNED_META_KEYS
from kiro_crew.members import DM_SLOT_KEY_PREFIX

CRON_SLOT = "cron-2c3b2e25"

#: Every verb a handoff is meant to widen. Spelled out rather than read off the
#: production set, so a verb quietly added to that set fails this file.
#: Every operation a handoff moves. ``summary`` is in here with ``read`` because it is
#: authorized as a transcript read is: a manager allowed the whole transcript is not
#: withheld the cached one-line summary of it.
WIDENED = ("send", "read", "summary", "stop", "close", "adopt", "release", "handoff")


@pytest.fixture(autouse=True)
def _saves(tmp_path, _floor_monkeypatch):
    """Report every slot save as landed, and record which slots were saved.

    The verb's commit is a REAL save on the live history; what these tests are about is
    what it writes to the slot and in what order, so the disk write is stubbed and the
    call is observed. The failure half is driven by overriding this per test.

    Patched through ``_floor_monkeypatch``, which is the rule for an AUTOUSE fixture:
    the shared ``monkeypatch`` stack is the one a test unwinds when it overrides this
    stub, and a floor patch sits under that stack rather than in it.
    """
    _floor_monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "crew-home"))
    handed_over.reset_caches()
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


def _team(state, *, handed: bool = False):
    """A lead, a conductor and a worker the LEAD created, as a dispatch leaves them.

    With *handed* the worker's ``_manager`` names the conductor, which is the state the
    verb writes -- set directly here so the fence can be read on its own, apart from
    the verb that produces it.
    """
    owner = _live(state, _slot(state, "chat-owner-tab"), "sid-owner")
    lead = _live(state, _slot(state, "chat-lead"), "sid-lead")
    conductor = _live(state, _slot(state, "chat-conductor"), "sid-conductor")
    worker = _live(state, _slot(state, "chat-worker"), "sid-worker")
    # The lead is itself agent-created, so it is FENCED -- which is the caller class
    # the fence is about. ``chat-owner-tab`` stays unattributed and stands in for the
    # person's own tab, the one caller the fence does not apply to at all.
    lead._created_by = owner.key
    conductor._created_by = lead.key
    worker._created_by = lead.key
    if handed:
        worker._manager = conductor.key
        worker._manager_tab = conductor._tab_id
    return lead, conductor, worker


# ── with nothing handed over, the fence is the fence it was ──────────────────


def test_a_session_nobody_handed_over_is_managed_by_its_creator(tmp_path):
    """The default that makes the field free to add: empty reads as the creator."""
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    assert worker._manager == ""
    assert sc.manager_of(worker) == lead.key
    # With the creator's own identity offered, which is what every asking site passes.
    # No record exists to compare it against, and an absent record must not be read as
    # naming somebody other than the asker -- that reading fences a creator out of a
    # session nobody ever handed anywhere.
    assert not sc._managed_by_other(state, worker, lead.key, lead._tab_id)
    assert not sc._created_by_other(worker, lead.key)


def test_the_creator_still_reaches_its_own_worker_on_every_widened_verb(tmp_path):
    """Byte-identical behaviour where no handoff happened, verb by verb."""
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    for operation in WIDENED:
        admitted = sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation=operation
        )
        assert admitted is worker, operation


def test_a_session_that_was_handed_nothing_still_refuses_a_peer(tmp_path):
    """The refusal the fence exists for is untouched: a peer is not a manager."""
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    for operation in WIDENED:
        with pytest.raises(sc.SessionControlError) as exc:
            sc.authorize_target(
                state, caller_session_key=_key(conductor), target=worker.key, operation=operation
            )
        assert exc.value.code == "not_creator", operation


# ── what a handoff widens ────────────────────────────────────────────────────


def test_the_new_manager_reaches_the_verbs_a_handoff_widens(tmp_path):
    """The point of the field: the conductor can now RUN the worker it was given."""
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state, handed=True)
    for operation in WIDENED:
        admitted = sc.authorize_target(
            state, caller_session_key=_key(conductor), target=worker.key, operation=operation
        )
        assert admitted is worker, operation


def test_the_giver_stops_reaching_what_it_handed_over(tmp_path):
    """One direct manager at a time: the reach MOVES rather than being shared.

    The giver keeps its creation record and nothing else, so a lead that handed a
    worker away cannot send to it behind the conductor's back.
    """
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state, handed=True)
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="send"
        )
    assert exc.value.code == "not_creator"
    assert worker._created_by == lead.key, "the creation record is not what moved"


def test_a_handoff_does_not_widen_fork(tmp_path):
    """A forked child inherits the CALLER's memory binding, so the fork gate stays on
    the creator: a manager that was given a worker must not be able to copy it into a
    session bound to the giver's store."""
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state, handed=True)
    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(conductor), target=worker.key, operation="fork"
        )
    assert exc.value.code == "not_creator"
    assert "fork" not in sc.HANDOFF_WIDENED_OPERATIONS
    assert "create" not in sc.HANDOFF_WIDENED_OPERATIONS


def test_memory_delegation_reads_the_creator_chain_and_not_the_manager(tmp_path):
    """``_delegation_lineage_fenced`` is the private-member dispatch gate, and a
    handoff must be unable to move it in either direction.

    Both directions are asserted, because reading ``_manager`` there would have shown
    up as one or the other: a member-rooted chain handed to an owner-rooted session
    would have gained dispatch, and an owner-rooted chain handed to a cron would have
    lost it.
    """
    state = _make_state(tmp_path)
    state.get_or_create_slot("chat-owner-tab")  # unattributed: the owner's own tab
    member_rooted = state.get_or_create_slot("chat-from-member")
    member_rooted._created_by = DM_SLOT_KEY_PREFIX + "radar"
    member_rooted._manager = "chat-owner-tab"
    assert sc._delegation_lineage_fenced(state, "chat-from-member")

    owner_rooted = state.get_or_create_slot("chat-from-owner")
    owner_rooted._created_by = "chat-owner-tab"
    owner_rooted._manager = CRON_SLOT
    assert not sc._delegation_lineage_fenced(state, "chat-from-owner")


# ── the verb ─────────────────────────────────────────────────────────────────


def test_the_manager_hands_a_worker_to_a_conductor(tmp_path, monkeypatch, _saves):
    """The happy path, end to end: the record moves, then the field does.

    The crew log entry is the one the tree already folds, with the RECEIVER as the
    parent -- so the sidebar nests the worker under the session that now runs it, and a
    cold start replays that edge like any other.
    """
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    result = _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )
    assert result == {
        "target": "chat-worker",
        "manager": "chat-conductor",
        "previous_manager": "chat-lead",
        "previous_parent": "chat-lead",
    }
    # BOTH durable writes, in the order the verb owes them: the edge the tree folds,
    # then the manager entry that is the commit point. The second is what a restart
    # rebuilds the field from, so a handoff that emitted only the first would move the
    # sidebar and hand the session back to its creator at the next start.
    assert calls == [
        {
            "op": "adopt",
            "sid": "sid-worker",
            "slot": "chat-worker",
            "parent_slot": "chat-conductor",
            "parent_sid": "sid-conductor",
            "previous_parent_slot": "chat-lead",
            "previous_parent_sid": "sid-lead",
        },
        {
            "op": "manager",
            "sid": "sid-worker",
            "slot": "chat-worker",
            # WHICH session under that name, because the name alone is a name: a
            # closed session's replacement can hold it, and a grant qualified by the
            # key would be restored to the replacement at the next start.
            "slot_tab": worker._tab_id,
            "manager": "chat-conductor",
            "previous": "chat-lead",
            # Both writers of this entry go through one assembly, which names every
            # field: a transfer FORWARD says so by not being a revert, and the
            # occupancy of the manager key lets a rebuild tell the session that was
            # handed this one from a later session that took its name.
            "to_creator": False,
            "manager_tab": conductor._tab_id,
        },
    ]
    assert worker._manager == "chat-conductor"
    assert worker._created_by == "chat-lead", "the creation record is never rewritten"
    # ONE durable write to the slot's line, and it is the marker, never the manager: no
    # slot save runs, so nothing caches who holds the session.
    assert _saves == [f"index:{worker.key}"], "the entry, then only the index write"
    assert worker.key in handed_over.handed_over_slots()[0]


def test_the_new_manager_can_hand_the_session_on(tmp_path, monkeypatch, _saves):
    """A transfer is started by the CURRENT manager, so one handoff cannot be the last."""
    _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state, handed=True)
    second = _live(state, _slot(state, "chat-second"), "sid-second")
    second._created_by = conductor.key
    _hold(state, worker.key, conductor.key, sid="sid-worker")
    result = _run(
        sc.handoff_target(
            state, caller_session_key=_key(conductor), target=worker.key, to=second.key
        )
    )
    assert result["manager"] == "chat-second"
    assert result["previous_manager"] == "chat-conductor"
    assert worker._manager == "chat-second"
    _ = lead


def test_a_session_cannot_hand_over_one_it_does_not_manage(tmp_path, monkeypatch):
    """Neither the report nor a bystander may start a transfer.

    The conductor here has a recipient it genuinely controls, so the only thing wrong
    with the call is the half the design cares about: it does not hold the target.
    """
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state)
    deputy = _live(state, _slot(state, "chat-deputy"), "sid-deputy")
    deputy._created_by = conductor.key
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(conductor), target=worker.key, to=deputy.key
            )
        )
    assert exc.value.code == "not_creator"
    assert calls == [], "a refused handoff writes nothing"
    assert worker._manager == ""


def test_the_recipient_must_be_a_session_the_giver_controls(tmp_path, monkeypatch):
    """You move people in your own org to a manager in your own org."""
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    stranger = _live(state, _slot(state, "chat-stranger"), "sid-stranger")
    stranger._created_by = "chat-somebody-else"
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=stranger.key
            )
        )
    assert exc.value.code == "not_creator"
    assert calls == []
    assert worker._manager == ""


def test_handing_a_session_to_itself_is_refused(tmp_path, monkeypatch):
    """An entry that records a change where none happened is worse than a refusal, and
    a session that is its own parent is a shape the tree cannot present."""
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=worker.key
            )
        )
    assert exc.value.code == "already_manager"
    assert calls == []
    assert worker._manager == ""


def test_handing_a_session_to_the_manager_it_already_has_is_refused(tmp_path, monkeypatch):
    """The owner's own tab is the caller that can reach this: it is unfenced, so it may
    name any session in the workspace -- including the one already holding the target.
    """
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state, handed=True)
    owner = state.get_slot("chat-owner-tab")
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(owner), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "already_manager"
    assert calls == []
    assert worker._manager == conductor.key


def test_a_handoff_to_the_session_answering_to_a_stale_record_is_not_refused(tmp_path, monkeypatch):
    """The refusal everywhere else says "hand it over again". That has to be runnable.

    A record names a KEY and a SESSION. When the session is gone and another answers to
    the key, every read of that record refuses with ``manager_session_changed``, and the
    way out it names is this verb -- so a gate that asked "is the receiver already the
    manager" by key alone refused the one handoff that could clear the record, and the
    refusal became permanent for the life of the slot.

    The receiver is not the session the record names, so there IS something to hand
    over, and the gate asks that through the same predicate the fence asks it with.
    """
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    _lead, conductor, worker = _team(state, handed=True)
    # The record's key still matches the live conductor; its IDENTITY names a session
    # that is gone. This is what a close-and-reoccupy of a deterministic key leaves.
    worker._manager_tab = "the-session-that-was-given-it"
    owner = state.get_slot("chat-owner-tab")

    _run(
        sc.handoff_target(
            state, caller_session_key=_key(owner), target=worker.key, to=conductor.key
        )
    )

    assert worker._manager == conductor.key
    assert worker._manager_tab == conductor._tab_id, "the record now names the live one"
    assert [call["op"] for call in calls].count("manager") == 1, "the move was recorded"


def test_an_append_that_never_landed_leaves_the_manager_where_it_was(tmp_path, monkeypatch):
    """The record is written FIRST, so a lost write costs the move and not the reach.

    The reverse order would be the unsafe one: a field written first and an append that
    then failed would leave the gateway admitting a manager no record names.
    """
    _emitted(monkeypatch, landed=False)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "tree_write_failed"
    assert worker._manager == "", "the reach must not move on a record that is not on disk"


def asyncio_cancelled() -> BaseException:
    """A ``BaseException`` that is not an ``Exception``, which is the case under test."""
    return asyncio.CancelledError()


def test_the_caller_rule_is_re_asked_inside_the_lock(tmp_path, monkeypatch):
    """The lock wait is unbounded, so a rule decided before it is decided on stale
    state -- and the way it goes stale here is the way that matters: another handoff
    lands first, so somebody else holds the target by the time the lock opens.

    Driven through the second warm, which is the suspension the in-lock gate sits
    behind: the first warm is the pre-lock gate's, the second is the re-check's.
    """
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    warms: list[int] = []

    async def _hand_it_away_on_the_final_warm():
        warms.append(1)
        if len(warms) == 2:
            # A real handoff landing in the wait publishes BOTH the key and its
            # occupancy, so the fixture does too -- a manager with no identifiable
            # occupant is a different refusal, and this test is about the caller rule.
            faster = _live(state, _slot(state, "chat-somebody-faster"), "sid-faster")
            worker._manager = faster.key
            worker._manager_tab = faster._tab_id

    monkeypatch.setattr(sc, "prewarm_enabled_check", _hand_it_away_on_the_final_warm)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "not_creator"
    assert len(warms) == 2, "the re-check has to be behind its own warm"
    assert calls == [], "nothing is appended once the re-check refuses"
    assert worker._manager == "chat-somebody-faster"


def test_a_recipient_replaced_under_the_same_key_is_refused(tmp_path, monkeypatch):
    """A key is not an identity on the RECIPIENT side either: a close plus a same-key
    reopen inside the lock wait would hand the session to a conversation that was
    never named."""
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    warms: list[int] = []

    async def _replace_the_recipient_on_the_final_warm():
        warms.append(1)
        if len(warms) == 2:
            del state._slots[conductor.key]
            replacement = _live(state, _slot(state, conductor.key), "sid-replacement")
            replacement._created_by = lead.key

    monkeypatch.setattr(sc, "prewarm_enabled_check", _replace_the_recipient_on_the_final_warm)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "recipient_replaced"
    assert calls == []
    assert worker._manager == ""


def test_a_handoff_is_refused_when_the_tree_is_not_recorded(tmp_path, monkeypatch):
    """With no crew log there is nowhere to write the edge, and a verb whose record
    cannot be written has not done anything -- so the field is not written either."""
    _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    monkeypatch.setattr(emit, "enabled", lambda: False)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "tree_unavailable"
    assert worker._manager == ""


def test_an_empty_recipient_is_refused_before_anything_is_read(tmp_path, monkeypatch):
    _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to="  "))
    assert exc.value.code == "recipient_required"


# ── the manager survives a restart ───────────────────────────────────────────


def test_the_slot_line_carries_no_manager_at_all(tmp_path, monkeypatch):
    """The crew log is the ONLY store. A save writes no manager onto the slot's line, so
    there is no second copy to prefer, to reconcile, or to be wrong -- and no line can
    hand a session to anybody.

    A copy here would be written on every save and read on every restore, and then
    overwritten by the log every time, because the start establishes the field from the
    log on the same path that restores these slots. It was also the way a revoked
    manager came back: a line outlives the grant it names.
    """
    state = _make_state(tmp_path)
    source = _slot(state, "chat-worker")
    source._created_by = "chat-lead"
    source._manager = "chat-conductor"
    meta = codec.encode(
        source, folds=codec.SaveFolds(memory_mode=source.memory_mode, created_at=source.created_at)
    )
    assert "manager" not in meta, "the live field is not persisted"
    assert meta["created_by"] == "chat-lead", "and the creator still is"


def test_a_restored_slot_holds_no_manager_until_the_log_says_so(tmp_path):
    """A slot comes up with NO manager, whatever its line says, because the line says
    nothing about one: the codec has no row for it, so the attribute is only ever set by
    the log reconstruction. Empty means the creator holds the session -- the reach the
    gateway had before any handoff."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-restored")
    codec.apply(
        state,
        slot,
        {"created_by": "chat-lead"},
        codec.Restore(name="chat-restored"),
        # The index this slot's own hydrate is given: nothing was handed over on this
        # store. Handed in as a value because the codec reads no file -- the read
        # belongs to the pass above it, once, off the loop.
        handoff_index=handed_over.handed_over_slots(),
    )
    assert slot._manager == ""
    assert sc.manager_of(slot) == "chat-lead"


# ── the verb is reachable the way every other session verb is ────────────────


def test_the_tool_the_route_and_the_schema_name_one_verb():
    """Four declarations have to agree or the verb is unreachable in production.

    The path has to be in the strict internal set, or the MCP caller's header is
    ignored and the route falls through to the cookie branch while handler tests still
    pass. The tool has to declare that route, and the argument schema has to exist
    under the tool's own name, because an unvalidated tool call is refused before the
    handler sees it.
    """
    from kiro_crew.dashboard.handlers import session_control as handlers
    from kiro_crew.dashboard.server import _STRICT_INTERNAL_API_PATHS
    from kiro_crew.mcp_dashboard import TABLE
    from kiro_crew.validation import MCP_DASHBOARD_SCHEMAS

    assert "/api/session-control/handoff" in _STRICT_INTERNAL_API_PATHS
    assert callable(handlers.api_session_control_handoff)
    tool = next(entry for entry in TABLE if entry.name == "session_handoff")
    assert tool.routes == ("POST /api/session-control/handoff",)
    assert sorted(tool.schema["required"]) == ["target", "to"]
    assert sorted(field.name for field in MCP_DASHBOARD_SCHEMAS["session_handoff"].fields) == [
        "target",
        "to",
    ]


def test_a_channel_agent_cannot_hand_sessions_around():
    """A channel agent acts on words from a thread other people are in, and this verb
    decides who may drive a session -- the same containment the tree verbs have."""
    from kiro_crew.channel import CHANNEL_AGENT_BLOCKED_TOOLS

    assert "session_handoff" in CHANNEL_AGENT_BLOCKED_TOOLS


# ── the record is the commit point; the slot line is its cache ───────────────
#
# What makes a handoff durable is the crew log entry: append-only, and awaited before
# any reach moves. ``_manager`` is set only after that, and the slot line is encoded
# from ``_manager`` alone -- so no writer, this verb's save or a turn's own, can
# persist a manager the gateway has not put in force. Each case asserts BOTH answers:
# the live one the fence reads, and the one a restart reads back off the line.


def _restored(state, slot) -> str:
    """The manager a restart would read back for *slot*, through the real codec."""
    line = codec.encode(
        slot, folds=codec.SaveFolds(memory_mode=slot.memory_mode, created_at=slot.created_at)
    )
    fresh = state.get_or_create_slot(f"restored-{slot.key}")
    fresh._created_by = slot._created_by
    codec.apply(
        state,
        fresh,
        line,
        codec.Restore(name=f"restored-{slot.key}"),
        handoff_index=handed_over.handed_over_slots(),
    )
    return sc.manager_of(fresh)


def test_an_entry_that_never_landed_leaves_both_managers_unchanged(tmp_path, monkeypatch):
    """The commit point is the ENTRY. A write the crew log gave up on is a handoff that
    did not happen, so neither the live manager nor the restored one moves."""
    _emitted(monkeypatch, landed=False)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    saves: list[str] = []

    async def _save(state_, slot, **kwargs):
        saves.append(slot.key)
        return True

    monkeypatch.setattr(sc, "save_slot_off_loop", _save)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "tree_write_failed"
    assert worker._manager == ""
    assert sc.manager_of(worker) == "chat-lead"
    assert _restored(state, worker) == "chat-lead"
    assert saves == [], "no line is written for a handoff that never committed"


def test_an_entry_that_landed_moves_the_manager_and_saves_nothing(tmp_path, monkeypatch, _saves):
    """The entry is on disk, so the reach moves -- and that is the whole operation.

    No slot save follows it and the reply carries no ``cached`` key, because there is
    no second copy whose landing could be reported. A handoff that returns at all is a
    handoff the crew log carries.
    """
    _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    result = _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )
    assert result["manager"] == "chat-conductor"
    assert "cached" not in result, "nothing to report the landing of"
    assert worker._manager == "chat-conductor"
    assert sc.manager_of(worker) == "chat-conductor"
    assert _saves == [f"index:{worker.key}"], "no slot save, only the index write"


def test_no_writer_can_persist_a_manager_at_all(tmp_path, monkeypatch):
    """The property the storage model rests on, asserted on the codec itself.

    A concurrent writer -- a turn's own save, the periodic flush -- encodes the slot's
    whole line, and the manager is not on it in either form. So there is nothing any
    writer can put on disk that a later start would read as a grant, which is a
    stronger statement than one about staging: it does not depend on the published field
    being the only staging point, because the field reaches disk by no path.

    The key is not in ``SLOT_OWNED_META_KEYS`` either, so nothing claims it on a save.
    It needs no tolerance: no shipped build ever wrote it, so no line in the field
    carries it, and an undeclared key is ignored on read in any case.
    """
    state = _make_state(tmp_path)
    _lead, _conductor, worker = _team(state)
    folds = codec.SaveFolds(memory_mode=worker.memory_mode, created_at=worker.created_at)
    worker._manager = "chat-conductor"
    assert "manager" not in codec.encode(worker, folds=folds)
    assert "manager" not in codec.encode(worker, merge=True, folds=folds)
    assert "manager" not in SLOT_OWNED_META_KEYS, "and no writer claims the key"
    assert sc.manager_of(worker) == "chat-conductor", "the live field still authorizes"


def test_a_recipient_in_another_memory_store_is_refused(tmp_path, monkeypatch):
    """A memory store is a separate silo, so a recipient bound to one the giver is not
    runs under another principal -- which is what the member refusal is really about.

    Established POSITIVELY from the two slots' own bindings. Asking the config whether
    the recipient is a crew member would answer False for every unreadable or
    incomplete record, and that direction admits the member: a V2 store whose record
    lost ``memory_version`` reads as a plain session.
    """
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    elsewhere = _live(state, _slot(state, "chat-elsewhere"), "sid-elsewhere")
    elsewhere._created_by = lead.key
    elsewhere.memory_store = "member-radar-private"
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=elsewhere.key
            )
        )
    assert exc.value.code == "recipient_other_store"
    assert calls == []
    assert worker._manager == ""


def test_a_recipient_sharing_the_givers_store_is_accepted(tmp_path, monkeypatch, _saves):
    """The positive half, so the rule above is not just a refusal of everything: a
    session the giver created inherits the giver's binding, which is the shape this
    verb exists for."""
    _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    lead.memory_store = "crew-alpha"
    conductor.memory_store = "crew-alpha"
    worker.memory_store = "crew-alpha"
    _hold(state, worker.key, lead.key, sid="sid-worker")
    result = _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )
    assert result["manager"] == "chat-conductor"
    assert worker._manager == "chat-conductor"


def test_the_member_dm_refusal_needs_no_config_record(tmp_path, monkeypatch):
    """A DM slot is the member's own operating surface, and the prefix is part of the
    slot KEY -- so the refusal holds whatever the config record says or fails to say."""
    calls = _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, _conductor, worker = _team(state)
    member = _live(state, _slot(state, "chat-stand-in"), "sid-member")
    member._created_by = lead.key
    monkeypatch.setattr(sc, "_member_caller", lambda state_, key: False)
    monkeypatch.setattr(sc, "_store_is_member_owned", lambda store: False)
    monkeypatch.setattr(member, "key", DM_SLOT_KEY_PREFIX + "radar", raising=False)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to="chat-stand-in"
            )
        )
    assert exc.value.code == "member_recipient"
    assert calls == []
    assert worker._manager == ""


# ── the recipient is re-established against the live object before the reach moves ──


@pytest.mark.parametrize(
    ("what", "change"),
    [
        ("rebound to another store", lambda state, slot: setattr(slot, "memory_store", "other")),
        (
            "channel-linked",
            lambda state, slot: setattr(slot, "linked_session_key", "slack:C1:123.45"),
        ),
        ("closed and reopened", lambda state, slot: state._slots.pop(slot.key, None)),
    ],
)
def test_a_recipient_that_changes_before_the_commit_is_not_given_the_session(
    what, change, tmp_path, monkeypatch
):
    """The gate's verdict about the recipient is formed before the entry is awaited, and
    that await suspends. The recipient's own routes take its slot lock and not this
    verb's, so inside that window its store can be rebound to another principal's, a
    channel link attached, or the session closed. Each changes WHO would hold the handed
    session, so the reach may not move on the earlier verdict.

    Injected from the entry's own settle callback, which is the suspension between the
    last gate and the commit step.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    saves: list[str] = []

    async def _record_save(state_, slot, **kwargs):
        saves.append(slot.key)
        return True

    def _change_then_settle(sid, **kw):
        change(state, conductor)
        settle = kw.get("on_settled")
        if settle is not None:
            settle(True)

    monkeypatch.setattr(sc, "save_slot_off_loop", _record_save)
    monkeypatch.setattr(emit, "on_session_adopted", _change_then_settle)
    with pytest.raises(sc.SessionControlError) as exc:
        _run(
            sc.handoff_target(
                state, caller_session_key=_key(lead), target=worker.key, to=conductor.key
            )
        )
    assert exc.value.code == "recipient_replaced", what
    assert worker._manager == "", what
    assert saves == [], "a refused commit writes no line"


def test_the_live_recipient_check_admits_an_unchanged_recipient(tmp_path, monkeypatch, _saves):
    """Control for the parametrized refusals above: with nothing changing, the same
    check publishes. Without this a predicate that always refused would pass them."""
    _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    _hold(state, worker.key, lead.key, sid="sid-worker")
    result = _run(
        sc.handoff_target(state, caller_session_key=_key(lead), target=worker.key, to=conductor.key)
    )
    assert result["manager"] == "chat-conductor"
    assert worker._manager == "chat-conductor"


def test_the_previous_manager_is_read_from_the_target_not_from_the_caller(
    tmp_path, monkeypatch, _saves
):
    """An owner's own tab is unfenced, so it can hand over a worker it never managed --
    and then the caller is not the manager being replaced. Read from the target."""
    _emitted(monkeypatch)
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)
    owner = state.get_slot("chat-owner-tab")
    _hold(state, worker.key, lead.key, sid="sid-worker")
    result = _run(
        sc.handoff_target(
            state, caller_session_key=_key(owner), target=worker.key, to=conductor.key
        )
    )
    assert result["previous_manager"] == "chat-lead", "the lead created it, so the lead held it"
    assert result["manager"] == "chat-conductor"


# ── the widened set and the broadcast audience agree with each other ─────────


def test_the_widened_set_is_exactly_what_the_fence_reads(tmp_path):
    """The set and this file's own list are pinned against each other, because the two
    ways they drift are opposite failures. An operation missing from the set is a reach
    the manager needs and does not get, and one that should not be there is a reach a
    handoff was never meant to move -- and neither shows up as a test failure anywhere
    else, because every other case in this file reads the set it is checking.
    """
    assert sc.HANDOFF_WIDENED_OPERATIONS == frozenset(WIDENED)


def test_the_giver_stops_reading_the_summary_of_what_it_handed_over(tmp_path):
    """``summary`` moves with ``read``. It is authorized as a transcript read is, and the
    summary is generated FROM the transcript -- so the set has to carry it: left out, the
    giver keeps a window onto a session it cannot open, and the manager is refused the
    one-line form of a transcript it is allowed in full.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state, handed=True)

    admitted = sc.authorize_target(
        state, caller_session_key=_key(conductor), target=worker.key, operation="summary"
    )
    assert admitted is worker, "the manager reads the summary"

    with pytest.raises(sc.SessionControlError) as exc:
        sc.authorize_target(
            state, caller_session_key=_key(lead), target=worker.key, operation="summary"
        )
    assert exc.value.code == "not_creator", "and the giver does not"


def test_the_default_broadcast_audience_follows_the_handoff(tmp_path):
    """The audience has to agree with the fence that judges each delivery, and ``send``
    is one of the verbs a handoff widens.

    Read from the creator instead, the audience was wrong in both directions at once: a
    handed-away session stayed in its GIVER's default audience, where every delivery is
    then refused, and it was missing from the audience of the only session that can
    actually reach it. The caller is excluded from its own audience either way, because
    a self-target is refused and a broadcast must not collect a refusal nobody could
    have avoided.
    """
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)

    assert worker.key in sc.broadcast_audience(state, lead.key), "the creator's, to start"
    assert worker.key not in sc.broadcast_audience(state, conductor.key)

    worker._manager = conductor.key

    worker._manager_tab = conductor._tab_id

    assert worker.key in sc.broadcast_audience(state, conductor.key), "the manager's, after"
    assert worker.key not in sc.broadcast_audience(state, lead.key), "and not the giver's"


def test_the_broadcast_audience_is_unchanged_where_nothing_was_handed_over(tmp_path):
    """The reading only differs once something has been handed over: ``manager_of``
    answers ``_created_by`` until a handoff writes the field, so a gateway that has
    never used the verb gets exactly the audience it got before the field existed."""
    state = _make_state(tmp_path)
    lead, conductor, worker = _team(state)

    assert all(not slot._manager for slot in state._slots.values())
    assert sc.broadcast_audience(state, lead.key) == sorted(
        key
        for key, slot in state._slots.items()
        if key != lead.key and not sc._created_by_other(slot, lead.key)
    )
    assert conductor.key in sc.broadcast_audience(state, lead.key)
