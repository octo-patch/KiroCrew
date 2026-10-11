"""Drain-time re-validation of queued prompts.

Authorization is decided at ADMISSION — ``authorize_target`` for
``session_send``, the authenticated composer for a human typing into a busy
session — but a busy target QUEUES the prompt and delivers it later, and the
containment those decisions rest on can change in between: a target authorized
while unlinked can gain a channel or mirror link before its queue drains.

These tests pin the three-part fix end to end: producers stamp the
admission-time containment on the queue entry (``containment_meta``), the drain
re-asserts the same constraints and drops what stops qualifying, and a drop
is loud (queue card retracted, visible transcript notice, SEL record) — never a
silent vanish. They also pin the two designed non-drops: a constraint already
held at admission is not a change (channel-born sessions keep draining), and
structural orchestration entries are the runner's own machinery, exempt.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from unittest.mock import MagicMock

import pytest
from chat_test_helpers import _make_state

from kiro_crew.dashboard import chat_runner as cr
from kiro_crew.dashboard import session_control as sc
from kiro_crew.dashboard.chat_utils import (
    CRON_NOTIFICATION_KIND,
    SUBAGENT_COMPLETION_KIND,
    SYNTHETIC_RECOVERY_KIND,
    slot_history_key,
)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    """Run in the shipped (enabled) session-control state without reading config."""
    monkeypatch.setattr(sc, "session_control_enabled", lambda: True)


@pytest.fixture(autouse=True)
def _inline_audit(monkeypatch):
    """Route SEL writes inline to a mock: assertable, and no executor thread
    outlives the test."""
    fake = MagicMock()
    monkeypatch.setattr(sc, "sel", lambda: fake)
    monkeypatch.setattr(sc, "_sel_off_loop", lambda write, what: write())
    return fake


def _busy(slot):
    """``running`` is derived (``task is not None and not task.done()``)."""
    task = MagicMock()
    task.done.return_value = False
    slot.task = task
    return slot


def _link(slot):
    slot.linked_session_key = "C0LINKED|1700000000.000100"
    return slot


async def _never_runs(state, slot, prompt):  # pragma: no cover - queued, not run
    raise AssertionError("a queued prompt must not start a turn at enqueue")


def _snapshot_of(entry):
    return entry.get("meta", {}).get(sc.QUEUED_CONTAINMENT_META_KEY)


# ── Producers stamp the admission-time snapshot ──────────────────────────────


def test_human_typed_enqueue_stamps_admission_snapshot(tmp_path):
    """The composer path: ``enqueue_or_run_prompt`` on a busy slot records the
    constraints that held at admission on the entry it queues."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))

    started = slot.enqueue_or_run_prompt("hello there", _never_runs, state)

    assert started is False
    snap = _snapshot_of(slot._queue[0])
    assert snap == {
        "linked": False,
        "mirrored": False,
        "mirror_identity": "",
        "ephemeral": False,
        "app": False,
        "unattended": False,
        "workspace": "default",
        # False at admission by construction: this slot's manager is known, which is
        # what let the entry be admitted for a verb that reads it at all.
        "manager_unknown": False,
    }


def test_session_send_enqueue_stamps_admission_snapshot(tmp_path):
    """The ``session_send`` path shares the same seam and the same stamp."""
    state = _make_state(tmp_path)
    caller = state.get_or_create_slot("chat-1")
    target = _busy(state.get_or_create_slot("chat-2"))

    out = asyncio.run(
        sc.send_to_target(
            state,
            caller_session_key=slot_history_key(caller),
            target="chat-2",
            message="queued message",
        )
    )

    assert out["started"] is False
    snap = _snapshot_of(target._queue[0])
    assert snap is not None
    assert not any(v for k, v in snap.items() if k != "workspace")
    assert isinstance(snap["workspace"], str) and snap["workspace"]


def test_channel_born_enqueue_records_linked_true(tmp_path):
    """A slot that is ALREADY channel-linked at admission records that fact, so
    its own channel's queued messages keep draining (designed behaviour)."""
    state = _make_state(tmp_path)
    slot = _busy(_link(state.get_or_create_slot("chat-1")))

    slot.queue_append("from the thread", meta=sc.containment_meta(state, slot))

    assert _snapshot_of(slot._queue[0])["linked"] is True


def test_requeued_steer_is_stamped(tmp_path):
    """A steer degraded to a queue card is plain user speech re-entering the queue,
    and it is stamped with the containment its SEND was admitted under.

    The stamp comes from the caller, not from reading the slot here: this requeue
    runs in the turn's teardown, past the steer RPC's suspension, so a slot read
    would fold a mirror linked during that suspension into the entry's own baseline
    and the drain would then read the widened audience as one the authorization saw.
    """
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-1")
    slot._pending_steers = ["steer me"]
    slot._steer_admissions["steer me"] = sc.containment_meta(state, slot)

    cr._requeue_unconsumed_steers(state, slot)

    assert _snapshot_of(slot._queue[0]) is not None


# ── The drain drops what stops qualifying, loudly ─────────────────────────


def test_linked_after_enqueue_drops_with_visible_notice(tmp_path, _inline_audit):
    """The issue's core scenario: authorized while unlinked, linked before the
    drain. The entry is dropped, the transcript says why, and the SEL records it."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("exfil me", _never_runs, state)

    _link(slot)
    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []
    notices = [m for m in slot.messages if m.get("role") == "notice"]
    assert notices and "linked to a channel" in notices[-1]["content"]
    assert "dropped" in notices[-1]["content"].lower()
    call = _inline_audit.log_tool_invocation.call_args
    assert call.kwargs["outcome"] == "denied"
    assert "linked" in call.kwargs["metadata"]["newly_held"]


def test_mirror_added_after_enqueue_drops(tmp_path):
    """The other mechanism the issue names: an OUTBOUND mirror link appearing
    between enqueue and drain (the ``_has_channel_mirror`` gap)."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("exfil me", _never_runs, state)

    state.sessions.set_mirror_link(slot_history_key(slot), "C0MIRROR", "1700000000.000200")
    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []
    notices = [m for m in slot.messages if m.get("role") == "notice"]
    assert notices and "outbound channel mirror" in notices[-1]["content"]


def test_mirror_retargeted_while_queued_drops(tmp_path, _inline_audit):
    """A mirror RETARGETED between enqueue and drain (A -> B) keeps the boolean
    true at both ends while substituting the audience — the identity comparison
    must catch what the boolean cannot. Directive provenance does not exempt it:
    the message's author does not control mirror links."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    state.sessions.set_mirror_link(slot_history_key(slot), "C0AUDIENCE_A", "1700000000.000400")
    slot.enqueue_or_run_prompt("meant for audience A", _never_runs, state)
    slot._queue[0]["_directive_user_origin"] = True

    state.sessions.set_mirror_link(slot_history_key(slot), "C0AUDIENCE_B", "1700000000.000500")
    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []
    notices = [m for m in slot.messages if m.get("role") == "notice"]
    assert notices and "retargeted" in notices[-1]["content"]
    call = _inline_audit.log_tool_invocation.call_args
    assert call.kwargs["outcome"] == "denied"
    assert "mirror_retarget" in call.kwargs["metadata"]["newly_held"]


def test_mirror_unchanged_identity_still_drains(tmp_path):
    """The identity comparison must not turn an UNCHANGED admitted mirror into a
    drop: a session mirrored to the same channel at both ends keeps its queue."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    state.sessions.set_mirror_link(slot_history_key(slot), "C0AUDIENCE_A", "1700000000.000400")
    slot.enqueue_or_run_prompt("same audience", _never_runs, state)

    cr._drop_stale_admissions(state, slot)

    assert [q["content"] for q in slot._queue] == ["same audience"]


def test_mirror_unlinked_while_queued_is_a_narrowing_and_drains(tmp_path, _inline_audit):
    """A mirror that goes AWAY while the entry waits is a narrowing, not a
    retarget: every room the delivery can now reach was admitted. The composed
    identity (mirror row + Slack thread) must not turn the thread's unlink into a
    ``mirror_retarget`` drop of a delivery whose audience only shrank -- while a
    rebind to a DIFFERENT thread is a room the admission never saw and drops."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    key = slot_history_key(slot)
    state.sessions.set_mirror_link(key, "C0AUDIENCE_A", "1700000000.000400")
    state.sessions.set_slack_link(key, "1700000000.000700", "C0OPSROOM")
    slot.enqueue_or_run_prompt("meant for both rooms", _never_runs, state)
    assert _snapshot_of(slot._queue[0])["mirror_identity"] == (
        "slack:C0AUDIENCE_A:1700000000.000400|slack:C0OPSROOM:1700000000.000700"
    )

    assert state.sessions.clear_slack_link(key) is True
    cr._drop_stale_admissions(state, slot)

    assert [q["content"] for q in slot._queue] == ["meant for both rooms"]
    assert [m for m in slot.messages if m.get("role") == "notice"] == []
    assert _inline_audit.log_tool_invocation.call_count == 0

    state.sessions.set_slack_link(key, "1700000000.000800", "C0OPSROOM")
    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []
    notices = [m for m in slot.messages if m.get("role") == "notice"]
    assert notices and "retargeted" in notices[-1]["content"]
    assert (
        "mirror_retarget"
        in _inline_audit.log_tool_invocation.call_args.kwargs["metadata"]["newly_held"]
    )


def test_unchanged_containment_drains(tmp_path):
    """No containment change, no drop — including a constraint that already
    held at admission (a channel-born session keeps its queue)."""
    state = _make_state(tmp_path)
    plain = _busy(state.get_or_create_slot("chat-1"))
    plain.enqueue_or_run_prompt("still fine", _never_runs, state)
    born_linked = _busy(_link(state.get_or_create_slot("chat-2")))
    born_linked.queue_append("thread msg", meta=sc.containment_meta(state, born_linked))

    cr._drop_stale_admissions(state, plain)
    cr._drop_stale_admissions(state, born_linked)

    assert [q["content"] for q in plain._queue] == ["still fine"]
    assert [q["content"] for q in born_linked._queue] == ["thread msg"]


def test_unmarked_legacy_entry_fails_closed(tmp_path):
    """An entry with no snapshot is checked against the FULL current-constraint
    set: an untagged producer can never ride a queued prompt past a boundary
    the tagged paths respect."""
    state = _make_state(tmp_path)
    slot = _link(state.get_or_create_slot("chat-1"))
    slot._queue = [{"id": "legacy1", "content": "hi"}]

    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []


def test_unmarked_entry_in_unconstrained_slot_survives(tmp_path):
    """Fail-closed is about constraints that HOLD: with none holding, an
    unmarked entry drains exactly as before this change."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-1")
    slot._queue = [{"id": "legacy1", "content": "hi"}]

    cr._drop_stale_admissions(state, slot)

    assert [q["content"] for q in slot._queue] == ["hi"]


def test_cron_and_subagent_entries_are_exempt(tmp_path):
    """Cron notifications and sub-agent completions are minted fresh by trusted
    internal producers for THIS slot's own turn lifecycle — channel-born
    sessions receive them by design, so the sweep must not touch them even
    unmarked."""
    state = _make_state(tmp_path)
    slot = _link(state.get_or_create_slot("chat-1"))
    slot.queue_append("[cron] tick", kind=CRON_NOTIFICATION_KIND)
    slot.queue_append("[Subagent completion event] done", kind=SUBAGENT_COMPLETION_KIND)

    cr._drop_stale_admissions(state, slot)

    assert len(slot._queue) == 2


def test_unmarked_recovery_entry_fails_closed(tmp_path):
    """A synthetic-recovery entry replays externally admitted content verbatim
    under a fresh queue id, so it is NOT exempt: unmarked, it is checked
    against the full boolean constraint set like any untagged producer — the
    retry window must not ride past a link the original admission never saw."""
    state = _make_state(tmp_path)
    slot = _link(state.get_or_create_slot("chat-1"))
    slot.queue_append("continue", kind=SYNTHETIC_RECOVERY_KIND)

    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []


def test_stamped_recovery_entry_follows_its_admission(tmp_path):
    """A recovery requeue stamps admission context at requeue time: linked
    recorded True keeps draining (channel-born recovery machinery survives),
    while a link appearing AFTER the requeue drops the retry."""
    state = _make_state(tmp_path)
    born_linked = _link(state.get_or_create_slot("chat-1"))
    born_linked.queue_append(
        "continue",
        kind=SYNTHETIC_RECOVERY_KIND,
        meta=sc.containment_meta(state, born_linked),
    )
    cr._drop_stale_admissions(state, born_linked)
    assert [q["content"] for q in born_linked._queue] == ["continue"]

    plain = state.get_or_create_slot("chat-2")
    plain.queue_append(
        "replayed send prompt",
        kind=SYNTHETIC_RECOVERY_KIND,
        meta=sc.containment_meta(state, plain),
    )
    _link(plain)
    cr._drop_stale_admissions(state, plain)
    assert plain._queue == []


# ── Helper semantics ─────────────────────────────────────────────────────────


def test_snapshot_probe_failure_directions(tmp_path):
    """The mirror probe fails closed in BOTH directions: at enqueue an
    unreadable link records not-mirrored (least-authorized admission, so the
    drain re-validates), at drain it reads as mirrored (refuse delivery) and
    carries the ``mirror_unverified`` marker so the notice can say the state
    could not be verified instead of asserting a mirror appeared."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-1")
    state.sessions.get_mirror_link = MagicMock(side_effect=RuntimeError("store down"))

    at_enqueue = sc.containment_snapshot(state, slot, on_probe_failure=False)
    at_drain = sc.containment_snapshot(state, slot, on_probe_failure=True)

    assert at_enqueue["mirrored"] is False
    assert "mirror_unverified" not in at_enqueue
    assert at_drain["mirrored"] is True
    assert at_drain["mirror_unverified"] is True
    # The marker is wording/telemetry, never a constraint to compare.
    assert "mirror_unverified" not in sc.newly_held_constraints(
        at_drain, {sc.QUEUED_CONTAINMENT_META_KEY: dict(at_drain)}
    )
    # An unverifiable drain-side probe fails closed EVEN for an entry admitted
    # under a mirror: the audience may have been retargeted since admission and
    # there is no identity to compare, so "mirrored at both ends" is not
    # positive authorization.
    assert "mirrored" in sc.newly_held_constraints(
        at_drain,
        {sc.QUEUED_CONTAINMENT_META_KEY: {"mirrored": True, "mirror_identity": "slack:C0A:1"}},
    )


def test_admitted_mirrored_entry_drops_on_drain_probe_failure(tmp_path):
    """End to end: a session mirrored at admission whose store stops answering
    by drain time REFUSES delivery — an unverifiable mirror state may hide a
    retarget, and `authorize_target`'s posture is refuse-not-open. The notice
    says the state could not be verified."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    state.sessions.set_mirror_link(slot_history_key(slot), "C0AUDIENCE_A", "1700000000.000600")
    slot.enqueue_or_run_prompt("admitted under mirror A", _never_runs, state)

    state.sessions.get_mirror_link = MagicMock(side_effect=RuntimeError("store down"))
    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []
    notices = [m for m in slot.messages if m.get("role") == "notice"]
    assert notices and "could not be verified" in notices[-1]["content"]


def test_malformed_snapshot_fails_closed():
    """Queue meta is untrusted plumbing: any shape other than a dict snapshot
    degrades to the all-False baseline."""
    now = {"linked": True, "mirrored": False}
    assert sc.newly_held_constraints(now, None) == ["linked"]
    assert sc.newly_held_constraints(now, {"other": 1}) == ["linked"]
    assert sc.newly_held_constraints(now, {sc.QUEUED_CONTAINMENT_META_KEY: "junk"}) == ["linked"]
    assert sc.newly_held_constraints(now, {sc.QUEUED_CONTAINMENT_META_KEY: {"linked": True}}) == []


def test_workspace_change_invalidates_admission(tmp_path):
    """`authorize_target`'s sixth refusal is `workspace_mismatch`, and
    `slot.workspace` is mutable while a queue waits (the agent-switch endpoint
    re-derives it): a prompt admitted under workspace A must not run with
    workspace B's memory, lessons and project context."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.workspace = "team-a"
    slot.enqueue_or_run_prompt("workspace-a prompt", _never_runs, state)

    slot.workspace = "team-b"
    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []
    notices = [m for m in slot.messages if m.get("role") == "notice"]
    assert notices and "different workspace" in notices[-1]["content"]


def test_workspace_not_compared_for_unmarked_entries(tmp_path):
    """An unmarked entry has no least-authorized workspace to assume, so its
    fail-closed floor stays the boolean set — it drains in any workspace when
    no boolean constraint holds."""
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-1")
    slot.workspace = "team-b"
    slot._queue = [{"id": "legacy1", "content": "hi"}]

    cr._drop_stale_admissions(state, slot)

    assert [q["content"] for q in slot._queue] == ["hi"]


def test_directive_user_origin_exempts_linked_only(tmp_path):
    """A human linking their OWN busy session must not destroy the messages
    they already typed (`api_chat` applies no linked refusal to composer
    input) — but the exemption covers ONLY the link: a NEW outbound mirror
    (which the message's author does not control) and a workspace change both
    still drop a directive entry."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.workspace = "team-a"
    slot.queue_append(
        "typed by the owner",
        meta=sc.containment_meta(state, slot),
        directive_user_origin=True,
    )

    _link(slot)
    cr._drop_stale_admissions(state, slot)
    assert [q["content"] for q in slot._queue] == ["typed by the owner"]

    state.sessions.set_mirror_link(slot_history_key(slot), "C0MIRROR", "1700000000.000300")
    cr._drop_stale_admissions(state, slot)
    assert slot._queue == []

    slot2 = _busy(state.get_or_create_slot("chat-2"))
    slot2.workspace = "team-a"
    slot2.queue_append(
        "typed by the owner",
        meta=sc.containment_meta(state, slot2),
        directive_user_origin=True,
    )
    slot2.workspace = "team-b"
    cr._drop_stale_admissions(state, slot2)
    assert slot2._queue == []


def test_drop_retracts_the_queue_card_unconditionally(tmp_path):
    """The frontend's queue card comes from the producer's queue_push, not a
    transcript placeholder row, so the retraction broadcast must not be gated
    on a placeholder existing."""
    state = _make_state(tmp_path)
    state.broadcast_ws = MagicMock()
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("exfil me", _never_runs, state)
    qid = slot._queue[0]["id"]

    _link(slot)
    cr._drop_stale_admissions(state, slot)

    pops = [c for c in state.broadcast_ws.call_args_list if c.args and c.args[0] == "queue_pop"]
    assert pops and pops[-1].args[1]["queue_id"] == qid


def test_probe_failure_drop_says_unverified_not_gained(tmp_path):
    """When the drain-side mirror probe fails, the refusal stands (fail closed)
    but the notice must describe an unverifiable state, not assert a mirror
    appeared."""
    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("exfil me", _never_runs, state)

    state.sessions.get_mirror_link = MagicMock(side_effect=RuntimeError("store down"))
    cr._drop_stale_admissions(state, slot)

    assert slot._queue == []
    notices = [m for m in slot.messages if m.get("role") == "notice"]
    assert notices and "could not be verified" in notices[-1]["content"]
    assert "gained an outbound channel mirror" not in notices[-1]["content"]


def test_drop_audit_uses_the_effective_session_key(tmp_path, _inline_audit):
    """A linked slot's turns run under `linked_session_key`; filing the drop
    under the slot key would hide exactly the drops this feature records."""
    from kiro_crew.dashboard.chat_utils import effective_session_key

    state = _make_state(tmp_path)
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("exfil me", _never_runs, state)

    _link(slot)
    cr._drop_stale_admissions(state, slot)

    call = _inline_audit.log_tool_invocation.call_args
    assert call.kwargs["session_key"] == effective_session_key(slot)
    assert call.kwargs["metadata"]["target"] == slot.key


@pytest.mark.asyncio
async def test_consumed_entry_emits_an_allowed_audit(tmp_path, monkeypatch, _inline_audit):
    """The ALLOW side of the drain's permission decision is audited too,
    matching `authorize_target`'s convention: an entry that passes
    re-validation and becomes a turn leaves an `allowed` SEL row naming its
    queue id — emitted at consumption, so waiting across sweeps does not
    multiply rows."""
    state = _make_state(tmp_path)
    state.subagents = None
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("still fine", _never_runs, state)
    qid = slot._queue[0]["id"]
    slot.task = None

    async def _stub_run_chat(_state, _slot, _prompt, **_kwargs):
        return None

    def _fake_spawn(_state, _slot, coro):
        coro.close()
        task = MagicMock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(cr, "_run_chat", _stub_run_chat)
    monkeypatch.setattr(cr, "spawn_guarded_turn", _fake_spawn)

    assert await cr._start_next_queued_turn(state, slot) is True

    allowed = [
        c
        for c in _inline_audit.log_tool_invocation.call_args_list
        if c.kwargs.get("outcome") == "allowed"
        and c.kwargs.get("tool_name") == "queue_drain_revalidation"
    ]
    assert allowed and qid in allowed[-1].kwargs["metadata"]["queue_ids"]


@pytest.mark.asyncio
async def test_channel_provenance_reaches_the_drained_turn(tmp_path, monkeypatch, _inline_audit):
    state = _make_state(tmp_path)
    state.subagents = None
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.queue_append(
        "watch the pull request",
        meta=sc.containment_meta(state, slot),
        directive_user_origin=True,
        directive_channel_origin=True,
    )
    slot.task = None
    captured: dict[str, object] = {}

    def _stub_run_chat(_state, _slot, _prompt, **kwargs):
        captured.update(kwargs)

        async def _done():
            return None

        return _done()

    def _fake_spawn(_state, _slot, coro):
        coro.close()
        task = MagicMock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(cr, "_run_chat", _stub_run_chat)
    monkeypatch.setattr(cr, "spawn_guarded_turn", _fake_spawn)

    assert await cr._start_next_queued_turn(state, slot) is True
    assert captured["_directive_user_origin"] is True
    assert captured["_directive_channel_origin"] is True


@pytest.mark.asyncio
async def test_mixed_origin_merge_keeps_channel_provenance(tmp_path, monkeypatch, _inline_audit):
    """Any channel entry makes the merged turn channel-authorized.

    Changing the origin reduction back to ``all`` would let an adjacent dashboard
    entry erase the channel boundary for the whole model turn.
    """
    state = _make_state(tmp_path)
    state.subagents = None
    slot = _busy(state.get_or_create_slot("chat-1"))
    admission = sc.containment_meta(state, slot)
    slot.queue_append(
        "from the linked channel",
        meta=admission,
        directive_user_origin=True,
        directive_channel_origin=True,
    )
    slot.queue_append(
        "from the dashboard",
        meta=admission,
        directive_user_origin=True,
    )
    slot.task = None
    config = MagicMock()
    config.dashboard.merge_queued_messages = True
    captured: dict[str, object] = {}

    def _stub_run_chat(_state, _slot, _prompt, **kwargs):
        captured.update(kwargs)

        async def _done():
            return None

        return _done()

    def _fake_spawn(_state, _slot, coro):
        coro.close()
        task = MagicMock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(cr.KiroCrewConfig, "load", lambda: config)
    monkeypatch.setattr(cr, "_run_chat", _stub_run_chat)
    monkeypatch.setattr(cr, "spawn_guarded_turn", _fake_spawn)

    assert await cr._start_next_queued_turn(state, slot) is True
    assert captured["_directive_channel_origin"] is True


def test_requeued_steer_in_a_plain_slot_carries_human_provenance(tmp_path):
    """A COMPOSER steer keeps the audience exemption: its author typed into this
    session's own surface, so linking the session is that owner's deliberate act.

    Provenance is now reported by the steer's caller rather than derived from the
    slot. It was derivable while ``steer_into_running_turn`` had exactly one caller
    (the api_chat composer branch); ``session_send``'s steer is a second caller with
    no such author, so the flag has to say which one produced the text. The composer
    records ``True`` -- ``user_origin`` defaults to it -- which is what this asserts.
    """
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-1")
    slot._pending_steers = ["steer me"]
    slot._steer_user_origin["steer me"] = True

    cr._requeue_unconsumed_steers(state, slot)

    assert slot._queue[0].get("_directive_user_origin") is True


def test_a_requeued_steer_with_no_recorded_provenance_fails_closed(tmp_path):
    """An unrecorded steer is not treated as the session owner's.

    The exemption is the one thing a peer must not inherit, so the absent case has
    to fall on the unexempted side: the entry faces the LINKED drop like any other
    queued prompt rather than riding past it on an assumption nobody made.
    """
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot("chat-1")
    slot._pending_steers = ["no provenance recorded"]

    cr._requeue_unconsumed_steers(state, slot)

    assert slot._queue[0].get("_directive_user_origin") is not True


# ── The full drain path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_drops_stale_entry_and_starts_nothing(tmp_path, monkeypatch):
    """Through ``_start_next_queued_turn``: a stale entry is dropped before the
    dequeue ever sees it, and no turn is spawned for it."""
    state = _make_state(tmp_path)
    state.subagents = None
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("exfil me", _never_runs, state)
    slot.task = None  # the turn ended; the tail-drain runs
    _link(slot)

    spawned = MagicMock()

    def _no_spawn(_state, _slot, coro):  # pragma: no cover - must not be reached
        coro.close()
        spawned()
        return MagicMock()

    monkeypatch.setattr(cr, "spawn_guarded_turn", _no_spawn)
    started = await cr._start_next_queued_turn(state, slot)

    assert started is False
    assert not spawned.called
    assert slot._queue == []
    assert any(m.get("role") == "notice" for m in slot.messages)


@pytest.mark.asyncio
async def test_drain_strips_snapshot_from_the_persisted_row(tmp_path, monkeypatch):
    """A surviving entry's snapshot is queue plumbing, consumed at the drain —
    it must not ride into the transcript row's persisted meta."""
    state = _make_state(tmp_path)
    state.subagents = None
    slot = _busy(state.get_or_create_slot("chat-1"))
    slot.enqueue_or_run_prompt("still fine", _never_runs, state)
    slot.task = None

    async def _stub_run_chat(_state, _slot, _prompt, **_kwargs):
        return None

    def _fake_spawn(_state, _slot, coro):
        coro.close()
        task = MagicMock()
        task.done.return_value = True
        return task

    monkeypatch.setattr(cr, "_run_chat", _stub_run_chat)
    monkeypatch.setattr(cr, "spawn_guarded_turn", _fake_spawn)

    started = await cr._start_next_queued_turn(state, slot)

    assert started is True
    user_rows = [m for m in slot.messages if m.get("role") == "user"]
    assert user_rows, "the drained entry must land as a user row"
    row_meta = user_rows[-1].get("meta") or {}
    assert sc.QUEUED_CONTAINMENT_META_KEY not in row_meta


# ── Constraint-set parity with authorize_target ──────────────────────
#
# The constraint set now has two hand-maintained spellings: `authorize_target`
# refuses admission inline, and `containment_snapshot` re-derives the same
# predicates so the drain can re-assert them. Drift between them fails OPEN --
# a refusal added to `authorize_target` alone is enforced at enqueue and NOT at
# delivery, silently reopening the enqueue->drain window this file exists to
# close, with nothing red. These tests make that divergence fail CI instead.
#
# ADDING A REFUSAL TO `authorize_target`? Classify it in exactly one of the two
# tables below. Either it is a per-slot containment constraint -- then it needs
# a `containment_snapshot` key, or the drain cannot re-check it -- or it is not,
# and it belongs in `_NON_CONTAINMENT_REFUSALS` with a reason.

# Target-side containment refusals, mapped to the snapshot key that re-asserts
# each one at drain time. `workspace_mismatch` is the sixth (see
# `test_workspace_change_invalidates_admission`); it is an identity rather than
# a boolean, but it is still a constraint the drain compares.
_TARGET_CONTAINMENT_REFUSALS = {
    "linked_session_target": "linked",
    "mirrored_target": "mirrored",
    "ephemeral_target": "ephemeral",
    "app_scoped_target": "app",
    "unattended_target": "unattended",
    "workspace_mismatch": "workspace",
    # Both ways `require_manager_of` can refuse, against the one snapshot boolean that
    # says who manages the slot is not known. Containment, not caller-side: the value
    # is a property of the TARGET, a handoff moves it while an entry waits, and a
    # failed read can retract it -- so the drain has something of the target's to
    # re-compare, unlike `not_creator` below.
    #
    # Compared only when the entry RECORDED it, like `workspace`: the boolean set is
    # the wrong fail-closed floor for this one, because the restore path strips the
    # stamp and every hydrated slot starts unresolved, so counting it against the
    # unmarked baseline would delete every prompt a person typed before a restart. The
    # population that skips the check is restores, whose admission was the composer and
    # never read the manager; every cross-session producer stamps at enqueue.
    "manager_unresolved": "manager_unknown",
    # The third way the value is not usable: read, decided, and the recorded manager
    # cannot hold the session. Same snapshot boolean, because the drain-time
    # consequence is identical -- there is no manager to authorize a delivery against.
    "manager_grant_inadmissible": "manager_unknown",
    # The fourth way the value is not usable: the key is a name a later session took,
    # so the recorded manager cannot be identified. Same snapshot boolean, same
    # drain-time consequence -- no manager to authorize a delivery against.
    "manager_session_changed": "manager_unknown",
}

_NON_CONTAINMENT_REFUSALS = {
    # Caller-side: a property of who is asking, not of the target slot. The
    # drain has nothing to re-validate -- by then the caller's turn is over.
    "caller_unidentified",
    "unattended_caller",
    "app_scoped_caller",
    "ephemeral_caller",
    "linked_session_caller",
    "mirrored_caller",
    "caller_gone",
    # Caller-relative ownership: fires only for member callers, comparing the
    # caller's key to the target's holder. For every verb OUTSIDE
    # `HANDOFF_WIDENED_OPERATIONS` that holder is `created_by`, and both sides are
    # written once at birth and never mutate, so the relation that admitted an entry
    # cannot flip while it waits -- and the drain has no caller left to re-evaluate.
    #
    # For a WIDENED verb the holder is `manager_of`, which a handoff MOVES, so the
    # relation can flip mid-wait: an entry admitted while A managed the target is
    # drained after the target was handed to B. `manager_unknown` above covers the
    # case where the value cannot be read; it does NOT cover a value that was read
    # and then moved. That half is not re-checked HERE because it is not a per-slot
    # constraint: it is a relation between the target's current manager and one entry's
    # stamped sender, so it is per-entry and lives beside this table rather than in it
    # (`session_control.sender_lost_its_reach`, called by `_drop_stale_admissions`).
    # `manager_unknown` above covers a value that cannot be READ; that one covers a
    # value read and then moved.
    "not_creator",
    # The three ways the manager LINE can refuse, which are `not_creator`'s own case one
    # level further up: the same caller-relative relation, read through more than one
    # hop. Not per-slot constraints, so they are not re-asserted from the snapshot --
    # the relation they describe is between the target's line and one entry's stamped
    # sender, which is per-entry and lives in `session_control.sender_lost_its_reach`
    # beside `not_creator`'s half.
    #
    # They are three codes rather than one because they are three different facts and a
    # person debugging a denial needs them apart: the record was DROPPED by the cap, the
    # key is held by a session the edges are not about, or a hop's own recorded manager
    # may not hold it. Each refuses; none explains either of the others.
    "manager_line_forgotten",
    "manager_line_ambiguous",
    "manager_line_inadmissible",
    # Not containment: global config state, a resolution failure, and the
    # self-target guard. None of the three can change while an entry waits in a
    # queue in a way the drain could act on.
    "session_control_disabled",
    "target_not_found",
    "self_target",
}

# Comparison detail rather than constraints: both are set by the mirror probe to
# tell the drain HOW to compare `mirrored`, and neither has its own refusal.
_SNAPSHOT_NON_CONSTRAINT_KEYS = {"mirror_identity", "mirror_unverified"}


def _authorize_target_refusal_codes() -> set[str]:
    """Every literal ``deny(..., code)`` in the target-authorization GATE, from source.

    Parsed rather than hand-listed on purpose. A hand-listed copy would be a
    THIRD spelling of the constraint set, free to drift from the other two --
    which is the failure this test exists to catch, not to reproduce.

    The gate is THREE functions, not one. ``authorize_target`` keeps the
    target-side refusals and delegates the caller-side ones to
    ``refuse_caller_identity`` (before it resolves a target) and
    ``refuse_caller_surface`` (after), so the targetless verbs can share one copy
    of those checks instead of growing a second sequence that drifts. All three
    are parsed together because the tables below describe the gate a queued
    prompt met, and it does not matter to a drain which of the three functions
    raised. Parsing only ``authorize_target`` silently loses eight caller-side
    refusals and reports every one of them as removed.

    One ``deny`` call re-raises a resolution failure with ``exc.code`` rather
    than a literal; it carries no new constraint, so a non-literal code is
    skipped instead of failing the parse.

    ``refuse_caller_surface`` delegates its slot-field half to
    ``_check_caller_slot_fields`` so ``revive_session`` can re-assert those
    refusals synchronously beside its publish; that helper is parsed too.
    The four live-target containment refusals live in ``_live_target_refusal``,
    which RETURNS ``(reason, code)`` for its caller to raise through ``deny``, so
    its returned tuples are read as codes the same way.

    ``_manager_unread_refusal`` is read the same way and for the same reason. The
    two refusals owed when who manages a slot has not been read deliberately do
    NOT live in the gate: they live on the READ (``require_manager_of``), because a
    condition in the gate had to name which operations consult the manager and that
    list was wrong at three sites which reach the value without passing the gate.
    They still reach the caller through ``authorize_target``, which re-raises them
    via ``deny`` with ``exc.code`` -- a variable the ``deny`` parse cannot see. So
    the helper that spells them is parsed directly; without this the two codes read
    as deleted and the tables they are classified in fail OPEN.
    """
    codes: set[str] = set()
    for fn in (
        sc.authorize_target,
        sc.refuse_caller_identity,
        sc.refuse_caller_surface,
        sc._check_caller_slot_fields,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        codes |= _deny_codes(tree)
    for returning in (
        sc._live_target_refusal,
        sc._manager_unread_refusal,
        sc._manager_record_refusal,
    ):
        codes |= _returned_refusal_codes(ast.parse(textwrap.dedent(inspect.getsource(returning))))
    return codes


def _returned_refusal_codes(tree: ast.AST) -> set[str]:
    """Codes from ``return ("reason", "code")`` statements in a refusal helper.

    A code spelled as a module CONSTANT is resolved to its value rather than skipped.
    One of them is: the single refusal a caller may wait on is named
    (``MANAGER_READ_PENDING``) because two readers decide hold-or-drop on it, and a
    parse that only understood literals would read it as deleted and fail OPEN.
    """
    codes: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)):
            continue
        elts = node.value.elts
        if len(elts) != 2:
            continue
        code = elts[1]
        if isinstance(code, ast.Constant) and isinstance(code.value, str):
            codes.add(code.value)
        elif isinstance(code, ast.Name) and isinstance(getattr(sc, code.id, None), str):
            codes.add(getattr(sc, code.id))
    return codes


def _deny_codes(tree: ast.AST) -> set[str]:
    codes: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "deny"):
            continue
        positional = node.args[1] if len(node.args) >= 2 else None
        keyword = next((kw.value for kw in node.keywords if kw.arg == "code"), None)
        for candidate in (keyword, positional):
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                codes.add(candidate.value)
                break
    return codes


def _base_snapshot(tmp_path) -> dict:
    """A snapshot of an unconstrained slot: every constraint key, no probe extras."""
    state = _make_state(tmp_path)
    return sc.containment_snapshot(
        state, state.get_or_create_slot("chat-1"), on_probe_failure=False
    )


def test_the_parse_finds_the_refusals_it_is_asked_to_pin():
    """Guard the guard: an empty or tiny parse would make the tests below vacuous.

    If the gate's refusals ever stop being spelled as ``deny(..., "code")`` the
    extraction silently returns less, and a parity test that compares against
    nothing passes while pinning nothing.

    One refusal from EACH of the gate's three functions is named, so the parse
    losing a whole function is caught here rather than surfacing as eight
    refusals that look deleted: ``workspace_mismatch`` is raised by
    ``authorize_target`` itself, ``unattended_caller`` by
    ``refuse_caller_identity`` before a target is resolved, and
    ``ephemeral_caller`` by ``refuse_caller_surface`` after.
    """
    codes = _authorize_target_refusal_codes()
    assert len(codes) >= len(_TARGET_CONTAINMENT_REFUSALS) + len(_NON_CONTAINMENT_REFUSALS)
    for code, fn in (
        ("workspace_mismatch", "authorize_target"),
        ("unattended_caller", "refuse_caller_identity"),
        ("ephemeral_caller", "refuse_caller_surface"),
    ):
        assert code in codes, f"{fn}'s refusals are not visible to the parse ({code} missing)"


def test_every_refusal_is_classified():
    """A new refusal in ``authorize_target`` must be classified, not ignored."""
    classified = set(_TARGET_CONTAINMENT_REFUSALS) | _NON_CONTAINMENT_REFUSALS
    unclassified = _authorize_target_refusal_codes() - classified
    assert not unclassified, (
        f"unclassified refusal(s) in authorize_target: {sorted(unclassified)}. "
        "Add each to _TARGET_CONTAINMENT_REFUSALS with the containment_snapshot key "
        "that re-asserts it at drain time, or to _NON_CONTAINMENT_REFUSALS with a "
        "reason it needs no drain-time check."
    )


def test_every_classified_refusal_still_exists():
    """A removed refusal must not leave a stale mapping claiming coverage."""
    codes = _authorize_target_refusal_codes()
    stale = (set(_TARGET_CONTAINMENT_REFUSALS) | _NON_CONTAINMENT_REFUSALS) - codes
    assert not stale, (
        f"refusal(s) mapped here but no longer raised by authorize_target: {sorted(stale)}. "
        "Drop the entry so the tables keep describing the code."
    )


def test_every_containment_refusal_has_a_snapshot_key(tmp_path):
    """The fail-open direction: enforced at admission, unchecked at drain."""
    snap = _base_snapshot(tmp_path)
    missing = set(_TARGET_CONTAINMENT_REFUSALS.values()) - set(snap)
    assert not missing, (
        f"containment_snapshot does not record {sorted(missing)}, so the drain cannot "
        "re-assert the matching refusal(s) -- a queued prompt rides past a constraint "
        "the enqueue side enforces."
    )


def test_the_snapshot_carries_no_unmapped_constraint(tmp_path):
    """The reverse drift: a snapshot key with no refusal behind it."""
    snap = _base_snapshot(tmp_path)
    extra = set(snap) - set(_TARGET_CONTAINMENT_REFUSALS.values()) - _SNAPSHOT_NON_CONSTRAINT_KEYS
    assert not extra, (
        f"containment_snapshot key(s) with no authorize_target refusal behind them: "
        f"{sorted(extra)}. Map each to its refusal, or add it to "
        "_SNAPSHOT_NON_CONSTRAINT_KEYS if it is comparison detail rather than a constraint."
    )


def test_a_handoff_between_enqueue_and_drain_drops_the_givers_queued_send(tmp_path, _inline_audit):
    """The caller-side half of the admission, re-asserted at the drain.

    Every other test in this file changes something about the TARGET. This one changes
    nothing about it: the worker is the same session, unlinked, unmirrored, in the same
    workspace, with a readable manager. What moved is WHO manages it, and the entry was
    admitted on the relation between that value and its sender.

    The relation needed no re-check while the fence read ``created_by``, which is
    written at birth and never moves. A handoff moves it, so the lead's queued message
    would otherwise run in a worker it had already been refused.
    """
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    conductor = state.get_or_create_slot("chat-conductor")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key

    worker.queue_append(
        "finish the migration",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )
    assert worker._queue, "the premise: the lead's delivery is waiting"

    # The containment snapshot is unchanged; only the manager moved.
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    cr._drop_stale_admissions(state, worker)

    assert worker._queue == [], "the giver's queued send does not survive the handoff"
    notices = [m for m in worker.messages if m.get("role") == "notice"]
    assert notices and "no longer manages this one" in notices[-1]["content"]
    call = _inline_audit.log_tool_invocation.call_args
    assert call.kwargs["outcome"] == "denied"
    assert sc.SENDER_LOST_REACH in call.kwargs["metadata"]["newly_held"]


def test_the_new_managers_own_queued_send_survives_the_drain(tmp_path):
    """The counterpart, so the check cannot pass by dropping everything.

    A delivery from the session that NOW manages the target is exactly what the handoff
    was for, and dropping it would make the feature useless the moment it was used.
    """
    state = _make_state(tmp_path)
    conductor = state.get_or_create_slot("chat-conductor")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = "chat-lead"
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id

    worker.queue_append(
        "carry on",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, conductor.key, manager_fenced=True),
        },
    )
    cr._drop_stale_admissions(state, worker)
    assert len(worker._queue) == 1, "the manager's own delivery is untouched"


def test_a_replacement_under_the_senders_key_does_not_inherit_the_queued_send(
    tmp_path, _inline_audit
):
    """A slot key is a NAME. The grant was given to a SESSION, and taken back from it.

    The sequence this drives is ordinary: a conductor is handed a worker and queues a
    message on it; the conductor is closed; another session comes up under the same key
    -- the explicitly-named keys are deterministic, so that happens by design, not by
    accident -- and the worker is handed to THAT one. The record is now consistent: its
    key and its identity both name the session in the chair, so nothing refuses on the
    target side.

    What is left over is one queued entry stamped by a session whose grant was revoked,
    under a name that still resolves. Compared by key, the drain reads the stamp as the
    current manager's and runs the dead predecessor's instruction under the live one's
    authority. The identity beside the key is what tells them apart.
    """
    state = _make_state(tmp_path)
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = "chat-lead"

    gone = state.get_or_create_slot("chat-conductor")
    worker._manager = gone.key
    worker._manager_tab = gone._tab_id
    worker.queue_append(
        "roll the migration back",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, gone.key, manager_fenced=True),
        },
    )
    assert worker._queue, "the premise: the session that held it queued a message"

    # Closed, and a DIFFERENT session takes the name. The worker is handed to that one,
    # so the record is sound again -- and names nobody who queued anything.
    state._slots.pop(gone.key, None)
    successor = state.get_or_create_slot(gone.key)
    assert successor._tab_id != gone._tab_id, "a new session mints its own identity"
    worker._manager_tab = successor._tab_id

    cr._drop_stale_admissions(state, worker)

    assert worker._queue == [], "a revoked sender's message does not run under its successor"
    call = _inline_audit.log_tool_invocation.call_args
    assert call.kwargs["outcome"] == "denied"
    assert sc.SENDER_LOST_REACH in call.kwargs["metadata"]["newly_held"]


def test_the_creators_own_queued_send_survives_a_target_nobody_handed_over(tmp_path):
    """The identity compare must not reach a slot that has no record to compare against.

    ``_created_by`` carries no recorded identity anywhere -- it is written at birth and
    never again -- so an absent manager record names NOBODY, and reading it as "names
    somebody other than the asker" fences every creator out of its own worker. On a
    gateway where nothing was ever handed over that is every delivery there is, which
    makes this the blast radius of the compare and not an edge of it.
    """
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key
    assert worker._manager == "", "the premise: nothing was handed over"

    worker.queue_append(
        "pick the next item up",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )
    cr._drop_stale_admissions(state, worker)

    assert len(worker._queue) == 1, "the creator's own delivery is untouched"


# ── The TIME axis: what the gate reads, and whether it can go stale ──────────


def test_every_mutable_target_value_the_gate_reads_is_re_checked_at_drain():
    """The temporal completeness argument, parsed rather than asserted in prose.

    Admission is decided at enqueue and re-asserted at drain only for what can CHANGE in
    between. That makes the exemption list a claim about MUTABILITY, and a claim like
    that rots silently: the day someone adds a writer for an exempt value, the drain
    stops re-checking something that now moves, and nothing fails.

    So the gate's target-side reads are extracted from source and each one must be
    accounted for -- either carried in `containment_snapshot`, so the drain compares it,
    or named here as immutable with the reason. A read this cannot classify fails.
    """
    import ast
    import inspect
    import textwrap

    #: Target-side values the gate reads that the drain does NOT re-compare, each with
    #: why it cannot go stale. Anything else must be in the snapshot.
    _IMMUTABLE_READS = {
        # Written at birth by the three mint paths and by restore onto a fresh slot
        # object, never on a live one -- which is what `test_created_by_is_only_ever_
        # written_at_birth` below pins. This is the premise that exempts `not_creator`.
        "_created_by": "written at birth only",
        # A slot key never changes, and `unattended` keys off its prefix.
        "key": "slot keys are immutable",
    }
    #: Snapshot fields that are not slot attribute names, mapped to the attribute each
    #: one carries. The drain compares the FIELD; this says which read it covers.
    _COVERED_BY = {
        "_app": "app",
        "linked_session_key": "linked",
        "memory_mode": "ephemeral",
        "workspace": "workspace",
        "_manager": "manager_unknown",
        "_manager_unresolved": "manager_unknown",
        "_manager_grant_inadmissible": "manager_unknown",
    }

    gate = (
        sc.authorize_target,
        sc._live_target_refusal,
        sc._managed_by_other,
        sc.manager_of,
        sc._created_by_other,
        sc._manager_unread_refusal,
    )
    reads: set[str] = set()
    for fn in gate:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "slot"
                and isinstance(node.args[1], ast.Constant)
            ):
                reads.add(node.args[1].value)
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "slot"
            ):
                reads.add(node.attr)
    assert reads, "the parse found nothing, so this test would pin nothing"

    snapshot_fields = set(_base_snapshot_fields())
    unaccounted = sorted(
        name
        for name in reads
        if name not in _IMMUTABLE_READS and _COVERED_BY.get(name) not in snapshot_fields
    )
    assert not unaccounted, (
        "the gate reads these target values and the drain does not re-compare them: "
        f"{unaccounted}. Either add the value to containment_snapshot and map it in "
        "_COVERED_BY, or add it to _IMMUTABLE_READS with the reason it cannot change "
        "between admission and delivery."
    )
    # And every mapping points at a field that really exists, so a renamed snapshot key
    # cannot leave a read silently 'covered' by a field that is gone.
    stale = sorted(f for f in _COVERED_BY.values() if f not in snapshot_fields)
    assert not stale, f"_COVERED_BY names snapshot fields that do not exist: {stale}"


def _base_snapshot_fields() -> set[str]:
    """Snapshot field names, read off the builder rather than a hand-kept copy."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(sc._snapshot_from_probe)))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    out.add(key.value)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                out.add(node.slice.value)
    return out


def test_created_by_is_only_ever_written_at_birth():
    """The premise that exempts `not_creator` from the drain re-check, pinned.

    `not_creator` is the one caller-relative refusal the drain does not re-evaluate, and
    the stated reason is that both sides are written once and never mutate. If a writer
    appears that stamps `_created_by` on a slot that is already open, a queued entry
    admitted under one creator can be delivered under another, and the exemption becomes
    a hole with nothing red.

    Every writer must therefore be a MINT or a RESTORE -- a slot object with no queue
    yet. Scanned from source, so a new writer has to come and argue with this list.
    """
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    #: Each writer, with why it cannot reach a slot that already has a queue.
    allowed = {
        "src/kiro_crew/dashboard/session_control.py": "create_session, at the mint",
        "src/kiro_crew/dashboard/chat_handlers.py": (
            "cron attribution, both sites guarded by created_in_send / is_new_slot"
        ),
        "src/kiro_crew/dashboard/slot_persistence/metadata_codec.py": (
            "restore, onto a freshly built slot object"
        ),
    }
    offenders: list[str] = []
    for path in sorted(repo_root.joinpath("src").rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^\s*\w+\._created_by\s*=\s*", text, re.M):
            if rel not in allowed:
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{rel}:{line}")
    assert not offenders, (
        "these write _created_by outside the mint and restore paths: "
        f"{offenders}. A slot that is already open can then change creator while an "
        "entry waits in its queue, which breaks the premise that exempts `not_creator` "
        "from the drain re-check. Either write it at the mint, or re-check it at the "
        "drain and move the refusal into _TARGET_CONTAINMENT_REFUSALS."
    )


def test_every_send_origin_stamp_records_a_resolved_fence_verdict():
    """The stamp's verdict must be the one authorization used, not a second derivation.

    ``caller_fenced`` arrives as ``bool | None``, and ``None`` means NOT RESOLVED YET --
    the HTTP handler's carried verdict is ``None`` for any caller that is not a crew
    member. ``bool(None)`` is ``False``, so coercing it at the stamp records an
    unresolved verdict as UNFENCED, which exempts that sender from the drain's manager
    re-check and lets the message it queued run after it has handed the session away.

    The verdict is therefore resolved ONCE per verb, before the gate, and both the gate
    and the stamp read that local. This parses every stamp site and fails on a coercion
    or on a site with no resolution above it in the same function, because the failure
    mode is silent: the stamp is well-formed and simply says the wrong thing.
    """
    import ast
    import inspect
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source = (repo_root / "src/kiro_crew/dashboard/session_control.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Parent links, because a stamp often sits in a CLOSURE over the verb's local --
    # `send_to_target` resolves the verdict once and its nested queue helpers read it.
    # A closure is the correct shape here, so the resolution is looked for in the
    # enclosing function too rather than only where the stamp is written.
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def _resolves_anywhere_up(node: ast.AST) -> bool:
        cur: ast.AST | None = node
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = ast.get_source_segment(source, cur) or ""
                if "caller_fenced is None" in body:
                    return True
            cur = parent.get(cur)
        return False

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not (
                isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "send_origin_meta"
            ):
                continue
            verdict = next((kw.value for kw in inner.keywords if kw.arg == "manager_fenced"), None)
            where = f"{node.name}:{inner.lineno}"
            if verdict is None:
                offenders.append(f"{where} (no manager_fenced argument)")
                continue
            if isinstance(verdict, ast.Call) and getattr(verdict.func, "id", "") == "bool":
                offenders.append(f"{where} (coerces the verdict with bool())")
                continue
            if not _resolves_anywhere_up(inner):
                offenders.append(f"{where} (no `caller_fenced is None` resolution above it)")

    assert not offenders, (
        "these send-origin stamps may record an unresolved fence verdict: "
        f"{offenders}. Resolve it once at the top of the verb -- "
        "`if caller_fenced is None: caller_fenced = bool(caller_key) and "
        "_caller_is_ownership_fenced(state, caller_key)` -- and pass that local to both "
        "authorize_target and the stamp."
    )
    # Guard the guard: the parse must actually be finding the stamps.
    stamp_sites = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "send_origin_meta"
    )
    assert stamp_sites >= 3, "the parse found almost no stamp sites, so this test would pin nothing"
    # And the parameter stays REQUIRED, so a new site cannot silently default to False.
    assert (
        inspect.signature(sc.send_origin_meta).parameters["manager_fenced"].default
        is inspect.Parameter.empty
    ), "manager_fenced must have no default: a forgotten stamp would read as unfenced"


# ── Fail-closed governs granting reach; it never governs keeping data ────────


def test_an_unreadable_manager_holds_a_queued_message_instead_of_destroying_it(
    tmp_path, _inline_audit
):
    """ "We cannot tell" is not a finding about the message.

    A queued entry is content somebody was already told had been accepted. An
    unreadable manager is not a statement that its authorization lapsed -- it is this
    gateway's own read not having landed. Dropping on it spends a person's work to
    protect a reach nobody can show was at issue, and the read is self-clearing, so
    waiting costs one pass.

    So the entry is HELD: not delivered, because delivering would grant a reach that
    cannot be verified, and not dropped, because nothing about it is known to be wrong.
    """
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key

    worker.queue_append(
        "finish the migration",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )
    assert worker._queue, "the premise"

    # The manager cannot be read. Nothing is known to have lapsed.
    worker._manager_unresolved = True
    held = cr._drop_stale_admissions(state, worker)

    assert held is True, "the drain reports the head as held"
    assert len(worker._queue) == 1, "the message is still there"
    notices = [m for m in worker.messages if m.get("role") == "notice"]
    assert not notices, "and the person is not told their message was discarded"
    assert not _inline_audit.log_tool_invocation.called, "nothing was denied"

    # Once the read lands, the same entry delivers.
    worker._manager_unresolved = False
    worker._manager = lead.key
    worker._manager_tab = lead._tab_id
    assert cr._drop_stale_admissions(state, worker) is False
    assert len(worker._queue) == 1, "still intact, now deliverable"


def test_a_sender_that_demonstrably_lost_its_reach_still_drops(tmp_path, _inline_audit):
    """The other half of the boundary, so holding does not become never-dropping.

    A handoff that LANDED is a finding about the message: the grant that admitted it
    ended, the sender is told, and holding it would deliver it the moment the manager
    came back. That is governing reach, which is what fail-closed is for.
    """
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    conductor = state.get_or_create_slot("chat-conductor")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key

    worker.queue_append(
        "finish the migration",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id

    assert cr._drop_stale_admissions(state, worker) is False, "dropped, not held"
    assert worker._queue == []
    notices = [m for m in worker.messages if m.get("role") == "notice"]
    assert notices and "no longer manages this one" in notices[-1]["content"]


def _settle_refusal(state, slot, shape: str) -> None:
    """Put *slot* in one of the two manager refusals that do NOT end by themselves."""
    if shape == "manager_grant_inadmissible":
        # The recorded manager was closed or rebound to another principal before a
        # restart, and no transfer back was ever recorded.
        slot._manager_grant_inadmissible = True
        return
    # The record's key is answered by a DIFFERENT session than the one it names.
    manager = state.get_or_create_slot("chat-conductor")
    slot._manager = manager.key
    slot._manager_tab = "the-session-that-was-given-it"


@pytest.mark.parametrize("shape", ["manager_grant_inadmissible", "manager_session_changed"])
def test_a_settled_manager_refusal_drops_instead_of_holding_the_queue_shut(
    tmp_path, _inline_audit, shape
):
    """A hold is a WAIT, so it is owed only where there is something to wait for.

    Three refusals arrive as one snapshot boolean. The read that has not landed arms
    its own re-read, so a hold on it costs a pass. These two do not: one says the
    recorded manager cannot hold the session, the other that a different session
    answers to its name, and both stand until a person hands the session over again.
    Waiting on them is waiting for the life of the process, with the queue shut behind
    the held head, and the entry is delivered to nobody in the end anyway.

    So a settled refusal DROPS, with the reason the refusal already carries and the
    sender told, which is the one outcome that leaves everybody informed.
    """
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key
    worker.queue_append(
        "finish the migration",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )
    _settle_refusal(state, worker, shape)
    assert sc.manager_read_is_pending(state, worker) is False, "the premise: settled"

    assert cr._drop_stale_admissions(state, worker) is False, "dropped, not held"
    assert worker._queue == [], "the queue is not left shut on an answer that never comes"
    notices = [m for m in worker.messages if m.get("role") == "notice"]
    assert notices, "and the message is not discarded silently"
    assert _inline_audit.log_tool_invocation.call_args.kwargs["outcome"] == "denied"


def test_a_settled_manager_refusal_does_not_bury_a_persons_own_typed_prompt(tmp_path):
    """What the unbounded hold costs, measured on the entry that has nothing to do
    with it.

    A held entry is marked so a merge run breaks at it, so everything BEHIND it waits
    too -- and a person's own typed prompt carries no sender stamp, is never re-checked
    against any manager, and is deliverable the whole time. Holding the fenced entry
    forever buries it.
    """
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key
    worker.queue_append(
        "the lead's, and settled against",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )
    worker.queue_append("mine, typed here", meta=sc.containment_meta(state, worker))
    _settle_refusal(state, worker, "manager_session_changed")

    assert cr._drop_stale_admissions(state, worker) is False
    assert [entry["content"] for entry in worker._queue] == ["mine, typed here"]
    assert "_held_for_manager" not in worker._queue[0]

    from kiro_crew.dashboard.chat_utils import _dequeue_next_message

    content, taken = _dequeue_next_message(worker, True)
    assert content == "mine, typed here", "the person's own prompt runs"
    assert len(taken) == 1


def test_a_merge_run_does_not_carry_a_held_entry_out(tmp_path):
    """A merge is a DELIVERY, so it must break at a held entry wherever it sits.

    Refusing only the HEAD is not enough. ``_dequeue_next_message`` folds a RUN of
    entries into one turn, and it breaks only at a system injection, an attachment, a
    possibly-delivered steer and an actor change. A held entry behind a deliverable one
    would be merged into the batch and run -- carried straight past the hold the sweep
    just decided.

    So the hold is marked per entry and the merge treats it as a stop, the same way it
    treats the other four.
    """
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key

    # A HUMAN-typed entry first, which is always deliverable, then a fenced
    # cross-session entry whose manager cannot be read.
    worker.queue_append("mine, and deliverable", meta=sc.containment_meta(state, worker))
    worker.queue_append(
        "the lead's, and not resolvable",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )
    worker._manager_unresolved = True

    assert cr._drop_stale_admissions(state, worker) is False, "the head is deliverable"
    assert len(worker._queue) == 2, "and nothing was dropped"
    assert worker._queue[1].get("_held_for_manager") is True, "the second is marked held"

    from kiro_crew.dashboard.chat_utils import _dequeue_next_message

    content, taken = _dequeue_next_message(worker, True)
    assert len(taken) == 1, "the merge stopped at the held entry"
    assert content == "mine, and deliverable"
    assert len(worker._queue) == 1, "the held entry is still queued"
    assert worker._queue[0].get("_held_for_manager") is True


def test_the_hold_mark_is_cleared_once_the_read_lands(tmp_path):
    """A hold must not outlive its reason, or the entry is stuck behind a stale mark."""
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key
    worker.queue_append(
        "the lead's",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )

    worker._manager_unresolved = True
    assert cr._drop_stale_admissions(state, worker) is True
    assert worker._queue[0].get("_held_for_manager") is True

    # The read lands and the lead is the manager, so the entry is deliverable again.
    worker._manager_unresolved = False
    worker._manager = lead.key
    worker._manager_tab = lead._tab_id
    assert cr._drop_stale_admissions(state, worker) is False
    assert "_held_for_manager" not in worker._queue[0], "the mark is gone with its reason"


# ── Provenance survives what routing does not ───────────────────────────────


def test_a_revoked_senders_message_does_not_survive_a_restart(tmp_path):
    """The restore strips the ROUTING stamp; it must not strip the re-check with it.

    The routing stamp names a write target, so the restore path drops it rather than
    trust an editable line. When the fence verdict and the sender rode in that same
    stamp, a restart turned a fenced cross-session entry into one with no provenance
    at all -- the drain read it as unfenced, skipped the sender re-check, and the
    message a revoked lead had queued ran after the restart.
    """
    from kiro_crew.dashboard.slot_queue_repository import sanitize_restored_queue

    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    conductor = state.get_or_create_slot("chat-conductor")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key
    worker.queue_append(
        "finish the migration",
        meta={
            **sc.containment_meta(state, worker),
            **sc.send_origin_meta(state, lead.key, manager_fenced=True),
        },
    )

    # The queue as a restart rebuilds it from the slot's own line.
    restored = sanitize_restored_queue(list(worker._queue))
    assert restored, "the premise: the entry is restored"
    meta = restored[0]["meta"]
    assert sc.SEND_ORIGIN_META_KEY not in meta, "routing is stripped, as designed"
    # The sender's IDENTITY survives the restore with its key, which is the reason the
    # two stamps are separate keys: routing is stripped and the re-check is not.
    assert meta[sc.ADMISSION_PROVENANCE_META_KEY] == {
        "sender": lead.key,
        "fenced": True,
        "tab": lead._tab_id,
        # The policy that admitted it travels with it: the drain re-asserts
        # that one and reads no config of its own.
        "chain": False,
    }

    # The worker was handed away before the restart; the restored entry is re-checked.
    worker._queue[:] = restored
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.sender_lost_its_reach(state, worker, meta) is True
    cr._drop_stale_admissions(state, worker)
    assert worker._queue == [], "the revoked sender's message does not run"


def test_a_tabless_senders_message_is_still_re_checked(tmp_path):
    """No tab means nobody to REPORT to. It does not mean nothing to re-check."""
    state = _make_state(tmp_path)
    lead = state.get_or_create_slot("chat-lead")
    lead._tab_id = ""
    conductor = state.get_or_create_slot("chat-conductor")
    worker = _busy(state.get_or_create_slot("chat-worker"))
    worker._created_by = lead.key

    meta = {
        **sc.containment_meta(state, worker),
        **sc.send_origin_meta(state, lead.key, manager_fenced=True),
    }
    assert sc.SEND_ORIGIN_META_KEY not in meta, "no routing without an identity"
    worker._manager = conductor.key
    worker._manager_tab = conductor._tab_id
    assert sc.sender_lost_its_reach(state, worker, meta) is True
