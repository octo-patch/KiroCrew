"""Route tests: execute, handoff, pause and stop.

Handoff authorizes and arms the build loop before it dispatches, refuses a second
execute while one is building, and unwinds every armed effect when authorization,
the index commit or a concurrent delete fails. Pause and stop halt this app's own
turn only; execute, handoff and stop are owner-gated, before any side effect; and
the app never grants worker trust.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.apps.builtins.spec_builder.tests.routes_facade import routes

# _never_touch_the_real_state is an autouse fixture: pytest finds it by the name bound here.
from kiro_crew.apps.builtins.spec_builder.tests.routes_helpers import (  # noqa: F401
    _BASE,
    _decision_spec,
    _make_client,
    _never_touch_the_real_state,
    _NoopState,
    _RecordingSlot,
    _redirect_state,
    _slot_stub,
)


def test_queued_dispatch_preserves_structural_directive_provenance():
    slot = _RecordingSlot()

    routes._dispatch_turn(_NoopState(), slot, "automated seed")
    routes._dispatch_turn(
        _NoopState(),
        slot,
        "human answer",
        directive_user_origin=True,
    )

    assert slot.queued_origins == [False, True]


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (routes._handle_create, False),
        (routes._handle_handoff, False),
        (routes._handle_message, True),
        (routes._deliver_pending_decision, True),
    ],
)
def test_spec_dispatch_callers_assign_directive_provenance(handler, expected):
    """Only direct human input may authorize session-retargeting directives."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_dispatch_turn"
    ]
    assert len(calls) == 1
    keyword = next(
        (kw for kw in calls[0].keywords if kw.arg == "directive_user_origin"),
        None,
    )
    if keyword is None:
        actual = False
    else:
        assert isinstance(keyword.value, ast.Constant)
        actual = keyword.value.value
    assert actual is expected


@pytest.mark.asyncio
async def test_handoff_refuses_when_its_own_slot_starts_during_authorization(tmp_path, monkeypatch):
    """Authorization awaits while channel traffic can start the same slot.

    The pre-arm snapshot is not authoritative after that await. Dispatching the
    build would queue it behind the new turn, and Pause deliberately clears that queue,
    despite this endpoint reporting that execution started.
    """
    client = _make_client(monkeypatch, tmp_path)
    spec_dir, _slot_key = _decision_spec(tmp_path, state={"phase": "tasks"})
    (Path(spec_dir) / "tasks.md").write_text("- [ ] a task\n")
    state, slot = _slot_stub()
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: object())

    async def _authorized(**_kw):
        slot.running = True
        return types.SimpleNamespace(id="loop-raced"), "", 200

    monkeypatch.setattr(routes, "authorize_and_add_nudge", _authorized)
    dispatched: list[str] = []
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *a, **k: dispatched.append(a[2]))
    released: list[str] = []

    async def _remove(_slot_key, *, only_loop_id=None, stop_reason=""):
        released.append(only_loop_id)

    monkeypatch.setattr(routes, "_remove_nudge_loop_for_slot", _remove)

    client.app["state"] = state
    await client.start_server()
    try:
        resp = await client.post(f"{_BASE}/specs/s/handoff", json={"spec_dir": spec_dir})
        payload = await resp.json()
    finally:
        await client.close()

    assert resp.status == 409, payload
    assert payload.get("code") == "spec_agent_busy", payload
    assert dispatched == []
    assert released == ["loop-raced"], "the armed timer survived the busy refusal"
    assert routes._load_index()["s"].get("status") == "planning"


@pytest.mark.asyncio
async def test_handoff_refuses_when_its_own_slot_starts_during_the_final_repin(
    tmp_path, monkeypatch
):
    """The final freshness write awaits, so channel traffic can start there too."""
    client = _make_client(monkeypatch, tmp_path)
    spec_dir, _slot_key = _decision_spec(tmp_path, state={"phase": "tasks"})
    (Path(spec_dir) / "tasks.md").write_text("- [ ] a task\n")
    state, slot = _slot_stub()
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: object())

    async def _authorized(**_kw):
        return types.SimpleNamespace(id="loop-repin-raced"), "", 200

    monkeypatch.setattr(routes, "authorize_and_add_nudge", _authorized)
    real_touch = routes._touch_spec

    async def _touch_then_start(*args, **kwargs):
        result = await real_touch(*args, **kwargs)
        if kwargs.get("exec_arming_at") == 0.0 and kwargs.get("status") is None:
            slot.running = True
        return result

    monkeypatch.setattr(routes, "_touch_spec", _touch_then_start)
    dispatched: list[str] = []
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *a, **k: dispatched.append(a[2]))
    released: list[str] = []

    async def _remove(_slot_key, *, only_loop_id=None, stop_reason=""):
        released.append(only_loop_id)

    monkeypatch.setattr(routes, "_remove_nudge_loop_for_slot", _remove)

    client.app["state"] = state
    await client.start_server()
    try:
        resp = await client.post(f"{_BASE}/specs/s/handoff", json={"spec_dir": spec_dir})
        payload = await resp.json()
    finally:
        await client.close()

    assert resp.status == 409, payload
    assert payload.get("code") == "spec_agent_busy", payload
    assert dispatched == []
    assert released == ["loop-repin-raced"], "the armed timer survived the busy refusal"
    assert routes._load_index()["s"].get("status") == "planning"


def test_the_busy_refusal_cannot_leak_an_armed_loop():
    """Source guard: the busy check must precede the arm.

    Arming after the check removes the leak outright, where releasing the timer inside
    the refusal only patches it. Ordering is the property worth pinning -- an edit that
    moves arming above the check reintroduces a hazard that only shows up as a build
    firing minutes after a 409.
    """
    src = inspect.getsource(routes._handle_handoff)
    lock = src.index("async with _turn_lock(handoff_dir_key):")
    busy = src.index("_busy_alias(", lock)
    arm = src.index("await authorize_and_add_nudge(", lock)
    assert busy < arm, (
        "the handoff arms its nudge loop before checking for a busy alias, so a "
        "refusal leaves a timer that dispatches the build it just denied"
    )
    assert lock < arm, (
        "the loop is armed outside the directory turn lock, so its idle timer can "
        "dispatch while this handler is still waiting for the lock"
    )


# ── owner gate on execute / handoff / stop ───────────────────────────────────


_GATE_OWNER = "owner-subject"


_GATE_NON_OWNER = "allowlisted-messaging-user"


class _ConfiguredOwnerState:
    """A state with a configured owner, so a different subject is a non-owner."""

    owner_id = _GATE_OWNER

    def get_or_create_slot(self, name, app=""):
        raise AssertionError("an owner-gated route acquired a worker slot")

    def get_slot(self, key):
        return None


def _claims_mw(user: str, app_claim: str):
    @web.middleware
    async def _mw(request, handler):
        request["user"] = user
        request["app"] = app_claim
        return await handler(request)

    return _mw


def _gated_client(monkeypatch, tmp_path, user: str, app_claim: str) -> TestClient:
    _redirect_state(monkeypatch, tmp_path)
    app = web.Application(middlewares=[_claims_mw(user, app_claim)])
    routes.register_routes(app)
    app["state"] = _ConfiguredOwnerState()
    return TestClient(TestServer(app))


def _gated_spec(tmp_path) -> Path:
    spec_dir = tmp_path / "wd" / ".kiro" / "specs" / "s"
    spec_dir.mkdir(parents=True)
    (spec_dir / "tasks.md").write_text("- [ ] task")
    routes._save_index(
        {
            "s": {
                "spec_dir": str(spec_dir),
                "working_dir": str(tmp_path / "wd"),
                "slot_key": "spec-builder-s",
            }
        }
    )
    return spec_dir


def _record_side_effects(monkeypatch) -> list[str]:
    effects: list[str] = []

    def _sync(label):
        def _spy(*_a, **_k):
            effects.append(label)
            return (False, "")

        return _spy

    def _async(label):
        async def _spy(*_a, **_k):
            effects.append(label)
            return None, "recorded by test", 403

        return _spy

    monkeypatch.setattr(routes, "_prepare_handoff", _sync("prepare_handoff"))
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: object())
    monkeypatch.setattr(routes, "authorize_and_add_nudge", _async("arm"))
    monkeypatch.setattr(routes, "_dispatch_turn", _sync("dispatch"))
    monkeypatch.setattr(routes, "_halt_execution", _async("halt"))
    monkeypatch.setattr(routes, "_remove_nudge_loop_for_slot", _async("remove_loop"))
    monkeypatch.setattr(routes, "_halt_active_turn", _async("halt_turn"))
    return effects


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["execute", "handoff"])
async def test_non_owner_dashboard_subject_cannot_execute_or_handoff(verb, tmp_path, monkeypatch):
    """An allow-listed non-owner holds a dashboard credential with an empty app
    claim. Arming the run's loop makes its message the owner session's next turn,
    so it gets the owner gate's 403 before anything -- including the STOP-sentinel
    clear in _prepare_handoff -- runs."""
    client = _gated_client(monkeypatch, tmp_path, _GATE_NON_OWNER, "")
    spec_dir = _gated_spec(tmp_path)
    effects = _record_side_effects(monkeypatch)

    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/s/{verb}",
            json={"spec_dir": str(spec_dir), "slot_key": "spec-builder-s"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert (resp.status, body.get("code")) == (403, "owner_only"), (resp.status, body)
    assert effects == [], f"a non-owner {verb} reached {effects}"


@pytest.mark.asyncio
async def test_non_owner_dashboard_subject_cannot_stop_spec_execution(tmp_path, monkeypatch):
    """Stop removes the owner session's loop, the partner of arming it, so a
    non-owner is refused before any halt, loop removal or STOP write."""
    client = _gated_client(monkeypatch, tmp_path, _GATE_NON_OWNER, "")
    spec_dir = _gated_spec(tmp_path)
    effects = _record_side_effects(monkeypatch)

    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/s/stop",
            json={"spec_dir": str(spec_dir), "slot_key": "spec-builder-s"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert (resp.status, body.get("code")) == (403, "owner_only"), (resp.status, body)
    assert effects == [], f"a non-owner stop reached {effects}"


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["execute", "handoff", "stop"])
async def test_spec_app_token_cannot_execute_handoff_or_stop(verb, tmp_path, monkeypatch):
    """The same refusal POST /api/autonudge gives an app token: even this app's
    own token carries the owner's subject but is not the owner's dashboard."""
    client = _gated_client(monkeypatch, tmp_path, _GATE_OWNER, "spec-builder")
    spec_dir = _gated_spec(tmp_path)
    effects = _record_side_effects(monkeypatch)

    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/s/{verb}",
            json={"spec_dir": str(spec_dir), "slot_key": "spec-builder-s"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert (resp.status, body.get("code")) == (403, "owner_only"), (resp.status, body)
    assert effects == [], f"an app token {verb} reached {effects}"


@pytest.mark.asyncio
async def test_non_owner_keeps_the_ungated_spec_routes(tmp_path, monkeypatch):
    """Only execute, handoff and stop take the owner gate; listing specs keeps
    its current callers."""
    client = _gated_client(monkeypatch, tmp_path, _GATE_NON_OWNER, "")
    _gated_spec(tmp_path)

    await client.start_server()
    try:
        resp = await client.get(f"{_BASE}/specs")
    finally:
        await client.close()

    assert resp.status == 200, resp.status
