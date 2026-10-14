"""Route tests: creating a spec.

Create refuses existing documents and an opt-in flag that is not a real boolean,
dispatches a self-contained, type-aware seed prompt, and cleans up orphaned
execution loops and turns before any filesystem side effect. A create that fails
unwinds only what it made: an abort spares a replacement spec, its worktree and
its slot; a failed archive restores the original name; and a name reserved by an
in-flight delete holds.
"""

from __future__ import annotations

import contextlib
import inspect
import types
from pathlib import Path

import pytest

from kiro_crew.apps.builtins.spec_builder.tests.routes_facade import routes

# _never_touch_the_real_state is an autouse fixture: pytest finds it by the name bound here.
from kiro_crew.apps.builtins.spec_builder.tests.routes_helpers import (  # noqa: F401
    _BASE,
    _make_client,
    _never_touch_the_real_state,
    _redirect_state,
)


def _maintenance_loader(service):
    """Build the AutoNudge class seam around one authoritative test service."""

    class _Loader:
        @classmethod
        @contextlib.asynccontextmanager
        async def maintenance_service(cls, base_dir=None):
            yield service

    return _Loader


@pytest.mark.asyncio
async def test_create_cleanup_removes_a_fully_orphaned_execution_loop(tmp_path, monkeypatch):
    """An empty index has an app-owned recovery path for a durable orphan."""
    _redirect_state(monkeypatch, tmp_path)
    orphan_slot = "spec-builder-s-1234abcd"
    loop = types.SimpleNamespace(
        id="orphan-loop",
        active=True,
        slot_key=orphan_slot,
        message=f"{routes._EXECUTION_HANDOFF_PREFIX}s'. Continue.",
        stop_sentinel_path=str(tmp_path / "old-spec" / "STOP"),
    )
    removed: list[str] = []
    torn_down: list[object] = []
    orphan_slot_state = types.SimpleNamespace(
        key=orphan_slot,
        _app=routes.APP_NAME,
        running=True,
    )

    class _State:
        def __init__(self):
            self.slot: object | None = orphan_slot_state

        def get_slot(self, key):
            return self.slot if key == orphan_slot else None

    state = _State()

    class _Svc:
        def list_all(self):
            return [] if removed else [loop]

        def get_by_slot(self, key):
            return loop if key == orphan_slot and not removed else None

        async def deactivate_and_wait(self, loop_id, *, stopped_reason=None):
            assert loop_id == loop.id
            stop_reasons.append(stopped_reason)
            loop.active = False
            return True

        async def remove(self, loop_id, *, stop_reason=""):
            removed.append(loop_id)
            loop.active = False

    async def _teardown(_state, _name, *, only_slot, require_archive):
        assert require_archive is True
        torn_down.append(only_slot)
        state.slot = None
        return True

    stop_reasons: list[str | None] = []
    service = _Svc()
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: service)
    monkeypatch.setattr(routes, "_AutoNudgeService", _maintenance_loader(service))
    monkeypatch.setattr(routes, "_teardown_worker_slot", _teardown)
    routes._save_index({})
    assert routes._reserve_pending_dispatch("/new/spec", "spec-builder-n-deadbeef", "n") == ""

    assert await routes._remove_orphaned_executions(state) == {orphan_slot}

    assert removed == [loop.id]
    assert stop_reasons == ["orphaned_worker"]
    assert torn_down == [orphan_slot_state]
    token = routes._reserve_pending_dispatch("/new/spec", "spec-builder-n-deadbeef", "n")
    assert token
    routes._drop_pending_dispatch(token)


@pytest.mark.asyncio
async def test_create_cleanup_rechecks_a_slot_materialized_while_its_loop_quiesces(
    tmp_path, monkeypatch
):
    """A firing timer cannot publish a worker after cleanup's slot snapshot."""
    _redirect_state(monkeypatch, tmp_path)
    orphan_slot = "spec-builder-s-1234abcd"
    loop = types.SimpleNamespace(
        id="orphan-loop",
        active=True,
        slot_key=orphan_slot,
        message=f"{routes._EXECUTION_HANDOFF_PREFIX}s'. Continue.",
        stop_sentinel_path=str(tmp_path / "old-spec" / "STOP"),
    )
    published = types.SimpleNamespace(
        key=orphan_slot,
        _app=routes.APP_NAME,
        running=True,
    )
    torn_down: list[object] = []

    class _State:
        slot = None

        def get_slot(self, key):
            return self.slot if key == orphan_slot else None

    state = _State()

    class _Svc:
        def list_all(self):
            return [loop]

        def get_by_slot(self, key):
            return loop if key == orphan_slot else None

        async def deactivate_and_wait(self, loop_id, *, stopped_reason=None):
            assert loop_id == loop.id
            loop.active = False
            state.slot = published
            return True

        async def remove(self, loop_id, *, stop_reason=""):
            assert loop_id == loop.id

    async def _teardown(_state, _name, *, only_slot, require_archive):
        assert require_archive is True
        torn_down.append(only_slot)
        state.slot = None
        return True

    service = _Svc()
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: service)
    monkeypatch.setattr(routes, "_AutoNudgeService", _maintenance_loader(service))
    monkeypatch.setattr(routes, "_teardown_worker_slot", _teardown)
    routes._save_index({})

    assert await routes._remove_orphaned_executions(state) == {orphan_slot}
    assert torn_down == [published]


@pytest.mark.asyncio
async def test_create_cleanup_keeps_a_quiesced_loop_until_worker_archive_succeeds(
    tmp_path, monkeypatch
):
    """A retry or restart retains the identity when cancellation times out."""
    _redirect_state(monkeypatch, tmp_path)
    orphan_slot = "spec-builder-s-1234abcd"
    loop = types.SimpleNamespace(
        id="orphan-loop",
        active=True,
        slot_key=orphan_slot,
        message=f"{routes._EXECUTION_HANDOFF_PREFIX}s'. Continue.",
        stop_sentinel_path=str(tmp_path / "old-spec" / "STOP"),
    )
    slot = types.SimpleNamespace(key=orphan_slot, _app=routes.APP_NAME, running=True)
    removed: list[str] = []

    class _State:
        def get_slot(self, key):
            return slot if key == orphan_slot else None

    class _Svc:
        def list_all(self):
            return [loop]

        def get_by_slot(self, key):
            return loop if key == orphan_slot else None

        async def deactivate_and_wait(self, loop_id, *, stopped_reason=None):
            assert loop_id == loop.id
            loop.active = False
            return True

        async def remove(self, loop_id, *, stop_reason=""):
            removed.append(loop_id)

    async def _failed_archive(*_args, **_kwargs):
        return False

    service = _Svc()
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: service)
    monkeypatch.setattr(routes, "_AutoNudgeService", _maintenance_loader(service))
    monkeypatch.setattr(routes, "_teardown_worker_slot", _failed_archive)
    routes._save_index({})

    with pytest.raises(RuntimeError, match="could not be archived"):
        await routes._remove_orphaned_executions(_State())

    assert loop.active is False
    assert removed == []
    assert routes._matching_execution_loops("", "", set(), include_orphans=True) == {
        orphan_slot: loop.id
    }


@pytest.mark.asyncio
async def test_create_cleanup_loads_durable_loops_when_autonudge_is_disabled(tmp_path, monkeypatch):
    """Disabling timers cannot hide an old loop from Create recovery."""
    _redirect_state(monkeypatch, tmp_path)
    orphan_slot = "spec-builder-s-1234abcd"
    loop = types.SimpleNamespace(
        id="orphan-loop",
        active=True,
        slot_key=orphan_slot,
        message=f"{routes._EXECUTION_HANDOFF_PREFIX}s'. Continue.",
        stop_sentinel_path=str(tmp_path / "old-spec" / "STOP"),
    )
    removed: list[str] = []

    class _OfflineSvc:
        def list_all(self):
            return [loop]

        def get_by_slot(self, key):
            return loop if key == orphan_slot else None

        async def deactivate_and_wait(self, loop_id, *, stopped_reason=None):
            assert loop_id == loop.id
            loop.active = False
            return True

        async def remove(self, loop_id, *, stop_reason=""):
            removed.append(loop_id)

    offline = _OfflineSvc()

    class _Loader:
        @classmethod
        @contextlib.asynccontextmanager
        async def maintenance_service(cls, base_dir=None):
            yield offline

    class _State:
        def get_slot(self, _key):
            return None

    monkeypatch.setattr(routes, "_autonudge_instance", lambda: None)
    monkeypatch.setattr(routes, "_AutoNudgeService", _Loader)
    routes._save_index({})

    assert await routes._remove_orphaned_executions(_State()) == {orphan_slot}
    assert removed == [loop.id]


def test_create_removes_orphaned_loops_before_filesystem_side_effects():
    """Recovery precedes worktree and spec-directory creation."""
    src = inspect.getsource(routes._handle_create)
    cleanup_at = src.index("await _remove_orphaned_executions(")
    worktree_at = src.index("await _create_worktree(")
    spec_dir_at = src.index("_prepare_spec_dir")

    assert cleanup_at < worktree_at < spec_dir_at


@pytest.mark.asyncio
async def test_create_refuses_an_unreadable_index_before_orphan_cleanup(tmp_path, monkeypatch):
    """Invalid index bytes cannot make durable workers look orphaned."""
    client = _make_client(monkeypatch, tmp_path)
    index_path = routes._index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("not-json")
    cleaned: list[bool] = []
    prepared: list[bool] = []

    async def _cleanup(_state):
        cleaned.append(True)
        return set()

    def _prepare(*_args, **_kwargs):
        prepared.append(True)
        return tmp_path / "spec", ""

    monkeypatch.setattr(routes, "_remove_orphaned_executions", _cleanup)
    monkeypatch.setattr(routes, "_prepare_spec_dir", _prepare)
    await client.start_server()
    try:
        response = await client.post(
            f"{_BASE}/specs",
            json={"name": "new", "working_dir": str(tmp_path), "spec_type": "quick"},
        )
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 503, payload
    assert payload.get("code") == "spec_index_unavailable"
    assert cleaned == []
    assert prepared == []
    assert index_path.read_text() == "not-json"


@pytest.mark.asyncio
async def test_create_refuses_before_side_effects_when_orphan_cleanup_fails(tmp_path, monkeypatch):
    """An unremovable orphan cannot overlap a partially-created replacement."""
    client = _make_client(monkeypatch, tmp_path)
    prepared: list[str] = []

    async def _fail_cleanup(_state):
        raise OSError("autonudge store unavailable")

    def _prepare(*_args, **_kwargs):
        prepared.append("prepared")
        return tmp_path / "spec", None

    monkeypatch.setattr(routes, "_remove_orphaned_executions", _fail_cleanup)
    monkeypatch.setattr(routes, "_prepare_spec_dir", _prepare)
    await client.start_server()
    try:
        response = await client.post(
            f"{_BASE}/specs",
            json={"name": "new", "working_dir": str(tmp_path), "spec_type": "quick"},
        )
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 503, payload
    assert payload.get("code") == "orphaned_execution_cleanup_failed"
    assert prepared == []


@pytest.mark.asyncio
async def test_create_cleanup_archives_an_orphaned_direct_turn_without_a_loop(
    tmp_path, monkeypatch
):
    """Removing the last index row cannot detach an already-running worker."""
    _redirect_state(monkeypatch, tmp_path)
    old_slot = "spec-builder-s-1234abcd"
    old_dir = str(tmp_path / "old-spec")
    routes._save_index({"s": {"spec_dir": old_dir, "slot_key": old_slot}})
    routes._save_index({})
    slot = types.SimpleNamespace(key=old_slot, _app=routes.APP_NAME, running=True)
    torn_down: list[object] = []

    class _Svc:
        def list_all(self):
            return []

    class _State:
        def __init__(self):
            self.slot: object | None = slot

        def get_slot(self, key):
            return self.slot if key == old_slot else None

    state = _State()

    async def _teardown(_state, _name, *, only_slot, require_archive):
        assert require_archive is True
        torn_down.append(only_slot)
        state.slot = None
        return True

    service = _Svc()
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: service)
    monkeypatch.setattr(routes, "_AutoNudgeService", _maintenance_loader(service))
    monkeypatch.setattr(routes, "_teardown_worker_slot", _teardown)

    assert await routes._remove_orphaned_executions(state) == {old_slot}
    assert torn_down == [slot]
    assert old_slot not in routes._OBSERVED_SLOT_KEYS.values()
    assert old_slot not in routes._OBSERVED_SPEC_DIRS


@pytest.mark.asyncio
async def test_create_cleanup_keeps_the_witness_when_orphan_archive_fails(tmp_path, monkeypatch):
    """A failed transcript archive remains retryable and blocks replacement."""
    _redirect_state(monkeypatch, tmp_path)
    old_slot = "spec-builder-s-1234abcd"
    old_dir = str(tmp_path / "old-spec")
    routes._save_index({"s": {"spec_dir": old_dir, "slot_key": old_slot}})
    routes._save_index({})
    slot = types.SimpleNamespace(key=old_slot, _app=routes.APP_NAME, running=True)

    class _Svc:
        def list_all(self):
            return []

    class _State:
        def get_slot(self, key):
            return slot if key == old_slot else None

    async def _failed_archive(*_args, **_kwargs):
        return False

    service = _Svc()
    monkeypatch.setattr(routes, "_autonudge_instance", lambda: service)
    monkeypatch.setattr(routes, "_AutoNudgeService", _maintenance_loader(service))
    monkeypatch.setattr(routes, "_teardown_worker_slot", _failed_archive)

    with pytest.raises(RuntimeError, match="could not be archived"):
        await routes._remove_orphaned_executions(_State())

    assert routes._OBSERVED_SLOT_KEYS["s"] == old_slot
    assert routes._OBSERVED_SPEC_DIRS[old_slot] == routes._decision_key(old_dir)


def test_global_orphan_scan_ignores_an_unrelated_monitor_without_a_sentinel(tmp_path, monkeypatch):
    """An empty target cannot turn two empty directory keys into a direct match."""
    _redirect_state(monkeypatch, tmp_path)
    loop = types.SimpleNamespace(
        id="ordinary-loop",
        active=True,
        slot_key="spec-builder-ordinary-1234abcd",
        message=f"{routes._EXECUTION_HANDOFF_PREFIX}ordinary'. Continue.",
        stop_sentinel_path="",
    )

    class _Svc:
        def list_all(self):
            return [loop]

    monkeypatch.setattr(routes, "_autonudge_instance", lambda: _Svc())
    routes._save_index({})

    assert routes._matching_execution_loops("", "", set(), include_orphans=True) == {}


@pytest.mark.parametrize("raw_slot", [None, "not-owned"])
def test_restart_keeps_a_bad_slot_row_from_orphaning_its_durable_loop(
    tmp_path, monkeypatch, raw_slot
):
    """Cold-start name and directory witnesses do not depend on a valid raw key."""
    _redirect_state(monkeypatch, tmp_path)
    old_dir = str(tmp_path / "old-spec")
    old_slot = "spec-builder-s-1234abcd"
    loop = types.SimpleNamespace(
        id="loop-s",
        active=True,
        slot_key=old_slot,
        message=f"{routes._EXECUTION_HANDOFF_PREFIX}s'. Continue.",
        stop_sentinel_path=str(Path(old_dir) / "STOP"),
    )

    class _Svc:
        def list_all(self):
            return [loop]

    monkeypatch.setattr(routes, "_autonudge_instance", lambda: _Svc())
    routes._save_index({"s": {"spec_dir": old_dir, "slot_key": raw_slot}})

    assert routes._OBSERVED_SLOT_KEYS == {}
    assert routes._SLOT_KEYS == {}
    assert routes._matching_execution_loops("", "", set(), include_orphans=True) == {}


def test_the_refusal_unwinds_a_worktree_it_created():
    """Source guard: the 503 must not leak the worktree create just made.

    Every other refusal in this handler removes it; this one is reached after the
    worktree exists, so omitting it would leave an orphaned checkout behind.
    """
    src = inspect.getsource(routes._handle_create)
    at = src.index("decision_record_not_cleared")
    # Look back from the refusal payload to the branch it belongs to.
    branch = src[src.index("if not cleared", 0) : at]
    assert (
        "_remove_worktree(" in branch
    ), "the stale-record refusal returns without removing a worktree it created"
