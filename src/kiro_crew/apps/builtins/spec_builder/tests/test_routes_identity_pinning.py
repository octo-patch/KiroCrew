"""Route tests: a request acts on the spec creation it names.

The client sends ``spec_dir`` and ``slot_key``, and every handler that re-reads
the index pins both, so a spec deleted and recreated at the same name or path
mid-request is refused instead of acted on. Covers per-creation slot keys, their
grammar and ownership, and observed slot identities that survive an index rewrite.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading

import pytest

from kiro_crew.apps.builtins.spec_builder.tests.routes_facade import routes

# _never_touch_the_real_state is an autouse fixture: pytest finds it by the name bound here.
from kiro_crew.apps.builtins.spec_builder.tests.routes_helpers import (  # noqa: F401
    _BASE,
    _decision_spec,
    _make_client,
    _never_touch_the_real_state,
    _redirect_state,
    _slot_stub,
)


@pytest.mark.asyncio
async def test_index_and_legacy_runtime_identity_are_captured_under_one_lock(tmp_path, monkeypatch):
    """A replacement writer cannot refresh the resolver between the two values."""
    _redirect_state(monkeypatch, tmp_path)
    spec_dir = str(tmp_path / "wd" / ".kiro" / "specs" / "legacy")
    routes._save_index({"legacy": {"spec_dir": spec_dir}})
    snapshot_read = threading.Event()
    release_snapshot = threading.Event()
    real_load = routes._load_index
    load_count = 0

    def _load_and_pause_once():
        nonlocal load_count
        index = real_load()
        load_count += 1
        if load_count == 1:
            snapshot_read.set()
            assert release_snapshot.wait(timeout=5)
        return index

    monkeypatch.setattr(routes, "_load_index", _load_and_pause_once)
    capture = asyncio.create_task(routes._aload_index_with_slot_identity("legacy"))
    assert await asyncio.to_thread(snapshot_read.wait, 5)
    replacement_key = routes._new_slot_key("legacy")

    def _replace(index):
        index["legacy"]["slot_key"] = replacement_key
        return True

    replacement = asyncio.create_task(routes._mutate_index(_replace))
    await asyncio.sleep(0)
    release_snapshot.set()

    index, runtime_key, observed_key = await capture
    assert await replacement
    assert "slot_key" not in index["legacy"]
    assert runtime_key == "spec-builder-legacy"
    assert observed_key == ""
    assert routes._load_index()["legacy"]["slot_key"] == replacement_key


def test_the_detail_read_scopes_the_slot_from_a_fresh_index_read():
    """Source guard: nothing may await between the detail handler's FRESH index read
    and the ``_ensure_worker_slot`` call that consumes its ``meta``.

    A delete-and-re-import in that window hands the replacement's slot the stale
    entry, and the agent's next turn then runs in the OLD project directory. Reading
    this backend's decision ledger separately opened exactly that window; the read
    belongs in the one filesystem hop above instead."""
    src = inspect.getsource(routes._handle_get)
    fresh_call = "await _aload_index_with_decision_alias_status("
    fresh = src.rindex(fresh_call)
    # The CALL, not the comment that names the helper a few lines above it.
    scope = src.index("await _ensure_worker_slot(")
    assert fresh < scope, "the slot is scoped before the index is re-read"
    between = src[fresh + len(fresh_call) : scope]
    awaits = [
        ln.strip()
        for ln in between.splitlines()
        if "await " in ln and not ln.strip().startswith("#")
    ]
    assert (
        not awaits
    ), f"an await sits between the fresh index read and the slot scoping: {awaits!r}"


@pytest.mark.asyncio
async def test_a_same_path_reimport_during_the_read_is_refused(tmp_path, monkeypatch):
    """A delete leaves the documents on disk, so a re-import at the same name AND path
    is a DIFFERENT creation. A spec_dir-only freshness check passed it, pairing the
    replacement's metadata with documents and a decision record read for the spec that
    is gone -- serving the deleted spec's locked answers on the new one."""
    client = _make_client(monkeypatch, tmp_path)
    spec_dir, slot_key = _decision_spec(tmp_path)
    assert (
        await routes._claim_decision(
            "s", "transport", "HTTPS", expect_spec_dir=spec_dir, expect_slot_key=slot_key
        )
    )[0] == routes._CLAIM_RECORDED

    real_collect = routes._collect_spec_documents

    def _collect_then_reimport(sd):
        # Same name, same path, NEW creation -- exactly what a delete + re-import
        # leaves behind.
        idx = routes._load_index()
        routes._forget_observed_slot_identity("s", str(idx["s"].get("slot_key", "")))
        idx["s"]["slot_key"] = routes._new_slot_key("s")
        routes._save_index(idx)
        return real_collect(sd)

    monkeypatch.setattr(routes, "_collect_spec_documents", _collect_then_reimport)

    client.app["state"] = _slot_stub()[0]
    await client.start_server()
    try:
        resp = await client.get(f"{_BASE}/specs/s")
        payload = await resp.json()
    finally:
        await client.close()

    assert resp.status == 409, payload
    assert payload.get("code") == "spec_changed_during_read", payload


@pytest.mark.asyncio
async def test_index_keeps_resolving_an_observed_per_creation_slot_key(tmp_path, monkeypatch):
    """An agent cannot detach a live turn by replacing its authenticated slot key."""
    _redirect_state(monkeypatch, tmp_path)
    spec_dir = str(tmp_path / "spec")
    original = "spec-builder-s-1234abcd"
    replacement = "spec-builder-s-deadbeef"
    routes._save_index({"s": {"spec_dir": spec_dir, "slot_key": original}})
    assert routes._load_index()["s"]["slot_key"] == original

    routes._index_path().write_text(
        json.dumps({"s": {"spec_dir": spec_dir, "slot_key": replacement}})
    )

    changed = routes._load_index()["s"]
    assert changed["slot_key"] == replacement
    assert routes._slot_key("s") == original
    assert routes._OBSERVED_SLOT_KEYS["s"] == original
    assert await routes._pin_legacy_slot_identity("s", changed) is None


@pytest.mark.asyncio
async def test_deleted_creation_releases_its_observed_slot_identity(tmp_path, monkeypatch):
    """A trusted delete permits a later creation to mint a fresh identity."""
    _redirect_state(monkeypatch, tmp_path)
    spec_dir = str(tmp_path / "spec")
    original = "spec-builder-s-1234abcd"
    replacement = "spec-builder-s-deadbeef"
    routes._save_index({"s": {"spec_dir": spec_dir, "slot_key": original}})

    routes._index_path().write_text(
        json.dumps({"s": {"spec_dir": spec_dir, "slot_key": replacement}})
    )
    routes._load_index()
    assert routes._slot_key("s") == original

    def _delete(index):
        del index["s"]
        return True

    assert await routes._mutate_index(
        _delete,
        on_commit=lambda: routes._forget_observed_slot_identity("s", replacement, original),
    )
    routes._save_index({"s": {"spec_dir": spec_dir, "slot_key": replacement}})

    assert routes._load_index()["s"]["slot_key"] == replacement


@pytest.mark.asyncio
async def test_legacy_message_pins_its_name_derived_slot_before_alias_scan(tmp_path, monkeypatch):
    """A genuine pre-key spec remains usable without weakening missing-key aliases."""
    client = _make_client(monkeypatch, tmp_path)
    spec_dir = tmp_path / "wd" / ".kiro" / "specs" / "s"
    spec_dir.mkdir(parents=True)
    routes._save_index(
        {
            "s": {
                "spec_dir": str(spec_dir),
                "working_dir": str(tmp_path / "wd"),
                "spec_type": "feature",
                "status": "planning",
            }
        }
    )
    state, _slot = _slot_stub()
    dispatched: list[str] = []
    monkeypatch.setattr(
        routes,
        "_dispatch_turn",
        lambda *_args, **_kwargs: dispatched.append("sent"),
    )

    client.app["state"] = state
    await client.start_server()
    try:
        response = await client.post(
            f"{_BASE}/specs/s/message",
            json={"spec_dir": str(spec_dir), "text": "continue"},
        )
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 200, payload
    assert routes._load_index()["s"]["slot_key"] == "spec-builder-s"
    assert dispatched == ["sent"]


@pytest.mark.asyncio
async def test_legacy_slot_read_cannot_erase_observed_per_creation_identity(tmp_path, monkeypatch):
    """A temporary legacy key cannot make later key removal look like migration."""
    _redirect_state(monkeypatch, tmp_path)
    spec_dir = str(tmp_path / "spec")
    per_creation = "spec-builder-s-1234abcd"
    routes._save_index({"s": {"spec_dir": spec_dir, "slot_key": per_creation}})
    routes._save_index({"s": {"spec_dir": spec_dir, "slot_key": "spec-builder-s"}})
    routes._save_index({"s": {"spec_dir": spec_dir}})

    assert routes._OBSERVED_SLOT_KEYS["s"] == per_creation
    assert routes._slot_key("s") == per_creation
    assert await routes._pin_legacy_slot_identity("s", {"spec_dir": spec_dir}) is None


@pytest.mark.asyncio
async def test_stop_refuses_a_spec_recreated_at_the_same_path(tmp_path, monkeypatch):
    """The directory cannot detect a delete + recreate; the slot key can.

    Rechecking the canonical directory after waiting for the lock is not enough: a
    recreate at the SAME path resolves to the same directory -- so the recheck
    passes and Stop cancels the replacement's run. Slot keys are minted per
    creation, so the replacement necessarily carries a different one.
    """
    client = _make_client(monkeypatch, tmp_path)
    spec_dir, slot_key = _decision_spec(tmp_path)
    dir_key = routes._decision_key(spec_dir)

    halted: list = []

    async def _halt_spy(*_a, **_kw):
        halted.append("halt")

    monkeypatch.setattr(routes, "_halt_execution", _halt_spy)

    state, _slot = _slot_stub()
    client.app["state"] = state
    await client.start_server()
    try:
        async with routes._turn_lock(dir_key):
            stop = asyncio.ensure_future(client.post(f"{_BASE}/specs/s/stop", json={}))
            # nosemgrep: kirocrew.test-sleep-as-barrier -- kept verbatim by the test_routes.py split
            await asyncio.sleep(0.75)  # let it reach the lock and block there
            # The spec is deleted and recreated at the SAME path while Stop waits:
            # same directory, new creation, new slot key.
            index = routes._load_index()
            routes._forget_observed_slot_identity("s", slot_key)
            index["s"]["slot_key"] = "spec-builder-s-99999999"
            routes._save_index(index)
        resp = await stop
        status, payload = resp.status, await resp.json()
    finally:
        await client.close()

    assert status == 409, f"{status}: {payload}"
    assert payload["code"] == "stale_client", payload
    assert halted == [], "Stop cancelled the run of a spec created after it started waiting"
    assert slot_key != "spec-builder-s-99999999"
