"""Route tests: deleting a spec.

Delete reserves the name, tombstones the directory, tears down the worker slot
and its loop, and only then drops the index entry; a failure restores what it
took. Pause and delete discard queued work before they stop the turn, and a new
spec never inherits a deleted one's conversation.
"""

from __future__ import annotations

import types

import pytest

from kiro_crew.apps.builtins.spec_builder.tests.routes_facade import routes

# _never_touch_the_real_state is an autouse fixture: pytest finds it by the name bound here.
from kiro_crew.apps.builtins.spec_builder.tests.routes_helpers import (  # noqa: F401
    _BASE,
    _decision_spec,
    _make_client,
    _never_touch_the_real_state,
)


@pytest.mark.asyncio
async def test_partial_slot_teardown_keeps_the_delete_reserved(tmp_path, monkeypatch):
    """A spec cannot reappear after one captured worker was already destroyed."""
    client = _make_client(monkeypatch, tmp_path)
    spec_dir, slot_key = _decision_spec(tmp_path)
    stale_slot_key = "spec-builder-s-feedbeef"
    routes._OBSERVED_SPEC_DIRS[stale_slot_key] = routes._decision_key(spec_dir)
    slots = {
        slot_key: types.SimpleNamespace(key=slot_key),
        stale_slot_key: types.SimpleNamespace(key=stale_slot_key),
    }

    class _State:
        def get_slot(self, key):
            return slots.get(key)

    teardown_results = iter((True, False))

    async def _partially_teardown(*_args, **_kwargs):
        return next(teardown_results)

    monkeypatch.setattr(routes, "_teardown_worker_slot", _partially_teardown)

    client.app["state"] = _State()
    await client.start_server()
    try:
        response = await client.delete(
            f"{_BASE}/specs/s",
            json={"spec_dir": spec_dir, "slot_key": slot_key},
        )
        body = await response.json()
    finally:
        await client.close()

    assert response.status == 503, body
    assert body["code"] == "archive_failed"
    assert "retry" in body["error"]
    entry = routes._load_index()["s"]
    assert entry.get(routes._DELETING), "the partially deleted spec was unhidden"
    assert spec_dir in routes._load_deleted(), "the partial delete lost its tombstone"
    monkeypatch.setattr(routes, "_PROCESS_ID", "replacement-process")
    restarted = routes._load_index()["s"]
    assert restarted.get(routes._DELETING), "restart exposed the partial delete"


@pytest.mark.asyncio
async def test_legacy_delete_refuses_same_path_replacement_before_reservation(
    tmp_path, monkeypatch
):
    """A missing legacy key still has the captured name-derived creation identity.

    Replacing that row at the same name and directory after DELETE's snapshot must
    not let the stale request reserve and remove the replacement.
    """
    client = _make_client(monkeypatch, tmp_path)
    base = tmp_path / "wd" / ".kiro" / "specs"
    (base / "legacy").mkdir(parents=True)
    spec_dir = str(base / "legacy")
    routes._save_index(
        {
            "legacy": {
                "spec_dir": spec_dir,
                "working_dir": str(tmp_path / "wd"),
                "spec_type": "feature",
                "status": "planning",
            }
        }
    )
    replacement_keys: list[str] = []
    real_alias_status = routes._aload_index_with_decision_alias_status

    async def _replace_then_check_aliases(directory):
        index = await routes._aload_index()
        replacement_key = routes._new_slot_key("legacy")
        replacement_keys.append(replacement_key)
        index["legacy"]["slot_key"] = replacement_key
        routes._save_index(index)
        return await real_alias_status(directory)

    monkeypatch.setattr(
        routes, "_aload_index_with_decision_alias_status", _replace_then_check_aliases
    )

    client.app["state"] = None
    await client.start_server()
    try:
        response = await client.delete(f"{_BASE}/specs/legacy")
        body = await response.json()
    finally:
        await client.close()

    assert response.status == 404, body
    current = routes._load_index()["legacy"]
    assert current["slot_key"] == replacement_keys[0]
    assert routes._DELETING not in current
    assert spec_dir not in routes._load_deleted()
