"""Route tests: duplicating a spec.

The copy is staged, written through the reservation's pinned stage directory,
fsynced and published without replacing anything; startup recovery finishes or
discards a duplicate a dead process left behind; and the duplicate route refuses
a busy, mixed, unreadable or name-colliding source.
"""

from __future__ import annotations

import asyncio
import os
import threading
import types
from pathlib import Path
from typing import cast

import pytest
from aiohttp import web

from kiro_crew.apps.builtins.spec_builder.tests.routes_facade import routes

# _never_touch_the_real_state is an autouse fixture: pytest finds it by the name bound here.
from kiro_crew.apps.builtins.spec_builder.tests.routes_helpers import (  # noqa: F401
    _BASE,
    _make_client,
    _never_touch_the_real_state,
    _redirect_state,
    _seed_spec,
    _spec_identity,
    _state_for,
)

# ── gap 1: documents are reviewable through their redacted rendering ─────────


def test_duplicate_doc_create_only_succeeds_while_the_file_is_absent(tmp_path):
    """O_EXCL makes an external writer the winner rather than its victim."""
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()

    result, identity = routes._create_spec_doc(spec_dir, "design.md", "# fresh")
    assert result == "" and identity is not None
    assert (spec_dir / "design.md").read_text() == "# fresh"

    result, identity = routes._create_spec_doc(spec_dir, "design.md", "# second")
    assert result == "conflict" and identity is None
    assert (spec_dir / "design.md").read_text() == "# fresh"


@pytest.mark.skipif(
    not routes._CAN_PIN_DIR,
    reason="duplicate document writes require descriptor-relative directory pinning",
)
def test_duplicate_doc_create_refuses_a_spec_directory_swapped_for_a_symlink(tmp_path):
    """Same attack the sentinel writers refuse: an agent replaces its own indexed
    directory with a symlink, and a path-based write then lands user-authored text
    somewhere else entirely."""
    real = tmp_path / "elsewhere"
    real.mkdir()
    attacker = tmp_path / "attacker-spec"
    os.symlink(real, attacker)

    assert routes._create_spec_doc(attacker, "design.md", "# text")[0] == "unsafe_dir"
    assert not (real / "design.md").exists()


def test_duplicate_doc_create_retries_short_writes(tmp_path, monkeypatch):
    """A successful save means the complete UTF-8 payload reached the temporary
    file; a regular-file write is allowed to accept only a prefix."""
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    real_write = routes.os.write
    writes: list[int] = []

    def _short_write(fd, data):
        chunk = data[:2]
        writes.append(len(chunk))
        return real_write(fd, chunk)

    monkeypatch.setattr(routes.os, "write", _short_write)
    content = "complete payload"

    assert routes._create_spec_doc(spec_dir, "design.md", content)[0] == ""
    assert (spec_dir / "design.md").read_text() == content
    assert len(writes) > 1


# ── gap 5: duplicate ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_copies_the_documents_into_a_fresh_spec(tmp_path, monkeypatch):
    """The recovery path rename cannot serve: a spec whose NAME is wrong once it
    already has a branch or history. The copy takes the documents and nothing
    else, so it gets its own slot key and therefore its own conversation."""
    client = _make_client(monkeypatch, tmp_path)
    _seed_spec(tmp_path, files={"requirements.md": "# reqs", "design.md": "# design"})
    state, _ = _state_for("live")
    sent: list[str] = []
    monkeypatch.setattr(routes, "_dispatch_turn", lambda st, slot, msg: sent.append(msg))

    client.app["state"] = state
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "live-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 201, body
    copy_dir = Path(body["spec_dir"])
    assert copy_dir.name == "live-copy"
    assert (copy_dir / "requirements.md").read_text() == "# reqs"
    assert (copy_dir / "design.md").read_text() == "# design"
    assert not (copy_dir / routes._DUPLICATE_MARKER).exists()

    index = routes._load_index()
    assert index["live"]["spec_dir"] != index["live-copy"]["spec_dir"]
    assert index["live-copy"]["slot_key"] != index["live"]["slot_key"], "shared a transcript"
    # A duplicate never inherits a worktree: branching off someone's repo is not a
    # copy operation, and it is an opt-in at create time.
    assert index["live-copy"]["worktree_branch"] == ""
    # The fresh conversation is told what it is looking at, and told not to rewrite it.
    assert len(sent) == 1 and "copy of 'live'" in sent[0] and "Do NOT rewrite" in sent[0]


@pytest.mark.asyncio
async def test_duplicate_refuses_while_the_source_agent_is_writing(tmp_path, monkeypatch):
    """A live agent turn can rewrite phase files while they are being copied."""
    client = _make_client(monkeypatch, tmp_path)
    _seed_spec(tmp_path, files={"requirements.md": "# reqs"})
    state, slots = _state_for("live")
    slots[routes._slot_key("live")].running = True
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *_args: None)

    client.app["state"] = state
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "live-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 409 and body["code"] == "agent_running"
    assert "live-copy" not in routes._load_index()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_refuses_a_mixed_source_document_snapshot(tmp_path, monkeypatch):
    """Every copied phase file must belong to one stable source snapshot."""
    client = _make_client(monkeypatch, tmp_path)
    spec_dir, _working_dir = _seed_spec(
        tmp_path,
        files={"requirements.md": "# requirements v1", "design.md": "# design v1"},
    )
    real_read = routes._read_spec_text
    changed = False

    def _change_plan_between_reads(directory, fname):
        nonlocal changed
        text = real_read(directory, fname)
        if fname == "requirements.md" and not changed:
            changed = True
            (spec_dir / "requirements.md").write_text("# requirements v2")
            (spec_dir / "design.md").write_text("# design v2")
        return text

    monkeypatch.setattr(routes, "_read_spec_text", _change_plan_between_reads)
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *_args: None)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "mixed-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 409 and body["code"] == "spec_changed_during_duplicate"
    assert "mixed-copy" not in routes._load_index()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_redacts_the_returned_spec_directory(tmp_path, monkeypatch):
    """A credential-like project path must not cross the HTTP response boundary raw."""
    client = _make_client(monkeypatch, tmp_path)
    secret_root = tmp_path / "secret-segment"
    _seed_spec(secret_root, files={"requirements.md": "# copied"})
    monkeypatch.setattr(
        routes,
        "_redact",
        lambda text: text.replace("secret-segment", "[REDACTED]"),
    )
    monkeypatch.setattr(routes, "_dispatch_turn", lambda _state, _slot, _message: None)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "live-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 201, body
    assert "secret-segment" not in body["spec_dir"]
    assert "[REDACTED]" in body["spec_dir"]
    assert Path(routes._load_index()["live-copy"]["spec_dir"]).is_dir()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_publishes_recovery_provenance_before_index_reservation(
    tmp_path, monkeypatch
):
    """A crash after reservation must leave proof that recovery may discard it."""
    client = _make_client(monkeypatch, tmp_path)
    _seed_spec(tmp_path, files={"requirements.md": "# copied"})
    real_mutate = routes._mutate_index

    async def _assert_provenance(fn):
        changed = await real_mutate(fn)
        if changed and getattr(fn, "__name__", "") == "_insert":
            meta = routes._load_index()["live-copy"]
            held = meta[routes._DUPLICATING]
            assert routes._duplicate_marker_matches(Path(held["stage_dir"]), held["token"])
        return changed

    monkeypatch.setattr(routes, "_mutate_index", _assert_provenance)
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *_args: None)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "live-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 201, body


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_cancelled_duplicate_finishes_its_published_transaction(tmp_path, monkeypatch):
    """Request cancellation cannot hide a published copy behind our reservation."""
    _redirect_state(monkeypatch, tmp_path)
    _seed_spec(tmp_path, files={"requirements.md": "# copied"})
    state, _ = _state_for("live")
    published = threading.Event()
    release_write = threading.Event()
    real_write = routes._write_and_publish_duplicate

    def _publish_then_block(*args, **kwargs):
        result = real_write(*args, **kwargs)
        assert result[0] == ""
        published.set()
        assert release_write.wait(5), "cancelled transaction was never released"
        return result

    async def _body(_request):
        return {"new_name": "cancelled-copy"}

    async def _fresh(_request, _name, _body):
        return routes._load_index()["live"]

    monkeypatch.setattr(routes, "_require_auth", lambda _request: None)
    monkeypatch.setattr(routes, "_read_json", _body)
    monkeypatch.setattr(routes, "_pinned_entry", _fresh)
    monkeypatch.setattr(routes, "_write_and_publish_duplicate", _publish_then_block)
    request = cast(
        web.Request,
        types.SimpleNamespace(match_info={"name": "live"}, app={"state": state}),
    )

    handler = asyncio.create_task(routes._handle_duplicate(request))
    try:
        assert await asyncio.to_thread(published.wait, 5)
        handler.cancel()
        release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await handler
    finally:
        release_write.set()

    meta = routes._load_index()["cancelled-copy"]
    target = Path(meta["spec_dir"])
    assert routes._DUPLICATING not in meta
    assert (target / "requirements.md").read_text() == "# copied"
    assert not (target / routes._DUPLICATE_MARKER).exists()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_preserves_an_existing_empty_document(tmp_path, monkeypatch):
    """An empty phase file is data, not the same thing as a missing document."""
    client = _make_client(monkeypatch, tmp_path)
    _seed_spec(tmp_path, files={"requirements.md": "# reqs", "design.md": ""})
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *_args: None)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "live-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 201, body
    copy_dir = Path(body["spec_dir"])
    assert (copy_dir / "requirements.md").read_text() == "# reqs"
    assert (copy_dir / "design.md").is_file()
    assert (copy_dir / "design.md").read_text() == ""


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_refuses_an_existing_document_that_cannot_be_read(tmp_path, monkeypatch):
    """An oversized source document must not be silently omitted from a copy."""
    client = _make_client(monkeypatch, tmp_path)
    _, working_dir = _seed_spec(
        tmp_path,
        files={
            "requirements.md": "# copied",
            "design.md": "x" * (routes._MAX_SPEC_BYTES + 1),
        },
    )
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *_args: None)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "incomplete-copy"},
        )
        assert resp.status == 409
        body = await resp.json()
    finally:
        await client.close()

    assert body["code"] == "spec_document_unreadable"
    assert "incomplete-copy" not in routes._load_index()
    assert not (working_dir / ".kiro" / "specs" / "incomplete-copy").exists()


@pytest.mark.asyncio
async def test_duplicate_refuses_a_name_already_in_use(tmp_path, monkeypatch):
    client = _make_client(monkeypatch, tmp_path)
    _seed_spec(tmp_path, files={"requirements.md": "# reqs"})

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        same = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "live"},
        )
        same_body = await same.json()
        bad = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "../escape"},
        )
        bad_body = await bad.json()
    finally:
        await client.close()

    assert same.status == 409 and same_body["code"] == "spec_exists"
    assert bad.status == 400 and bad_body["code"] == "invalid_name"


@pytest.mark.asyncio
async def test_duplicate_refuses_a_spec_with_no_documents_yet(tmp_path, monkeypatch):
    """Nothing to copy is a refusal, not an empty spec created as a side effect."""
    client = _make_client(monkeypatch, tmp_path)
    _seed_spec(tmp_path)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "empty-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 409 and body["code"] == "nothing_to_copy"
    assert "empty-copy" not in routes._load_index()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_reserves_its_name_before_populating_the_destination(tmp_path, monkeypatch):
    """A concurrent create must not win the destination index entry while the
    duplicate is already copying documents into that same directory."""
    client = _make_client(monkeypatch, tmp_path)
    _, working_dir = _seed_spec(tmp_path, files={"requirements.md": "# copied"})
    state, _ = _state_for("live")
    real_create = routes._create_spec_doc
    save_started = threading.Event()
    release_save = threading.Event()

    def _held_save(*args, **kwargs):
        save_started.set()
        assert release_save.wait(5), "concurrent create never reached arbitration"
        return real_create(*args, **kwargs)

    monkeypatch.setattr(routes, "_create_spec_doc", _held_save)
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *_args, **_kwargs: None)

    client.app["state"] = state
    await client.start_server()
    try:
        duplicate_request = asyncio.create_task(
            client.post(
                f"{_BASE}/specs/live/duplicate",
                json={**_spec_identity(), "new_name": "shared-name"},
            )
        )
        assert await asyncio.to_thread(save_started.wait, 5)
        create = await client.post(
            f"{_BASE}/specs",
            json={
                "name": "shared-name",
                "working_dir": str(working_dir),
                "spec_type": "feature",
                "description": "concurrent create",
                "use_worktree": False,
            },
        )
        create_body = await create.json()
        release_save.set()
        duplicate = await duplicate_request
        duplicate_body = await duplicate.json()
    finally:
        release_save.set()
        await client.close()

    assert create.status == 409 and create_body["code"] == "spec_exists"
    assert duplicate.status == 201, duplicate_body
    copy_dir = Path(duplicate_body["spec_dir"])
    assert (copy_dir / "requirements.md").read_text() == "# copied"
    assert routes._load_index()["shared-name"]["spec_dir"] == str(copy_dir)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_refuses_when_the_destination_changes_after_reservation(
    tmp_path, monkeypatch
):
    """The index reservation and copied files must always name one directory."""
    client = _make_client(monkeypatch, tmp_path)
    _, working_dir = _seed_spec(tmp_path, files={"requirements.md": "# copied"})
    moved_base = tmp_path / "moved-specs"
    moved_base.mkdir()
    state, _ = _state_for("live")
    real_mutate = routes._mutate_index
    moved = False

    async def _move_settings_after_reservation(fn):
        nonlocal moved
        changed = await real_mutate(fn)
        if changed and not moved and "live-copy" in routes._load_index():
            moved = True
            routes._save_settings({"base_path": str(moved_base), "model": ""})
        return changed

    monkeypatch.setattr(routes, "_mutate_index", _move_settings_after_reservation)
    monkeypatch.setattr(routes, "_dispatch_turn", lambda *_args, **_kwargs: None)

    client.app["state"] = state
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "live-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    assert resp.status == 409 and body["code"] == "spec_destination_changed"
    assert "live-copy" not in routes._load_index()
    assert not (moved_base / "live-copy").exists()
    assert not (working_dir / ".kiro" / "specs" / "live-copy").exists()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_rolls_back_only_files_it_created_after_a_partial_failure(
    tmp_path, monkeypatch
):
    """A failed copy must not become discoverable as an incomplete spec."""
    client = _make_client(monkeypatch, tmp_path)
    _, working_dir = _seed_spec(
        tmp_path,
        files={"requirements.md": "# copied", "design.md": "# fails"},
    )
    state, _ = _state_for("live")
    real_create = routes._create_spec_doc
    calls = 0

    def _fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return "write_failed", None
        return real_create(*args, **kwargs)

    monkeypatch.setattr(routes, "_create_spec_doc", _fail_second)

    client.app["state"] = state
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "partial-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    target = working_dir / ".kiro" / "specs" / "partial-copy"
    assert resp.status == 400 and body["code"] == "doc_write_failed"
    assert "partial-copy" not in routes._load_index()
    assert not target.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_preserves_a_published_copy_when_index_finalization_fails(
    tmp_path, monkeypatch
):
    """A post-rename failure remains recoverable instead of poisoning retries."""
    client = _make_client(monkeypatch, tmp_path)
    _, working_dir = _seed_spec(tmp_path, files={"requirements.md": "# copied"})
    real_mutate = routes._mutate_index

    async def _fail_finish(fn):
        if getattr(fn, "__name__", "") == "_finish":
            return False
        return await real_mutate(fn)

    monkeypatch.setattr(routes, "_mutate_index", _fail_finish)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "recoverable-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    target = working_dir / ".kiro" / "specs" / "recoverable-copy"
    assert resp.status == 409 and body["code"] == "spec_changed_during_create"
    assert (target / "requirements.md").read_text() == "# copied"
    assert (target / routes._DUPLICATE_MARKER).is_file()
    assert routes._DUPLICATING in routes._load_index()["recoverable-copy"]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_contains_an_index_finalization_exception(tmp_path, monkeypatch):
    """A post-publication index error must return the recoverable-copy response."""
    client = _make_client(monkeypatch, tmp_path)
    _, working_dir = _seed_spec(tmp_path, files={"requirements.md": "# copied"})
    real_mutate = routes._mutate_index

    async def _raise_finish(fn):
        if getattr(fn, "__name__", "") == "_finish":
            raise OSError("index storage unavailable")
        return await real_mutate(fn)

    monkeypatch.setattr(routes, "_mutate_index", _raise_finish)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "recoverable-copy"},
        )
        assert resp.status == 409
        body = await resp.json()
    finally:
        await client.close()

    target = working_dir / ".kiro" / "specs" / "recoverable-copy"
    assert body["code"] == "spec_changed_during_create"
    assert (target / "requirements.md").read_text() == "# copied"
    assert (target / routes._DUPLICATE_MARKER).is_file()
    assert routes._DUPLICATING in routes._load_index()["recoverable-copy"]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not routes._CAN_PUBLISH_DIR_NOREPLACE,
    reason="duplicate publication requires atomic no-replace directory publication",
)
async def test_duplicate_keeps_its_committed_copy_when_slot_setup_is_refused(tmp_path, monkeypatch):
    """Session arbitration must not delete a copy already committed to the index."""
    client = _make_client(monkeypatch, tmp_path)
    _, working_dir = _seed_spec(tmp_path, files={"requirements.md": "# copied"})

    async def _refuse_slot(*_args, **_kwargs):
        return None

    monkeypatch.setattr(routes, "_ensure_worker_slot", _refuse_slot)

    client.app["state"] = _state_for("live")[0]
    await client.start_server()
    try:
        resp = await client.post(
            f"{_BASE}/specs/live/duplicate",
            json={**_spec_identity(), "new_name": "committed-copy"},
        )
        body = await resp.json()
    finally:
        await client.close()

    target = working_dir / ".kiro" / "specs" / "committed-copy"
    assert resp.status == 409 and body["code"] == "slot_owned_by_another_app"
    assert (target / "requirements.md").read_text() == "# copied"
    assert "committed-copy" in routes._load_index()
