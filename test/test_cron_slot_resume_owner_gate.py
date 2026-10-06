"""Resuming a cron slot serves a non-owner only the rows slot detail would.

``POST /api/chat/slots/{slot}/resume`` returns the slot's window on both of its
branches: the already-live dedup body and the freshly restored slot. Both go
through :func:`_cron_slot_rows_for_caller`, the filter slot detail applies, so a
non-owner sees only the ``cron:{id}`` rows ``_non_owner_transcript_rows`` proves
came from an unbound run. These tests pin the helper and both branches.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.dashboard import chat_handlers

_OWNER = "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request"
_ROWS = "kiro_crew.dashboard.handlers.cron._non_owner_transcript_rows"
_KEY = "kiro_crew.dashboard.chat_handlers.slot_history_key"
_FACADE = "kiro_crew.dashboard.chat_handlers"

_UNBOUND = {"role": "assistant", "content": "public result", "ts": "t1"}
_BOUND = {"role": "assistant", "content": "agent read /private/proj/secret.txt", "ts": "t2"}
_WINDOW = [_UNBOUND, _BOUND]


def _slot(key="cron-abcd1234", linked=""):
    return SimpleNamespace(
        key=key,
        linked_session_key=linked,
        messages=list(_WINDOW),
        running=False,
        workspace="",
        _queue=[],
        _disk_older_count=0,
        memory_mode="persistent",
        mode="",
    )


def _gate(*, owner, key="cron:abcd1234"):
    """Patch the owner check, the slot's history key and the row filter."""
    rows_filter = AsyncMock(return_value=([_UNBOUND], True))
    patches = (
        patch(_OWNER, lambda request: owner),
        patch(_KEY, lambda slot: key),
        patch(_ROWS, rows_filter),
    )
    return patches, rows_filter


class TestTheHelper:
    @pytest.mark.asyncio
    async def test_a_non_owner_keeps_only_the_rows_the_filter_proves_unbound(self):
        patches, rows_filter = _gate(owner=False)
        with patches[0], patches[1], patches[2]:
            kept = await chat_handlers._cron_slot_rows_for_caller(
                MagicMock(), MagicMock(), _slot(), list(_WINDOW)
            )
        assert kept == [_UNBOUND]
        assert rows_filter.await_args.args[1:] == ("abcd1234", _WINDOW)

    @pytest.mark.asyncio
    async def test_the_owner_gets_every_row_unchecked(self):
        patches, rows_filter = _gate(owner=True)
        with patches[0], patches[1], patches[2]:
            kept = await chat_handlers._cron_slot_rows_for_caller(
                MagicMock(), MagicMock(), _slot(), list(_WINDOW)
            )
        assert kept == _WINDOW
        rows_filter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_slot_that_is_not_a_cron_transcript_is_not_filtered(self):
        patches, rows_filter = _gate(owner=False, key="dashboard:chat-1")
        with patches[0], patches[1], patches[2]:
            kept = await chat_handlers._cron_slot_rows_for_caller(
                MagicMock(), MagicMock(), _slot(key="chat-1"), list(_WINDOW)
            )
        assert kept == _WINDOW
        rows_filter.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "slot_key, linked",
        [
            # Any name a resume request carries, over the job's transcript.
            ("report", "cron:abcd1234"),
            # The History pane's own spelling: the row's transcript stem.
            ("cron_abcd1234", ""),
        ],
    )
    async def test_a_cron_transcript_resumed_under_another_name_is_still_filtered(
        self, slot_key, linked
    ):
        patches, rows_filter = _gate(owner=False, key="dashboard:" + slot_key)
        with patches[0], patches[1], patches[2]:
            kept = await chat_handlers._cron_slot_rows_for_caller(
                MagicMock(), MagicMock(), _slot(key=slot_key, linked=linked), list(_WINDOW)
            )
        assert kept == [_UNBOUND]
        assert rows_filter.await_args.args[1:] == ("abcd1234", _WINDOW)


async def _resume(*, owner, already_live):
    """POST a resume for a cron slot and return (status, body)."""
    slot = _slot()
    outcome = chat_handlers.ResumeOutcome(
        slot=slot, already_live=already_live, total=len(slot.messages)
    )
    state = MagicMock(conversation_log=MagicMock())
    app = web.Application()
    app["state"] = state
    app.router.add_post("/api/chat/slots/{slot}/resume", chat_handlers.api_chat_slot_resume)
    patches, _rows_filter = _gate(owner=owner)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch(f"{_FACADE}.resume_slot_from_history", AsyncMock(return_value=outcome)),
        patch(f"{_FACADE}.load_window_off_loop", AsyncMock()),
        patch(f"{_FACADE}._reconcile_slot_window", AsyncMock()),
        patch(f"{_FACADE}._collapse_wire_rows", lambda rows: list(rows)),
        patch(f"{_FACADE}._prepare_messages", lambda rows, *_a, **_k: list(rows)),
        patch(f"{_FACADE}._live_child_instance", lambda *_a: None),
    ):
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/cron-abcd1234/resume", json={})
            return resp.status, await resp.json()


class TestBothResumeBranchesFilter:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("already_live", [True, False], ids=["live", "restored"])
    async def test_a_non_owner_never_receives_a_bound_row(self, already_live):
        status, body = await _resume(owner=False, already_live=already_live)

        assert status == 200
        assert body["messages"] == [_UNBOUND]
        # Paging is counted on the raw window, so the hidden row does not
        # turn into a phantom older page.
        assert body["total"] == len(_WINDOW)
        assert body["has_more"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("already_live", [True, False], ids=["live", "restored"])
    async def test_the_owner_receives_every_row(self, already_live):
        status, body = await _resume(owner=True, already_live=already_live)

        assert status == 200
        assert body["messages"] == _WINDOW
