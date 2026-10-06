"""History reads apply the project-output owner boundary to ``cron:`` transcripts.

A project-bound job's output is owner-only. The chat-slot routes filter a cron
slot's rows per run, but History reads the same transcript FILE by key, so a
non-owner could otherwise read it through ``GET /api/sessions/{key}``, the list
preview, content search or a summary. Detail keeps only the rows the run-window
filter proves unbound; the preview, search hits and summaries, which cannot be
filtered per row, skip cron transcripts for a non-owner.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.cron import cron_job_id_from_transcript_key
from kiro_crew.dashboard.handlers import (
    api_session_detail,
    api_sessions,
    api_sessions_search,
    api_sessions_summarize,
)
from kiro_crew.history import ConversationLog

_OWNER = "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request"
_ROWS = "kiro_crew.dashboard.handlers.cron._non_owner_transcript_rows"
_SUMMARIZE = "kiro_crew.dashboard.handlers.sessions._summarize_one"

BOUND = "agent read the private QUOKKA-77 roadmap"
UNBOUND = "public digest QUOKKA-77 posted"


def _log(tmp_path) -> ConversationLog:
    log = ConversationLog(base_dir=tmp_path)
    log.append("cron:abcd1234", "assistant", UNBOUND)
    log.append("cron:abcd1234", "assistant", BOUND)
    log.append("dashboard:chat-1", "user", "my own QUOKKA-77 notes")
    return log


def _client(log: ConversationLog) -> TestClient:
    app = web.Application()
    app["state"] = SimpleNamespace(conversation_log=log)
    app.router.add_get("/api/sessions", api_sessions)
    app.router.add_get("/api/sessions/search", api_sessions_search)
    app.router.add_post("/api/sessions/summarize", api_sessions_summarize)
    app.router.add_get("/api/sessions/{key}", api_session_detail)
    return TestClient(TestServer(app))


def _keep_unbound():
    async def _filter(_state, job_id, rows):
        assert job_id == "abcd1234"
        return [r for r in rows if r.get("content") != BOUND], True

    return AsyncMock(side_effect=_filter)


class TestTheTranscriptIdentity:
    @pytest.mark.parametrize(
        "key", ["cron:abcd1234", "cron_abcd1234", "cron:abcd1234:run7", "cron_abcd1234_run7"]
    )
    def test_every_spelling_of_a_cron_transcript_names_the_job(self, key):
        assert cron_job_id_from_transcript_key(key) == "abcd1234"

    @pytest.mark.parametrize("key", ["dashboard:chat-1", "slack:1.2", "cron-abcd1234", "", None])
    def test_other_transcripts_name_no_job(self, key):
        assert cron_job_id_from_transcript_key(key) == ""


class TestDetail:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["cron:abcd1234", "cron_abcd1234"])
    async def test_a_non_owner_reads_only_rows_proved_unbound(self, tmp_path, key):
        rows_filter = _keep_unbound()
        with patch(_OWNER, lambda request: False), patch(_ROWS, rows_filter):
            async with _client(_log(tmp_path)) as client:
                body = await (await client.get(f"/api/sessions/{key}")).json()
        contents = [m["content"] for m in body]
        assert UNBOUND in contents and BOUND not in contents
        rows_filter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_owner_reads_every_row_unfiltered(self, tmp_path):
        rows_filter = _keep_unbound()
        with patch(_OWNER, lambda request: True), patch(_ROWS, rows_filter):
            async with _client(_log(tmp_path)) as client:
                body = await (await client.get("/api/sessions/cron_abcd1234")).json()
        assert BOUND in [m["content"] for m in body]
        rows_filter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_ordinary_transcript_is_not_filtered(self, tmp_path):
        rows_filter = _keep_unbound()
        with patch(_OWNER, lambda request: False), patch(_ROWS, rows_filter):
            async with _client(_log(tmp_path)) as client:
                body = await (await client.get("/api/sessions/dashboard_chat-1")).json()
        assert [m["content"] for m in body] == ["my own QUOKKA-77 notes"]
        rows_filter.assert_not_awaited()


class TestListPreview:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("owner", [False, True])
    async def test_a_cron_rows_preview_reaches_only_the_owner(self, tmp_path, owner):
        with patch(_OWNER, lambda request: owner):
            async with _client(_log(tmp_path)) as client:
                body = await (await client.get("/api/sessions?preview=1")).json()
        rows = {r["key"]: r for r in body["sessions"]}
        assert ("preview" in rows["cron_abcd1234"]) is owner
        assert "preview" in rows["dashboard_chat-1"]


class TestSearch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("owner, expected", [(False, False), (True, True)])
    async def test_a_cron_transcript_matches_only_for_the_owner(self, tmp_path, owner, expected):
        with patch(_OWNER, lambda request: owner):
            async with _client(_log(tmp_path)) as client:
                body = await (await client.get("/api/sessions/search?q=QUOKKA-77")).json()
        keys = {s["key"] for s in body["sessions"]}
        assert ("cron_abcd1234" in keys) is expected
        assert "dashboard_chat-1" in keys


class TestSummarize:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("owner", [False, True])
    async def test_a_cron_transcript_is_summarized_only_for_the_owner(self, tmp_path, owner):
        summarize = AsyncMock(return_value="a summary")
        with patch(_OWNER, lambda request: owner), patch(_SUMMARIZE, summarize):
            async with _client(_log(tmp_path)) as client:
                resp = await client.post(
                    "/api/sessions/summarize", json={"keys": ["cron_abcd1234", "dashboard_chat-1"]}
                )
                body = await resp.json()
        assert ("cron_abcd1234" in body["summaries"]) is owner
        assert "dashboard_chat-1" in body["summaries"]
