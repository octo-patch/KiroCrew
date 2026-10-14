"""Dashboard chat: the ``/api/chat`` handler — body forwarding, drain on disconnect, browse gating."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_app, _make_state


@pytest.mark.asyncio
class TestApiChatDrainOnDisconnect:
    """Cover the slot.drain() call in chat_handlers' SSE finally block."""

    async def test_sse_reader_drains_pending_on_cancel(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")

        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            assert _directive_user_origin is True
            sl.append("chunk", "partial answer", "chunk")
            # nosemgrep: kirocrew.test-sleep-as-barrier -- the fake turn parks here until the disconnect cancels it
            await asyncio.sleep(60)

        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={"message": "hello", "slot": "s1"},
                timeout=None,
            )
            line = b""
            async for chunk in resp.content.iter_any():
                line += chunk
                if b"partial answer" in line:
                    break

            resp.close()
            # nosemgrep: kirocrew.test-sleep-as-barrier -- the SSE handler's finally block drains after the close and sets no signal to await
            await asyncio.sleep(0.1)

        assert slot._pending == []
        assert slot._has_reader is False


@pytest.mark.asyncio
class TestApiChatMemoryModeForwarding:
    """api_chat propagates body.memory_mode to auto-created slot.

    AgentRock skill dispatch sends `memory_mode: "temporary"` so one-shot
    skill invocations don't bleed into the user's persistent memory. The
    chat endpoint must honor this on the auto-create path because
    AgentRock does not pre-create the slot via /api/chat/slots.
    """

    async def test_temporary_memory_mode_propagates_to_new_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            assert _directive_user_origin is True
            sl.append("chunk", "ack", "chunk")

        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "hello",
                    "slot": "agentrock-skill-1",
                    "memory_mode": "temporary",
                },
                timeout=None,
            )
            async for _chunk in resp.content.iter_any():
                break  # only need to drive slot creation
            resp.close()
            # nosemgrep: kirocrew.test-sleep-as-barrier -- the closed request's handler sets no signal to await before the slot is read
            await asyncio.sleep(0.05)

        slot = state._slots.get("agentrock-skill-1")
        assert slot is not None
        assert slot.memory_mode == "temporary"

    async def test_missing_memory_mode_defaults_to_persistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            assert _directive_user_origin is True
            sl.append("chunk", "ack", "chunk")

        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={"message": "hello", "slot": "default-slot"},
                timeout=None,
            )
            async for _chunk in resp.content.iter_any():
                break
            resp.close()
            # nosemgrep: kirocrew.test-sleep-as-barrier -- the closed request's handler sets no signal to await before the slot is read
            await asyncio.sleep(0.05)

        slot = state._slots.get("default-slot")
        assert slot is not None
        assert slot.memory_mode == "persistent"

    async def test_invalid_memory_mode_is_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            assert _directive_user_origin is True
            sl.append("chunk", "ack", "chunk")

        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "hello",
                    "slot": "garbage-mode-slot",
                    "memory_mode": "garbage",
                },
                timeout=None,
            )
            async for _chunk in resp.content.iter_any():
                break
            resp.close()
            # nosemgrep: kirocrew.test-sleep-as-barrier -- the closed request's handler sets no signal to await before the slot is read
            await asyncio.sleep(0.05)

        slot = state._slots.get("garbage-mode-slot")
        assert slot is not None
        assert slot.memory_mode == "persistent"

    async def test_mismatched_memory_mode_on_existing_slot_returns_409(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mock_sel = MagicMock()
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: mock_sel)
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        # Pre-create a persistent slot
        state.get_or_create_slot("locked", memory_mode="persistent")

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "hello",
                    "slot": "locked",
                    "memory_mode": "temporary",
                },
                timeout=None,
            )
            assert resp.status == 409
            data = await resp.json()
            assert "memory_mode" in data["error"]

        # SEL audit event for the denial
        denied_calls = [
            c for c in mock_sel.log_api_access.call_args_list if c[1].get("outcome") == "denied"
        ]
        assert len(denied_calls) == 1
        kw = denied_calls[0][1]
        assert kw["operation"] == "chat_send"
        assert kw["source"] == "memory_mode_mismatch"
        assert "slot=locked" in kw["resources"]


@pytest.mark.asyncio
class TestApiChatModeForwarding:
    """api_chat propagates a validated body.mode to the auto-created slot.

    The Design Critique app's worker slot (mode="design-critique") lives only
    in gateway memory. After a gateway restart the app's next send() recreates
    the slot through this auto-create path; without mode forwarding the slot
    is reborn with mode="" — it then serializes surface="" and passes the chat
    sidebar's surface allowlist, leaking the throwaway worker session into the
    user's chat list. Mirrors TestApiChatMemoryModeForwarding above.
    """

    async def _post_chat(self, state, body):
        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            assert _directive_user_origin is True
            sl.append("chunk", "ack", "chunk")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat", json=body, timeout=None)
                assert resp.status == 200
                async for _chunk in resp.content.iter_any():
                    break  # only need to drive slot creation
                resp.close()
                await asyncio.sleep(0.05)

    async def test_design_critique_mode_propagates_to_new_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        await self._post_chat(
            state,
            {
                "message": "critique this",
                "slot": "dc-12345",
                "memory_mode": "temporary",
                "mode": "design-critique",
            },
        )

        slot = state._slots.get("dc-12345")
        assert slot is not None
        assert slot.mode == "design-critique"
        # The leak this guards against: the sidebar allowlist admits the plain
        # "" surface — the recreated worker must not serialize it.
        assert slot.mode != ""

    async def test_bogus_mode_is_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        await self._post_chat(
            state,
            {"message": "hello", "slot": "bogus-mode-slot", "mode": "not-a-mode"},
        )

        slot = state._slots.get("bogus-mode-slot")
        assert slot is not None
        assert slot.mode == ""

    async def test_non_string_mode_is_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        await self._post_chat(
            state,
            {"message": "hello", "slot": "nonstring-mode-slot", "mode": 42},
        )

        slot = state._slots.get("nonstring-mode-slot")
        assert slot is not None
        assert slot.mode == ""

    async def test_retired_crew_mode_is_dropped(self, tmp_path, monkeypatch):
        """Crew Mode retired: a caller still sending ``mode: "crew"`` (an old
        client, a saved script) gets an ordinary slot, not an error — the mode
        is absent from ``_CREATABLE_MODES``, so it is dropped like any other
        unknown value."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        await self._post_chat(
            state,
            {"message": "hello", "slot": "old-crew-client", "mode": "crew"},
        )

        slot = state._slots.get("old-crew-client")
        assert slot is not None, "slot should still be created, just as plain chat"
        assert slot.mode == ""

    async def test_mode_ignored_for_existing_slot(self, tmp_path, monkeypatch):
        """Repeating mode on send() must be safe when the slot survived: an
        existing slot keeps its stored mode and no error is raised."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("dc-alive", mode="design-critique", memory_mode="temporary")

        await self._post_chat(
            state,
            {
                "message": "hello again",
                "slot": "dc-alive",
                "memory_mode": "temporary",
                "mode": "design-critique",
            },
        )

        slot = state._slots.get("dc-alive")
        assert slot is not None
        assert slot.mode == "design-critique"


@pytest.mark.asyncio
class TestApiChatNoBrowseMarker:
    """Browse is gated by tool AVAILABILITY, not a per-message marker: the chat
    handler injects nothing into the user message, and the agent itself decides
    whether to operate a browser or read with web_fetch. This pins that the
    persisted message is verbatim (no `[BROWSE]` prefix), regardless of any legacy
    `browse` field a client might still send."""

    async def _send(self, tmp_path, monkeypatch, *, body_extra: dict):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            assert _directive_user_origin is True
            sl.append("chunk", "ack", "chunk")

        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)

        slot_key = body_extra.get("slot", "browse-slot")
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={"message": "look at example.com", **body_extra},
                timeout=None,
            )
            async for _chunk in resp.content.iter_any():
                break
            resp.close()
            await asyncio.sleep(0.05)
        return state._slots.get(slot_key)

    async def test_message_is_never_marked(self, tmp_path, monkeypatch):
        slot = await self._send(tmp_path, monkeypatch, body_extra={"slot": "plain-slot"})
        assert slot is not None
        user_msgs = [m for m in slot.messages if m.get("role") == "user"]
        assert user_msgs and user_msgs[-1]["content"] == "look at example.com"

    async def test_legacy_browse_field_is_ignored(self, tmp_path, monkeypatch):
        # A client that still sends the old `browse` field must not change the
        # stored message: the marker mechanism is gone entirely.
        slot = await self._send(
            tmp_path, monkeypatch, body_extra={"slot": "legacy-slot", "browse": True}
        )
        assert slot is not None
        user_msgs = [m for m in slot.messages if m.get("role") == "user"]
        assert user_msgs and not user_msgs[-1]["content"].startswith("[BROWSE]")
