"""Tests for dashboard chat session — runtime wiring, tool calls, refusal fallback, approvals,
folders, regenerate and fork, stop, provider recovery, empty-response and transient
retries, model switching and reload.

The other dashboard chat tests live in the sibling ``test_dashboard_chat_*.py``
files, one topic each; shared fakes are in ``dashboard_chat_helpers.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import (
    AsyncIterator,
    _make_app,
    _make_app_with_agent_routes,
    _make_folder_app,
    _make_ready_kiro_prerequisite,
    _make_state,
    await_successor,
    chat_done_frames,
    close_before_resume,
    run_as_slot_task,
)
from dashboard_chat_helpers import _provider_mock
from dashboard_owner_helpers import as_owner

from kiro_crew.acp.types import ACP_BACKEND_CLAUDE, ACP_BACKEND_KIRO
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.config.loader import ResolvedBindings
from kiro_crew.dashboard.chat_runner import _tool_call_ws_payload
from kiro_crew.dashboard.recovery_replays import LiveSlot, ReplayFamily
from kiro_crew.dashboard.state import SUBAGENT_SYNTHESIS_PROMPT, DashboardState, _ChatSlot
from kiro_crew.history import ConversationLog
from kiro_crew.testing.wait import async_wait_until, await_until, until_parked


def test_tool_call_ws_payload_preserves_shell_capability_signal():
    """The dashboard receives an explicit shell signal for indeterminate UX.

    Keep this contract at the backend boundary so a future percentage-based
    progress mode can extend the payload without making the frontend infer
    tool type from a display title.
    """
    event = MagicMock(
        title="bash",
        tool_kind="execute",
        tool_call_id="tc-shell",
        tool_purpose="Run a command",
        tool_input="echo hello",
        is_shell=True,
    )

    payload = _tool_call_ws_payload(event)

    assert payload["tool"] == "bash"
    assert payload["kind"] == "execute"
    assert payload["is_shell"] is True
    assert payload["tool_call_id"] == "tc-shell"


# ── Runtime wiring tests (multi-agent-orchestration) ──


class TestRuntimeWiring:
    """Tests for multi-agent-orchestration runtime wiring.

    Requirements: 1.3, 2.3, 2.4, 3.1
    """

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_response_includes_workspace(self, tmp_path, monkeypatch):
        """api_chat_slot_agent response includes resolved workspace field.

        Requirements: 1.3
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")
        state.sessions.reset = AsyncMock()

        # Mock config loading to return a config with a known agent
        mock_cfg = MagicMock()
        mock_cfg.agents = {"oncall": MagicMock(workspace="oncall-ws", memory_store="oncall-mem")}
        mock_cfg.workspaces = {"oncall-ws": MagicMock(dir="/tmp/oncall")}
        mock_cfg.default_workspace = "default"
        mock_cfg.default_memory_store = "default"
        mock_cfg.memory_stores = {"oncall-mem": MagicMock()}
        mock_cfg.memory = MagicMock()

        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/tmp/oncall")
        mock_bindings.memory_store_name = "oncall-mem"
        mock_bindings.model = ""

        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat._workspace_name_for_dir",
            lambda cfg, ws_dir: "oncall-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "oncall-ws",
        )

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "oncall"})
            data = await resp.json()
            assert resp.status == 200
            assert data["ok"] is True
            assert data["agent"] == "oncall"
            assert "workspace" in data
            assert data["workspace"] == "oncall-ws"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_failed_resolution_reports_slot_workspace(
        self, tmp_path, monkeypatch
    ):
        """A binding-resolution failure must not fabricate workspace='default'.

        The handler logs and proceeds when resolve_agent_bindings raises, and
        slot.workspace keeps its previous value — so the response must name
        THAT value. The acting tab writes the response's workspace
        into its store optimistically; a fabricated 'default' would pin the
        chip to a workspace the slot does not hold, and with the websocket
        down (the optimistic write's whole premise) nothing corrects it.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.workspace = "research-ws"
        state.sessions.reset = AsyncMock()

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", _boom)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", _boom)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "oncall"})
            data = await resp.json()
            assert resp.status == 200
            assert data["ok"] is True
            assert data["agent"] == "oncall"
            # The slot's workspace was not changed by the failed resolution,
            # and the response reports the value the slot actually holds.
            assert slot.workspace == "research-ws"
            assert data["workspace"] == "research-ws"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_reset_failure_keeps_committed_agent(
        self, tmp_path, monkeypatch
    ):
        """A throwing teardown reports SUCCESS with a warning; the agent stays.

        The reset pops the session before its shutdown can fail, so by the
        failure point the old binding's session does not exist — every
        future or replacement session cold-starts on the NEW agent. The
        switch therefore succeeded: answering 500 would make the acting
        tab's performSlotSwitch keep the OLD store value for a switch that
        actually happened.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "oncall"
        slot.workspace = "oncall-ws"
        slot.project = "/tmp/oncall"
        old = MagicMock()
        old.has_active_turn.return_value = False
        state.sessions.get_provider = MagicMock(return_value=old)

        async def _pop_then_raise(*_a, **_k):
            # The canonical post-pop shape: the provider is deregistered
            # before the shutdown raises, so the helper's identity probe
            # classifies the raise as post-pop and the handler answers the
            # committed switch.
            state.sessions.get_provider = MagicMock(return_value=None)
            raise RuntimeError("shutdown blew up")

        state.sessions.reset = AsyncMock(side_effect=_pop_then_raise)

        mock_cfg = MagicMock()
        mock_cfg.agents = {"research": MagicMock(workspace="research-ws", memory_store="default")}
        mock_cfg.workspaces = {"research-ws": MagicMock(dir="/tmp/research")}
        mock_cfg.default_workspace = "default"
        mock_cfg.default_memory_store = "default"
        mock_cfg.memory_stores = {}
        mock_cfg.memory = MagicMock()
        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/tmp/research")
        mock_bindings.memory_store_name = "default"
        mock_bindings.model = ""
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            data = await resp.json()
            # Success with a warning — the switch happened; only the old
            # process's teardown degraded.
            assert resp.status == 200
            assert data["ok"] is True
            assert data["agent"] == "research"
            assert data["warning"] == "old session teardown incomplete"
            # The committed bindings stay (the popped old session cannot come
            # back) — agent and the derived workspace both advertise the new
            # binding.
            assert slot.agent == "research"
            assert slot.workspace == "research-ws"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_failed_reset_spares_concurrent_writes(
        self, tmp_path, monkeypatch
    ):
        """A failed switch never disturbs values written while it was in flight.

        The handler commits before the reset and its failure path touches
        nothing further, so state
        written by a concurrent actor during the (failing) reset — e.g. the
        project endpoint, which does not take the slot lock — survives
        untouched. Restoring captured priors here (the old rollback shape)
        would silently erase that concurrent success.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "oncall"
        slot.workspace = "oncall-ws"
        slot.project = "/tmp/oncall"

        async def _concurrent_switch_lands_then_reset_fails(*args, **kwargs):
            # Simulate a concurrent request winning the race while this
            # request's reset is in flight.
            slot.agent = "writer"
            slot.workspace = "writing-ws"
            slot.project = "/tmp/writing"
            raise RuntimeError("shutdown blew up")

        state.sessions.reset = AsyncMock(side_effect=_concurrent_switch_lands_then_reset_fails)
        # No live session was ever registered for the key, so the helper's
        # identity probe classifies the raise as post-pop and answers the
        # committed switch rather than a 500.
        state.sessions.get_provider = MagicMock(return_value=None)

        mock_cfg = MagicMock()
        mock_cfg.agents = {"research": MagicMock(workspace="research-ws", memory_store="default")}
        mock_cfg.workspaces = {"research-ws": MagicMock(dir="/tmp/research")}
        mock_cfg.default_workspace = "default"
        mock_cfg.default_memory_store = "default"
        mock_cfg.memory_stores = {}
        mock_cfg.memory = MagicMock()
        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/tmp/research")
        mock_bindings.memory_store_name = "default"
        mock_bindings.model = ""
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            assert resp.status == 200
            # The concurrent winner's values survive — no field snaps back to
            # this request's priors.
            assert slot.agent == "writer"
            assert slot.workspace == "writing-ws"
            assert slot.project == "/tmp/writing"

    def _patch_agent_resolution(self, monkeypatch):
        """Stub config + binding resolution so the agent switch reaches its reset."""
        mock_cfg = MagicMock()
        mock_cfg.agents = {"research": MagicMock(workspace="research-ws", memory_store="default")}
        mock_cfg.workspaces = {"research-ws": MagicMock(dir="/tmp/research")}
        mock_cfg.default_workspace = "default"
        mock_cfg.default_memory_store = "default"
        mock_cfg.memory_stores = {}
        mock_cfg.memory = MagicMock()
        mock_bindings = ResolvedBindings(
            workspace_dir=Path("/tmp/research"),
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/tmp/research")
        mock_bindings.memory_store_name = "default"
        mock_bindings.model = ""
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_reset_raise_before_pop_propagates(
        self, tmp_path, monkeypatch
    ):
        """A raise with the session STILL REGISTERED propagates as a 500.

        The SAME provider instance registered before and after the raise
        means the failure came BEFORE the session pop (e.g. in the
        pending-wait unblock): the old session survives on the old agent, so
        a 200 + warning would report a switch that did not take. The
        committed values are rolled back before the raise escapes — the
        acting tab keeps its old store value on a non-2xx, and the probe has
        proven the surviving session still runs the old binding.
        """
        from kiro_crew.providers.base import LLMProvider

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "oncall"
        slot.model = "oncall-model"
        slot._model_pick_gen = 3
        slot._active_fallback_model = "oncall-fallback"
        slot._fallback_primary_model = "oncall-model"
        slot._fallback_slot_model = "oncall-model"
        slot._fallback_pick_gen = 3
        alive = MagicMock(spec=LLMProvider)
        alive.has_active_turn.return_value = False
        state.sessions.get_provider = MagicMock(return_value=alive)
        state.sessions.reset = AsyncMock(side_effect=RuntimeError("pre-pop boom"))
        self._patch_agent_resolution(monkeypatch)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            assert resp.status == 500
            assert slot.agent == "oncall"
            assert slot.model == "oncall-model"
            assert slot._model_pick_gen == 3
            assert slot._active_fallback_model == "oncall-fallback"
            assert slot._fallback_primary_model == "oncall-model"
            assert slot._fallback_slot_model == "oncall-model"
            assert slot._fallback_pick_gen == 3

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_reset_raise_with_successor_succeeds(
        self, tmp_path, monkeypatch
    ):
        """A successor registered post-pop is not the unpopped old session.

        A concurrent send can register a SUCCESSOR session for the same key
        after the pop and before the old session's shutdown raises: the
        helper's probe compares instance IDENTITY, so a different registered
        provider still classifies as post-pop — the switch is committed, the
        successor cold-started from the committed bindings, and the answer is
        200 + warning.
        """
        from kiro_crew.providers.base import LLMProvider

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "oncall"
        old = MagicMock(spec=LLMProvider)
        old.has_active_turn.return_value = False
        state.sessions.get_provider = MagicMock(return_value=old)

        async def _pop_register_successor_and_raise(*_a, **_k):
            successor = MagicMock(spec=LLMProvider)
            successor.has_active_turn.return_value = False
            state.sessions.get_provider = MagicMock(return_value=successor)
            raise RuntimeError("shutdown boom")

        state.sessions.reset = AsyncMock(side_effect=_pop_register_successor_and_raise)
        self._patch_agent_resolution(monkeypatch)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            data = await resp.json()
            assert resp.status == 200
            assert data["ok"] is True
            assert data["agent"] == "research"
            assert data["warning"] == "old session teardown incomplete"
            assert slot.agent == "research"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_success_commit_spares_concurrent_pick(
        self, tmp_path, monkeypatch
    ):
        """A pick landing during a SUCCESSFUL reset wins over derived values.

        The project/workspace endpoints do not take the slot lock, so a
        user's explicit pick can land while the agent switch's reset await is
        in flight. The commit is compare-and-set against a pre-await
        baseline: the switch's DERIVED workspace/project must not overwrite
        the newer explicit pick, and the response names the slot's final
        reality. `agent` itself still commits — it is owned exclusively by
        this endpoint.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "oncall"
        slot.workspace = "oncall-ws"
        slot.project = "/tmp/oncall"
        seen_during_reset: list[tuple[str, str]] = []

        async def _concurrent_pick_lands_then_reset_succeeds(*args, **kwargs):
            # The derived bindings must already be committed when the reset
            # runs: a send in this window cold-starts the replacement session
            # from the slot's CURRENT values, and starting the new agent in
            # the old project would run its tools in the wrong repository.
            seen_during_reset.append((slot.workspace, slot.project))
            # A project pick (unlocked endpoint) lands mid-reset.
            slot.project = "/tmp/user-picked"
            return True

        state.sessions.reset = AsyncMock(side_effect=_concurrent_pick_lands_then_reset_succeeds)

        mock_cfg = MagicMock()
        mock_cfg.agents = {"research": MagicMock(workspace="research-ws", memory_store="default")}
        mock_cfg.workspaces = {"research-ws": MagicMock(dir="/tmp/research")}
        mock_cfg.default_workspace = "default"
        mock_cfg.default_memory_store = "default"
        mock_cfg.memory_stores = {}
        mock_cfg.memory = MagicMock()
        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/tmp/research")
        mock_bindings.memory_store_name = "default"
        mock_bindings.model = ""
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            data = await resp.json()
            assert resp.status == 200
            # The agent itself committed; the derived workspace committed
            # (untouched since baseline); the user's mid-reset project pick
            # SURVIVED the commit instead of being overwritten by the
            # derived project reset.
            assert slot.agent == "research"
            assert slot.workspace == "research-ws"
            assert slot.project == "/tmp/user-picked"
            # The derived workspace was already committed when the reset ran
            # (a replacement session in that window starts under the NEW
            # binding, not the old repository).
            assert seen_during_reset[0][0] == "research-ws"
            # The response names the slot's final reality.
            assert data["workspace"] == "research-ws"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_new_agent_visible_during_reset(self, tmp_path, monkeypatch):
        """The new agent is committed before the reset runs.

        A message send landing while the reset await is in flight creates a
        fresh session from the slot's CURRENT bindings — if the agent were
        committed only after the reset, that session would cold-start on the
        OLD agent and stay stale while the switch reports success.
        (`agent` has no unlocked writers, so committing before the reset is
        safe: the failure path's rollback races nobody.)
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "oncall"
        seen_during_reset: list[str] = []

        async def _observe_then_succeed(*args, **kwargs):
            seen_during_reset.append(slot.agent)
            return True

        state.sessions.reset = AsyncMock(side_effect=_observe_then_succeed)

        def _boom():
            raise RuntimeError("config unreadable")

        # Resolution outcome is irrelevant to the visibility property.
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", _boom)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", _boom)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            assert resp.status == 200
            assert seen_during_reset == ["research"]
            assert slot.agent == "research"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_response_names_workspace_picked_during_persist(
        self, tmp_path, monkeypatch
    ):
        """The response snapshots the workspace LAST, before leaving the lock.

        The metadata persist awaits a worker thread, yielding the event loop
        — a concurrent /workspace pick landing in that window must be what
        the response names, because the acting tab writes the response's
        workspace into its store optimistically. A snapshot taken before the
        persist would hand the tab a stale value that the pick had already
        superseded.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.workspace = "oncall-ws"
        state.sessions.reset = AsyncMock()

        def _boom():
            raise RuntimeError("config unreadable")

        # Resolution outcome is irrelevant; keep the derived commit a no-op.
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", _boom)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", _boom)

        # A conversation_log whose update_metadata simulates a concurrent
        # /workspace pick landing while the persist thread runs.
        conv = MagicMock()

        def _pick_lands_during_persist(*args, **kwargs):
            slot.workspace = "user-picked-ws"

        conv.update_metadata = MagicMock(side_effect=_pick_lands_during_persist)
        state.conversation_log = conv

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            data = await resp.json()
            assert resp.status == 200
            # The response names the newest reality, not a pre-persist
            # snapshot.
            assert data["workspace"] == "user-picked-ws"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_updates_project_dir(self, tmp_path, monkeypatch):
        """Switching agent also updates slot.project to the new workspace dir.

        Without this, the dashboard file search stays scoped to the previous
        workspace even after the user selects a different agent.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.project = "/old/project"
        state.sessions.reset = AsyncMock()

        mock_cfg = MagicMock()
        mock_cfg.agents = {"dev": MagicMock(workspace="dev-ws", memory_store="default")}
        mock_cfg.workspaces = {"dev-ws": MagicMock(dir="/workspace/dev")}
        mock_cfg.default_workspace = "default"
        mock_cfg.default_memory_store = "default"
        mock_cfg.memory_stores = {"default": MagicMock()}
        mock_cfg.memory = MagicMock()

        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/workspace/dev")
        mock_bindings.memory_store_name = "default"
        mock_bindings.model = ""

        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat._workspace_name_for_dir",
            lambda cfg, ws_dir: "dev-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "dev-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.default_project_dir",
            lambda ws: "/workspace/dev",
        )

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "dev"})
            assert resp.status == 200
            assert slot.project == "/workspace/dev"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_keeps_project_for_project_agent(self, tmp_path, monkeypatch):
        """Selecting a PROJECT-scope agent must not reset slot.project.

        kiro-cli resolves --agent against $PWD/.kiro/agents, so clobbering the
        project here makes the just-selected agent unresolvable on the next
        turn: the slot advertises it while the default answers — the
        silent-substitution bug this exists to remove.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.project = str(tmp_path / "repo")
        state.sessions.reset = AsyncMock()

        mock_cfg = MagicMock()
        mock_cfg.agents = {}  # not an alias — resolvable only via the project scope

        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/workspace/default")
        mock_bindings.requested_resolved = True

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "default",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.warm_project_agent_names", AsyncMock()
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.cached_project_agent_names",
            lambda project_dir: frozenset({"repo-bot"}),
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.default_project_dir",
            lambda ws: "/workspace/default",
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "repo-bot"})
            assert resp.status == 200
            assert slot.project == str(
                tmp_path / "repo"
            ), f"project agent selection clobbered slot.project: {slot.project!r}"

    @pytest.mark.asyncio
    async def test_api_chat_slot_workspace_updates_project_dir(self, tmp_path, monkeypatch):
        """Switching workspace also updates slot.project to the new workspace dir."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.project = "/old/project"
        state.sessions.reset = AsyncMock()

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.default_project_dir",
            lambda ws: "/workspace/new-ws",
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/workspace", json={"workspace": "new-ws"})
            assert resp.status == 200
            assert slot.project == "/workspace/new-ws"

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_resets_fallback_and_persists_bindings(
        self, tmp_path, monkeypatch
    ):
        """Switching agents resets provider-specific state and persists both bindings.

        The pin is KEPT (the read side withholds it from a provider that does
        not serve it); the fallback walk, which nothing re-scopes at read, is
        reset. Without the persistence, a session resumed after a gateway
        restart reverts to whatever agent (if any) was recorded in the initial
        metadata line.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "old-agent"
        slot.model = "old-provider-model"
        slot._model_pick_gen = 4
        slot._fallback_candidate_idx = 2
        slot._fallback_walked = ["old-fallback"]
        slot._active_fallback_model = "old-fallback"
        slot._fallback_primary_model = "old-provider-model"
        slot._fallback_slot_model = "old-provider-model"
        slot._fallback_pick_gen = 4
        state.sessions.reset = AsyncMock()

        # Seed a session file so update_metadata has something to patch.
        # Use the canonical colon-separated key that the API handler derives
        # via _history_key_for("s1") → "dashboard:s1".  Using "dashboard_s1"
        # (underscore) maps to the same *file* on disk (_safe_key converts
        # both to "dashboard_s1.jsonl") but creates a different *cache key*,
        # so update_metadata's cache invalidation for "dashboard:s1" would
        # leave the "dashboard_s1" cache entry stale.
        history_key = "dashboard:s1"
        state.conversation_log.append(history_key, "user", "hi", agent="old-agent")
        state.conversation_log.update_metadata(history_key, {"model": "old-provider-model"})
        assert state.conversation_log.get_metadata(history_key).get("agent") == "old-agent"

        # Minimal config stub (agent-binding resolution is exercised by the
        # workspace-focused test above; here we only care about persistence).
        mock_cfg = MagicMock()
        mock_cfg.agents = {}
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)

        async with TestClient(TestServer(as_owner(_make_app_with_agent_routes(state)))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "new-agent"})
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True
            assert data["agent"] == "new-agent"

        meta = state.conversation_log.get_metadata(history_key)
        assert (
            meta.get("agent") == "new-agent"
        ), f"expected new-agent in metadata, got {meta.get('agent')!r}"
        assert meta.get("model") == "old-provider-model"
        assert slot.model == "old-provider-model"
        assert slot._model_pick_gen == 4
        assert slot._fallback_candidate_idx == 0
        assert slot._fallback_walked == []
        assert slot._active_fallback_model == ""
        assert slot._fallback_primary_model == ""
        assert slot._fallback_slot_model == ""
        assert slot._fallback_pick_gen == 0

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_same_agent_keeps_model_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "same-agent"
        slot.model = "same-provider-model"
        slot._model_pick_gen = 4
        slot._active_fallback_model = "same-fallback"
        slot._fallback_primary_model = "same-provider-model"
        slot._fallback_slot_model = "same-provider-model"
        slot._fallback_pick_gen = 4
        state.sessions.reset = AsyncMock()

        mock_cfg = MagicMock()
        mock_cfg.agents = {}
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        # The handler imports the owner check locally from source_providers, so
        # that is the namespace to patch; short-circuiting it keeps this test on
        # the same-agent path instead of the binding-resolution refusal.
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: True,
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "same-agent"})
            assert resp.status == 200

        assert slot.model == "same-provider-model"
        assert slot._model_pick_gen == 4
        assert slot._active_fallback_model == "same-fallback"
        assert slot._fallback_primary_model == "same-provider-model"
        assert slot._fallback_slot_model == "same-provider-model"
        assert slot._fallback_pick_gen == 4

    @pytest.mark.asyncio
    async def test_api_chat_slot_agent_same_name_kind_switch_clears_the_binding(
        self, tmp_path, monkeypatch
    ):
        """A same-name member->template pick is a rebind, not an affirmation.

        The pin is kept (the read side withholds it from the replacement
        provider); only the fallback walk, which nothing re-scopes at read,
        is reset. Only the kind distinguishes the two picks, so the reset
        keys on it as well as the name.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "same-agent"
        slot.agent_kind = "member"
        slot.model = "member-provider-model"
        slot._model_pick_gen = 4
        slot._active_fallback_model = "member-fallback"
        slot._fallback_primary_model = "member-provider-model"
        slot._fallback_slot_model = "member-provider-model"
        slot._fallback_pick_gen = 4
        state.sessions.reset = AsyncMock()

        mock_cfg = MagicMock()
        mock_cfg.agents = {}
        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request",
            lambda request: True,
        )
        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/agent", json={"agent": "same-agent", "agent_kind": "template"}
            )
            assert resp.status == 200

        assert slot.agent_kind == "template"
        assert slot.model == "member-provider-model"
        assert slot._model_pick_gen == 4
        assert slot._active_fallback_model == ""
        assert slot._fallback_primary_model == ""
        assert slot._fallback_slot_model == ""
        assert slot._fallback_pick_gen == 0

    @pytest.mark.asyncio
    async def test_rolled_back_owner_switch_persists_the_restored_model(
        self, tmp_path, monkeypatch
    ):
        """A refused agent switch preserves the pin in memory and metadata."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.agent = "oncall"
        slot.model = "pinned-model"
        slot._active_fallback_model = "old-fallback"
        slot.workspace = "oncall-ws"
        slot.project = "/tmp/oncall"
        state.conversation_log.append("dashboard:s1", "user", "hi", agent="oncall")
        state.conversation_log.update_metadata("dashboard:s1", {"model": "pinned-model"})

        state.sessions.reset = AsyncMock()
        state.sessions.get_provider = MagicMock(return_value=None)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._record_explicit_agent_selection",
            AsyncMock(side_effect=ValueError("member memory write failed")),
        )
        self._patch_agent_resolution(monkeypatch)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/agent", json={"agent": "research"})
            assert resp.status != 200

        assert slot.agent == "oncall"
        assert slot.model == "pinned-model"
        metadata = state.conversation_log.get_metadata("dashboard:s1")
        assert metadata.get("agent") == "oncall"
        assert metadata.get("model") == "pinned-model"

    @pytest.mark.asyncio
    async def test_api_chat_slot_create_response_includes_workspace(self, tmp_path, monkeypatch):
        """api_chat_slot_create response includes resolved workspace field.

        Requirements: 2.4
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        mock_cfg = MagicMock()
        mock_cfg.agents = {"research": MagicMock(workspace="research-ws", memory_store="default")}
        mock_cfg.workspaces = {"research-ws": MagicMock(dir="/tmp/research")}
        mock_cfg.default_workspace = "default"
        mock_cfg.default_memory_store = "default"
        mock_cfg.memory_stores = {}
        mock_cfg.memory = MagicMock()

        mock_bindings = ResolvedBindings(
            workspace_dir=tmp_path,
            memory_store_name="",
            effective_memory_config={},
            kiro_agent="kirocrew",
            selection_kind="template",
        )
        mock_bindings.workspace_dir = Path("/tmp/research")
        mock_bindings.memory_store_name = "default"
        mock_bindings.model = ""

        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._workspace_name_for_dir",
            lambda cfg, ws_dir: "research-ws",
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post(
                "/api/chat/slots",
                json={"name": "new-slot", "agent": "research"},
            )
            data = await resp.json()
            assert resp.status == 200
            assert data["workspace"] == "research-ws"

    def test_get_or_create_slot_accepts_workspace_parameter(self, tmp_path, monkeypatch):
        """get_or_create_slot accepts workspace parameter and sets it on the slot.

        Requirements: 2.3
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        slot = state.get_or_create_slot("ws-test", agent="oncall", workspace="oncall-ws")
        assert slot.workspace == "oncall-ws"
        assert slot.agent == "oncall"

        # Default workspace when not specified
        slot2 = state.get_or_create_slot("ws-default")
        assert slot2.workspace == "default"

        # Mode parameter
        slot3 = state.get_or_create_slot("mode-test", mode="design-critique")
        assert slot3.mode == "design-critique"
        assert state.get_or_create_slot("ws-default").mode == ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("binding_change", ["none", "project", "replacement", "unavailable"])
    async def test_run_chat_passes_memory_store_to_build_message(
        self, tmp_path, monkeypatch, binding_change
    ):
        """_run_chat resolves agent bindings and passes memory_store to build_message.

        Requirements: 3.1
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        # Track calls to build_message
        build_message_calls: list[dict] = []

        def mock_build_message(self_ctx, text, is_new, session_key=None, **kwargs):
            build_message_calls.append({"text": text, "session_key": session_key, "kwargs": kwargs})
            return text, MagicMock(action=None, text="")

        from kiro_crew.config.loader import (
            KiroCrewAgentConfig,
            KiroCrewConfig,
            WorkspaceConfig,
            resolve_agent_bindings,
        )
        from kiro_crew.memory_stores import provision_member_memory

        mock_cfg = KiroCrewConfig.load()
        mock_cfg.agents["oncall"] = KiroCrewAgentConfig(
            kiro_agent="kirocrew", workspace="oncall-ws"
        )
        mock_cfg.workspaces["oncall-ws"] = WorkspaceConfig(dir=str(tmp_path / "oncall-ws"))
        private_store = provision_member_memory(mock_cfg, "oncall")
        mock_cfg.save()
        mock_bindings = resolve_agent_bindings(mock_cfg, "oncall")
        # The provider and context are doubles; model a supported runtime while
        # keeping member ownership and persisted conversation metadata real.

        monkeypatch.setattr("kiro_crew.dashboard.chat.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.KiroCrewConfig.load", lambda: mock_cfg)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner.resolve_agent_bindings",
            lambda cfg, name, project_dir=None, **kwargs: mock_bindings,
        )

        # Create a context builder with mocked build_message
        from kiro_crew.context import ContextBuilder
        from kiro_crew.memory import MemoryStore
        from kiro_crew.skills import SkillsLoader

        ctx_builder = ContextBuilder(
            memory=MemoryStore(workspace=tmp_path / "ws"),
            skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        )
        monkeypatch.setattr(
            ctx_builder, "build_message", lambda *a, **kw: mock_build_message(ctx_builder, *a, **kw)
        )
        monkeypatch.setattr(ctx_builder, "ensure_store", AsyncMock(return_value=object()))
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.title_then_refresh", AsyncMock())
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.generate_session_summary", AsyncMock())

        state = _make_state(tmp_path, context_builder=ctx_builder)

        # Create a slot with retained history, then verify a cold start replays it.
        slot = state.get_or_create_slot("mem-test", agent="oncall")
        if binding_change == "none":
            from kiro_crew.execution_context import bind_session_execution, resolve_member_execution

            captured_execution = await asyncio.to_thread(
                resolve_member_execution, mock_cfg, "oncall"
            )
            await asyncio.to_thread(
                bind_session_execution, "dashboard:mem-test", captured_execution
            )
        conversation_log = ConversationLog(base_dir=tmp_path / "sessions")
        conversation_log.init()
        await asyncio.to_thread(
            conversation_log.append, "dashboard:mem-test", "user", "frozen retained question"
        )
        await asyncio.to_thread(
            conversation_log.append,
            "dashboard:mem-test",
            "assistant",
            "frozen retained answer",
        )
        await asyncio.to_thread(
            conversation_log.append, "dashboard:mem-test", "user", "test message"
        )
        ctx_builder.conversation_log = conversation_log
        state.conversation_log = conversation_log
        slot.append("user", "retained question", "msg msg-u")
        slot.append("assistant", "retained answer", "msg msg-a")
        slot.append("user", "test message", "msg msg-u")

        # Mock session manager to return a mock client
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            return_value=AsyncIterator(
                [LLMEvent(kind=EVENT_TEXT_CHUNK, text="done"), LLMEvent(kind=EVENT_COMPLETE)]
            )
        )
        state.sessions.get_or_create = AsyncMock(return_value=(mock_client, True, False))
        state.sessions.get_pid = MagicMock(return_value=None)
        state.sessions.consume_replay_suppression = MagicMock(return_value=False)
        state.sessions.record_failure = AsyncMock()

        loop = asyncio.get_running_loop()
        binding_reads = []

        def checked_bindings(cfg, name, project_dir=None, **kwargs):
            from kiro_crew.memory_stores import UnknownMemoryStore

            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.get_running_loop()
            assert kwargs["validate_memory_files"] is False
            binding_reads.append((name, project_dir))
            if binding_change == "unavailable":
                raise UnknownMemoryStore("member identity offline")
            resolved = resolve_agent_bindings(cfg, name, project_dir, **kwargs)
            if binding_change == "project":
                loop.call_soon_threadsafe(setattr, slot, "project", str(tmp_path / "new-project"))
            elif binding_change == "replacement":
                loop.call_soon_threadsafe(state._slots.__setitem__, slot.key, _ChatSlot(slot.key))
            return resolved

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner.resolve_agent_bindings", checked_bindings
        )

        # Import and run _run_chat
        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "test message")
        await asyncio.gather(*state._background_tasks)

        assert len(binding_reads) == 1
        assert binding_reads[0][0] == "oncall"
        if binding_change != "none":
            from kiro_crew.member_memory_auth import read_private_session_store

            state.sessions.get_or_create.assert_not_awaited()
            state.sessions.record_failure.assert_awaited_once_with("dashboard:mem-test")
            assert build_message_calls == []
            assert read_private_session_store("dashboard:mem-test") is None
            assert any(
                message.get("role") == "error"
                and message.get("meta", {}).get("code") == "memory_unavailable"
                for message in slot.messages
            )
            return

        # Verify build_message was called with memory_store
        state.sessions.record_failure.assert_not_awaited()
        assert len(build_message_calls) == 1
        assert build_message_calls[0]["kwargs"].get("memory_store") == private_store
        assert build_message_calls[0]["session_key"] == "dashboard:mem-test"
        metadata = conversation_log.get_metadata("dashboard:mem-test")
        assert metadata["memory_store"] == private_store
        assert metadata["agent"] == "oncall"
        from kiro_crew.execution_context import read_session_execution

        retained_execution = read_session_execution("dashboard:mem-test")
        assert retained_execution.member_id == captured_execution.member_id
        assert retained_execution.store == captured_execution.store
        assert retained_execution.template_id == captured_execution.template_id
        replay = build_message_calls[0]["kwargs"].get("compressed_history")
        assert "frozen retained question" in replay
        assert "frozen retained answer" in replay

    @pytest.mark.parametrize(
        ("message_kind", "resumed", "expect_restore"),
        [
            ("manual_resume", True, True),
            ("conn_recover", True, True),
            ("busy_recover", True, True),
            # A cold start that did NOT natively resume carries the Kiro Crew
            # replay, which already holds the interrupted turn.
            ("manual_resume", False, False),
            # Continue on a turn that ended normally: kiro-cli logged it.
            ("manual_continue", True, False),
            # A cron delivery opened the interrupted turn as an ``inject``.
            ("cron_opener", True, True),
        ],
    )
    @pytest.mark.asyncio
    async def test_resume_on_a_natively_loaded_session_restores_the_interrupted_turn(
        self, tmp_path, monkeypatch, message_kind, resumed, expect_restore
    ):
        """kiro-cli logs a prompt only once it is answered, so a turn cut off by
        the process dying is absent from what ``session/load`` restores. A
        resumed session gets no replay, so the Resume press on that turn must
        carry the turn itself, or the model is told to finish a request it cannot
        see and answers the one before it (or reports nothing to continue)."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        from kiro_crew.context import ContextBuilder
        from kiro_crew.dashboard.chat_utils import (
            _BUSY_RECOVER_MSG,
            _CONN_RECOVER_MSG,
            _MANUAL_CONTINUE_MSG,
            _MANUAL_RESUME_MSG,
        )
        from kiro_crew.memory import MemoryStore
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent
        from kiro_crew.skills import SkillsLoader

        resume_text = {
            "manual_resume": _MANUAL_RESUME_MSG,
            "conn_recover": _CONN_RECOVER_MSG,
            "busy_recover": _BUSY_RECOVER_MSG,
            "manual_continue": _MANUAL_CONTINUE_MSG,
            "cron_opener": _MANUAL_RESUME_MSG,
        }[message_kind]
        built: list[str] = []

        def mock_build_message(text, context_is_new, session_key=None, **kwargs):
            built.append(text)
            return text, MagicMock(action=None, text="")

        ctx_builder = ContextBuilder(
            memory=MemoryStore(workspace=tmp_path / "ws"),
            skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        )
        ctx_builder.conversation_log = MagicMock()
        monkeypatch.setattr(ctx_builder, "build_message", mock_build_message)
        monkeypatch.setattr("kiro_crew.context.build_session_replay", lambda *a, **k: "")

        state = _make_state(tmp_path, context_builder=ctx_builder)
        hook_store = MagicMock()
        hook_store.fire = AsyncMock(return_value=[])
        state._hook_store = hook_store
        slot = state.get_or_create_slot("resume-native")
        slot.append("user", "Earlier request, answered long ago")
        slot.append("assistant", "Earlier answer")
        if message_kind == "cron_opener":
            slot.append(
                "inject",
                "Summarise the incident timeline for TICKET-42",
                meta={"injectKind": "cron"},
            )
        else:
            slot.append("user", "Summarise the incident timeline for TICKET-42")
        slot.append("assistant", "Pulling the timeline now; first event at 09:14")
        resume_row = slot.append("inject", resume_text, meta={"injectKind": "recovery"})

        streamed: list[str] = []

        async def stream(stream_message):
            streamed.append(stream_message)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

        mock_client = MagicMock()
        mock_client.stream = stream
        mock_client.stream_command = stream
        mock_client.context_usage_pct = MagicMock(return_value=10.0)
        state.sessions.get_or_create = AsyncMock(return_value=(mock_client, True, resumed))
        state.sessions.get_pid = MagicMock(return_value=None)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(
            state,
            slot,
            resume_text,
            _current_message=resume_row,
            _synthetic_recovery_turn=True,
        )

        assert len(built) == 1 and len(streamed) == 1
        sent = streamed[0]
        # Minted after the egress scrub, as a prefix of the final prompt: the
        # builder's own input stays the bare recovery text.
        assert built[0] == resume_text
        assert sent.startswith("[INTERRUPTED TURN") is expect_restore
        assert ("Summarise the incident timeline for TICKET-42" in sent) is expect_restore
        assert ("first event at 09:14" in sent) is expect_restore
        # Only the interrupted turn is restored, never the one before it.
        assert "Earlier request, answered long ago" not in sent

    @pytest.mark.parametrize(
        ("accepts_inbound", "expected_channel"),
        [(True, "chat-42"), (False, None)],
    )
    @pytest.mark.asyncio
    async def test_run_chat_passes_only_resumable_mirror_channel_to_provider(
        self, tmp_path, monkeypatch, accepts_inbound, expected_channel
    ):
        """Only an inbound-capable persisted mirror identifies a dispatcher.

        Telegram/Discord mirrors can reuse a ``dashboard:*`` key while their
        identity exists only in ``SessionMap.mirror``. Two-way links must retain
        native resume; outbound-only mirrors still run as direct dashboard turns
        and need the Tool Search resume workaround.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        from kiro_crew.messaging.link import ChannelLink
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        state = _make_state(tmp_path)
        state.context_builder = None
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("linked-provider")
        slot._slack_channel = ""
        state.sessions.get_mirror_link = MagicMock(
            return_value=ChannelLink("telegram", channel_id="chat-42", thread_id="thread-7")
        )
        state.sessions.mirror_accepts_inbound = MagicMock(return_value=accepts_inbound)

        async def stream(_message):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

        mock_client = MagicMock()
        mock_client.stream = stream
        mock_client.stream_command = stream
        mock_client.context_usage_pct = MagicMock(return_value=10.0)
        state.sessions.get_or_create = AsyncMock(return_value=(mock_client, True, False))
        state.sessions.get_pid = MagicMock(return_value=None)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "linked dashboard turn")

        assert state.sessions.get_or_create.await_args.kwargs["channel_id"] == expected_channel

    @pytest.mark.asyncio
    async def test_first_slash_preserves_pending_replay_for_next_prompt(
        self, tmp_path, monkeypatch
    ):
        """A fresh session's slash command bypasses ContextBuilder entirely.

        The command may consume SessionManager's one-shot ``is_new`` observation,
        but it must not consume the conversation replay debt. Pre-dispatch Stop
        leaves it armed; an accepted turn consumes it on the first provider event;
        and a cancelled terminal re-arms it because the provider discards that
        turn. The next accepted ordinary prompt finally clears the lease.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        build_message_calls: list[dict] = []
        build_behavior = {"stop_during_build": False}

        def mock_build_message(text, context_is_new, session_key=None, **kwargs):
            build_message_calls.append(
                {
                    "text": text,
                    "context_is_new": context_is_new,
                    "session_key": session_key,
                    "kwargs": kwargs,
                }
            )
            if build_behavior["stop_during_build"]:
                slot._stop_generation += 1
                build_behavior["stop_during_build"] = False
            return text, MagicMock(action=None, text="")

        from kiro_crew.acp.types import STOP_REASON_CANCELLED
        from kiro_crew.context import ContextBuilder
        from kiro_crew.hooks import HOOK_EVENT_AGENT_SPAWN
        from kiro_crew.memory import MemoryStore
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent
        from kiro_crew.skills import SkillsLoader

        ctx_builder = ContextBuilder(
            memory=MemoryStore(workspace=tmp_path / "ws"),
            skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        )
        ctx_builder.conversation_log = MagicMock()
        monkeypatch.setattr(ctx_builder, "build_message", mock_build_message)
        monkeypatch.setattr(
            "kiro_crew.context.build_session_replay",
            lambda *args, **kwargs: "retained conversation replay",
        )

        state = _make_state(tmp_path, context_builder=ctx_builder)
        hook_store = MagicMock()
        hook_store.fire = AsyncMock(return_value=[])
        state._hook_store = hook_store
        slot = state.get_or_create_slot("slash-replay")
        slot.folder_id = "folder-1"
        state.folder_breadcrumb = MagicMock(return_value="Workspace / Demo")
        persona_first_turn: list[bool] = []

        def inject_persona(message, _theme, context_is_new, **_kwargs):
            persona_first_turn.append(context_is_new)
            return message

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._maybe_inject_persona",
            inject_persona,
        )

        dispatched: list[str] = []
        stream_behavior = {
            "cancel_after_accept": False,
            "raise_after_accept": False,
            "synthetic_end_turn": False,
            "empty_end_turn": False,
        }

        async def stream(stream_message):
            dispatched.append(stream_message)
            if stream_message != "/tools":
                assert replay["pending"] is True
            if stream_behavior["cancel_after_accept"]:
                stream_behavior["cancel_after_accept"] = False
                yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED)
                return
            if stream_behavior["synthetic_end_turn"]:
                stream_behavior["synthetic_end_turn"] = False
                yield LLMEvent(
                    kind=EVENT_COMPLETE,
                    stop_reason="end_turn",
                    synthetic_completion=True,
                )
                return
            if stream_behavior["empty_end_turn"]:
                stream_behavior["empty_end_turn"] = False
                yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")
                return
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok")
            if stream_message != "/tools":
                assert replay["pending"] is True
            if stream_behavior["raise_after_accept"]:
                stream_behavior["raise_after_accept"] = False
                raise RuntimeError("stream failed after accepting replay")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

        mock_client = MagicMock()
        mock_client.stream = stream
        mock_client.stream_command = stream
        mock_client.context_usage_pct = MagicMock(return_value=10.0)
        state.sessions.get_or_create = AsyncMock(
            side_effect=[
                (mock_client, True, False),
                (mock_client, False, False),
                (mock_client, False, False),
                (mock_client, False, False),
                (mock_client, False, False),
                (mock_client, False, False),
                (mock_client, False, False),
                (mock_client, False, False),
            ]
        )
        state.sessions.get_pid = MagicMock(return_value=None)
        state.sessions.record_failure = AsyncMock()
        state.sessions.consume_replay_suppression = MagicMock(return_value=False)

        replay = {"pending": True}
        durable = {"sid": "old-full-history-sid"}

        def replay_pending(_key):
            return replay["pending"]

        def consume_replay(_key):
            was_pending = replay["pending"]
            replay["pending"] = False
            return was_pending

        def mark_replay(_key):
            replay["pending"] = True
            return True

        def commit_replay(_key):
            durable["sid"] = "fresh-replayed-sid"
            replay["pending"] = False
            return True

        state.sessions.provider_switch_replay_pending = MagicMock(side_effect=replay_pending)
        state.sessions.consume_provider_switch_replay = MagicMock(side_effect=consume_replay)
        state.sessions.mark_provider_switch_replay = MagicMock(side_effect=mark_replay)
        state.sessions.commit_provider_switch_replay_sid = MagicMock(side_effect=commit_replay)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "/tools")

        assert build_message_calls == []
        assert replay["pending"] is True
        assert durable["sid"] == "old-full-history-sid"
        assert dispatched == ["/tools"]
        state.sessions.consume_provider_switch_replay.assert_not_called()
        state.sessions.commit_provider_switch_replay_sid.assert_not_called()

        build_behavior["stop_during_build"] = True
        await _run_chat(state, slot, "stop during replay prep")

        assert len(build_message_calls) == 1
        assert build_message_calls[0]["context_is_new"] is True
        assert build_message_calls[0]["kwargs"]["folder_path"] == "Workspace / Demo"
        assert (
            build_message_calls[0]["kwargs"]["compressed_history"] == "retained conversation replay"
        )
        assert persona_first_turn == [True]
        assert replay["pending"] is True
        assert durable["sid"] == "old-full-history-sid"
        assert dispatched == ["/tools"]
        state.sessions.consume_provider_switch_replay.assert_not_called()
        state.sessions.commit_provider_switch_replay_sid.assert_not_called()

        stream_behavior["cancel_after_accept"] = True
        await _run_chat(state, slot, "cancel accepted replay")

        assert len(build_message_calls) == 2
        cancelled_call = build_message_calls[1]
        assert cancelled_call["context_is_new"] is True
        assert cancelled_call["kwargs"]["folder_path"] == "Workspace / Demo"
        assert cancelled_call["kwargs"]["compressed_history"] == "retained conversation replay"
        assert persona_first_turn == [True, True]
        assert dispatched == ["/tools", "cancel accepted replay"]
        assert replay["pending"] is True
        assert durable["sid"] == "old-full-history-sid"
        state.sessions.consume_provider_switch_replay.assert_not_called()
        state.sessions.mark_provider_switch_replay.assert_called_once_with("dashboard:slash-replay")
        state.sessions.commit_provider_switch_replay_sid.assert_not_called()

        stream_behavior["raise_after_accept"] = True
        await _run_chat(state, slot, "raise after accepted replay")

        assert len(build_message_calls) == 3
        assert replay["pending"] is True
        assert durable["sid"] == "old-full-history-sid"
        state.sessions.consume_provider_switch_replay.assert_not_called()
        assert state.sessions.mark_provider_switch_replay.call_count == 2
        state.sessions.commit_provider_switch_replay_sid.assert_not_called()

        stream_behavior["synthetic_end_turn"] = True
        slot._empty_response_retries = 2
        await _run_chat(state, slot, "synthetic replay completion")

        assert len(build_message_calls) == 4
        assert replay["pending"] is True
        assert durable["sid"] == "old-full-history-sid"
        state.sessions.consume_provider_switch_replay.assert_not_called()
        assert state.sessions.mark_provider_switch_replay.call_count == 3
        state.sessions.commit_provider_switch_replay_sid.assert_not_called()

        stream_behavior["empty_end_turn"] = True
        slot._empty_response_retries = 2
        await _run_chat(state, slot, "real empty replay with exhausted budget")

        assert len(build_message_calls) == 5
        assert replay["pending"] is True
        assert durable["sid"] == "old-full-history-sid"
        state.sessions.consume_provider_switch_replay.assert_not_called()
        assert state.sessions.mark_provider_switch_replay.call_count == 4
        state.sessions.commit_provider_switch_replay_sid.assert_not_called()

        stream_behavior["empty_end_turn"] = True
        slot._empty_response_retries = 1
        with patch(
            "kiro_crew.dashboard.chat_runner._empty_auto_continue_enabled",
            return_value=False,
        ):
            await _run_chat(state, slot, "real empty replay with continuation off")

        assert len(build_message_calls) == 6
        assert replay["pending"] is True
        assert durable["sid"] == "old-full-history-sid"
        state.sessions.consume_provider_switch_replay.assert_not_called()
        assert state.sessions.mark_provider_switch_replay.call_count == 5
        state.sessions.commit_provider_switch_replay_sid.assert_not_called()

        await _run_chat(state, slot, "continue after uncommitted replay turns")

        assert len(build_message_calls) == 7
        call = build_message_calls[6]
        assert call["context_is_new"] is True
        assert call["kwargs"]["folder_path"] == "Workspace / Demo"
        assert call["kwargs"]["compressed_history"] == "retained conversation replay"
        assert persona_first_turn == [True, True, True, True, True, True, True]
        assert dispatched == [
            "/tools",
            "cancel accepted replay",
            "raise after accepted replay",
            "synthetic replay completion",
            "real empty replay with exhausted budget",
            "real empty replay with continuation off",
            "continue after uncommitted replay turns",
        ]
        assert replay["pending"] is False
        assert durable["sid"] == "fresh-replayed-sid"
        state.sessions.consume_provider_switch_replay.assert_not_called()
        assert state.sessions.mark_provider_switch_replay.call_count == 5
        state.sessions.commit_provider_switch_replay_sid.assert_called_once_with(
            "dashboard:slash-replay"
        )
        agent_spawn_calls = [
            call
            for call in hook_store.fire.await_args_list
            if call.args and call.args[0] == HOOK_EVENT_AGENT_SPAWN
        ]
        assert len(agent_spawn_calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("native_flag", [False, True])
    async def test_resumed_observation_does_not_drop_an_owed_replay(
        self, tmp_path, monkeypatch, native_flag
    ):
        """A replay lease outranks a ``resumed`` observation.

        Allocation arms the lease only when the native session started fresh, as
        a Tool Search resume does when it replaces ``session/load``. If the turn
        also reads ``resumed=True`` (from the allocation tuple or the native
        client flag) it must still build a full first turn with the replay, not
        the slim resume that claims the history was restored.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        build_message_calls: list[dict] = []

        def mock_build_message(text, context_is_new, session_key=None, **kwargs):
            build_message_calls.append({"context_is_new": context_is_new, "kwargs": kwargs})
            return text, MagicMock(action=None, text="")

        from kiro_crew.context import ContextBuilder
        from kiro_crew.memory import MemoryStore
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent
        from kiro_crew.skills import SkillsLoader

        ctx_builder = ContextBuilder(
            memory=MemoryStore(workspace=tmp_path / "ws"),
            skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        )
        ctx_builder.conversation_log = MagicMock()
        monkeypatch.setattr(ctx_builder, "build_message", mock_build_message)
        monkeypatch.setattr(
            "kiro_crew.context.build_session_replay",
            lambda *args, **kwargs: "retained conversation replay",
        )

        state = _make_state(tmp_path, context_builder=ctx_builder)
        hook_store = MagicMock()
        hook_store.fire = AsyncMock(return_value=[])
        state._hook_store = hook_store
        slot = state.get_or_create_slot("resume-replay")

        async def stream(_message):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

        mock_client = MagicMock()
        mock_client.stream = stream
        mock_client.stream_command = stream
        mock_client.context_usage_pct = MagicMock(return_value=10.0)
        mock_client.client.resumed = native_flag
        state.sessions.get_or_create = AsyncMock(return_value=(mock_client, True, not native_flag))
        state.sessions.get_pid = MagicMock(return_value=None)
        state.sessions.record_failure = AsyncMock()
        state.sessions.consume_replay_suppression = MagicMock(return_value=False)
        state.sessions.provider_switch_replay_pending = MagicMock(return_value=True)
        state.sessions.consume_provider_switch_replay = MagicMock(return_value=True)
        state.sessions.mark_provider_switch_replay = MagicMock(return_value=True)
        state.sessions.commit_provider_switch_replay_sid = MagicMock(return_value=True)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "what were we doing?")

        assert len(build_message_calls) == 1
        call = build_message_calls[0]
        assert call["context_is_new"] is True
        assert call["kwargs"]["resumed"] is False
        assert call["kwargs"]["compressed_history"] == "retained conversation replay"

    @pytest.mark.asyncio
    async def test_run_chat_forwards_and_clears_the_reinjection_flag(self, tmp_path, monkeypatch):
        """A compaction flags the session; the NEXT _run_chat must forward
        needs_reinjection=True to build_message and clear the flag so the turn
        after that does not re-inject again.

        Regression guard: without this wiring the flag is set by the compact
        callback and read by nobody, so the whole feature is dead code.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        build_message_calls: list[dict] = []

        def mock_build_message(self_ctx, text, is_new, session_key=None, **kwargs):
            build_message_calls.append({"text": text, "kwargs": kwargs})
            return text, MagicMock(action=None, text="")

        from kiro_crew.context import ContextBuilder
        from kiro_crew.memory import MemoryStore
        from kiro_crew.skills import SkillsLoader

        ctx_builder = ContextBuilder(
            memory=MemoryStore(workspace=tmp_path / "ws"),
            skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        )
        monkeypatch.setattr(
            ctx_builder, "build_message", lambda *a, **kw: mock_build_message(ctx_builder, *a, **kw)
        )

        state = _make_state(tmp_path, context_builder=ctx_builder)
        slot = state.get_or_create_slot("reinject-test")

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=AsyncIterator([]))
        state.sessions.get_or_create = AsyncMock(return_value=(mock_client, True, False))
        state.sessions.get_pid = MagicMock(return_value=None)

        # Stand in for the real SessionManager flag store, including the
        # mark/consume round-trip so the empty-response re-queue path is
        # exercised the way production behaves.
        flag = {"set": True}

        def _consume(key):
            was = flag["set"]
            flag["set"] = False
            return was

        def _mark(key):
            flag["set"] = True

        state.sessions.consume_needs_reinjection = MagicMock(side_effect=_consume)
        state.sessions.mark_needs_reinjection = MagicMock(side_effect=_mark)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "first turn after compaction")

        # The stream is empty, so this turn does not land. The flag must be put
        # BACK -- asserting on the mark call directly, because asserting that some
        # build carried True is tautological (the first assertion already pins it,
        # and the re-queue is drained by an outer worker, not an inner loop).
        assert build_message_calls, "build_message must have been called"
        assert (
            build_message_calls[0]["kwargs"].get("needs_reinjection") is True
        ), "the turn right after a compaction must re-inject the skills index"
        assert state.sessions.mark_needs_reinjection.called, (
            "a turn that consumed the flag but did not land must restore it, "
            "or the skills index is lost for the rest of the session"
        )
        assert flag["set"] is True, "the flag must be back on after a non-landing turn"

        # Once a turn genuinely lands, the flag is gone and later turns are clean.
        flag["set"] = False
        build_message_calls.clear()
        await _run_chat(state, slot, "a later turn")
        assert build_message_calls, "build_message must have been called again"
        assert all(
            c["kwargs"].get("needs_reinjection") is False for c in build_message_calls
        ), "the flag must be one-shot, not sticky for every later turn"

    @pytest.mark.asyncio
    async def test_member_first_turn_that_never_lands_rearms_reinjection(
        self, tmp_path, monkeypatch
    ):
        """A member DM's FIRST turn that ends without landing — a user Stop's
        cancelled completion, an empty stream, a graceful cancel — never enters
        the except arm, so an except-arm-only re-arm leaves the warm session
        running WITHOUT [PERMANENT RULES] for the rest of its life. The re-arm
        must live in the finally, on every exit path."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)

        from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig
        from kiro_crew.context import ContextBuilder
        from kiro_crew.memory import MemoryStore
        from kiro_crew.skills import SkillsLoader

        ctx_builder = ContextBuilder(
            memory=MemoryStore(workspace=tmp_path / "ws"),
            skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        )
        monkeypatch.setattr(
            ctx_builder,
            "build_message",
            lambda text, is_new, session_key=None, **kw: (text, MagicMock(action=None, text="")),
        )

        state = _make_state(tmp_path, context_builder=ctx_builder)
        cfg = KiroCrewConfig()
        cfg.agents["rearm-test"] = KiroCrewAgentConfig(kiro_agent="kirocrew")
        cfg.default_agent = "rearm-test"
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.KiroCrewConfig.load", lambda: cfg)
        slot = state.get_or_create_slot("member-rearm-test", agent="rearm-test", mode="member")

        mock_client = MagicMock()
        # Empty stream: the turn ends with no landing and NO exception — the
        # same observable shape as a Stop's cancelled completion.
        mock_client.stream = MagicMock(return_value=AsyncIterator([]))
        # is_new=True: this is the member session's FIRST turn.
        state.sessions.get_or_create = AsyncMock(return_value=(mock_client, True, False))
        state.sessions.get_pid = MagicMock(return_value=None)
        # The compaction flag is OFF, so only the member condition can re-arm.
        state.sessions.consume_needs_reinjection = MagicMock(return_value=False)
        state.sessions.mark_needs_reinjection = MagicMock()

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "first member turn")

        assert state.sessions.mark_needs_reinjection.called, (
            "a member first turn that never landed must re-arm reinjection on "
            "EVERY exit path (finally), or the next warm turn runs without the "
            "member section and its [PERMANENT RULES]"
        )


class TestRunChatToolBoundarySegments:
    """Test that _run_chat inserts whitespace across tool call boundaries."""

    @pytest.mark.asyncio
    async def test_tool_boundary_splits_segments(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import LLMEvent

        monkeypatch.setattr("kiro_crew.dashboard.chat.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat.sel", lambda: MagicMock())

        state = _make_state(tmp_path)
        state.broadcast_ws = MagicMock()
        state._hook_store = None

        events = [
            LLMEvent(kind="text_chunk", text="Let me check."),
            LLMEvent(kind="tool_call", title="Read File", tool_kind="read"),
            LLMEvent(kind="text_chunk", text="Done!"),
            LLMEvent(kind="complete"),
        ]

        fake_client = _provider_mock()

        async def _stream(msg):
            for e in events:
                yield e

        fake_client.stream = _stream
        fake_client.context_usage_pct = MagicMock(return_value=0.0)
        state.sessions.get_or_create = AsyncMock(return_value=(fake_client, True, False))
        state.sessions.get_pid = MagicMock(return_value=None)
        state.sessions.check_context_usage = MagicMock()
        state.sessions.record_success = MagicMock()
        state.sessions.record_failure = AsyncMock()
        state.sessions.release = MagicMock()

        slot = state.get_or_create_slot("s1")
        await _run_chat(state, slot, "do it")

        assistant_msgs = [m for m in slot.messages if m.get("role") == "assistant"]
        # With _flush_segment, text is split into separate segments at tool boundaries
        # so gluing can't happen — each segment is independent
        assert len(assistant_msgs) == 2
        assert "Let me check." in assistant_msgs[0]["content"]
        assert "Done!" in assistant_msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_tool_boundary_empty_chunk_still_splits(self, tmp_path, monkeypatch):
        """Empty text chunk after tool call doesn't prevent segment splitting."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import LLMEvent

        monkeypatch.setattr("kiro_crew.dashboard.chat.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat.sel", lambda: MagicMock())

        state = _make_state(tmp_path)
        state.broadcast_ws = MagicMock()
        state._hook_store = None

        events = [
            LLMEvent(kind="text_chunk", text="Before."),
            LLMEvent(kind="tool_call", title="T", tool_kind="read"),
            LLMEvent(kind="text_chunk", text=""),  # empty chunk
            LLMEvent(kind="text_chunk", text="After!"),
            LLMEvent(kind="complete"),
        ]

        fake_client = _provider_mock()

        async def _stream(msg):
            for e in events:
                yield e

        fake_client.stream = _stream
        fake_client.context_usage_pct = MagicMock(return_value=0.0)
        state.sessions.get_or_create = AsyncMock(return_value=(fake_client, True, False))
        state.sessions.get_pid = MagicMock(return_value=None)
        state.sessions.check_context_usage = MagicMock()
        state.sessions.record_success = MagicMock()
        state.sessions.record_failure = AsyncMock()
        state.sessions.release = MagicMock()

        slot = state.get_or_create_slot("s1")
        await _run_chat(state, slot, "do it")

        assistant_msgs = [m for m in slot.messages if m.get("role") == "assistant"]
        # Segments are flushed at tool boundaries; empty chunks don't create segments
        assert len(assistant_msgs) == 2
        assert "Before." in assistant_msgs[0]["content"]
        assert "After!" in assistant_msgs[1]["content"]


class TestRunChatToolCallUpdate:
    """EVENT_TOOL_CALL_UPDATE handler — claude-agent-acp emits a refinement
    once the streamed tool input is complete. The handler patches the in-place
    pill, persisted message, _pending_tools, and the SEL audit trail."""

    @staticmethod
    def _make_mock_client(events):
        client = _provider_mock()
        client.context_usage_pct = MagicMock(return_value=10.0)

        async def _stream(msg):
            for ev in events:
                yield ev

        client.stream = _stream
        client.stream_command = _stream
        return client

    @staticmethod
    def _make_state_for_run_chat(tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.context_builder = None
        state.consolidator = None
        state._hook_store = None
        state._yolo = False
        return state

    @pytest.mark.asyncio
    async def test_refinement_patches_pill_content_and_meta(self, tmp_path, monkeypatch):
        """An initial tool_call with a stub title is overwritten by the refined
        title and the meta picks up the populated input."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL, title="Terminal", tool_kind="execute", tool_call_id="tc-1"
            ),
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_input='{"command": "ls /tmp"}',
                tool_call_id="tc-1",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")

        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        tool_msgs = [m for m in slot.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        # The persisted content should reflect the refined title, not the stub.
        assert tool_msgs[0]["content"] == "🔧 ls /tmp"
        assert tool_msgs[0]["meta"]["tool_call_id"] == "tc-1"
        # The refined input is patched into meta.
        assert "ls /tmp" in tool_msgs[0]["meta"]["input"]

    @pytest.mark.asyncio
    async def test_refinement_carries_purpose_when_it_has_one(self, tmp_path, monkeypatch):
        """The update frame must carry the purpose so the live status line keeps
        showing the agent's own reason for the call."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Terminal",
                tool_kind="execute",
                tool_purpose="List the temp dir",
                tool_call_id="tc-p1",
            ),
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_purpose="List the temp dir",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-p1",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        ws_calls = [(c.args[0], c.args[1]) for c in state.broadcast_ws.call_args_list]
        (update,) = [p for k, p in ws_calls if k == "tool_call" and p.get("is_update")]
        assert update["purpose"] == "List the temp dir"

    @pytest.mark.asyncio
    async def test_purposeless_refinement_omits_the_key(self, tmp_path, monkeypatch):
        """A refinement with no purpose omits the key entirely rather than
        sending an empty string: consumers merge field-by-field and read an
        absent ``purpose`` as "keep what the initial tool_call supplied", so an
        empty value would replace a good purpose with the raw command."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Terminal",
                tool_kind="execute",
                tool_purpose="List the temp dir",
                tool_call_id="tc-p2",
            ),
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-p2",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        ws_calls = [(c.args[0], c.args[1]) for c in state.broadcast_ws.call_args_list]
        (update,) = [p for k, p in ws_calls if k == "tool_call" and p.get("is_update")]
        assert "purpose" not in update

    @pytest.mark.asyncio
    async def test_refinement_persists_a_recovered_purpose(self, tmp_path, monkeypatch):
        """A refinement's purpose must reach the PERSISTED meta, not just the live
        status: when the initial tool_call streamed an empty rawInput, _tool_meta
        wrote an empty purpose, and the reloaded transcript reads meta.purpose —
        so a live-only patch loses it on the next reload."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            # Streaming backend: no purpose and no input on the initial frame.
            LLMEvent(
                kind=EVENT_TOOL_CALL, title="Terminal", tool_kind="execute", tool_call_id="tc-p3"
            ),
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_purpose="List the temp dir",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-p3",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        (tool_msg,) = [m for m in slot.messages if m.get("role") == "tool"]
        assert tool_msg["meta"]["purpose"] == "List the temp dir"
        # And the same patch goes out live so an open tab does not wait for a reload.
        ws_calls = [(c.args[0], c.args[1]) for c in state.broadcast_ws.call_args_list]
        (msg_update,) = [p for k, p in ws_calls if k == "chat_message_update"]
        assert msg_update["meta"]["purpose"] == "List the temp dir"

    @pytest.mark.asyncio
    async def test_tool_meta_persists_kind(self, tmp_path, monkeypatch):
        """The persisted tool-message meta carries the ACP tool kind. The
        dashboard gates the inline diff-card promotion on kind == 'edit'
        (a shell command whose input looks like a diff must never promote),
        and historical rows can only be gated from persisted meta."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            LLMEvent,
        )

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="fs_write",
                tool_kind="edit",
                tool_purpose="Edit app.py",
                tool_input="--- /a/app.py\n+++ /a/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2",
                tool_call_id="tc-kind1",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        (tool_msg,) = [m for m in slot.messages if m.get("role") == "tool"]
        assert tool_msg["meta"]["kind"] == "edit"

    @pytest.mark.asyncio
    async def test_refinement_broadcasts_chat_message_update(self, tmp_path, monkeypatch):
        """The handler broadcasts a chat_message_update WS event so the
        frontend can patch the persisted tile in place without a reload."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL, title="Terminal", tool_kind="execute", tool_call_id="tc-2"
            ),
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-2",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        ws_calls = [(c.args[0], c.args[1]) for c in state.broadcast_ws.call_args_list]
        kinds = [k for k, _ in ws_calls]
        # Refinement broadcasts both a fresh tool_call (toolLog merge) AND a
        # chat_message_update so the persisted tile updates live too.
        assert kinds.count("tool_call") >= 2
        assert "chat_message_update" in kinds
        # The second tool_call carries is_update:True so the frontend merges.
        update_payloads = [p for k, p in ws_calls if k == "tool_call" and p.get("is_update")]
        assert len(update_payloads) == 1
        assert update_payloads[0]["tool"] == "ls /tmp"
        assert update_payloads[0]["tool_call_id"] == "tc-2"
        # And the chat_message_update carries the patched content + meta.
        msg_updates = [p for k, p in ws_calls if k == "chat_message_update"]
        assert msg_updates[0]["tool_call_id"] == "tc-2"
        assert msg_updates[0]["content"] == "🔧 ls /tmp"
        assert "input" in msg_updates[0]["meta"]

    @pytest.mark.asyncio
    async def test_refinement_preserves_existing_icon(self, tmp_path, monkeypatch):
        """Auto-approved tools may already carry a ✅ marker on the message
        with the same tool_call_id. The patch must preserve that prefix."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        # Pre-seed a message that matches the tool_call_id with a ✅ icon —
        # mirrors the auto-approved-tool case where the post-approval marker
        # is already in place.
        slot.append("tool", "✅ Terminal", "msg msg-tool", meta={"tool_call_id": "tc-3"})

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-3",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        tool_msgs = [m for m in slot.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        # ✅ icon survives the patch; only the title changes.
        assert tool_msgs[0]["content"] == "✅ ls /tmp"

    @pytest.mark.asyncio
    async def test_refinement_breaks_on_first_match_walking_reverse(self, tmp_path, monkeypatch):
        """When two messages share the tool_call_id (auto-approved double-emit
        with 🔧 then ✅), only the most recent one is patched."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("tool", "🔧 Terminal", "msg msg-tool", meta={"tool_call_id": "tc-4"})
        slot.append("tool", "✅ Terminal", "msg msg-tool", meta={"tool_call_id": "tc-4"})

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_call_id="tc-4",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        tool_msgs = [m for m in slot.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        # The earlier 🔧 message is untouched.
        assert tool_msgs[0]["content"] == "🔧 Terminal"
        # The later ✅ message is patched, with its icon preserved.
        assert tool_msgs[1]["content"] == "✅ ls /tmp"

    @pytest.mark.asyncio
    async def test_refinement_strips_running_prefix_from_pending_tools(self, tmp_path, monkeypatch):
        """_pending_tools feeds PostToolUse hooks by tool name. The refinement
        must strip the "Running: " prefix exactly like EVENT_TOOL_CALL does so
        hooks matching by name keep working after the refinement event."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Running: stub",
                tool_kind="execute",
                tool_call_id="tc-5",
            ),
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="Running: ls /tmp",
                tool_kind="execute",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-5",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        # Patch fire_tool_hooks so we can inspect the name passed in. The hook
        # only fires on EVENT_TOOL_CALL — but the refinement updates
        # _pending_tools, which is read by the EVENT_TOOL_RESULT handler. We
        # don't drive a tool_result here; instead we inspect _pending_tools
        # was updated correctly via the WS-broadcast surface area: the
        # refinement broadcasts the refined title without the "Running: "
        # prefix on the handler's local copy.
        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        # Indirectly verify via the persisted message — the title displayed on
        # the pill should be the refined one, sans "Running: " prefix because
        # the refinement code strips it for _pending_tools (the displayed
        # title still carries the prefix, but the hook-name copy doesn't).
        # The persisted pill carries the refined title verbatim — this is the
        # display surface, not the hook surface.
        tool_msgs = [m for m in slot.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "🔧 Running: ls /tmp"

    @pytest.mark.asyncio
    async def test_refinement_logs_sel_audit_event(self, tmp_path, monkeypatch):
        """The handler logs a `tool_invocation` audit event with
        outcome="refined" so the audit trail captures the refined name."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        captured = []

        class _FakeSel:
            def log_tool_invocation(self, **kw):
                captured.append(kw)

        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.sel", lambda: _FakeSel())

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-6",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        refined = [c for c in captured if c.get("outcome") == "refined"]
        assert len(refined) == 1
        assert refined[0]["tool_name"] == "ls /tmp"
        assert refined[0]["tool_kind"] == "execute"
        assert refined[0]["source"] == "dashboard"

    @pytest.mark.asyncio
    async def test_refinement_no_tool_call_id_skipped(self, tmp_path, monkeypatch):
        """Refinement events without a tool_call_id are silently dropped —
        we have nothing to merge against."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            LLMEvent(kind=EVENT_TOOL_CALL_UPDATE, title="ls /tmp", tool_call_id=""),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        # No tool messages should have been added.
        tool_msgs = [m for m in slot.messages if m.get("role") == "tool"]
        assert tool_msgs == []
        # No chat_message_update broadcast either.
        kinds = [c.args[0] for c in state.broadcast_ws.call_args_list]
        assert "chat_message_update" not in kinds

    @pytest.mark.asyncio
    async def test_refinement_no_matching_message_no_chat_message_update(
        self, tmp_path, monkeypatch
    ):
        """When no persisted tool message matches the tool_call_id, the
        handler still broadcasts the tool_call merge but skips
        chat_message_update (nothing to patch)."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_input='{"command":"ls /tmp"}',
                tool_call_id="tc-orphan",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        kinds = [c.args[0] for c in state.broadcast_ws.call_args_list]
        # tool_call still broadcast (toolLog merge by id is still meaningful).
        assert "tool_call" in kinds
        # Nothing to patch in slot.messages, so no chat_message_update.
        assert "chat_message_update" not in kinds

    @pytest.mark.asyncio
    async def test_refinement_redacts_credentials_in_input(self, tmp_path, monkeypatch):
        """Credentials in tool_input must be redacted before the broadcast
        and the persisted meta. _redact_tool_field applies both
        redact_exfiltration_urls and redact_credentials."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("tool", "🔧 Terminal", "msg msg-tool", meta={"tool_call_id": "tc-cred"})

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="echo secret",
                tool_kind="execute",
                tool_input='{"command":"echo AKIAIOSFODNN7EXAMPLE"}',
                tool_call_id="tc-cred",
            ),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        tool_msgs = [m for m in slot.messages if m.get("role") == "tool"]
        # Credential must not appear in the persisted meta.
        assert "AKIAIOSFODNN7EXAMPLE" not in tool_msgs[0]["meta"].get("input", "")
        # Or in any of the broadcast payloads.
        for call in state.broadcast_ws.call_args_list:
            assert "AKIAIOSFODNN7EXAMPLE" not in str(call.args[1])

    @pytest.mark.asyncio
    async def test_refinement_handler_swallows_exceptions(self, tmp_path, monkeypatch):
        """A malformed broadcast or other exception inside the handler must
        not tear down the run loop. The try/except logs and continues."""
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL_UPDATE,
            LLMEvent,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")

        # Make broadcast_ws raise on the first call (the tool_call broadcast).
        # The try/except in the handler must catch it so subsequent events
        # (the trailing EVENT_TEXT_CHUNK) still process.
        original_broadcast = state.broadcast_ws

        def _raising_then_normal(*args, **kwargs):
            if args and args[0] == "tool_call":
                raise RuntimeError("simulated broadcast failure")
            return original_broadcast(*args, **kwargs)

        state.broadcast_ws = MagicMock(side_effect=_raising_then_normal)

        events = [
            LLMEvent(
                kind=EVENT_TOOL_CALL_UPDATE,
                title="ls /tmp",
                tool_kind="execute",
                tool_call_id="tc-boom",
            ),
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="after the boom"),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        # Must not raise — the run loop should continue past the exception.
        await _run_chat(state, slot, "hello")

        # The trailing assistant message proves the run loop survived.
        assistant_msgs = [m for m in slot.messages if m.get("role") == "assistant"]
        assert any("after the boom" in m["content"] for m in assistant_msgs)


# ── Model refusal (kiro-cli `refusal` stop reason) ──


class TestRunChatModelRefusal:
    """A turn that ends on stop_reason 'refusal' with no text is a DETERMINISTIC
    model-side content refusal — it must surface a distinct, non-retried card
    instead of falling into the blind empty-response retry loop (which just
    re-hits the same refusal and burns credits)."""

    @staticmethod
    def _make_mock_client(events, prompts: list[str] | None = None):
        client = _provider_mock()
        client.context_usage_pct = MagicMock(return_value=10.0)

        async def _stream(msg):
            if prompts is not None:
                prompts.append(str(msg))
            for ev in events:
                yield ev

        client.stream = _stream
        client.stream_command = _stream
        return client

    @staticmethod
    def _make_state_for_run_chat(tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.context_builder = None
        state.consolidator = None
        state._hook_store = None
        state._yolo = False
        return state

    @pytest.mark.asyncio
    async def test_stale_recover_notice_is_tagged_and_emits_one_frame(self, tmp_path, monkeypatch):
        """The LIVE frame must carry the retry tag, and there must be exactly one."""
        from kiro_crew.acp.types import STOP_REASON_STALE_RECOVER
        from kiro_crew.dashboard.chat_utils import TRANSIENT_RETRY_KIND
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        events = [LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_STALE_RECOVER)]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        _notices = [
            m
            for m in slot.messages
            if m.get("role") == "error" and "Recovering a stalled turn" in m.get("content", "")
        ]
        assert _notices, "stale-recovery notice not emitted"
        assert all((m.get("meta") or {}).get("kind") == TRANSIENT_RETRY_KIND for m in _notices)
        # slot.append already emits ONE tagged chat_message; an explicit frame here is an
        # untagged duplicate, and a client rendering it re-offers the executing choice.
        _dupes = [
            c
            for c in state.broadcast_ws.call_args_list
            if c.args
            and c.args[0] == "chat_message"
            and isinstance(c.args[1], dict)
            and "Recovering a stalled turn" in str(c.args[1].get("content", ""))
        ]
        assert not _dupes, f"untagged explicit chat_message duplicate(s): {_dupes}"

    @pytest.mark.asyncio
    async def test_refusal_shows_declined_card_and_does_not_retry(self, tmp_path, monkeypatch):
        from kiro_crew.acp.types import STOP_REASON_REFUSAL
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        events = [LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_REFUSAL)]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        # A single declined card is surfaced, NOT the empty-response card.
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("declined" in m.get("content", "").lower() for m in error_msgs)
        assert not any("empty response" in m.get("content", "").lower() for m in error_msgs)
        # No blind re-queue: the message is not re-enqueued and the retry budget
        # is untouched.
        assert not slot._queue
        assert slot._empty_response_retries == 0


class TestRunChatRefusalFallback:
    """agent.refusal_fallback_model: one single-message retry on a DIFFERENT
    model after a content-filter refusal, then restore the primary.

    A refusal is deterministic for the model that issued it, but not across
    model families — the whole point of the retry. The feature ships OFF
    (empty default); every test here pins the seam explicitly."""

    _make_state_for_run_chat = staticmethod(TestRunChatModelRefusal._make_state_for_run_chat)

    @staticmethod
    def _make_refusing_client(events, prompts: list[str] | None = None):
        client = TestRunChatModelRefusal._make_mock_client(events, prompts)
        # provider_active_model / provider_raw_model read `served_model` first;
        # the witness needs set_model to observably move it.
        client.served_model = "fable-5"
        # Real-int epoch: a bare MagicMock auto-attribute here would compare
        # unequal to the slot's integer snapshot and trip the alias-pick guard.
        client._explicit_pick_epoch = 0

        def _apply(mid: str) -> None:
            client.served_model = mid

        client.set_model = AsyncMock(side_effect=_apply)
        return client

    @staticmethod
    def _refusal_events(**refusal_kwargs):
        from kiro_crew.acp.types import STOP_REASON_REFUSAL, RefusalInfo
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        refusal = RefusalInfo(**refusal_kwargs) if refusal_kwargs else None
        return [LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_REFUSAL, refusal=refusal)]

    @pytest.mark.asyncio
    async def test_refusal_retries_once_on_configured_model_then_terminal(
        self, tmp_path, monkeypatch
    ):
        """First refusal swaps + replays the SAME message; a second refusal
        (from the fallback) is terminal and says the fallback also declined."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        events = self._refusal_events(category="CYBER")
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        # Turn 1: the swap landed, the retry notice (not the terminal card)
        # rendered, and the one-per-user-message allowance is spent.
        client.set_model.assert_awaited_once_with("opus-test")
        notices = [
            m
            for m in slot.messages
            if m.get("role") == "error" and "retrying once on 'opus-test'" in m.get("content", "")
        ]
        assert len(notices) == 1, f"expected exactly one retry notice, got {slot.messages}"
        # The notice stays plain-language: the raw classifier category
        # ("cyber") is jargon and never reaches the user.
        assert "(cyber)" not in notices[0]["content"]
        assert slot._refusal_fallback_attempted is True
        assert slot._refusal_fallback_primary == "fable-5"
        assert not any(
            "Response declined by the model." in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "error"
        ), "terminal card must not render on the retried turn"

        # The replay was dispatched as a background task; run it. The mock
        # stream refuses again — the fallback also declined.
        assert slot.task is not None, "queued replay was not dispatched"
        await slot.task

        cards = [
            m
            for m in slot.messages
            if m.get("role") == "error"
            and "Response declined by the model." in m.get("content", "")
        ]
        assert len(cards) == 1
        assert (
            "The configured fallback model ('opus-test') also declined this request."
            in cards[0]["content"]
        )
        # Still exactly one swap (no retry loop), replay record consumed.
        client.set_model.assert_awaited_once_with("opus-test")
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert not slot._queue

    @pytest.mark.asyncio
    async def test_primary_restored_on_next_genuine_turn(self, tmp_path, monkeypatch):
        """The swap is single-message: the first turn that is not the replay
        moves the session back to the primary before dispatching."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")
        assert slot.task is not None
        await slot.task  # the retry turn (fallback also refuses; terminal)
        assert slot._refusal_fallback_primary == "fable-5"

        # Next GENUINE message completes normally; restore precedes dispatch.
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        async def _ok_stream(msg):
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _ok_stream
        client.stream_command = _ok_stream
        await _run_chat(state, slot, "a fresh question")

        assert [c.args[0] for c in client.set_model.await_args_list] == ["opus-test", "fable-5"]
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_candidate == ""
        # The fresh message also re-armed the one-retry allowance.
        assert slot._refusal_fallback_attempted is False

    @pytest.mark.asyncio
    async def test_disabled_seam_keeps_terminal_card_and_never_swaps(self, tmp_path, monkeypatch):
        """Empty config (the default) is byte-for-byte the pre-feature path."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        client.set_model.assert_not_awaited()
        assert not slot._queue
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("Response declined by the model." in m.get("content", "") for m in error_msgs)
        # No fallback ran, so the card must NOT claim one declined.
        assert not any("fallback model also declined" in m.get("content", "") for m in error_msgs)

    @pytest.mark.asyncio
    async def test_auto_uses_provider_recommended_model(self, tmp_path, monkeypatch):
        """'auto' defers to the refusal envelope's recommended_model."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "auto",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(
            self._refusal_events(category="CYBER", recommended_model="rec-model-1")
        )
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        client.set_model.assert_awaited_once_with("rec-model-1")
        assert slot._refusal_fallback_candidate == "rec-model-1"

    @pytest.mark.asyncio
    async def test_auto_without_recommendation_is_terminal(self, tmp_path, monkeypatch):
        """'auto' with no recommended_model has nothing to retry on."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "auto",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        client.set_model.assert_not_awaited()
        assert not slot._queue
        assert any(
            "Response declined by the model." in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "error"
        )

    @pytest.mark.asyncio
    async def test_failed_swap_falls_back_to_terminal_card(self, tmp_path, monkeypatch):
        """A set_model failure surfaces the ordinary card — never a dead retry."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        client.set_model = AsyncMock(side_effect=RuntimeError("model unavailable"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        assert not slot._queue
        assert slot._refusal_fallback_attempted is False
        assert any(
            "Response declined by the model." in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "error"
        )

    @pytest.mark.asyncio
    async def test_retry_after_tool_dispatch_continues_instead_of_replaying(
        self, tmp_path, monkeypatch
    ):
        """A refusal terminal arriving AFTER the turn dispatched a tool still
        moves the session to the fallback model, but the replay is the
        Continue-style continuation, never the user's message: replaying the
        message would run the dispatched tool a second time."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        from kiro_crew.dashboard.chat_utils import (
            _REFUSAL_FALLBACK_RESUME_MSG,
            RecoveryPayload,
        )
        from kiro_crew.providers.base import EVENT_TOOL_CALL, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TOOL_CALL, title="write_file", tool_kind="write"),
            *self._refusal_events(category="CYBER"),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        # Capture what the swap queues: the entry is consumed by the drain as
        # soon as the turn ends, so it cannot be inspected on ``slot._queue``.
        # ``_ChatSlot`` declares ``__slots__``, so the spy goes on the class.
        queued: list[tuple[tuple, dict]] = []
        _orig_queue_insert = type(slot).queue_insert

        def _spy(self, *args, **kwargs):
            if self is slot:
                queued.append((args, kwargs))
            return _orig_queue_insert(self, *args, **kwargs)

        monkeypatch.setattr(type(slot), "queue_insert", _spy)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello", _attachments=[str(tmp_path / "a.txt")])

        client.set_model.assert_awaited_once_with("opus-test")
        assert slot._refusal_fallback_attempted is True
        assert slot._refusal_fallback_primary == "fable-5"
        assert len(queued) == 1, f"expected exactly one queued continuation, got {queued}"
        (_index, body), kwargs = queued[0]
        assert _index == 0
        assert body == _REFUSAL_FALLBACK_RESUME_MSG
        assert "hello" not in body
        # Runner text, so a linked thread must not mirror it as user speech...
        assert kwargs["payload"] == RecoveryPayload.CONTINUATION
        # ...and the original's attachments do not ride along — the refused
        # turn already delivered them into the session.
        assert not any(k in kwargs.get("meta", {}) for k in ("files", "dirs"))
        notices = [
            m
            for m in slot.messages
            if m.get("role") == "error" and "continuing on 'opus-test'" in m.get("content", "")
        ]
        assert len(notices) == 1, f"expected exactly one continue notice, got {slot.messages}"
        assert "after 1 tool call —" in notices[0]["content"]
        assert not any(
            "Response declined by the model." in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "error"
        ), "terminal card must not render on the continued turn"

        # The continuation runs on the fallback; the mock stream refuses again
        # (after a tool call again), and the allowance is spent — terminal.
        assert slot.task is not None, "queued continuation was not dispatched"
        await slot.task
        cards = [
            m
            for m in slot.messages
            if m.get("role") == "error"
            and "Response declined by the model." in m.get("content", "")
        ]
        assert len(cards) == 1
        assert "The configured fallback model ('opus-test') also declined" in cards[0]["content"]
        assert client.set_model.await_count == 1, "the continuation must not swap again"

    def test_resume_msg_fails_safe_when_completed_work_is_not_in_view(self):
        """The continuation is sent ONLY after the turn dispatched a tool, so
        "nothing was done yet" is never true of it: a clause letting the
        fallback model start the request over could only ever fire on a
        session that did not keep the partial turn — re-running the writes.
        The nudge must tell such a model to stop instead."""
        from kiro_crew.dashboard.chat_utils import _REFUSAL_FALLBACK_RESUME_MSG

        assert "start the request now" not in _REFUSAL_FALLBACK_RESUME_MSG
        assert "do NOT start the request over" in _REFUSAL_FALLBACK_RESUME_MSG
        assert "say so and stop" in _REFUSAL_FALLBACK_RESUME_MSG

    @pytest.mark.asyncio
    async def test_clean_refusal_still_replays_the_message_verbatim(self, tmp_path, monkeypatch):
        """With no tool dispatched, the retry stays the verbatim replay of the
        user's own words, carrying the original attachments."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        from kiro_crew.dashboard.chat_utils import RecoveryPayload

        events = self._refusal_events(category="CYBER")
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        queued: list[tuple[tuple, dict]] = []
        _orig_queue_insert = type(slot).queue_insert

        def _spy(self, *args, **kwargs):
            if self is slot:
                queued.append((args, kwargs))
            return _orig_queue_insert(self, *args, **kwargs)

        monkeypatch.setattr(type(slot), "queue_insert", _spy)

        from kiro_crew.dashboard.chat import _run_chat

        _file = str(tmp_path / "a.txt")
        await _run_chat(state, slot, "hello", _attachments=[_file])

        assert len(queued) == 1
        (_index, body), kwargs = queued[0]
        assert _index == 0
        assert body == "hello"
        assert kwargs["payload"] == RecoveryPayload.ORIGINAL
        assert kwargs["meta"].get("files") == [_file]
        assert any(
            "retrying once on 'opus-test'" in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "error"
        )
        if slot.task is not None:
            await slot.task

    @pytest.mark.asyncio
    async def test_replay_turn_does_not_pin_unpinned_slot(self, tmp_path, monkeypatch):
        """The replay turn runs with the candidate active while
        ``_active_fallback_model`` is empty — the slot.model backfill must not
        persist the temporary candidate into the (durable) pin."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        # Simulate the real provider contract the backfill documents: the
        # resolved model readable off the client IS the currently served one.
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._backfill_canonical_model",
            lambda client, provider_name: getattr(client, "served_model", "") or "",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        assert not slot.model, "precondition: slot starts unpinned"
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")
        # Turn 1 backfilled the PRIMARY (legitimate). Clear it to model the
        # documented CC case — the canonical id is unresolvable until a later
        # turn — so the replay turn runs against a genuinely unpinned slot.
        slot.model = ""
        assert slot.task is not None
        await slot.task  # the replay turn — candidate active, throttle state empty

        assert (
            not slot.model
        ), f"replay turn persisted the temporary candidate into slot.model: {slot.model!r}"

    @pytest.mark.asyncio
    async def test_synthetic_recovery_does_not_restore(self, tmp_path, monkeypatch):
        """A synthetic recovery turn after the replay keeps the fallback
        active; only a genuine turn restores the primary."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")
        assert slot.task is not None
        await slot.task
        assert slot._refusal_fallback_primary == "fable-5"

        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.dashboard.chat_utils import _SYNTHETIC_RECOVERY_MSGS
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        async def _ok_stream(msg):
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _ok_stream
        client.stream_command = _ok_stream

        # A runner-authored continuation must finish on the model that
        # produced the work it continues — no restore.
        await _run_chat(state, slot, _SYNTHETIC_RECOVERY_MSGS[0])
        assert slot._refusal_fallback_primary == "fable-5"
        assert [c.args[0] for c in client.set_model.await_args_list] == ["opus-test"]

        # The next genuine turn restores.
        await _run_chat(state, slot, "a fresh question")
        assert slot._refusal_fallback_primary == ""
        assert [c.args[0] for c in client.set_model.await_args_list] == [
            "opus-test",
            "fable-5",
        ]

    @pytest.mark.asyncio
    async def test_restore_keeps_record_on_silent_noop(self, tmp_path, monkeypatch):
        """A non-raising ``set_model`` that does not move the model must NOT
        clear the restore record — the next turn retries instead of stranding
        the session on the fallback with nothing left to restore from."""
        from kiro_crew.dashboard.chat_runner import _restore_refusal_fallback

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "opus-test"  # fallback currently active
        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        client.set_model = AsyncMock()  # silent no-op: served_model unchanged

        await _restore_refusal_fallback(slot, client)
        client.set_model.assert_awaited_once_with("fable-5")
        assert (
            slot._refusal_fallback_primary == "fable-5"
        ), "restore record cleared on a silent set_model no-op"

        # A working set_model on the next attempt restores and clears.
        def _apply(mid: str) -> None:
            client.served_model = mid

        client.set_model = AsyncMock(side_effect=_apply)
        await _restore_refusal_fallback(slot, client)
        assert slot._refusal_fallback_primary == ""
        assert client.served_model == "fable-5"

    @pytest.mark.asyncio
    async def test_explicit_pick_after_retry_wins_over_restore(self, tmp_path, monkeypatch):
        """An explicit model pick landing between the replay and the next
        genuine turn drops the restore record without touching the model —
        even a pick of the fallback candidate itself, which current-model
        equality alone cannot distinguish from the automatic swap."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")
        assert slot.task is not None
        await slot.task
        assert slot._refusal_fallback_primary == "fable-5"

        # An explicit pick moves the generation (api_chat_slot_model's seam).
        slot._model_pick_gen += 1

        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        async def _ok_stream(msg):
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _ok_stream
        client.stream_command = _ok_stream
        await _run_chat(state, slot, "a fresh question")

        # No restore call — only the original swap — and the record is gone.
        assert [c.args[0] for c in client.set_model.await_args_list] == ["opus-test"]
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_candidate == ""

    @pytest.mark.asyncio
    async def test_no_retry_when_follow_up_queued(self, tmp_path, monkeypatch):
        """A queued follow-up is the user's next intent — often a correction
        of the refused message. The replay inserts at queue index 0, so it
        would run BEFORE that correction; the retry must refuse instead and
        leave the queue untouched."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        slot.queue_append("actually, drop that request")

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        client.set_model.assert_not_awaited()
        assert slot._refusal_fallback_attempted is False
        assert slot._refusal_fallback_primary == ""
        cards = [
            i
            for i, m in enumerate(slot.messages)
            if m.get("role") == "error"
            and "Response declined by the model." in m.get("content", "")
        ]
        assert len(cards) == 1, "the terminal card must render when the retry is refused"
        # No replay was recorded or announced — the follow-up owns the next
        # turn instead of being jumped by a re-send of the refused message.
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert not any("retrying once on" in m.get("content", "") for m in slot.messages)

    @pytest.mark.asyncio
    async def test_swap_unwound_when_follow_up_arrives_during_swap(self, tmp_path, monkeypatch):
        """A follow-up landing while set_model is in flight changes intent
        under the swap: the swap is unwound (witnessed restore), the refusal
        surfaces, and nothing is queued ahead of the follow-up."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))

        def _apply_and_interrupt(mid: str) -> None:
            client.served_model = mid
            if mid == "opus-test":
                slot.queue_append("wait — different plan")

        client.set_model = AsyncMock(side_effect=_apply_and_interrupt)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        # Swap then unwind: candidate first, primary back, record cleared.
        assert [c.args[0] for c in client.set_model.await_args_list] == ["opus-test", "fable-5"]
        assert client.served_model == "fable-5"
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_attempted is False
        assert any(
            m.get("role") == "error" and "Response declined by the model." in m.get("content", "")
            for m in slot.messages
        )
        # No replay was recorded or announced ahead of the follow-up.
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert not any("retrying once on" in m.get("content", "") for m in slot.messages)

    @pytest.mark.asyncio
    async def test_swap_unwound_when_stop_completes_during_swap(self, tmp_path, monkeypatch):
        """A Stop that presses AND resolves back to idle while set_model is in
        flight leaves ``_stop_state == "idle"``, so the state check alone
        misses it. The monotonic stop generation still moved: the swap must
        unwind and nothing may be queued — otherwise the stopped request
        replays, and the enqueue snapshot would bake the moved counter in so
        the drain purge could never catch it either."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))

        def _apply_and_stop(mid: str) -> None:
            client.served_model = mid
            if mid == "opus-test":
                # Stop pressed and fully resolved during the swap await:
                # the counter moved, the state is back to idle.
                slot._stop_generation = getattr(slot, "_stop_generation", 0) + 1
                slot._stop_state = "idle"

        client.set_model = AsyncMock(side_effect=_apply_and_stop)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        # Swap then unwind: candidate first, primary back, record cleared.
        assert [c.args[0] for c in client.set_model.await_args_list] == ["opus-test", "fable-5"]
        assert client.served_model == "fable-5"
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_attempted is False
        # No replay recorded or announced: the stopped request must not rerun.
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert not any("retrying once on" in m.get("content", "") for m in slot.messages)

    @pytest.mark.asyncio
    async def test_chained_swap_preserves_original_primary(self, tmp_path, monkeypatch):
        """A second swap while the restore record is still live must NOT
        overwrite the recorded primary. After a failed/no-op restore the
        session is stranded on fallback A with the record still naming the
        TRUE primary P; a second refusal's swap A->B reads A as "active" —
        recording A would lose P permanently (restore would return to A,
        never to P)."""
        from kiro_crew.dashboard.chat_runner import (
            _refusal_fallback_swap,
            _restore_refusal_fallback,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        # Stranded state: restore to fable-5 failed last turn, record kept.
        client.served_model = "opus-test"
        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"

        def _apply(mid: str) -> None:
            client.served_model = mid

        client.set_model = AsyncMock(side_effect=_apply)

        # Second refusal on the fallback: auto recommends a different model.
        returned = await _refusal_fallback_swap(slot, client, "sonnet-test")

        assert returned == "opus-test", "the notice names the model that refused"
        assert client.served_model == "sonnet-test"
        assert (
            slot._refusal_fallback_primary == "fable-5"
        ), "chained swap overwrote the true primary with the stale fallback"
        assert slot._refusal_fallback_candidate == "sonnet-test"

        # The restore returns the session to the TRUE primary directly.
        await _restore_refusal_fallback(slot, client)
        assert client.served_model == "fable-5"
        assert slot._refusal_fallback_primary == ""

    @pytest.mark.asyncio
    async def test_swap_skipped_when_auto_primary_has_no_provable_restore(
        self, tmp_path, monkeypatch
    ):
        """Unpinned session (the "auto" sentinel): the restore leg must
        set_model("auto"), which partitions that do not advertise the
        sentinel refuse — the restore then fails every turn and the session
        stays stranded on the candidate. The swap must not run when the
        return leg is unprovable."""
        from kiro_crew.dashboard.chat_runner import _refusal_fallback_swap

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "auto"  # provider_active_model() -> ""
        client.available_models = lambda: [
            {"modelId": "fable-5"},
            {"modelId": "opus-test"},
        ]  # no "auto" advertised: the restore target is unprovable

        set_model = AsyncMock()
        client.set_model = set_model

        returned = await _refusal_fallback_swap(slot, client, "opus-test")

        assert returned is None, "swap must be skipped when the restore leg is unprovable"
        set_model.assert_not_awaited()
        assert getattr(slot, "_refusal_fallback_primary", "") in ("", None)

    @pytest.mark.asyncio
    async def test_swap_proceeds_when_auto_is_advertised(self, tmp_path, monkeypatch):
        """When the backend advertises the "auto" sentinel as a target the
        round trip is provable: the swap runs and records "auto" as the
        primary for the restore."""
        from kiro_crew.dashboard.chat_runner import _refusal_fallback_swap

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "auto"
        client.available_models = lambda: [
            {"modelId": "auto"},
            {"modelId": "opus-test"},
        ]

        def _apply(mid: str) -> None:
            client.served_model = mid

        client.set_model = AsyncMock(side_effect=_apply)

        returned = await _refusal_fallback_swap(slot, client, "opus-test")

        assert returned == "auto"
        assert client.served_model == "opus-test"
        assert slot._refusal_fallback_primary == "auto"
        assert slot._refusal_fallback_candidate == "opus-test"

    @pytest.mark.asyncio
    async def test_orchestration_queue_entry_does_not_suppress_retry(self, tmp_path, monkeypatch):
        """A queued cron notification is orchestration, not a user correction:
        it must not suppress the configured retry. Only USER-authored queue
        entries express superseding intent (`_has_user_queued_followup`)."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat_utils import CRON_NOTIFICATION_KIND

        slot.queue_append(
            '[Cron notification from "scanner"]\nno new items', kind=CRON_NOTIFICATION_KIND
        )

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        # The retry proceeded despite the queued orchestration entry.
        client.set_model.assert_awaited_once_with("opus-test")
        assert slot._refusal_fallback_attempted is True
        assert any(
            "retrying once on 'opus-test'" in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "error"
        ), "retry notice missing — the cron entry suppressed the retry"

    @pytest.mark.asyncio
    async def test_unattended_actor_keeps_terminal_refusal(self, tmp_path, monkeypatch):
        """A cron/autonudge/sub-agent wake keeps the terminal refusal: nobody
        attends the announced swap, and an unattended replay doubles whatever
        the wake was about. The gate reads the turn's ledger actor."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "nightly scan wake", _turn_actor="subagent")

        client.set_model.assert_not_awaited()
        assert slot._refusal_fallback_attempted is False
        assert slot._refusal_fallback_primary == ""
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert any(
            m.get("role") == "error" and "Response declined by the model." in m.get("content", "")
            for m in slot.messages
        ), "the unattended turn must surface the ordinary terminal card"
        assert not any("retrying once on" in m.get("content", "") for m in slot.messages)

    @pytest.mark.asyncio
    async def test_a_loops_refused_cycle_records_the_loop_on_the_slot(self, tmp_path, monkeypatch):
        """A loop's own cycle that ends refused names that loop on the slot.

        The fire guard reads the pair before the next cycle and stops the loop
        rather than sending the same prompt to the same model again. Recorded
        with a fallback configured too, because the fallback never runs for an
        unattended cycle, so the card is terminal either way.
        """
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(
            state,
            slot,
            "patrol wake",
            _directive_self_wake=True,
            _directive_loop_id="loop-42",
            _directive_loop_gen=7,
        )

        client.set_model.assert_not_awaited()
        assert slot._last_turn_model_refused_loop_id == "loop-42"
        assert slot._last_turn_model_refused_loop_gen == 7

    @pytest.mark.asyncio
    async def test_a_refused_person_turn_names_no_loop(self, tmp_path, monkeypatch):
        """A person's refused turn must not stop a loop that shares the slot."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        assert slot._last_turn_model_refused_loop_id == ""
        assert slot._last_turn_model_refused_loop_gen == 0

    @pytest.mark.asyncio
    async def test_a_genuine_new_turn_clears_the_refused_cycle(self, tmp_path, monkeypatch):
        """A person's next message is a new context, so the loop may fire again."""
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot._last_turn_model_refused_loop_id = "loop-42"
        slot._last_turn_model_refused_loop_gen = 7
        client = self._make_refusing_client(
            [
                LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok"),
                LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn"),
            ]
        )
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "try a different angle")

        assert slot._last_turn_model_refused_loop_id == ""
        assert slot._last_turn_model_refused_loop_gen == 0

    @pytest.mark.asyncio
    async def test_throttle_walk_divergence_redirects_walk_restore_target(
        self, tmp_path, monkeypatch
    ):
        """A throttle walk advancing OFF the refusal candidate mid-retry must
        not cost the true primary: the walk's restore target is rewritten to
        the refusal primary and the refusal record hands over, with the wire
        model left on the walk's live choice."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "haiku-test"  # the walk's live choice
        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        slot._refusal_pick_gen = slot._model_pick_gen
        slot._active_fallback_model = "haiku-test"
        slot._fallback_primary_model = "opus-test"  # walk departed FROM our candidate

        from kiro_crew.dashboard.chat_runner import _restore_refusal_fallback

        await _restore_refusal_fallback(slot, client)

        assert (
            slot._fallback_primary_model == "fable-5"
        ), "walk restore target must become the refusal primary"
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_candidate == ""
        client.set_model.assert_not_awaited()
        assert client.served_model == "haiku-test", "the walk's live model must not be touched"

    @pytest.mark.asyncio
    async def test_alias_pick_epoch_blocks_restore(self, tmp_path, monkeypatch):
        """An explicit pick through a session ALIAS bumps the shared client's
        pick epoch without touching this slot's generation; the restore must
        drop its record without moving the model — otherwise it silently
        overrides the pick the user made through the other slot. Cross-layer
        on purpose: the pick handler stamps the INNER wrapped client while
        this restore holds the provider WRAPPER, so both ends must resolve
        the same epoch host."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "opus-test"  # still on the candidate

        class _InnerClient:
            """Non-callable inner wrapped client, as pick_epoch_host resolves."""

        inner = _InnerClient()
        client.client = inner  # the wrapper's .client, like AcpProvider.client
        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        slot._refusal_pick_gen = slot._model_pick_gen  # THIS slot saw no pick
        slot._refusal_client_pick_epoch = 0
        # The alias's live-switch pick stamps the INNER client (the handler's
        # side of pick_epoch_host) — even a pick of exactly the candidate,
        # which model equality alone cannot see.
        inner._explicit_pick_epoch = 1

        from kiro_crew.dashboard.chat_runner import _restore_refusal_fallback

        await _restore_refusal_fallback(slot, client)

        client.set_model.assert_not_awaited()
        assert client.served_model == "opus-test", "the alias's pick must stand"
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_candidate == ""

    @pytest.mark.asyncio
    async def test_pick_during_restore_set_model_await_is_not_overwritten(
        self, tmp_path, monkeypatch
    ):
        """A pick landing INSIDE the restore's ``set_model`` await must win.

        The epoch snapshot is read once before the await, so without
        serialization an alias pick that lands while the restore's RPC is in
        flight is applied first and then silently overwritten when the
        restore's ``set_model(primary)`` completes LAST. The restore now
        holds the session-scoped switch lock across the whole
        check-and-restore, so the pick (which takes the same lock) is
        strictly ordered after it and its choice stands.
        """
        import asyncio as _asyncio

        from kiro_crew.dashboard.chat_runner import _restore_refusal_fallback
        from kiro_crew.dashboard.chat_utils import effective_session_key
        from kiro_crew.llm_helpers import pick_epoch_host, slot_switch_session_lock

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "opus-test"  # wire is on the candidate

        # Gate set_model so the RPC "lands" only on release — modelling the
        # in-flight window the finding names: the model applies when the
        # await COMPLETES, so an unserialized concurrent pick applies first
        # and the restore's write lands last.
        restore_entered = _asyncio.Event()
        release_restore = _asyncio.Event()

        async def _gated_set_model(model: str) -> None:
            restore_entered.set()
            await release_restore.wait()
            client.served_model = model

        client.set_model = AsyncMock(side_effect=_gated_set_model)

        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        slot._refusal_pick_gen = slot._model_pick_gen
        slot._refusal_client_pick_epoch = 0  # epoch unmoved when restore starts

        restore_task = _asyncio.create_task(_restore_refusal_fallback(slot, client))
        await await_until(restore_entered, what="the restore reaching set_model")

        async def _alias_pick() -> None:
            # The pick handler's essential moves, under the SAME session
            # lock it takes at its acquisition site: bump the shared epoch,
            # apply the model on the wire.
            async with slot_switch_session_lock(effective_session_key(slot)):
                host = pick_epoch_host(client)
                host._explicit_pick_epoch = getattr(host, "_explicit_pick_epoch", 0) + 1
                client.served_model = "user-picked"

        pick_task = _asyncio.create_task(_alias_pick())
        for _ in range(10):  # let the pick reach the lock (or, unfixed, run)
            await _asyncio.sleep(0)
        release_restore.set()
        await await_until(restore_task, what="the restore finishing")
        await await_until(pick_task, what="the aliased pick finishing")

        client.set_model.assert_awaited_once_with("fable-5")
        assert client.served_model == "user-picked", (
            "the pick that landed during the restore's set_model await must "
            "apply last — the restore overwrote the user's selection"
        )

    @pytest.mark.asyncio
    async def test_pick_during_swap_set_model_await_is_not_reverted(self, tmp_path, monkeypatch):
        """A pick landing INSIDE the swap's ``set_model`` await must survive.

        The swap snapshots the client pick epoch AFTER its await, so without
        the session lock an alias pick landing while the swap's RPC is in
        flight is folded into the snapshot — the restore's alias-pick guard
        then reads equal and restores the primary over the user's choice.
        The swap now holds the session-scoped switch lock across the await
        and the snapshot (same order as the restore), so the pick is
        strictly ordered after the snapshot and the restore drops its
        record instead of touching the model.
        """
        import asyncio as _asyncio

        from kiro_crew.dashboard.chat_runner import (
            _refusal_fallback_swap,
            _restore_refusal_fallback,
        )
        from kiro_crew.dashboard.chat_utils import effective_session_key
        from kiro_crew.llm_helpers import pick_epoch_host, slot_switch_session_lock

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "fable-5"  # wire on the primary pre-swap

        swap_entered = _asyncio.Event()
        release_swap = _asyncio.Event()

        async def _gated_set_model(model: str) -> None:
            swap_entered.set()
            await release_swap.wait()
            client.served_model = model

        client.set_model = AsyncMock(side_effect=_gated_set_model)

        swap_task = _asyncio.create_task(_refusal_fallback_swap(slot, client, "opus-test"))
        await await_until(swap_entered, what="the swap reaching set_model")

        async def _alias_pick() -> None:
            # The pick handler's essential moves, under the SAME session
            # lock it takes at its acquisition site: bump the shared epoch,
            # apply the model on the wire.
            async with slot_switch_session_lock(effective_session_key(slot)):
                host = pick_epoch_host(client)
                host._explicit_pick_epoch = getattr(host, "_explicit_pick_epoch", 0) + 1
                client.served_model = "user-picked"

        pick_task = _asyncio.create_task(_alias_pick())
        for _ in range(10):  # let the pick reach the lock (or, unfixed, run)
            await _asyncio.sleep(0)
        release_swap.set()
        assert await await_until(swap_task, what="the swap finishing") == "fable-5"
        await await_until(pick_task, what="the aliased pick finishing")

        # The retry ran; the next genuine turn restores. The record must be
        # dropped on the epoch mismatch without touching the model.
        client.set_model = AsyncMock()
        await _restore_refusal_fallback(slot, client)

        client.set_model.assert_not_awaited()
        assert client.served_model == "user-picked", (
            "the pick that landed during the swap's set_model await must "
            "stand — the restore reverted the user's selection"
        )
        assert slot._refusal_fallback_primary == ""

    @pytest.mark.asyncio
    async def test_drain_purges_replay_after_stop(self, tmp_path, monkeypatch):
        """A Stop landing between the replay's enqueue and the drain kills the
        replay: the index-0 entry must never dispatch superseded work."""
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.chat_runner import (
            SYNTHETIC_RECOVERY_KIND,
            _start_next_queued_turn,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        qid = slot.queue_insert(0, "retry me", kind=SYNTHETIC_RECOVERY_KIND)
        slot.replays.arm(
            ReplayFamily.CONTENT_FILTER,
            entry_id=qid,
            session_key=slot._refusal_fallback_session_key,
            stop_gen=slot._stop_generation,
            session_stop_gen=0,
        )
        # The Stop that landed while the replay waited.
        slot._stop_generation += 1
        _dispatched = MagicMock()
        monkeypatch.setattr(chat_runner, "_run_chat", _dispatched)

        assert await _start_next_queued_turn(state, slot) is False
        _dispatched.assert_not_called()
        assert all(q.get("id") != qid for q in slot._queue)
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert any(
            "Content-filter retry cancelled" in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "notice"
        ), "the transcript must say the retry was cancelled"

    @pytest.mark.asyncio
    async def test_drain_purges_replay_when_user_correction_queued(self, tmp_path, monkeypatch):
        """A user correction queued behind the index-0 replay supersedes it:
        the correction dispatches, the replay dies."""
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.chat_runner import (
            SYNTHETIC_RECOVERY_KIND,
            _start_next_queued_turn,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        qid = slot.queue_insert(0, "retry me", kind=SYNTHETIC_RECOVERY_KIND)
        slot.replays.arm(
            ReplayFamily.CONTENT_FILTER,
            entry_id=qid,
            session_key=slot._refusal_fallback_session_key,
            stop_gen=slot._stop_generation,
            session_stop_gen=0,
        )
        # The correction the refusal card asked for, queued while the replay waited.
        slot.queue_insert(1, "actually, do this instead")

        _seen: list[str] = []

        def _fake_run_chat(_state, _slot, message, **kwargs):
            _seen.append(message)

            async def _noop():
                return None

            return _noop()

        monkeypatch.setattr(chat_runner, "_run_chat", _fake_run_chat)

        assert await _start_next_queued_turn(state, slot) is True
        assert _seen == [
            "actually, do this instead"
        ], "the correction must dispatch, never the purged replay"
        assert all(q.get("id") != qid for q in slot._queue)
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)

    @pytest.mark.asyncio
    async def test_replay_carries_original_turn_attachments(self, tmp_path, monkeypatch):
        """The replay is the SAME turn again: a refused message with files
        retries with its files riding the queue entry's meta."""
        from kiro_crew.dashboard.state import _ChatSlot

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        _recovery_meta: list[dict] = []
        _orig_insert = _ChatSlot.queue_insert

        def _spy_insert(self_slot, index, content, kind="", payload="", meta=None, **kw):
            if kind:
                _recovery_meta.append(dict(meta or {}))
            return _orig_insert(
                self_slot, index, content, kind=kind, payload=payload, meta=meta, **kw
            )

        monkeypatch.setattr(_ChatSlot, "queue_insert", _spy_insert)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello", _attachments=["/tmp/evidence.png"])
        if slot.task is not None:
            await slot.task

        assert _recovery_meta, "the refusal replay was never queued"
        assert _recovery_meta[0].get("files") == [
            "/tmp/evidence.png"
        ], "the replay entry must carry the refused turn's attachment list"

    @pytest.mark.asyncio
    async def test_replay_preserves_folder_attachment_typing(self, tmp_path, monkeypatch):
        """A folder attachment replays under ``dirs``: rebucketing the flat
        list under ``files`` would retype it and resolve its
        ``[attached_dir N]`` marker against the wrong list."""
        from kiro_crew.dashboard.state import _ChatSlot

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._refusal_events(category="CYBER"))
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        _recovery_meta: list[dict] = []
        _orig_insert = _ChatSlot.queue_insert

        def _spy_insert(self_slot, index, content, kind="", payload="", meta=None, **kw):
            if kind:
                _recovery_meta.append(dict(meta or {}))
            return _orig_insert(
                self_slot, index, content, kind=kind, payload=payload, meta=meta, **kw
            )

        monkeypatch.setattr(_ChatSlot, "queue_insert", _spy_insert)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(
            state,
            slot,
            "hello",
            _attachments=["/tmp/report.pdf", "/tmp/project-dir"],
            _attachment_meta={
                "files": ["/tmp/report.pdf"],
                "dirs": ["/tmp/project-dir"],
            },
        )
        if slot.task is not None:
            await slot.task

        assert _recovery_meta, "the refusal replay was never queued"
        assert _recovery_meta[0].get("dirs") == [
            "/tmp/project-dir"
        ], "the folder attachment must replay under dirs, not be retyped as a file"
        assert _recovery_meta[0].get("files") == [
            "/tmp/report.pdf"
        ], "the file attachment must stay under files with no dir mixed in"

    @pytest.mark.asyncio
    async def test_drain_purges_replay_when_binding_changes(self, tmp_path, monkeypatch):
        """A relink moving the slot to a DIFFERENT session between the
        replay's enqueue and the drain kills the replay: it belongs to the
        session the refused turn ran on, never the newly bound one. (The
        unbound->bound case is already dropped by the admission sweep's
        fail-closed booleans; a bound->bound relink is invisible to them —
        this branch is what catches it and clears the episode record.)"""
        from kiro_crew.dashboard import chat_runner, session_control
        from kiro_crew.dashboard.chat_runner import (
            SYNTHETIC_RECOVERY_KIND,
            _start_next_queued_turn,
        )

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        # The turn ran on binding A; stamp admission containment as the real
        # producer (_queue_recovery) does, so the admission sweep sees an
        # unchanged boolean set and lets the entry through to the replay gate.
        slot.linked_session_key = "session-a"
        slot._refusal_fallback_session_key = "session-a"
        qid = slot.queue_insert(
            0,
            "retry me",
            kind=SYNTHETIC_RECOVERY_KIND,
            meta=session_control.containment_meta(state, slot),
        )
        slot.replays.arm(
            ReplayFamily.CONTENT_FILTER,
            entry_id=qid,
            session_key=slot._refusal_fallback_session_key,
            stop_gen=slot._stop_generation,
            session_stop_gen=0,
        )
        # The relink that landed while the replay waited.
        slot.linked_session_key = "session-b"
        _dispatched = MagicMock()
        monkeypatch.setattr(chat_runner, "_run_chat", _dispatched)

        assert await _start_next_queued_turn(state, slot) is False
        _dispatched.assert_not_called()
        assert all(q.get("id") != qid for q in slot._queue)
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER), (
            "the dispatch-gate record must die with the replay — a later "
            "identical message on session-b is a genuine turn, not a retry"
        )
        assert any(
            "this chat moved to another session" in m.get("content", "")
            for m in slot.messages
            if m.get("role") == "notice"
        ), "the transcript must say the retry died with the old binding"

    @pytest.mark.asyncio
    async def test_swept_replay_clears_dispatch_gate_record(self, tmp_path, monkeypatch):
        """A replay entry the admission sweep removed leaves the queue-id
        pointing at nothing. The drain's replay gate must clear the
        dispatch-gate text with it: otherwise an identical later resend
        matches as the retry — running on the fallback with the restore
        probe skipped and the one-retry allowance kept spent."""
        from kiro_crew.dashboard import chat_runner, session_control
        from kiro_crew.dashboard.chat_runner import _start_next_queued_turn

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        # Episode state as the swap left it — but the replay entry itself is
        # absent (the admission sweep dropped it on a containment change).
        slot.replays.arm(
            ReplayFamily.CONTENT_FILTER,
            entry_id="qid-swept-away",
            session_key="",
            stop_gen=0,
            session_stop_gen=0,
        )
        slot._refusal_fallback_attempted = True
        # The user's identical resend queued behind the (now gone) replay.
        slot.queue_insert(
            0,
            "retry me",
            meta=session_control.containment_meta(state, slot),
        )
        _dispatched = MagicMock()
        monkeypatch.setattr(chat_runner, "_run_chat", _dispatched)

        await _start_next_queued_turn(state, slot)
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER), (
            "the dispatch-gate record must die with the swept replay — the "
            "identical resend is a genuine turn, not a mistaken retry"
        )

    @pytest.mark.asyncio
    async def test_swap_locks_the_turns_captured_binding(self, tmp_path, monkeypatch):
        """The swap serializes on the TURN's captured key and stamps it: a
        binding that moved before the refusal arrived must not re-route the
        lock domain."""
        import asyncio as _asyncio

        from kiro_crew.dashboard.chat_runner import _refusal_fallback_swap
        from kiro_crew.llm_helpers import slot_switch_session_lock

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        # The slot was bound AFTER the turn started: live derivation now
        # disagrees with the turn's captured key.
        slot.linked_session_key = "cron:job-123"
        client = self._make_refusing_client([])
        client.served_model = "fable-5"

        async def _apply_set_model(model: str) -> None:
            client.served_model = model

        client.set_model = AsyncMock(side_effect=_apply_set_model)

        _turn_key = "dash:turn-key"
        _held = slot_switch_session_lock(_turn_key)
        await _held.acquire()
        try:
            swap_task = _asyncio.create_task(
                _refusal_fallback_swap(slot, client, "opus-test", session_key=_turn_key)
            )
            for _ in range(10):
                await _asyncio.sleep(0)
            assert not swap_task.done(), (
                "the swap must serialize on the turn's captured key, "
                "not the newly bound session's"
            )
        finally:
            _held.release()
        assert await await_until(swap_task, what="the swap finishing") == "fable-5"
        assert (
            slot._refusal_fallback_session_key == _turn_key
        ), "the swap must stamp the binding it ran under"

    @pytest.mark.asyncio
    async def test_restore_locks_the_swap_recorded_binding(self, tmp_path, monkeypatch):
        """The restore serializes on the binding the SWAP recorded — a rebind
        between swap and restore must not split the two seams into disjoint
        lock domains."""
        import asyncio as _asyncio

        from kiro_crew.dashboard.chat_runner import _restore_refusal_fallback
        from kiro_crew.llm_helpers import slot_switch_session_lock

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "opus-test"
        client.set_model = AsyncMock()

        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        slot._refusal_pick_gen = slot._model_pick_gen
        slot._refusal_client_pick_epoch = 0
        slot._refusal_fallback_session_key = "dash:turn-key"
        # The rebind that landed after the swap.
        slot.linked_session_key = "cron:job-123"

        _held = slot_switch_session_lock("dash:turn-key")
        await _held.acquire()
        try:
            restore_task = _asyncio.create_task(_restore_refusal_fallback(slot, client))
            for _ in range(10):
                await _asyncio.sleep(0)
            assert not restore_task.done(), (
                "the restore must serialize on the swap's recorded key, "
                "not the rebound session's"
            )
        finally:
            _held.release()
        await await_until(restore_task, what="the restore finishing")
        # Once inside the lock, the rebind guard rules: the record belongs to
        # the recorded session, and ``client`` now serves the rebound one —
        # the restore must DROP the record, never move the rebound session's
        # model to a primary it never chose.
        client.set_model.assert_not_awaited()
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_candidate == ""

    @pytest.mark.asyncio
    async def test_replay_recognized_after_credential_redaction(self, tmp_path, monkeypatch):
        """Replay identity is the queue entry, not the text: the drain redacts
        credentials AFTER the swap records the raw message, so a refused
        message carrying a token must still be recognized as the one retry —
        exactly one swap, no second 'retrying once' notice, no restore probe
        running the replay back on the primary that just refused."""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        events = self._refusal_events(category="CYBER")
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        secret_msg = "use token ghp_" + "A" * 36 + " to fetch the repo"
        await _run_chat(state, slot, secret_msg)

        assert slot.task is not None, "queued replay was not dispatched"
        await slot.task

        retry_notices = [
            m
            for m in slot.messages
            if m.get("role") == "error" and "retrying once on 'opus-test'" in m.get("content", "")
        ]
        assert len(retry_notices) == 1, (
            "the redacted replay must be recognized as the retry — a second "
            f"notice means it re-armed as a genuine turn: {retry_notices}"
        )
        # Exactly one model move: the swap. No restore probe ran inside the
        # replay (unrecognized, it would move the model back to the primary
        # mid-retry), and no second swap happened.
        client.set_model.assert_awaited_once_with("opus-test")
        assert slot._refusal_fallback_attempted is True, "allowance must stay spent"

    @staticmethod
    def _arm_refusal_replay(slot, *, session_key: str | None = None) -> None:
        """The slot state a refusal swap leaves behind before its replay runs."""
        from kiro_crew.dashboard.chat_utils import effective_session_key

        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        slot._refusal_fallback_attempted = True
        slot._refusal_fallback_session_key = (
            effective_session_key(slot) if session_key is None else session_key
        )
        slot.replays.arm(
            ReplayFamily.CONTENT_FILTER,
            entry_id="q-retry",
            session_key=slot._refusal_fallback_session_key,
            stop_gen=getattr(slot, "_stop_generation", 0),
            session_stop_gen=0,
        )

    @pytest.mark.asyncio
    async def test_replay_consume_aborts_after_stop(self, tmp_path, monkeypatch):
        """A Stop completing between the drain's dequeue validation and the
        replay turn's consume must abort the replay: the current stop
        generation differs from the recorded snapshot, so the stale content
        never runs. With nothing queued, the exit guard ends the cycle: one
        serializable chat_done, the done row, and the slot's task released."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        # Snapshot taken at enqueue…
        self._arm_refusal_replay(slot)
        # …then a Stop pressed AND resolved in the spawn→consume window.
        slot._stop_generation = getattr(slot, "_stop_generation", 0) + 1

        await run_as_slot_task(
            slot,
            _run_chat(state, slot, "retry me", _replay=frozenset({ReplayFamily.CONTENT_FILTER})),
        )

        cancelled = [
            m
            for m in slot.messages
            if m.get("role") == "notice"
            and "Content-filter retry cancelled" in m.get("content", "")
            and "the turn was stopped." in m.get("content", "")
        ]
        assert len(cancelled) == 1, f"expected the consume-time abort notice, got {slot.messages}"
        client.set_model.assert_not_awaited()
        state.sessions.get_or_create.assert_not_awaited()
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert slot._refusal_fallback_attempted is True, "allowance stays spent on abort"
        frames = chat_done_frames(state)
        assert len(frames) == 1 and isinstance(frames[0], dict), frames
        assert slot.messages[-1].get("role") == "done"
        assert slot.task is None

    @pytest.mark.asyncio
    async def test_replay_consume_aborts_after_rebind(self, tmp_path, monkeypatch):
        """A session rebind landing between dequeue and consume must abort the
        replay — the stale content belongs to the OLD session and must not
        enter the newly bound one."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        # The binding the swap recorded differs from the slot's live binding.
        self._arm_refusal_replay(slot, session_key="dash:old-session")

        await _run_chat(state, slot, "retry me", _replay=frozenset({ReplayFamily.CONTENT_FILTER}))

        cancelled = [
            m
            for m in slot.messages
            if m.get("role") == "notice"
            and "Content-filter retry cancelled" in m.get("content", "")
            and "this chat moved to another session." in m.get("content", "")
        ]
        assert len(cancelled) == 1, f"expected the rebind abort notice, got {slot.messages}"
        client.set_model.assert_not_awaited()
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)

    @pytest.mark.asyncio
    async def test_synthetic_recovery_dispatch_does_not_rearm(self, tmp_path, monkeypatch):
        """A kind-tagged recovery requeue of the user's OWN words is the same
        turn retried, not a genuine new message: it must not re-arm the
        one-retry allowance (re-arming lets a crashing fallback replay cycle
        primary-refusal → swap → crash indefinitely)."""
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([LLMEvent(kind=EVENT_COMPLETE)])
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        slot._refusal_fallback_attempted = True

        await _run_chat(state, slot, "my own words again", _synthetic_recovery_turn=True)

        assert slot._refusal_fallback_attempted is True, (
            "a synthetic recovery requeue must not re-arm the refusal-retry "
            "allowance — only a genuine user-origin dispatch does"
        )

    @pytest.mark.asyncio
    async def test_synthetic_recovery_dispatch_skips_restore_probe(self, tmp_path, monkeypatch):
        """A kind-tagged recovery requeue of the user's OWN words must finish
        on the model that produced the original attempt: the restore probe at
        turn start must not move the session back to the primary mid-recovery.
        The fixed-text membership check cannot recognize a requeued user
        message, so the gate needs the structural signal."""
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([LLMEvent(kind=EVENT_COMPLETE)])
        client.served_model = "opus-test"  # still on the fallback candidate
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.dashboard.chat_utils import effective_session_key

        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        slot._refusal_fallback_attempted = True
        slot._refusal_pick_gen = slot._model_pick_gen
        slot._refusal_client_pick_epoch = 0
        slot._refusal_fallback_session_key = effective_session_key(slot)

        await _run_chat(state, slot, "my own words again", _synthetic_recovery_turn=True)

        client.set_model.assert_not_awaited()
        assert slot._refusal_fallback_primary == "fable-5", (
            "the restore record must survive a recovery dispatch — the next "
            "genuine turn owns the restore"
        )

    @pytest.mark.asyncio
    async def test_restore_dropped_when_slot_rebound(self, tmp_path, monkeypatch):
        """A restore record made under session A must never be applied through
        the provider of a session the slot has since rebound to: the record is
        dropped without touching the model."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "opus-test"
        client.set_model = AsyncMock()

        slot._refusal_fallback_primary = "fable-5"
        slot._refusal_fallback_candidate = "opus-test"
        slot._refusal_pick_gen = slot._model_pick_gen
        slot._refusal_client_pick_epoch = 0
        # Swap recorded session A; the slot has since rebound elsewhere.
        slot._refusal_fallback_session_key = "dash:session-a"

        from kiro_crew.dashboard.chat_runner import _restore_refusal_fallback

        await _restore_refusal_fallback(slot, client)

        client.set_model.assert_not_awaited()
        assert slot._refusal_fallback_primary == ""
        assert slot._refusal_fallback_candidate == ""

    @pytest.mark.asyncio
    async def test_recovery_requeue_of_retry_turn_carries_replay_identity(
        self, tmp_path, monkeypatch
    ):
        """A pre-output process failure during the retry turn requeues the
        user's message verbatim through a GENERIC recovery family. The requeue
        must carry the refusal-replay identity and fresh stop snapshots onto
        the new entry — otherwise the requeued entry escapes the drain's
        refusal-specific validation, and a correction queued in the
        requeue-to-dispatch window loses to superseded work."""
        from kiro_crew.acp.client import AcpError

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        state.sessions.reset = AsyncMock()

        async def _raise_dead(msg):
            raise AcpError("ACP process exited (code=-15)")
            yield  # make it an async generator

        client.stream = _raise_dead
        client.stream_command = _raise_dead
        client.shutdown = AsyncMock()

        # Keep the requeued entry IN the queue: the post-turn drain would
        # otherwise dequeue and spawn it immediately, racing the assertions.
        async def _no_drain(_state, _slot, **_kwargs) -> bool:
            return False

        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._start_next_queued_turn", _no_drain)

        from kiro_crew.dashboard.chat import _run_chat

        self._arm_refusal_replay(slot)

        await _run_chat(state, slot, "retry me", _replay=frozenset({ReplayFamily.CONTENT_FILTER}))

        requeued = [q for q in slot._queue if q.get("content") == "retry me"]
        assert requeued, f"expected the verbatim recovery requeue, queue={slot._queue}"
        assert slot.replays.entry_id(ReplayFamily.CONTENT_FILTER) == requeued[0]["id"], (
            "the requeued entry must carry the refusal-replay identity so the "
            "drain applies the refusal-specific validation at dispatch"
        )

        # Re-armed at the requeue's Stop counts: no Stop since, so it may run on
        # the session the swap ran under, and reads as moved anywhere else.
        def _view(session_key: str) -> LiveSlot:
            return LiveSlot(
                session_key=session_key,
                stop_generation=getattr(slot, "_stop_generation", 0),
                stopping=False,
                user_input=False,
                session_stop_generation=lambda _key: 0,
            )

        replay = ReplayFamily.CONTENT_FILTER
        assert not slot.replays.revalidate(
            replay, _view(slot._refusal_fallback_session_key)
        ).revoked
        assert slot.replays.revalidate(replay, _view("dash:other")).rebound

    @pytest.mark.asyncio
    async def test_swap_during_throttle_fallback_records_true_primary(self, tmp_path, monkeypatch):
        """A refusal landing while a THROTTLE fallback is serving must record
        the walk's true primary, not the wire model: the replay turn's
        throttle probe clears the walk's sticky state via moved-off, so a
        record naming the throttle candidate would strand the session on a
        model the user never chose, with no machinery left to recover."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client([])
        client.served_model = "haiku-throttle"  # wire = the throttle candidate

        # The throttle walk is actively serving and knows the true primary.
        slot._active_fallback_model = "haiku-throttle"
        slot._fallback_primary_model = "fable-5"

        from kiro_crew.dashboard.chat_runner import _refusal_fallback_swap
        from kiro_crew.dashboard.chat_utils import effective_session_key

        result = await _refusal_fallback_swap(
            slot, client, "opus-test", session_key=effective_session_key(slot)
        )

        assert result == "haiku-throttle", (
            "the return feeds the 'declined on X' notice — it must name the "
            "model that actually refused (the wire model)"
        )
        assert (
            slot._refusal_fallback_primary == "fable-5"
        ), "the record must name the user's model, not the throttle candidate"
        assert slot._refusal_fallback_candidate == "opus-test"

    @pytest.mark.asyncio
    async def test_replay_aborts_when_correction_queued_during_preparation(
        self, tmp_path, monkeypatch
    ):
        """The consume-seam supersession check runs at turn ENTRY, but turn
        preparation awaits (session acquisition, prompt assembly) before the
        stream opens. A correction queued in that window must abort the
        replay before any model call — a stale replay dispatching first could
        run destructive tools the correction cancelled."""
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(
            [
                LLMEvent(kind=EVENT_TEXT_CHUNK, text="stale replay output"),
                LLMEvent(kind=EVENT_COMPLETE),
            ]
        )

        _orig_get_or_create = AsyncMock(return_value=(client, True, False))

        async def _acquire_and_inject(*args, **kwargs):
            # A user correction lands DURING preparation — after the
            # consume-seam check, before the stream opens.
            slot.queue_insert(0, "actually do the other thing", kind="")
            return await _orig_get_or_create(*args, **kwargs)

        state.sessions.get_or_create = _acquire_and_inject

        from kiro_crew.dashboard.chat import _run_chat

        self._arm_refusal_replay(slot)

        await _run_chat(state, slot, "retry me", _replay=frozenset({ReplayFamily.CONTENT_FILTER}))

        cancelled = [
            m
            for m in slot.messages
            if m.get("role") == "notice"
            and "Content-filter retry cancelled" in m.get("content", "")
        ]
        assert len(cancelled) == 1, f"expected the pre-stream abort notice, got {slot.messages}"
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert not any(
            m.get("role") == "assistant" for m in slot.messages
        ), "the stale replay must never reach the model"

    @staticmethod
    def _answering_events():
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        return [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="follow-up answer"),
            LLMEvent(kind=EVENT_COMPLETE),
        ]

    @staticmethod
    def _arm_model_access_replay(slot, entry_id: str = "q-replay") -> None:
        from kiro_crew.dashboard.chat_utils import effective_session_key

        slot.replays.arm(
            ReplayFamily.MODEL_ACCESS,
            entry_id=entry_id,
            session_key=effective_session_key(slot),
            stop_gen=getattr(slot, "_stop_generation", 0),
            session_stop_gen=0,
        )

    @staticmethod
    def _arm_image_recovery(slot) -> None:
        from kiro_crew.dashboard.chat_utils import effective_session_key

        slot.replays.arm(
            ReplayFamily.IMAGE_HISTORY,
            entry_id="q-image",
            session_key=effective_session_key(slot),
            stop_gen=getattr(slot, "_stop_generation", 0),
            session_stop_gen=0,
        )

    _ARM = {
        "refusal": "_arm_refusal_replay",
        "model-access": "_arm_model_access_replay",
        "image": "_arm_image_recovery",
    }

    @pytest.mark.parametrize(
        ("family", "message", "kwargs", "notice"),
        [
            (
                "refusal",
                "retry me",
                {"_replay": frozenset({ReplayFamily.CONTENT_FILTER})},
                "Content-filter retry cancelled",
            ),
            (
                "model-access",
                "original prompt",
                {"_replay": frozenset({ReplayFamily.MODEL_ACCESS})},
                "Model-fallback retry cancelled",
            ),
            (
                "image",
                "original prompt",
                {"_replay": frozenset({ReplayFamily.IMAGE_HISTORY})},
                "Image-history recovery cancelled",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_superseded_replay_abort_runs_the_queued_followup(
        self, tmp_path, monkeypatch, family, message, kwargs, notice
    ):
        """A replay cancelled at its consume seam returns before ``_run_chat``'s
        try, so the exit guard hands off the queue: the newer message that
        superseded the replay is dispatched once, and only its turn sends the
        cycle's one terminal chat_done."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        prompts: list[str] = []
        client = self._make_refusing_client(self._answering_events(), prompts)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        getattr(self, self._ARM[family])(slot)
        slot.queue_insert(0, "actually do the other thing", kind="")

        replay = await run_as_slot_task(slot, _run_chat(state, slot, message, **kwargs))

        assert any(
            m.get("role") == "notice" and notice in m.get("content", "") for m in slot.messages
        ), f"expected the consume-time abort notice, got {slot.messages}"
        assert chat_done_frames(state) == [], "the cycle ended before the follow-up ran"
        await await_successor(slot, replay)

        assert not any(message in p for p in prompts), "the cancelled replay must not run"
        assert sum("actually do the other thing" in p for p in prompts) == 1, prompts
        assert not slot._queue
        assert slot.task is None
        frames = chat_done_frames(state)
        assert len(frames) == 1 and isinstance(frames[0], dict), frames

    @pytest.mark.asyncio
    async def test_real_refusal_replay_dropped_after_relink(self, tmp_path, monkeypatch):
        """End to end through the real refuse -> swap -> enqueue path: a relink that
        lands before the drain moves the replay's session away, so the drain drops
        it with the move notice instead of dispatching it onto the new binding."""
        from kiro_crew.acp.types import STOP_REASON_REFUSAL
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.linked_session_key = "session-a"
        prompts: list[str] = []
        client = self._make_refusing_client([])

        async def _stream(msg):
            prompts.append(str(msg))
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_REFUSAL)

        client.stream = _stream
        client.stream_command = _stream
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        real_drain = chat_runner._start_next_queued_turn

        async def _relink_then_drain(drain_state, drain_slot, **kwargs):
            # The relink that lands while the replay waits in the queue.
            drain_slot.linked_session_key = "session-b"
            return await real_drain(drain_state, drain_slot, **kwargs)

        monkeypatch.setattr(chat_runner, "_start_next_queued_turn", _relink_then_drain)

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")

        client.set_model.assert_awaited_once_with("opus-test")
        assert len(prompts) == 1, "the replay was dispatched onto the new binding"
        assert slot.task is None and slot._queue == []
        assert not slot.replays.armed(ReplayFamily.CONTENT_FILTER)
        assert any(
            m.get("role") == "notice"
            and m.get("content")
            == "ℹ️ Content-filter retry cancelled — this chat moved to another session."
            for m in slot.messages
        ), slot.messages

    @pytest.mark.asyncio
    async def test_real_refusal_replay_superseded_at_consume_runs_the_followup(
        self, tmp_path, monkeypatch
    ):
        """End to end through the real refuse -> swap -> enqueue -> drain path: a
        correction queued after the drain spawned the replay cancels it at the
        consume seam, and the correction runs next."""
        from kiro_crew.acp.types import STOP_REASON_REFUSAL
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._configured_refusal_fallback",
            lambda: "opus-test",
        )
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        prompts: list[str] = []
        client = self._make_refusing_client([])
        replies = [
            [LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_REFUSAL)],
            [LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok"), LLMEvent(kind=EVENT_COMPLETE)],
        ]

        async def _stream(msg):
            prompts.append(str(msg))
            for ev in replies[min(len(prompts), len(replies)) - 1]:
                yield ev

        client.stream = _stream
        client.stream_command = _stream
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "hello")
        replay = slot.task
        assert replay is not None, "the refusal replay was not dispatched"
        # The correction lands after the drain spawned the replay, before it runs.
        slot.queue_insert(0, "actually do the other thing", kind="")
        await await_until(replay, what="the refusal replay finishing")
        await await_successor(slot, replay)

        assert [p for p in prompts if "hello" in p] == [prompts[0]], "the replay ran"
        assert sum("actually do the other thing" in p for p in prompts) == 1, prompts
        assert any("Content-filter retry cancelled" in m.get("content", "") for m in slot.messages)
        assert not slot._queue and slot.task is None
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_aborted_replay_starts_a_pending_synthesis(self, tmp_path, monkeypatch):
        """The consume-seam exit never reached the turn's own memory-preparation
        barrier, and no shared preparation holds turns back, so an armed
        synthesis starts, as it would after any finished turn, and the cycle's
        chat_done is left to it."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        from kiro_crew.dashboard.chat_utils import SYNTHESIS_CLEAR

        state.subagents = MagicMock()
        slot = state.get_or_create_slot("s1")
        state.sessions.get_or_create = AsyncMock()
        synthesis = AsyncMock()
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._run_pending_synthesis", synthesis)
        # No child holds the synthesis back (the fire gate's own reads are
        # covered by its tests).
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner.synthesis_fire_verdict",
            AsyncMock(return_value=SYNTHESIS_CLEAR),
        )

        from kiro_crew.dashboard.chat import _run_chat

        self._arm_refusal_replay(slot)
        slot._stop_generation = getattr(slot, "_stop_generation", 0) + 1
        slot._pending_synthesis = True

        await _run_chat(state, slot, "retry me", _replay=frozenset({ReplayFamily.CONTENT_FILTER}))
        await await_until(slot.task, what="the slot's turn finishing")

        synthesis.assert_awaited_once()
        assert chat_done_frames(state) == []

    @pytest.mark.asyncio
    async def test_the_drain_names_the_model_access_replay_by_its_queue_id(
        self, tmp_path, monkeypatch
    ):
        """The swap records its replay's queue id, and the drain passes the
        replay identity only for that entry: the replay keeps the spent one-shot,
        while an ordinary entry queued behind it refreshes it."""
        from kiro_crew.dashboard import chat_runner

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        prompts: list[str] = []
        client = self._make_refusing_client(self._answering_events(), prompts)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        self._arm_model_access_replay(
            slot,
            slot.queue_insert(0, "original prompt", kind=chat_runner.SYNTHETIC_RECOVERY_KIND),
        )
        slot._model_access_fallback_used = True
        seen: list[bool] = []
        real_stream = client.stream

        async def _stream(msg):
            seen.append(slot._model_access_fallback_used)
            async for event in real_stream(msg):
                yield event

        client.stream = _stream
        client.stream_command = _stream

        assert await chat_runner._start_next_queued_turn(state, slot)
        await await_successor(slot, None)

        assert seen == [True], "the replay turn must keep the one-shot it was queued under"
        assert not slot.replays.armed(ReplayFamily.MODEL_ACCESS)

        # A genuine entry is not the replay, so its turn refreshes the one-shot.
        slot.queue_insert(0, "a fresh message", kind="")
        assert await chat_runner._start_next_queued_turn(state, slot)
        await await_successor(slot, None)
        assert seen == [True, False]

    @pytest.mark.parametrize("caller", ["direct-send", "nudge-wake"])
    @pytest.mark.asyncio
    async def test_a_stale_model_access_record_does_not_cancel_the_next_turn(
        self, tmp_path, monkeypatch, caller
    ):
        """A record whose entry left the queue another way (a hard kill's clear,
        the stop counter moved since) is not what identifies a replay: only the
        drain's queue-id match does. So the next turn, a direct send or a nudge
        wake, runs normally."""
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        prompts: list[str] = []
        client = self._make_refusing_client(self._answering_events(), prompts)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        self._arm_model_access_replay(slot, "q-cleared-by-the-kill")
        slot._stop_generation = getattr(slot, "_stop_generation", 0) + 1

        extra = (
            {"_directive_self_wake": True, "_turn_actor": "autonudge"}
            if caller == "nudge-wake"
            else {}
        )
        await _run_chat(state, slot, "the next message", **extra)

        assert any("the next message" in p for p in prompts), prompts
        assert not any("retry cancelled" in m.get("content", "") for m in slot.messages)

    @pytest.mark.asyncio
    async def test_a_synthetic_text_replay_consumes_the_model_access_record(
        self, tmp_path, monkeypatch
    ):
        """A swap during a synthetic recovery turn replays the synthetic text. The
        replay is still recognized (by queue id, not text), so it consumes the
        record instead of leaving it for the next turn."""
        from kiro_crew.dashboard.chat_runner import _CONN_RECOVER_MSG

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_refusing_client(self._answering_events())
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        from kiro_crew.dashboard.chat import _run_chat

        self._arm_model_access_replay(slot)

        await _run_chat(
            state, slot, _CONN_RECOVER_MSG, _replay=frozenset({ReplayFamily.MODEL_ACCESS})
        )

        assert not slot.replays.armed(ReplayFamily.MODEL_ACCESS)


class TestRunChatEarlyExitHandOff:
    """Every ``_run_chat`` exit runs the turn's tail exactly once.

    A local slash command answers without a provider and returns before the
    turn's ``finally``. The exit guard runs the tail for it, so a message queued
    behind the command runs next, the cycle sends exactly one terminal
    chat_done, and the slot's task is released -- and the tail takes its policy
    from what the turn and the slot recorded, never from where the turn
    stopped."""

    _make_state_for_run_chat = staticmethod(TestRunChatModelRefusal._make_state_for_run_chat)

    def _setup(self, tmp_path, monkeypatch, events=None):
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        prompts: list[str] = []
        client = TestRunChatRefusalFallback._make_refusing_client(
            TestRunChatRefusalFallback._answering_events() if events is None else events, prompts
        )
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.get_instance", lambda: None)
        state.push_refresh = MagicMock()
        state.refresh_slot_source_status = MagicMock()
        return state, slot, prompts

    @staticmethod
    def _run(state, slot, message, **kwargs):
        from kiro_crew.dashboard.chat import _run_chat

        return run_as_slot_task(slot, _run_chat(state, slot, message, **kwargs))

    @staticmethod
    def _count_nudge_notifies(monkeypatch) -> list[dict]:
        notifies: list[dict] = []
        nudge = MagicMock()
        nudge.notify_turn_complete = MagicMock(side_effect=lambda _key, **kw: notifies.append(kw))
        monkeypatch.setattr("kiro_crew.autonudge.get_instance", lambda: nudge)
        return notifies

    @pytest.mark.parametrize(
        "command", ["/quit", "/goal status", "/workflow", "/prompts", "/prompts get no-such-prompt"]
    )
    @pytest.mark.asyncio
    async def test_a_local_command_runs_the_message_queued_behind_it(
        self, tmp_path, monkeypatch, command
    ):
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        slot.queue_insert(0, "do X please", kind="")

        first = await self._run(state, slot, command)
        await await_successor(slot, first)

        assert sum("do X please" in p for p in prompts) == 1, prompts
        assert not any(command in p for p in prompts), "a local command reached the provider"
        assert not slot._queue
        assert slot.task is None
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_a_lone_local_command_ends_the_cycle_without_titling(self, tmp_path, monkeypatch):
        """No provider saw the cycle, so there is no reply to title or summarize
        and no remote work to re-read, and the session list is left alone."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        notifies = self._count_nudge_notifies(monkeypatch)
        title = AsyncMock()
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.title_then_refresh", title)

        await self._run(state, slot, "/prompts")

        assert prompts == []
        assert slot.task is None
        assert [m.get("role") for m in slot.messages][-2:] == ["assistant", "done"]
        frames = chat_done_frames(state)
        assert len(frames) == 1 and isinstance(frames[0], dict), frames
        state.push_refresh.assert_not_called()
        state.refresh_slot_source_status.assert_not_called()
        title.assert_not_called()
        assert len(notifies) == 1, "the AutoNudge timer re-arms once per exit"

    @pytest.mark.asyncio
    async def test_a_provider_turn_handing_off_to_a_local_command_still_titles(
        self, tmp_path, monkeypatch
    ):
        """Titling belongs to the cycle: a provider reply followed by a queued
        local command is still titled when the command ends the cycle."""
        state, slot, _prompts = self._setup(tmp_path, monkeypatch)
        title = AsyncMock()
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.title_then_refresh", title)
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner.generate_session_summary", AsyncMock())
        slot.queue_insert(0, "/prompts", kind="")

        first = await self._run(state, slot, "answer me")
        await await_successor(slot, first)
        for task in list(state._background_tasks):
            await await_until(task, what="a background task finishing")

        title.assert_awaited_once()
        state.refresh_slot_source_status.assert_called_once_with(slot.key)
        assert slot._cycle_reached_provider is False

    @pytest.mark.parametrize("command", ["/quit", "/goal status", "/prompts"])
    @pytest.mark.asyncio
    async def test_a_local_command_keeps_a_queue_held_after_a_sign_in_failure(
        self, tmp_path, monkeypatch, command
    ):
        """The command learned nothing about sign-in, so the slot keeps the
        outcome the last turn recorded and the held queue stays held: draining
        it would walk each prompt into the same signed-out CLI."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        slot._last_turn_auth_required = True
        slot.queue_insert(0, "held until login", kind="")

        await self._run(state, slot, command)

        assert prompts == []
        assert [q.get("content") for q in slot._queue] == ["held until login"]
        assert slot._last_turn_auth_required is True
        assert slot.task is None
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_a_local_command_keeps_the_queue_while_memory_preparation_runs(
        self, tmp_path, monkeypatch
    ):
        """The command never reached the barrier, so the shared preparation task
        decides: while it still runs, a drained prompt would only wait at the
        same barrier, so the queue is held."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        preparation = asyncio.get_running_loop().create_future()
        state.memory_startup_task = preparation
        slot.queue_insert(0, "after memory is ready", kind="")
        try:
            await self._run(state, slot, "/prompts")
        finally:
            preparation.cancel()

        assert prompts == []
        assert [q.get("content") for q in slot._queue] == ["after memory is ready"]
        assert slot.task is None
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_a_refused_command_leaves_a_deferred_reset_armed(self, tmp_path, monkeypatch):
        """A turn that acquired no session consumes no deferred reset: a refused
        ``/compact`` or ``/quit`` behaves as if the turn never started."""
        state, slot, _prompts = self._setup(tmp_path, monkeypatch)
        consume = AsyncMock(return_value=False)
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._consume_pending_reset", consume)

        await self._run(state, slot, "/quit")

        consume.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_setup_error_holds_the_queue(self, tmp_path, monkeypatch):
        """An exception above the main try comes from setup the next queued
        entry would run through too, so the cycle ends with the queue intact
        rather than failing every entry in turn, and the error propagates."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._discard_stale_decision",
            MagicMock(side_effect=RuntimeError("setup broke")),
        )
        slot.queue_insert(0, "second", kind="")
        slot.queue_insert(1, "third", kind="")

        with pytest.raises(RuntimeError, match="setup broke"):
            await self._run(state, slot, "first")

        assert prompts == []
        assert [q.get("content") for q in slot._queue] == ["second", "third"]
        assert slot.task is None
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_a_cancel_cutting_the_finally_short_keeps_the_turns_own_policy(
        self, tmp_path, monkeypatch
    ):
        """A second cancel landing on an await in the finally skips the rest of
        it. The outcome was recorded first, so the guard runs the tail with this
        turn's policy: a sign-in failure still holds the queue and is still
        recorded, and the AutoNudge timer re-arms exactly once, with its facts."""
        from kiro_crew.acp.transport_errors import AcpAuthRequired

        async def _signed_out(_msg):
            raise AcpAuthRequired("Please sign in")
            yield  # pragma: no cover - makes this an async generator

        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        client = TestRunChatRefusalFallback._make_refusing_client([])
        client.stream = _signed_out
        client.stream_command = _signed_out
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        notifies = self._count_nudge_notifies(monkeypatch)

        async def _cancelled_reset(_key, **_kw):
            raise asyncio.CancelledError

        state.sessions.reset = AsyncMock(side_effect=_cancelled_reset)
        slot.queue_insert(0, "held until login", kind="")

        with pytest.raises(asyncio.CancelledError):
            await self._run(state, slot, "first")

        assert [q.get("content") for q in slot._queue] == ["held until login"]
        assert slot._last_turn_auth_required is True
        assert len(notifies) == 1 and "tool_calls" in notifies[0], notifies
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_a_cancel_inside_the_tail_still_hands_off_the_queue(self, tmp_path, monkeypatch):
        """A cancel landing while the tail awaits its deferred-reset consume (a
        turn deadline) is deferred until the hand-off ran: the queued message
        still runs, and the turn then ends cancelled."""
        from kiro_crew.dashboard import chat_runner

        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        entered = asyncio.Event()
        release = asyncio.Event()
        parked: list[bool] = []

        async def _consume(_state, _slot, *, allow_discard=False, **_kw):
            if allow_discard and not parked:
                parked.append(True)
                entered.set()
                await release.wait()
            return False

        monkeypatch.setattr(chat_runner, "_consume_pending_reset", _consume)
        slot.queue_insert(0, "do X please", kind="")

        runner = asyncio.ensure_future(self._run(state, slot, "answer me"))
        await await_until(entered, what="the turn reaching its gate")
        first = slot.task
        first.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await runner

        await await_successor(slot, first)
        assert sum("do X please" in p for p in prompts) == 1, prompts
        assert not slot._queue
        assert slot.task is None

    @pytest.mark.asyncio
    async def test_a_closed_coroutine_runs_no_tail(self, tmp_path, monkeypatch):
        """A coroutine closed while suspended above its main try (a pending task
        destroyed at loop or harness teardown) cannot await, so the guard skips
        the tail instead of raising out of the close."""
        from kiro_crew.dashboard import chat_runner

        state, slot, _prompts = self._setup(tmp_path, monkeypatch)
        tails = AsyncMock()
        monkeypatch.setattr(chat_runner, "_end_turn_tail", tails)
        release = asyncio.Event()

        async def _parked(*_args, **_kwargs):
            await release.wait()
            return True

        monkeypatch.setattr(chat_runner, "_expand_prompt_mention_off_loop", _parked)
        coro = chat_runner._run_chat(state, slot, "/prompts get review")
        assert coro.send(None) is not None  # parked inside the turn
        coro.close()
        tails.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_nested_prompts_get_turn_hands_off_exactly_once(self, tmp_path, monkeypatch):
        """``/prompts get`` runs the expanded prompt as a nested turn, which runs
        the tail itself; the outer call must not run it a second time."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._expand_prompt_mention_off_loop",
            AsyncMock(return_value=("Execute the following instructions: review it", "ok")),
        )
        slot.queue_insert(0, "then this", kind="")

        first = await self._run(state, slot, "/prompts get review")
        await await_successor(slot, first)

        assert sum("review it" in p for p in prompts) == 1, prompts
        assert sum("then this" in p for p in prompts) == 1, prompts
        assert not slot._queue and slot.task is None
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_the_remote_slot_refusal_does_not_drain_locally(self, tmp_path, monkeypatch):
        """A remote-bound slot's queue drains on its peer; the local refusal sends
        its chat_done and leaves the queue alone."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        slot.executor = "remote"
        slot.queue_insert(0, "for the peer", kind="")

        first = await self._run(state, slot, "hello")

        assert prompts == []
        assert [q.get("content") for q in slot._queue] == ["for the peer"]
        assert slot.task is first
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.asyncio
    async def test_a_send_during_the_payload_read_queues_and_runs(self, tmp_path, monkeypatch):
        """The cycle's chat_done payload is read while the finishing turn still
        holds the slot, so a send arriving in that read sees a busy slot and
        queues -- no reservation can take the floor either -- and the cycle
        hands it the floor instead of ending: one terminal chat_done, after the
        message it ran."""
        from kiro_crew.apps.builtins.spec_builder.backend.runtime import _reserve_slot_turn
        from kiro_crew.dashboard import chat_runner

        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        real_payload = chat_runner.chat_done_payload
        reservations: list = []

        async def _payload(the_state, the_slot, **kw):
            if not reservations:
                assert the_slot.turn_running, "the floor was released before the read"
                reservations.append(_reserve_slot_turn(the_state, the_slot))
                the_slot.queue_insert(0, "sent during the read", kind="")
            return await real_payload(the_state, the_slot, **kw)

        monkeypatch.setattr(chat_runner, "chat_done_payload", _payload)

        first = await self._run(state, slot, "/prompts")
        await await_successor(slot, first)

        assert reservations == [None], "a reservation took the floor mid-read"
        assert sum("sent during the read" in p for p in prompts) == 1, prompts
        assert not slot._queue and slot.task is None
        assert len(chat_done_frames(state)) == 1

    @staticmethod
    def _fail_first_setup(monkeypatch) -> None:
        """Make the first ``_run_chat`` raise above its main try, and no later one."""
        calls: list[int] = []

        def _discard(_slot):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("setup broke")

        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._discard_stale_decision", _discard)

    @pytest.mark.parametrize("hold", ["sign-in", "setup"])
    @pytest.mark.asyncio
    async def test_a_send_during_the_read_of_a_held_queue_runs(self, tmp_path, monkeypatch, hold):
        """A held queue keeps what it held, but a send arriving during the cycle's
        payload read is the user's next send, the one a hold resumes on: it runs
        instead of waiting on an idle slot, and the held entries drain after it."""
        from kiro_crew.dashboard import chat_runner

        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        if hold == "sign-in":
            slot._last_turn_auth_required = True
            message = "/prompts"
        else:
            self._fail_first_setup(monkeypatch)
            message = "first"
        slot.queue_insert(0, "held entry", kind="")
        real_payload = chat_runner.chat_done_payload
        sent: list[str] = []

        async def _payload(the_state, the_slot, **kw):
            if not sent:
                sent.append(the_slot.queue_insert(1, "sent during the read", kind=""))
            return await real_payload(the_state, the_slot, **kw)

        monkeypatch.setattr(chat_runner, "chat_done_payload", _payload)

        try:
            await self._run(state, slot, message)
        except RuntimeError:
            assert hold == "setup"
        assert slot.task is not None, "the send was stranded on an idle slot"
        await await_until(slot.task, what="the slot's turn finishing")
        while slot.task is not None and not slot.task.done():
            await await_until(slot.task, what="the slot's turn finishing")

        ran = [p for p in prompts if "sent during the read" in p or "held entry" in p]
        assert len(ran) == 2 and "sent during the read" in ran[0], prompts
        assert not slot._queue and slot.task is None

    @pytest.mark.asyncio
    async def test_a_send_during_the_read_stays_queued_while_sub_agents_run(
        self, tmp_path, monkeypatch
    ):
        """With sub-agents attached a user send waits, as it would on an idle
        slot. The hand-off for a send that arrived during the read of a held
        queue must not start a held sub-agent completion in its place."""
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.chat_utils import SUBAGENT_COMPLETION_KIND

        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        state.subagents = MagicMock()
        state.subagents.running_agents_for = MagicMock(return_value=["child-2"])
        slot._last_turn_auth_required = True
        slot.queue_insert(0, "child-1 done", kind=SUBAGENT_COMPLETION_KIND)
        real_payload = chat_runner.chat_done_payload
        sent: list[str] = []

        async def _payload(the_state, the_slot, **kw):
            if not sent:
                sent.append(the_slot.queue_insert(1, "sent during the read", kind=""))
            return await real_payload(the_state, the_slot, **kw)

        monkeypatch.setattr(chat_runner, "chat_done_payload", _payload)

        await self._run(state, slot, "/prompts")

        assert prompts == []
        assert [q.get("content") for q in slot._queue] == ["child-1 done", "sent during the read"]
        assert slot.task is None
        assert len(chat_done_frames(state)) == 1

    @pytest.mark.parametrize("command", ["/prompts", "/goal status"])
    @pytest.mark.asyncio
    async def test_a_local_command_leaves_a_sub_agents_wait_state_alone(
        self, tmp_path, monkeypatch, command
    ):
        """A sub-agent's ``wait`` keeps its countdown on the parent slot. An exit
        that never reached the main try ran no wait of its own, so it leaves the
        countdown, an uncollected End-wait request and the contested latch to
        the sleep they belong to."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        wait = {"wait_id": "w1", "seconds": 60, "deadline_ts": 1e12}
        slot._wait_state = dict(wait)
        slot._end_wait_request = "w1"
        slot._wait_contested = True

        await self._run(state, slot, command)

        assert prompts == []
        assert slot._wait_state == wait
        assert slot._end_wait_request == "w1"
        assert slot._wait_contested is True

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_a_setup_error_is_a_server_error_on_the_openai_endpoint(
        self, tmp_path, monkeypatch, stream
    ):
        """The cycle still ends with a ``done`` row, which the OpenAI-compatible
        readers wait for, so the failed turn's error row is what tells them to
        answer ``server_error`` rather than an empty successful completion."""
        from kiro_crew.dashboard import openai_compat
        from kiro_crew.dashboard.chat import _run_chat

        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        self._fail_first_setup(monkeypatch)
        with slot.pending_consumer():
            task = asyncio.ensure_future(_run_chat(state, slot, "first"))
            slot.task = task
            if stream:
                written: list[bytes] = []
                resp = MagicMock()
                resp.headers = {}
                resp.prepare = AsyncMock()
                resp.write = AsyncMock(side_effect=written.append)
                monkeypatch.setattr(openai_compat.web, "StreamResponse", lambda: resp)
                await await_until(
                    asyncio.ensure_future(
                        openai_compat._stream_response(
                            MagicMock(), state, slot, "id1", "m", 0, False
                        )
                    ),
                    what="the error stream response finishing",
                )
                body = b"".join(written).decode()
                assert "server_error" in body and '"finish_reason": "stop"' not in body, body
            else:
                response = await await_until(
                    asyncio.ensure_future(
                        openai_compat._blocking_response(state, slot, "id1", "m", 0, False)
                    ),
                    what="the error blocking response finishing",
                )
                assert response.status == 500
                assert json.loads(response.text)["error"]["type"] == "server_error"
        with pytest.raises(RuntimeError, match="setup broke"):
            await task
        assert prompts == []

    @pytest.mark.asyncio
    async def test_a_setup_error_shows_an_error_and_a_finished_frame(self, tmp_path, monkeypatch):
        """The dispatcher only logs the exception, so the turn says it failed; and
        the queue it held runs on the user's next send, not by itself, so the
        terminal frame does not report it as work that continues."""
        state, slot, _prompts = self._setup(tmp_path, monkeypatch)
        self._fail_first_setup(monkeypatch)
        slot.queue_insert(0, "second", kind="")

        with pytest.raises(RuntimeError, match="setup broke"):
            await self._run(state, slot, "first")

        errors = [m.get("content", "") for m in slot.messages if m.get("role") == "error"]
        assert any("failed before it started" in e for e in errors), errors
        [frame] = chat_done_frames(state)
        assert frame["continuing"] is False, frame

    @pytest.mark.asyncio
    async def test_a_failing_local_command_does_not_hold_the_queue(self, tmp_path, monkeypatch):
        """A local command's own handler raising is that entry's failure, not
        setup every queued entry shares: the user sees it failed and the message
        queued behind it runs."""
        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._handle_workflow_command",
            AsyncMock(side_effect=OSError("store unavailable")),
        )
        slot.queue_insert(0, "summarize the PR", kind="")

        with pytest.raises(OSError, match="store unavailable"):
            await self._run(state, slot, "/workflow nightly")
        assert slot.task is not None, "the queued message was held"
        await await_until(slot.task, what="the slot's turn finishing")

        assert sum("summarize the PR" in p for p in prompts) == 1, prompts
        errors = [m.get("content", "") for m in slot.messages if m.get("role") == "error"]
        assert any("`/workflow` failed" in e for e in errors), errors
        assert not slot._queue and slot.task is None

    @pytest.mark.asyncio
    async def test_a_local_command_keeps_the_queue_after_memory_preparation_failed(
        self, tmp_path, monkeypatch
    ):
        """A preparation task can finish without being cancelled and still have
        recorded a failure the turn's barrier refuses. The tail asks the barrier's
        own rule, so the queue is held rather than walked into that refusal."""
        from kiro_crew import memory_startup

        state, slot, prompts = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(memory_startup, "_active", None)
        startup = memory_startup.MemoryStartup.begin()
        startup.fail(RuntimeError("structural"))
        finished = asyncio.get_running_loop().create_future()
        finished.set_result(None)
        state.memory_startup_task = finished
        slot.queue_insert(0, "after memory is ready", kind="")

        await self._run(state, slot, "/prompts")

        assert prompts == []
        assert [q.get("content") for q in slot._queue] == ["after memory is ready"]
        assert slot.task is None


# ── Mode/approval policy propagation (HTTP handlers) ──


class TestApiChatModePropagation:
    """api_chat_mode propagates approval policy to all session slots."""

    @pytest.mark.asyncio
    async def test_yolo_mode_propagates_auto_policy(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")
        state.get_or_create_slot("s2")

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "yolo"})
            data = await resp.json()
            assert data["ok"] is True

        calls = state.sessions.set_approval_policy.call_args_list
        keys = [c.args[0] for c in calls]
        policies = [c.args[1] for c in calls]
        assert "dashboard:s1" in keys
        assert "dashboard:s2" in keys
        assert all(p == "auto" for p in policies)

    @pytest.mark.asyncio
    async def test_normal_mode_clears_policy(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.safety_override import safety_override

        safety_override().activate("test")
        state = _make_state(tmp_path)
        state.enable_yolo()
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("s1")
        slot._trust = False

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "normal"})
            data = await resp.json()
            assert data["ok"] is True

        state.sessions.set_approval_policy.assert_called_with("dashboard:s1", "")

    @pytest.mark.asyncio
    async def test_trust_mode_scoped_to_slot_channel(self, tmp_path, monkeypatch):
        """Trust with slot_key only trusts that slot's linked channel."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("s1")
        slot._slack_channel = "ch1"

        ch1 = MagicMock(trusted=False)
        ch2 = MagicMock(trusted=False)
        mgr = MagicMock(_channels={"ch1": ch1, "ch2": ch2})
        state.channel_manager = mgr

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "trust", "slot": "s1"})
            assert (await resp.json())["ok"] is True

        assert ch1.trusted is True
        ch1._save.assert_called_once()
        assert ch2.trusted is False
        ch2._save.assert_not_called()

    @pytest.mark.asyncio
    async def test_trust_mode_all_channels_when_no_slot(self, tmp_path, monkeypatch):
        """Trust without slot_key trusts all channels."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        ch = MagicMock(trusted=False)
        mgr = MagicMock(_channels={"ch1": ch})
        state.channel_manager = mgr

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "trust"})
            assert (await resp.json())["ok"] is True

        assert ch.trusted is True
        ch._save.assert_called_once()

    @pytest.mark.asyncio
    async def test_normal_mode_scoped_resets_only_linked_channel(self, tmp_path, monkeypatch):
        """Normal mode with slot_key should only reset that slot's linked channel."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("s1")
        slot._slack_channel = "ch1"
        state.get_or_create_slot("s2")

        ch1 = MagicMock(trusted=True)
        ch2 = MagicMock(trusted=True)
        mgr = MagicMock(_channels={"ch1": ch1, "ch2": ch2})
        state.channel_manager = mgr

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "normal", "slot": "s1"})
            assert (await resp.json())["ok"] is True

        assert ch1.trusted is False
        ch1._save.assert_called_once()
        assert ch2.trusted is True
        ch2._save.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_mode_resets_all_channels_when_no_slot(self, tmp_path, monkeypatch):
        """Normal mode without slot_key resets all channel trust."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        ch = MagicMock(trusted=True)
        mgr = MagicMock(_channels={"ch1": ch})
        state.channel_manager = mgr

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "normal"})
            assert (await resp.json())["ok"] is True

        assert ch.trusted is False
        ch._save.assert_called_once()

    @pytest.mark.asyncio
    async def test_trust_mode_unknown_slot_returns_400(self, tmp_path, monkeypatch):
        """Trust with unknown slot_key must return 400, not trust all."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/mode", json={"mode": "trust", "slot": "nonexistent"}
            )
            assert resp.status == 400
            assert (await resp.json())["error"] == "unknown slot"

    @pytest.mark.asyncio
    async def test_normal_mode_unknown_slot_returns_400(self, tmp_path, monkeypatch):
        """Normal with unknown slot_key must return 400, not reset all."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/mode", json={"mode": "normal", "slot": "nonexistent"}
            )
            assert resp.status == 400
            assert (await resp.json())["error"] == "unknown slot"

    @pytest.mark.asyncio
    async def test_trust_slot_preserves_other_slot_trust(self, tmp_path, monkeypatch):
        """trusting slot B must not wipe trust from slot A."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        s1 = state.get_or_create_slot("s1")
        s2 = state.get_or_create_slot("s2")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": "trust", "slot": "s1"})
            assert s1._trust is True

            await client.post("/api/chat/mode", json={"mode": "trust", "slot": "s2"})
            assert s2._trust is True
            assert s1._trust is True  # must survive

    @pytest.mark.asyncio
    async def test_yolo_restores_per_slot_trust(self, tmp_path, monkeypatch):
        """YOLO does not mutate per-slot trust; disabling preserves it."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        s1 = state.get_or_create_slot("s1")
        s2 = state.get_or_create_slot("s2")

        async with TestClient(TestServer(_make_app(state))) as client:
            # Set per-slot modes: s1=trust, s2=trust_reads
            await client.post("/api/chat/mode", json={"mode": "trust", "slot": "s1"})
            await client.post("/api/chat/mode", json={"mode": "trust_reads", "slot": "s2"})
            assert s1._trust is True
            assert s2._trust_reads is True

            # YOLO overrides everything
            await client.post("/api/chat/mode", json={"mode": "yolo"})
            assert s1._trust is True  # unchanged
            assert s2._trust_reads is True  # unchanged

            # Set s1 to normal (leaving YOLO) — s2 should be untouched
            await client.post("/api/chat/mode", json={"mode": "normal", "slot": "s1"})
            assert s1._trust is False
            assert s1._trust_reads is False
            assert s2._trust_reads is True  # preserved

    def test_yolo_auto_expires_and_clears_untrusted_policies(self, tmp_path, monkeypatch):
        """YOLO expiry clears policies for untrusted slots only."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.broadcast_ws = MagicMock()
        s1 = state.get_or_create_slot("s1")
        state.get_or_create_slot("s2")
        s1._trust = True

        from unittest.mock import patch

        from kiro_crew.safety_override import safety_override

        # Wire the on_expired callback (as server.py does at startup)
        def _on_expired(source: str) -> None:
            if state.sessions is not None:
                for slot in state._slots.values():
                    if not slot._trust and not slot._trust_reads:
                        state.sessions.set_approval_policy(f"dashboard:{slot.key}", "")

        with patch("kiro_crew.safety_override.sel"):
            safety_override().activate("dashboard")
        safety_override().on_expired = _on_expired
        safety_override()._expires_at = 0  # already expired

        assert state.is_yolo_active() is False
        assert s1._trust is True  # per-slot trust survives expiry

        cleared = [
            c[0][0] for c in state.sessions.set_approval_policy.call_args_list if c[0][1] == ""
        ]
        assert "dashboard:s2" in cleared
        assert "dashboard:s1" not in cleared

    @pytest.mark.asyncio
    async def test_trust_mode_propagates_approval_policy_to_session(self, tmp_path, monkeypatch):
        """Trust mode must set session approval_policy='auto' so subagents inherit."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.slack.handler.is_yolo_mode", lambda: False)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": "trust", "slot": "s1"})

        state.sessions.set_approval_policy.assert_any_call("dashboard:s1", "auto")

    @pytest.mark.asyncio
    async def test_normal_mode_resets_approval_policy(self, tmp_path, monkeypatch):
        """Normal mode must reset session approval_policy so subagents require approval."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.slack.handler.is_yolo_mode", lambda: False)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": "normal", "slot": "s1"})

        state.sessions.set_approval_policy.assert_any_call("dashboard:s1", "")

    @pytest.mark.asyncio
    async def test_trust_reads_mode_resets_approval_policy(self, tmp_path, monkeypatch):
        """trust_reads must reset approval_policy (not auto-approve writes)."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.slack.handler.is_yolo_mode", lambda: False)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": "trust_reads", "slot": "s1"})

        state.sessions.set_approval_policy.assert_any_call("dashboard:s1", "")

    @pytest.mark.asyncio
    async def test_trust_mode_all_slots_propagates_approval_policy(self, tmp_path, monkeypatch):
        """Trust without slot_key must set approval_policy on all slots."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.slack.handler.is_yolo_mode", lambda: False)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")
        state.get_or_create_slot("s2")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": "trust"})

        state.sessions.set_approval_policy.assert_any_call("dashboard:s1", "auto")
        state.sessions.set_approval_policy.assert_any_call("dashboard:s2", "auto")


class TestApiChatModeGlobalOverrideScope:
    """A slot-scoped mode change must not revoke the process-global YOLO grant.

    The grant covers every slot, so ending it on behalf of a request that named
    ONE slot changes slots the request never mentioned — the shape that let a
    per-slot `trust` setup call drop the whole gateway out of YOLO.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["trust", "trust_reads"])
    async def test_slot_scoped_trust_preserves_global_grant(self, mode, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.safety_override import safety_override

        safety_override().activate("dashboard")
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")
        state.get_or_create_slot("s2")

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": mode, "slot": "s1"})
            assert (await resp.json())["ok"] is True

        assert safety_override().is_active() is True
        # The regression itself: a slot the request never named keeps auto-approving.
        state.sessions.set_approval_policy.assert_any_call("dashboard:s2", "auto")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["trust", "trust_reads"])
    async def test_global_trust_revokes_global_grant(self, mode, tmp_path, monkeypatch):
        """Without a slot the request IS global, so it may end the grant."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.safety_override import safety_override

        safety_override().activate("dashboard")
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": mode})

        assert safety_override().is_active() is False

    @pytest.mark.asyncio
    async def test_slot_scoped_normal_revokes_global_grant(self, tmp_path, monkeypatch):
        """`normal` asks for NO auto-approval, so it revokes at any scope.

        This is the dashboard picker's YOLO off-switch: the picker always names
        the slot it is attached to.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.safety_override import safety_override

        safety_override().activate("dashboard")
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": "normal", "slot": "s1"})

        assert safety_override().is_active() is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["trust", "trust_reads"])
    async def test_declared_grant_is_revoked_at_any_scope(self, mode, tmp_path, monkeypatch):
        """A declared grant has no TTL, so any mode selection is its off-switch."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.safety_override import safety_override

        safety_override().activate_declared()
        assert safety_override().is_declared is True
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": mode, "slot": "s1"})

        assert safety_override().is_active() is False

    @pytest.mark.asyncio
    async def test_until_shutdown_grant_is_ad_hoc_and_protected(self, tmp_path, monkeypatch):
        """`until_shutdown` is permanent but AD HOC, so the scope rule applies.

        Classifying it by permanence instead of source would revoke the grant of
        every operator who picked "until Kiro Crew restarts".
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.safety_override import safety_override

        so = safety_override()
        so.adhoc_until_shutdown = True
        so.activate("dashboard")
        assert so.is_permanent is True
        assert so.is_declared is False
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post("/api/chat/mode", json={"mode": "trust", "slot": "s1"})

        assert so.is_active() is True

    @pytest.mark.asyncio
    async def test_non_string_mode_does_not_raise(self, tmp_path, monkeypatch):
        """The membership test must answer for an unhashable body value."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.get_or_create_slot("s1")

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": {}, "slot": "s1"})

        assert resp.status == 200  # falls through to the normal branch, as before


class TestApproveYoloPropagation:
    """api_chat_slot_approve with yolo action propagates policy to all slots."""

    @pytest.mark.asyncio
    async def test_yolo_approve_propagates_to_all_slots(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        s1 = state.get_or_create_slot("s1")
        state.get_or_create_slot("s2")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        s1._approval_futures["test"] = fut

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/approve", json={"action": "yolo"})
            data = await resp.json()
            assert data["ok"] is True

        calls = state.sessions.set_approval_policy.call_args_list
        keys = [c.args[0] for c in calls]
        assert "dashboard:s1" in keys
        assert "dashboard:s2" in keys

    @pytest.mark.asyncio
    async def test_trust_approve_propagates_to_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("s1")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        slot._approval_futures["test"] = fut
        slot.messages.append(
            {
                "role": "permission",
                "content": "Running: ls",
                "cls": json.dumps(
                    {
                        "request_id": "test",
                        "full_command": "ls",
                        "trust_grantable": "1",
                    }
                ),
            }
        )

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/approve", json={"action": "trust"})
            data = await resp.json()
            assert data["ok"] is True

        state.sessions.set_approval_policy.assert_called_with("dashboard:s1", "auto")


# ── Coverage: bulk-approve broadcasts ──


class TestBulkApproveBroadcast:
    """Trust/YOLO mode change bulk-approve must broadcast approval_resolved."""

    @pytest.mark.asyncio
    async def test_mode_yolo_broadcasts_for_pending(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat.sel", lambda: MagicMock())
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.broadcast_ws = MagicMock()
        slot = state.get_or_create_slot("s1")
        loop = asyncio.get_running_loop()
        f1: asyncio.Future[str] = loop.create_future()
        f2: asyncio.Future[str] = loop.create_future()
        slot._approval_futures["req-1"] = f1
        slot._approval_futures["req-2"] = f2

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "yolo"})
            assert (await resp.json())["ok"] is True

        broadcast_calls = [
            c for c in state.broadcast_ws.call_args_list if c.args[0] == "approval_resolved"
        ]
        ids = {c.args[1]["id"] for c in broadcast_calls}
        assert "req-1" in ids
        assert "req-2" in ids

    @pytest.mark.asyncio
    async def test_slot_scoped_trust_leaves_other_slots_pending(self, tmp_path, monkeypatch):
        """A slot-scoped ``trust`` must NOT sweep other slots' pending approvals.

        Regression: bulk auto-approve iterated EVERY slot regardless of scope, so
        trusting one chat resolved the pending approval card in unrelated chats
        (making them LOOK approved) while their ``_trust`` flag stayed False — so
        their next tool call prompted again. Scope the sweep to the target
        session; other slots keep their pending future AND their untrusted state.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat.sel", lambda: MagicMock())
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.broadcast_ws = MagicMock()
        s1 = state.get_or_create_slot("s1")
        s2 = state.get_or_create_slot("s2")
        loop = asyncio.get_running_loop()
        f1: asyncio.Future[str] = loop.create_future()
        f2: asyncio.Future[str] = loop.create_future()
        s1._approval_futures["req-1"] = f1
        s2._approval_futures["req-2"] = f2

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "trust", "slot": "s1"})
            assert (await resp.json())["ok"] is True

        # Target slot: approved + broadcast + trusted.
        assert f1.done() and f1.result() == "approved"
        assert s1._trust is True
        # Other slot: still pending, never broadcast, still untrusted.
        assert not f2.done()
        assert s2._trust is False
        resolved_ids = {
            c.args[1]["id"]
            for c in state.broadcast_ws.call_args_list
            if c.args[0] == "approval_resolved"
        }
        assert "req-1" in resolved_ids
        assert "req-2" not in resolved_ids
        # Clean up the still-pending future so the loop does not warn on GC.
        f2.set_result("rejected")

    @pytest.mark.asyncio
    async def test_unscoped_trust_still_sweeps_all_slots(self, tmp_path, monkeypatch):
        """An all-slots ``trust`` (no slot named) keeps sweeping every slot."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat.sel", lambda: MagicMock())
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.broadcast_ws = MagicMock()
        s1 = state.get_or_create_slot("s1")
        s2 = state.get_or_create_slot("s2")
        loop = asyncio.get_running_loop()
        f1: asyncio.Future[str] = loop.create_future()
        f2: asyncio.Future[str] = loop.create_future()
        s1._approval_futures["req-1"] = f1
        s2._approval_futures["req-2"] = f2

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/mode", json={"mode": "trust"})
            assert (await resp.json())["ok"] is True

        assert f1.done() and f2.done()
        assert s1._trust is True and s2._trust is True
        resolved_ids = {
            c.args[1]["id"]
            for c in state.broadcast_ws.call_args_list
            if c.args[0] == "approval_resolved"
        }
        assert {"req-1", "req-2"} <= resolved_ids


# ── Coverage: multi-pending approval 400 and trust auto-approve ──


class TestMultiPendingApproval:
    """Cover the 400 response when multiple approvals are pending without request_id."""

    @pytest.mark.asyncio
    async def test_multi_pending_returns_400(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        loop = asyncio.get_running_loop()
        slot._approval_futures["a1"] = loop.create_future()
        slot._approval_futures["a2"] = loop.create_future()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/approve", json={"action": "approved"})
            assert resp.status == 400
            data = await resp.json()
            assert "pending" in data
            assert set(data["pending"]) == {"a1", "a2"}

    @pytest.mark.asyncio
    async def test_approve_with_request_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        slot._approval_futures["specific"] = fut

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/approve",
                json={"action": "approved", "request_id": "specific"},
            )
            assert resp.status == 200
            assert fut.result() == "approved"


# ── Agent passing via /api/chat (AgentRock integration) ──


class TestApiChatAgentPassing:
    @pytest.mark.asyncio
    async def test_agent_set_on_new_slot(self, tmp_path, monkeypatch):
        from kiro_crew.config import loader

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agentrock--my-aim-agent.json").write_text(
            json.dumps({"name": "my-aim-agent"}), encoding="utf-8"
        )
        monkeypatch.setattr(loader, "kiro_agents_dir", lambda: agents_dir)
        monkeypatch.setattr(loader, "_MATERIALIZED_AGENTS", frozenset())
        monkeypatch.setattr(loader, "_MATERIALIZED_AGENTS_READY", False)
        await asyncio.to_thread(loader.refresh_materialized_agents)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat?ws=1",
                json={"message": "hello", "slot": "agentrock-my-skill", "agent": "my-aim-agent"},
            )
            data = await resp.json()
            assert resp.status == 200, data
            assert data["ok"] is True
            assert state._slots["agentrock-my-skill"].agent == "my-aim-agent"

    @pytest.mark.asyncio
    async def test_agent_mismatch_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("slot-x")
        slot.agent = "agent-a"
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat?ws=1",
                json={"message": "hello", "slot": "slot-x", "agent": "agent-b"},
            )
            assert resp.status == 409

    @pytest.mark.asyncio
    async def test_empty_agent_on_agent_slot_allowed(self, tmp_path, monkeypatch):
        """Follow-up message with no agent on an agent-bound slot must not 409."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("slot-y")
        slot.agent = "agent-a"
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat?ws=1",
                json={"message": "follow-up", "slot": "slot-y", "agent": ""},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_invalid_agent_name_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from unittest.mock import patch

        state = _make_state(tmp_path)
        with patch("kiro_crew.dashboard.chat_handlers._emit_agent_assignment") as mock_emit:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hello", "slot": "s1", "agent": "../evil"},
                )
            assert resp.status == 400
            mock_emit.assert_called_once_with("s1", "../evil", outcome="denied_invalid")

    @pytest.mark.asyncio
    async def test_credential_shaped_invalid_agent_is_redacted_in_sel(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from unittest.mock import MagicMock, patch

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("member-dr-eggbot", agent="dr. eggbot", mode="member")
        unsafe = "crew password=shortvalue"
        audit = MagicMock()
        with patch("kiro_crew.dashboard.chat_utils.sel", return_value=audit):
            async with TestClient(TestServer(_make_app(state))) as client:
                response = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hello", "slot": slot.key, "agent": unsafe},
                )

        assert response.status == 400
        event = audit.log.call_args.args[0]
        assert event.agent == "crew password=[REDACTED]"
        assert unsafe not in event.agent

    @pytest.mark.asyncio
    async def test_non_string_agent_logs_actual_value(self, tmp_path, monkeypatch):
        """Fix for Post 22: str(agent) preserves malicious input in audit trail."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from unittest.mock import patch

        state = _make_state(tmp_path)
        with patch("kiro_crew.dashboard.chat_handlers._emit_agent_assignment") as mock_emit:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hello", "slot": "s1", "agent": 123},
                )
            assert resp.status == 400
            mock_emit.assert_called_once_with("s1", "123", outcome="denied_invalid")

    @pytest.mark.asyncio
    async def test_no_agent_no_emit(self, tmp_path, monkeypatch):
        """Fix for Post 23: no SEL event when no agent involved (reduces audit noise)."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from unittest.mock import patch

        state = _make_state(tmp_path)
        with patch("kiro_crew.dashboard.chat_handlers._emit_agent_assignment") as mock_emit:
            async with TestClient(TestServer(_make_app(state))) as client:
                await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hi", "slot": "no-agent-slot"},
                )
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_sel_event_on_running_slot_rejection(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from unittest.mock import MagicMock, patch

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("slot-r")
        mock_task = MagicMock()
        mock_task.done.return_value = False
        slot.task = mock_task
        with patch("kiro_crew.dashboard.chat_handlers._emit_agent_assignment") as mock_emit:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hi", "slot": "slot-r", "agent": "new-agent"},
                )
                assert resp.status == 409
            mock_emit.assert_called_once_with("slot-r", "new-agent", outcome="denied_running")


class TestApiChatSendReceiptMid:
    """The immediate-dispatch receipt carries the user row's server-minted `mid`.

    Correlated dashboard sends also echo the row before their reply starts.
    The receipt and echo must agree on identity regardless of arrival order.
    """

    @pytest.mark.asyncio
    async def test_idle_steer_echoes_user_before_the_reply(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("echo-slot")
        echoed: list[dict] = []
        broadcast_ws = state.broadcast_ws

        def record_echo(kind, data):
            if kind == "chat_message":
                echoed.append(data)
            broadcast_ws(kind, data)

        monkeypatch.setattr(state, "broadcast_ws", record_echo)

        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            # The frontend can see this turn's first chunk before the HTTP
            # receipt. Its user row must already be on the ordered event stream.
            assert [(row["slot"], row["role"]) for row in echoed] == [("echo-slot", "user")]
            sl.append("assistant", "reply")

        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat?ws=1",
                json={
                    "message": "pasted\ttable\r\n1\t42",
                    "slot": "echo-slot",
                    "steer": True,
                    "meta": {"sendId": "s-echo", "pastes": [{"seq": 1, "content": "table"}]},
                },
            )
            assert resp.status == 200
            receipt = await resp.json()
            await slot.task

        assert len(echoed) == 1
        assert [row["role"] for row in slot.messages] == ["user", "assistant"]
        user = echoed[0]
        assert user["content"] == "pasted\ttable\r\n1\t42"
        assert user["meta"]["sendId"] == "s-echo"
        assert user["meta"]["mid"] == receipt["mid"]
        assert user["meta"]["pastes"][0]["content"] == "table"

    @pytest.mark.asyncio
    async def test_immediate_dispatch_receipt_carries_the_user_row_mid(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
            sl.append("chunk", "ack", "chunk")

        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", fake_run_chat)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat?ws=1",
                json={"message": "hello", "slot": "mid-slot"},
            )
            assert resp.status == 200
            data = await resp.json()

        assert data["ok"] is True
        # The receipt id must be the SAME id stamped on the appended user row —
        # that is the identity the client stamps on the optimistic bubble and
        # the pin endpoint keys on.
        slot = state._slots["mid-slot"]
        user_rows = [m for m in slot.messages if m["role"] == "user"]
        assert user_rows, "the send must have appended a user row"
        row_mid = user_rows[-1]["meta"]["mid"]
        assert row_mid
        assert data["mid"] == row_mid

    @pytest.mark.asyncio
    async def test_queued_send_receipt_carries_no_mid(self, tmp_path, monkeypatch):
        """A busy slot queues the message and broadcasts its own queue card; the
        optimistic bubble is not this row's receipt, so no `mid` is returned
        (mirrors `confirmedDelivered`, which excludes a queued acceptance)."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("busy-slot")
        mock_task = MagicMock()
        mock_task.done.return_value = False
        slot.task = mock_task  # slot.running is True → the busy/queue branch

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat?ws=1",
                json={"message": "queue me", "slot": "busy-slot"},
            )
            assert resp.status == 200
            data = await resp.json()

        assert data["queued"] is True
        assert "mid" not in data


class TestPromptBusyRecovery:
    """When kiro-cli returns 'Prompt already in progress', _run_chat must
    reset the session and re-queue the message so the next attempt cold-starts."""

    @pytest.mark.asyncio
    async def test_prompt_busy_resets_session_and_requeues(self, tmp_path: Path) -> None:
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        state = _make_state(tmp_path)
        state.sessions.get_or_create = AsyncMock(return_value=(MagicMock(), False, False))
        state.sessions.release = MagicMock()
        state.sessions.reset = AsyncMock()
        state.sessions.set_approval_policy = MagicMock()
        state.sessions.check_context_usage = MagicMock()
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.is_yolo_active = MagicMock(return_value=False)
        state._background_tasks = set()

        slot = state.get_or_create_slot("busy-slot")
        slot.append("user", "hello", "msg msg-u")

        # Make client.stream raise "already in progress"
        mock_client = state.sessions.get_or_create.return_value[0]

        async def _raise_busy(msg):
            raise AcpError("Prompt error: {'data': 'Prompt already in progress'}")
            yield  # make it an async generator  # noqa: E501

        mock_client.stream = _raise_busy
        mock_client.stream_command = _raise_busy
        mock_client.shutdown = AsyncMock()

        await _run_chat(state, slot, "test message")

        # Session must be reset (kill the stuck kiro-cli process)
        state.sessions.reset.assert_awaited_once()
        # The finally block drains the re-queued message into a new task
        assert slot.task is not None
        # No ❌ error shown to the user for the busy case
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert not any("already in progress" in m.get("content", "") for m in error_msgs)

    @pytest.mark.asyncio
    async def test_process_exited_resets_session_and_requeues(self, tmp_path: Path) -> None:
        """When ACP subprocess dies (SIGTERM/SIGKILL), _run_chat must reset
        the session and re-queue the message so autonudges land on a fresh
        provider instead of a bare ❌ error card with no work done."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        state = _make_state(tmp_path)
        state.sessions.get_or_create = AsyncMock(return_value=(MagicMock(), False, False))
        state.sessions.release = MagicMock()
        state.sessions.reset = AsyncMock()
        state.sessions.set_approval_policy = MagicMock()
        state.sessions.check_context_usage = MagicMock()
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.is_yolo_active = MagicMock(return_value=False)
        state._background_tasks = set()

        slot = state.get_or_create_slot("dead-slot")
        slot.append("user", "hello", "msg msg-u")

        mock_client = state.sessions.get_or_create.return_value[0]

        async def _raise_dead(msg):
            raise AcpError("ACP process exited (code=-15)")
            yield  # make it an async generator  # noqa: E501

        mock_client.stream = _raise_dead
        mock_client.stream_command = _raise_dead
        mock_client.shutdown = AsyncMock()

        await _run_chat(state, slot, "test message")

        state.sessions.reset.assert_awaited_once()
        assert slot.task is not None
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert not any("process exited" in m.get("content", "") for m in error_msgs)


# ── Tests: slot.task None guard ──


class TestSlotTaskNoneGuard:
    """stop/delete must not crash when slot.task is None."""

    @pytest.mark.asyncio
    async def test_stop_not_running(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("s1")
        # task is None → running is False → stop is a no-op
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_delete_not_running(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")
        # task is None → running is False → delete skips cancel
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.delete("/api/chat/slots/s1")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_stop_with_real_task_cancels(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        state.sessions.stop_turn = AsyncMock(return_value="soft")
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.get_running_loop().create_future()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200
            state.sessions.stop_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_with_real_task_cancels(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.get_running_loop().create_future()

        async with TestClient(TestServer(_make_app(state))) as client:
            with patch("kiro_crew.dashboard.chat_handlers.save_slot_off_loop"):
                resp = await client.delete("/api/chat/slots/s1")
            assert resp.status == 200
            assert slot.task.cancelled()


# ── Bulk cleanup tests ──


class TestBulkCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_archives_stale_sessions(self, tmp_path, monkeypatch):
        """Stale sessions are archived; fresh and pinned are kept."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        fresh_ts = datetime.now(timezone.utc).isoformat()

        stale = state.get_or_create_slot("stale1")
        stale.append("user", "old msg", ts=old_ts)
        stale.drain()

        fresh = state.get_or_create_slot("fresh1")
        fresh.append("user", "new msg", ts=fresh_ts)
        fresh.drain()

        pinned = state.get_or_create_slot("pinned1")
        pinned.pinned = True
        pinned.append("user", "pinned msg", ts=old_ts)
        pinned.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 3, "active_slot": "fresh1"},
            )
            data = await resp.json()
            assert data["ok"] is True
            assert data["archived"] == 1
            assert "stale1" in data["keys"]

        assert "stale1" not in state._slots
        assert "fresh1" in state._slots
        assert "pinned1" in state._slots

    @pytest.mark.asyncio
    async def test_cleanup_skips_active_slot(self, tmp_path, monkeypatch):
        """The active slot is never archived even if stale."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        slot = state.get_or_create_slot("active")
        slot.append("user", "old", ts=old_ts)
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 1, "active_slot": "active"},
            )
            data = await resp.json()
            assert data["archived"] == 0
        assert "active" in state._slots

    @pytest.mark.asyncio
    async def test_cleanup_saves_to_history(self, tmp_path, monkeypatch):
        """Archived sessions are persisted to conversation log."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        slot = state.get_or_create_slot("to-archive")
        slot.append("user", "save me", ts=old_ts)
        slot.append("assistant", "saved", ts=old_ts)
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 3},
            )

        msgs = state.conversation_log.read_messages("dashboard:to-archive")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "save me"

    @pytest.mark.asyncio
    async def test_cleanup_defaults_to_3_days(self, tmp_path, monkeypatch):
        """Without max_inactive_days, defaults to 3."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        ts_2d = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        slot = state.get_or_create_slot("recent")
        slot.append("user", "hi", ts=ts_2d)
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/cleanup", json={})
            data = await resp.json()
            assert data["archived"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_empty_slots_uses_created_at(self, tmp_path, monkeypatch):
        """Slots with no messages use created_at for staleness."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        slot = state.get_or_create_slot("empty-old")
        slot.created_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 7},
            )
            data = await resp.json()
            assert data["archived"] == 1
            assert "empty-old" in data["keys"]

    @pytest.mark.asyncio
    async def test_cleanup_no_stale_returns_zero(self, tmp_path, monkeypatch):
        """When all sessions are fresh, nothing is archived."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timezone

        fresh_ts = datetime.now(timezone.utc).isoformat()
        slot = state.get_or_create_slot("fresh")
        slot.append("user", "hi", ts=fresh_ts)
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 1},
            )
            data = await resp.json()
            assert data["ok"] is True
            assert data["archived"] == 0
            assert data["keys"] == []

    @pytest.mark.asyncio
    async def test_cleanup_rollback_on_save_failure(self, tmp_path, monkeypatch):
        """When _save_slot_to_history raises, slot is restored and reported as failed."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        slot = state.get_or_create_slot("fail-save")
        slot.append("user", "msg", ts=old_ts)
        slot.drain()

        with patch(
            "kiro_crew.dashboard.chat_handlers.save_slot_off_loop",
            side_effect=OSError("disk full"),
        ):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat/slots/cleanup",
                    json={"max_inactive_days": 1},
                )
                data = await resp.json()
                assert data["archived"] == 0
                assert "fail-save" in data["failed"]

        # Slot must be restored (not lost)
        assert "fail-save" in state._slots
        # No history entry should exist (save failed)
        msgs = state.conversation_log.read_messages("dashboard:fail-save")
        assert len(msgs) == 0

    @pytest.mark.asyncio
    async def test_cleanup_cancels_running_task(self, tmp_path, monkeypatch):
        """Running tasks on stale slots are cancelled after archive."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        slot = state.get_or_create_slot("running1")
        slot.append("user", "msg", ts=old_ts)
        slot.drain()
        slot.task = asyncio.get_running_loop().create_future()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 1},
            )
            data = await resp.json()
            assert data["archived"] == 1
        assert slot.task.cancelled()

    @pytest.mark.asyncio
    async def test_cleanup_skips_unparseable_timestamps(self, tmp_path, monkeypatch):
        """Slots with unparseable timestamps are skipped, not archived."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        slot = state.get_or_create_slot("bad-ts")
        slot.append("user", "hi", ts="not-a-date")
        slot.created_at = "also-not-a-date"
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 1},
            )
            data = await resp.json()
            assert data["archived"] == 0
        assert "bad-ts" in state._slots

    @pytest.mark.asyncio
    async def test_cleanup_dry_run_returns_keys_without_archiving(self, tmp_path, monkeypatch):
        """dry_run=True returns stale keys and active_is_stale but does not archive anything."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        fresh_ts = datetime.now(timezone.utc).isoformat()

        stale = state.get_or_create_slot("stale1")
        stale.append("user", "old msg", ts=old_ts)
        stale.drain()

        active_stale = state.get_or_create_slot("active1")
        active_stale.append("user", "old active msg", ts=old_ts)
        active_stale.drain()

        fresh = state.get_or_create_slot("fresh1")
        fresh.append("user", "new msg", ts=fresh_ts)
        fresh.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/cleanup",
                json={"max_inactive_days": 3, "active_slot": "active1", "dry_run": True},
            )
            data = await resp.json()
            assert data["ok"] is True
            assert data["dry_run"] is True
            assert "stale1" in data["keys"]
            assert "active1" not in data["keys"]
            assert data["count"] == 1
            assert data["active_is_stale"] is True

        # Slots should NOT have been removed
        assert "stale1" in state._slots
        assert "active1" in state._slots
        assert "fresh1" in state._slots


class TestHistoryKeyFor:
    """Tests for _history_key_for — canonical history key from slot key."""

    def test_already_canonical(self):
        from kiro_crew.dashboard.chat import _history_key_for

        assert _history_key_for("dashboard:chat-1-100") == "dashboard:chat-1-100"

    def test_strips_single_prefix(self):
        from kiro_crew.dashboard.chat import _history_key_for

        assert _history_key_for("dashboard_chat-1-100") == "dashboard:chat-1-100"

    def test_strips_double_prefix(self):
        from kiro_crew.dashboard.chat import _history_key_for

        assert _history_key_for("dashboard_dashboard_chat-1-100") == "dashboard:chat-1-100"

    def test_strips_triple_prefix(self):
        from kiro_crew.dashboard.chat import _history_key_for

        assert _history_key_for("dashboard_dashboard_dashboard_x") == "dashboard:x"

    def test_raw_key_gets_prefix(self):
        from kiro_crew.dashboard.chat import _history_key_for

        assert _history_key_for("chat-1-100") == "dashboard:chat-1-100"


# ── Folder CRUD tests ──


class TestFolderCRUD:
    @pytest.mark.asyncio
    async def test_list_folders_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/chat/folders")
            assert resp.status == 200
            assert await resp.json() == []

    @pytest.mark.asyncio
    async def test_create_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Oncall"})
            assert resp.status == 201
            data = await resp.json()
            assert data["name"] == "Oncall"
            assert "id" in data
            assert data["collapsed"] is False
            # Persisted to disk
            assert (tmp_path / "folders.json").exists()

    @pytest.mark.asyncio
    async def test_create_folder_with_parent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Parent"})
            parent = await resp.json()
            resp = await client.post(
                "/api/chat/folders", json={"name": "Child", "parent_id": parent["id"]}
            )
            child = await resp.json()
            assert child["parent_id"] == parent["id"]

    @pytest.mark.asyncio
    async def test_create_folder_accepts_default_agent(self, tmp_path, monkeypatch):
        """The create modal collects the full folder config, so POST must take it."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders",
                json={"name": "Payments", "default_agent": "kirocrew-dev"},
            )
            assert resp.status == 201
            data = await resp.json()
            assert data["default_agent"] == "kirocrew-dev"

    @pytest.mark.asyncio
    async def test_create_folder_accepts_palette_color(self, tmp_path, monkeypatch):
        """Create persists an allowlisted palette color and rejects others."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "Redteam", "color": "#ef4444"}
            )
            assert resp.status == 201
            assert (await resp.json())["color"] == "#ef4444"
            bad = await client.post("/api/chat/folders", json={"name": "Bad", "color": "#123456"})
            assert bad.status == 400
            assert (await bad.json())["code"] == "color_invalid"

    def test_folder_color_palette_matches_frontend_catalog(self):
        """The backend allowlist and the frontend catalog must offer the same
        hues: the modal renders FOLDER_COLOR_PALETTE (folderColorCatalog.tsx)
        while create/PATCH validate against _FOLDER_COLOR_PALETTE, so drift
        means the UI offers a swatch the backend rejects with a 400."""
        from pathlib import Path

        from kiro_crew.dashboard.chat_folders import _FOLDER_COLOR_PALETTE

        catalog = (
            Path(__file__).resolve().parent.parent
            / "website"
            / "src"
            / "components"
            / "folderColorCatalog.tsx"
        )
        frontend = set(re.findall(r"#[0-9a-f]{6}", catalog.read_text(encoding="utf-8")))
        assert frontend == set(_FOLDER_COLOR_PALETTE)

    @pytest.mark.asyncio
    async def test_patch_color_set_and_clear(self, tmp_path, monkeypatch):
        """PATCH color: allowlisted value sets, empty string clears the key."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Blue"})
            fid = (await resp.json())["id"]
            resp = await client.patch(f"/api/chat/folders/{fid}", json={"color": "#3b82f6"})
            assert resp.status == 200
            assert state._folders[0]["color"] == "#3b82f6"
            resp = await client.patch(f"/api/chat/folders/{fid}", json={"color": ""})
            assert resp.status == 200
            assert "color" not in state._folders[0]

    @pytest.mark.asyncio
    async def test_create_folder_invalid_parent_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "Orphan", "parent_id": "nonexistent"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_create_folder_empty_name_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": ""})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_update_folder_rename(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Old"})
            folder = await resp.json()
            resp = await client.patch(f"/api/chat/folders/{folder['id']}", json={"name": "New"})
            assert resp.status == 200
            data = await resp.json()
            assert data["name"] == "New"

    @pytest.mark.asyncio
    async def test_a_failed_folder_write_restores_the_slots_it_unfiled(self, tmp_path, monkeypatch):
        """Delete must not half-land: unfiled slots go back if the store write fails.

        The handler unfiles the folder's slots first, then commits the folder
        removal. If that commit fails, leaving the slots unfiled strands
        conversations outside a folder that still exists. Restoring them is
        deliberately order-neutral — the reverse order strands a dangling
        folder_id instead, so undoing whichever half landed is what closes both.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            folder = await (await client.post("/api/chat/folders", json={"name": "Work"})).json()
            slot = state.get_or_create_slot("in-folder")
            slot.folder_id = folder["id"]

            def boom(path, data):
                raise OSError("disk full")

            monkeypatch.setattr(state, "_atomic_write_json", boom)
            resp = await client.delete(f"/api/chat/folders/{folder['id']}")
            assert resp.status >= 500

        assert slot.folder_id == folder["id"], (
            "the slot stayed unfiled after the folder removal failed; the delete "
            "half-landed and the conversation left a folder that still exists"
        )

    @pytest.mark.asyncio
    async def test_a_parent_deleted_under_the_lock_rejects_the_create(self, tmp_path, monkeypatch):
        """A child must not land with a dangling parent_id.

        Parent existence is validated before the store lock is taken, so a
        concurrent delete of that parent would otherwise persist an orphan. Both
        the sidebar and the folder picker surface orphans at the top level rather
        than hiding them, so this is a tree-shape defect rather than a
        disappearance — but still a folder nobody asked to put at the root.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            parent = await (await client.post("/api/chat/folders", json={"name": "P"})).json()

            real = state.mutate_folders

            async def delete_parent_then_apply(mutate):
                # Stand in for a concurrent DELETE landing while this create
                # waits for the store lock.
                state._folders[:] = [f for f in state._folders if f["id"] != parent["id"]]
                return await real(mutate)

            monkeypatch.setattr(state, "mutate_folders", delete_parent_then_apply)
            resp = await client.post(
                "/api/chat/folders", json={"name": "Child", "parent_id": parent["id"]}
            )
            assert resp.status == 400, (
                f"create was accepted with a deleted parent (status {resp.status}); "
                "the folder persists as an orphan"
            )
        assert not any(f.get("name") == "Child" for f in state._folders)

    @pytest.mark.asyncio
    async def test_update_folder_collapse(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "F"})
            folder = await resp.json()
            resp = await client.patch(f"/api/chat/folders/{folder['id']}", json={"collapsed": True})
            data = await resp.json()
            assert data["collapsed"] is True

    @pytest.mark.asyncio
    async def test_update_folder_reparent_into_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            a = await (await client.post("/api/chat/folders", json={"name": "A"})).json()
            b = await (await client.post("/api/chat/folders", json={"name": "B"})).json()
            resp = await client.patch(f"/api/chat/folders/{b['id']}", json={"parent_id": a["id"]})
            assert resp.status == 200
            assert (await resp.json())["parent_id"] == a["id"]
            assert state._folders[1]["parent_id"] == a["id"]

    @pytest.mark.asyncio
    async def test_update_folder_reparent_to_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            a = await (await client.post("/api/chat/folders", json={"name": "A"})).json()
            b = await (
                await client.post("/api/chat/folders", json={"name": "B", "parent_id": a["id"]})
            ).json()
            # Empty string and null both mean "move to top level".
            resp = await client.patch(f"/api/chat/folders/{b['id']}", json={"parent_id": ""})
            assert resp.status == 200
            assert (await resp.json())["parent_id"] == ""
            resp = await client.patch(f"/api/chat/folders/{b['id']}", json={"parent_id": a["id"]})
            assert (await resp.json())["parent_id"] == a["id"]
            resp = await client.patch(f"/api/chat/folders/{b['id']}", json={"parent_id": None})
            assert resp.status == 200
            assert (await resp.json())["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_concurrent_opposite_reparents_cannot_persist_a_cycle(
        self, tmp_path, monkeypatch
    ):
        """Two opposite reparents must not both apply and orphan the pair.

        The tree-shape rules (parent exists, target is not a descendant) are
        checked once before the store lock as a fast reject, but a folder write
        is serialized and off-loop, so while one is in flight BOTH requests can
        validate against the same pre-state. If the winner's mutation were not
        re-tested under the lock, `A -> B` and `B -> A` would both persist and
        neither folder would be reachable from the root any more.

        The lock is held deliberately here so both handlers are provably queued
        on it before either applies -- the window is opened, not raced for.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            a = await (await client.post("/api/chat/folders", json={"name": "A"})).json()
            b = await (await client.post("/api/chat/folders", json={"name": "B"})).json()

            gate = asyncio.Event()

            async def _hold_the_store() -> None:
                async with state._folders_lock:
                    await gate.wait()

            holder = asyncio.create_task(_hold_the_store())
            while not state._folders_lock.locked():
                await asyncio.sleep(0)

            t1 = asyncio.create_task(
                client.patch(f"/api/chat/folders/{a['id']}", json={"parent_id": b["id"]})
            )
            t2 = asyncio.create_task(
                client.patch(f"/api/chat/folders/{b['id']}", json={"parent_id": a["id"]})
            )

            # Both requests have passed their pre-lock validation against the
            # SAME pre-state and are now waiting on the store. _folders_lock is a
            # LoopBoundLock; until_parked reads this loop's inner asyncio.Lock.
            await until_parked(state._folders_lock, count=2)
            gate.set()
            r1, r2 = await asyncio.gather(t1, t2)
            await holder

            assert sorted([r1.status, r2.status]) == [200, 409], (
                "both opposite reparents were accepted; the tree-shape rules are "
                "not being re-tested under the store lock"
            )

            parents = {f["id"]: f.get("parent_id", "") for f in state._folders}

            def _reaches_root(fid: str) -> bool:
                seen: set[str] = set()
                cur = fid
                while cur:
                    if cur in seen:
                        return False
                    seen.add(cur)
                    cur = parents.get(cur, "")
                return True

            assert all(
                _reaches_root(fid) for fid in parents
            ), f"a parent cycle was persisted: {parents}"
            stored = json.loads((tmp_path / state._FOLDERS_FILE).read_text(encoding="utf-8"))
            assert {f["id"]: f.get("parent_id", "") for f in stored} == parents

    @pytest.mark.asyncio
    async def test_update_folder_reparent_self_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            a = await (await client.post("/api/chat/folders", json={"name": "A"})).json()
            resp = await client.patch(f"/api/chat/folders/{a['id']}", json={"parent_id": a["id"]})
            assert resp.status == 400
            assert state._folders[0]["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_update_folder_reparent_unknown_parent_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            a = await (await client.post("/api/chat/folders", json={"name": "A"})).json()
            resp = await client.patch(f"/api/chat/folders/{a['id']}", json={"parent_id": "nope"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_update_folder_reparent_cycle_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            a = await (await client.post("/api/chat/folders", json={"name": "A"})).json()
            b = await (
                await client.post("/api/chat/folders", json={"name": "B", "parent_id": a["id"]})
            ).json()
            c = await (
                await client.post("/api/chat/folders", json={"name": "C", "parent_id": b["id"]})
            ).json()
            # A -> C would create A > B > C > A
            resp = await client.patch(f"/api/chat/folders/{a['id']}", json={"parent_id": c["id"]})
            assert resp.status == 400
            assert "descendant" in (await resp.json())["error"]
            # Direct child is also a descendant
            resp = await client.patch(f"/api/chat/folders/{a['id']}", json={"parent_id": b["id"]})
            assert resp.status == 400
            assert state._folders[0]["parent_id"] == ""

    @pytest.mark.asyncio
    async def test_update_folder_rejected_field_leaves_no_partial_mutation(
        self, tmp_path, monkeypatch
    ):
        """A PATCH mixing a VALID field (name) with an INVALID one (bad parent_id)
        must be all-or-nothing: the 400 rejection must NOT persist the name change
        (validate-all-before-mutate)."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            f = await (await client.post("/api/chat/folders", json={"name": "Original"})).json()
            resp = await client.patch(
                f"/api/chat/folders/{f['id']}",
                json={"name": "Renamed", "parent_id": "does-not-exist"},
            )
            assert resp.status == 400
            # The name change from the SAME rejected request must not have landed.
            assert state._folders[0]["name"] == "Original"
            # And a later valid update must not resurrect the rejected name.
            resp = await client.patch(f"/api/chat/folders/{f['id']}", json={"collapsed": True})
            assert resp.status == 200
            assert state._folders[0]["name"] == "Original"

    @pytest.mark.asyncio
    async def test_create_folder_hidden_defaults_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "F"})
            assert (await resp.json())["hidden"] is False
            resp = await client.get("/api/chat/folders")
            assert (await resp.json())[0]["hidden"] is False

    @pytest.mark.asyncio
    async def test_update_folder_hidden(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "F"})
            folder = await resp.json()
            resp = await client.patch(f"/api/chat/folders/{folder['id']}", json={"hidden": True})
            assert (await resp.json())["hidden"] is True
            assert state._folders[0]["hidden"] is True
            resp = await client.patch(f"/api/chat/folders/{folder['id']}", json={"hidden": False})
            assert (await resp.json())["hidden"] is False

    @pytest.mark.asyncio
    async def test_folders_get_includes_history_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state._folders = [
            {"id": "f1", "name": "A", "order": 0, "collapsed": False},
            {"id": "f2", "name": "B", "order": 1, "collapsed": False},
        ]
        # History count is computed from the full session list (authoritative),
        # not the paginated client history window.
        monkeypatch.setattr(
            state.conversation_log,
            "list_sessions",
            lambda: [
                {"key": "a", "folder_id": "f1"},
                {"key": "b", "folder_id": "f1"},
                {"key": "c", "folder_id": "f2"},
                {"key": "d"},  # unfiled — ignored
            ],
        )
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/chat/folders")
            folders = {f["id"]: f for f in await resp.json()}
            assert folders["f1"]["history_count"] == 2
            assert folders["f2"]["history_count"] == 1

    @pytest.mark.asyncio
    async def test_assign_slot_to_folder_unhides(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("myslot")
        state._folders = [
            {"id": "f1", "name": "Test", "order": 0, "collapsed": False, "hidden": True}
        ]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            # Moving a session into a hidden folder re-engages it (Model B) → un-hides.
            resp = await client.patch("/api/chat/slots/myslot/folder", json={"folder_id": "f1"})
            assert resp.status == 200
            assert state._folders[0]["hidden"] is False

    @pytest.mark.asyncio
    async def test_resume_session_unhides_folder(self, tmp_path, monkeypatch):
        """Reviving a history session filed in a hidden folder un-hides it (Model B).

        Complements test_assign_slot_to_folder_unhides (the move path): here the
        re-engage happens via api_chat_slot_resume loading an archived session
        from history, which is the revive path that lets a hidden folder reappear.
        """
        from kiro_crew.dashboard.chat import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state._folders = [
            {"id": "f1", "name": "Test", "order": 0, "collapsed": False, "hidden": True}
        ]
        # Create a session filed in f1, persist it to history, then drop the active
        # slot so resume loads it fresh from history (the revive path).
        slot = state.get_or_create_slot("revive1")
        slot.folder_id = "f1"
        slot.append("user", "old msg")
        slot.drain()
        _save_slot_to_history(state, slot, closed=True)
        # Closed BEFORE the resume begins, whatever the clock's resolution.
        close_before_resume(state.conversation_log, "dashboard:revive1")
        state._slots.pop("revive1", None)
        assert state._folders[0]["hidden"] is True  # still hidden before revive

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/revive1/resume", json={"key": "dashboard:revive1"}
            )
            assert resp.status == 200
        assert state._folders[0]["hidden"] is False  # revive re-engaged → un-hid

    @pytest.mark.asyncio
    async def test_update_folder_default_agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "MS&AD"})
            folder = await resp.json()
            resp = await client.patch(
                f"/api/chat/folders/{folder['id']}", json={"default_agent": "msad"}
            )
            data = await resp.json()
            assert data["default_agent"] == "msad"
            # Verify persistence
            assert state._folders[0]["default_agent"] == "msad"

    @pytest.mark.asyncio
    async def test_update_folder_clear_default_agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state._folders = [
            {"id": "f1", "name": "Test", "order": 0, "collapsed": False, "default_agent": "nissay"}
        ]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/folders/f1", json={"default_agent": ""})
            data = await resp.json()
            assert data["default_agent"] == ""

    @pytest.mark.asyncio
    async def test_create_folder_with_project_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        proj = tmp_path / "proj"
        proj.mkdir()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "P", "project_dir": str(proj)}
            )
            assert resp.status == 201
            data = await resp.json()
            assert data["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_create_folder_relative_project_dir_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "P", "project_dir": "relative/path"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_create_folder_nonexistent_project_dir_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        missing = tmp_path / "does-not-exist"
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "P", "project_dir": str(missing)}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_create_folder_sensitive_project_dir_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "P", "project_dir": "~/.ssh"}
            )
            assert resp.status == 400
            data = await resp.json()
            assert "sensitive" in data["error"]

    @pytest.mark.asyncio
    async def test_update_folder_project_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        proj = tmp_path / "proj2"
        proj.mkdir()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "P"})
            folder = await resp.json()
            resp = await client.patch(
                f"/api/chat/folders/{folder['id']}", json={"project_dir": str(proj)}
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["project_dir"] == os.path.realpath(str(proj))

    @pytest.mark.asyncio
    async def test_slot_create_inherits_nearest_folder_project(self, tmp_path, monkeypatch):
        """The server owns folder inheritance when the client cache omits project."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        root_project = tmp_path / "root-project"
        parent_project = tmp_path / "parent-project"
        root_project.mkdir()
        parent_project.mkdir()
        state._folders = [
            {
                "id": "root",
                "name": "Root",
                "order": 0,
                "parent_id": "",
                "project_dir": str(root_project),
            },
            {
                "id": "parent",
                "name": "Parent",
                "order": 1,
                "parent_id": "root",
                "project_dir": str(parent_project),
            },
            {
                "id": "child",
                "name": "Child",
                "order": 2,
                "parent_id": "parent",
                "project_dir": "",
            },
        ]
        mock_cfg = MagicMock()
        mock_cfg.dashboard.default_project = ""
        # A bare MagicMock leaks into the slot: the handler stamps
        # cfg.default_agent (a truthy MagicMock) as the slot's agent when the
        # request names none, and the coalesced slots broadcast then dies in
        # json.dumps ("Object of type MagicMock is not JSON serializable"),
        # 500ing the create. Pin it to a string like the sibling cfg mocks.
        mock_cfg.default_agent = ""
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.KiroCrewConfig.load", lambda: mock_cfg
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.default_project_dir",
            lambda _workspace: str(root_project),
        )
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.schedule_eager_spawn",
            lambda *_args, **_kwargs: None,
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post(
                "/api/chat/slots",
                json={"name": "folder-project", "folder_id": "child"},
            )
            data = await resp.json()

        assert resp.status == 200
        assert data["folder_id"] == "child"
        assert data["project"] == os.path.realpath(str(parent_project))
        assert state._slots["folder-project"].project == data["project"]

    @pytest.mark.asyncio
    async def test_slot_create_rejects_invalid_inherited_project(self, tmp_path, monkeypatch):
        """A stale folder path fails before a partially configured slot is created."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state._folders = [
            {
                "id": "folder",
                "name": "Folder",
                "order": 0,
                "parent_id": "",
                "project_dir": str(tmp_path / "missing"),
            }
        ]

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post(
                "/api/chat/slots",
                json={"name": "invalid-folder-project", "folder_id": "folder"},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["code"] == "folder_project_invalid"
        assert state._slots == {}

    @pytest.mark.asyncio
    async def test_slot_create_does_not_rescope_existing_named_slot(self, tmp_path, monkeypatch):
        """Folder inheritance initializes new slots; explicit project changes stay explicit."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        old_project = tmp_path / "old-project"
        folder_project = tmp_path / "folder-project"
        old_project.mkdir()
        folder_project.mkdir()
        slot = state.get_or_create_slot("existing")
        slot.project = str(old_project)
        slot.append("user", "existing conversation")
        slot.drain()
        state._folders = [
            {
                "id": "folder",
                "name": "Folder",
                "order": 0,
                "parent_id": "",
                "project_dir": str(folder_project),
            }
        ]
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.schedule_eager_spawn",
            lambda *_args, **_kwargs: None,
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post(
                "/api/chat/slots",
                json={"name": "existing", "folder_id": "folder"},
            )

        assert resp.status == 200
        assert slot.project == str(old_project)

    def test_resolve_folder_project_dir_terminates_on_cycle(self):
        from kiro_crew.dashboard.chat_folders import _resolve_folder_project_dir

        folders = [
            {"id": "a", "parent_id": "b", "project_dir": ""},
            {"id": "b", "parent_id": "a", "project_dir": ""},
        ]
        assert _resolve_folder_project_dir(folders, "a") == ("", None)

    @pytest.mark.asyncio
    async def test_update_folder_empty_name_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Keep"})
            folder = await resp.json()
            resp = await client.patch(f"/api/chat/folders/{folder['id']}", json={"name": "  "})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_update_nonexistent_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/folders/nonexistent", json={"name": "X"})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Delete Me"})
            folder = await resp.json()
            resp = await client.delete(f"/api/chat/folders/{folder['id']}")
            assert resp.status == 200
            resp = await client.get("/api/chat/folders")
            assert await resp.json() == []

    @pytest.mark.asyncio
    async def test_delete_folder_reparents_children(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state._folders = [
            {"id": "parent", "name": "Parent", "order": 0, "collapsed": False},
            {"id": "child", "name": "Child", "order": 1, "collapsed": False, "parent_id": "parent"},
        ]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            await client.delete("/api/chat/folders/parent")
            assert len(state._folders) == 1
            assert state._folders[0]["id"] == "child"
            assert state._folders[0].get("parent_id") == ""

    @pytest.mark.asyncio
    async def test_delete_folder_ungroups_slots(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.folder_id = "f-del"
        state._folders.append({"id": "f-del", "name": "X", "order": 0, "collapsed": False})
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            await client.delete("/api/chat/folders/f-del")
            assert slot.folder_id == ""

    @pytest.mark.asyncio
    async def test_assign_slot_to_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("myslot")
        state._folders = [{"id": "f1", "name": "Test", "order": 0, "collapsed": False}]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/slots/myslot/folder", json={"folder_id": "f1"})
            assert resp.status == 200
            data = await resp.json()
            assert data["folder_id"] == "f1"
            assert state._slots["myslot"].folder_id == "f1"

    @pytest.mark.asyncio
    async def test_slot_folder_change_sets_reinject_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("myslot")
        state._folders = [{"id": "f1", "name": "Test", "order": 0, "collapsed": False}]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            # Moving into a folder flags the slot for a one-shot [FOLDER] re-inject.
            resp = await client.patch("/api/chat/slots/myslot/folder", json={"folder_id": "f1"})
            assert resp.status == 200
            assert slot._folder_changed is True
            # A no-op PATCH (same folder) must not re-flag.
            slot._folder_changed = False
            resp = await client.patch("/api/chat/slots/myslot/folder", json={"folder_id": "f1"})
            assert resp.status == 200
            assert slot._folder_changed is False

    @pytest.mark.asyncio
    async def test_unassign_slot_from_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("myslot")
        slot.folder_id = "f1"
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/slots/myslot/folder", json={"folder_id": ""})
            assert resp.status == 200
            assert state._slots["myslot"].folder_id == ""

    @pytest.mark.asyncio
    async def test_assign_folder_nonexistent_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/slots/nope/folder", json={"folder_id": "f1"})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_expected_created_matching_the_live_slot_files_it(self, tmp_path, monkeypatch):
        """The generation token is optional; when it matches the slot's own
        ``created_at`` the write lands exactly as an untokened one does."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("myslot")
        state._folders = [{"id": "f1", "name": "Test", "order": 0, "collapsed": False}]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch(
                "/api/chat/slots/myslot/folder",
                json={"folder_id": "f1", "expected_created": slot.created_at},
            )
            assert resp.status == 200
            assert slot.folder_id == "f1"

    @pytest.mark.asyncio
    async def test_expected_created_from_a_replaced_slot_is_refused_409(
        self, tmp_path, monkeypatch
    ):
        """A caller that resolved a slot in an earlier request, then saw its
        tab close and the same key recreated for another conversation, carries
        the OLD slot's ``created_at``. The recreated slot has the same key and
        the same ``dashboard:<key>`` transcript key, so only the token can tell
        them apart — and it must, or the stale write files someone else's
        session."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        first = state.get_or_create_slot("myslot")
        stale_token = first.created_at
        # Close the tab and recreate the key: a different slot object, a
        # different birth stamp, the same key.
        del state._slots["myslot"]
        replacement = state.get_or_create_slot("myslot")
        replacement.created_at = stale_token + "-later"
        state._folders = [{"id": "f1", "name": "Test", "order": 0, "collapsed": False}]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch(
                "/api/chat/slots/myslot/folder",
                json={"folder_id": "f1", "expected_created": stale_token},
            )
            assert resp.status == 409
            assert (await resp.json())["code"] == "session_gone"
            # The replacement conversation was not filed.
            assert replacement.folder_id == ""
            assert replacement._folder_changed is False

    @pytest.mark.asyncio
    async def test_assign_nonexistent_folder_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("myslot")
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch(
                "/api/chat/slots/myslot/folder", json={"folder_id": "nonexistent"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_pin_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("myslot")
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/slots/myslot/pin", json={"pinned": True})
            assert resp.status == 200
            data = await resp.json()
            assert data["pinned"] is True
            assert state._slots["myslot"].pinned is True

    @pytest.mark.asyncio
    async def test_unpin_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("myslot")
        slot.pinned = True
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/slots/myslot/pin", json={"pinned": False})
            assert resp.status == 200
            assert state._slots["myslot"].pinned is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
    async def test_pin_slot_rejects_non_boolean_values(self, tmp_path, monkeypatch, value):
        """The API must not apply Python truthiness to JSON metadata."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("myslot")
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/slots/myslot/pin", json={"pinned": value})
            assert resp.status == 400
            payload = await resp.json()
            assert payload["error"] == "pinned must be a boolean"
            assert payload["code"] == "pinned_not_bool"
        assert slot.pinned is False

    @pytest.mark.asyncio
    async def test_slots_include_pinned(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.pinned = True
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/chat/slots")
            slots = await resp.json()
            assert any(s.get("pinned") is True for s in slots)

    @pytest.mark.asyncio
    async def test_slots_include_folder_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.folder_id = "f-abc"
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/chat/slots")
            slots = await resp.json()
            assert any(s["folder_id"] == "f-abc" for s in slots)


class TestFolderTags:
    """Folder `tags` — vocabulary-constrained list persisted on create/update,
    and stripped from folders when a tag is deleted."""

    @staticmethod
    def _seed_vocabulary(state, tag_ids):
        state._tags = [
            {"id": tid, "name": tid, "color": "#6b7280", "order": i}
            for i, tid in enumerate(tag_ids)
        ]
        # A directly-seeded vocabulary is authoritative by definition — the
        # constructor fails closed (False) until load_tags() runs.
        state._tags_authoritative = True

    @staticmethod
    def _app_with_tag_delete(state):
        """Folder endpoints plus the tag-delete route, on one app."""
        from kiro_crew.dashboard.chat_tags import api_chat_tag_delete

        app = _make_folder_app(state)
        app.router.add_delete("/api/chat/tags/{id}", api_chat_tag_delete)
        return app

    @pytest.mark.asyncio
    async def test_create_and_update_with_valid_tags_persists(self, tmp_path, monkeypatch):
        """(a) Create with valid tags persists them; PATCH replaces the set."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1", "t2", "t3"])
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "Payments", "tags": ["t1", "t2"]}
            )
            assert resp.status == 201
            data = await resp.json()
            assert data["tags"] == ["t1", "t2"]
            fid = data["id"]
            # Reloaded from the store reflects the persisted list.
            assert state._folders[0]["tags"] == ["t1", "t2"]

            # PATCH replaces the tag set.
            resp = await client.patch(f"/api/chat/folders/{fid}", json={"tags": ["t3"]})
            assert resp.status == 200
            assert state._folders[0]["tags"] == ["t3"]

            # An empty list clears the key entirely (absent means no tags).
            resp = await client.patch(f"/api/chat/folders/{fid}", json={"tags": []})
            assert resp.status == 200
            assert "tags" not in state._folders[0]

    @pytest.mark.asyncio
    async def test_create_dedupes_tags(self, tmp_path, monkeypatch):
        """Duplicates collapse (order preserved) and a valid set persists."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1", "t2"])
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "Dupes", "tags": ["t1", "t1", "t2"]}
            )
            assert resp.status == 201
            assert (await resp.json())["tags"] == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_malformed_vocabulary_id_does_not_crash_validation(self, tmp_path, monkeypatch):
        """A malformed persisted vocabulary entry (non-string ``id`` — e.g. a
        hand-edited tags.json carrying a list) must degrade to "unknown id"
        rather than crash the validator's set build with an unhashable type
        (which would 500 every folder-tag operation)."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1"])
        # Corrupt entries alongside the valid one: unhashable id, missing id.
        state._tags.append({"id": ["not", "a", "string"], "name": "bad"})
        state._tags.append({"name": "no-id"})
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "Robust", "tags": ["t1", "t9"]}
            )
            assert resp.status == 201
            # The valid id survives; the unknown one is filtered; no crash.
            assert (await resp.json())["tags"] == ["t1"]

    @pytest.mark.asyncio
    async def test_unreadable_vocab_does_not_wipe_folder_tags(self, tmp_path, monkeypatch):
        """FAIL-OPEN parity with ``validate_folder_tag_ids``: when tags.json
        was unreadable at boot (vocabulary UNKNOWN, ``_tags_authoritative``
        False), a folder PATCH carrying ``tags`` must keep the ids rather than
        intersecting them against the unknown (empty) vocabulary — that
        intersection would silently wipe the folder's tags and the PATCH would
        persist the loss."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1", "t2"])
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "Payments", "tags": ["t1", "t2"]}
            )
            assert resp.status == 201
            fid = (await resp.json())["id"]

            # Simulate a later boot where tags.json failed to load.
            state._tags = []
            state._tags_authoritative = False

            # A PATCH that echoes the stored tags back (any rename flow does
            # this) must not wipe them; shape guards and dedupe still apply.
            resp = await client.patch(
                f"/api/chat/folders/{fid}",
                json={"name": "Renamed", "tags": ["t1", "t1", "t2", 7]},
            )
            assert resp.status == 200
            assert state._folders[0]["tags"] == ["t1", "t2"]  # preserved, not wiped

    @pytest.mark.asyncio
    async def test_folder_tag_writes_hold_the_tags_write_lock(self, tmp_path, monkeypatch):
        """The point-of-application intersection and the folders write are ONE
        critical section under ``tags_write_lock`` (the invariant every tag
        consumer follows): a tag deletion committing between them would slip a
        just-deleted id past the strip pass and back onto the folder."""
        from kiro_crew.dashboard.chat_tags import tags_write_lock

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1"])
        held: list[bool] = []
        orig_mutate = state.mutate_folders

        async def _spy(fn, **kwargs):
            held.append(tags_write_lock(state).locked())
            return await orig_mutate(fn, **kwargs)

        monkeypatch.setattr(state, "mutate_folders", _spy)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Locked", "tags": ["t1"]})
            assert resp.status == 201
            fid = (await resp.json())["id"]
            resp = await client.patch(f"/api/chat/folders/{fid}", json={"tags": ["t1"]})
            assert resp.status == 200
        assert held[:2] == [True, True], (
            "a folder tags write ran outside tags_write_lock; a concurrent tag "
            "deletion could resurrect a deleted id onto the folder"
        )

    @pytest.mark.asyncio
    async def test_unknown_tag_id_is_filtered_not_400(self, tmp_path, monkeypatch):
        """(b) An unknown tag id is silently dropped on create and update.

        Mirrors ``api_chat_slot_tags`` exactly: a dangling id can legitimately
        exist (the tag delete's best-effort folder strip can fail), and a
        strict 400 here would make every subsequent save of that folder fail —
        the "permanently uneditable folder" class. Filtering at the write
        sheds the stale reference instead.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1"])
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/folders", json={"name": "Mixed", "tags": ["nope", "t1"]}
            )
            assert resp.status == 201
            fid = (await resp.json())["id"]
            # Unknown id dropped, known id kept.
            assert state._folders[0].get("tags") == ["t1"]

            bad_patch = await client.patch(f"/api/chat/folders/{fid}", json={"tags": ["ghost"]})
            assert bad_patch.status == 200
            # An all-unknown list filters down to empty — tags cleared, not 400.
            assert not state._folders[0].get("tags")

    @pytest.mark.asyncio
    async def test_non_array_tags_rejected_400(self, tmp_path, monkeypatch):
        """A non-array `tags` payload is a 400, not a 500."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1"])
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/folders", json={"name": "Bad", "tags": "t1"})
            assert resp.status == 400
            assert (await resp.json())["code"] == "tags_invalid"

    @pytest.mark.asyncio
    async def test_deleting_a_tag_strips_it_from_folders(self, tmp_path, monkeypatch):
        """(f) Deleting a tag from the vocabulary removes its id from every folder."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        self._seed_vocabulary(state, ["t1", "t2"])
        app = self._app_with_tag_delete(state)
        async with TestClient(TestServer(app)) as client:
            # Two folders carrying t1; one also carries t2.
            a = await (
                await client.post("/api/chat/folders", json={"name": "A", "tags": ["t1", "t2"]})
            ).json()
            b = await (
                await client.post("/api/chat/folders", json={"name": "B", "tags": ["t1"]})
            ).json()

            resp = await client.delete("/api/chat/tags/t1")
            assert resp.status == 200

        fa = next(f for f in state._folders if f["id"] == a["id"])
        fb = next(f for f in state._folders if f["id"] == b["id"])
        # t1 stripped everywhere; t2 untouched; a folder left with no tags loses
        # the key entirely.
        assert fa["tags"] == ["t2"]
        assert "tags" not in fb


class TestFolderPersistence:
    def test_load_folders_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        import json

        (tmp_path / "folders.json").write_text(
            json.dumps([{"id": "f1", "name": "Test", "order": 0}])
        )
        state = _make_state(tmp_path)
        state.load_folders()
        assert len(state._folders) == 1
        assert state._folders[0]["name"] == "Test"

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state._folders = [{"id": "f1", "name": "Roundtrip", "order": 0, "collapsed": True}]
        state.save_folders()
        state._folders = []
        state.load_folders()
        assert state._folders[0]["name"] == "Roundtrip"
        assert state._folders[0]["collapsed"] is True

    def test_load_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.load_folders()
        assert state._folders == []

    def test_load_corrupted_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        (tmp_path / "folders.json").write_text("not json")
        state = _make_state(tmp_path)
        state.load_folders()
        assert state._folders == []


class TestGenerateEmojiForName:
    @pytest.mark.asyncio
    async def test_redaction_applied(self, tmp_path, monkeypatch):
        """generate_emoji_for_name (artifact-folder path) still redacts replies."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from kiro_crew.dashboard.chat_folders import generate_emoji_for_name

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        mock_event = MagicMock()
        mock_event.kind = "text_chunk"
        mock_event.text = "🔥"
        done_event = MagicMock()
        done_event.kind = "complete"
        monkeypatch.setattr("kiro_crew.providers.base.EVENT_TEXT_CHUNK", "text_chunk")
        monkeypatch.setattr("kiro_crew.providers.base.EVENT_COMPLETE", "complete")
        monkeypatch.setattr("kiro_crew.providers.base.EVENT_PERMISSION_REQUEST", "permission")

        mock_client = _provider_mock()
        mock_client.prompt = MagicMock(return_value=AsyncIterator([mock_event, done_event]))
        state.sessions.get_bg_session = AsyncMock(return_value=mock_client)
        state.push_slots_update = MagicMock()

        with (
            patch(
                "kiro_crew.dashboard.chat_folders.redact_exfiltration_urls",
                return_value=("🔥", False),
            ) as mock_url,
            patch(
                "kiro_crew.dashboard.chat_folders.redact_credentials", return_value=("🔥", False)
            ) as mock_cred,
        ):
            icon = await generate_emoji_for_name(state, "Oncall")
            assert icon == "🔥"
            mock_url.assert_called_once()
            mock_cred.assert_called_once()

    @pytest.mark.asyncio
    async def test_variation_selector_emoji_accepted(self, tmp_path, monkeypatch):
        """Emoji with U+FE0F variation selector (e.g. ❤️) accepted by the
        emoji generator (still used for artifact-library folders)."""
        from unittest.mock import AsyncMock, MagicMock

        from kiro_crew.dashboard.chat_folders import generate_emoji_for_name

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)

        mock_event = MagicMock()
        mock_event.kind = "text_chunk"
        mock_event.text = "\u2764\ufe0f"  # ❤️
        done_event = MagicMock()
        done_event.kind = "complete"
        monkeypatch.setattr("kiro_crew.providers.base.EVENT_TEXT_CHUNK", "text_chunk")
        monkeypatch.setattr("kiro_crew.providers.base.EVENT_COMPLETE", "complete")
        monkeypatch.setattr("kiro_crew.providers.base.EVENT_PERMISSION_REQUEST", "permission")

        mock_client = _provider_mock()
        mock_client.prompt = MagicMock(return_value=AsyncIterator([mock_event, done_event]))
        state.sessions.get_bg_session = AsyncMock(return_value=mock_client)

        icon = await generate_emoji_for_name(state, "Love")
        assert icon == "\u2764\ufe0f"


class TestFolderAssignmentPersistence:
    @pytest.mark.asyncio
    async def test_folder_assignment_saves_to_history(self, tmp_path, monkeypatch):
        """api_chat_slot_folder should call _save_slot_to_history for new sessions."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("myslot")
        slot.append("user", "hello")
        slot.drain()
        state._folders = [{"id": "f1", "name": "Test", "order": 0, "collapsed": False}]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            await client.patch("/api/chat/slots/myslot/folder", json={"folder_id": "f1"})
            path = tmp_path / "dashboard_myslot.jsonl"
            assert path.exists()
            import json

            meta = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
            assert meta["folder_id"] == "f1"

    @pytest.mark.asyncio
    async def test_folder_assignment_persists_on_resumed_session(self, tmp_path, monkeypatch):
        """Regression: folder_id must reach disk even when slot is a resumed
        session with no new messages.

        Root cause: _save_slot_to_history had an early-return guard that
        skipped disk writes when ``slot._resumed_count > 0 and
        len(messages) <= slot._resumed_count``. Metadata-only changes like
        folder assignment don't grow the message count past the resumed
        marker, so the save was silently dropped — folder_id never reached
        disk and the move was lost on the next gateway restart.

        Fix: folder endpoint passes ``force=True`` which bypasses the guard.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("resumedslot")
        slot.append("user", "old message from before restart")
        slot.drain()
        # A genuinely resumed slot's file exists on disk (the restore read it).
        # Persist first so the fixture matches reality — a slot claiming
        # on-disk history whose file is missing is the delete-won state the
        # save now refuses to recreate.
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        _save_slot_to_history(state, slot, force=True)
        # Mark slot as a resumed session (simulates being restored from disk).
        # The guard fires when _resumed_count >= len(messages).
        slot._resumed_count = len(slot.messages)
        state._folders = [{"id": "f-resumed", "name": "Build", "order": 0, "collapsed": False}]
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch(
                "/api/chat/slots/resumedslot/folder",
                json={"folder_id": "f-resumed"},
            )
            assert resp.status == 200
            path = tmp_path / "dashboard_resumedslot.jsonl"
            assert path.exists(), "fixture save should have created the session file"
            import json

            meta = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
            assert meta.get("folder_id") == "f-resumed", (
                "folder_id was silently dropped on resumed session — "
                "force=True must bypass the _resumed_count guard"
            )

    @pytest.mark.asyncio
    async def test_pin_toggle_persists_on_resumed_session(self, tmp_path, monkeypatch):
        """Regression: pinned flag must reach disk on resumed sessions.

        Same root cause as the folder regression — the resumed-count guard
        in _save_slot_to_history was blocking metadata-only writes. Pin
        endpoint now passes ``force=True``.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("pinslot")
        slot.append("user", "old message")
        slot.drain()
        # Persist first: a resumed slot's file exists on disk (see the folder
        # regression above — the missing-file variant is the delete-won state
        # the save refuses to recreate).
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        _save_slot_to_history(state, slot, force=True)
        slot._resumed_count = len(slot.messages)
        app = _make_folder_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.patch("/api/chat/slots/pinslot/pin", json={"pinned": True})
            assert resp.status == 200
            path = tmp_path / "dashboard_pinslot.jsonl"
            assert path.exists(), "fixture save should have created the session file"
            import json

            meta = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
            assert meta.get("pinned") is True, (
                "pinned was silently dropped on resumed session — "
                "force=True must bypass the _resumed_count guard"
            )

    def test_save_slot_force_bypasses_resumed_guard(self, tmp_path, monkeypatch):
        """Unit test: ``force=True`` must bypass the resumed-session guard.

        Without force, resumed sessions with no new messages skip the write.
        With force, the metadata-only mutation reaches disk regardless.
        """
        from kiro_crew.dashboard.chat import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("forceslot")
        slot.append("user", "hello")
        slot.drain()
        # Persist first: a genuinely resumed slot's file exists on disk. (The
        # missing-file variant is the delete-won state the save refuses to
        # recreate.)
        _save_slot_to_history(state, slot, force=True)
        path = tmp_path / "dashboard_forceslot.jsonl"
        assert path.exists()
        slot._resumed_count = len(slot.messages)
        # Model a genuinely-resumed, UNCHANGED slot: restore sets _dirty=False.
        # (A dirty slot — e.g. an in-place stop-event edit — must NOT be skipped;
        # see test_inplace_stop_resolution_persists.)
        slot._dirty = False
        slot.folder_id = "f-force"

        import json

        # Without force — the resumed-count guard skips the save, so the
        # metadata mutation must NOT reach disk.
        _save_slot_to_history(state, slot)
        meta = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
        assert meta.get("folder_id") is None, "guard must skip save when not forced"

        # With force — save bypasses the guard, folder_id is written.
        _save_slot_to_history(state, slot, force=True)

        meta = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
        assert meta.get("folder_id") == "f-force"


# ── Regenerate + variant switching ──


class TestRegenerateAndVariants:
    @pytest.mark.asyncio
    async def test_regenerate_truncates_and_stashes_variant(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "hello v1")
        slot.drain()
        captured = []

        async def _capture(*a, **kw):
            captured.extend(list(slot._pending_variants))

        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=_capture):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
                await asyncio.sleep(0)
        assert [m["role"] for m in slot.messages] == ["user"]
        assert len(captured) == 1
        assert captured[0]["content"] == "hello v1"

    @pytest.mark.asyncio
    async def test_regenerate_rejects_when_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "hello")

        # Simulate running task
        async def _noop():
            await asyncio.sleep(10)

        slot.task = asyncio.create_task(_noop())
        try:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 409
        finally:
            slot.task.cancel()

    @pytest.mark.asyncio
    async def test_regenerate_requires_prior_assistant(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "only user")
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/regenerate")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_switch_variant_updates_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "v2")
        slot.messages[-1]["variants"] = [
            {"content": "v1", "ts": "t1"},
            {"content": "v2", "ts": "t2"},
        ]
        slot.messages[-1]["variant_idx"] = 1
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", json={"index": 0})
            assert resp.status == 200
            assert slot.messages[-1]["content"] == "v1"
            assert slot.messages[-1]["variant_idx"] == 0

    @pytest.mark.asyncio
    async def test_switch_variant_index_out_of_range(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "v1")
        slot.messages[-1]["variants"] = [{"content": "v1"}]
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", json={"index": 5})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_regenerate_passes_hint_to_run_chat(self, tmp_path, monkeypatch):
        """_run_chat should receive a non-empty regenerate_hint kwarg."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "reply")
        slot.drain()
        mock_run = AsyncMock()
        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=mock_run):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
                # Let the scheduled task actually run so the mock records args
                await asyncio.sleep(0)
        mock_run.assert_called_once()
        _args, kwargs = mock_run.call_args
        assert kwargs.get("regenerate_hint"), "regenerate_hint must be non-empty"

    @pytest.mark.asyncio
    async def test_regenerate_preserves_existing_variants(self, tmp_path, monkeypatch):
        """When assistant already has variants[], regenerate keeps them and adds current."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "v2")
        slot.messages[-1]["variants"] = [
            {"content": "v1", "ts": "t1"},
            {"content": "v2", "ts": "t2"},
        ]
        slot.messages[-1]["variant_idx"] = 1
        slot.drain()
        captured = []

        async def _capture(*a, **kw):
            captured.extend(list(slot._pending_variants))

        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=_capture):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
                await asyncio.sleep(0)
        assert [v["content"] for v in captured] == ["v1", "v2"]

    @pytest.mark.asyncio
    async def test_regenerate_when_active_is_old_variant_no_dup(self, tmp_path, monkeypatch):
        """If user switched back to v1 then regenerates, v1 should not be appended twice."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "v1")
        slot.messages[-1]["variants"] = [
            {"content": "v1", "ts": "t1"},
            {"content": "v2", "ts": "t2"},
        ]
        slot.messages[-1]["variant_idx"] = 0
        slot.drain()
        captured = []

        async def _capture(*a, **kw):
            captured.extend(list(slot._pending_variants))

        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=_capture):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
                await asyncio.sleep(0)
        assert [v["content"] for v in captured] == ["v1", "v2"]

    @pytest.mark.asyncio
    async def test_regenerate_caps_variants(self, tmp_path, monkeypatch):
        """Variant list is capped; oldest entries drop when over _MAX_VARIANTS."""
        from kiro_crew.dashboard.chat import _MAX_VARIANTS

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "newest")
        existing = [{"content": f"v{i}", "ts": f"t{i}"} for i in range(_MAX_VARIANTS)]
        slot.messages[-1]["variants"] = existing
        slot.messages[-1]["variant_idx"] = len(existing) - 1
        slot.drain()
        captured = []

        async def _capture(*a, **kw):
            captured.extend(list(slot._pending_variants))

        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=_capture):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
                await asyncio.sleep(0)
        assert len(captured) <= _MAX_VARIANTS
        assert captured[-1]["content"] == "newest"

    @pytest.mark.asyncio
    async def test_regenerate_rejects_missing_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/nonexistent/regenerate")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_regenerate_rejects_empty_user_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "")
        slot.append("assistant", "reply")
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/regenerate")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_regenerate_persists_to_disk(self, tmp_path, monkeypatch):
        """After regenerate, on-disk history should reflect the truncation."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "old")
        slot.drain()
        # Save first so a file exists
        from kiro_crew.dashboard.chat import _history_key_for, _save_slot_to_history

        _save_slot_to_history(state, slot)
        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
        # File should now only contain the user message (assistant truncated)
        key = _history_key_for(slot.key)
        persisted = state.conversation_log.read_messages(key)
        roles = [m.get("role") for m in persisted]
        assert roles == ["user"]

    @pytest.mark.asyncio
    async def test_save_slot_redacts_variants(self, tmp_path, monkeypatch):
        """Variants written to disk must have credentials/exfil URLs redacted."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "safe content")
        # Plant a fake credential inside a variant
        slot.messages[-1]["variants"] = [
            {"content": "AKIAIOSFODNN7EXAMPLE secret stuff", "ts": "t1"},
            {"content": "safe content", "ts": "t2"},
        ]
        slot.messages[-1]["variant_idx"] = 1
        slot.drain()
        from kiro_crew.dashboard.chat import _history_key_for, _save_slot_to_history

        _save_slot_to_history(state, slot)
        key = _history_key_for(slot.key)
        persisted = state.conversation_log.read_messages(key)
        ai = [m for m in persisted if m.get("role") == "assistant"][0]
        assert "variants" in ai
        # The AKIA key must not appear in either variant after redaction
        for v in ai["variants"]:
            assert "AKIAIOSFODNN7EXAMPLE" not in v.get("content", "")

    @pytest.mark.asyncio
    async def test_switch_variant_rejects_when_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "v2")
        slot.messages[-1]["variants"] = [
            {"content": "v1", "ts": "t1"},
            {"content": "v2", "ts": "t2"},
        ]

        async def _noop():
            await asyncio.sleep(10)

        slot.task = asyncio.create_task(_noop())
        try:
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/switch-variant", json={"index": 0})
                assert resp.status == 409
        finally:
            slot.task.cancel()

    @pytest.mark.asyncio
    async def test_switch_variant_missing_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/none/switch-variant", json={"index": 0})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_switch_variant_no_variants(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "plain")  # no variants[]
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", json={"index": 0})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_switch_variant_invalid_json_body(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "v1")
        slot.messages[-1]["variants"] = [{"content": "v1"}]
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", data="not-json")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_switch_variant_non_int_index(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "v1")
        slot.messages[-1]["variants"] = [{"content": "v1"}]
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", json={"index": "abc"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_edit_resend_non_dict_body_is_400_not_500(self, tmp_path, monkeypatch):
        # api_chat_slot_edit_resend parsed the body but never checked
        # isinstance(dict); a valid-JSON array/scalar reached body.get("index")
        # and raised AttributeError -> 500. Must be 400, like switch_variant.
        from kiro_crew.dashboard.chat import api_chat_slot_edit_resend

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")
        app = web.Application()
        app["state"] = state
        app.router.add_post("/api/chat/slots/{slot}/edit-resend", api_chat_slot_edit_resend)
        async with TestClient(TestServer(app)) as client:
            for bad in ("[1, 2]", '"hi"', "42"):
                resp = await client.post(
                    "/api/chat/slots/s1/edit-resend",
                    data=bad,
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 400, f"body={bad!r} gave {resp.status}"

    @pytest.mark.asyncio
    async def test_regenerate_clears_pending_on_task_error(self, tmp_path, monkeypatch):
        """If _run_chat raises, _pending_variants must be cleared to prevent leak."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "reply")
        slot.drain()

        async def _boom(*a, **kw):
            raise RuntimeError("llm blew up")

        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=_boom):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
                # Let the failing task propagate through done_callback
                for _ in range(5):
                    await asyncio.sleep(0)
        assert slot._pending_variants == [], "pending variants must be cleared when task errors"

    @pytest.mark.asyncio
    async def test_flush_segment_attaches_pending_variants(self, tmp_path, monkeypatch):
        """_flush_segment should attach _pending_variants to the new assistant message."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        # Simulate pending variants from a regenerate
        slot._pending_variants = [
            {"content": "old v1", "ts": "t1"},
            {"content": "old v2", "ts": "t2"},
        ]
        from kiro_crew.dashboard.chat import _flush_segment

        _flush_segment(state, slot, "new reply", broadcast=False)
        last = slot.messages[-1]
        assert last["role"] == "assistant"
        assert last["content"] == "new reply"
        assert len(last["variants"]) == 3  # old v1, old v2, new reply
        assert last["variant_idx"] == 2
        assert slot._pending_variants == []

    @pytest.mark.asyncio
    async def test_switch_variant_negative_index(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "v1")
        slot.messages[-1]["variants"] = [{"content": "v1"}]
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", json={"index": -1})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_regenerate_only_system_and_assistant(self, tmp_path, monkeypatch):
        """Regenerate should fail if there's no user message (only system + assistant)."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("system", "you are helpful")
        slot.append("assistant", "hello")
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/regenerate")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_flush_segment_no_pending_no_variants(self, tmp_path, monkeypatch):
        """Normal flush without pending variants should not add variants field."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        from kiro_crew.dashboard.chat import _flush_segment

        _flush_segment(state, slot, "reply", broadcast=False)
        last = slot.messages[-1]
        assert "variants" not in last

    @pytest.mark.asyncio
    async def test_switch_variant_missing_index_key(self, tmp_path, monkeypatch):
        """Request body without 'index' key should return 400."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "v1")
        slot.messages[-1]["variants"] = [{"content": "v1"}]
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_restore_preserves_variants(self, tmp_path, monkeypatch):
        """Variants written to disk should be restored via production code path."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "v2")
        slot.messages[-1]["variants"] = [
            {"content": "v1", "ts": "t1"},
            {"content": "v2", "ts": "t2"},
        ]
        slot.messages[-1]["variant_idx"] = 1
        slot.drain()
        from kiro_crew.dashboard.chat import _save_slot_to_history, restore_recent_sessions

        _save_slot_to_history(state, slot)
        # Clear in-memory state and restore via production path
        state._slots.clear()
        restore_recent_sessions(state, window_minutes=9999)
        restored_slot = state._slots.get("s1")
        assert restored_slot is not None
        ai = [m for m in restored_slot.messages if m.get("role") == "assistant"][0]
        assert "variants" in ai
        assert len(ai["variants"]) == 2
        assert ai["variant_idx"] == 1

    @pytest.mark.asyncio
    async def test_regenerate_clears_pending_on_cancel(self, tmp_path, monkeypatch):
        """If user stops a regeneration (cancel), _pending_variants must be cleared."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "reply")
        slot.drain()

        async def _hang(*a, **kw):
            await asyncio.sleep(999)

        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=_hang):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post("/api/chat/slots/s1/regenerate")
                assert resp.status == 200
                assert slot._pending_variants != []
                # Cancel the task (simulates user clicking Stop)
                slot.task.cancel()
                for _ in range(5):
                    await asyncio.sleep(0)
        assert (
            slot._pending_variants == []
        ), "pending variants must be cleared when task is cancelled"

    @pytest.mark.asyncio
    async def test_prepare_messages_redacts_variant_content(self, tmp_path, monkeypatch):
        """Variant content exposed via API must have credentials redacted."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "safe")
        slot.messages[-1]["variants"] = [
            {"content": "AKIAIOSFODNN7EXAMPLE leaked key", "ts": "t1"},
            {"content": "safe", "ts": "t2"},
        ]
        from kiro_crew.dashboard.chat import _prepare_messages

        prepared = _prepare_messages(slot.messages, False, live_child="")
        ai = [m for m in prepared if m.get("role") == "assistant"][0]
        for v in ai["variants"]:
            assert "AKIAIOSFODNN7EXAMPLE" not in v.get("content", "")

    @pytest.mark.asyncio
    async def test_switch_variant_corrupt_entry(self, tmp_path, monkeypatch):
        """If a variant entry is not a dict, switch-variant should return 400."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "v1")
        slot.messages[-1]["variants"] = ["not-a-dict", {"content": "v1"}]
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/switch-variant", json={"index": 0})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_concurrent_regenerate_one_succeeds_one_409(self, tmp_path, monkeypatch):
        """Two simultaneous regenerate requests: one gets 200, the other gets 409."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hi")
        slot.append("assistant", "reply")
        slot.drain()

        async def _hang(*a, **kw):
            await asyncio.sleep(999)

        with patch("kiro_crew.dashboard.chat_regenerate._run_chat", new=_hang):
            async with TestClient(TestServer(_make_app(state))) as client:
                r1, r2 = await asyncio.gather(
                    client.post("/api/chat/slots/s1/regenerate"),
                    client.post("/api/chat/slots/s1/regenerate"),
                )
                statuses = sorted([r1.status, r2.status])
                assert statuses == [200, 409], f"Expected one 200 and one 409, got {statuses}"
        # Cleanup
        if slot.task:
            slot.task.cancel()


class TestForkSlot:
    """Tests for POST /api/chat/slots/{slot}/fork."""

    @pytest.mark.asyncio
    async def test_fork_copies_all_messages(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.title = "My Chat"
        slot._titled = True
        slot.append("user", "hello", "msg msg-u")
        slot.append("assistant", "hi there", "msg msg-a")
        slot.append("user", "how are you", "msg msg-u")
        slot.append("assistant", "good", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True
            assert data["messages"] == 4
            assert data["title"] == "↳ Fork of My Chat"

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        assert new_slot.forked_from == "dashboard:src"
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert len(visible) == 4

    @pytest.mark.asyncio
    async def test_fork_inherits_tags_under_a_fresh_revision(self, tmp_path, monkeypatch):
        """The fork is constructed with an empty tag list under its birth
        revision, then inherits the parent's tags. Those two states must never
        share a `tags_revision`: a slot fetch racing the fork could snapshot the
        empty newborn, and a client seeing that frame under the same revision as
        the inherited list would take it as the next toggle base and erase the
        inherited tags on its first write."""
        from kiro_crew.dashboard.state import _ChatSlot

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.tags = ["tag-a", "tag-b"]
        slot.append("user", "hello", "msg msg-u")
        slot.drain()
        birth_revisions: list[str] = []
        original_init = _ChatSlot.__init__

        def _recording_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            birth_revisions.append(self.tags_revision)

        monkeypatch.setattr(_ChatSlot, "__init__", _recording_init)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot.tags == ["tag-a", "tag-b"]
        assert new_slot.tags is not slot.tags
        assert new_slot.tags_revision not in birth_revisions
        assert new_slot.tags_revision > max(birth_revisions)

    @pytest.mark.asyncio
    async def test_fork_at_message_id(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "msg1", "msg msg-u")
        slot.append("assistant", "reply1", "msg msg-a")
        slot.append("user", "msg2", "msg msg-u")
        slot.append("assistant", "reply2", "msg msg-a")
        slot.drain()
        target_id = slot.messages[1]["meta"]["mid"]

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_id": target_id},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == 2

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert [m["content"] for m in visible] == ["msg1", "reply1"]

    @pytest.mark.asyncio
    async def test_fork_at_message_id_takes_precedence_over_window_index(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "msg1", "msg msg-u")
        slot.append("assistant", "reply1", "msg msg-a")
        slot.append("user", "msg2", "msg msg-u")
        slot.append("assistant", "reply2", "msg msg-a")
        slot.drain()
        target_id = slot.messages[3]["meta"]["mid"]

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_id": target_id, "at_message_index": 0},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == 4

    @pytest.mark.asyncio
    async def test_fork_at_index(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "msg1", "msg msg-u")
        slot.append("assistant", "reply1", "msg msg-a")
        slot.append("user", "msg2", "msg msg-u")
        slot.append("assistant", "reply2", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"at_message_index": 1})
            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == 2

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert len(visible) == 2
        assert visible[-1]["content"] == "reply1"

    @pytest.mark.asyncio
    async def test_fork_aborts_and_keeps_dirty_when_source_save_fails(self, tmp_path):
        # Regression (double-persistence data-loss): the fork persists the dirty
        # source slot before reading it as the source of truth, then clears
        # `_dirty` (which also disables the periodic retry). If that save is
        # best-effort it can silently drop the write under a lock timeout / I/O
        # error, yet `_dirty` would still be cleared — permanently losing the
        # unwritten source messages on the next restart. The fork must persist
        # with best_effort=False and, on failure, abort (503) WITHOUT clearing
        # `_dirty`, so the periodic flush still retries.
        from unittest.mock import AsyncMock, patch

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "unsaved-source-msg", "msg msg-u")
        slot.append("assistant", "unsaved-reply", "msg msg-a")
        slot.drain()
        # Force the save branch: pretend nothing has reached disk yet.
        slot._dirty = True
        slot._resumed_count = 0

        failing_save = AsyncMock(side_effect=RuntimeError("lock timeout"))
        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            with patch("kiro_crew.dashboard.chat_fork.save_slot_off_loop", failing_save):
                resp = await client.post("/api/chat/slots/src/fork", json={})
                assert resp.status == 503

        # best_effort must be explicitly False so the failure propagated.
        assert failing_save.await_count == 1
        assert failing_save.await_args.kwargs.get("best_effort") is False
        # Slot stays dirty → the periodic flush will retry; messages not lost.
        assert slot._dirty is True

    @pytest.mark.asyncio
    async def test_fork_preserves_meta(self, tmp_path):
        # chat_fork.py must preserve the `meta` dict when copying messages into
        # the new slot; dropping it silently breaks every meta-based feature
        # (knowledge chips, paste refs, future inline-comment rewrite badges).
        # Fork must preserve meta verbatim on copied messages.
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append(
            "user",
            "find me a citation",
            "msg msg-u",
            meta={"paste_refs": ["ref-abc123"]},
        )
        slot.append(
            "assistant",
            "Here you go",
            "msg msg-a",
            meta={"knowledge_chips": [{"id": "kb-42", "title": "Cite-X"}]},
        )
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == 2

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert len(visible) == 2

        # User message: meta.paste_refs survived. Compared as a subset, not an
        # exact dict: `_ChatSlot.append` also stamps `meta.mid` (the per-row
        # delivery identity) on every persisted row, so an equality assertion
        # here would be asserting the absence of that id rather than the survival
        # of the caller's keys, which is what this regression is about.
        assert visible[0]["role"] == "user"
        user_meta = visible[0].get("meta") or {}
        assert user_meta.get("paste_refs") == [
            "ref-abc123"
        ], f"Fork dropped user meta. Got: {visible[0].get('meta')!r}"

        # Assistant message: meta.knowledge_chips survived
        assert visible[1]["role"] == "assistant"
        assistant_meta = visible[1].get("meta") or {}
        assert assistant_meta.get("knowledge_chips") == [
            {"id": "kb-42", "title": "Cite-X"}
        ], f"Fork dropped assistant meta. Got: {visible[1].get('meta')!r}"

    @pytest.mark.asyncio
    async def test_fork_handles_messages_without_meta(self, tmp_path):
        # The mirror of test_fork_preserves_meta: fork must not INVENT meta keys
        # the parent row did not have. Since `_ChatSlot.append` now stamps
        # `meta.mid` on every row, "no meta at all" is not the observable
        # invariant; the equivalent one is that the forked row's meta matches the
        # parent's exactly -- nothing added, nothing dropped.
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "plain msg", "msg msg-u")
        slot.append("assistant", "plain reply", "msg msg-a")
        slot.drain()
        parent_meta = [m.get("meta") for m in slot.messages if m["role"] in ("user", "assistant")]

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert len(visible) == len(parent_meta)
        for m, expected in zip(visible, parent_meta):
            assert (
                m.get("meta") == expected
            ), f"Fork changed a message's meta. Got: {m.get('meta')!r}, want: {expected!r}"

    @pytest.mark.asyncio
    async def test_fork_not_found(self, tmp_path):
        state = _make_state(tmp_path)
        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/nope/fork", json={})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_fork_empty_slot(self, tmp_path):
        state = _make_state(tmp_path)
        state.get_or_create_slot("empty")
        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/empty/fork", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_fork_inherits_agent_and_workspace(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src", agent="my-agent", workspace="my-ws")
        slot.model = "custom-model"
        slot.mode = "custom-mode"
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot.agent == "my-agent"
        assert new_slot.workspace == "my-ws"
        assert new_slot.model == "custom-model"
        assert new_slot.mode == "custom-mode"

    @pytest.mark.asyncio
    async def test_fork_of_member_slot_is_not_member_mode(self, tmp_path):
        """A fork of a member DM thread must be an ordinary chat, never a second
        "member" slot: the fork mints a chat-* key, so member mode would make it
        invisible everywhere (excluded from Sessions by surface mode, and absent
        from the roster, whose threads live only on member-<slug> keys)."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.mode = "member"
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        assert new_slot.mode == ""

    @pytest.mark.asyncio
    async def test_fork_mode_override_wins_over_inheritance(self, tmp_path):
        """An allowlisted ``mode`` in the body overrides the source slot's mode.
        The empty string is a legal override, so the override arm is selected on
        ``is not None``, not on truthiness."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.mode = "custom-mode"
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"mode": ""})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        assert new_slot.mode == ""

    @pytest.mark.asyncio
    async def test_fork_with_retired_orchestrator_mode_is_plain_chat(self, tmp_path):
        """Autopilot retired: a fork that still asks for it is coerced, not refused."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"mode": "orchestrator"})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        assert new_slot.mode == ""

    @pytest.mark.asyncio
    async def test_create_with_retired_orchestrator_mode_is_plain_chat(self, tmp_path):
        """Autopilot retired: a create that still asks for it is coerced, not refused."""
        state = _make_state(tmp_path)

        app = _make_app_with_agent_routes(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots", json={"name": "retired-mode", "mode": "orchestrator"}
            )
            assert resp.status == 200

        new_slot = state._slots.get("retired-mode")
        assert new_slot is not None
        assert new_slot.mode == ""

    @pytest.mark.asyncio
    async def test_fork_inherits_folder(self, tmp_path):
        """Fork must land in the same project folder as the source slot."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.folder_id = "proj-abc"
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        assert new_slot.folder_id == "proj-abc"

    @pytest.mark.asyncio
    async def test_fork_inherits_empty_folder(self, tmp_path):
        """Fork of an unfoldered slot stays unfoldered (root)."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot.folder_id == ""

    @pytest.mark.asyncio
    async def test_fork_inherits_project(self, tmp_path):
        """Fork must carry the source slot's active project directory."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.project = str(tmp_path / "repo")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        assert new_slot.project == str(tmp_path / "repo")

    @pytest.mark.asyncio
    async def test_fork_inherits_empty_project(self, tmp_path):
        """Fork of a projectless slot stays projectless."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot.project == ""

    @pytest.mark.asyncio
    async def test_fork_inherits_tags(self, tmp_path):
        """Fork must carry the source slot's assigned tag ids."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.tags = ["tag-a", "tag-b"]
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        assert new_slot.tags == ["tag-a", "tag-b"]
        # Copied, not aliased: mutating the fork's list must not touch the parent.
        new_slot.tags.append("tag-c")
        assert slot.tags == ["tag-a", "tag-b"]

    @pytest.mark.asyncio
    async def test_fork_inherits_empty_tags(self, tmp_path):
        """Fork of an untagged slot stays untagged."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert new_slot.tags == []

    @pytest.mark.asyncio
    async def test_fork_with_prompt(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "context", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"prompt": "fix the bug"},
            )
            data = await resp.json()
            assert data["ok"] is True
            assert data["prompt"] == "fix the bug"
            assert data["messages"] == 1

        # Prompt is returned for frontend to send separately — must NOT be
        # injected into the forked slot server-side.
        new_slot = state._slots.get(data["key"])
        assert all(m["content"] != "fix the bug" for m in new_slot.messages)

    @pytest.mark.asyncio
    async def test_fork_redacts_credentials_in_assistant_messages(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "show me the key", "msg msg-u")
        slot.append("assistant", "Here: AKIAIOSFODNN7EXAMPLE", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()
            assert data["ok"] is True

        new_slot = state._slots.get(data["key"])
        assistant_msgs = [m for m in new_slot.messages if m["role"] == "assistant"]
        assert "AKIAIOSFODNN7EXAMPLE" not in assistant_msgs[0]["content"]
        assert "[REDACTED" in assistant_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_fork_redacts_credentials_in_llm_generated_title(self, tmp_path):
        """Parent title is LLM-generated (via /api/chat/generate-title) and
        flows into the new slot's title + API response + dashboard JSON.
        Must be redacted like any other LLM output (AUTOSDE security-controls).
        """
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.title = "Leaked AKIAIOSFODNN7EXAMPLE key"
        slot._titled = True
        slot.append("user", "hi", "msg msg-u")
        slot.append("assistant", "ok", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()

        assert "AKIAIOSFODNN7EXAMPLE" not in data["title"]
        assert "[REDACTED" in data["title"]
        assert data["title"].startswith("↳ Fork of ")
        new_slot = state._slots.get(data["key"])
        assert "AKIAIOSFODNN7EXAMPLE" not in new_slot.title

    @pytest.mark.asyncio
    async def test_fork_rejects_bool_index(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"at_message_index": True})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_fork_rejects_negative_index(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"at_message_index": -1})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_fork_excludes_system_messages(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("system", "you are helpful", "msg msg-s")
        slot.append("user", "hello", "msg msg-u")
        slot.append("assistant", "hi", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()
            assert data["messages"] == 2

        new_slot = state._slots.get(data["key"])
        roles = [m["role"] for m in new_slot.messages]
        assert "system" not in roles

    @pytest.mark.asyncio
    async def test_fork_persists_to_disk(self, tmp_path):
        """Forked slot (and forked_from metadata) must survive a save/restore cycle."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.append("assistant", "hello", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()
            new_key = data["key"]

        # Simulate a gateway restart by reading messages + metadata from disk
        from kiro_crew.dashboard.chat import _history_key_for

        hk = _history_key_for(new_key)
        meta = state.conversation_log.get_metadata(hk)
        disk_msgs = state.conversation_log.read_messages(hk)
        assert meta.get("forked_from") == "dashboard:src", f"forked_from not persisted; meta={meta}"
        assert len(disk_msgs) == 2, f"forked messages not persisted (got {len(disk_msgs)})"

    @pytest.mark.asyncio
    async def test_fork_rejects_oversized_prompt(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"prompt": "x" * 40_000},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_fork_rejects_out_of_range_index(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.append("assistant", "hello", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"at_message_index": 5})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_fork_succeeds_while_streaming(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.append("assistant", "done reply", "msg msg-a")
        slot.drain()

        # Simulate a running session: task attribute non-None + not done
        class _FakeTask:
            def done(self):
                return False

        slot.task = _FakeTask()  # type: ignore[assignment]
        assert slot.running is True

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == 2

    @pytest.mark.asyncio
    async def test_fork_emits_sel_audit_event(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mock_sel = MagicMock()
        monkeypatch.setattr("kiro_crew.dashboard.chat_fork.sel", lambda: mock_sel)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        mock_sel.log_api_access.assert_called_once()
        kw = mock_sel.log_api_access.call_args[1]
        assert kw["operation"] == "chat.slot_fork"
        assert kw["outcome"] == "allowed"
        assert "from=src" in kw["resources"]
        assert f"to={data['key']}" in kw["resources"]
        # L5 audit enrichment: at_index + prompt_len present
        assert "at_index=last" in kw["resources"]
        assert "prompt_len=0" in kw["resources"]

    @pytest.mark.asyncio
    async def test_fork_of_incognito_parent_inherits_incognito(self, tmp_path):
        """A restricted session forks; the child is born restricted.

        A refusal would buy no privacy -- the parent's transcript is already on
        disk for tab recovery -- and would only cost the user the conversation.
        What must hold is that the child never LOOSENS the mode: it is registered
        restricted at birth, so consolidation and lessons stay closed to the
        copied content.
        """
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src", memory_mode="incognito")
        slot.append("user", "secret", "msg msg-u")
        slot.append("assistant", "kept between us", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        assert data["ok"] is True
        assert data["memory_mode"] == "incognito"
        child = state._slots[data["key"]]
        assert child.memory_mode == "incognito"
        assert child.is_restricted is True
        assert child.blocks_reads is False
        assert f"dashboard:{data['key']}" in state._restricted_keys
        assert [m["content"] for m in child.messages] == ["secret", "kept between us"]

    @pytest.mark.asyncio
    async def test_fork_of_temporary_parent_stays_temporary_with_its_history(self, tmp_path):
        """Temporary blocks memory READS, not the session's own thread history.

        Cold provider replay uses the child's live message window, and the
        copied turns also reach the child's own conversation log -- a temporary
        fork is a chat the user reopens from History like any other -- under a
        metadata line that names the temporary mode, so nothing learns from it.
        """
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src", memory_mode="temporary")
        slot.append("user", "parent question", "msg msg-u")
        slot.append("assistant", "parent answer", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        child = state._slots[data["key"]]
        assert child.memory_mode == "temporary"
        assert child.blocks_reads is True
        assert f"dashboard:{data['key']}" in state._restricted_keys

        from kiro_crew.context import build_session_replay
        from kiro_crew.dashboard.chat import _history_key_for

        history_key = _history_key_for(data["key"])
        visible = [m["content"] for m in child.messages if m.get("role") in ("user", "assistant")]
        assert visible == ["parent question", "parent answer"]
        replay = await asyncio.to_thread(
            build_session_replay,
            state.conversation_log,
            history_key,
            pending_messages=list(child.messages),
        )
        assert "parent question" in replay
        assert "parent answer" in replay
        recent = state.conversation_log.recent(history_key)
        assert [m["content"] for m in recent if m.get("role") in ("user", "assistant")] == [
            "parent question",
            "parent answer",
        ]
        assert state.conversation_log.get_metadata(history_key).get("memory_mode") == "temporary"

    @pytest.mark.asyncio
    async def test_fork_history_visible_to_new_kiro_via_context_builder(self, tmp_path):
        """Forked JSONL is the source build_session_context reads for the new slot.

        Guarantees the fresh kiro-cli process in the forked tab receives the
        copied user/assistant turns as thread-history context.
        """
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "parent question", "msg msg-u")
        slot.append("assistant", "parent answer", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()
            new_key = data["key"]

        # conversation_log.recent(forked_key) is what ContextBuilder.build_session_context
        # calls to assemble the thread-history section for the new kiro process.
        from kiro_crew.dashboard.chat import _history_key_for

        recent = state.conversation_log.recent(_history_key_for(new_key))
        visible = [m for m in recent if m.get("role") in ("user", "assistant")]
        assert [m["content"] for m in visible] == [
            "parent question",
            "parent answer",
        ], f"fork history not readable as new-session context: {visible}"

    @pytest.mark.asyncio
    async def test_fork_does_not_clone_parent_kiro_session_id(self, tmp_path, monkeypatch):
        """Parent's kiro-cli session id (session_map sid) must NOT carry to fork.

        Cloning the sid would make both tabs share one kiro process state and
        corrupt each other's view. Fork creates a FRESH kiro session on first
        prompt by leaving session_map unset for the new key.
        """
        from kiro_crew.session import SessionMap

        monkeypatch.setattr("kiro_crew.session_map.config_dir", lambda: tmp_path)
        session_map = SessionMap()
        session_map.set("dashboard:src", "parent-kiro-sid-abc123")
        # A loop-side mutation defers its disk write; the fresh-instance
        # readback below needs the file current NOW.
        session_map.flush()

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            data = await resp.json()
            new_key = data["key"]

        # Re-read from disk so we're not trusting an in-process cache.
        # Inspect _data directly to skip SessionMap.get()'s kiro-session file
        # existence check (we don't spawn real kiro processes in unit tests).
        reloaded = SessionMap()
        assert (
            reloaded._data.get("dashboard:src", {}).get("sid") == "parent-kiro-sid-abc123"
        ), "parent's kiro sid should survive fork unchanged"
        assert (
            f"dashboard:{new_key}" not in reloaded._data
        ), "forked slot must NOT inherit parent's kiro sid"

    @pytest.mark.asyncio
    async def test_fork_of_fork_chains_forked_from(self, tmp_path):
        """M10: fork of a fork titles correctly and `forked_from` points to intermediate, not root."""
        state = _make_state(tmp_path)
        root = state.get_or_create_slot("root")
        root.title = "Original"
        root._titled = True
        root.append("user", "q1", "msg msg-u")
        root.append("assistant", "a1", "msg msg-a")
        root.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            r1 = await client.post("/api/chat/slots/root/fork", json={})
            d1 = await r1.json()
            mid_key = d1["key"]
            assert d1["title"] == "↳ Fork of Original"

            mid = state._slots.get(mid_key)
            mid.append("user", "q2", "msg msg-u")
            mid.append("assistant", "a2", "msg msg-a")
            mid.drain()

            r2 = await client.post(f"/api/chat/slots/{mid_key}/fork", json={})
            d2 = await r2.json()

        leaf = state._slots.get(d2["key"])
        assert d2["title"] == "↳ Fork of Fork of Original"
        assert (
            leaf.forked_from == f"dashboard:{mid_key}"
        ), f"leaf forked_from should point to intermediate, got {leaf.forked_from}"
        assert leaf.forked_from != "dashboard:root"
        visible = [m for m in leaf.messages if m["role"] in ("user", "assistant")]
        assert [m["content"] for m in visible] == ["q1", "a1", "q2", "a2"]

    @pytest.mark.asyncio
    async def test_fork_reads_full_history_from_disk_when_memory_capped(self, tmp_path):
        """M12: when in-memory snapshot is smaller than full history, fork reads from disk."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        for i in range(250):
            slot.append("user" if i % 2 == 0 else "assistant", f"m{i}", "msg")
        slot.drain()
        from kiro_crew.dashboard.chat import _save_slot_to_history

        _save_slot_to_history(state, slot)
        # Simulate restore cap: keep only last 50 in memory.
        # Clear _dirty so the endpoint's flush-if-dirty path doesn't overwrite disk.
        slot.messages = slot.messages[-50:]
        slot._dirty = False

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        assert (
            data["messages"] == 250
        ), f"fork should read full history from disk, got {data['messages']}"
        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert len(visible) == 250
        assert visible[0]["content"] == "m0"
        assert visible[-1]["content"] == "m249"

    @pytest.mark.asyncio
    async def test_fork_preserves_full_history_when_dirty_and_capped(self, tmp_path):
        """A1 regression: _dirty=True + capped in-memory must NOT truncate disk history."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        for i in range(250):
            slot.append("user" if i % 2 == 0 else "assistant", f"m{i}", "msg")
        slot.drain()
        from kiro_crew.dashboard.chat import _save_slot_to_history

        _save_slot_to_history(state, slot)
        # Simulate restore with cap: real path caps messages then sets
        # _resumed_count to the capped length. User then sends new messages.
        slot.messages = slot.messages[-50:]
        slot._resumed_count = len(slot.messages)
        slot.append("user", "new1", "msg")
        slot.append("assistant", "new2", "msg")
        slot.drain()
        assert slot._dirty is True

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        # Full 250 on disk + 2 new dirty messages = 252 total.
        assert (
            data["messages"] == 252
        ), f"fork must preserve full disk history + dirty tail, got {data['messages']}"
        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert visible[0]["content"] == "m0"
        assert visible[-2]["content"] == "new1"
        assert visible[-1]["content"] == "new2"

    @pytest.mark.asyncio
    async def test_fork_concurrent_requests_both_succeed(self, tmp_path):
        """R2-7: two rapid fork requests on the same slot both return 200 with
        identical visible-message counts. Each fork produces an independent new
        slot; no messages lost or duplicated."""
        import asyncio

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "q1", "msg msg-u")
        slot.append("assistant", "a1", "msg msg-a")
        slot.append("user", "q2", "msg msg-u")
        slot.append("assistant", "a2", "msg msg-a")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            r1, r2 = await asyncio.gather(
                client.post("/api/chat/slots/src/fork", json={}),
                client.post("/api/chat/slots/src/fork", json={}),
            )
            assert r1.status == 200 and r2.status == 200
            d1, d2 = await r1.json(), await r2.json()

        assert d1["key"] != d2["key"], "concurrent forks must produce distinct slot keys"
        assert (
            d1["messages"] == d2["messages"] == 4
        ), f"both forks must copy all 4 visible messages, got {d1['messages']}/{d2['messages']}"
        for key in (d1["key"], d2["key"]):
            new_slot = state._slots.get(key)
            visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
            assert [m["content"] for m in visible] == ["q1", "a1", "q2", "a2"]

    @pytest.mark.asyncio
    async def test_fork_audit_records_inherited_memory_mode(self, tmp_path, monkeypatch):
        """The SEL event for a restricted fork is ``allowed`` and names the mode.

        A restricted parent is not a denial ground, but the boundary the child
        inherited must still be visible in the audit log.
        """
        from unittest.mock import MagicMock

        mock_sel = MagicMock()
        monkeypatch.setattr("kiro_crew.dashboard.chat_fork.sel", lambda: mock_sel)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src", memory_mode="incognito")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200

        calls = [c[1] for c in mock_sel.log_api_access.call_args_list]
        assert calls, "fork emitted no SEL event"
        assert all(kw["operation"] == "chat.slot_fork" for kw in calls)
        assert not [kw for kw in calls if kw["outcome"] == "denied"]
        allowed = [kw for kw in calls if kw["outcome"] == "allowed"]
        assert len(allowed) == 1
        assert "memory_mode=incognito" in allowed[0]["resources"]

    @pytest.mark.asyncio
    async def test_fork_app_isolation_rejects_cross_app(self, tmp_path, monkeypatch):
        """M-2 regression: app A cannot fork a slot owned by app B."""
        from unittest.mock import MagicMock

        mock_sel = MagicMock()
        # The refusal is the per-slot checkpoint's, so its row is written there.
        monkeypatch.setattr("kiro_crew.dashboard.slot_ownership.sel", lambda: mock_sel)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src", app="app-B")
        slot.append("user", "secret", "msg msg-u")
        slot.drain()

        # aiohttp middleware populates request["app"]; test injects via middleware.
        @web.middleware
        async def inject_app(request, handler):
            request["app"] = "app-A"
            return await handler(request)

        app_obj = _make_app(state)
        app_obj.middlewares.insert(0, inject_app)

        async with TestClient(TestServer(app_obj)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            # Cross-app access now returns 404 (indistinguishable from a missing
            # slot) to prevent slot enumeration — the true reason is still
            # recorded server-side via SEL (asserted below).
            assert resp.status == 404
            data = await resp.json()
            assert data["error"] == "not found"

        # denied event logged
        denied_calls = [
            c for c in mock_sel.log_api_access.call_args_list if c[1].get("outcome") == "denied"
        ]
        assert len(denied_calls) == 1
        assert denied_calls[0][1]["source"] == "app_isolation"

    @pytest.mark.asyncio
    async def test_fork_inherits_app_ownership(self, tmp_path):
        """I-1 regression: new_slot._app is the requesting app (or empty for dashboard)."""
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src", app="app-X")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()

        @web.middleware
        async def inject_app(request, handler):
            request["app"] = "app-X"
            return await handler(request)

        app_obj = _make_app(state)
        app_obj.middlewares.insert(0, inject_app)

        async with TestClient(TestServer(app_obj)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        assert (
            new_slot._app == "app-X"
        ), f"forked slot must inherit caller's app, got {new_slot._app!r}"

    @pytest.mark.asyncio
    async def test_fork_rejects_when_slot_cap_reached(self, tmp_path, monkeypatch):
        """Review finding: fork must return 429 + denied audit when slot cap hit."""
        from unittest.mock import MagicMock

        mock_sel = MagicMock()
        monkeypatch.setattr("kiro_crew.dashboard.chat_fork.sel", lambda: mock_sel)
        # Lower the cap so we don't need to create hundreds of slots.
        monkeypatch.setattr("kiro_crew.dashboard.chat_fork.MAX_LIVE_SLOTS", 3)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("src")
        slot.append("user", "hi", "msg msg-u")
        slot.drain()
        # Pre-populate to hit the cap (src + 2 dummies = 3).
        state.get_or_create_slot("dummy1")
        state.get_or_create_slot("dummy2")
        assert len(state._slots) == 3

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={})
            assert resp.status == 429
            data = await resp.json()
            assert "cap" in data["error"].lower()

        denied = [
            c for c in mock_sel.log_api_access.call_args_list if c[1].get("outcome") == "denied"
        ]
        assert len(denied) == 1
        assert denied[0][1]["source"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_fork_at_index_spans_chained_session_files(self, tmp_path):
        """Index space must match the frontend (chained), not the current file alone.

        The slot detail endpoint returns
        ``read_messages_chained`` (all sibling session files sharing the
        slot's ``tab_id``), and the frontend builds its fork-button index
        against that list. Pre-fix, fork called ``read_messages`` and
        rejected any index past the current file's boundary.
        """
        from kiro_crew.dashboard.chat_utils import _history_key_for

        state = _make_state(tmp_path)
        tab_id = "tab12345abcd"

        # Older sibling session file with same tab_id (4 visible messages).
        # The chained-read glob matches ``dashboard_chat-*.jsonl`` so the key
        # must start with ``chat-`` to participate in chaining. The chained
        # walker uses ``sorted(glob)`` so file names must lexicographically
        # match chronological order — production uses ``chat-N-<ts>`` which
        # naturally sorts; mirror that with explicit ordering here.
        older_key = "dashboard:chat-tab-1-old"
        state.conversation_log.append(older_key, "user", "old-q1", tab_id=tab_id)
        state.conversation_log.append(older_key, "assistant", "old-a1")
        state.conversation_log.append(older_key, "user", "old-q2")
        state.conversation_log.append(older_key, "assistant", "old-a2")

        # Current session file with same tab_id (2 visible messages).
        current_key = "dashboard:chat-tab-2-new"
        state.conversation_log.append(current_key, "user", "new-q1", tab_id=tab_id)
        state.conversation_log.append(current_key, "assistant", "new-a1")
        state.conversation_log.invalidate_tab_id_cache()

        # In-memory slot mirrors the current file's persisted view; mark
        # clean so the fork handler relies on the chained read alone.
        slot = state.get_or_create_slot("chat-tab-2-new")
        slot._tab_id = tab_id
        slot.append("user", "new-q1", "msg msg-u")
        slot.append("assistant", "new-a1", "msg msg-a")
        slot.drain()
        slot._resumed_count = len(slot.messages)
        slot._dirty = False

        # Frontend visibleIndexMap assigns index 5 to the last "new-a1"
        # (chained list has 6 user/assistant entries: 4 older + 2 new).
        assert _history_key_for("chat-tab-2-new") == current_key
        chained = state.conversation_log.read_messages_chained(current_key)
        assert len([m for m in chained if m.get("role") in ("user", "assistant")]) == 6

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/chat-tab-2-new/fork",
                json={"at_message_index": 5},
            )
            assert resp.status == 200, await resp.text()
            data = await resp.json()
            assert data["ok"] is True
            assert data["messages"] == 6  # full chained history up to index 5

        new_slot = state._slots.get(data["key"])
        assert new_slot is not None
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert [m["content"] for m in visible] == [
            "old-q1",
            "old-a1",
            "old-q2",
            "old-a2",
            "new-q1",
            "new-a1",
        ]


# ── Color theme & persona injection tests ──


class TestColorTheme:
    """Tests for color_theme validation and slot assignment. Only "" and
    ``custom-<slug>`` (installed packs) are valid; any built-in visual-theme
    slug or junk value is coerced to "" (no persona path)."""

    @pytest.mark.asyncio
    async def test_color_theme_set_on_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        with patch("kiro_crew.dashboard.chat_handlers._run_chat", new=AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hi", "slot": "theme-slot", "color_theme": "custom-mypack"},
                )
                assert resp.status == 200
                assert state._slots["theme-slot"].color_theme == "custom-mypack"

    @pytest.mark.asyncio
    async def test_color_theme_cleared_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("theme-slot")
        slot.color_theme = "custom-mypack"
        with patch("kiro_crew.dashboard.chat_handlers._run_chat", new=AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hi", "slot": "theme-slot", "color_theme": ""},
                )
                assert resp.status == 200
                assert slot.color_theme == ""

    @pytest.mark.asyncio
    async def test_color_theme_not_cleared_when_absent(self, tmp_path, monkeypatch):
        """Omitting color_theme from body must not reset an existing theme."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("theme-slot")
        slot.color_theme = "custom-mypack"
        with patch("kiro_crew.dashboard.chat_handlers._run_chat", new=AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hi", "slot": "theme-slot"},
                )
                assert resp.status == 200
                assert slot.color_theme == "custom-mypack"

    @pytest.mark.asyncio
    async def test_invalid_color_theme_coerced_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        with patch("kiro_crew.dashboard.chat_handlers._run_chat", new=AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hi", "slot": "theme-slot", "color_theme": "evil"},
                )
                assert resp.status == 200
                assert state._slots["theme-slot"].color_theme == ""

    @pytest.mark.asyncio
    async def test_non_string_color_theme_coerced(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        with patch("kiro_crew.dashboard.chat_handlers._run_chat", new=AsyncMock()):
            async with TestClient(TestServer(_make_app(state))) as client:
                resp = await client.post(
                    "/api/chat?ws=1",
                    json={"message": "hi", "slot": "theme-slot", "color_theme": 42},
                )
                assert resp.status == 200
                assert state._slots["theme-slot"].color_theme == ""


class TestInstalledPackConsentInjection:
    """Content-bound (sha256) consent gate for INSTALLED pack personas
    (``custom-<slug>``). Injection requires the caller's ``theme_consent_sha``
    to equal sha256 of the persona text actually read from disk; anything else
    fails closed. Guards the Codex HIGH reinstall-swap fix."""

    PERSONA = "Speak like a friendly installed-pack host."

    @staticmethod
    def _sha(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_matching_sha_injects(self):
        from kiro_crew.dashboard.chat import _maybe_inject_persona

        with patch(
            "kiro_crew.dashboard.chat_utils._installed_theme_persona",
            return_value=self.PERSONA,
        ):
            result = _maybe_inject_persona(
                "hello",
                "custom-mypack",
                True,
                theme_consent_sha=self._sha(self.PERSONA),
            )
        assert "[THEME PERSONA]" in result
        assert self.PERSONA in result

    def test_stale_sha_not_injected(self):
        # Reinstall rewrote persona.md; the stored hash does not match the
        # on-disk text -> the new, never-consented persona must NOT be injected.
        from kiro_crew.dashboard.chat import _maybe_inject_persona

        with patch(
            "kiro_crew.dashboard.chat_utils._installed_theme_persona",
            return_value=self.PERSONA,
        ):
            result = _maybe_inject_persona(
                "hello",
                "custom-mypack",
                True,
                theme_consent_sha=self._sha("OLD PERSONA THE USER CONSENTED TO"),
            )
        assert result == "hello"

    def test_absent_sha_not_injected(self):
        from kiro_crew.dashboard.chat import _maybe_inject_persona

        with patch(
            "kiro_crew.dashboard.chat_utils._installed_theme_persona",
            return_value=self.PERSONA,
        ):
            result = _maybe_inject_persona("hello", "custom-mypack", True)
        assert result == "hello"

    def test_legacy_boolean_alone_not_injected(self):
        # The legacy boolean consent field grants nothing on its own: with no
        # content-bound sha, an installed pack persona is never injected even
        # though the pack ships one.
        from kiro_crew.dashboard.chat import _maybe_inject_persona

        with patch(
            "kiro_crew.dashboard.chat_utils._installed_theme_persona",
            return_value=self.PERSONA,
        ):
            result = _maybe_inject_persona(
                "hello",
                "custom-mypack",
                True,
                theme_consent_sha=None,
            )
        assert result == "hello"

    def test_not_injected_on_followup(self):
        from kiro_crew.dashboard.chat import _maybe_inject_persona

        with patch(
            "kiro_crew.dashboard.chat_utils._installed_theme_persona",
            return_value=self.PERSONA,
        ):
            result = _maybe_inject_persona(
                "hello",
                "custom-mypack",
                False,
                theme_consent_sha=self._sha(self.PERSONA),
            )
        assert result == "hello"

    def test_malformed_sha_no_crash_no_injection(self):
        # GPT HIGH: a non-ASCII (or otherwise malformed) theme_consent_sha
        # must never reach hmac.compare_digest (which raises TypeError on
        # non-ASCII str, aborting the whole chat turn). The compare-site guard
        # treats anything that is not exactly 64 lowercase-hex as ABSENT:
        # no exception AND no injection.
        from kiro_crew.dashboard.chat import _maybe_inject_persona

        valid = self._sha(self.PERSONA)
        malformed = [
            "é",  # non-ASCII -> would TypeError in compare_digest
            valid.upper(),  # uppercase hex (raw, un-normalized) -> not 64-lower
            valid[:-1],  # 63 chars
            valid + "a",  # 65 chars
            "",  # empty
            "  ",  # whitespace only
            12345,  # non-str
            None,  # absent
            valid[:-2] + "gg",  # non-hex chars
        ]
        with patch(
            "kiro_crew.dashboard.chat_utils._installed_theme_persona",
            return_value=self.PERSONA,
        ):
            for bad in malformed:
                # The call must not raise for any malformed input...
                result = _maybe_inject_persona(
                    "hello",
                    "custom-mypack",
                    True,
                    theme_consent_sha=bad,
                )
                # ...and must not inject the persona.
                assert result == "hello", f"unexpected injection for {bad!r}"

    def test_normalizer_fail_closed_and_salvage(self):
        # The parse-site normalizer (validation.normalize_theme_consent_sha)
        # rejects malformed values (-> None, fail closed) and salvages a valid
        # sha wrapped in surrounding whitespace / uppercase (strip + lower).
        from kiro_crew.validation import normalize_theme_consent_sha

        valid = self._sha(self.PERSONA)
        assert normalize_theme_consent_sha("é") is None
        assert normalize_theme_consent_sha(valid[:-1]) is None
        assert normalize_theme_consent_sha("") is None
        assert normalize_theme_consent_sha(12345) is None
        assert normalize_theme_consent_sha(None) is None
        assert normalize_theme_consent_sha(valid) == valid
        # salvage: leading/trailing whitespace + uppercase normalize to canonical
        assert normalize_theme_consent_sha("  " + valid.upper() + "\n") == valid
        # a normalized value then injects through the real gate
        from kiro_crew.dashboard.chat import _maybe_inject_persona

        with patch(
            "kiro_crew.dashboard.chat_utils._installed_theme_persona",
            return_value=self.PERSONA,
        ):
            norm = normalize_theme_consent_sha("  " + valid.upper() + "\n")
            result = _maybe_inject_persona(
                "hello",
                "custom-mypack",
                True,
                theme_consent_sha=norm,
            )
        assert "[THEME PERSONA]" in result


class TestStopReasonCancelled:
    """Phase 4: handler response to stopReason='cancelled'."""

    @staticmethod
    def _make_mock_client(events):
        client = _provider_mock()
        client.context_usage_pct = MagicMock(return_value=10.0)

        async def _stream(msg):
            for ev in events:
                yield ev

        client.stream = _stream
        client.stream_command = _stream
        return client

    @staticmethod
    def _make_state_for_run_chat(tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.context_builder = None
        state.consolidator = MagicMock()
        state._hook_store = None
        state._yolo = False
        return state

    @pytest.mark.asyncio
    async def test_handler_stop_reason_cancelled_skips_record_success(self, tmp_path, monkeypatch):
        """When EVENT_COMPLETE carries stop_reason='cancelled', neither
        record_success nor record_failure should be called."""
        from kiro_crew.acp.types import STOP_REASON_CANCELLED
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial"),
            LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        state.sessions.record_success = MagicMock()
        state.sessions.record_failure = AsyncMock()

        await _run_chat(state, slot, "hello")

        state.sessions.record_success.assert_not_called()
        state.sessions.record_failure.assert_not_called()

    @staticmethod
    def _wire_reinjection(state, tmp_path, monkeypatch):
        """Give the state a context builder so the consume site actually runs.

        The class's default harness sets `context_builder = None`, which skips
        the leg that consumes the flag -- with no consume there is nothing to
        restore, so a test without this would pass vacuously.
        """
        from kiro_crew.context import ContextBuilder
        from kiro_crew.memory import MemoryStore
        from kiro_crew.skills import SkillsLoader

        ctx_builder = ContextBuilder(
            memory=MemoryStore(workspace=tmp_path / "ws"),
            skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        )
        monkeypatch.setattr(
            ctx_builder,
            "build_message",
            lambda text, *a, **kw: (text, MagicMock(action=None, text="")),
        )
        state.context_builder = ctx_builder
        state.sessions.consume_needs_reinjection = MagicMock(return_value=True)
        state.sessions.mark_needs_reinjection = MagicMock()
        return state

    @pytest.mark.asyncio
    async def test_cancelled_turn_restores_the_reinjection_flag(self, tmp_path, monkeypatch):
        """A graceful cancel discards the prompt, so the one-shot
        post-compaction re-injection flag must be put back.

        Regression guard for the path a narrower fix missed: restoring only on
        the empty-response re-queue left a soft-stop losing the skills index for
        the rest of the session.
        """
        from kiro_crew.acp.types import STOP_REASON_CANCELLED
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial"),
            LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        self._wire_reinjection(state, tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        await _run_chat(state, slot, "hello")

        state.sessions.consume_needs_reinjection.assert_called()
        state.sessions.mark_needs_reinjection.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_mid_turn_restores_the_reinjection_flag(self, tmp_path, monkeypatch):
        """An exception between the consume and the success check must not
        swallow the flag either -- the restore lives in the `finally`, so every
        non-landing exit is covered, not only the ones that reach the end."""
        from kiro_crew.dashboard.chat import _run_chat

        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        self._wire_reinjection(state, tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")

        client = MagicMock()

        def _boom(_msg):
            raise RuntimeError("provider exploded mid-turn")

        client.stream = _boom
        client.context_usage_pct = MagicMock(return_value=0.0)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        state.sessions.record_failure = AsyncMock()

        await _run_chat(state, slot, "hello")

        state.sessions.mark_needs_reinjection.assert_called_once()

    @pytest.mark.asyncio
    async def test_landed_turn_does_not_restore_the_reinjection_flag(self, tmp_path, monkeypatch):
        """The complement: a turn that lands must leave the flag cleared,
        otherwise the index is re-paid on every subsequent turn."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="a real answer"),
            LLMEvent(kind=EVENT_COMPLETE),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        self._wire_reinjection(state, tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        state.sessions.record_success = MagicMock()

        await _run_chat(state, slot, "hello")

        state.sessions.mark_needs_reinjection.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_stop_reason_cancelled_skips_consolidation(self, tmp_path, monkeypatch):
        """When cancelled, maybe_consolidate must not be called."""
        from kiro_crew.acp.types import STOP_REASON_CANCELLED
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial"),
            LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        await _run_chat(state, slot, "hello")

        state.consolidator.maybe_consolidate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_stop_reason_end_turn_preserves_existing_behavior(
        self, tmp_path, monkeypatch
    ):
        """When stop_reason='end_turn', record_success and maybe_consolidate fire."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="done"),
            LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        state.sessions.record_success = MagicMock()

        await _run_chat(state, slot, "hello")

        state.sessions.record_success.assert_called_once()
        state.consolidator.maybe_consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_wake_turn_does_not_write_user_memory(self, tmp_path, monkeypatch):
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.dashboard.state import MONITOR_WAKE_PREFIX
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="monitor action complete"),
            LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        state.sessions.record_success = MagicMock()

        await _run_chat(state, slot, f"{MONITOR_WAKE_PREFIX}\nChecks failed.")

        state.sessions.record_success.assert_called_once()
        state.consolidator.maybe_consolidate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_stop_reason_cancelled_flushes_partial_text(self, tmp_path, monkeypatch):
        """Partial text chunks before cancel must be flushed to the slot."""
        from kiro_crew.acp.types import STOP_REASON_CANCELLED
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events = [
            LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial output here"),
            LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED),
        ]
        state = self._make_state_for_run_chat(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        client = self._make_mock_client(events)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        await _run_chat(state, slot, "hello")

        assistant_msgs = [m for m in slot.messages if m.get("role") == "assistant"]
        assert any("partial output here" in m["content"] for m in assistant_msgs)


# ── Phase 5: Soft-stop dashboard backend tests ──


class TestStopTurnSlotState:
    """Tests for api_chat_slot_stop soft/hard state transitions."""

    def _make_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        sessions = MagicMock(count=0)
        sessions.stop_turn = AsyncMock(return_value="soft")
        sessions.reset = AsyncMock()
        sessions.get_pid = MagicMock(return_value=None)
        state = DashboardState(
            sessions=sessions,
            crons=MagicMock(
                list_jobs=MagicMock(return_value=[]),
                status=MagicMock(return_value={}),
            ),
            lessons=MagicMock(load_all=MagicMock(return_value=[])),
            start_time=0.0,
            conversation_log=ConversationLog(base_dir=tmp_path),
        )
        state.kiro_prerequisite_service = _make_ready_kiro_prerequisite()
        return state

    @pytest.mark.asyncio
    async def test_stop_turn_slot_state_transitions_soft(self, tmp_path, monkeypatch):
        """POST stop → idle→soft_pending; after on_soft → idle."""
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))

        captured_states: list[str] = []

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            captured_states.append(slot._stop_state)
            if on_soft:
                await on_soft()
            return "soft"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200

        assert captured_states == ["soft_pending"]
        assert slot._stop_state == "idle"
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_stop_turn_slot_state_transitions_hard(self, tmp_path, monkeypatch):
        """POST stop with hard outcome → idle→soft_pending→idle after on_hard."""
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            if on_hard:
                await on_hard()
            return "hard"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200

        assert slot._stop_state == "idle"
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_stop_turn_force_query_param(self, tmp_path, monkeypatch):
        """POST stop?force=true when soft_pending → skips cancel, hard kill."""
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))
        slot._stop_state = "soft_pending"

        force_called = []

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            force_called.append(force)
            if on_hard:
                await on_hard()
            return "hard"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop?force=true")
            assert resp.status == 200

        assert force_called == [True]
        assert slot._stop_state == "idle"
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_stop_turn_second_press_escalates_without_force_flag(self, tmp_path, monkeypatch):
        """A second stop press while soft_pending hard-kills even when the
        client did NOT send force=true. The client derives force from the
        WS-echoed stop_state, which lags on a slow connection; the backend's
        own soft_pending state is authoritative, so any second press escalates.
        """
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))
        slot._stop_state = "soft_pending"

        force_called = []

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            force_called.append(force)
            if on_hard:
                await on_hard()
            return "hard"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            # No ?force=true — the lagging client still thinks state is idle.
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200

        # Backend escalated to a hard kill anyway.
        assert force_called == [True]
        assert slot._stop_state == "idle"
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_stop_turn_first_press_preserves_queue(self, tmp_path, monkeypatch):
        """Queue populated; POST stop; queue preserved for dequeue loop."""
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))
        slot._queue.extend(["msg1", "msg2"])

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            assert preserve_queue is True
            if on_soft:
                await on_soft()
            return "soft"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200
        assert len(slot._queue) == 2
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_double_press_force_kill_clears_queue(self, tmp_path, monkeypatch):
        """Double-press (force kill) clears slot._queue so no dequeue fires."""
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))
        slot._stop_state = "soft_pending"
        for content in ("msg1", "msg2", "msg3"):
            slot.queue_append(content)

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            if on_hard:
                await on_hard()
            return "hard"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200
        assert len(slot._queue) == 0
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_single_press_empty_queue_goes_idle(self, tmp_path, monkeypatch):
        """Single press with no queued messages goes idle without error."""
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))
        assert len(slot._queue) == 0

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            assert preserve_queue is True
            if on_soft:
                await on_soft()
            return "soft"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200
        assert len(slot._queue) == 0
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_double_press_empty_queue_no_crash(self, tmp_path, monkeypatch):
        """Double press with empty queue clears without error."""
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))
        slot._stop_state = "soft_pending"
        assert len(slot._queue) == 0

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            if on_hard:
                await on_hard()
            return "hard"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200
        assert len(slot._queue) == 0
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_stop_event_appears_in_transcript(self, tmp_path, monkeypatch):
        """After stop, slot messages contain a stop_event entry."""
        import json

        def _is_stop_event(m: dict) -> bool:
            cls = m.get("cls", "")
            if not isinstance(cls, str) or not cls.startswith("{"):
                return False
            try:
                return json.loads(cls).get("kind") == "stop_event"
            except (json.JSONDecodeError, TypeError):
                return False

        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            if on_soft:
                await on_soft()
            return "soft"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/stop")
            assert resp.status == 200

        stop_msgs = [m for m in slot.messages if _is_stop_event(m)]
        assert len(stop_msgs) == 1
        data = json.loads(stop_msgs[0]["content"])
        assert data["kind"] == "stop_event"
        assert data["state"] == "stopped"
        assert data["outcome"] == "soft"
        slot.task.cancel()

    @pytest.mark.asyncio
    async def test_stop_event_replace_in_place(self, tmp_path, monkeypatch):
        """Stop event has stable id across state transitions (one entry)."""
        import json

        def _is_stop_event(m: dict) -> bool:
            cls = m.get("cls", "")
            if not isinstance(cls, str) or not cls.startswith("{"):
                return False
            try:
                return json.loads(cls).get("kind") == "stop_event"
            except (json.JSONDecodeError, TypeError):
                return False

        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.task = asyncio.ensure_future(asyncio.sleep(999))

        async def fake_stop_turn(
            key, *, force=False, preserve_queue=False, on_soft=None, on_hard=None
        ):
            # Verify the stop_event was inserted before callbacks
            stop_msgs = [m for m in slot.messages if _is_stop_event(m)]
            assert len(stop_msgs) == 1
            pre_data = json.loads(stop_msgs[0]["content"])
            assert pre_data["state"] == "stopping"
            if on_soft:
                await on_soft()
            return "soft"

        state.sessions.stop_turn = fake_stop_turn

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/chat/slots/s1/stop")

        # Still only one stop_event message
        stop_msgs = [m for m in slot.messages if _is_stop_event(m)]
        assert len(stop_msgs) == 1
        data = json.loads(stop_msgs[0]["content"])
        assert data["state"] == "stopped"
        slot.task.cancel()


class TestStopHistoryBanner:
    """Tests for history re-injection banner skip on soft stop."""

    @staticmethod
    def _last_stop_soft(slot: _ChatSlot) -> bool:
        """Replicates the detection logic in chat.py:_run_chat."""
        import json

        for m in reversed(slot.messages):
            cls_val = m.get("cls", "")
            if not isinstance(cls_val, str) or not cls_val.startswith("{"):
                continue
            try:
                _cls = json.loads(cls_val)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(_cls, dict) or _cls.get("kind") != "stop_event":
                continue
            return _cls.get("outcome") == "soft"
        return False

    def test_soft_stop_preserves_session_no_history_banner(self):
        """After a soft stop, _build_history_prefix is skipped."""
        import json

        slot = _ChatSlot("s1")
        slot.append("user", "hello")
        slot.append("assistant", "hi there")
        # cls must be a JSON-encoded dict (same format api_chat_slot_stop uses)
        cls_json = json.dumps(
            {
                "kind": "stop_event",
                "id": "stop-abc",
                "state": "stopped",
                "outcome": "soft",
            }
        )
        slot.append("system", cls_json, cls_json)
        assert self._last_stop_soft(slot) is True

    def test_hard_stop_still_injects_history_banner(self):
        """After a hard stop, the banner detection returns False."""
        import json

        slot = _ChatSlot("s1")
        slot.append("user", "hello")
        slot.append("assistant", "hi there")
        cls_json = json.dumps(
            {
                "kind": "stop_event",
                "id": "stop-abc",
                "state": "stop_failed_reset",
                "outcome": "hard",
            }
        )
        slot.append("system", cls_json, cls_json)
        assert self._last_stop_soft(slot) is False

    def test_plain_string_cls_does_not_match(self):
        """Plain-string cls (legacy format) is ignored — no false positive."""
        slot = _ChatSlot("s1")
        slot.append("user", "hello")
        slot.append("system", "{}", "stop_event")  # plain string cls
        assert self._last_stop_soft(slot) is False


# ── Tests: AcpProcessDied handler in _run_chat ──


class TestStopDuringSessionPrep:
    """A Stop pressed while the turn is still being prepared must not open it.

    During ``get_or_create``'s cold start the session is not registered yet,
    so ``SessionManager.stop_turn`` answers ``"idle"`` and the stop resolves
    without cancelling anything. The dispatch gate in ``_run_chat`` is what
    honors that stop: it compares ``slot._stop_generation`` against the
    turn-entry snapshot and refuses to open the turn ([Stopped] card
    shown while the response streams to completion).
    """

    def _make_state_and_slot(self, tmp_path):
        from kiro_crew.dashboard.chat_runner import _run_chat

        state = _make_state(tmp_path)
        state.sessions.release = MagicMock()
        state.sessions.reset = AsyncMock()
        state.sessions.set_approval_policy = MagicMock()
        state.sessions.check_context_usage = MagicMock()
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.is_yolo_active = MagicMock(return_value=False)
        state._background_tasks = set()

        slot = state.get_or_create_slot("stop-prep-slot")
        slot.append("user", "hello", "msg msg-u")
        return state, slot, _run_chat

    def _make_client(self, stream_calls):
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        client = MagicMock()
        client.shutdown = AsyncMock()

        async def _stream(msg):
            stream_calls.append(msg)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="full response")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

        client.stream = _stream
        client.stream_command = _stream
        return client

    @pytest.mark.asyncio
    async def test_stop_during_get_or_create_aborts_dispatch(self, tmp_path: Path) -> None:
        """Stop lands mid-cold-start → the turn never opens, nothing streams.

        The stop is simulated exactly as the /stop handler leaves it for this
        race: ``_stop_state`` flips idle → soft_pending (bumping
        ``_stop_generation``) and, because ``stop_turn`` found no session and
        answered "idle", snaps straight back to idle. A point-in-time state
        check at dispatch sees nothing — only the generation delta survives.
        """
        state, slot, _run_chat = self._make_state_and_slot(tmp_path)
        stream_calls: list[str] = []
        client = self._make_client(stream_calls)

        async def _slow_create(*args, **kwargs):
            # Stop pressed while the session is still being created.
            slot._stop_state = "soft_pending"
            slot._stop_state = "idle"
            return client, True, False

        state.sessions.get_or_create = AsyncMock(side_effect=_slow_create)

        await _run_chat(state, slot, "hello")

        assert stream_calls == [], (
            "the turn was dispatched despite a Stop during session prep — "
            "the full response would stream behind a [Stopped] card (#5464)"
        )

    @pytest.mark.asyncio
    async def test_stop_during_get_or_create_does_not_accept_monitor_wake(
        self, tmp_path: Path
    ) -> None:
        """A stopped pre-stream wake stays retryable instead of becoming DISPATCHED."""
        from kiro_crew.monitoring.completion import MonitorCompletionHook

        state, slot, _run_chat = self._make_state_and_slot(tmp_path)
        stream_calls: list[str] = []
        client = self._make_client(stream_calls)

        async def _slow_create(*args, **kwargs):
            slot._stop_state = "soft_pending"
            slot._stop_state = "idle"
            return client, True, False

        state.sessions.get_or_create = AsyncMock(side_effect=_slow_create)
        accepted = MagicMock()
        completion = MonitorCompletionHook(
            "monitor1",
            "failure-a",
            AsyncMock(),
            authorization_callback=AsyncMock(return_value=True),
            acceptance_callback=accepted,
        )

        await _run_chat(state, slot, "hello", monitor_completion=completion)

        assert completion.accepted is False
        accepted.assert_not_called()
        assert stream_calls == []

    @pytest.mark.asyncio
    async def test_no_stop_dispatches_normally(self, tmp_path: Path) -> None:
        """Control: without a Stop, the same setup opens the turn exactly once."""
        state, slot, _run_chat = self._make_state_and_slot(tmp_path)
        stream_calls: list[str] = []
        client = self._make_client(stream_calls)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        await _run_chat(state, slot, "hello")

        assert len(stream_calls) == 1, "the dispatch gate must not fire without a Stop"

    @pytest.mark.asyncio
    async def test_stop_of_a_previous_turn_does_not_abort_the_next(self, tmp_path: Path) -> None:
        """A stop generation inherited from an EARLIER turn must not trip the gate.

        The gate compares against the snapshot taken at THIS turn's entry, so a
        slot whose previous turn was stopped (generation already > 0) still
        dispatches its next turn normally.
        """
        state, slot, _run_chat = self._make_state_and_slot(tmp_path)
        # A previous turn was stopped and resolved before this turn began.
        slot._stop_state = "soft_pending"
        slot._stop_state = "idle"

        stream_calls: list[str] = []
        client = self._make_client(stream_calls)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))

        await _run_chat(state, slot, "hello")

        assert (
            len(stream_calls) == 1
        ), "a stale stop generation from a previous turn aborted a fresh turn"


class TestAcpProcessDiedRecovery:
    """Verify _run_chat handles AcpProcessDied with retry logic, redaction, and session reset."""

    def _make_state_and_slot(self, tmp_path):
        from kiro_crew.dashboard.chat_runner import _run_chat

        state = _make_state(tmp_path)
        state.sessions.get_or_create = AsyncMock(return_value=(MagicMock(), False, False))
        state.sessions.release = MagicMock()
        state.sessions.reset = AsyncMock()
        state.sessions.set_approval_policy = MagicMock()
        state.sessions.check_context_usage = MagicMock()
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.is_yolo_active = MagicMock(return_value=False)
        state._background_tasks = set()

        slot = state.get_or_create_slot("pipe-death-slot")
        slot.append("user", "hello", "msg msg-u")

        mock_client = state.sessions.get_or_create.return_value[0]
        mock_client.shutdown = AsyncMock()
        return state, slot, mock_client, _run_chat

    def _make_stream_raise(self, mock_client, exc):
        async def _raise(msg):
            raise exc
            yield  # noqa: E501

        mock_client.stream = _raise
        mock_client.stream_command = _raise

    @pytest.mark.asyncio
    async def test_retry_at_depth_0_requeues_message(self, tmp_path: Path) -> None:
        """First pipe death at depth 0 → message re-queued, retrying shown."""
        from kiro_crew.acp.client import AcpProcessDied

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))

        await _run_chat(state, slot, "test message")

        state.sessions.reset.assert_awaited_once()
        assert slot._acp_pipe_death_retries == 1
        # Retry shows a single PERSISTED error card (reliably visible at
        # turn-teardown, unlike an ephemeral chat_status which the frontend drops
        # once the streaming turn ends). slot.append both persists the card and
        # emits a single chat_message via _on_message.
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("retrying" in m.get("content", "") for m in error_msgs)
        # Regression guard for the duplicate-card bug: the retry card must NOT be
        # ALSO explicitly broadcast over chat_message (slot.append already emits it
        # once via _on_message). A redundant broadcast_ws renders a second card.
        dup_broadcasts = [
            c
            for c in state.broadcast_ws.call_args_list
            if c.args
            and c.args[0] == "chat_message"
            and len(c.args) > 1
            and "retrying" in c.args[1].get("content", "")
        ]
        assert dup_broadcasts == [], "retry card must not be double-emitted via broadcast_ws"

    @pytest.mark.parametrize("ambiguous", [False, True])
    @pytest.mark.asyncio
    async def test_an_ambiguous_death_requeues_written_steers_as_possibly_delivered(
        self, tmp_path: Path, ambiguous: bool
    ) -> None:
        """A steer written into a turn sits in the same pipe as its prompt, so
        when the turn's death is ambiguous the teardown requeues it with the
        possibly-delivered mark; a plain death requeues it plainly."""
        from kiro_crew.acp.client import AcpProcessDied
        from kiro_crew.dashboard.chat_utils import STEER_POSSIBLY_DELIVERED_META

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _steered_then_died(msg):
            slot._pending_steers.append("also tag it")  # written mid-turn
            raise AcpProcessDied("stdin stalled", ambiguous_delivery=ambiguous)
            yield  # noqa: E501

        client.stream = _steered_then_died
        client.stream_command = _steered_then_died
        from kiro_crew.dashboard import chat_runner

        requeued: list[dict] = []
        _real = chat_runner._requeue_unconsumed_steers

        def _spy(state_, slot_):
            _real(state_, slot_)
            requeued.extend(e for e in slot_._queue if e.get("content") == "also tag it")

        with patch.object(chat_runner, "_requeue_unconsumed_steers", _spy):
            await _run_chat(state, slot, "test message")

        assert len(requeued) == 1
        assert bool((requeued[0].get("meta") or {}).get(STEER_POSSIBLY_DELIVERED_META)) is ambiguous

    @pytest.mark.asyncio
    async def test_a_possibly_delivered_steer_note_leads_only_the_model_input(
        self, tmp_path: Path
    ) -> None:
        """The note reaches the model; slash-command dispatch and the mirror still
        read the user's own text."""
        from kiro_crew.dashboard.chat_utils import STEER_POSSIBLY_DELIVERED_NOTE

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        sent: list[str] = []

        async def _capture(msg):
            sent.append(msg)
            return
            yield  # noqa: E501

        client.stream = _capture
        await _run_chat(state, slot, "also tag it", _steer_possibly_delivered=True)

        assert sent, "the turn must reach the model"
        assert STEER_POSSIBLY_DELIVERED_NOTE.strip() in sent[0]
        assert sent[0].rstrip().endswith("also tag it")

    @pytest.mark.asyncio
    async def test_budget_exhaustion_shows_stuck(self, tmp_path: Path) -> None:
        """4th pipe death → 'Session stuck' shown, no re-queue."""
        from kiro_crew.acp.client import AcpProcessDied

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._acp_pipe_death_retries = 3  # already exhausted
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))

        await _run_chat(state, slot, "test message")

        assert slot._acp_pipe_death_retries == 4
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("stuck" in m.get("content", "").lower() for m in error_msgs)

    @pytest.mark.asyncio
    async def test_nested_depth_shows_please_retry(self, tmp_path: Path) -> None:
        """Pipe death at depth > 0 → 'please retry' shown, no re-queue."""
        from kiro_crew.acp.client import AcpProcessDied

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))

        await _run_chat(state, slot, "test message", _prompt_depth=1)

        assert slot._acp_pipe_death_retries == 1
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("please retry" in m.get("content", "").lower() for m in error_msgs)

    @pytest.mark.asyncio
    async def test_partial_assistant_text_redacted(self, tmp_path: Path) -> None:
        """Pipe death mid-stream preserves redacted output and queues a continuation."""
        from kiro_crew.acp.client import AcpProcessDied
        from kiro_crew.dashboard.chat_utils import (
            _CONN_RECOVER_MSG,
            SYNTHETIC_RECOVERY_KIND,
            RecoveryPayload,
        )
        from kiro_crew.providers.base import EVENT_TEXT_CHUNK, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._titled = True

        async def _stream_then_die(msg):
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK, text="partial output with AKIA1234567890ABCDEF secret"
            )
            raise AcpProcessDied("pipe broken")

        client.stream = _stream_then_die
        client.stream_command = _stream_then_die

        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await _run_chat(state, slot, "test message")

        assistant_msgs = [m for m in slot.messages if m.get("role") == "assistant"]
        assert assistant_msgs, "Expected at least one assistant message with redacted content"
        for m in assistant_msgs:
            assert "AKIA1234567890ABCDEF" not in m.get("content", "")
        assert slot._queue == [
            {
                "id": slot._queue[0]["id"],
                "content": _CONN_RECOVER_MSG,
                "kind": SYNTHETIC_RECOVERY_KIND,
                # A continuation, not the user's request: the turn had emitted, so
                # this text is the runner's and must not mirror as user speech.
                "payload": RecoveryPayload.CONTINUATION,
                # Admission stamp: recovery requeues record the containment
                # that held at requeue so the drain can re-validate the retry.
                "meta": slot._queue[0]["meta"],
            }
        ]

    @pytest.mark.asyncio
    async def test_prompt_busy_requeue_does_not_claim_a_lost_connection(
        self, tmp_path: Path
    ) -> None:
        """A busy-session reset must requeue the busy continuation, not the connection one.

        Both causes reset the session and requeue a continuation, and the queued
        marker is what the transcript renders -- so borrowing the connection
        marker here reports a dropped connection to a user whose status card
        says the session was busy.
        """
        from kiro_crew.dashboard.chat_runner import PromptBusyExhaustedError
        from kiro_crew.dashboard.chat_utils import (
            _BUSY_RECOVER_MSG,
            SYNTHETIC_RECOVERY_KIND,
            RecoveryPayload,
        )
        from kiro_crew.providers.base import EVENT_TEXT_CHUNK, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._titled = True

        async def _stream_then_busy(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial output")
            raise PromptBusyExhaustedError("busy")

        client.stream = _stream_then_busy
        client.stream_command = _stream_then_busy

        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await _run_chat(state, slot, "test message")

        assert slot._queue == [
            {
                "id": slot._queue[0]["id"],
                "content": _BUSY_RECOVER_MSG,
                "kind": SYNTHETIC_RECOVERY_KIND,
                # A continuation, not the user's request: the turn had emitted, so
                # this text is the runner's and must not mirror as user speech.
                "payload": RecoveryPayload.CONTINUATION,
                # Admission stamp: recovery requeues record the containment
                # that held at requeue so the drain can re-validate the retry.
                "meta": slot._queue[0]["meta"],
            }
        ]
        # The status card and the queued marker describe the same event to two
        # audiences; they disagreed until the continuation became cause-aware.
        errors = [m.get("content", "") for m in slot.messages if m.get("role") == "error"]
        assert any("Session busy" in text for text in errors)
        assert not any("Connection lost" in text for text in errors)

    @pytest.mark.asyncio
    async def test_a_second_failure_before_output_keeps_the_text_machine_authored(
        self, tmp_path: Path
    ) -> None:
        """A recovery turn that dies again before emitting must stay machine-authored.

        The requeue replays ``message`` unchanged when nothing was emitted -- but on a
        second consecutive failure that message is the runner's own continuation from
        the previous recovery, not the user's request. Tagging it ORIGINAL makes the
        next dequeue mirror internal orchestration to a linked thread as user speech.
        """
        from kiro_crew.acp.client import AcpProcessDied
        from kiro_crew.dashboard.chat_utils import (
            _CONN_RECOVER_MSG,
            SYNTHETIC_RECOVERY_KIND,
            RecoveryPayload,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._titled = True
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))

        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await _run_chat(state, slot, _CONN_RECOVER_MSG, _synthetic_payload=True)

        assert slot._queue == [
            {
                "id": slot._queue[0]["id"],
                "content": _CONN_RECOVER_MSG,
                "kind": SYNTHETIC_RECOVERY_KIND,
                "payload": RecoveryPayload.CONTINUATION,
                # Admission stamp: recovery requeues record the containment
                # that held at requeue so the drain can re-validate the retry.
                "meta": slot._queue[0]["meta"],
            }
        ]

    @pytest.mark.asyncio
    async def test_a_recovery_turns_reply_still_reaches_the_linked_thread(
        self, tmp_path: Path
    ) -> None:
        """Withholding the user echo must not also withhold the assistant reply.

        A Slack-linked turn that emits output and then loses its connection recovers as
        a synthetic continuation. Skipping the whole mirror SETUP for that turn leaves
        no thread to reply into, so the continuation's answer is never delivered and the
        question asked on Slack stays unanswered.
        """
        from kiro_crew.dashboard.chat_utils import _CONN_RECOVER_MSG
        from kiro_crew.providers.base import EVENT_TEXT_CHUNK, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._titled = True
        state.sessions.get_slack_link = MagicMock(return_value=("ts-1", "C123"))
        state.slack_client = AsyncMock()
        state.slack_client.start_stream = AsyncMock(return_value="")

        async def _stream_text(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="the finished answer")

        client.stream = _stream_text
        client.stream_command = _stream_text

        await _run_chat(state, slot, _CONN_RECOVER_MSG, _synthetic_payload=True)

        posted = [str(c.args) for c in state.slack_client.post_message.await_args_list]
        assert any(
            "the finished answer" in a for a in posted
        ), f"recovery reply never delivered to the linked thread; posted={posted}"
        # The user echo stays withheld: that text is the runner's, not the user's.
        assert not any("\U0001f4ac" in a for a in posted), f"echoed runner text: {posted}"

    @pytest.mark.asyncio
    async def test_session_reset_propagated(self, tmp_path: Path) -> None:
        """Verify the finally block resets the session after AcpProcessDied."""
        from kiro_crew.acp.client import AcpProcessDied

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))

        await _run_chat(state, slot, "test message")

        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_state_suppresses_requeue(self, tmp_path: Path) -> None:
        """AcpProcessDied during active stop → no re-queue."""
        from kiro_crew.acp.client import AcpProcessDied

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._stop_state = "killing"
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))

        await _run_chat(state, slot, "test message")

        assert not slot._queue, "message should not be re-queued during active stop"

    @pytest.mark.asyncio
    async def test_stop_state_suppresses_requeue_prompt_busy(self, tmp_path: Path) -> None:
        """PromptBusyExhaustedError during active stop → no re-queue."""
        from kiro_crew.dashboard.chat_runner import PromptBusyExhaustedError

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._stop_state = "soft_pending"
        self._make_stream_raise(client, PromptBusyExhaustedError("busy"))

        await _run_chat(state, slot, "test message")

        assert not slot._queue, "message should not be re-queued during active stop"

    @pytest.mark.asyncio
    async def test_stop_state_suppresses_requeue_acp_error(self, tmp_path: Path) -> None:
        """AcpError retry-eligible during active stop → no re-queue."""
        from kiro_crew.acp.client import AcpError

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._stop_state = "killing"
        self._make_stream_raise(client, AcpError("process exited"))

        await _run_chat(state, slot, "test message")

        assert not slot._queue, "message should not be re-queued during active stop"

    @pytest.mark.asyncio
    async def test_should_suppress_requeue_helper(self, tmp_path: Path) -> None:
        """_should_suppress_requeue returns True for non-idle states."""
        from kiro_crew.dashboard.chat_runner import _should_suppress_requeue

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("helper-test")

        slot._stop_state = "idle"
        assert _should_suppress_requeue(slot) is False

        slot._stop_state = "soft_pending"
        assert _should_suppress_requeue(slot) is True

        slot._stop_state = "killing"
        assert _should_suppress_requeue(slot) is True

    @pytest.mark.asyncio
    async def test_cancelled_error_redacts_partial_text(self, tmp_path: Path) -> None:
        """CancelledError mid-stream → partial output redacted before display."""
        from kiro_crew.providers.base import EVENT_TEXT_CHUNK, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _stream_then_cancel(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial with AKIA1234567890ABCDEF key")
            raise asyncio.CancelledError()

        client.stream = _stream_then_cancel
        client.stream_command = _stream_then_cancel

        await _run_chat(state, slot, "test message")

        assistant_msgs = [m for m in slot.messages if m.get("role") == "assistant"]
        assert assistant_msgs, "Expected at least one assistant message with redacted content"
        for m in assistant_msgs:
            assert "AKIA1234567890ABCDEF" not in m.get("content", "")

    @pytest.mark.asyncio
    async def test_retry_requeues_via_queue_insert(self, tmp_path: Path) -> None:
        """First pipe death at depth 0 → queue_insert is called."""
        from unittest.mock import patch as _patch

        from kiro_crew.acp.client import AcpProcessDied
        from kiro_crew.dashboard.state import _ChatSlot

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))

        calls = []
        orig = _ChatSlot.queue_insert

        def spy(self_slot, *a, **kw):
            calls.append(a)
            return orig(self_slot, *a, **kw)

        with _patch.object(_ChatSlot, "queue_insert", spy):
            await _run_chat(state, slot, "test message")

        assert (0, "test message") in calls

    @pytest.mark.asyncio
    async def test_retry_requeues_with_consumption_callback(self, tmp_path: Path) -> None:
        """A pre-consumption pipe-death retry keeps its producer callback."""
        from unittest.mock import Mock
        from unittest.mock import patch as _patch

        from kiro_crew.acp.client import AcpProcessDied
        from kiro_crew.dashboard.state import _ChatSlot

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpProcessDied("pipe broken"))
        on_consumed = Mock()
        on_irreversibly_consumed = Mock()
        callbacks = []
        orig = _ChatSlot.queue_insert

        def spy(self_slot, *args, **kwargs):
            callbacks.append(
                (
                    kwargs.get("on_consumed"),
                    kwargs.get("on_irreversibly_consumed"),
                )
            )
            return orig(self_slot, *args, **kwargs)

        with (
            _patch.object(_ChatSlot, "queue_insert", spy),
            _patch(
                "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
                new=AsyncMock(return_value=False),
            ),
            _patch("kiro_crew.dashboard.chat_runner._finish_queue_cycle"),
        ):
            await _run_chat(
                state,
                slot,
                "test message",
                _on_consumed=on_consumed,
                _on_irreversibly_consumed=on_irreversibly_consumed,
            )

        assert callbacks[0] == (on_consumed, on_irreversibly_consumed)

    @pytest.mark.parametrize("event_type", ["text", "tool"])
    @pytest.mark.asyncio
    async def test_output_reports_irreversible_consumption_before_turn_end(
        self, tmp_path: Path, event_type: str
    ) -> None:
        """First token or tool closes durable replay windows immediately."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        irreversible_reported = asyncio.Event()
        allow_turn_finish = asyncio.Event()

        async def _stream(_message):
            if event_type == "text":
                yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="started")
            else:
                yield LLMEvent(kind=EVENT_TOOL_CALL, title="read_file", tool_kind="read")
            await allow_turn_finish.wait()
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream
        turn = asyncio.create_task(
            _run_chat(
                state,
                slot,
                "test message",
                _on_irreversibly_consumed=irreversible_reported.set,
            )
        )

        report = asyncio.create_task(irreversible_reported.wait())
        done, _pending = await asyncio.wait(
            (turn, report),
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert report in done, turn.exception()
        assert not turn.done()

        allow_turn_finish.set()
        await turn

    @pytest.mark.asyncio
    async def test_acperror_process_exited_uses_pipe_death_counter(self, tmp_path: Path) -> None:
        """Option Y: AcpError 'process exited' increments the pipe-death counter, not busy."""
        from kiro_crew.acp.client import AcpError

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpError("ACP process exited (code=1)"))

        await _run_chat(state, slot, "test message")

        assert slot._acp_pipe_death_retries == 1
        assert slot._prompt_busy_retries == 0
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("Connection lost" in m.get("content", "") for m in error_msgs)
        # A recovery IS queued here, so the notice must be tagged: an untagged row is
        # indistinguishable from a terminal failure and re-offers an executing choice.
        from kiro_crew.dashboard.chat_utils import TRANSIENT_RETRY_KIND

        _retrying = [m for m in error_msgs if "retrying" in m.get("content", "").lower()]
        assert _retrying, "no retrying notice to inspect"
        assert all((m.get("meta") or {}).get("kind") == TRANSIENT_RETRY_KIND for m in _retrying)
        # LIVE path: slot.append already emits one tagged chat_message, so an explicit
        # frame here would be an untagged duplicate that re-arms the pills mid-backoff.
        _dupes = [
            c
            for c in state.broadcast_ws.call_args_list
            if c.args
            and c.args[0] == "chat_message"
            and isinstance(c.args[1], dict)
            and "retrying" in str(c.args[1].get("content", "")).lower()
        ]
        assert not _dupes, f"untagged explicit chat_message frame(s): {_dupes}"

    @pytest.mark.asyncio
    async def test_acperror_already_in_progress_uses_busy_counter(self, tmp_path: Path) -> None:
        """Option Y: AcpError 'already in progress' increments the busy counter, not pipe-death."""
        from kiro_crew.acp.client import AcpError

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpError("Prompt already in progress"))

        await _run_chat(state, slot, "test message")

        assert slot._prompt_busy_retries == 1
        assert slot._acp_pipe_death_retries == 0
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("Session busy" in m.get("content", "") for m in error_msgs)

    @pytest.mark.asyncio
    async def test_prompt_busy_exhausted_depth_gt0_shows_please_retry(self, tmp_path: Path) -> None:
        """PromptBusyExhaustedError at _prompt_depth>0 with budget remaining hits the
        else branch: surfaces a 'Session busy — please retry' card (not a silent
        failure) and does NOT re-queue (mirrors the AcpProcessDied depth>0 handling)."""
        from kiro_crew.llm_helpers import PromptBusyExhaustedError

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, PromptBusyExhaustedError("busy"))

        await _run_chat(state, slot, "test message", _prompt_depth=1)

        # depth>0 + counter (1) <= 3: neither the retry-and-requeue `if` nor the
        # `elif > 3` fires, so the new `else` surfaces feedback instead of failing
        # silently. No re-queue at depth>0.
        assert slot._prompt_busy_retries == 1
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("please retry" in m.get("content", "").lower() for m in error_msgs)
        assert any("Session busy" in m.get("content", "") for m in error_msgs)

    @pytest.mark.asyncio
    async def test_acperror_process_exited_depth_gt0_resets_no_requeue(
        self, tmp_path: Path
    ) -> None:
        """A pipe-death AcpError ('process exited') at _prompt_depth>0 must STILL reset
        the dead session and increment the pipe-death counter (mirrors AcpProcessDied /
        PromptBusyExhaustedError), and surface a 'Connection lost — please retry' card —
        but NOT re-queue (re-queue is depth-0 only). Gating the whole reset/counter
        block on `_prompt_depth == 0` makes a depth>0 pipe-death fall through to
        the generic else: no session reset (the next turn hit the dead process) and the
        failure never counted toward the exhaustion threshold."""
        from kiro_crew.acp.client import AcpError

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_stream_raise(client, AcpError("ACP process exited (code=1)"))

        await _run_chat(state, slot, "test message", _prompt_depth=1)

        # The dead process IS reset and the failure IS counted, regardless of depth.
        state.sessions.reset.assert_awaited_once()
        assert slot._acp_pipe_death_retries == 1
        # A friendly feedback card (not a bare ❌ raw-error card), and NO re-queue.
        error_msgs = [m for m in slot.messages if m.get("role") == "error"]
        assert any("Connection lost" in m.get("content", "") for m in error_msgs)
        assert any("please retry" in m.get("content", "").lower() for m in error_msgs)
        assert not slot._queue, "depth>0 must not re-queue"
        # Discriminating control: nothing is queued, so this notice is genuinely terminal
        # and must NOT be tagged -- tagging it would hide pills a user still needs.
        from kiro_crew.dashboard.chat_utils import TRANSIENT_RETRY_KIND

        assert all((m.get("meta") or {}).get("kind") != TRANSIENT_RETRY_KIND for m in error_msgs)


class TestEmptyResponseRetry:
    """Verify _run_chat retries once on empty model response, then shows error."""

    def _make_state_and_slot(self, tmp_path):
        from kiro_crew.dashboard.chat_runner import _run_chat

        state = _make_state(tmp_path)
        state.sessions.get_or_create = AsyncMock(return_value=(MagicMock(), False, False))
        state.sessions.release = MagicMock()
        state.sessions.reset = AsyncMock()
        state.sessions.set_approval_policy = MagicMock()
        state.sessions.check_context_usage = MagicMock()
        state.sessions.record_success = MagicMock()
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.is_yolo_active = MagicMock(return_value=False)
        state._background_tasks = set()

        slot = state.get_or_create_slot("empty-resp-slot")
        slot.append("user", "hello", "msg msg-u")

        mock_client = state.sessions.get_or_create.return_value[0]
        mock_client.is_kiro_backend = True
        mock_client.context_usage_pct = MagicMock(return_value=50.0)
        mock_client.shutdown = AsyncMock()
        return state, slot, mock_client, _run_chat

    def _make_empty_stream(self, mock_client):
        """Stream that completes immediately with no text."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        mock_client.stream = _stream
        mock_client.stream_command = _stream

    @staticmethod
    async def _cancel_background_tasks(state) -> None:
        """Cancel AND await every task this isolated state spawned.

        `_run_chat` starts title/summary work that is irrelevant to these
        recovery assertions. A bare `task.cancel()` leaves the coroutine pending
        until the loop gets another tick, which surfaces as unawaited-coroutine
        and destroyed-pending-task warnings under xdist. Await the cancellation
        exactly; never sleep and never let one test's teardown spill into another.
        """
        tasks = list(state._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("resumed", (False, True))
    async def test_read_then_false_tool_budget_claim_auto_continues(
        self, tmp_path: Path, resumed: bool
    ) -> None:
        """A false blocker after a read replays only the authenticated request."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.dashboard.chat_utils import (
            FALSE_TOOL_BLOCKER_REPLAY_KIND,
            RecoveryPayload,
        )
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        state.sessions.get_or_create.return_value = (client, False, resumed)
        state.context_builder = MagicMock()
        state.context_builder.conversation_log = None
        state.context_builder.build_message.side_effect = lambda message, *_args, **_kwargs: (
            message,
            None,
        )
        hidden_context = "SECRET APP CONTEXT: never mirror this"
        slot._pending_context = [{"content": hidden_context, "source": "test-app"}]
        provider_prompts: list[str] = []

        async def _stream(msg):
            provider_prompts.append(msg)
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Reading CR workflow",
                tool_kind="read",
                tool_input="{}",
                tool_call_id="read-1",
                tool_name="fs_read",
                tool_identity_trusted=True,
            )
            yield LLMEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id="read-1",
                tool_output="workflow loaded",
                tool_final=True,
            )
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=(
                    "I’m proceeding, but this turn’s tool budget was exhausted immediately "
                    "after loading the workflow. No publish or deployment has happened yet."
                ),
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        file_path = "/tmp/My Report.pdf"
        dir_path = "/tmp/My Reports/"
        request = (
            "Publish and deploy the demo\n"
            f"[attached_file 1] {file_path}\n"
            f"[attached_dir 1] {dir_path}"
        )
        slot.append(
            "user",
            request,
            "msg msg-u",
            meta={"files": [file_path], "dirs": [dir_path]},
        )
        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new=AsyncMock(return_value=False),
        ):
            await _run_chat(
                state,
                slot,
                request,
                _directive_user_origin=True,
            )
            await self._cancel_background_tasks(state)

        replay = next(item for item in slot._queue if item.get("content") == request)
        assert hidden_context in provider_prompts[0]
        assert replay["content"] == request
        assert hidden_context not in replay["content"]
        assert replay["kind"] == FALSE_TOOL_BLOCKER_REPLAY_KIND
        assert replay["payload"] == RecoveryPayload.ORIGINAL
        assert replay["meta"]["files"] == [file_path]
        assert replay["meta"]["dirs"] == [dir_path]
        assert replay["_directive_user_origin"] is True
        assert slot._promise_only_retries == 1
        state.sessions.record_success.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("result_shape", ["failed", "missing"])
    async def test_uncompleted_read_then_false_blocker_does_not_replay(
        self, tmp_path: Path, result_shape: str
    ) -> None:
        """Canonical read identity is insufficient without completed result proof."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _stream(msg):
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Reading workflow",
                tool_kind="read",
                tool_call_id="read-1",
                tool_name="fs_read",
                tool_identity_trusted=True,
            )
            if result_shape == "failed":
                yield LLMEvent(
                    kind=EVENT_TOOL_RESULT,
                    tool_call_id="read-1",
                    tool_output="read failed",
                    tool_final=False,
                )
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=(
                    "I’m blocked from further tool execution in the resumed session: "
                    "the screenshot delivery tools are no longer callable here."
                ),
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new=AsyncMock(return_value=False),
        ):
            await _run_chat(
                state,
                slot,
                "Publish and deploy the demo",
                _directive_user_origin=True,
            )
            await self._cancel_background_tasks(state)

        assert not slot._queue
        assert slot._promise_only_retries == 0
        state.sessions.record_success.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("identity_trusted", "kiro_backend"),
        ((False, True), (True, False)),
    )
    async def test_untrusted_or_non_kiro_identity_does_not_replay(
        self,
        tmp_path: Path,
        identity_trusted: bool,
        kiro_backend: bool,
    ) -> None:
        """A familiar name needs extractor provenance and a Kiro backend."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        client.is_kiro_backend = kiro_backend

        async def _stream(msg):
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="grep",
                tool_kind="read",
                tool_call_id="read-1",
                tool_name="grep",
                tool_identity_trusted=identity_trusted,
            )
            yield LLMEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id="read-1",
                tool_output="matches",
                tool_final=True,
            )
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=(
                    "I’m blocked from further tool execution in the resumed session: "
                    "the screenshot delivery tools are no longer callable here."
                ),
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new=AsyncMock(return_value=False),
        ):
            await _run_chat(
                state,
                slot,
                "Publish and deploy the demo",
                _directive_user_origin=True,
            )
            await self._cancel_background_tasks(state)

        assert not slot._queue
        assert slot._promise_only_retries == 0
        state.sessions.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_kiro_blocker_near_miss_logs_without_replay(self, tmp_path: Path, caplog) -> None:
        """Wording drift is observable but never grants replay authority."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        near_miss = "I'm blocked from further tool execution."

        async def _stream(msg):
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Reading workflow",
                tool_kind="read",
                tool_call_id="read-1",
                tool_name="fs_read",
                tool_identity_trusted=True,
            )
            yield LLMEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id="read-1",
                tool_output="workflow loaded",
                tool_final=True,
            )
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text=near_miss)
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        with (
            caplog.at_level("WARNING", logger="kiro_crew.dashboard.chat_runner"),
            patch(
                "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
                new=AsyncMock(return_value=False),
            ),
        ):
            await _run_chat(
                state,
                slot,
                "Publish and deploy the demo",
                _directive_user_origin=True,
            )
            await self._cancel_background_tasks(state)

        assert not slot._queue
        assert slot._promise_only_retries == 0
        assert "False tool-blocker wording drift" in caplog.text
        assert near_miss not in caplog.text
        state.sessions.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_yolo_read_then_false_blocker_stays_notice_only(self, tmp_path: Path) -> None:
        """Auto-approve never dispatches even an authenticated blocker replay."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        state.is_yolo_active.return_value = True

        async def _stream(msg):
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Reading workflow",
                tool_kind="read",
                tool_input="{}",
                tool_call_id="read-1",
                tool_name="fs_read",
                tool_identity_trusted=True,
            )
            yield LLMEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id="read-1",
                tool_output="workflow loaded",
                tool_final=True,
            )
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=(
                    "I’m blocked from further tool execution in the resumed session: "
                    "the screenshot delivery tools are no longer callable here."
                ),
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new=AsyncMock(return_value=False),
        ):
            await _run_chat(
                state,
                slot,
                "Publish and deploy the demo",
                _directive_user_origin=True,
            )
            await self._cancel_background_tasks(state)

        assert not slot._queue
        assert slot._promise_only_retries == 0
        notices = [m.get("content", "") for m in slot.messages if m.get("role") == "notice"]
        assert any("Auto-continue is skipped under auto-approve mode" in m for m in notices)
        state.sessions.record_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_user_read_then_false_blocker_does_not_replay(self, tmp_path: Path) -> None:
        """App/system text cannot acquire authenticated-user replay provenance."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _stream(msg):
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Reading workflow",
                tool_kind="read",
                tool_input="{}",
                tool_call_id="read-1",
                tool_name="fs_read",
                tool_identity_trusted=True,
            )
            yield LLMEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id="read-1",
                tool_output="workflow loaded",
                tool_final=True,
            )
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=(
                    "I’m blocked from further tool execution in the resumed session: "
                    "the screenshot delivery tools are no longer callable here."
                ),
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new=AsyncMock(return_value=False),
        ):
            await _run_chat(state, slot, "Untrusted app injection")
            await self._cancel_background_tasks(state)

        assert not slot._queue
        assert slot._promise_only_retries == 0
        state.sessions.record_success.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("intervention", ["stop", "followup", "steer"])
    async def test_late_user_intervention_purges_false_blocker_replay(
        self, tmp_path: Path, intervention: str
    ) -> None:
        """A revocation landing after enqueue still wins at dispatch time."""
        from kiro_crew.dashboard.chat_runner import _start_next_queued_turn
        from kiro_crew.dashboard.chat_utils import (
            FALSE_TOOL_BLOCKER_REPLAY_KIND,
            RecoveryPayload,
        )

        state = _make_state(tmp_path)
        state.broadcast_ws = MagicMock()
        slot = state.get_or_create_slot(f"late-{intervention}")
        request = "Delete the deployment"
        slot.queue_insert(
            0,
            request,
            kind=FALSE_TOOL_BLOCKER_REPLAY_KIND,
            payload=RecoveryPayload.ORIGINAL,
            directive_user_origin=True,
        )
        slot._promise_only_retries = 1
        slot.replays.arm(
            ReplayFamily.CONTINUATION,
            entry_id="",
            session_key="",
            stop_gen=slot._stop_generation,
            session_stop_gen=0,
        )

        if intervention == "stop":
            slot._stop_generation += 1
        elif intervention == "followup":
            slot.queue_append("Never mind", directive_user_origin=True)
            # A running child holds the user's follow-up after the purge.
            state.subagents = MagicMock(running_agents_for=MagicMock(return_value=["child"]))
        else:
            slot._pending_steers = [{"content": "Stop; keep it"}]

        cfg = MagicMock()
        cfg.dashboard.merge_queued_messages = False
        with patch(
            "kiro_crew.dashboard.chat_runner.KiroCrewConfig.load",
            return_value=cfg,
        ):
            started = await _start_next_queued_turn(state, slot)

        assert started is False
        assert all(item.get("kind") != FALSE_TOOL_BLOCKER_REPLAY_KIND for item in slot._queue)
        assert slot._promise_only_retries == 0
        corrections = [
            msg.get("content", "")
            for msg in slot.messages
            if msg.get("role") == "notice" and "Auto-continue cancelled" in msg.get("content", "")
        ]
        assert len(corrections) == 1
        if intervention == "stop":
            assert "nothing was run" in corrections[0]
        else:
            assert "takes over" in corrections[0]

    @pytest.mark.asyncio
    async def test_session_scoped_stop_after_enqueue_purges_false_blocker_replay(
        self, tmp_path: Path
    ) -> None:
        """The false-blocker arm snapshots the SESSION stop counter at enqueue.

        A Stop issued on a linked channel surface moves only the session-scoped
        count (the slot's own ``_stop_generation`` never ticks), and the
        dispatch-point purge compares that count against its value AT ENQUEUE
        (the continuation record in ``slot.replays``). Without the arm's snapshot
        the comparison reads a count the arm never took, and the replay
        dispatches over the user's Stop.
        """
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.dashboard.chat_runner import _start_next_queued_turn
        from kiro_crew.dashboard.chat_utils import (
            FALSE_TOOL_BLOCKER_REPLAY_KIND,
            effective_session_key,
        )
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        state.context_builder = MagicMock()
        state.context_builder.conversation_log = None
        state.context_builder.build_message.side_effect = lambda message, *_args, **_kwargs: (
            message,
            None,
        )
        # A session-scoped Stop counter the runner really reads: the arm's
        # enqueue snapshot and the purge's live read both go through
        # ``sessions.stop_generation(session_key)``.
        stop_counts: dict[str, int] = {}
        state.sessions.stop_generation = lambda key: stop_counts.get(key, 0)
        session_key = effective_session_key(slot)
        # Non-zero baseline: the record a slot starts with counts zero, so the
        # revalidation below reads "moved" unless the arm took the enqueue-time
        # count.
        stop_counts[session_key] = 5

        async def _stream(msg):
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                title="Reading CR workflow",
                tool_kind="read",
                tool_input="{}",
                tool_call_id="read-1",
                tool_name="fs_read",
                tool_identity_trusted=True,
            )
            yield LLMEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id="read-1",
                tool_output="workflow loaded",
                tool_final=True,
            )
            yield LLMEvent(
                kind=EVENT_TEXT_CHUNK,
                text=(
                    "I’m proceeding, but this turn’s tool budget was exhausted "
                    "immediately after loading the workflow. No publish or "
                    "deployment has happened yet."
                ),
            )
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        request = "Publish and deploy the demo"
        slot.append("user", request, "msg msg-u")
        with patch(
            "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
            new=AsyncMock(return_value=False),
        ):
            await _run_chat(state, slot, request, _directive_user_origin=True)
            await self._cancel_background_tasks(state)

        replay = next(item for item in slot._queue if item.get("content") == request)
        assert replay["kind"] == FALSE_TOOL_BLOCKER_REPLAY_KIND
        # The arm must snapshot the ENQUEUE-time session count, exactly as the
        # sibling recovery arms do: with no Stop since, the record is not moved.
        assert not slot.replays.revalidate(
            ReplayFamily.CONTINUATION,
            LiveSlot(
                session_key=session_key,
                stop_generation=slot._stop_generation,
                stopping=False,
                user_input=False,
                session_stop_generation=lambda key: stop_counts.get(key, 0),
            ),
        ).stop_moved

        # A channel-side Stop lands while the replay waits: only the session
        # count moves. The dispatch-point purge must drop the replay.
        stop_counts[session_key] = 6
        cfg = MagicMock()
        cfg.dashboard.merge_queued_messages = False
        with patch(
            "kiro_crew.dashboard.chat_runner.KiroCrewConfig.load",
            return_value=cfg,
        ):
            started = await _start_next_queued_turn(state, slot)

        assert started is False
        assert all(item.get("kind") != FALSE_TOOL_BLOCKER_REPLAY_KIND for item in slot._queue)
        assert slot._promise_only_retries == 0
        assert slot._stop_generation == 0, "the slot's own Stop state was never touched"
        assert any(
            m.get("role") == "notice" and "nothing was run" in m.get("content", "")
            for m in slot.messages
        )

    @pytest.mark.asyncio
    async def test_first_empty_response_requeues_message(self, tmp_path: Path) -> None:
        """First empty response at depth 0 → message re-queued silently."""
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_empty_stream(client)

        # Spy on queue_insert to verify the message is ACTUALLY re-queued (the
        # behavior the test name promises) — not merely that the counter ticked.
        calls = []
        callbacks = []
        directive_origins = []
        directive_channel_origins = []
        on_consumed = MagicMock()
        orig = _ChatSlot.queue_insert

        def spy(self_slot, *a, **kw):
            calls.append(a)
            callbacks.append(kw.get("on_consumed"))
            directive_origins.append(kw.get("directive_user_origin"))
            directive_channel_origins.append(kw.get("directive_channel_origin"))
            return orig(self_slot, *a, **kw)

        with (
            patch.object(_ChatSlot, "queue_insert", spy),
            patch("kiro_crew.dashboard.chat_runner.save_slot_off_loop") as mock_save,
            patch("kiro_crew.dashboard.chat_runner._maybe_consolidate") as mock_consolidate,
            patch("kiro_crew.dashboard.chat_runner._flush_file_changes") as mock_flush,
            patch(
                "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
                new=AsyncMock(return_value=False),
            ),
        ):
            # The behavior under test is the re-queue itself. Keep the queued
            # item in the slot, but stop the finally block from immediately
            # launching a second turn. Patching the queue-drain boundary avoids
            # globally replacing asyncio.create_task, which can otherwise let
            # unrelated lifecycle tasks race the assertions under xdist load.
            await _run_chat(
                state,
                slot,
                "test message",
                _directive_user_origin=True,
                _directive_channel_origin=True,
                _on_consumed=on_consumed,
            )
            background_tasks = list(state._background_tasks)
            for _bg_task in background_tasks:
                _bg_task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)

        assert slot._empty_response_retries == 1
        # The message must be re-queued at the front of the queue.
        assert (0, "test message") in calls
        assert callbacks[0] is on_consumed
        assert directive_origins == [True]
        assert directive_channel_origins == [True]
        assert [args.args for args in on_consumed.call_args_list] == [(True,), (False,)]
        # No notice card shown on first attempt — the empty is silently re-queued
        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert not any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        # Re-queue path must NOT persist/consolidate the spurious empty turn or
        # record success (item 3 of the CR). The only saves are the in-flight
        # marker's: this slot has no transcript yet, so admission falls back to
        # the forced save that creates it, and the finally clears the marker
        # with a second one before the queued retry receives its own generation.
        assert mock_save.await_count == 2
        for marker_save in mock_save.await_args_list:
            assert marker_save.args == (state, slot)
            # Both marker saves are pinned to this slot incarnation so a
            # cancelled runner outliving a same-key replacement writes nothing.
            assert marker_save.kwargs == {
                "force": True,
                "expected_history_key": "dashboard:empty-resp-slot",
                "expected_slot_name": "empty-resp-slot",
            }
        mock_consolidate.assert_not_called()
        state.sessions.record_success.assert_not_called()
        # _flush_file_changes is intentionally NOT skipped: the try-body call (inside
        # the `if not _retrying_empty` guard) is skipped, but the finally block calls
        # it once unconditionally ("ensure file changes always surface, even on
        # cancel/error"). Scope the assertion to this slot because finalization of a
        # task created by another test can run while this module-level function is
        # patched and must not change the behavior observed for this turn.
        own_flushes = [
            mock_call for mock_call in mock_flush.call_args_list if mock_call.args[:1] == (slot,)
        ]
        assert len(own_flushes) == 1

    @pytest.mark.asyncio
    async def test_empty_response_at_depth_gt0_shows_error_immediately(
        self, tmp_path: Path
    ) -> None:
        """At _prompt_depth>0 an empty response is NOT silently retried — it shows the
        terminal empty-response notice card on the FIRST empty. The silent
        re-queue is intentionally depth-0 only (nested tool-use turns must not silently
        re-queue and risk a runaway loop)."""
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_empty_stream(client)

        await _run_chat(state, slot, "test message", _prompt_depth=1)

        # depth>0: the `if _prompt_depth == 0 ...` guard is false, so the else fires —
        # terminal card immediately, no silent retry, no increment of the counter.
        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        assert slot._empty_response_retries == 0

    @pytest.mark.asyncio
    async def test_second_empty_response_auto_continues(self, tmp_path: Path) -> None:
        """Second consecutive empty response → ONE synthetic continue nudge is
        queued (same live session) with a transcript-visible notice. Re-sending
        the identical prompt reproduces the identical empty generation; a
        DIFFERENT message reliably recovers — this automates the user manually
        typing "continue"."""
        from kiro_crew.dashboard.chat_runner import _EMPTY_AUTO_CONTINUE_MSG

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 1  # already silently retried once
        self._make_empty_stream(client)

        calls = []
        orig = _ChatSlot.queue_insert

        def spy(self_slot, *a, **kw):
            calls.append(a)
            return orig(self_slot, *a, **kw)

        with patch.object(_ChatSlot, "queue_insert", spy):
            await _run_chat(state, slot, "test message")
            await self._cancel_background_tasks(state)

        # The nudge (NOT the original message) is queued at the front.
        assert (0, _EMPTY_AUTO_CONTINUE_MSG) in calls
        assert (0, "test message") not in calls
        # Visible recovery notice, counter advanced to the terminal rung, and
        # the recovery turn is excluded from the cycle-complete counter reset.
        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("auto-continuing once" in m.get("content", "") for m in notice_msgs)
        assert slot._empty_response_retries == 2

    @pytest.mark.asyncio
    async def test_auto_continue_nudge_drains_as_inject_not_user(self, tmp_path: Path) -> None:
        """The queued nudge is runner orchestration, not user speech: when the
        queue drains it, the transcript append MUST use the "inject" recovery
        role (never "user" — a user-role append would persist an internal
        instruction as user-authored history and mirror it to linked
        channels), and it MUST NOT cancel a pending synthesis (the user did
        not take over the conversation)."""
        from kiro_crew.dashboard.chat_runner import _EMPTY_AUTO_CONTINUE_MSG

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 1  # next empty triggers the nudge rung
        slot._pending_synthesis = True
        self._make_empty_stream(client)

        await _run_chat(state, slot, "test message")
        # Let the REAL detached drain task run: it pops the nudge, classifies
        # it, appends it to the transcript, and runs the (also-empty) nudge
        # turn, which terminates the ladder at the give-up notice.
        for _bg_task in list(state._background_tasks):
            try:
                await _bg_task
            except Exception:
                pass

        nudge_msgs = [m for m in slot.messages if m.get("content") == _EMPTY_AUTO_CONTINUE_MSG]
        assert nudge_msgs, "drained nudge never reached the transcript"
        # The nudge must NEVER carry the user role (that would persist an
        # internal instruction as user-authored history and mirror it to
        # linked channels) — the ORIGINAL user message keeps its user role.
        assert all(m.get("role") == "inject" for m in nudge_msgs)
        # Draining a synthetic recovery message is not a user takeover.
        assert slot._pending_synthesis is True

    @pytest.mark.asyncio
    async def test_third_empty_response_shows_notice(self, tmp_path: Path) -> None:
        """Third consecutive empty (the auto-continue nudge ALSO produced
        nothing) → terminal notice card; the ladder is bounded, never loops."""
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 2  # re-queue + nudge both spent
        self._make_empty_stream(client)

        await _run_chat(state, slot, "test message")

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        # After the terminal notice, the counter resets so the NEXT independent
        # user turn gets a fresh budget (not sticky).
        assert slot._empty_response_retries == 0

    @pytest.mark.asyncio
    async def test_productive_giveup_notice_does_not_claim_nothing_happened(
        self, tmp_path: Path
    ) -> None:
        """A PRODUCTIVE turn reaching give-up (its one continuation also ended
        without a closing reply) must not be described as "returned nothing":
        the card is read by the model via the transcript, and telling it the
        completed work produced no output invites a redo of side effects that
        already landed. Same distinction the continue rung draws one rung up."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 2  # the single continuation is spent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TOOL_CALL, tool_call_id="tc-1", title="send_message")
            yield LLMEvent(kind=EVENT_TOOL_RESULT, tool_call_id="tc-1", text="sent")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        await _run_chat(state, slot, "test message")
        await self._cancel_background_tasks(state)

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert notice_msgs, "give-up produced no notice card"
        assert not any("returned nothing" in m.get("content", "") for m in notice_msgs), (
            "the give-up card claims the turn produced nothing, but this turn "
            "ran a tool whose side effects already landed"
        )
        assert any("without a closing reply" in m.get("content", "") for m in notice_msgs)
        # The card reassures rather than inviting a redo: completed steps stay done.
        assert any("will not re-run" in m.get("content", "") for m in notice_msgs)
        # Terminal rung still resets the budget for the next genuine user turn.
        assert slot._empty_response_retries == 0

    @pytest.mark.asyncio
    async def test_productive_episode_giveup_keeps_no_rerun_wording(self, tmp_path: Path) -> None:
        """A give-up whose EPISODE was productive must keep the productive
        wording, even though the final continuation turn itself produced
        nothing.

        The activity snapshot is rebuilt per turn, so the continuation that
        streams nothing arrives at give-up with `productive=False`. The counter
        cannot rescue it either: the productive jump (`= 2`) and the plain
        non-productive ladder (`+= 1` twice) both land on 2. Without
        episode-scoped state the card takes the counter>0 wording, whose "Just
        send your message again to continue." is the same redo invitation the
        continue rung removes one rung up — and the first turn of THIS episode
        already ran a tool whose side effects landed.
        """
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        _turns = {"n": 0}

        async def _stream(msg):
            _turns["n"] += 1
            if _turns["n"] == 1:
                # Productive: a tool ran, no closing reply. Takes the CONTINUE
                # rung and queues the activity continuation.
                yield LLMEvent(kind=EVENT_TOOL_CALL, tool_call_id="tc-1", title="send_message")
                yield LLMEvent(kind=EVENT_TOOL_RESULT, tool_call_id="tc-1", text="sent")
                yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)
            else:
                # The continuation returns nothing at all -> give-up.
                yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        await _run_chat(state, slot, "test message")
        # Let the real drain task run the queued continuation turn, which is
        # the turn that reaches give-up.
        for _bg_task in list(state._background_tasks):
            try:
                await _bg_task
            except Exception:
                pass
        await self._cancel_background_tasks(state)

        assert _turns["n"] >= 2, "the queued continuation never ran"
        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        giveup = [
            m
            for m in notice_msgs
            if "will not re-run" in m.get("content", "")
            or "returned nothing this turn" in m.get("content", "")
        ]
        assert giveup, "give-up produced no notice card"
        # The episode ran a tool: no card may invite the user to re-send.
        assert not any("send your message again" in m.get("content", "") for m in notice_msgs), (
            "the give-up card invites a resend, but the first turn of this "
            "episode ran a tool whose side effects already landed"
        )
        assert any("will not re-run" in m.get("content", "") for m in notice_msgs)
        # Terminal rung clears both the counter and the episode flag.
        assert slot._empty_response_retries == 0
        assert slot._empty_episode_productive is False

    @pytest.mark.asyncio
    async def test_discarded_episode_cannot_speak_for_a_later_turn(self, tmp_path: Path) -> None:
        """An episode whose continuation was thrown away must not lend its
        productive wording to the user's next, unrelated turn.

        Several controls discard a queued entry -- the hard-kill Stop, a plan
        Cancel, a rewind commit -- and none of them resets the recovery counter,
        so the spent counter routes the next empty turn straight to give-up. The
        flag is therefore read only on the ladder's OWN continuation: a discarded
        continuation takes that turn with it, so no per-site bookkeeping is
        needed and a control added later cannot reopen this.
        """
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        # What a productive CONTINUE rung leaves behind when its continuation
        # never runs: budget spent, flag set, no landed turn to clear either.
        slot._empty_response_retries = 2
        slot._empty_episode_productive = True
        self._make_empty_stream(client)

        # A genuine user turn, NOT the ladder's continuation.
        await _run_chat(state, slot, "a different request")
        await self._cancel_background_tasks(state)

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert notice_msgs, "give-up produced no notice card"
        assert not any("will not re-run" in m.get("content", "") for m in notice_msgs), (
            "a discarded episode's flag told an unrelated turn that its "
            "completed steps will not re-run, but this turn ran nothing"
        )
        assert any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        assert slot._empty_response_retries == 0
        assert slot._empty_episode_productive is False

    @pytest.mark.asyncio
    async def test_user_typed_continuation_text_ends_the_episode(self, tmp_path: Path) -> None:
        """The continuation bodies are fixed runner-authored strings, so a user
        can type or paste one. Such a turn is still user speech: it ends the
        episode rather than continuing it.

        User speech carries no structural payload marker, which is what separates
        it from the runner's own recovery dispatches. Keying the episode on the
        bodies instead would hand a user-authored turn the productive wording for
        work it never ran.
        """
        from kiro_crew.dashboard.chat_utils import _ACTIVITY_NO_REPLY_CONTINUE_MSG

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 2
        slot._empty_episode_productive = True
        self._make_empty_stream(client)

        # The exact continuation body, but dispatched as user speech: no
        # synthetic-recovery marker rides with it.
        await _run_chat(state, slot, _ACTIVITY_NO_REPLY_CONTINUE_MSG)
        await self._cancel_background_tasks(state)

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert notice_msgs, "give-up produced no notice card"
        assert not any("will not re-run" in m.get("content", "") for m in notice_msgs), (
            "a user-authored turn matching the continuation body was read as "
            "the ladder's own continuation, so a stale flag spoke for it"
        )
        assert any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)

    @pytest.mark.asyncio
    async def test_a_runner_queued_continuation_keeps_the_productive_wording(
        self, tmp_path: Path
    ) -> None:
        """A runner-authored continuation from ANY family, running inside an
        active productive episode, keeps the no-rerun wording.

        The runner can put another recovery ahead of the ladder's own: a Stop
        hook answering `decision: block` prepends at index 0 in the same teardown
        that just queued the ladder's continuation there, so it dispatches first.
        It is the same episode -- the tools of its productive first turn have
        already landed -- so if it also returns nothing, its give-up must not
        invite a resend. Keying the episode on the two continuation bodies would
        fail exactly here, which is why it is keyed on user origin instead.
        """
        from kiro_crew.dashboard.chat_utils import _PROMISE_ONLY_CONTINUE_MSG

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        # Mid-episode: a productive turn has spent the budget and set the flag.
        slot._empty_response_retries = 2
        slot._empty_episode_productive = True
        self._make_empty_stream(client)

        # Another family's continuation: runner-authored, so it carries the
        # structural payload marker, and its body is not the ladder's.
        await _run_chat(
            state,
            slot,
            _PROMISE_ONLY_CONTINUE_MSG,
            _synthetic_payload=True,
            _synthetic_recovery_turn=True,
        )
        await self._cancel_background_tasks(state)

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert notice_msgs, "give-up produced no notice card"
        assert not any("send your message again" in m.get("content", "") for m in notice_msgs), (
            "a runner-queued continuation inside a productive episode was told "
            "to resend, but that episode's first turn already ran a tool"
        )
        assert any("will not re-run" in m.get("content", "") for m in notice_msgs)

    @pytest.mark.asyncio
    async def test_discarded_episode_cannot_ride_a_later_turns_own_continuation(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """With a continue budget above one, a later turn earns a continuation of
        its own -- and that turn must not inherit a discarded episode's wording.

        This is the case a guard at the give-up rung cannot catch. A productive
        turn leaves the counter at 2 and the flag set; its continuation is
        discarded, so nothing lands and neither resets. With a budget of 2 the
        next unrelated empty turn still passes rung 2 (`2 < 1 + 2`) and queues its
        OWN continuation, which genuinely IS one of the ladder's bodies carrying
        the payload marker. The evidence therefore has to be dropped when the
        episode ends, not merely disbelieved when it is read.
        """
        from kiro_crew.dashboard import chat_runner as cr

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        monkeypatch.setattr(cr, "_empty_max_auto_continues", lambda: 2)
        # What a productive rung-2 turn leaves behind when its continuation is
        # thrown away: budget spent, flag set, no landed turn.
        slot._empty_response_retries = 2
        slot._empty_episode_productive = True
        self._make_empty_stream(client)

        # A genuine user turn, which under this budget gets its own continuation.
        await _run_chat(state, slot, "a different request")
        for _bg_task in list(state._background_tasks):
            try:
                await _bg_task
            except Exception:
                pass
        await self._cancel_background_tasks(state)

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert notice_msgs, "give-up produced no notice card"
        assert not any("will not re-run" in m.get("content", "") for m in notice_msgs), (
            "a discarded episode's flag rode a later turn's own ladder "
            "continuation and claimed its completed steps will not re-run"
        )

    @pytest.mark.asyncio
    async def test_productive_episode_survives_a_second_continuation(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A productive episode keeps its wording across EVERY continuation it is
        budgeted for, not just the first.

        With a budget of 2 the episode runs two continuations: the first carries
        `_ACTIVITY_NO_REPLY_CONTINUE_MSG`, the second `_EMPTY_AUTO_CONTINUE_MSG`
        (the non-productive branch queues that body once the continuation itself
        streams nothing). Both are this ladder's own, so both must preserve the
        episode's evidence -- if either body is missing from the identity pair,
        that turn looks like an unrelated one, the episode-end clear fires, and
        give-up falls back to inviting a resend of tools that already ran.
        """
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.dashboard import chat_runner as cr
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        monkeypatch.setattr(cr, "_empty_max_auto_continues", lambda: 2)

        _turns = {"n": 0}

        async def _stream(msg):
            _turns["n"] += 1
            if _turns["n"] == 1:
                # Productive: a tool ran, no closing reply.
                yield LLMEvent(kind=EVENT_TOOL_CALL, tool_call_id="tc-1", title="send_message")
                yield LLMEvent(kind=EVENT_TOOL_RESULT, tool_call_id="tc-1", text="sent")
                yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)
            else:
                # Both continuations return nothing, walking the ladder to give-up.
                yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

        await _run_chat(state, slot, "test message")
        # Drain repeatedly: each continuation queues the next one.
        for _round in range(4):
            for _bg_task in list(state._background_tasks):
                try:
                    await _bg_task
                except Exception:
                    pass
        await self._cancel_background_tasks(state)

        assert _turns["n"] >= 3, f"the second continuation never ran (turns={_turns['n']})"
        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert not any("send your message again" in m.get("content", "") for m in notice_msgs), (
            "give-up invited a resend after a productive episode spent its full "
            "continuation budget, so the episode's evidence was lost partway"
        )
        assert any("will not re-run" in m.get("content", "") for m in notice_msgs)

    @pytest.mark.asyncio
    async def test_giveup_without_recoveries_drops_retry_claim(self, tmp_path: Path) -> None:
        """A give-up reached with the recovery counter still at zero (here: a
        nested depth>0 turn) must not assert that a retry and an auto-continue
        ran — neither did. The counter is the evidence, not the rung."""
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._make_empty_stream(client)

        await _run_chat(state, slot, "test message", _prompt_depth=1)

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        assert not any("retried" in m.get("content", "") for m in notice_msgs), (
            "the give-up card claims a retry ran, but slot._empty_response_retries "
            "is still 0 on this path"
        )
        assert not any("auto-continued" in m.get("content", "") for m in notice_msgs)
        assert not any("recovery was attempted" in m.get("content", "") for m in notice_msgs)

    @pytest.mark.asyncio
    async def test_giveup_after_spent_recoveries_reports_recovery(self, tmp_path: Path) -> None:
        """When the recovery budget was actually spent (counter exhausted, turn
        not productive), the card reports that automatic recovery was
        attempted. Phrased phase-neutrally on purpose: the counter counts
        budget spent, not which rungs ran."""
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 2  # re-queue + nudge both spent
        self._make_empty_stream(client)

        await _run_chat(state, slot, "test message")

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("automatic recovery was attempted" in m.get("content", "") for m in notice_msgs)
        # The counter alone cannot prove WHICH rungs ran (a productive turn's
        # continuation arrives here at 2 with no verbatim retry), so the card
        # must not name specific recovery phases.
        assert not any("auto-continued automatically" in m.get("content", "") for m in notice_msgs)

    @pytest.mark.asyncio
    async def test_second_empty_flag_off_shows_notice(self, tmp_path: Path) -> None:
        """With session.empty_response_auto_continue disabled, the second empty
        surfaces the terminal notice immediately (pre-feature behavior)."""
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 1
        self._make_empty_stream(client)

        # Exercise the REAL config path (loader wiring included): persist the
        # flag as false in a config file and point KiroCrewConfig.load() at it,
        # rather than patching the gate function (which would pass even if the
        # loader dropped the field — the exact regression this test guards).
        cfg_file = tmp_path / "flag-off-config.json"
        cfg_file.write_text('{"session": {"empty_response_auto_continue": false}}')
        with patch("kiro_crew.config.loader.config_path", return_value=cfg_file):
            await _run_chat(state, slot, "test message")

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        # Counter is 1 here: ONLY the silent verbatim re-queue ran (rung 1's
        # guard ignores the gate); the auto-continue was forbidden by config.
        # The card reports budget spent phase-neutrally and must not assert
        # the auto-continue that provably did not run. This is also the
        # boundary case for the counter gate: budget WAS spent, so the bare
        # zero-recoveries wording would be wrong too.
        assert any("automatic recovery was attempted" in m.get("content", "") for m in notice_msgs)
        assert not any("auto-continued" in m.get("content", "") for m in notice_msgs)
        assert slot._empty_response_retries == 0

    @pytest.mark.asyncio
    async def test_max_continues_config_extends_the_ladder(self, tmp_path: Path) -> None:
        """With session.empty_response_max_continues=3, the second consecutive
        continuation failure keeps continuing — numbered so the transcript
        reads as advancing, not looping — instead of giving up after one.
        Exercises the REAL config path like the flag-off test above, so a
        loader that drops the field fails this test too."""
        from kiro_crew.dashboard.chat_runner import _EMPTY_AUTO_CONTINUE_MSG

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 2  # replay + first continue spent
        self._make_empty_stream(client)

        calls = []
        orig = _ChatSlot.queue_insert

        def spy(self_slot, *a, **kw):
            calls.append(a)
            return orig(self_slot, *a, **kw)

        cfg_file = tmp_path / "max-continues-config.json"
        cfg_file.write_text('{"session": {"empty_response_max_continues": 3}}')
        with (
            patch("kiro_crew.config.loader.config_path", return_value=cfg_file),
            patch.object(_ChatSlot, "queue_insert", spy),
        ):
            await _run_chat(state, slot, "test message")
            await self._cancel_background_tasks(state)

        assert (0, _EMPTY_AUTO_CONTINUE_MSG) in calls
        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("(recovery 2 of 3)" in m.get("content", "") for m in notice_msgs)
        assert slot._empty_response_retries == 3

    @pytest.mark.asyncio
    async def test_max_continues_budget_still_terminates(self, tmp_path: Path) -> None:
        """The raised budget is still a budget: with max=3 and the replay plus
        all three continues spent, the give-up card fires and the counter
        resets for the next independent user turn."""
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 4  # replay + 3 continues spent
        self._make_empty_stream(client)

        cfg_file = tmp_path / "max-continues-config.json"
        cfg_file.write_text('{"session": {"empty_response_max_continues": 3}}')
        with patch("kiro_crew.config.loader.config_path", return_value=cfg_file):
            await _run_chat(state, slot, "test message")

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        assert slot._empty_response_retries == 0

    @pytest.mark.asyncio
    async def test_successful_response_resets_counter(self, tmp_path: Path) -> None:
        """A successful (non-empty) response resets the retry counter."""
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        slot._empty_response_retries = 1  # had a prior empty

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="Hello!")
            yield LLMEvent(kind=EVENT_COMPLETE)

        client.stream = _stream
        client.stream_command = _stream

        await _run_chat(state, slot, "test message")

        assert slot._empty_response_retries == 0

    @pytest.mark.asyncio
    async def test_compaction_turn_no_empty_response_error(self, tmp_path: Path) -> None:
        """Compaction turns set assistant_text='' but should NOT trigger the empty-response notice."""
        from kiro_crew.providers.base import EVENT_COMPACTION_STATUS, EVENT_COMPLETE, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_COMPACTION_STATUS, text="completed", title="summary")
            yield LLMEvent(kind=EVENT_COMPLETE)

        client.stream = _stream
        client.stream_command = _stream

        await _run_chat(state, slot, "/compact")

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert not any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)

    @pytest.mark.asyncio
    async def test_clear_turn_no_empty_response_error(self, tmp_path: Path) -> None:
        """Clear turns set assistant_text='' but should NOT trigger the empty-response notice."""
        from kiro_crew.providers.base import EVENT_CLEAR_STATUS, EVENT_COMPLETE, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        state.sessions.provider_switch_replay_pending = MagicMock(return_value=True)
        state.sessions.consume_provider_switch_replay = MagicMock(return_value=True)
        state.sessions.mark_provider_switch_replay = MagicMock(return_value=True)
        state.sessions.commit_provider_switch_replay_sid = MagicMock(return_value=True)

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_CLEAR_STATUS)
            yield LLMEvent(kind=EVENT_COMPLETE)

        client.stream = _stream
        client.stream_command = _stream

        await _run_chat(state, slot, "/clear")

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert not any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)
        state.sessions.consume_provider_switch_replay.assert_called_once_with(
            f"dashboard:{slot.key}"
        )
        state.sessions.commit_provider_switch_replay_sid.assert_called_once_with(
            f"dashboard:{slot.key}"
        )
        state.sessions.mark_provider_switch_replay.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_switch_turn_no_empty_response_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Agent switch turns set assistant_text='' but should NOT trigger the empty-response notice."""
        from kiro_crew.providers.base import EVENT_AGENT_SWITCHED, EVENT_COMPLETE, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        monkeypatch.setattr(
            "kiro_crew.config.loader._materialized_kiro_agent",
            lambda name, project_dir=None: name if name == "new-agent" else "",
        )

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_AGENT_SWITCHED, text="new-agent")
            yield LLMEvent(kind=EVENT_COMPLETE)

        client.stream = _stream
        client.stream_command = _stream

        await _run_chat(state, slot, "test message")

        notice_msgs = [m for m in slot.messages if m.get("role") == "notice"]
        assert not any("returned nothing this turn" in m.get("content", "") for m in notice_msgs)


class TestProductiveTurnNeverReplaysVerbatim:
    """A turn that DID work must never have the user's message replayed verbatim.

    ``assistant_text`` is reset at every tool boundary, so a turn that streamed a
    preamble and then called a tool arrives at the terminal chain with an empty
    final segment and takes the empty-response branch. Rung 1 of that ladder
    re-queues the ORIGINAL message, which re-runs every tool call that already
    completed — a second ``send_message``, a second write, a second PR — and
    re-derives an answer the user has already read.

    The field incident these tests pin: two consecutive billed turns, each with an
    assistant preamble and successful tool calls and each ending on a clean
    ``end_turn``, were both classified empty; the first verbatim-replayed the
    user's message.

    Reuses ``TestEmptyResponseRetry``'s harness rather than a second one, so a
    change to how a slot is built cannot leave these tests driving a different
    runner than the ladder tests beside them.
    """

    _make_state_and_slot = TestEmptyResponseRetry._make_state_and_slot

    @staticmethod
    def _spy_queue(monkeypatch=None):
        """Record ``(index, content)`` for every queue insert."""
        calls: list[tuple] = []
        orig = _ChatSlot.queue_insert

        def spy(self_slot, *a, **kw):
            calls.append(a)
            return orig(self_slot, *a, **kw)

        return calls, spy

    _cancel_background_tasks = staticmethod(TestEmptyResponseRetry._cancel_background_tasks)

    @staticmethod
    def _text_then_tool_stream(client, executed):
        """TEXT -> TOOL_CALL -> TOOL_RESULT -> COMPLETE(end_turn).

        ``executed`` counts dispatches so a replay of the original message would
        be observable as a second execution rather than only as a queue entry.
        """
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="checking")
            executed.append(msg)
            yield LLMEvent(
                kind=EVENT_TOOL_CALL,
                tool_call_id="tc-1",
                title="send_message",
            )
            yield LLMEvent(kind=EVENT_TOOL_RESULT, tool_call_id="tc-1", text="sent")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

    @staticmethod
    def _tool_only_stream(client):
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TOOL_CALL,
            EVENT_TOOL_RESULT,
            LLMEvent,
        )

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TOOL_CALL, tool_call_id="tc-1", title="send_message")
            yield LLMEvent(kind=EVENT_TOOL_RESULT, tool_call_id="tc-1", text="sent")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

    @staticmethod
    def _thinking_only_stream(client):
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_THINKING_CHUNK, LLMEvent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_THINKING_CHUNK, text="hmm")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _stream
        client.stream_command = _stream

    @pytest.mark.asyncio
    async def test_text_then_tool_turn_continues_instead_of_replaying(self, tmp_path: Path) -> None:
        """The incident's own shape: preamble + a completed tool call, clean
        end_turn. The original message must NOT be re-queued; exactly one
        synthetic continuation may be, and the completed tool must not re-run."""
        from kiro_crew.dashboard.chat_utils import (
            _ACTIVITY_NO_REPLY_CONTINUE_MSG,
            _EMPTY_AUTO_CONTINUE_MSG,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        executed: list[str] = []
        self._text_then_tool_stream(client, executed)
        calls, spy = self._spy_queue()

        with patch.object(_ChatSlot, "queue_insert", spy):
            await _run_chat(state, slot, "send the summary")
            await self._cancel_background_tasks(state)

        # The verbatim replay is the defect. It must be absent.
        assert (0, "send the summary") not in calls, (
            "the ORIGINAL user message was re-queued after a turn that already "
            "ran a tool — the replay re-executes completed side effects"
        )
        # Exactly one continuation, and it is the one whose body does not claim
        # the turn produced nothing (which would invite the model to redo the
        # completed call).
        _continuations = [c for c in calls if c and c[1] == _ACTIVITY_NO_REPLY_CONTINUE_MSG]
        assert len(_continuations) == 1, f"expected one continuation, got {calls}"
        # The productive turn skipped rung 1 (verbatim replay), so this one
        # continuation consumes the whole bounded ladder. If the continuation
        # itself returns empty, it gives up rather than enqueueing a SECOND
        # continuation whose wording would again invite rework.
        assert slot._empty_response_retries == 2
        assert (0, _EMPTY_AUTO_CONTINUE_MSG) not in calls, (
            "the empty-response body tells the model its turn produced no "
            "output, which is false for a turn that ran a tool"
        )
        # The tool ran once. The queue was not drained in this test, so a second
        # execution could only come from a replay dispatched inside this turn.
        assert executed == ["send the summary"], f"tool dispatched more than once: {executed}"

    @pytest.mark.asyncio
    async def test_tool_only_turn_does_not_replay_verbatim(self, tmp_path: Path) -> None:
        """A turn with tool calls and no text at all still counts as productive."""
        from kiro_crew.dashboard.chat_utils import _ACTIVITY_NO_REPLY_CONTINUE_MSG

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._tool_only_stream(client)
        calls, spy = self._spy_queue()

        with patch.object(_ChatSlot, "queue_insert", spy):
            await _run_chat(state, slot, "post it")
            await self._cancel_background_tasks(state)

        assert (0, "post it") not in calls, "tool-only turn verbatim-replayed the prompt"
        assert (0, _ACTIVITY_NO_REPLY_CONTINUE_MSG) in calls

    @pytest.mark.asyncio
    async def test_thinking_only_turn_does_not_replay_verbatim(self, tmp_path: Path) -> None:
        """Thinking is work the replay would discard and re-charge for."""
        from kiro_crew.dashboard.chat_utils import _ACTIVITY_NO_REPLY_CONTINUE_MSG

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        self._thinking_only_stream(client)
        calls, spy = self._spy_queue()

        with patch.object(_ChatSlot, "queue_insert", spy):
            await _run_chat(state, slot, "think about it")
            await self._cancel_background_tasks(state)

        assert (0, "think about it") not in calls, "thinking-only turn verbatim-replayed the prompt"
        assert (0, _ACTIVITY_NO_REPLY_CONTINUE_MSG) in calls

    @pytest.mark.asyncio
    async def test_activity_free_turn_keeps_the_verbatim_replay_rung(self, tmp_path: Path) -> None:
        """The guard is scoped: a GENUINELY activity-free empty turn keeps rung 1.

        Without this, the fix could be 'never replay', which would silently
        remove the self-heal that a real provider-side empty depends on.
        """
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        TestEmptyResponseRetry._make_empty_stream(self, client)
        calls, spy = self._spy_queue()

        with (
            patch.object(_ChatSlot, "queue_insert", spy),
            patch(
                "kiro_crew.dashboard.chat_runner._start_next_queued_turn",
                new=AsyncMock(return_value=False),
            ),
        ):
            await _run_chat(state, slot, "test message")
            await self._cancel_background_tasks(state)

        assert (0, "test message") in calls, "the activity-free replay rung was removed"
        assert slot._empty_response_retries == 1


class TestEmptyTurnDiagnostics:
    """The empty-response WARNING must name a CLOSED cause, and carry no content.

    The incident it exists for had three attempts with three different causes and
    one log line that could not tell them apart. These tests assert on the closed
    vocabulary only — never on a prompt, a response, a tool argument, a path, an
    identity, a token count or a cost, because asserting on those would pin
    exactly the leak the diagnostics are designed to avoid.
    """

    _make_state_and_slot = TestEmptyResponseRetry._make_state_and_slot
    _cancel_background_tasks = staticmethod(TestEmptyResponseRetry._cancel_background_tasks)

    @staticmethod
    def _causes(caplog):
        """The ``cause=`` values of every empty-response warning captured."""
        out = []
        for rec in caplog.records:
            msg = rec.getMessage()
            if "Empty model response" not in msg:
                continue
            for field in msg.split():
                if field.startswith("cause="):
                    out.append(field.split("=", 1)[1])
        return out

    @staticmethod
    def _fields(caplog):
        """``key=value`` pairs of the first empty-response warning captured."""
        for rec in caplog.records:
            msg = rec.getMessage()
            if "Empty model response" not in msg:
                continue
            return dict(
                tuple(f.split("=", 1)) for f in msg.split() if "=" in f and not f.startswith("%")
            )
        return {}

    def test_the_cause_and_rung_vocabularies_are_closed(self) -> None:
        """Pure-unit: every cause the classifier can return, from its own inputs.

        Kept as a unit test over ``classify_empty_turn`` rather than seven
        ``_run_chat`` drives: the ranking between overlapping causes is the part
        that can regress silently, and a stream fixture cannot express
        'synthesized terminal AND tool activity' as cleanly as the snapshot can.
        """
        from kiro_crew.dashboard.chat_utils import (
            EMPTY_CAUSE_NO_TERMINAL,
            EMPTY_CAUSE_OTHER,
            EMPTY_CAUSE_PROVIDER_EMPTY,
            EMPTY_CAUSE_SYNTHETIC,
            EMPTY_CAUSE_THINKING_ONLY,
            EMPTY_CAUSE_TOOL_ONLY,
            EMPTY_CAUSE_VISIBLE_PARTIAL,
            STOP_REASON_ABSENT,
            EmptyTurnActivity,
            classify_empty_turn,
        )

        # No terminal outranks everything, including activity.
        assert (
            classify_empty_turn(EmptyTurnActivity(saw_terminal=False, had_tools=True))
            == EMPTY_CAUSE_NO_TERMINAL
        )
        # A synthesized terminal outranks activity: the emptiness is ours.
        assert (
            classify_empty_turn(
                EmptyTurnActivity(saw_terminal=True, terminal_synthetic=True, had_tools=True)
            )
            == EMPTY_CAUSE_SYNTHETIC
        )
        # A flushed visible segment outranks tools: the user read an answer.
        assert (
            classify_empty_turn(
                EmptyTurnActivity(saw_terminal=True, flushed_visible=True, had_tools=True)
            )
            == EMPTY_CAUSE_VISIBLE_PARTIAL
        )
        assert (
            classify_empty_turn(EmptyTurnActivity(saw_terminal=True, had_tools=True))
            == EMPTY_CAUSE_TOOL_ONLY
        )
        assert (
            classify_empty_turn(EmptyTurnActivity(saw_terminal=True, had_thinking=True))
            == EMPTY_CAUSE_THINKING_ONLY
        )
        # A clean end_turn with nothing at all is the only genuine provider empty.
        assert (
            classify_empty_turn(EmptyTurnActivity(saw_terminal=True, stop_reason="end_turn"))
            == EMPTY_CAUSE_PROVIDER_EMPTY
        )
        # An OMITTED stop reason is NOT a provider empty — the distinction the
        # incident's third attempt needed and did not have.
        assert (
            classify_empty_turn(
                EmptyTurnActivity(saw_terminal=True, stop_reason=STOP_REASON_ABSENT)
            )
            == EMPTY_CAUSE_OTHER
        )

    def test_normalize_stop_reason_never_returns_a_raw_backend_string(self) -> None:
        """Closed output set, and the ACP spellings it mirrors are pinned.

        ``chat_utils`` may not import ``kiro_crew.acp.types`` (the agent-SDK
        boundary gate baselines it at one edge and a baselined file may not grow),
        so it spells the three stop reasons as literals. The test tree is outside
        that gate, so the pin lives here — a change to the backend's vocabulary
        reddens here instead of silently reclassifying every turn as ``other``.
        """
        from kiro_crew.acp.types import (
            STOP_REASON_CANCELLED,
            STOP_REASON_END_TURN,
            STOP_REASON_REFUSAL,
        )
        from kiro_crew.dashboard.chat_utils import (
            _STOP_CANCELLED_REASON,
            _STOP_END_TURN,
            _STOP_REFUSAL,
            STOP_REASON_ABSENT,
            STOP_REASON_OTHER,
            normalize_stop_reason,
        )

        assert _STOP_END_TURN == STOP_REASON_END_TURN
        assert _STOP_CANCELLED_REASON == STOP_REASON_CANCELLED
        assert _STOP_REFUSAL == STOP_REASON_REFUSAL

        assert normalize_stop_reason(None) == STOP_REASON_ABSENT
        assert normalize_stop_reason("") == STOP_REASON_ABSENT
        assert normalize_stop_reason(STOP_REASON_END_TURN) == STOP_REASON_END_TURN
        assert normalize_stop_reason("error: tool stall") == "error"
        # An invented backend reason is folded, never echoed.
        _invented = "the model said stop because of /home/someone/secret.txt"
        assert normalize_stop_reason(_invented) == STOP_REASON_OTHER

    @pytest.mark.asyncio
    async def test_provider_empty_and_no_terminal_event_report_distinct_causes(
        self, tmp_path: Path, caplog
    ) -> None:
        """Two turns that look identical to the old log line must not any more."""
        from kiro_crew.acp.types import STOP_REASON_END_TURN
        from kiro_crew.dashboard.chat_utils import (
            EMPTY_CAUSE_NO_TERMINAL,
            EMPTY_CAUSE_PROVIDER_EMPTY,
        )
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        # A clean end_turn with nothing in it.
        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _clean(msg):
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)

        client.stream = _clean
        client.stream_command = _clean
        with caplog.at_level(logging.WARNING, logger="kiro_crew.dashboard.chat_runner"):
            await _run_chat(state, slot, "a", _prompt_depth=1)
        await self._cancel_background_tasks(state)
        assert self._causes(caplog) == [EMPTY_CAUSE_PROVIDER_EMPTY]

        # A stream that ends without any terminal event at all.
        caplog.clear()
        state2, slot2, client2, _run_chat2 = self._make_state_and_slot(tmp_path)

        async def _no_terminal(msg):
            return
            yield  # pragma: no cover -- makes this an async generator

        client2.stream = _no_terminal
        client2.stream_command = _no_terminal
        with caplog.at_level(logging.WARNING, logger="kiro_crew.dashboard.chat_runner"):
            await _run_chat2(state2, slot2, "a", _prompt_depth=1)
        await self._cancel_background_tasks(state2)
        assert self._causes(caplog) == [EMPTY_CAUSE_NO_TERMINAL]

    @pytest.mark.asyncio
    async def test_omitted_stop_reason_is_not_reported_as_provider_empty(
        self, tmp_path: Path, caplog
    ) -> None:
        """A terminal with no stopReason is its own observation."""
        from kiro_crew.dashboard.chat_utils import (
            EMPTY_CAUSE_OTHER,
            STOP_REASON_ABSENT,
        )
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_COMPLETE)  # no stop_reason

        client.stream = _stream
        client.stream_command = _stream
        with caplog.at_level(logging.WARNING, logger="kiro_crew.dashboard.chat_runner"):
            await _run_chat(state, slot, "a", _prompt_depth=1)
        await self._cancel_background_tasks(state)

        assert self._causes(caplog) == [EMPTY_CAUSE_OTHER]
        assert self._fields(caplog).get("stop_reason") == STOP_REASON_ABSENT

    @pytest.mark.asyncio
    async def test_productive_turn_reports_visible_partial_and_the_continue_rung(
        self, tmp_path: Path, caplog
    ) -> None:
        """The incident's shape gets its own cause AND the rung it took.

        Asserting the rung is what makes the diagnostic answer 'what did the
        runner DO', not only 'what did it see'.
        """
        from kiro_crew.dashboard.chat_utils import (
            EMPTY_CAUSE_VISIBLE_PARTIAL,
            EMPTY_RUNG_CONTINUE,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)
        TestProductiveTurnNeverReplaysVerbatim._text_then_tool_stream(client, [])

        with caplog.at_level(logging.WARNING, logger="kiro_crew.dashboard.chat_runner"):
            await _run_chat(state, slot, "send the summary")
            await self._cancel_background_tasks(state)

        assert self._causes(caplog) == [EMPTY_CAUSE_VISIBLE_PARTIAL]
        _fields = self._fields(caplog)
        assert _fields.get("rung") == EMPTY_RUNG_CONTINUE
        assert _fields.get("flushed_visible") == "True"
        assert _fields.get("tools") == "True"

    @pytest.mark.asyncio
    async def test_the_warning_carries_no_content_and_no_billing_amounts(
        self, tmp_path: Path, caplog
    ) -> None:
        """The privacy floor, asserted against the rendered line.

        Every diagnostic value is a bool or a closed constant, so the prompt text,
        the streamed text, the tool title and the billed amounts must be absent
        from the line even though the turn carried all of them.
        """
        from kiro_crew.acp.types import STOP_REASON_END_TURN, TurnUsage
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            LLMEvent,
        )

        state, slot, client, _run_chat = self._make_state_and_slot(tmp_path)

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="SECRETPREAMBLE")
            yield LLMEvent(kind=EVENT_TOOL_CALL, tool_call_id="tc-1", title="SECRETTOOL")
            yield LLMEvent(
                kind=EVENT_COMPLETE,
                stop_reason=STOP_REASON_END_TURN,
                usage=TurnUsage(credits=4.25, input_tokens=1234, output_tokens=99),
            )

        client.stream = _stream
        client.stream_command = _stream
        with caplog.at_level(logging.WARNING, logger="kiro_crew.dashboard.chat_runner"):
            await _run_chat(state, slot, "SECRETPROMPT")
            await self._cancel_background_tasks(state)

        _line = next(
            rec.getMessage() for rec in caplog.records if "Empty model response" in rec.getMessage()
        )
        for forbidden in ("SECRETPROMPT", "SECRETPREAMBLE", "SECRETTOOL", "4.25", "1234", "99"):
            assert forbidden not in _line, f"{forbidden!r} leaked into {_line!r}"
        # Billing presence IS reported — as a bool, which is the point.
        assert "billed=True" in _line


class TestExpandDollarSkills:
    """Runner-side ``_expand_dollar_skills``: redaction, chip,
    SEL audit, and empty/exception branches. The pure resolution logic is
    covered separately by test_skills.TestResolveDollarSkills; here we
    exercise the chat_runner wrapper that adds runner concerns."""

    def _make_skill(self, skills_dir: Path, name: str, body: str) -> None:
        d = skills_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(body, encoding="utf-8")

    def _state_with_skills(self, tmp_path, monkeypatch, *skills):
        """A DashboardState whose _get_skills() returns a hermetic loader
        pointed at *skills* (each a ``(name, body)`` pair)."""
        # Keep edition-contributed skill roots out of the hermetic loader.
        from kiro_crew.platform.defaults import DefaultMcpToolingProvider
        from kiro_crew.skills import SkillsLoader

        monkeypatch.setattr(DefaultMcpToolingProvider, "extra_skills", lambda self: [])
        skills_dir = tmp_path / "skills"
        for name, body in skills:
            self._make_skill(skills_dir, name, body)
        loader = SkillsLoader(skills_path=skills_dir, install_builtins=False)

        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        state.context_builder = None
        state._standalone_skills = loader  # _get_skills returns this verbatim
        return state

    def test_no_dollar_short_circuits(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat_runner import _expand_dollar_skills

        state = self._state_with_skills(tmp_path, monkeypatch)
        slot = _ChatSlot("s1")
        out, n = _expand_dollar_skills("plain message", state, slot, "sess")
        assert out == "plain message"
        assert n == 0
        state.push_slots_update.assert_not_called()

    def test_resolves_appends_block_and_chip(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat_runner import _expand_dollar_skills

        state = self._state_with_skills(
            tmp_path,
            monkeypatch,
            ("oncall-handover", "---\nname: oncall-handover\n---\n# Handover\nBODY-A"),
        )
        slot = _ChatSlot("s1")
        out, n = _expand_dollar_skills("please $oncall-handover now", state, slot, "sess")
        assert n == 1
        # original message preserved, skill body appended as a [Skill: name] block
        assert out.startswith("please $oncall-handover now")
        assert "[Skill: oncall-handover]" in out
        assert "BODY-A" in out
        # a system chip is appended and the UI is poked
        notice = next(
            m
            for m in slot.messages
            if "Loaded skill(s)" in m.get("content", "") and m.get("role") == "system"
        )
        assert notice["meta"]["kind"] == "skill_load"
        assert notice["meta"]["skills"] == [
            {"name": "oncall-handover", "body": "# Handover\nBODY-A"}
        ]
        state.push_slots_update.assert_called_once()

    def test_skill_notice_preserves_multiple_resolved_bodies_in_order(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat_runner import _expand_dollar_skills

        state = self._state_with_skills(
            tmp_path,
            monkeypatch,
            ("first", "---\nname: first\n---\n# First\nONE"),
            ("second", "---\nname: second\n---\n# Second\nTWO"),
        )
        slot = _ChatSlot("s1")

        out, n = _expand_dollar_skills("use $first then $second", state, slot, "sess")

        assert n == 2
        assert out.index("[Skill: first]") < out.index("[Skill: second]")
        notice = next(m for m in slot.messages if m.get("role") == "system")
        assert notice["meta"] == {
            "kind": "skill_load",
            "skills": [
                {"name": "first", "body": "# First\nONE"},
                {"name": "second", "body": "# Second\nTWO"},
            ],
            "mid": notice["meta"]["mid"],
        }

    def test_unresolved_token_no_chip(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import chat_runner

        state = self._state_with_skills(
            tmp_path,
            monkeypatch,
            ("oncall-handover", "---\nname: oncall-handover\n---\nBODY"),
        )
        slot = _ChatSlot("s1")
        sel_mock = MagicMock()
        monkeypatch.setattr(chat_runner, "sel", lambda: sel_mock)

        out, n = chat_runner._expand_dollar_skills("$does-not-exist hi", state, slot, "sess")
        assert (out, n) == ("$does-not-exist hi", 0)
        state.push_slots_update.assert_not_called()
        # a real $skill-shaped token that didn't resolve IS audited as not_found
        assert sel_mock.log_tool_invocation.called
        _, kwargs = sel_mock.log_tool_invocation.call_args
        assert kwargs.get("outcome") == "not_found"
        assert kwargs.get("tool_name") == "skill_dollar_expansion"

    def test_incidental_dollar_not_audited(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import chat_runner

        state = self._state_with_skills(
            tmp_path,
            monkeypatch,
            ("oncall-handover", "---\nname: oncall-handover\n---\nBODY"),
        )
        slot = _ChatSlot("s1")
        sel_mock = MagicMock()
        monkeypatch.setattr(chat_runner, "sel", lambda: sel_mock)

        # $5 / $PATH / bare $ are NOT skill-shaped → no not_found noise.
        out, n = chat_runner._expand_dollar_skills("it costs $5 not $PATH", state, slot, "sess")
        assert (out, n) == ("it costs $5 not $PATH", 0)
        sel_mock.log_tool_invocation.assert_not_called()

    def test_credentials_redacted_in_loaded_body(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat_runner import _expand_dollar_skills

        secret = "AKIAIOSFODNN7EXAMPLE"
        state = self._state_with_skills(
            tmp_path,
            monkeypatch,
            ("leaky", f"---\nname: leaky\n---\n# Leaky\naws_key={secret}"),
        )
        slot = _ChatSlot("s1")
        out, n = _expand_dollar_skills("use $leaky", state, slot, "sess")
        assert n == 1
        # the raw credential must not survive into the expanded message
        assert secret not in out

    def test_resolution_exception_audited_and_swallowed(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import chat_runner

        state = self._state_with_skills(tmp_path, monkeypatch)
        slot = _ChatSlot("s1")

        boom = MagicMock()
        boom.resolve_dollar_skills.side_effect = RuntimeError("kaboom")
        # Force _get_skills(state) to return the exploding loader.
        monkeypatch.setattr(chat_runner, "_get_skills", lambda _s: boom)

        sel_mock = MagicMock()
        monkeypatch.setattr(chat_runner, "sel", lambda: sel_mock)

        out, n = chat_runner._expand_dollar_skills("try $whatever", state, slot, "sess")
        # message returned unchanged, no crash
        assert (out, n) == ("try $whatever", 0)
        # the failure was audited via SEL with outcome="error"
        assert sel_mock.log_tool_invocation.called
        _, kwargs = sel_mock.log_tool_invocation.call_args
        assert kwargs.get("outcome") == "error"
        assert kwargs.get("tool_name") == "skill_dollar_expansion"


# ── Transient backend 5xx retry on the interactive chat path ──


class TestRunChatTransientRetry:
    """The interactive _run_chat stream loop reuses the llm_helpers transient
    classifier + backoff to retry a transient
    backend 5xx WITHOUT resetting the live session, guarded so a partial
    (already-streamed) response is never re-run."""

    _TRANSIENT = "Prompt error: {'message': 'Internal error: API Error: Internal server error'}"
    _AUTH = "Bedrock authentication failed. Run 'ada credentials update'"

    @staticmethod
    def _make_state(tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.broadcast_ws = MagicMock()
        state.push_slots_update = MagicMock()
        state.push_refresh = MagicMock()
        state.context_builder = None
        state.consolidator = None
        state._hook_store = None
        state._yolo = False
        return state

    @staticmethod
    def _wire_sessions(state, client):
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        state.sessions.get_pid = MagicMock(return_value=None)
        state.sessions.check_context_usage = MagicMock()
        state.sessions.record_success = MagicMock()
        state.sessions.record_failure = AsyncMock()
        state.sessions.release = MagicMock()
        # reset is an AsyncMock so we can assert it is NEVER awaited — a
        # transient 5xx must NOT reset the (still-alive) session.
        state.sessions.reset = AsyncMock()
        # discard_conversation is the poisoned-conversation escalation (clears
        # the resume sid, keeps the session-map entry with its channel
        # linkage); mocked so tests can assert exactly when it fires.
        state.sessions.discard_conversation = AsyncMock()
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))

    @staticmethod
    def _client(stream):
        client = _provider_mock()
        client.context_usage_pct = MagicMock(return_value=0.0)
        # These are sync accessors on the real provider client; _run_chat
        # calls them without awaiting, so a bare AsyncMock attribute here
        # would leave an unawaited coroutine per turn.
        client.context_window_tokens = MagicMock(return_value=0)
        client.context_used_tokens = MagicMock(return_value=0)
        client.mcp_session_report = MagicMock(return_value=None)
        client.pop_pending_oauth_requests = MagicMock(return_value=[])
        client.available_models = MagicMock(return_value=[])
        # _drain_session_init_oauth_requests reaches pop_pending_oauth_requests
        # via client.client (the nested ACP client), not the front client
        # itself; leaving client.client as a bare AsyncMock attribute would
        # make that nested lookup resolve to another never-awaited coroutine.
        client.client = None
        client.stream = stream
        client.stream_command = stream
        # The poisoned-conversation canary reads the session's served model
        # via the provider's PUBLIC served_model accessor — mock that public
        # seam, not the provider internals it happens to be backed by.
        client.served_model = "claude-test-model"
        return client

    @staticmethod
    async def _drain_bg(state, limit=30):
        """Drive the finally-block re-queue cascade (each retry spawns a
        background _run_chat task) to completion. Bounded by the retry budget."""
        for _ in range(limit):
            pending = [t for t in list(state._background_tasks) if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    def _err_texts(self, slot):
        return [m["content"] for m in slot.messages if m.get("role") == "error"]

    def _assistant_texts(self, slot):
        return [m["content"] for m in slot.messages if m.get("role") == "assistant"]

    @staticmethod
    def _image_error():
        from kiro_crew.acp.client import AcpError

        exc = AcpError("The model could not process an image", transient=False)
        exc.structural_terminal = True
        exc.image_format_unsupported = True
        return exc

    @pytest.mark.asyncio
    async def test_retained_tool_image_discards_native_history_and_continues(
        self, tmp_path, monkeypatch
    ):
        """A typed image rejection after a tool result cannot be repaired by
        retrying the native conversation. With no new user attachment, discard
        the resume SID once and continue from Kiro Crew's text transcript without
        replaying the original request or its completed tool side effects."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            LLMEvent,
        )

        calls: list[str] = []

        async def _stream(msg):
            calls.append(msg)
            if len(calls) == 1:
                yield LLMEvent(kind=EVENT_TOOL_CALL, title="read_file", tool_kind="read")
                raise self._image_error()
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="recovered without old image bytes")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "compose the avatar")
            await self._drain_bg(state)

        assert len(calls) == 2
        assert "Continue from where it stopped" in calls[1]
        state.sessions.discard_conversation.assert_awaited_once()
        state.sessions.reset.assert_not_awaited()
        assert any("without binary image history" in t for t in self._err_texts(slot))
        assert any("recovered without old image bytes" in t for t in self._assistant_texts(slot))
        assert slot._poisoned_reset_used is False

    @staticmethod
    def _record_queue_inserts(monkeypatch, slot) -> list[dict]:
        inserts: list[dict] = []
        repo = slot._queue_repository
        real_insert = repo.queue_insert

        def _record(owner, index, content, kind="", *args, **kw):
            inserts.append({"content": content, "kind": kind})
            return real_insert(owner, index, content, kind, *args, **kw)

        monkeypatch.setattr(repo, "queue_insert", _record)
        return inserts

    @pytest.mark.asyncio
    async def test_an_owed_completion_rejected_for_a_retained_image_runs_before_the_user(
        self, tmp_path, monkeypatch
    ):
        """A sub-agent completion the model never consumed is a result the parent
        is still owed. Its image-history replay is queued as the completion it is,
        so a newer user message does not cancel it: the completion runs again on
        the fresh conversation, reports itself consumed, and the user's message
        runs after it."""
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        calls: list[str] = []

        async def _stream(msg):
            calls.append(msg)
            if len(calls) == 1:
                raise self._image_error()
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="answered")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        self._wire_sessions(state, self._client(_stream))
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        inserts = self._record_queue_inserts(monkeypatch, slot)
        slot.queue_insert(0, "a newer user message", kind="")
        consumed = MagicMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(
                state,
                slot,
                "[Subagent completion event] the child finished",
                _turn_actor="subagent",
                _on_consumed=consumed,
            )
            await self._drain_bg(state)

        replays = [i for i in inserts if "the child finished" in i["content"]]
        assert replays == [
            {
                "content": "[Subagent completion event] the child finished",
                "kind": chat_runner.SUBAGENT_COMPLETION_KIND,
            }
        ], inserts
        assert not slot.replays.armed(
            ReplayFamily.IMAGE_HISTORY
        ), "an owed completion has no recovery record"
        assert ["the child finished" in c for c in calls[:2]] == [True, True], calls
        assert any("a newer user message" in c for c in calls[2:]), calls
        consumed.assert_any_call(True)
        assert not any("recovery cancelled" in m.get("content", "") for m in slot.messages)

    @pytest.mark.parametrize("turn", ["continuation", "synthesis"])
    @pytest.mark.asyncio
    async def test_a_turn_that_owes_nothing_yields_to_the_users_message(
        self, tmp_path, monkeypatch, turn
    ):
        """Only a completion still owed outranks the user. A completion whose
        output already landed would replay as a continuation, and the synthesis
        turn is runner-authored: with a user message queued, neither is retried,
        and the user's message runs next."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            LLMEvent,
        )

        calls: list[str] = []

        async def _stream(msg):
            calls.append(msg)
            if len(calls) == 1:
                if turn == "continuation":
                    yield LLMEvent(kind=EVENT_TOOL_CALL, title="read_file", tool_kind="read")
                raise self._image_error()
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="answered")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        self._wire_sessions(state, self._client(_stream))
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        slot.queue_insert(0, "a newer user message", kind="")
        if turn == "continuation":
            message = "[Subagent completion event] the child finished"
            kwargs = {"_turn_actor": "subagent", "_on_consumed": MagicMock()}
        else:
            message = SUBAGENT_SYNTHESIS_PROMPT
            # Even with a settlement callback attached, synthesis owes nothing.
            kwargs = {
                "_turn_actor": "subagent",
                "_synthetic_payload": True,
                "_on_consumed": MagicMock(),
            }

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, message, **kwargs)
            await self._drain_bg(state)

        assert len(calls) == 2, calls
        assert "a newer user message" in calls[1]
        state.sessions.discard_conversation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_new_image_attachment_is_not_automatically_discarded(self, tmp_path, monkeypatch):
        """A current attachment may itself be invalid. Preserve the healthy
        native conversation and let the actionable terminal error tell the user
        to remove or re-encode that image instead of discarding history."""
        from kiro_crew.dashboard.chat import _run_chat

        async def _fail(msg):
            raise self._image_error()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        await _run_chat(state, slot, "inspect this", _attachments=["/tmp/current.png"])

        state.sessions.discard_conversation.assert_not_awaited()
        state.sessions.reset.assert_not_awaited()
        assert slot._queue == []
        assert any(t.startswith("❌") for t in self._err_texts(slot))
        assert slot._poisoned_reset_used is False

    @pytest.mark.asyncio
    async def test_retained_image_discard_is_one_shot_until_a_turn_lands(
        self, tmp_path, monkeypatch
    ):
        """If the fresh conversation also rejects an image, surface the error;
        never loop through repeated discard and synthetic replay."""
        from kiro_crew.dashboard.chat import _run_chat

        async def _fail(msg):
            raise self._image_error()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        slot._poisoned_reset_used = True

        await _run_chat(state, slot, "continue")

        state.sessions.discard_conversation.assert_not_awaited()
        assert slot._queue == []
        assert any(t.startswith("❌") for t in self._err_texts(slot))
        assert slot._poisoned_reset_used is True

    @pytest.mark.asyncio
    async def test_a_current_turn_image_path_in_text_is_not_retained_history(
        self, tmp_path, monkeypatch
    ):
        """Empty dashboard attachment lists do not prove the turn shipped no
        image: a channel turn (and a dashboard turn that types a path) carries it
        as a bare path in the message text, which ``build_prompt_blocks`` inlines
        as a CURRENT-turn image block. Such a rejection is of the image the user
        just sent, so the healthy conversation must survive and the verbatim
        replay must never re-inline the same bytes."""
        from kiro_crew.dashboard.chat import _run_chat

        async def _fail(msg):
            raise self._image_error()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        # No _attachments: the image rides in the text, exactly as the Slack
        # event handler appends its attachment paths. Use the host's own path
        # grammar: ``image_refs._PATH_RE`` is platform-gated, so a POSIX path
        # is prose on Windows and would not be inlined there.
        shot = r"C:\Users\me\shot.bmp" if os.name == "nt" else "/home/me/shot.bmp"
        await _run_chat(state, slot, f"{shot} look at this")

        state.sessions.discard_conversation.assert_not_awaited()
        state.sessions.reset.assert_not_awaited()
        assert slot._queue == []
        assert any(t.startswith("❌") for t in self._err_texts(slot))
        assert slot._poisoned_reset_used is False

    @pytest.mark.asyncio
    async def test_a_stop_during_the_discard_drops_the_queued_recovery(self, tmp_path, monkeypatch):
        """The discard is awaited between the recovery's enqueue and the drain's
        dispatch. A soft Stop landing in that window preserves the queue and
        leaves ``_stopping`` back at idle, so only the enqueue-time stop-generation
        snapshot can see it. The recovery must be dropped, not dispatched: its
        replay would re-run a cancelled turn's remaining destructive step."""
        from kiro_crew.dashboard.chat import _run_chat

        calls: list[str] = []

        async def _fail(msg):
            calls.append(msg)
            raise self._image_error()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        async def _discard_then_stop(*_a, **_kw):
            # A Stop initiated while the discard is in flight. It resolves back
            # to idle, so the counter is the only surviving evidence.
            slot._stop_generation = getattr(slot, "_stop_generation", 0) + 1

        state.sessions.discard_conversation = AsyncMock(side_effect=_discard_then_stop)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "keep going")
            await self._drain_bg(state)

        # The original prompt ran once; the recovery was purged before dispatch.
        assert len(calls) == 1
        state.sessions.discard_conversation.assert_awaited_once()
        assert slot._queue == []
        assert any("Image-history recovery cancelled" in m["content"] for m in slot.messages)
        # The aborted episode refunds the shared one-shot.
        assert slot._poisoned_reset_used is False
        assert not slot.replays.armed(ReplayFamily.IMAGE_HISTORY)

    @pytest.mark.asyncio
    async def test_stop_after_image_recovery_dispatch_aborts_before_provider(
        self, tmp_path, monkeypatch
    ):
        """A Stop after dequeue but before the guarded task consumes the replay
        must still veto the recovery. The queue entry is already consumed at
        this seam, so the runner must use the identity and snapshots passed by
        the drain."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.dashboard.chat_utils import effective_session_key
        from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent

        provider_calls = 0

        async def _stream(msg):
            nonlocal provider_calls
            provider_calls += 1
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        session_key = effective_session_key(slot)
        state.sessions.stop_generation = lambda key: 4
        slot._poisoned_reset_used = True
        slot.replays.arm(
            ReplayFamily.IMAGE_HISTORY,
            entry_id="image-recovery-qid",
            session_key=session_key,
            stop_gen=7,
            session_stop_gen=4,
        )
        # The Stop lands after the drain's validation and task spawn.
        slot._stop_generation = 8

        await _run_chat(
            state,
            slot,
            "continue the destructive step",
            _replay=frozenset({ReplayFamily.IMAGE_HISTORY}),
            _synthetic_recovery_turn=True,
        )

        assert provider_calls == 0
        assert not slot.replays.armed(ReplayFamily.IMAGE_HISTORY)
        assert slot._poisoned_reset_used is False
        assert any(
            m.get("role") == "notice" and "Image-history recovery cancelled" in m.get("content", "")
            for m in slot.messages
        )

    _BINDING_MISMATCH = (
        "Invalid `signature` in `thinking` block. "
        "The block is bound to a different conversation."
    )

    @staticmethod
    def _binding_failure():
        from kiro_crew.acp.client import _raise_acp_error

        _raise_acp_error(
            {
                "code": -32603,
                "message": "Prompt failed",
                "data": TestRunChatTransientRetry._BINDING_MISMATCH,
            }
        )

    @pytest.mark.asyncio
    async def test_thinking_binding_mismatch_rebuilds_once_on_fresh_conversation(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        events: list[str] = []

        async def poisoned_stream(_message):
            self._binding_failure()
            yield  # pragma: no cover

        async def fresh_stream(_message):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="recovered-on-fresh-conversation")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

        state = self._make_state(tmp_path, monkeypatch)
        poisoned = self._client(poisoned_stream)
        fresh = self._client(fresh_stream)
        self._wire_sessions(state, poisoned)
        clients = iter((poisoned, fresh))

        async def get_or_create(*_args, **_kwargs):
            client = next(clients)
            events.append("get-poisoned" if client is poisoned else "get-fresh")
            return client, True, False

        async def discard(*_args, **_kwargs):
            events.append("discard")
            return True

        state.sessions.get_or_create = AsyncMock(side_effect=get_or_create)
        state.sessions.discard_conversation = AsyncMock(side_effect=discard)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        await asyncio.wait_for(_run_chat(state, slot, "follow up"), timeout=30)
        await self._drain_bg(state)

        # A plain discard, as the image-history recovery does: no durable
        # replay state and no binding-only session-map argument.
        assert events == ["get-poisoned", "discard", "get-fresh"]
        state.sessions.discard_conversation.assert_awaited_once_with("dashboard:s1")
        state.sessions.reset.assert_not_awaited()
        assert any(
            "recovered-on-fresh-conversation" in text for text in self._assistant_texts(slot)
        )
        # The replay ran and its ledger record was consumed with it.
        assert not slot.replays.armed(ReplayFamily.THINKING_BINDING)

    @pytest.mark.asyncio
    async def test_thinking_binding_mismatch_arms_the_ledger_family(self, tmp_path, monkeypatch):
        """The queued retry is a ledger family, so the drain and consume seams own
        its Stop, steer, follow-up and rebind vetoes."""
        from kiro_crew.dashboard.chat import _run_chat

        async def fail(_message):
            self._binding_failure()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(fail)
        self._wire_sessions(state, client)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        from kiro_crew.dashboard import chat_runner

        monkeypatch.setattr(chat_runner, "_start_next_queued_turn", AsyncMock(return_value=False))

        await asyncio.wait_for(_run_chat(state, slot, "follow up"), timeout=30)

        assert len(slot._queue) == 1
        assert slot.replays.armed(ReplayFamily.THINKING_BINDING)
        assert slot.replays.entry_id(ReplayFamily.THINKING_BINDING) == slot._queue[0]["id"]
        assert slot._poisoned_reset_used is True

    @pytest.mark.asyncio
    async def test_thinking_binding_mismatch_discard_is_one_shot(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat import _run_chat

        calls = 0

        async def always_fail(_message):
            nonlocal calls
            calls += 1
            self._binding_failure()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(always_fail)
        self._wire_sessions(state, client)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        await asyncio.wait_for(_run_chat(state, slot, "follow up"), timeout=30)
        await self._drain_bg(state)

        assert calls == 2
        state.sessions.discard_conversation.assert_awaited_once()
        assert slot._poisoned_reset_used is True

    @pytest.mark.asyncio
    async def test_unrecoverable_binding_mismatch_is_a_terminal_cycle_failure(
        self, tmp_path, monkeypatch
    ):
        """A spent one-shot refuses the rebuild; the mismatch falls through to the
        terminal error, so a self-wake loop's stand-down bound sees it."""
        from kiro_crew.dashboard.chat import _run_chat

        async def fail(_message):
            self._binding_failure()
            yield  # pragma: no cover

        noted: list[tuple[str, bool]] = []

        async def _note(slot_key, _exc, *, self_wake, **_kwargs):
            noted.append((slot_key, self_wake))

        from kiro_crew.dashboard import chat_runner

        monkeypatch.setattr(chat_runner, "_note_cycle_failure", _note)
        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        slot._poisoned_reset_used = True
        slot._prestream_exhausted_cycles = 2

        await asyncio.wait_for(
            _run_chat(state, slot, "loop cycle", _directive_self_wake=True), timeout=30
        )
        await self._drain_bg(state)

        assert noted == [(slot.key, True)]
        assert slot._queue == []
        assert slot._prestream_exhausted_cycles == 0
        state.sessions.discard_conversation.assert_not_awaited()
        assert not slot.replays.armed(ReplayFamily.THINKING_BINDING)
        assert any(
            m.get("role") == "error" and m.get("content", "").startswith("❌")
            for m in slot.messages
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("produced", ["text", "thought", "tool_call"])
    async def test_binding_mismatch_after_output_never_replays(
        self, tmp_path, monkeypatch, produced
    ):
        """A mismatch that lands after the turn already produced output (a reply
        chunk, a thought or a tool call) must not discard and replay: the replay
        could repeat what the user already saw or a side effect that ran."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import (
            EVENT_TEXT_CHUNK,
            EVENT_THINKING_CHUNK,
            EVENT_TOOL_CALL,
            LLMEvent,
        )

        first = {
            "text": LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial reply"),
            "thought": LLMEvent(kind=EVENT_THINKING_CHUNK, text="partial thought"),
            "tool_call": LLMEvent(kind=EVENT_TOOL_CALL, title="read_file", tool_kind="read"),
        }[produced]

        async def produce_then_fail(_message):
            yield first
            self._binding_failure()

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(produce_then_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        await asyncio.wait_for(_run_chat(state, slot, "follow up"), timeout=30)
        await self._drain_bg(state)

        state.sessions.discard_conversation.assert_not_awaited()
        assert not slot.replays.armed(ReplayFamily.THINKING_BINDING)
        assert slot._poisoned_reset_used is False
        assert slot._queue == []

    @pytest.mark.asyncio
    async def test_an_owed_completion_hit_by_a_binding_mismatch_runs_before_the_user(
        self, tmp_path, monkeypatch
    ):
        """A sub-agent completion the model never consumed is still owed to the
        parent. A newer user message queued before the mismatch must not refuse
        the rebuild, and the replay is queued as the completion it is (no
        THINKING_BINDING record), so nothing can cancel it: the completion runs
        again on the fresh conversation, then the user's message."""
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        calls: list[str] = []

        async def _stream(msg):
            calls.append(msg)
            if len(calls) == 1:
                self._binding_failure()
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="answered")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        self._wire_sessions(state, self._client(_stream))
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        inserts = self._record_queue_inserts(monkeypatch, slot)
        slot.queue_insert(0, "a newer user message", kind="")
        consumed = MagicMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(
                state,
                slot,
                "[Subagent completion event] the child finished",
                _turn_actor="subagent",
                _on_consumed=consumed,
            )
            await self._drain_bg(state)

        replays = [i for i in inserts if "the child finished" in i["content"]]
        assert replays == [
            {
                "content": "[Subagent completion event] the child finished",
                "kind": chat_runner.SUBAGENT_COMPLETION_KIND,
            }
        ], inserts
        assert not slot.replays.armed(ReplayFamily.THINKING_BINDING)
        state.sessions.discard_conversation.assert_awaited()
        assert ["the child finished" in c for c in calls[:2]] == [True, True], calls
        assert any("a newer user message" in c for c in calls[2:]), calls
        consumed.assert_any_call(True)
        assert not any("recovery cancelled" in m.get("content", "") for m in slot.messages)

    @pytest.mark.asyncio
    async def test_recoverable_binding_mismatch_resets_outage_bookkeeping(
        self, tmp_path, monkeypatch
    ):
        """The recovery clears retry/fallback counters before the discard."""
        from kiro_crew.dashboard.chat import _run_chat

        async def fail(_message):
            self._binding_failure()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(fail)
        self._wire_sessions(state, client)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        slot._prestream_exhausted_cycles = 2
        slot._transient_5xx_retries = 1
        slot._infra_retries = 1
        seen: dict[str, int] = {}

        async def discard_then_stop(*_args, **_kwargs):
            seen["prestream"] = slot._prestream_exhausted_cycles
            seen["transient"] = slot._transient_5xx_retries
            seen["infra"] = slot._infra_retries
            # Stop the queued retry so only the first mismatch is observed.
            slot._stop_generation += 1
            return True

        state.sessions.discard_conversation = AsyncMock(side_effect=discard_then_stop)

        await asyncio.wait_for(_run_chat(state, slot, "follow up"), timeout=30)
        await self._drain_bg(state)

        assert seen == {"prestream": 0, "transient": 0, "infra": 0}

    @pytest.mark.asyncio
    async def test_thinking_binding_retry_is_dropped_when_stop_lands_during_discard(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.dashboard.recovery_replays import THINKING_BINDING_CANCELLED_TEXT

        calls = 0

        async def fail(_message):
            nonlocal calls
            calls += 1
            self._binding_failure()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(fail)
        self._wire_sessions(state, client)
        state.sessions.get_or_create = AsyncMock(return_value=(client, True, False))
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        async def discard_then_stop(*_args, **_kwargs):
            slot._stop_generation += 1
            return True

        state.sessions.discard_conversation = AsyncMock(side_effect=discard_then_stop)

        await asyncio.wait_for(_run_chat(state, slot, "follow up"), timeout=30)
        await self._drain_bg(state)

        assert calls == 1
        assert state.sessions.get_or_create.await_count == 1
        assert slot._queue == []
        assert any(m.get("content") == THINKING_BINDING_CANCELLED_TEXT for m in slot.messages)
        # The ledger refunds the shared one-shot of a replay that never ran.
        assert slot._poisoned_reset_used is False

    @pytest.mark.asyncio
    async def test_transient_pre_token_retries_then_recovers_no_reset(self, tmp_path, monkeypatch):
        """A transient 5xx before any token streams is retried on the SAME live
        session (no reset); the second attempt succeeds."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AcpError(self._TRANSIENT)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True  # skip background auto-title

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == 2  # initial transient failure + one retry
        # Recovered: the response is persisted, no ❌ error surfaced.
        assert any("ok-result" in t for t in self._assistant_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))
        # The transient branch fired (status surfaced) ...
        assert any("Connection unstable" in t for t in self._err_texts(slot))
        # ... and the notice is TAGGED, which is the only thing telling the UI a
        # recovery is pending; without it a stale pill re-runs the queued choice.
        from kiro_crew.dashboard.chat_utils import TRANSIENT_RETRY_KIND

        _hiccups = [
            m
            for m in slot.messages
            if m.get("role") == "error" and "Connection unstable" in m.get("content", "")
        ]
        assert _hiccups, "no hiccup row to inspect"
        assert all((m.get("meta") or {}).get("kind") == TRANSIENT_RETRY_KIND for m in _hiccups)
        # Negative control: a TERMINAL error row must NOT carry the tag, or the
        # discriminator would classify every failure as a pending retry.
        slot.append("error", "❌ terminal for the control", "msg msg-err")
        assert (slot.messages[-1].get("meta") or {}).get("kind") != TRANSIENT_RETRY_KIND
        # ... and the live session was NOT reset.
        state.sessions.reset.assert_not_awaited()
        # Budget reset to 0 after the successful turn.
        assert slot._transient_5xx_retries == 0

    @pytest.mark.asyncio
    async def test_transient_post_token_textonly_retries_once(self, tmp_path, monkeypatch):
        """A transient 5xx AFTER text streamed (no tool call) is retried EXACTLY
        ONCE, APPEND-ONLY: the streamed partial is PRESERVED (finalized as a
        normal assistant message, exactly like the terminal else: branch), a
        recovery notice is surfaced, and a CONTINUE instruction — NOT the
        original prompt — is re-queued onto the same live session. The retry
        appends the continued answer as a NEW message below. The user sees an
        append-only sequence [partial] [recover notice] [continued answer] with
        nothing retracted. No chat_stream_reset event exists anymore."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        call_count = 0
        captured: list = []

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            captured.append(msg)
            if call_count == 1:
                # First attempt streams a partial, then hits a transient 5xx.
                yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial ")
                raise AcpError(self._TRANSIENT)
            # Retry streams a clean, continued answer.
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == 2  # one post-token retry
        # The recovery notice surfaced and no ❌ error was raised.
        assert any("Connection unstable" in t for t in self._err_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))
        # APPEND-ONLY: the partial is PRESERVED as a finalized assistant message,
        # AND the continued retry answer is appended below it. Both are present;
        # the live chunk placeholders are finalized away.
        assert not any(m.get("role") == "chunk" for m in slot.messages)
        assistant = self._assistant_texts(slot)
        assert any("partial" in t for t in assistant), "partial must be preserved"
        assert any("ok-result" in t for t in assistant), "continued answer must append below"
        # The RE-QUEUED prompt is the CONTINUE instruction, NOT the original.
        assert len(captured) == 2
        assert (
            "Continue from where it stopped" in captured[1]
        ), "the retry must re-queue the continue instruction, not the original prompt"
        assert "hello" not in captured[1], "the original prompt must NOT be re-queued"
        # No chat_stream_reset broadcast — that frontend reconcile event was
        # removed entirely by the append-only rework.
        assert not any(
            c.args and c.args[0] == "chat_stream_reset" for c in state.broadcast_ws.call_args_list
        ), "chat_stream_reset must no longer be broadcast"
        # Live session was NOT reset. The one-shot allowance stays consumed
        # (True) across the synthetic recovery turn — it is refreshed only at the
        # start of the NEXT genuine user turn, never on the recovery turn itself
        # (finding #3, prevents a recovery-turn re-failure from looping).
        state.sessions.reset.assert_not_awaited()
        assert slot._posttoken_retry_used is True

    @pytest.mark.asyncio
    async def test_transient_post_token_suppressed_preserves_partial(self, tmp_path, monkeypatch):
        """When Stop is active (_should_suppress_requeue True) during a post-token
        transient 5xx, the partial assistant text is PRESERVED (persisted as an
        assistant message) and the message is NOT re-queued. Under the
        append-only design the partial + a retry notice are shown regardless of
        eligibility; only the re-queue is gated."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_TEXT_CHUNK, LLMEvent

        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            # Stream a partial, then hit a transient 5xx (post-token, no tool).
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial answer")
            raise AcpError(self._TRANSIENT)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        # Stop is active → suppress the re-queue for the whole handler.
        monkeypatch.setattr(chat_runner, "_should_suppress_requeue", lambda s: True)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # No re-queue: the stream ran exactly once (Stop is active). Finding #3:
        # a suppressed recovery does NOT consume the one-shot allowance — the
        # flag is only set when a recovery is actually enqueued, so it stays
        # False and a later genuine turn can still recover once.
        assert call_count == 1
        assert slot._posttoken_retry_used is False
        # HIGH: the partial survives as a persisted assistant message.
        assert any("partial answer" in t for t in self._assistant_texts(slot))
        # Chunk placeholders were finalized (not left as role "chunk").
        assert not any(m.get("role") == "chunk" for m in slot.messages)
        # The retry notice is shown (append-only: partial + notice regardless of
        # eligibility); no ❌ terminal error is raised here.
        assert any("Connection unstable" in t for t in self._err_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))
        # No chat_stream_reset broadcast — that event was removed entirely.
        assert not any(
            c.args and c.args[0] == "chat_stream_reset" for c in state.broadcast_ws.call_args_list
        )
        # Transient path never resets the (still-alive) session.
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transient_post_toolcall_recovers(self, tmp_path, monkeypatch):
        """A transient 5xx AFTER a TOOL CALL fired RECOVERS instead of failing
        fast. Because the retry re-queues a CONTINUE instruction onto the
        SAME live session — which still holds the completed tool results — the
        model resumes from where it stopped instead of blindly re-running the
        tool. Exactly one post-token recovery fires, the partial is preserved,
        and the continue instruction (not the original prompt) is re-queued."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_TOOL_CALL,
            LLMEvent,
        )

        call_count = 0
        captured: list = []

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            captured.append(msg)
            if call_count == 1:
                yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="working ")
                yield LLMEvent(
                    kind=EVENT_TOOL_CALL,
                    text="",
                    title="fs_read",
                    tool_kind="read",
                    tool_call_id="tc-1",
                )
                raise AcpError(self._TRANSIENT)
            # Retry continues from the preserved context and finishes cleanly.
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == 2  # post-tool transient now RECOVERS via continue
        # Recovery notice surfaced, no ❌ terminal error.
        assert any("Connection unstable" in t for t in self._err_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))
        # The partial text is preserved and the continued answer appended below.
        assert any("working" in t for t in self._assistant_texts(slot)), "partial must be preserved"
        assert any(
            "ok-result" in t for t in self._assistant_texts(slot)
        ), "continued answer must append below"
        # The CONTINUE instruction — not the original prompt — was re-queued.
        assert len(captured) == 2
        assert "Continue from where it stopped" in captured[1]
        assert "hello" not in captured[1]
        # The one-shot allowance stays consumed (True) across the recovery turn;
        # it is refreshed only by the next genuine user turn. Live session never
        # reset.
        assert slot._posttoken_retry_used is True
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_posttoken_suppressed_leaves_allowance_for_later_turn(
        self, tmp_path, monkeypatch
    ):
        """Finding #3(a): a Stop-suppressed post-token transient must NOT consume
        the one-shot allowance (the flag is set only when a recovery is actually
        enqueued). A LATER genuine user turn on the same slot can therefore still
        recover once."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        # ── Turn 1: Stop active → suppressed post-token transient. ──
        async def _stream_suppressed(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial-1")
            raise AcpError(self._TRANSIENT)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream_suppressed)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        monkeypatch.setattr(chat_runner, "_should_suppress_requeue", lambda s: True)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "first question")
            await self._drain_bg(state)

        # Allowance untouched — the suppressed path never enqueued a recovery.
        assert slot._posttoken_retry_used is False
        # And the row says so: nothing resumes, so it is the TERMINAL notice
        # (give-up text + token, no retry kind) — never a pending "resuming…"
        # the dashboard would draw as a soft notice with no way to continue.
        from kiro_crew.dashboard.chat_utils import (
            TRANSIENT_GIVE_UP_TEXT,
            TRANSIENT_NOTICE_GIVE_UP,
            TRANSIENT_RETRY_KIND,
        )

        _terminal = [
            m
            for m in slot.messages
            if m.get("role") == "error" and "Connection unstable" in m.get("content", "")
        ]
        assert len(_terminal) == 1
        assert _terminal[0]["content"] == TRANSIENT_GIVE_UP_TEXT
        assert _terminal[0]["meta"].get("notice") == TRANSIENT_NOTICE_GIVE_UP
        assert _terminal[0]["meta"].get("kind") != TRANSIENT_RETRY_KIND

        # ── Turn 2: Stop cleared → a genuine turn recovers once. ──
        call_count = 0

        async def _stream_recover(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial-2")
                raise AcpError(self._TRANSIENT)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        client.stream = _stream_recover
        client.stream_command = _stream_recover
        monkeypatch.setattr(chat_runner, "_should_suppress_requeue", lambda s: False)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "second question")
            await self._drain_bg(state)

        # The later real turn recovered: it re-ran once and produced the answer.
        assert call_count == 2
        assert any("ok-result" in t for t in self._assistant_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))

    @pytest.mark.asyncio
    async def test_posttoken_recovery_turn_refailure_does_not_loop(self, tmp_path, monkeypatch):
        """Finding #3(b): a post-token transient that re-fires DURING the
        synthetic recovery turn must NOT recover again (no infinite re-queue).
        The recovery turn inherits the already-consumed allowance (it is not
        refreshed for the recover message), so the post-token branch is skipped
        and a clean terminal error surfaces instead of a second recovery."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.dashboard.chat_runner import _POSTTOKEN_RECOVER_MSG
        from kiro_crew.providers.base import EVENT_TEXT_CHUNK, LLMEvent

        captured: list = []

        async def _stream(msg):
            captured.append(msg)
            # The recovery turn emits a partial then hits ANOTHER transient 5xx.
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="still-failing")
            raise AcpError(self._TRANSIENT)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        # Simulate the state left by the originating turn that enqueued recovery.
        slot._posttoken_retry_used = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Drive the recovery turn directly with the continue instruction.
            await _run_chat(state, slot, _POSTTOKEN_RECOVER_MSG)
            await self._drain_bg(state)

        # Ran exactly once — no second recovery was enqueued (no loop).
        assert len(captured) == 1
        # The recover instruction was NOT re-queued a second time.
        assert not any(m == _POSTTOKEN_RECOVER_MSG for m in captured[1:])
        # The flag was NOT refreshed by the recovery turn (stays consumed).
        assert slot._posttoken_retry_used is True
        # A clean terminal error surfaced (branch skipped, fell to else:).
        assert any(t.startswith("❌") for t in self._err_texts(slot))
        # The partial from the recovery turn is still preserved (append-only).
        assert any("still-failing" in t for t in self._assistant_texts(slot))
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_error_fails_fast_no_retry(self, tmp_path, monkeypatch):
        """An auth failure is excluded from the transient set — it fails fast
        with a clean error and is never retried (a retry can't fix a token)."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            raise AcpError(self._AUTH)
            yield  # pragma: no cover — makes this an async generator

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == 1  # fail-fast, no retry
        assert slot._transient_5xx_retries == 0
        assert any(t.startswith("❌") for t in self._err_texts(slot))
        assert not any("Connection unstable" in t for t in self._err_texts(slot))
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("partial", ["", "Here is the first half"])
    async def test_codex_provider_failure_renders_as_an_error_row(
        self, tmp_path, monkeypatch, partial
    ):
        """codex-acp writes a terminal provider error as message text and answers
        ``end_turn``. The session handle turns that into the turn's AcpError
        (``test_codex_terminal_error_fidelity``); here the error the handle
        actually raises is fed to the chat loop, which must show an error row,
        keep any partial answer, never store the provider error as assistant
        text, and not retry a 400."""
        from test_codex_terminal_error_fidelity import (
            _PROVIDER_MESSAGE,
            END_TURN,
            ERROR_CHUNK_2_0,
            _handle,
            _run,
        )

        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_TEXT_CHUNK, LLMEvent

        _events, failure = await _run(_handle([ERROR_CHUNK_2_0, END_TURN]))
        assert failure is not None
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            if partial:
                yield LLMEvent(kind=EVENT_TEXT_CHUNK, text=partial)
            raise failure

        state = self._make_state(tmp_path, monkeypatch)
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._agent_fallback_chain", lambda: ())
        self._wire_sessions(state, self._client(_stream))
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == 1, "a deterministic 400 was retried"
        errors = self._err_texts(slot)
        assert any(_PROVIDER_MESSAGE in t for t in errors), errors
        assistant = self._assistant_texts(slot)
        assert not any(_PROVIDER_MESSAGE in t for t in assistant), assistant
        assert not any("invalid_request_error" in t for t in assistant), assistant
        if partial:
            assert any(partial in t for t in assistant), assistant
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transient_exhausts_budget_then_clean_error_resumable(
        self, tmp_path, monkeypatch
    ):
        """Persistent transient 5xx is retried up to the budget, then surfaces a
        clean ❌ error while leaving the session resumable (never reset)."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES

        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        # fallback_model="" (disabled): this test pins the PRE-FEATURE ladder
        # (the default is now "auto", which would walk the fallback branch —
        # that branch has its own class, TestRunChatModelFallback).
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._agent_fallback_chain", lambda: ())
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # Initial attempt + TRANSIENT_RETRIES re-prompts, then it gives up.
        assert call_count == TRANSIENT_RETRIES + 1
        # Clean error surfaced; session left resumable (no reset).
        assert any(t.startswith("❌") for t in self._err_texts(slot))
        state.sessions.reset.assert_not_awaited()
        # The terminal ❌ ENDS the cycle, so the budget is refreshed for the
        # next one — the Continue press the error message invites must get the
        # full ladder again, not inherit an exhausted counter.
        assert slot._transient_5xx_retries == 0

    @staticmethod
    def _connection_drop():
        """A network-path drop raised through the real ACP raise path."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.acp.transport_errors import _raise_acp_error

        try:
            _raise_acp_error(
                {
                    "code": -32603,
                    "message": "Internal error",
                    "data": "dispatch failure: io error: connection reset by peer",
                }
            )
        except AcpError as exc:
            assert exc.connection_failure is True
            return exc
        raise AssertionError("_raise_acp_error did not raise")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("synthetic", [False, True], ids=["user-turn", "recovery-turn"])
    async def test_connection_drop_shorter_than_window_recovers(
        self, tmp_path, monkeypatch, synthetic
    ):
        """A network drop that outlasts the fixed TRANSIENT_RETRIES count but
        ends inside the recovery window resumes the turn on the SAME session,
        with one standing retry notice rather than one per extra attempt. A
        runner recovery turn replays as a different inject kind and gets the
        same single notice."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.dashboard.chat_utils import (
            _SYNTHETIC_RECOVERY_MSGS,
            TRANSIENT_NOTICE_META_KEY,
            TRANSIENT_NOTICE_RETRYING,
        )
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        drop_attempts = TRANSIENT_RETRIES + 3
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            if call_count <= drop_attempts:
                raise self._connection_drop()
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="back-online")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._agent_fallback_chain", lambda: ())
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        sleeps: list[float] = []

        async def _fake_sleep(secs, *a, **k):
            sleeps.append(secs)

        with patch("asyncio.sleep", side_effect=_fake_sleep):
            if synthetic:
                await _run_chat(
                    state,
                    slot,
                    _SYNTHETIC_RECOVERY_MSGS[0],
                    _synthetic_recovery_turn=True,
                    _synthetic_payload=True,
                )
            else:
                await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == drop_attempts + 1
        assert any("back-online" in t for t in self._assistant_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))
        state.sessions.reset.assert_not_awaited()
        retrying = [
            m
            for m in slot.messages
            if (m.get("meta") or {}).get(TRANSIENT_NOTICE_META_KEY) == TRANSIENT_NOTICE_RETRYING
        ]
        assert len(retrying) == TRANSIENT_RETRIES
        # The landed turn hands the whole ladder back.
        assert slot._transient_5xx_retries == 0

    @pytest.mark.asyncio
    async def test_connection_drop_longer_than_window_fails_clearly(self, tmp_path, monkeypatch):
        """A drop that outlasts the recovery window ends in the same clean,
        resumable ❌ as an exhausted provider-error ladder. Elapsed time comes
        from one fake monotonic clock that both modules read and each attempt
        advances, so the result does not depend on the host's uptime."""
        import types

        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import CONNECTION_RECOVERY_WINDOW_SECS, TRANSIENT_RETRIES

        step = CONNECTION_RECOVERY_WINDOW_SECS / 5
        clock = {"now": 10_000.0}
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                clock["now"] += step
            raise self._connection_drop()
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._agent_fallback_chain", lambda: ())
        import kiro_crew.dashboard.chat_runner as runner_mod
        import kiro_crew.llm_helpers as helpers_mod

        for mod in (runner_mod, helpers_mod):
            fake = types.SimpleNamespace(
                **{k: getattr(mod.time, k) for k in dir(mod.time) if not k.startswith("__")}
            )
            fake.monotonic = lambda: clock["now"]
            monkeypatch.setattr(mod, "time", fake)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # Attempt N sees N-1 steps elapsed; the window closes at five steps,
        # so the sixth attempt is the last, two beyond the fixed count.
        assert call_count == 6
        assert call_count > TRANSIENT_RETRIES + 1
        # The terminal row names the network as the cause, not a retry notice.
        errs = self._err_texts(slot)
        assert errs and "Could not reach the model backend" in errs[-1]
        state.sessions.reset.assert_not_awaited()
        assert slot._transient_5xx_retries == 0

    @pytest.mark.asyncio
    async def test_provider_5xx_keeps_fixed_budget_inside_window(self, tmp_path, monkeypatch):
        """A provider 5xx answer is not a network drop: it keeps the fixed
        TRANSIENT_RETRIES count even while the recovery window is open."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES

        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            exc = AcpError(self._TRANSIENT, transient=True)
            exc.connection_failure = False
            raise exc
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._agent_fallback_chain", lambda: ())
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == TRANSIENT_RETRIES + 1
        assert any(t.startswith("❌") for t in self._err_texts(slot))

    @pytest.mark.asyncio
    async def test_transient_budget_refreshed_after_terminal_error(self, tmp_path, monkeypatch):
        """Regression: a cycle that EXHAUSTS the transient budget and surfaces
        the terminal ❌ must refresh the budget for the NEXT cycle. The
        happy-path reset only runs when a cycle COMPLETES, so before the fix
        the exhausted counter leaked into every later cycle: the very next
        5xx — e.g. right after the Continue press the ❌ message itself
        invites ("retry in a moment") — failed instantly with ZERO retries
        until some turn happened to finish cleanly."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        # ── Cycle 1: persistent 5xx exhausts the budget → terminal ❌. ──
        async def _always_fail(msg):
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_always_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "first question")
            await self._drain_bg(state)

        assert any(t.startswith("❌") for t in self._err_texts(slot))
        assert slot._transient_5xx_retries == 0
        n_before = len(slot.messages)

        # ── Cycle 2: a 5xx on the next turn retries again (fresh budget). ──
        call_count = 0

        async def _fail_once_then_ok(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AcpError(self._TRANSIENT)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        client.stream = _fail_once_then_ok
        client.stream_command = _fail_once_then_ok

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "second question")
            await self._drain_bg(state)

        # Before the fix: 3 < TRANSIENT_RETRIES was False on the first 5xx, so
        # cycle 2 died instantly (call_count == 1, a second ❌, no answer).
        assert call_count == 2
        tail = slot.messages[n_before:]
        tail_errs = [m["content"] for m in tail if m.get("role") == "error"]
        tail_answers = [m["content"] for m in tail if m.get("role") == "assistant"]
        assert any("Connection unstable" in t for t in tail_errs)
        assert not any(t.startswith("❌") for t in tail_errs)
        assert any("ok-result" in t for t in tail_answers)
        # Completed cycle → budget back to 0 via the happy-path reset.
        assert slot._transient_5xx_retries == 0
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_first_exhausted_cycle_never_discards(self, tmp_path, monkeypatch):
        """A SINGLE cycle that exhausts the pre-stream transient ladder
        surfaces the terminal ❌ without discarding the native conversation —
        one exhaustion is still plausibly a momentary outage, and discarding
        on it would throw away a healthy conversation on every blip that
        outlasts the ladder."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        async def _always_fail(msg):
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_always_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert any(t.startswith("❌") for t in self._err_texts(slot))
        state.sessions.discard_conversation.assert_not_awaited()
        state.sessions.reset.assert_not_awaited()
        # The exhausted cycle is counted toward the consecutive streak.
        assert slot._prestream_exhausted_cycles == 1
        assert slot._poisoned_reset_used is False

    @pytest.mark.asyncio
    async def test_poisoned_conversation_discarded_on_second_exhausted_cycle(
        self, tmp_path, monkeypatch
    ):
        """Regression (poisoned persisted conversation): when TWO consecutive
        cycles each exhaust the full pre-stream transient ladder with zero
        output, the backend is deterministically rejecting this session's
        native conversation — retrying into it can never succeed (observed
        live: a session/load'ed conversation failing pre-stream identically
        11 hours apart while a NEW session on the same gateway+model answered
        instantly). The second exhaustion must escalate: discard the
        native conversation (discard_conversation() clears the resume sid —
        reset() would session/load the poison right back — while keeping the
        session-map entry with its channel linkage) and re-queue the message
        once, so the recovery cycle cold-starts a fresh conversation and the
        slot self-heals instead of telling the user to 'retry in a moment'
        forever."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        # Fail every attempt of two full cycles (the "poisoned conversation"),
        # then succeed — the success models the fresh post-discard conversation.
        _poisoned_calls = 2 * (TRANSIENT_RETRIES + 1)
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            if call_count <= _poisoned_calls:
                raise AcpError(self._TRANSIENT)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="recovered-after-discard")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        # fallback_model="" (disabled): pins the PRE-FEATURE escalation ladder
        # (the "auto" default would walk the fallback branch instead).
        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._agent_fallback_chain", lambda: ())
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "kiro_crew.dashboard.chat_runner.run_bg_oneliner",
                new_callable=AsyncMock,
                return_value="OK",
            ) as canary,
        ):
            # Cycle 1: ladder exhausts → terminal ❌ (streak = 1, no discard,
            # no canary — below the consecutive threshold).
            await _run_chat(state, slot, "first try")
            await self._drain_bg(state)
            state.sessions.discard_conversation.assert_not_awaited()
            canary.assert_not_awaited()
            # Cycle 2 (the Continue press the ❌ invites): ladder exhausts
            # again → the canary probes a FRESH conversation and succeeds →
            # conversation-specific rejection proven → escalation fires and
            # the re-queued recovery cycle lands on the fresh session.
            await _run_chat(state, slot, "continue")
            await self._drain_bg(state)
            canary.assert_awaited_once()
            # The probe is only meaningful on the SAME served model as the
            # failing session, with fallback disabled (GPT review finding:
            # a canary succeeding on a different model must never justify
            # discarding this conversation).
            assert canary.await_args.kwargs["model"] == "claude-test-model"
            assert canary.await_args.kwargs["strict_model"] is True

        # Escalation discarded the native conversation exactly once; plain
        # reset (which preserves the poisoned resume sid) was never used.
        state.sessions.discard_conversation.assert_awaited_once()
        state.sessions.reset.assert_not_awaited()
        # Cycle 1 ❌ + cycle 2 escalation notice, then the recovered answer.
        errs = self._err_texts(slot)
        assert any(t.startswith("❌") for t in errs)
        assert any("keeps rejecting" in t for t in errs)
        assert any("recovered-after-discard" in t for t in self._assistant_texts(slot))
        # Attempt accounting: two full ladders + the single recovery prompt.
        assert call_count == _poisoned_calls + 1
        # The landed recovery turn re-arms the one-shot and breaks the streak.
        assert slot._poisoned_reset_used is False
        assert slot._prestream_exhausted_cycles == 0

    @pytest.mark.asyncio
    async def test_poisoned_discard_is_one_shot_until_a_turn_lands(self, tmp_path, monkeypatch):
        """If even the fresh post-discard conversation keeps failing (genuine
        prolonged outage), no second discard fires: the one-shot is consumed by
        the first escalation and only a LANDED turn re-arms it, so a discard
        loop is impossible. Later cycles surface the terminal ❌ as before."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        async def _always_fail(msg):
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_always_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "kiro_crew.dashboard.chat_runner.run_bg_oneliner",
                new_callable=AsyncMock,
                return_value="OK",
            ),
        ):
            # Cycle 1: exhausts → ❌ (streak 1).
            await _run_chat(state, slot, "first try")
            await self._drain_bg(state)
            # Cycle 2: exhausts → canary succeeds → discard #1 → recovery
            # cycle ALSO exhausts → terminal ❌ (one-shot consumed, so the
            # eligibility gate rejects before the canary even runs again).
            await _run_chat(state, slot, "continue")
            await self._drain_bg(state)
            # Cycle 3: user retries once more — still failing. Streak keeps
            # counting but the one-shot is spent: ❌ again, NO second discard.
            await _run_chat(state, slot, "continue again")
            await self._drain_bg(state)

        state.sessions.discard_conversation.assert_awaited_once()
        state.sessions.reset.assert_not_awaited()
        # No turn ever landed: the one-shot stays consumed.
        assert slot._poisoned_reset_used is True
        # Terminal ❌ surfaced after the failed recovery and again on cycle 3.
        assert sum(1 for t in self._err_texts(slot) if t.startswith("❌")) >= 2

    @pytest.mark.asyncio
    async def test_cancelled_turn_does_not_rearm_poisoned_one_shot(self, tmp_path, monkeypatch):
        """Regression: the poisoned-discard one-shot re-arms only on a LANDED
        turn. But the STREAK is evidence-based: a cancelled turn that EMITTED
        output proves the backend accepts this conversation, so it breaks the
        streak (GPT review finding — without this, exhaustion → stopped-but-
        streaming turn → exhaustion would discard a healthy conversation),
        while a cancelled turn with NO output proves nothing and preserves
        it. In both cases the spent one-shot stays consumed."""
        from kiro_crew.acp.types import STOP_REASON_CANCELLED
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        # ── Cancelled turn with NO output: both guards preserved. ──
        async def _cancelled_silent(msg):
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_cancelled_silent)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        # Arrange: an escalation already consumed the one-shot mid-streak.
        slot._poisoned_reset_used = True
        slot._prestream_exhausted_cycles = 3

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "user stops before any output")
            await self._drain_bg(state)

        assert slot._poisoned_reset_used is True
        assert slot._prestream_exhausted_cycles == 3

        # ── Cancelled turn WITH output: streak broken, one-shot still spent. ──
        async def _cancelled_after_output(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="partial before stop")
            yield LLMEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED)

        client.stream = _cancelled_after_output
        client.stream_command = _cancelled_after_output
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "recovery attempt the user stops mid-answer")
            await self._drain_bg(state)

        assert slot._prestream_exhausted_cycles == 0  # output = evidence
        assert slot._poisoned_reset_used is True  # cancel ≠ landed

        # ── A genuinely LANDED turn re-arms the one-shot too. ──
        async def _ok(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="landed")
            yield LLMEvent(kind=EVENT_COMPLETE)

        client.stream = _ok
        client.stream_command = _ok
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "works now")
            await self._drain_bg(state)

        assert slot._poisoned_reset_used is False
        assert slot._prestream_exhausted_cycles == 0

    @pytest.mark.asyncio
    async def test_canary_failure_blocks_discard_and_preserves_one_shot(
        self, tmp_path, monkeypatch
    ):
        """GPT-review fix: two exhausted ladders alone are NOT
        conversation-specific evidence — a sustained backend-wide outage or
        throttle produces the identical pattern. The canary probe (one prompt
        on a fresh background conversation) is the discriminator: when it
        ALSO fails, no discard fires, the one-shot stays unconsumed, and the
        streak stays accrued so a later user-initiated cycle re-probes once
        the outage ends."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        async def _always_fail(msg):
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_always_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "kiro_crew.dashboard.chat_runner.run_bg_oneliner",
                new_callable=AsyncMock,
                side_effect=AcpError(self._TRANSIENT),
            ) as canary,
        ):
            # Two consecutive exhausted cycles — the pattern that WOULD
            # discard if the canary had succeeded.
            await _run_chat(state, slot, "first try")
            await self._drain_bg(state)
            await _run_chat(state, slot, "continue")
            await self._drain_bg(state)

        canary.assert_awaited_once()  # probed at the threshold, cycle 2
        state.sessions.discard_conversation.assert_not_awaited()
        state.sessions.reset.assert_not_awaited()
        # Backend-wide failure consumes NOTHING: the one-shot survives for a
        # later cycle whose canary succeeds, and the streak keeps accruing.
        assert slot._poisoned_reset_used is False
        assert slot._prestream_exhausted_cycles == 2
        assert sum(1 for t in self._err_texts(slot) if t.startswith("❌")) >= 2

    @pytest.mark.asyncio
    async def test_canary_empty_reply_is_not_positive_evidence(self, tmp_path, monkeypatch):
        """A canary that completes with EMPTY output proves nothing about the
        fresh conversation working — fail-safe to no-discard."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        async def _always_fail(msg):
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_always_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "kiro_crew.dashboard.chat_runner.run_bg_oneliner",
                new_callable=AsyncMock,
                return_value="  ",
            ),
        ):
            await _run_chat(state, slot, "first try")
            await self._drain_bg(state)
            await _run_chat(state, slot, "continue")
            await self._drain_bg(state)

        state.sessions.discard_conversation.assert_not_awaited()
        assert slot._poisoned_reset_used is False

    @pytest.mark.asyncio
    async def test_stop_during_canary_vetoes_discard_and_requeue(self, tmp_path, monkeypatch):
        """GPT-review fix: a Stop initiated DURING the (up to 30s) canary probe
        must veto the discard and the synthetic re-queue even when the canary
        succeeds — the user just cancelled this work, and re-queueing would
        re-execute it with tool side effects. The veto keys on the monotonic
        _stop_generation, not _stop_state, because teardown can drive
        _stop_state back to "idle" before the check (a stop that fired AND
        resolved mid-probe must still veto). Nothing is consumed: the one-shot
        stays armed and the streak stays accrued."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        async def _always_fail(msg):
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_always_fail)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        async def _canary_with_midflight_stop(*a, **kw):
            # User presses Stop while the probe is in flight; the stop then
            # RESOLVES (teardown drives _stop_state back to idle) before the
            # probe returns — the generation edge is the only surviving trace.
            slot._stop_state = "soft_pending"
            slot._stop_state = "idle"
            return "OK"  # the canary itself succeeds

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "kiro_crew.dashboard.chat_runner.run_bg_oneliner",
                new_callable=AsyncMock,
                side_effect=_canary_with_midflight_stop,
            ) as canary,
        ):
            await _run_chat(state, slot, "first try")
            await self._drain_bg(state)
            await _run_chat(state, slot, "continue")
            await self._drain_bg(state)

        canary.assert_awaited_once()  # probe ran at the threshold
        # ...but the mid-probe stop vetoed everything downstream: no discard,
        # no synthetic recovery turn, nothing consumed.
        state.sessions.discard_conversation.assert_not_awaited()
        assert slot._queue == []  # no synthetic recovery turn queued
        assert slot._poisoned_reset_used is False
        assert slot._prestream_exhausted_cycles == 2

    @pytest.mark.asyncio
    async def test_unreadable_session_model_skips_canary_and_discard(self, tmp_path, monkeypatch):
        """When the session's served model cannot be read, the canary cannot
        be pinned to it, so the probe is meaningless — fail-safe: no probe,
        no discard, one-shot preserved."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat

        async def _always_fail(msg):
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_always_fail)
        client.served_model = ""  # unreadable/unresolved
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "kiro_crew.dashboard.chat_runner.run_bg_oneliner",
                new_callable=AsyncMock,
                return_value="OK",
            ) as canary,
        ):
            await _run_chat(state, slot, "first try")
            await self._drain_bg(state)
            await _run_chat(state, slot, "continue")
            await self._drain_bg(state)

        canary.assert_not_awaited()
        state.sessions.discard_conversation.assert_not_awaited()
        assert slot._poisoned_reset_used is False

    @pytest.mark.asyncio
    async def test_transient_after_thinking_only_retries(self, tmp_path, monkeypatch):
        """A transient 5xx that lands AFTER reasoning streamed but before any
        answer token or tool call is still retried: thinking is ephemeral,
        broadcast-only output, so it does NOT flip _turn_emitted. This pins the
        deliberate decision in the EVENT_THINKING_CHUNK branch — if thinking
        ever starts being persisted, this guard must become a turn-emit to
        avoid a double-emit."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            EVENT_THINKING_CHUNK,
            LLMEvent,
        )

        call_count = 0
        thinking_seen = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            # Reasoning streams on BOTH attempts (cosmetic re-stream is the
            # accepted cost of retrying a thinking-only turn).
            yield LLMEvent(kind=EVENT_THINKING_CHUNK, text="let me think...")
            if call_count == 1:
                # Transient 5xx after thinking, before any answer token.
                raise AcpError(self._TRANSIENT)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)

        # Count chat_thinking broadcasts to prove reasoning re-streamed.
        _orig_broadcast = state.broadcast_ws

        def _count_thinking(event, payload, *a, **kw):
            nonlocal thinking_seen
            if event == "chat_thinking":
                thinking_seen += 1
            return _orig_broadcast(event, payload, *a, **kw)

        state.broadcast_ws = MagicMock(side_effect=_count_thinking)

        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # Thinking did NOT block the retry: initial transient + one retry.
        assert call_count == 2
        # Reasoning re-streamed on the retry. The streaming redaction buffer
        # (issue 3) may split/coalesce a thinking chunk across broadcasts, so
        # assert "streamed on both attempts" rather than an exact per-chunk count.
        assert thinking_seen >= 2
        # Recovered cleanly on the live session (no reset, no ❌).
        assert any("ok-result" in t for t in self._assistant_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))
        assert any("Connection unstable" in t for t in self._err_texts(slot))
        state.sessions.reset.assert_not_awaited()
        assert slot._transient_5xx_retries == 0

    @pytest.mark.asyncio
    async def test_thinking_only_failures_never_accrue_discard_streak(self, tmp_path, monkeypatch):
        """A turn that streamed REASONING before dying is a mid-generation
        failure — the backend demonstrably serves this conversation — not the
        poisoned pre-stream signature. Repeated thinking-then-transient-death
        turns must never accrue the discard streak (and so can never reach the
        canary/discard), even though they leave _turn_emitted False for retry
        purposes. Locks in the _turn_thought gate on _prestream_exhausted."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_THINKING_CHUNK, LLMEvent

        async def _think_then_die(msg):
            yield LLMEvent(kind=EVENT_THINKING_CHUNK, text="reasoning...")
            raise AcpError(self._TRANSIENT)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_think_then_die)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Well past POISONED_SESSION_CYCLES worth of exhausted ladders.
            for i in range(3):
                await _run_chat(state, slot, f"msg-{i}")
                await self._drain_bg(state)

        # Thinking activity keeps the streak at zero every cycle: never
        # eligible, so no canary probe and no discard — the conversation is
        # being served, just dying mid-generation.
        assert slot._prestream_exhausted_cycles == 0
        assert not slot._poisoned_reset_used
        state.sessions.discard_conversation.assert_not_awaited()


# ── Bulk model switch (api_chat_slots_model) ──


class TestRunChatModelFallback:
    """Throttle-exhaustion model fallback (agent.fallback_model) on the
    interactive path: budget exhausts → substitute set_model → visible notice
    card → re-queue on the SAME live session; empty chain is byte-for-byte
    today's terminal error; the sticky swap is restored at the next genuine
    turn start."""

    _TRANSIENT = "Prompt error: {'message': 'Internal error: API Error: Internal server error'}"

    @pytest.mark.asyncio
    async def test_fallback_swap_refreshes_the_served_model_cache(self, tmp_path, monkeypatch):
        """The swap moves the LIVE session onto the candidate without a spawn,
        so the spawn-time cache would keep naming the primary; it must follow
        the provider's served model instead."""
        from kiro_crew.dashboard.chat_runner import _fallback_swap_for_turn

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._agent_fallback_chain",
            lambda: ("fallback-model",),
        )
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.record_served_model("primary-model")

        class _Client:
            served_model = "primary-model"

            async def set_model(self, target):
                self._model = target
                self.served_model = target

        client = _Client()
        candidate = await _fallback_swap_for_turn(slot, client)
        assert candidate == "fallback-model"
        assert slot.served_model == "fallback-model"

    @pytest.mark.asyncio
    async def test_fallback_swap_serialises_on_the_pick_lock(self, tmp_path, monkeypatch):
        """REGRESSION (GPT finding on c97f2f2d): the swap awaits set_model, and
        an explicit pick landing during that await could be overwritten and
        snapshotted as fallback state. The swap must hold _model_pick_lock —
        with the lock pre-acquired, the swap may not advance until release.
        This closes the LAST writer: explicit pick, bulk pick, restore probe,
        and swap all serialise on the same per-slot lock."""
        import asyncio

        from kiro_crew.dashboard.chat_runner import _fallback_swap_for_turn

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._agent_fallback_chain",
            lambda: ("fallback-model",),
        )
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")

        entered = asyncio.Event()

        class _Client:
            async def set_model(self, target):
                entered.set()
                self._model = target

        client = _Client()

        await slot._model_pick_lock.acquire()
        try:
            task = asyncio.create_task(_fallback_swap_for_turn(slot, client))
            await asyncio.sleep(0.05)
            # Lock held ⇒ the swap must not have reached set_model yet.
            assert not entered.is_set()
            assert not task.done()
        finally:
            slot._model_pick_lock.release()
        candidate = await await_until(task, what="the fallback swap finishing")
        assert candidate == "fallback-model"
        assert slot._active_fallback_model == "fallback-model"

    @pytest.mark.asyncio
    async def test_fallback_swap_serialises_on_the_session_lock(self, tmp_path, monkeypatch):
        """The swap awaits set_model and then snapshots the shared client pick
        epoch. The per-slot pick lock is disjoint across aliases, so a pick made
        through a DIFFERENT alias of the same wire session could land in that
        await and be absorbed into the snapshot, after which the restore reads
        not-stale and silently overwrites the user's choice. The swap must hold
        the session-scoped switch lock too — with it pre-acquired, the swap may
        not advance until release. Mirrors the restore probe and refusal path.

        Mutation guard: drop the slot_switch_session_lock acquisition around the
        swap and this test reddens, because the swap reaches set_model while the
        session lock is held elsewhere."""
        import asyncio

        from kiro_crew.dashboard.chat_runner import _fallback_swap_for_turn
        from kiro_crew.dashboard.chat_utils import effective_session_key
        from kiro_crew.llm_helpers import slot_switch_session_lock

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._agent_fallback_chain",
            lambda: ("fallback-model",),
        )
        state = self._make_state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")

        entered = asyncio.Event()

        class _Client:
            async def set_model(self, target):
                entered.set()
                self._model = target

        client = _Client()

        _session_lock = slot_switch_session_lock(effective_session_key(slot))
        await _session_lock.acquire()
        try:
            task = asyncio.create_task(_fallback_swap_for_turn(slot, client))
            await asyncio.sleep(0.05)
            # Session lock held ⇒ the swap must not have reached set_model yet.
            assert not entered.is_set()
            assert not task.done()
        finally:
            _session_lock.release()
        candidate = await await_until(task, what="the fallback swap finishing")
        assert candidate == "fallback-model"
        assert slot._active_fallback_model == "fallback-model"

    @pytest.mark.asyncio
    async def test_stop_during_fallback_backoff_drops_the_requeue(self, tmp_path, monkeypatch):
        """REGRESSION (review finding on 1a61ddcf): a Stop pressed during the
        fallback backoff sleep resolves while no prompt is active; without the
        post-sleep guard the cancelled prompt is requeued and executes on the
        fallback anyway."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._agent_fallback_chain",
            lambda: ("fallback-model",),
        )
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        async def _stop_during_sleep(_secs):
            # Simulate the user pressing Stop while the backoff sleeps: the
            # monotonic generation increments even though the stop resolves
            # back to "idle" before the sleep returns. Scoped to the backoff this
            # test is about, because the same-model arm re-reads the stop signals
            # after ITS wait too: an unscoped press lands on the FIRST backoff,
            # drops the replay there, and the fallback arm this test measures is
            # then never reached at all.
            if slot._fallback_walked:
                slot._stop_generation = getattr(slot, "_stop_generation", 0) + 1

        with patch("asyncio.sleep", new=AsyncMock(side_effect=_stop_during_sleep)):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # Only the primary's budget ran; the cancelled prompt was NOT replayed
        # on the fallback (unfixed code replays it: +FALLBACK_CANDIDATE_ATTEMPTS).
        assert call_count == TRANSIENT_RETRIES + 1
        assert not any(q for q in getattr(slot, "_queue", []) if "hello" in str(q))
        # The drop arm ends the turn, so it must mirror the landed/terminal
        # per-turn resets: a fresh transient budget and a cleared walk (which
        # re-enables the restore probe). Sticky _active_fallback_model
        # deliberately survives for the probe to heal.
        assert slot._transient_5xx_retries == 0
        assert slot._fallback_candidate_idx == 0
        assert slot._fallback_walked == []

    _make_state = staticmethod(TestRunChatTransientRetry._make_state)
    _wire_sessions = staticmethod(TestRunChatTransientRetry._wire_sessions)
    _drain_bg = staticmethod(TestRunChatTransientRetry._drain_bg)

    def _err_texts(self, slot):
        return [m["content"] for m in slot.messages if m.get("role") == "error"]

    def _notice_texts(self, slot):
        return [m["content"] for m in slot.messages if m.get("role") == "notice"]

    def _assistant_texts(self, slot):
        return [m["content"] for m in slot.messages if m.get("role") == "assistant"]

    @staticmethod
    def _client(stream, advertised=("fallback-model",)):
        client = TestRunChatTransientRetry._client(stream)
        client._model = "claude-test-model"
        client.available_models = MagicMock(return_value=[{"modelId": m} for m in advertised])

        async def _set_model(model_id):
            # Mimic the real substitute path's bookkeeping so read_turn_model
            # (and therefore meta.turn_stats.model) reports the candidate.
            client._model = model_id
            client.served_model = model_id

        client.set_model = AsyncMock(side_effect=_set_model)
        return client

    @pytest.mark.asyncio
    async def test_budget_exhaustion_swaps_and_recovers(self, tmp_path, monkeypatch):
        """After the same-model budget exhausts, the session is moved onto the
        chain's candidate, the swap is ANNOUNCED as a persisted notice card,
        the turn recovers, and turn stats record the candidate."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._agent_fallback_chain",
            lambda: ("fallback-model",),
        )
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            if call_count <= TRANSIENT_RETRIES + 1:
                raise AcpError(self._TRANSIENT)
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="fb-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # 4 attempts on the primary, 1 on the candidate (success).
        assert call_count == TRANSIENT_RETRIES + 2
        client.set_model.assert_awaited_once_with("fallback-model")
        # The swap is announced — a persisted notice card, never silent.
        notices = self._notice_texts(slot)
        assert any(
            "claude-test-model" in t and "fallback-model" in t and "throttled" in t for t in notices
        ), f"expected a fallback notice card, got {notices!r}"
        # Recovered on the candidate; no terminal ❌.
        assert any("fb-result" in t for t in self._assistant_texts(slot))
        assert not any(t.startswith("❌") for t in self._err_texts(slot))
        # The live session was never reset — same-session recovery.
        state.sessions.reset.assert_not_awaited()
        # Sticky fallback state armed for the next turn's restore probe...
        assert slot._active_fallback_model == "fallback-model"
        assert slot._fallback_primary_model == "claude-test-model"
        # ...while the per-cycle walk state reset with the budgets.
        assert slot._transient_5xx_retries == 0
        assert slot._fallback_candidate_idx == 0
        assert slot._fallback_walked == []
        # Turn stats carry the model that ACTUALLY served the turn: the
        # substitute set_model updated the provider's bookkeeping (mimicked by
        # this mock exactly as AcpSessionHandle/AcpClient.set_model do), so the
        # read_turn_model seam — the sole source for meta.turn_stats.model
        # (see _attach_turn_stats, pinned by test_turn_stats.py) — reports the
        # candidate. The mocked provider yields no duration, so the stats dict
        # itself is not attached here; the model field's attach contract is
        # already pinned by test_turn_stats.py::test_model_included_when_resolved.
        from kiro_crew.dashboard.handlers.usage import read_turn_model

        assert read_turn_model(client) == "fallback-model"

    @pytest.mark.asyncio
    async def test_empty_chain_is_todays_terminal_error(self, tmp_path, monkeypatch):
        """REGRESSION PIN: with no chain configured, budget exhaustion surfaces
        the terminal ❌ exactly as before — no swap, no notice, no extra
        attempts."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES

        monkeypatch.setattr("kiro_crew.dashboard.chat_runner._agent_fallback_chain", lambda: ())
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        assert call_count == TRANSIENT_RETRIES + 1
        client.set_model.assert_not_awaited()
        assert not self._notice_texts(slot)
        assert any(t.startswith("❌") for t in self._err_texts(slot))
        assert slot._active_fallback_model == ""

    @pytest.mark.asyncio
    async def test_chain_exhaustion_surfaces_error_with_story(self, tmp_path, monkeypatch):
        """Every candidate also fails: the terminal ❌ carries the chain's
        story (primary throttled + which fallbacks were tried), and each
        candidate got exactly two attempts."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._agent_fallback_chain",
            lambda: ("fallback-model",),
        )
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # 4 on the primary + 2 on the candidate (initial + one retry).
        assert call_count == TRANSIENT_RETRIES + 3
        client.set_model.assert_awaited_once_with("fallback-model")
        errs = self._err_texts(slot)
        assert any(
            "fallback-model" in t and "also unavailable" in t for t in errs
        ), f"expected the chain story on the terminal error, got {errs!r}"
        # Per-cycle walk state refreshed for the next user-initiated cycle.
        assert slot._fallback_candidate_idx == 0
        assert slot._fallback_walked == []

    @pytest.mark.asyncio
    async def test_candidate_budget_routes_through_shared_rewind_body(self, tmp_path, monkeypatch):
        """DRIFT PIN: the post-swap counter rewind must come
        from llm_helpers.fallback_rewound_transient_budget — not a re-encoded
        local constant. Patching the shared body to grant no extra pass drops
        the candidate to a single attempt."""
        from kiro_crew.acp.client import AcpError
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.llm_helpers import TRANSIENT_RETRIES

        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner._agent_fallback_chain",
            lambda: ("fallback-model",),
        )
        # Rewind to the exhaustion threshold: the re-queued attempt runs, and
        # its failure immediately re-enters the fallback branch (no same-model
        # retry pass) — one attempt on the candidate instead of two.
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_runner.fallback_rewound_transient_budget",
            lambda: TRANSIENT_RETRIES,
        )
        call_count = 0

        async def _stream(msg):
            nonlocal call_count
            call_count += 1
            raise AcpError(self._TRANSIENT)
            yield  # pragma: no cover

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # 4 on the primary + only 1 on the candidate (vs 2 unpatched — see
        # test_chain_exhaustion_surfaces_error_with_story).
        assert call_count == TRANSIENT_RETRIES + 2
        client.set_model.assert_awaited_once_with("fallback-model")

    @pytest.mark.asyncio
    async def test_restore_probe_fires_on_next_genuine_turn(self, tmp_path, monkeypatch):
        """A sticky fallback from an earlier cycle is probed back to the
        primary at the start of the next genuine user turn — quietly (no
        chat card), clearing the sticky state on success."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        client._model = "fallback-model"
        client.served_model = "fallback-model"
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        slot._active_fallback_model = "fallback-model"
        slot._fallback_primary_model = "claude-test-model"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello again")
            await self._drain_bg(state)

        client.set_model.assert_awaited_once_with("claude-test-model")
        assert slot._active_fallback_model == ""
        assert slot._fallback_primary_model == ""
        # Quiet restore: no notice card for recovery.
        assert not self._notice_texts(slot)
        assert any("ok-result" in t for t in self._assistant_texts(slot))

    @pytest.mark.asyncio
    async def test_restore_probe_skipped_when_session_moved_on(self, tmp_path, monkeypatch):
        """If the session is not on our fallback (explicit user pick),
        the stale sticky state is dropped WITHOUT touching the model."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        client._model = "user-picked-model"
        client.served_model = "user-picked-model"
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        slot._active_fallback_model = "fallback-model"
        slot._fallback_primary_model = "claude-test-model"
        # A stale provider marker from the abandoned fallback must be dropped
        # WITH the slot state — a surviving marker would re-seed the long-dead
        # primary into a later, unrelated fallback walk.
        from kiro_crew.llm_helpers import TURN_FALLBACK_ATTR

        setattr(client, TURN_FALLBACK_ATTR, ("claude-test-model", "fallback-model"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello again")
            await self._drain_bg(state)

        client.set_model.assert_not_awaited()
        assert slot._active_fallback_model == ""
        assert getattr(client, TURN_FALLBACK_ATTR) is None

    @pytest.mark.asyncio
    async def test_restore_probe_honors_explicit_pick_of_the_fallback_itself(
        self, tmp_path, monkeypatch
    ):
        """REGRESSION (review finding): a user who EXPLICITLY picks the very
        model the session fell back to changes slot.model but not the served
        model — the probe must drop the sticky state without restoring the
        old primary over their pick."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        client._model = "fallback-model"
        client.served_model = "fallback-model"
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        # Fallback activated while the slot was configured on the primary...
        slot._active_fallback_model = "fallback-model"
        slot._fallback_primary_model = "claude-test-model"
        slot._fallback_slot_model = "claude-test-model"
        slot._fallback_pick_gen = slot._model_pick_gen
        # ...then the user explicitly picked the fallback model themselves
        # (the pick surface writes slot.model AND bumps the pick generation).
        slot.model = "fallback-model"
        slot._model_pick_gen += 1

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello again")
            await self._drain_bg(state)

        # No restore fired — the explicit pick wins — and the sticky state is gone.
        client.set_model.assert_not_awaited()
        assert slot._active_fallback_model == ""
        assert slot._fallback_slot_model == ""
        # The explicit pick itself is untouched.
        assert slot.model == "fallback-model"

    @pytest.mark.asyncio
    async def test_restore_probe_heals_automatic_backfill(self, tmp_path, monkeypatch):
        """REGRESSION (verifier finding): on an UNPINNED slot, a served-
        fallback value that reached slot.model without a pick-generation bump
        (pre-guard pollution) must NOT be misread as an explicit pick: the
        probe restores the primary, drops the sticky state, and heals
        slot.model back to the activation snapshot (else the fallback id would
        be re-sent as a pin on resume)."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        client._model = "fallback-model"
        client.served_model = "fallback-model"
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        # Fallback activated on an UNPINNED slot (empty snapshot, no pick)...
        slot._active_fallback_model = "fallback-model"
        slot._fallback_primary_model = "claude-test-model"
        slot._fallback_slot_model = ""
        slot._fallback_pick_gen = slot._model_pick_gen
        # ...then the automatic backfill wrote the served fallback into
        # slot.model WITHOUT bumping the pick generation.
        slot.model = "fallback-model"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello again")
            await self._drain_bg(state)

        # The restore fired (backfill is not a pick) and healed the slot pin.
        client.set_model.assert_awaited_once_with("claude-test-model")
        assert slot._active_fallback_model == ""
        assert slot.model == ""

    @pytest.mark.asyncio
    async def test_backfill_never_persists_the_fallback_candidate(self, tmp_path, monkeypatch):
        """REGRESSION (review finding, the span invariant): while a fallback is
        ACTIVE, the automatic model backfill must not write the served
        candidate into slot.model — slot.model is PERSISTED, and the in-memory
        sticky state is not, so a gateway restart would turn the temporary
        fallback into a permanent pin. An unpinned slot stays unpinned for the
        fallback's duration."""
        from kiro_crew.dashboard.chat import _run_chat
        from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_TEXT_CHUNK, LLMEvent

        async def _stream(msg):
            yield LLMEvent(kind=EVENT_TEXT_CHUNK, text="ok-result")
            yield LLMEvent(kind=EVENT_COMPLETE)

        state = self._make_state(tmp_path, monkeypatch)
        client = self._client(_stream)
        # The provider resolves to the FALLBACK candidate (it is serving).
        client._model = "fallback-model"
        client.served_model = "fallback-model"
        self._wire_sessions(state, client)
        slot = state.get_or_create_slot("s1")
        slot._titled = True
        slot.model = ""  # unpinned
        slot._active_fallback_model = "fallback-model"
        slot._fallback_primary_model = ""  # primary unknown -> no restore
        slot._fallback_pick_gen = slot._model_pick_gen

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await _run_chat(state, slot, "hello")
            await self._drain_bg(state)

        # The turn ran, but the persisted pin was never written with the
        # candidate. (The unknown-primary probe clears the sticky state — the
        # pin staying empty is the property under test.)
        assert slot.model == ""


class TestSlotProbeWrapsSharedRestoreBody:
    """DRIFT PIN: the slot restore probe is a thin adapter over
    llm_helpers.probe_fallback_restore — only the slot pick-gen/heal/clear
    logic lives locally. These tests pin the delegation and the slot-specific
    hooks it hands over; the probe mechanics themselves are pinned in
    test_llm_helpers.py and the end-to-end behavior in
    TestRunChatModelFallback above."""

    @staticmethod
    def _slot(**overrides):
        from types import SimpleNamespace

        defaults = dict(
            key="s1",
            model="",
            _model_pick_lock=None,
            _active_fallback_model="fb-1",
            _fallback_primary_model="primary-m",
            _fallback_slot_model="",
            _model_pick_gen=3,
            _fallback_pick_gen=3,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @pytest.mark.asyncio
    async def test_delegates_slot_state_to_the_shared_body(self):
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.llm_helpers import TURN_FALLBACK_ATTR

        slot = self._slot()
        client = MagicMock()
        setattr(client, TURN_FALLBACK_ATTR, ("primary-m", "fb-1"))
        with patch(
            "kiro_crew.dashboard.chat_runner.probe_fallback_restore", new_callable=AsyncMock
        ) as probe:
            await chat_runner._probe_fallback_restore_for_slot(slot, client)

        probe.assert_awaited_once()
        kwargs = probe.await_args.kwargs
        assert kwargs["surface"] == "dashboard"
        assert kwargs["state"] == ("primary-m", "fb-1")
        assert kwargs["stale"] is False
        assert kwargs["log_suffix"] == ", slot=s1"
        # The handed-over clear drops slot fields AND the provider marker as
        # one logical record.
        kwargs["clear"]()
        assert slot._active_fallback_model == ""
        assert slot._fallback_primary_model == ""
        assert getattr(client, TURN_FALLBACK_ATTR) is None

    @pytest.mark.asyncio
    async def test_explicit_pick_generation_bump_rides_in_as_stale(self):
        from kiro_crew.dashboard import chat_runner

        slot = self._slot(_model_pick_gen=4)  # picked after the swap
        with patch(
            "kiro_crew.dashboard.chat_runner.probe_fallback_restore", new_callable=AsyncMock
        ) as probe:
            await chat_runner._probe_fallback_restore_for_slot(slot, MagicMock())
        assert probe.await_args.kwargs["stale"] is True

    @pytest.mark.asyncio
    async def test_heal_hook_restores_the_activation_snapshot(self):
        from kiro_crew.dashboard import chat_runner

        slot = self._slot(model="fb-1", _fallback_slot_model="")  # backfilled pin
        with patch(
            "kiro_crew.dashboard.chat_runner.probe_fallback_restore", new_callable=AsyncMock
        ) as probe:
            await chat_runner._probe_fallback_restore_for_slot(slot, MagicMock())
        probe.await_args.kwargs["on_restored"]()
        assert slot.model == ""

    @pytest.mark.asyncio
    async def test_no_active_fallback_never_reaches_the_shared_body(self):
        from kiro_crew.dashboard import chat_runner

        slot = self._slot(_active_fallback_model="")
        with patch(
            "kiro_crew.dashboard.chat_runner.probe_fallback_restore", new_callable=AsyncMock
        ) as probe:
            await chat_runner._probe_fallback_restore_for_slot(slot, MagicMock())
        probe.assert_not_awaited()

    def test_sticky_clear_drops_marker_and_slot_fields(self):
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.llm_helpers import TURN_FALLBACK_ATTR

        slot = self._slot()
        client = MagicMock()
        setattr(client, TURN_FALLBACK_ATTR, ("primary-m", "fb-1"))
        chat_runner._clear_fallback_sticky_state(slot, client)
        assert getattr(client, TURN_FALLBACK_ATTR) is None
        assert slot._active_fallback_model == ""
        assert slot._fallback_primary_model == ""
        assert slot._fallback_slot_model == ""

    def test_sticky_clear_keeps_slot_record_when_marker_clear_fails(self):
        """DRIFT PIN: a failed provider-marker clear must NOT
        blank the slot fields — the slot probe keys on _active_fallback_model,
        so keeping the record is what makes the clear retryable next turn.
        Blanking the fields around a surviving marker would orphan it."""
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.llm_helpers import TURN_FALLBACK_ATTR

        class _HostileClient:
            @property
            def _kc_active_fallback(self):  # TURN_FALLBACK_ATTR
                raise RuntimeError("hostile marker")

        assert TURN_FALLBACK_ATTR == "_kc_active_fallback"
        slot = self._slot()
        chat_runner._clear_fallback_sticky_state(slot, _HostileClient())  # must not raise
        assert slot._active_fallback_model == "fb-1"
        assert slot._fallback_primary_model == "primary-m"


class TestSlotsGetWarmsGitLabAllowlist:
    """GET /api/chat/slots warms the self-managed GitLab allowlist first.

    Slot source-link extraction is synchronous and cannot load the allowlist, so
    a cold direct GET (no WebSocket yet) would serialize against the empty
    snapshot and omit every configured self-hosted MR chip. The WS path has its
    own test; this pins the direct-fetch path, which is a separate entry point.
    """

    @pytest.mark.asyncio
    async def test_cold_get_returns_the_authorized_self_hosted_link(self, monkeypatch) -> None:
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from kiro_crew.dashboard.chat_handlers import api_chat_slots
        from kiro_crew.dashboard.handlers import source_providers as sp

        url = "https://gitlab.acme.internal/team/api/-/merge_requests/7"
        order: list[str] = []

        # Cold snapshot: nothing has loaded the allowlist yet.
        monkeypatch.setattr(sp, "_gitlab_hosts_snapshot", frozenset())
        monkeypatch.setattr(sp, "_gitlab_hosts_loaded_at", 0.0)
        monkeypatch.setattr(sp, "_gitlab_hosts_generation", 0)

        async def fake_ensure() -> frozenset:
            order.append("ensure")
            sp._publish_provider_hosts(frozenset({"gitlab.acme.internal"}), frozenset())
            return frozenset({"gitlab.acme.internal"})

        monkeypatch.setattr(sp, "ensure_gitlab_hosts_loaded", fake_ensure)
        monkeypatch.setattr(sp, "schedule_check_refresh", lambda *a, **k: [])

        state = MagicMock()
        slot = _ChatSlot("s1")
        slot.append("assistant", f"Opened {url}", ts="t1")

        def serialize(**_kwargs):
            order.append("serialize")
            return [slot.to_dict()]

        state.serialize_slots.side_effect = serialize

        @web.middleware
        async def dashboard_auth_marker(request, handler):
            request["app"] = ""
            request["user"] = ""
            return await handler(request)

        app = web.Application(middlewares=[dashboard_auth_marker])
        app["state"] = state
        app.router.add_get("/api/chat/slots", api_chat_slots)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/chat/slots")
            assert resp.status == 200
            payloads = await resp.json()

        # Warm-up strictly precedes serialization, and the authorized MR is a link.
        assert order[:2] == ["ensure", "serialize"]
        links = [link["url"] for p in payloads for link in p.get("source_links", [])]
        assert url in links


class TestSourceLinkUrlsExcludeIssues:
    """The chip-status sweep reaches `gh pr view`, which has no meaning for an
    issue. Issue links must be filtered out before scheduling, not merely
    refused downstream, so a session that mentions only issues never generates
    provider work at all."""

    def test_state_sweep_skips_issue_links(self, tmp_path):
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("source")
        pr = "https://github.com/acme/widgets/pull/12"
        slot.append("assistant", f"{pr} and https://github.com/acme/widgets/issues/13")

        assert state.source_link_urls() == [pr]
        assert state.source_link_urls_for_slot("source") == [pr]

    @pytest.mark.asyncio
    async def test_slots_get_schedules_only_change_links(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.handlers import source_providers as sp

        scheduler = MagicMock(return_value=[])
        monkeypatch.setattr(sp, "schedule_check_refresh", scheduler)
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.owner_id = "U_OWNER"
        slot = state.get_or_create_slot("source")
        pr = "https://github.com/acme/widgets/pull/12"
        slot.append("assistant", f"{pr} and https://github.com/acme/widgets/issues/13")

        app = _make_app(state)

        @web.middleware
        async def owner_auth(request, handler):
            request["user"] = "U_OWNER"
            request["app"] = ""
            return await handler(request)

        app.middlewares.append(owner_auth)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/chat/slots")
            assert resp.status == 200
            payload = await resp.json()

        # Both links are serialized for the sidebar...
        links = next(item for item in payload if item["key"] == "source")["source_links"]
        assert len(links) == 2
        # ...but only the pull request is scheduled for a status read.
        assert scheduler.call_args.args[0] == [pr]


class TestBulkModelSwitch:
    """POST /api/chat/slots/model — switch every live slot to one model."""

    @staticmethod
    def _app(state: DashboardState) -> web.Application:
        from kiro_crew.dashboard.chat import api_chat_slots_model

        # Mirror production: token_auth middleware sets request["app"] on
        # every authenticated path ("" = dashboard user). Tests that model an
        # app-token caller insert their own middleware at index 0, which runs
        # first; this default only fills in the marker when absent.
        @web.middleware
        async def dashboard_auth_marker(request, handler):
            if "app" not in request:
                request["app"] = ""
            return await handler(request)

        app = web.Application(middlewares=[dashboard_auth_marker])
        app["state"] = state
        app.router.add_post("/api/chat/slots/model", api_chat_slots_model)
        return app

    @staticmethod
    def _mark_running(slot: _ChatSlot) -> None:
        """Give the slot a live task so the ``running`` property is True."""
        task = MagicMock()
        task.done.return_value = False
        slot.task = task

    @pytest.mark.asyncio
    async def test_switches_all_differing_slots_and_resets_each(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-opus-4.6")
        state.get_or_create_slot("b", model="claude-sonnet-4.6")
        state.push_slots_update = MagicMock()  # mock after creation: get_or_create_slot pushes

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "claude-opus-4.8"})
            data = await resp.json()

        assert resp.status == 200
        assert sorted(data["switched"]) == ["a", "b"]
        assert data["skipped_running"] == []
        assert state._slots["a"].model == "claude-opus-4.8"
        assert state._slots["b"].model == "claude-opus-4.8"
        assert state.sessions.reset.await_count == 2
        state.push_slots_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_running_slot_by_default(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.push_slots_update = MagicMock()
        idle = state.get_or_create_slot("idle", model="claude-opus-4.6")
        busy = state.get_or_create_slot("busy", model="claude-opus-4.6")
        self._mark_running(busy)

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "claude-opus-4.8"})
            data = await resp.json()

        assert resp.status == 200
        assert data["switched"] == ["idle"]
        assert data["skipped_running"] == ["busy"]
        assert idle.model == "claude-opus-4.8"
        assert busy.model == "claude-opus-4.6"  # untouched mid-turn
        assert state.sessions.reset.await_count == 1

    @pytest.mark.asyncio
    async def test_skip_running_false_forces_running_slot(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.push_slots_update = MagicMock()
        busy = state.get_or_create_slot("busy", model="claude-opus-4.6")
        self._mark_running(busy)

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/model",
                json={"model": "claude-opus-4.8", "skip_running": False},
            )
            data = await resp.json()

        assert resp.status == 200
        assert data["switched"] == ["busy"]
        assert data["skipped_running"] == []
        assert busy.model == "claude-opus-4.8"
        assert state.sessions.reset.await_count == 1

    @pytest.mark.asyncio
    async def test_already_on_target_is_unchanged_not_reset(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()  # mock after creation: get_or_create_slot pushes

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "claude-opus-4.8"})
            data = await resp.json()

        assert resp.status == 200
        assert data["unchanged"] == ["a"]
        assert data["switched"] == []
        state.sessions.reset.assert_not_awaited()
        state.push_slots_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_failure_is_isolated_and_reported(self, tmp_path):
        state = _make_state(tmp_path)
        # First reset (slot "a") succeeds, second ("b") raises. The failure must
        # be isolated: "a" still switches, "b" lands in `failed` with its old
        # model intact, and partial progress is still broadcast.
        state.sessions.reset = AsyncMock(side_effect=[None, RuntimeError("boom")])
        state.get_or_create_slot("a", model="claude-opus-4.6")
        state.get_or_create_slot("b", model="claude-opus-4.6")
        state.push_slots_update = MagicMock()  # mock after creation: get_or_create_slot pushes

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "claude-opus-4.8"})
            data = await resp.json()

        assert resp.status == 200
        assert data["switched"] == ["a"]
        assert data["failed"] == ["b"]
        assert state._slots["a"].model == "claude-opus-4.8"
        assert state._slots["b"].model == "claude-opus-4.6"  # untouched on reset failure
        state.push_slots_update.assert_called_once()  # partial progress still broadcast

    @pytest.mark.asyncio
    async def test_app_caller_only_switches_own_slots(self, tmp_path):
        """App Kit ownership isolation: an app token can only bulk-switch its
        own slots -- other apps' and the dashboard user's slots are untouched
        (mirrors api_chat_slots_cleanup)."""
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("mine", model="claude-opus-4.6", app="app-A")
        state.get_or_create_slot("theirs", model="claude-opus-4.6", app="app-B")
        state.get_or_create_slot("dashboard", model="claude-opus-4.6")
        state.push_slots_update = MagicMock()  # mock after creation: get_or_create_slot pushes

        @web.middleware
        async def inject_app(request, handler):
            request["app"] = "app-A"
            return await handler(request)

        app_obj = self._app(state)
        app_obj.middlewares.insert(0, inject_app)

        async with TestClient(TestServer(app_obj)) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "claude-opus-4.8"})
            data = await resp.json()

        assert resp.status == 200
        assert data["switched"] == ["mine"]
        assert state._slots["mine"].model == "claude-opus-4.8"
        # Non-owned slots: not switched, not reset, not even reported.
        assert state._slots["theirs"].model == "claude-opus-4.6"
        assert state._slots["dashboard"].model == "claude-opus-4.6"
        assert "theirs" not in data["unchanged"] + data["skipped_running"] + data["failed"]
        assert state.sessions.reset.await_count == 1

    @pytest.mark.asyncio
    async def test_missing_auth_marker_is_denied(self, tmp_path):
        """Deny-by-default: if the auth middleware never ran (request["app"]
        absent), the endpoint refuses instead of granting all-slot access."""
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-opus-4.6")

        from kiro_crew.dashboard.chat import api_chat_slots_model

        bare_app = web.Application()  # deliberately NO auth-marker middleware
        bare_app["state"] = state
        bare_app.router.add_post("/api/chat/slots/model", api_chat_slots_model)

        async with TestClient(TestServer(bare_app)) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 403
        assert state._slots["a"].model == "claude-opus-4.6"
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falsy_non_string_marker_fails_closed(self, tmp_path):
        """A buggy middleware setting request["app"] to a falsy non-string
        (None) must NOT be treated as a dashboard user -- the ownership check
        applies and no slot matches, so nothing is switched."""
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-opus-4.6")

        from kiro_crew.dashboard.chat import api_chat_slots_model

        @web.middleware
        async def buggy_auth_marker(request, handler):
            request["app"] = None  # falsy, but NOT the explicit "" dashboard marker
            return await handler(request)

        app_obj = web.Application(middlewares=[buggy_auth_marker])
        app_obj["state"] = state
        app_obj.router.add_post("/api/chat/slots/model", api_chat_slots_model)

        async with TestClient(TestServer(app_obj)) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "claude-opus-4.8"})
            data = await resp.json()

        assert resp.status == 200
        assert data["switched"] == []
        assert state._slots["a"].model == "claude-opus-4.6"
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_boolean_skip_running_is_rejected(self, tmp_path):
        state = _make_state(tmp_path)
        state.get_or_create_slot("a", model="claude-opus-4.6")

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/model",
                json={"model": "claude-opus-4.8", "skip_running": "yes"},
            )

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_canonical_key_rejected_no_slot_touched(self, tmp_path):
        # A canonical registry key (the /api/models cold-start fallback trap)
        # is rejected 400 for the acp provider; no slot is switched or reset.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-fable-5")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/model", json={"model": "fable-5-1m"})
            data = await resp.json()

        assert resp.status == 400
        assert "fable-5-1m" in data["error"]
        assert state._slots["a"].model == "claude-fable-5"
        state.sessions.reset.assert_not_awaited()


class TestSlotModelGuard:
    """POST /api/chat/slots/{slot}/model — reject canonical registry keys the
    ACP CLI cannot accept (the /api/models cold-start fallback -32603 trap)."""

    @staticmethod
    def _app(state: DashboardState) -> web.Application:
        from kiro_crew.dashboard.chat import api_chat_slot_model

        app = web.Application()
        app["state"] = state
        app.router.add_post("/api/chat/slots/{slot}/model", api_chat_slot_model)
        return app

    @pytest.mark.asyncio
    async def test_canonical_key_rejected_slot_unchanged(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-fable-5")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "fable-5-1m"})
            data = await resp.json()

        assert resp.status == 400
        assert "fable-5-1m" in data["error"]
        # The slot keeps its valid model and the session is NOT reset.
        assert state._slots["a"].model == "claude-fable-5"
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_auto_default_is_allowed(self, tmp_path):
        # The frontend maps 'auto' -> '' before calling; '' is the provider
        # default and always passes.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-fable-5")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": ""})

        assert resp.status == 200
        assert state._slots["a"].model == ""
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_literal_is_allowed(self, tmp_path):
        # A literal 'auto' (registry key, but the safe sentinel) always passes.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-fable-5")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "auto"})

        assert resp.status == 200
        assert state._slots["a"].model == "auto"

    @pytest.mark.asyncio
    async def test_valid_kiro_alias_is_allowed(self, tmp_path):
        # kiro/acp ids (registry aliases, not top-level keys) pass through.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.get_or_create_slot("a", model="claude-fable-5")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        assert state._slots["a"].model == "claude-opus-4.8"
        state.sessions.reset.assert_awaited_once()


class TestSlotModelLiveSwitch:
    """POST /api/chat/slots/{slot}/model — prefer an in-place session/set_model
    over tearing the session down.

    A reset costs a full process-tree teardown now plus a cold start and
    transcript replay on the next message. kiro-cli's ``session/set_model``
    switches a live session instead (verified against 2.15.1: acked
    synchronously, conversation carried across the switch, sticky over later
    turns), so the reset is only a fallback.
    """

    @staticmethod
    def _app(state: DashboardState) -> web.Application:
        from kiro_crew.dashboard.chat import api_chat_slot_model

        app = web.Application()
        app["state"] = state
        app.router.add_post("/api/chat/slots/{slot}/model", api_chat_slot_model)
        return app

    @staticmethod
    def _provider(
        *,
        claude: bool = False,
        active_turn: bool = False,
        models=("auto",),
        supports_effort: bool = False,
        change_effort: bool = True,
    ):
        """A live AcpProvider double. ``spec=`` keeps isinstance() working.

        Carries the real capability record for the backend it stands in for:
        ``capabilities_of`` requires a genuine ``SessionCapabilities``, so an
        attribute-shaped flag on a spec'd mock would claim every capability at once.
        """
        from kiro_crew.providers.acp import AcpProvider

        provider = MagicMock(spec=AcpProvider)
        provider.capabilities = capabilities_for(ACP_BACKEND_CLAUDE if claude else ACP_BACKEND_KIRO)
        provider.has_active_turn.return_value = active_turn
        provider.available_models.return_value = [{"modelId": m} for m in models]
        provider.supports_effort.return_value = supports_effort
        provider.change_effort = AsyncMock(return_value=change_effort)
        provider.clear_effort = AsyncMock(return_value=False)
        provider.client = MagicMock()
        provider.client.set_model = AsyncMock()
        return provider

    @pytest.mark.asyncio
    async def test_idle_switch_goes_live_without_reset(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        assert state._slots["a"].model == "gpt-5.6-sol"
        provider.client.set_model.assert_awaited_once_with("gpt-5.6-sol")
        # The whole point: the session survives.
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_canonical_key_is_translated_to_the_acp_id(self, tmp_path):
        # slot.model is a canonical/wire value; set_model only accepts kiro ids.
        from kiro_crew import model_registry

        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.6")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        sent = provider.client.set_model.await_args.args[0]
        assert sent == model_registry.to_acp_id("claude-opus-4.8")
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_active_turn_answers_409_without_reset(self, tmp_path):
        # Awaiting a response mid-turn would race the streaming prompt loop on
        # stdout for the non-multiplexed client (same hazard the effort handler
        # documents), and the old reset fallback tore down the in-flight turn
        # mid-stream for any programmatic caller. A turn in flight now answers
        # busy: no live switch, no reset, slot model untouched.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(active_turn=True)
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})
            data = await resp.json()

        assert resp.status == 409
        assert data["code"] == "turn_in_flight"
        provider.client.set_model.assert_not_awaited()
        state.sessions.reset.assert_not_awaited()
        assert state._slots["a"].model == "claude-opus-4.8"

    @pytest.mark.asyncio
    async def test_live_switch_failure_falls_back_to_reset(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        provider.client.set_model = AsyncMock(side_effect=RuntimeError("boom"))
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        # The slot still lands on the new model, via the reset path.
        assert resp.status == 200
        assert state._slots["a"].model == "gpt-5.6-sol"
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_uses_the_advertised_auto_id(self, tmp_path):
        # Back-to-default is the most common click; kiro expresses it as a real
        # model id, so it should not cost a teardown either.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(models=("auto", "claude-opus-4.8"))
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": ""})

        assert resp.status == 200
        assert state._slots["a"].model == ""
        provider.client.set_model.assert_awaited_once_with("auto")
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_without_advertised_auto_falls_back_to_reset(self, tmp_path):
        # Never invent an id the backend did not advertise.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(models=("claude-opus-4.8",))
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": ""})

        assert resp.status == 200
        provider.client.set_model.assert_not_awaited()
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_claude_backend_default_falls_back_to_reset(self, tmp_path):
        # The claude backend has no "let the server choose" id.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(claude=True)
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": ""})

        assert resp.status == 200
        provider.client.set_model.assert_not_awaited()
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_claude_backend_switch_uses_provider_id(self, tmp_path):
        from kiro_crew import model_registry

        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(claude=True)
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.6")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        sent = provider.client.set_model.await_args.args[0]
        assert sent == model_registry.to_provider_id("claude-opus-4.8", "claude_code")
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_live_session_still_routes_through_reset(self, tmp_path):
        # No session to update: the reset is an O(1) no-op teardown, but it also
        # carries _reset_slot_session's pending-wait cleanup, so keep taking it.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.sessions.get_provider = MagicMock(return_value=None)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        assert state._slots["a"].model == "gpt-5.6-sol"
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unchanged_model_touches_nothing(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="gpt-5.6-sol")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        provider.client.set_model.assert_not_awaited()
        state.sessions.reset.assert_not_awaited()

    # ── reasoning effort ─────────────────────────────────────────────
    # The kiro effort overlay is written at (re)spawn, so a cold start applies
    # the slot's level for free. An in-place switch never respawns, so it must
    # push the level itself or the new model silently runs at its own default
    # while the UI still reports the slot's level.

    @pytest.mark.asyncio
    async def test_persisted_effort_is_pushed_to_the_new_model(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(supports_effort=True)
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="claude-haiku-4.5")
        slot.reasoning_effort = "high"
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        provider.change_effort.assert_awaited_once_with("high")
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_effort_push_rejected_falls_back_to_reset(self, tmp_path):
        # change_effort returning False means the level never reached the
        # session — reset so the cold start applies it via the overlay.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(supports_effort=True, change_effort=False)
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="claude-haiku-4.5")
        slot.reasoning_effort = "high"
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_effort_push_raising_falls_back_to_reset(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(supports_effort=True)
        provider.change_effort = AsyncMock(side_effect=RuntimeError("rejected"))
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="claude-haiku-4.5")
        slot.reasoning_effort = "high"
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        state.sessions.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_effort_incapable_target_keeps_the_live_switch(self, tmp_path):
        # Switching to a model with no effort selector is a persisted no-op for
        # effort — it must not cost a teardown.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(supports_effort=False)
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="claude-opus-4.8")
        slot.reasoning_effort = "high"
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-haiku-4.5"})

        assert resp.status == 200
        provider.change_effort.assert_not_awaited()
        # The level stays persisted for a later switch back to a capable model.
        assert state._slots["a"].reasoning_effort == "high"
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_effort_override_re_resolves_without_resetting(self, tmp_path):
        # With no slot override, clear_effort re-resolves any workspace default.
        # Its False return is benign here (nothing to push, nothing stale for a
        # model the user never set a level on), so it must not force a reset.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(supports_effort=True)
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="claude-haiku-4.5")
        slot.reasoning_effort = ""
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        provider.clear_effort.assert_awaited_once()
        provider.change_effort.assert_not_awaited()
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unavailable_model_is_4xx_and_keeps_the_session(self, tmp_path):
        """An entitlement refusal must NOT take the reset fallback.

        The generic ``except Exception`` here treats
        every set_model failure as "the call didn't land" and recovers with a
        reset. For a model the account cannot use that recovery is wrong twice
        over — it destroys the live conversation AND cold-starts on a different
        model, while the handler still answers ok:True. The slot must also keep
        its previous model, or the picker asserts a model that was refused.
        """
        from kiro_crew.acp.client import AcpModelUnavailable

        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        provider.client.set_model = AsyncMock(
            side_effect=AcpModelUnavailable(
                "claude-opus-4.8", ["gpt-5.6-sol"], backend=ACP_BACKEND_KIRO
            )
        )
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="gpt-5.6-sol")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})
            body = await resp.json()

        assert resp.status == 400
        assert "not available on your account" in body["error"]
        # Names what the account CAN use.
        assert "gpt-5.6-sol" in body["error"]
        # The conversation survives — no reset fallback.
        state.sessions.reset.assert_not_awaited()
        # And the slot still reports the model that is actually running.
        assert state._slots["a"].model == "gpt-5.6-sol"

    @pytest.mark.asyncio
    async def test_refused_pick_rolls_back_the_pick_generation(self, tmp_path):
        """REGRESSION (review finding): a refused pick changed nothing, so the
        explicit-pick generation must roll back with the model. Leaving the
        bump in place would make the model-fallback restore probe read the
        FAILED attempt as an explicit choice and silently abandon restoring
        the primary — the session would stay on the fallback forever with no
        card and no probe."""
        from kiro_crew.acp.client import AcpModelUnavailable

        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        provider.client.set_model = AsyncMock(
            side_effect=AcpModelUnavailable(
                "claude-opus-4.8", ["gpt-5.6-sol"], backend=ACP_BACKEND_KIRO
            )
        )
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="gpt-5.6-sol")
        state.push_slots_update = MagicMock()
        gen_before = slot._model_pick_gen

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 400
        assert slot._model_pick_gen == gen_before

    @pytest.mark.asyncio
    async def test_refused_pick_rollback_spares_a_newer_concurrent_pick(self, tmp_path):
        """REGRESSION (review finding on eb3cf067): two picks arriving together
        must not roll back each other's state. The pick transaction is
        serialised per slot (_model_pick_lock), so the refused pick unwinds
        cleanly before the next pick's snapshot is taken — the surviving state
        is the later, successful pick's."""
        import asyncio

        from kiro_crew.acp.client import AcpModelUnavailable

        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        slot = state.get_or_create_slot("a", model="gpt-5.6-sol")
        state.push_slots_update = MagicMock()

        release_slow = asyncio.Event()

        async def _set_model(target):
            if target == "claude-opus-4.8":
                # The SLOW, ultimately-refused pick: park inside the locked
                # transaction, then refuse.
                await release_slow.wait()
                raise AcpModelUnavailable(
                    "claude-opus-4.8", ["gpt-5.6-sol"], backend=ACP_BACKEND_KIRO
                )
            return None

        provider.client.set_model = AsyncMock(side_effect=_set_model)
        state.sessions.get_provider = MagicMock(return_value=provider)
        gen_before = slot._model_pick_gen

        async with TestClient(TestServer(self._app(state))) as client:
            slow = asyncio.create_task(
                client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})
            )
            await asyncio.sleep(0.05)  # slow pick holds the lock, parked at its await
            fast = asyncio.create_task(
                client.post("/api/chat/slots/a/model", json={"model": "deepseek-3.2"})
            )
            await asyncio.sleep(0.05)  # fast pick queued on the lock
            release_slow.set()
            slow_resp = await slow
            fast_resp = await fast

        assert slow_resp.status == 400
        assert fast_resp.status == 200
        # The successful pick's state survives: the refused pick's rollback
        # unwound only its own speculative writes.
        assert slot.model == "deepseek-3.2"
        assert slot._model_pick_gen == gen_before + 1

    @pytest.mark.asyncio
    async def test_two_refused_picks_leave_the_slot_untouched(self, tmp_path):
        """Two UNAVAILABLE picks overlapping must not fail out of order: without
        serialisation the later rollback would restore
        the earlier pick's already-refused model, leaving the slot advertising
        a rejected model with a phantom pick generation. Serialised, each
        transaction unwinds itself and the slot ends exactly where it began."""
        import asyncio

        from kiro_crew.acp.client import AcpModelUnavailable

        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        slot = state.get_or_create_slot("a", model="gpt-5.6-sol")
        state.push_slots_update = MagicMock()

        release_first = asyncio.Event()
        release_second = asyncio.Event()

        async def _set_model(target):
            # Park each pick independently so the FIRST refusal can complete
            # while the SECOND is still pending — the out-of-order shape that
            # made the pre-lock CAS restore an already-refused model.
            if target == "claude-opus-4.8":
                await release_first.wait()
            else:
                await release_second.wait()
            raise AcpModelUnavailable(target, ["gpt-5.6-sol"], backend=ACP_BACKEND_KIRO)

        provider.client.set_model = AsyncMock(side_effect=_set_model)
        state.sessions.get_provider = MagicMock(return_value=provider)
        gen_before = slot._model_pick_gen

        async with TestClient(TestServer(self._app(state))) as client:
            first = asyncio.create_task(
                client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})
            )
            await asyncio.sleep(0.05)
            second = asyncio.create_task(
                client.post("/api/chat/slots/a/model", json={"model": "deepseek-3.2"})
            )
            await asyncio.sleep(0.05)
            # First refusal completes while the second pick is still pending.
            release_first.set()
            first_resp = await first
            release_second.set()
            second_resp = await second

        assert first_resp.status == 400
        assert second_resp.status == 400
        # No phantom state: the slot still reports what is actually running.
        assert slot.model == "gpt-5.6-sol"
        assert slot._model_pick_gen == gen_before

    @pytest.mark.asyncio
    async def test_successful_pick_bumps_the_pick_generation(self, tmp_path):
        """The counterpart pin: a pick that LANDS must keep its bump — that is
        what stops the fallback restore probe from overriding the choice."""
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="gpt-5.6-sol")
        state.push_slots_update = MagicMock()
        gen_before = slot._model_pick_gen

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "claude-opus-4.8"})

        assert resp.status == 200
        assert slot._model_pick_gen == gen_before + 1

    @pytest.mark.asyncio
    async def test_same_value_pick_still_bumps_the_pick_generation(self, tmp_path):
        """REGRESSION (review finding): picking the model the slot already
        reports takes the equality early-return, but it is still an EXPLICIT
        affirmation — without the bump, a user who deliberately re-picks the
        very model the session fell back to would have that choice silently
        overridden by the next fallback restore probe."""
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="gpt-5.6-sol")
        state.push_slots_update = MagicMock()
        gen_before = slot._model_pick_gen

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        assert slot._model_pick_gen == gen_before + 1
        # Same-value: no switch, no reset.
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_value_pick_during_fallback_takes_the_live_switch_path(self, tmp_path):
        """REGRESSION (review finding on 1a61ddcf): with a fallback actively
        serving the session, a pick of the DISPLAYED primary equals slot.model
        but the wire model is the fallback — the equality early-return would
        record the pick and switch nothing, stranding the session on the
        fallback while usage is attributed to the primary. The live-switch
        path must run instead."""
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider()
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="gpt-5.6-sol")
        slot._active_fallback_model = "fallback-model"
        state.push_slots_update = MagicMock()
        gen_before = slot._model_pick_gen

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        # The pick is recorded AND the switch actually went to the wire.
        assert slot._model_pick_gen == gen_before + 1
        provider.client.set_model.assert_awaited_once_with("gpt-5.6-sol")


class TestSlotModelSwitchContextBroadcast:
    """POST /api/chat/slots/{slot}/model — one ``context_usage`` event per
    switch so the meter updates immediately.

    Regression: nothing broadcast on a model switch, so the frontend's stored
    ``slotContextTokens`` kept the OLD model's {used, window} until the next
    turn. The event carries ``reset: true`` so the reducer may replace or
    delete the stored entry (per-turn events never delete).
    """

    _app = staticmethod(TestSlotModelLiveSwitch._app)
    _provider = staticmethod(TestSlotModelLiveSwitch._provider)

    @staticmethod
    def _find_context_events(broadcast_mock):
        return [
            call.args[1]
            for call in broadcast_mock.call_args_list
            if call.args and call.args[0] == "context_usage"
        ]

    @pytest.mark.asyncio
    async def test_live_switch_broadcasts_rebased_stats(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.broadcast_ws = MagicMock()
        provider = self._provider()
        # The provider accessors report the freshly rebased stats set_model left.
        provider.context_usage_pct.return_value = 36.8
        provider.context_used_tokens.return_value = 100_000
        provider.context_window_tokens.return_value = 272_000
        state.sessions.get_provider = MagicMock(return_value=provider)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        events = self._find_context_events(state.broadcast_ws)
        assert events == [
            {
                "slot": state._slots["a"].key,
                "pct": 36.8,
                "used_tokens": 100_000,
                "window_tokens": 272_000,
                "reset": True,
            }
        ]

    @pytest.mark.asyncio
    async def test_live_switch_to_auto_refreshes_the_served_model_cache(self, tmp_path):
        """REGRESSION (GPT finding on 33ef1122e): ``slot.served_model`` was
        written only at spawn, so selecting Auto on a warm, preserved session
        left the concrete id the session was spawned with in the cache and the
        composer chip kept naming the OLD model as "the default". After the live
        switch lands the cache follows the provider -- ``""`` here, because
        AcpProvider collapses the ``auto`` sentinel (chip reads "auto")."""
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(models=("auto", "claude-opus-4.8"))
        provider.served_model = ""
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="claude-opus-4.8")
        slot.record_served_model("claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "auto"})

        assert resp.status == 200
        provider.client.set_model.assert_awaited_once_with("auto")
        state.sessions.reset.assert_not_awaited()
        assert slot.served_model == ""

    @pytest.mark.asyncio
    async def test_live_switch_to_a_pin_refreshes_the_served_model_cache(self, tmp_path):
        # Pin -> pin: the cache names the model the live session runs NOW.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        provider = self._provider(models=("auto", "claude-opus-4.8", "gpt-5.6-sol"))
        provider.served_model = "gpt-5.6-sol"
        state.sessions.get_provider = MagicMock(return_value=provider)
        slot = state.get_or_create_slot("a", model="claude-opus-4.8")
        slot.record_served_model("claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        provider.client.set_model.assert_awaited_once_with("gpt-5.6-sol")
        assert slot.served_model == "gpt-5.6-sol"

    @pytest.mark.asyncio
    async def test_reset_fallback_broadcasts_token_clearing_event(self, tmp_path):
        # No live provider -> the reset path: no token counts, so the frontend
        # deletes its stored entry and falls back to the model-derived window.
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.broadcast_ws = MagicMock()
        state.sessions.get_provider = MagicMock(return_value=None)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        state.sessions.reset.assert_awaited_once()
        events = self._find_context_events(state.broadcast_ws)
        assert events == [{"slot": state._slots["a"].key, "pct": 0.0, "reset": True}]

    @pytest.mark.asyncio
    async def test_broadcast_failure_does_not_fail_the_switch(self, tmp_path):
        state = _make_state(tmp_path)
        state.sessions.reset = AsyncMock()
        state.broadcast_ws = MagicMock(side_effect=RuntimeError("ws down"))
        state.sessions.get_provider = MagicMock(return_value=None)
        state.get_or_create_slot("a", model="claude-opus-4.8")
        state.push_slots_update = MagicMock()

        async with TestClient(TestServer(self._app(state))) as client:
            resp = await client.post("/api/chat/slots/a/model", json={"model": "gpt-5.6-sol"})

        assert resp.status == 200
        assert state._slots["a"].model == "gpt-5.6-sol"


def _make_tail_fork_slot(state):
    slot = state.get_or_create_slot("src")
    slot.title = "My Chat"
    slot._titled = True
    slot.append("user", "msg1", "msg msg-u")
    slot.append("assistant", "reply1", "msg msg-a")
    slot.append("user", "msg2", "msg msg-u")
    slot.append("assistant", "reply2", "msg msg-a")
    slot.drain()
    return slot


class TestForkSlotTail:
    """Tests for tail-only fork (direction="tail") on POST /api/chat/slots/{slot}/fork."""

    @pytest.fixture(autouse=True)
    def _tail_fork_enabled(self, monkeypatch):
        """Default tail_fork_enabled=True so these tests exercise the tail path;
        the B1 gate test below overrides this to False for its own assertion."""
        mock_cfg = MagicMock()
        mock_cfg.dashboard.tail_fork_enabled = True
        monkeypatch.setattr("kiro_crew.dashboard.chat_fork.KiroCrewConfig.load", lambda: mock_cfg)

    @pytest.mark.asyncio
    async def test_tail_fork_keeps_messages_after_message_id(self, tmp_path):
        state = _make_state(tmp_path)
        slot = _make_tail_fork_slot(state)
        target_id = slot.messages[1]["meta"]["mid"]

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_id": target_id, "direction": "tail"},
            )

            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == 2
            assert data["direction"] == "tail"

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert [m["content"] for m in visible] == ["msg2", "reply2"]

    @pytest.mark.asyncio
    async def test_tail_fork_keeps_messages_after_index(self, tmp_path):
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_index": 1, "direction": "tail"},
            )

            assert resp.status == 200
            data = await resp.json()
            assert data["messages"] == 2
            assert data["direction"] == "tail"

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert len(visible) == 2
        assert visible[0]["content"] == "msg2"
        assert visible[-1]["content"] == "reply2"

    @pytest.mark.asyncio
    async def test_tail_fork_discard_has_no_summary(self, tmp_path):
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_index": 1, "direction": "tail"},
            )

            assert resp.status == 200
            data = await resp.json()

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert all((m.get("meta") or {}).get("source") != "head_summary" for m in visible)

    @pytest.mark.asyncio
    async def test_tail_fork_requires_at_index(self, tmp_path):
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"direction": "tail"})

            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_tail_fork_nothing_after_last_message(self, tmp_path):
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_index": 3, "direction": "tail"},
            )

            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_tail_fork_title_and_provenance(self, tmp_path):
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_index": 1, "direction": "tail"},
            )

            assert resp.status == 200
            data = await resp.json()
            assert data["title"] == "↳ Tail of My Chat"

        new_slot = state._slots.get(data["key"])
        assert new_slot.forked_from == "dashboard:src"

    @pytest.mark.asyncio
    async def test_default_direction_is_head(self, tmp_path):
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"at_message_index": 1})

            assert resp.status == 200
            data = await resp.json()
            assert data["direction"] == "head"
            assert data["messages"] == 2

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert visible[-1]["content"] == "reply1"

    @pytest.mark.asyncio
    async def test_invalid_direction_rejected(self, tmp_path):
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/src/fork", json={"direction": "sideways"})

            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_tail_fork_gated_off_falls_back_to_head(self, tmp_path, monkeypatch):
        """B1 gate (D1): tail_fork_enabled=False silently downgrades tail -> head."""
        mock_cfg = MagicMock()
        mock_cfg.dashboard.tail_fork_enabled = False
        monkeypatch.setattr("kiro_crew.dashboard.chat_fork.KiroCrewConfig.load", lambda: mock_cfg)
        state = _make_state(tmp_path)
        _make_tail_fork_slot(state)

        app = _make_app(state)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat/slots/src/fork",
                json={"at_message_index": 1, "direction": "tail"},
            )

            assert resp.status == 200
            data = await resp.json()
            assert data["direction"] == "head"
            assert data["messages"] == 2

        new_slot = state._slots.get(data["key"])
        visible = [m for m in new_slot.messages if m["role"] in ("user", "assistant")]
        assert visible[-1]["content"] == "reply1"


# ── Session reload endpoint (POST /api/chat/slots/{slot}/reload) ──


class TestSessionReload:
    """api_chat_slot_reload — relaunch the slot's agent process in place.

    The endpoint's contract: refuse while a turn is in flight (409), otherwise
    tear down via reset(skip_if_busy=True), append the session_reload feed
    notice, and eagerly re-arm the resume spawn.
    """

    @staticmethod
    def _idle_provider():
        provider = MagicMock()
        provider.has_active_turn = MagicMock(return_value=False)
        return provider

    @pytest.mark.asyncio
    async def test_reload_unknown_slot_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/nope/reload")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_reload_resets_notices_and_respawns(self, tmp_path, monkeypatch):
        """Happy path: atomic reset, feed notice tagged session_reload, eager resume."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        state.sessions.get_provider = MagicMock(return_value=self._idle_provider())
        state.sessions.reset = AsyncMock(return_value=True)
        eager = MagicMock(return_value=None)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", eager)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")
            data = await resp.json()

        assert resp.status == 200
        assert data == {"ok": True}
        state.sessions.reset.assert_awaited_once_with("dashboard:s1", skip_if_busy=True)
        # The feed carries the durable confirmation, tagged so the follow-up
        # options / continuable scans skip it (isSystemNoticeKind frontend twin).
        notice = slot.messages[-1]
        assert notice["role"] == "assistant"
        assert notice["meta"]["kind"] == "session_reload"
        assert "Reloading session" in notice["content"]
        eager.assert_called_once()
        assert eager.call_args.kwargs.get("allow_resume") is True

    @pytest.mark.asyncio
    async def test_reload_refused_while_turn_in_flight(self, tmp_path, monkeypatch):
        """409 with no teardown, no notice, no eager spawn while a turn runs."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        busy = MagicMock()
        busy.has_active_turn = MagicMock(return_value=True)
        state.sessions.get_provider = MagicMock(return_value=busy)
        state.sessions.reset = AsyncMock()
        eager = MagicMock(return_value=None)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", eager)
        before = len(slot.messages)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")

        assert resp.status == 409
        state.sessions.reset.assert_not_awaited()
        assert len(slot.messages) == before
        eager.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_race_turn_started_after_fast_path(self, tmp_path, monkeypatch):
        """A turn slipping in between the fast path and the atomic guard is a 409.

        reset(skip_if_busy=True) returning False is ambiguous (no session OR
        busy); the re-check of has_active_turn is what disambiguates. Nothing
        may be appended or respawned for the busy case.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        idle = self._idle_provider()
        busy = MagicMock()
        busy.has_active_turn = MagicMock(return_value=True)
        # First lookup (fast path) sees idle, as does the teardown's
        # pre-reset identity capture; the post-reset re-check sees busy.
        state.sessions.get_provider = MagicMock(side_effect=[idle, idle, busy])
        state.sessions.reset = AsyncMock(return_value=False)
        eager = MagicMock(return_value=None)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", eager)
        before = len(slot.messages)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")

        assert resp.status == 409
        assert len(slot.messages) == before
        eager.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_retries_when_racing_turn_already_finished(self, tmp_path, monkeypatch):
        """A declined reset whose racing turn already FINISHED retries once.

        Falling through would report success while the stale live process
        survives untouched -- the silent false-success the endpoint exists to
        prevent. The retry succeeds here; a second decline is the 409 case
        (covered below).
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        state.sessions.get_provider = MagicMock(return_value=self._idle_provider())
        state.sessions.reset = AsyncMock(side_effect=[False, True])
        eager = MagicMock(return_value=None)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", eager)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")
            data = await resp.json()

        assert resp.status == 200
        assert data == {"ok": True}
        assert state.sessions.reset.await_count == 2
        assert slot.messages[-1]["meta"]["kind"] == "session_reload"
        eager.assert_called_once()

    @pytest.mark.asyncio
    async def test_reload_double_decline_is_turn_in_flight(self, tmp_path, monkeypatch):
        """Two consecutive declines with a live session mean a genuinely racing
        turn: 409, no notice, no respawn -- never a false success."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        state.sessions.get_provider = MagicMock(return_value=self._idle_provider())
        state.sessions.reset = AsyncMock(return_value=False)
        eager = MagicMock(return_value=None)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", eager)
        before = len(slot.messages)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")
            data = await resp.json()

        assert resp.status == 409
        assert data["code"] == "turn_in_flight"
        assert state.sessions.reset.await_count == 2
        assert len(slot.messages) == before
        eager.assert_not_called()

    def test_system_notice_kinds_backend_frontend_parity(self):
        """The two SYSTEM_NOTICE_KINDS twins must hold the same set.

        A kind skipped on one side but not the other leaves the sidebar showing
        notice boilerplate while the chat pane shows the real turn (or vice
        versa) -- the exact divergence the shared predicates exist to prevent.
        Asserted on the TS source because nothing else couples the two files.
        """
        import re
        from pathlib import Path

        from kiro_crew.dashboard.system_notices import SYSTEM_NOTICE_KINDS

        ts_path = Path(__file__).resolve().parents[1] / "website/src/lib/systemNotice.ts"
        ts_src = ts_path.read_text(encoding="utf-8")
        m = re.search(r"new Set\(\[([^\]]*)\]\)", ts_src)
        assert m, "SYSTEM_NOTICE_KINDS Set literal not found in systemNotice.ts"
        frontend = set(re.findall(r"'([^']+)'", m.group(1)))
        assert frontend == set(SYSTEM_NOTICE_KINDS)

    @pytest.mark.asyncio
    async def test_reload_without_live_session_is_ok(self, tmp_path, monkeypatch):
        """No live session: reload is a no-op teardown but still ok + notice + respawn.

        The goal state (next spawn picks up fresh config) already holds, so
        this must not be an error; reloaded=False tells the caller no process
        was actually torn down.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        state.sessions.get_provider = MagicMock(return_value=None)
        state.sessions.reset = AsyncMock(return_value=False)
        eager = MagicMock(return_value=None)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", eager)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")
            data = await resp.json()

        assert resp.status == 200
        assert data == {"ok": True}
        assert slot.messages[-1]["meta"]["kind"] == "session_reload"
        eager.assert_called_once()

    @pytest.mark.asyncio
    async def test_reload_cancels_pending_question_waits(self, tmp_path, monkeypatch):
        """A pending ask_question wait is released by the reload teardown.

        The card lives in dashboard state, not the session — without the
        unblock it would survive the reset and hold its MCP worker until
        timeout with no agent left to receive the answer.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")
        state.sessions.get_provider = MagicMock(return_value=self._idle_provider())
        state.sessions.reset = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.schedule_eager_spawn",
            MagicMock(return_value=None),
        )
        cancelled = MagicMock(return_value=1)
        monkeypatch.setattr(state, "cancel_questions_for_slot", cancelled)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")

        assert resp.status == 200
        cancelled.assert_called_once_with("s1")

    @pytest.mark.asyncio
    async def test_reload_app_token_denied_on_foreign_slot(self, tmp_path, monkeypatch):
        """An app token gets the indistinguishable 404, and nothing is torn down.

        Reload is a teardown; without the cancel-route ownership policy, an app
        allowed /api/chat/* could reset a foreign slot's process and cancel its
        pending waits.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot._app = "other-app"
        state.sessions.get_provider = MagicMock(return_value=None)
        state.sessions.reset = AsyncMock()
        app = _make_app_with_agent_routes(state)

        @web.middleware
        async def _as_app(request, handler):
            request["app"] = "attacker-app"
            return await handler(request)

        app.middlewares.append(_as_app)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat/slots/s1/reload")
            assert resp.status == 404
            assert (await resp.json())["code"] == "slot_not_found"
        state.sessions.reset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reload_targets_the_linked_session_key(self, tmp_path, monkeypatch):
        """A channel-born slot reloads its LINKED session, not the phantom
        dashboard-prefixed key — otherwise reload reports success while the
        live process keeps its stale config.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.linked_session_key = "slack:123.456"
        state.sessions.get_provider = MagicMock(return_value=None)
        state.sessions.reset = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers.schedule_eager_spawn",
            MagicMock(return_value=None),
        )

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")

        assert resp.status == 200
        state.sessions.reset.assert_awaited_once_with("slack:123.456", skip_if_busy=True)

    @pytest.mark.asyncio
    async def test_reload_refused_while_subagents_attached(self, tmp_path, monkeypatch):
        """409 while children are running/queued/delivering — the reset would
        tear down the shared subagent runtime and discard their work.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        state.sessions.get_provider = MagicMock(return_value=self._idle_provider())
        state.sessions.reset = AsyncMock()
        state.subagents = MagicMock()
        state.subagents.running_agents_for = MagicMock(return_value=["child-1"])
        state.subagents._queued_depth = MagicMock(return_value=0)
        eager = MagicMock(return_value=None)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.schedule_eager_spawn", eager)
        before = len(slot.messages)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")
            data = await resp.json()

        assert resp.status == 409
        assert data["code"] == "slot_subagents_running"
        state.sessions.reset.assert_not_awaited()
        assert len(slot.messages) == before
        eager.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_fails_closed_on_unreadable_children_probe(self, tmp_path, monkeypatch):
        """A None running-probe is the probe FAILING, not zero children."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")
        state.sessions.get_provider = MagicMock(return_value=None)
        state.sessions.reset = AsyncMock()
        state.subagents = MagicMock()
        state.subagents.running_agents_for = MagicMock(return_value=None)

        async with TestClient(TestServer(_make_app_with_agent_routes(state))) as client:
            resp = await client.post("/api/chat/slots/s1/reload")

        assert resp.status == 409
        state.sessions.reset.assert_not_awaited()


class TestCloseBroadcastDurability:
    """A slot-list frame must never omit a slot that is coming back.

    Clients treat a frame omitting a slot as licence to drop that slot's cached
    per-slot state, including follow-up and folder-suggestion cards that only
    ever exist in the browser and are never re-sent. So the close may only
    broadcast once the history save is durable: past that point no rollback can
    put the slot back, and the omission can never be retracted.

    Every test records the ORDER of the close steps, because asserting only that
    a broadcast happened passes against any placement and proves nothing.
    """

    @staticmethod
    def _instrument(state, slot, monkeypatch, frames=None):
        """Record close steps in call order; append each frame's slot keys."""
        from kiro_crew.dashboard import chat_handlers

        calls: list[str] = []

        def _push():
            calls.append("broadcast")
            if frames is not None:
                frames.append(set(state._slots))

        state.push_slots_update = MagicMock(side_effect=_push)
        # A successful close announces the removal through push_slot_removed,
        # which is the same slot-list broadcast for this test's purpose.
        state.push_slot_removed = MagicMock(side_effect=lambda _key: _push())

        real_sync = chat_handlers._sync_dashboard_slots

        def _sync(st):
            calls.append("sync")
            return real_sync(st)

        monkeypatch.setattr(chat_handlers, "_sync_dashboard_slots", _sync)

        real_unblock = chat_handlers._unblock_pending_waits

        def _unblock(st, sl):
            calls.append("unblock_pending_waits")
            return real_unblock(st, sl)

        monkeypatch.setattr(chat_handlers, "_unblock_pending_waits", _unblock)

        async def _save(*args, **kwargs):
            calls.append("save")

        monkeypatch.setattr(chat_handlers, "save_slot_off_loop", _save)

        state.sessions.remove = AsyncMock(side_effect=lambda *a, **k: calls.append("remove"))

        eager = MagicMock()
        eager.done = MagicMock(return_value=False)
        eager.cancel = MagicMock(side_effect=lambda: calls.append("cancel_eager"))
        slot._eager_spawn_task = eager

        async def _never() -> None:
            await asyncio.sleep(30)

        task = asyncio.get_running_loop().create_task(_never())
        real_cancel = task.cancel

        def _task_cancel(*args, **kwargs):
            calls.append("cancel_task")
            return real_cancel(*args, **kwargs)

        task.cancel = _task_cancel  # type: ignore[method-assign]
        # `running` is derived from `task`, so assigning the task is enough to
        # reach the cancel-and-wait_for branch.
        slot.task = task

        return calls

    @pytest.mark.asyncio
    async def test_failed_close_never_emits_a_frame_without_the_slot(self, tmp_path, monkeypatch):
        """The regression guard: a rollback must not be preceded by an omission.

        A frame omitting the slot is what makes another client drop this slot's
        follow-up and folder-suggestion cards. Those are browser-only and the
        backend re-offers a folder suggestion at most once per slot, so a prune
        on a close that then rolls back loses them permanently.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard import chat_handlers

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hello")
        slot.drain()
        frames: list[set] = []
        calls = self._instrument(state, slot, monkeypatch, frames=frames)

        async def _boom(*args, **kwargs):
            calls.append("save")
            raise RuntimeError("disk full")

        monkeypatch.setattr(chat_handlers, "save_slot_off_loop", _boom)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.delete("/api/chat/slots/s1")
            assert resp.status == 500

        assert frames, "no frame was broadcast at all, so this proves nothing"
        missing = [i for i, f in enumerate(frames) if "s1" not in f]
        assert not missing, (
            f"frame(s) {missing} omitted a slot that the rollback restores, so a "
            f"client pruned its cards for it: frames={frames} calls={calls}"
        )
        assert state._slots.get("s1") is slot, "slot was not restored"

    @pytest.mark.asyncio
    async def test_broadcast_precedes_the_session_teardown(self, tmp_path, monkeypatch):
        """On success, broadcast after the save but before the slow teardown."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hello")
        slot.drain()
        calls = self._instrument(state, slot, monkeypatch)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.delete("/api/chat/slots/s1")
            assert resp.status == 200

        for step in ("save", "remove", "broadcast"):
            assert step in calls, f"{step} never ran, so this proves nothing: {calls}"
        assert calls.index("save") < calls.index(
            "broadcast"
        ), f"broadcast ran before the save was durable: {calls}"
        assert calls.index("broadcast") < calls.index(
            "remove"
        ), f"broadcast waited out the session teardown: {calls}"

    @pytest.mark.asyncio
    async def test_broadcast_beats_a_slow_session_teardown(self, tmp_path, monkeypatch):
        """A session teardown that never returns must not hold the broadcast."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hello")
        slot.drain()
        calls = self._instrument(state, slot, monkeypatch)
        seen = asyncio.Event()
        state.push_slots_update = MagicMock(
            side_effect=lambda: (calls.append("broadcast"), seen.set())
        )
        state.push_slot_removed = MagicMock(
            side_effect=lambda _key: (calls.append("broadcast"), seen.set())
        )

        async def _hang(*args, **kwargs):
            calls.append("remove")
            await asyncio.sleep(30)

        state.sessions.remove = AsyncMock(side_effect=_hang)

        async with TestClient(TestServer(_make_app(state))) as client:
            close = asyncio.get_running_loop().create_task(client.delete("/api/chat/slots/s1"))
            await await_until(seen, what="the broadcast firing")
            assert "save" in calls and calls.index("save") < calls.index("broadcast")
            assert "s1" not in state._slots
            close.cancel()
            try:
                await close
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_active_slot_set_is_published_only_after_the_teardown(
        self, tmp_path, monkeypatch
    ):
        """_sync_dashboard_slots must stay after the teardown.

        Publishing the shrunken set earlier lets the idle sweep classify this
        session as orphaned and reap it mid-teardown, and that reap releases a
        shared subagent runtime which outlives the turn — the turn semaphore, the
        sweep's only busy guard, is free while subagents run.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "hello")
        slot.drain()
        calls = self._instrument(state, slot, monkeypatch)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.delete("/api/chat/slots/s1")
            assert resp.status == 200

        assert "sync" in calls, f"the trailing sync never ran: {calls}"
        assert calls.index("save") < calls.index(
            "sync"
        ), f"active-slot set published before the save was durable: {calls}"
        assert calls.index("remove") < calls.index(
            "sync"
        ), f"active-slot set published before the session teardown: {calls}"


class TestUnflushedTailOrderingAndSnapshot:
    """The owed tail must land in window order, from a snapshot taken on the loop."""

    @staticmethod
    def _row(role: str, body: str, mid: str, ts: str) -> dict:
        return {
            "role": role,
            "content": body,
            "cls": f"msg msg-{role[0]}",
            "ts": ts,
            "meta": {"mid": mid},
        }

    def test_owed_row_merges_at_its_window_position_not_after_the_disk_slice(self):
        """A persisted row LATER in window order must not be rendered before an owed one.

        ``_flush_segment`` pulls a ``stop_event`` out of the trailing chunk run and
        re-appends it AFTER the finalized assistant row (chat_runner.py:2686-2687).
        So a stop that reached disk during streaming sits, in window order, behind a
        reply that is still owed. Concatenating the owed rows after the whole disk
        slice therefore renders the pair inverted -- the stop card above the prose it
        terminated.

        Window order is authoritative and every row in this arm carries an id, so the
        position is derivable without the body matching the other arm needs.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._row("user", "run it", "u1", "2026-08-19T10:00:00+00:00")
        reply = self._row("assistant", "done", "a1", "2026-08-19T10:00:03+00:00")
        stop = self._row("assistant", "stopped", "s1", "2026-08-19T10:00:02+00:00")

        window = [q, reply, stop]
        all_msgs = [q, stop]  # the reply has not been flushed yet
        slot = SimpleNamespace(messages=window, _disk_older_count=0)

        disk_mids = {m["meta"]["mid"] for m in all_msgs}
        assert reply["meta"]["mid"] not in disk_mids, "fixture needs the reply un-persisted"
        assert stop["meta"]["mid"] in disk_mids, "fixture needs the stop already on disk"
        assert window.index(reply) < window.index(stop), (
            "fixture must place the owed reply BEFORE the persisted stop, else there "
            "is no inversion to detect"
        )

        out = _append_unflushed_tail(slot, all_msgs)  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]
        assert (
            bodies.count("stopped") == 1 and bodies.count("done") == 1
        ), f"a row was dropped or duplicated; got {bodies}"
        assert bodies.index("done") < bodies.index("stopped"), (
            f"the persisted stop was rendered before the owed reply it follows in the "
            f"window; got {bodies}"
        )

    def test_tail_with_no_inversion_still_appends_after_the_disk_slice(self):
        """Negative control: with no inversion the result must be the old concatenation.

        Fails for the intended reason if the merge reorders rows it should leave alone
        -- every owed row here already follows every persisted one in window order, so
        the answer must be exactly ``all_msgs + owed``.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._row("user", "run it", "u1", "2026-08-19T10:00:00+00:00")
        stop = self._row("assistant", "stopped", "s1", "2026-08-19T10:00:02+00:00")
        reply = self._row("assistant", "done", "a1", "2026-08-19T10:00:03+00:00")

        all_msgs = [q, stop]
        slot = SimpleNamespace(messages=[q, stop, reply], _disk_older_count=0)
        out = _append_unflushed_tail(slot, all_msgs)  # type: ignore[arg-type]
        assert out == [q, stop, reply], (
            f"the no-inversion case must be untouched concatenation; got "
            f"{[m['content'] for m in out]}"
        )

    @pytest.mark.asyncio
    async def test_tail_snapshot_is_captured_on_the_loop_not_inside_the_worker(
        self, tmp_path, monkeypatch
    ):
        """The window snapshot must be taken before the offload, not inside the thread.

        ``_flush_segment`` assigns ``slot.messages = head`` and only THEN appends the
        finalized assistant row, so between those two statements the window is a
        transient chunk-free head missing that row. It is a plain ``def`` that is never
        handed to ``to_thread``, so the event loop cannot interleave with it -- but the
        threaded tail scan can land exactly there and snapshot the hole, and
        ``_disk_older_count`` is untouched by that mutation so the bounded re-read
        cannot detect it.

        Simulated deterministically: the mutation is applied at the moment the worker
        is entered. A snapshot captured on the loop predates it and keeps the row; one
        taken inside the worker sees the hole and drops the row from the response.
        """
        import kiro_crew.dashboard.chat_handlers as ch
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("loopsnap")
        key = "dashboard:loopsnap"

        slot.append("user", "q1")
        slot.drain()
        _save_slot_to_history(state, slot, force=True)
        disk = state.conversation_log.read_messages_chained(key)
        disk_mids = {
            m["meta"]["mid"]
            for m in disk
            if isinstance(m.get("meta"), dict) and isinstance(m["meta"].get("mid"), str)
        }
        assert disk and len(disk_mids) == len(disk), (
            f"fixture needs every disk row to carry an id so the id arm runs; "
            f"disk={len(disk)} mids={len(disk_mids)}"
        )

        owed = {
            "role": "assistant",
            "content": "finalized-reply",
            "cls": "msg msg-a",
            "ts": "2026-08-19T10:00:03+00:00",
            "meta": {"mid": "m-finalizedneverpersisted"},
        }
        slot.messages.append(owed)
        assert owed["meta"]["mid"] not in disk_mids, "fixture row must be un-persisted"

        original = ch._append_unflushed_tail_from_offset
        seen: dict[str, object] = {}

        def worker_entered(slot_arg, all_msgs_arg, **kwargs):
            # The worker is now running. Reproduce the mid-finalization window: the
            # finalized assistant row is not in slot.messages yet.
            seen["snapshot_kwarg"] = "snapshot" in kwargs
            slot_arg.messages = [
                m for m in slot_arg.messages if m.get("content") != "finalized-reply"
            ]
            return original(slot_arg, all_msgs_arg, **kwargs)

        monkeypatch.setattr(ch, "_append_unflushed_tail_from_offset", worker_entered)

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/loopsnap?limit=10")
            assert resp.status == 200, await resp.text()
            data = await resp.json()

        assert (
            seen.get("snapshot_kwarg") is not None
        ), "the tail match never ran, so this test asserts nothing"
        assert seen["snapshot_kwarg"] is True, (
            "the caller did not pass a snapshot, so the window is still read inside "
            "the worker thread where a mid-finalization window is observable"
        )
        contents = [m["content"] for m in data["messages"]]
        assert "finalized-reply" in contents, (
            f"the worker snapshotted the transient chunk-free window and dropped the "
            f"finalized row; got {contents}"
        )

    @staticmethod
    def _idless(role: str, body: str, ts: str, cls: str | None = None) -> dict:
        """A row with NO ``meta.mid`` -- forces the id-less body-matching arm."""
        r = {"role": role, "content": body, "ts": ts}
        if cls is not None:
            r["cls"] = cls
        return r

    def test_live_streamed_row_before_a_persisted_row_is_not_dropped(self):
        """An owed transient row must survive the suffix boundary in the id-less arm.

        The matcher skipped a transient row WITHOUT advancing ``start``; a later
        persisted match then set ``start = i + 1``, moving the boundary PAST the skipped
        row, and ``window[start:]`` excluded it. So a live streamed chunk sitting before
        a persisted ``stop_event`` was dropped from the bounded response entirely.

        A disk read omits transient roles, so such a row can never be matched and is
        always owed -- which is precisely the rule the id-based arm already applies.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._idless("user", "run it", "2026-08-19T15:00:00+00:00")
        chunk = self._idless("chunk", "partial an", "2026-08-19T15:00:01+00:00")
        stop = self._idless("assistant", "stopped", "2026-08-19T15:00:02+00:00")

        all_msgs = [q, stop]  # a disk read holds no transient rows
        window = [q, chunk, stop]
        assert not any(
            "meta" in m for m in all_msgs
        ), "fixture must be id-less, else the id arm runs and this asserts nothing"
        assert window.index(chunk) < window.index(stop), (
            "fixture must place the transient BEFORE the persisted row, else the "
            "suffix boundary never moves past it"
        )

        slot = SimpleNamespace(messages=window, _disk_older_count=0)
        out = _append_unflushed_tail(slot, all_msgs)  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]
        assert (
            "partial an" in bodies
        ), f"the live streamed row was dropped by the suffix boundary; got {bodies}"
        assert bodies.index("partial an") < bodies.index("stopped"), (
            f"the owed transient must keep its window position ahead of the persisted "
            f"row; got {bodies}"
        )

    def test_trailing_transient_row_is_emitted_exactly_once(self):
        """Negative control for the merge: an owed transient must not be emitted twice.

        Rows left in ``pending`` sit at or after ``start`` and are therefore already
        carried by ``window[start:]``. Emitting them again as well would re-append a
        live row -- the duplication class this change exists to avoid. Fails for the
        intended reason if the merge and the suffix both claim the same row.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._idless("user", "run it", "2026-08-19T15:00:00+00:00")
        stop = self._idless("assistant", "stopped", "2026-08-19T15:00:02+00:00")
        chunk = self._idless("chunk", "partial an", "2026-08-19T15:00:03+00:00")

        slot = SimpleNamespace(messages=[q, stop, chunk], _disk_older_count=0)
        out = _append_unflushed_tail(slot, [q, stop])  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]
        assert (
            bodies.count("partial an") == 1
        ), f"the trailing transient was emitted more than once; got {bodies}"
        assert bodies == ["run it", "stopped", "partial an"], (
            f"a transient already after the last match must stay where it was; got " f"{bodies}"
        )

    def test_unowed_and_answered_rows_stay_excluded_from_the_idless_tail(self):
        """Negative control: only genuinely owed transients may be merged.

        ``done``/``queued`` are ``_UNOWED_WINDOW_ROLES`` and an answered ``permission``
        has already been dispositioned, so neither is owed. Merging transients
        unconditionally would re-append them, which is the same defect in the opposite
        direction from the drop above. A still-pending permission IS owed, so it is
        included -- that arm of the assertion is what stops the fix from over-narrowing
        into a second drop.
        """
        import json
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._idless("user", "run it", "2026-08-19T15:00:00+00:00")
        stop = self._idless("assistant", "stopped", "2026-08-19T15:00:02+00:00")
        done = self._idless("done", "unowed-marker", "2026-08-19T15:00:01+00:00")
        answered = self._idless(
            "permission",
            "answered-approval",
            "2026-08-19T15:00:01+00:00",
            cls=json.dumps({"resolved": "approve"}),
        )
        pending = self._idless(
            "permission",
            "pending-approval",
            "2026-08-19T15:00:01+00:00",
            cls=json.dumps({}),
        )

        for extra, body, owed in (
            (done, "unowed-marker", False),
            (answered, "answered-approval", False),
            (pending, "pending-approval", True),
        ):
            slot = SimpleNamespace(messages=[q, extra, stop], _disk_older_count=0)
            out = _append_unflushed_tail(slot, [q, stop])  # type: ignore[arg-type]
            bodies = [m["content"] for m in out]
            if owed:
                assert body in bodies, (
                    f"{extra['role']} row {body!r} is owed and must be merged; got " f"{bodies}"
                )
            else:
                assert body not in bodies, (
                    f"{extra['role']} row {body!r} is NOT owed and must not be "
                    f"re-appended; got {bodies}"
                )

    def test_unmatched_row_does_not_strand_the_persisted_rows_behind_it(self):
        """An owed row must not make the id-less arm re-emit later persisted rows.

        The arm forward-scans the disk slice for each window row. When a row was NOT
        found the loop would ``break`` outright, which left ``start`` pointing AT that
        row -- so ``window[start:]`` carried it AND every later window row, including
        rows the disk slice already holds. Those came back a second time, and out of
        order, which is the duplication this function exists to remove.

        The reachable shape is a ``stop_event`` flushed before reply finalization:
        ``_flush_segment`` re-appends the stop AFTER the finalized assistant row, so the
        window reads ``[... unflushed reply, flushed stop]``. The reply misses, and on
        the old code the persisted stop was emitted twice.

        The sibling id-carrying arm already holds an unmatched row and keeps walking, so
        this mirrors that rule rather than inventing a second one.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._idless("user", "run it", "2026-08-19T16:00:00+00:00")
        reply = self._idless("assistant", "the reply", "2026-08-19T16:00:01+00:00")
        stop = self._idless("assistant", "stopped", "2026-08-19T16:00:02+00:00")

        # The reply is still owed; the stop already reached disk ahead of it.
        all_msgs = [q, stop]
        window = [q, reply, stop]
        assert not any(
            "meta" in m for m in all_msgs
        ), "fixture must be id-less, else the id arm runs and this asserts nothing"
        assert window.index(reply) < window.index(stop), (
            "fixture must place the UNMATCHED row before the persisted one, else the "
            "loop never abandons the rows behind it"
        )
        assert not any(
            m["content"] == "the reply" for m in all_msgs
        ), "fixture must keep the reply off disk, else there is nothing unmatched"

        slot = SimpleNamespace(messages=window, _disk_older_count=0)
        out = _append_unflushed_tail(slot, all_msgs)  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]

        assert bodies.count("stopped") == 1, (
            f"the persisted stop was re-emitted because an earlier unmatched row "
            f"abandoned the scan; got {bodies}"
        )
        assert bodies == [
            "run it",
            "the reply",
            "stopped",
        ], f"output must follow window order with each row exactly once; got {bodies}"

    def test_unmatched_row_is_still_emitted_and_not_dropped(self):
        """Negative control: holding the unmatched row must not silently drop it.

        The fix stops an unmatched row from stranding the rows behind it. The opposite
        failure is just as silent -- discarding that row instead of retaining it would
        also stop the duplicate, and would pass a test that only counted the persisted
        row. So assert the owed row itself survives, in both the with-a-later-match and
        no-later-match shapes.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._idless("user", "ask", "2026-08-19T16:10:00+00:00")
        owed = self._idless("assistant", "owed body", "2026-08-19T16:10:01+00:00")
        persisted = self._idless("assistant", "on disk", "2026-08-19T16:10:02+00:00")

        # (a) a later row DOES match, so the owed row is flushed at that disk position
        slot = SimpleNamespace(messages=[q, owed, persisted], _disk_older_count=0)
        out = _append_unflushed_tail(slot, [q, persisted])  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]
        assert (
            bodies.count("owed body") == 1
        ), f"the owed row must be retained exactly once, not dropped; got {bodies}"
        assert bodies.index("owed body") < bodies.index(
            "on disk"
        ), f"the retained row must keep its window position; got {bodies}"

        # (b) NO later row matches, so the owed row can only arrive via the tail
        slot = SimpleNamespace(messages=[q, owed], _disk_older_count=0)
        out = _append_unflushed_tail(slot, [q])  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]
        assert (
            bodies.count("owed body") == 1
        ), f"with no later match the owed row must still be emitted once; got {bodies}"

    def test_trailing_unowed_and_answered_rows_are_excluded_from_the_idless_tail(self):
        """An unowed row in the LAST window position must not reach the tail.

        ``start`` only advances on a match, and an unowed row takes the transient branch
        and ``continue``s -- skipping ``start = i + 1``. So when such a row TRAILS the last
        match, a raw ``window[start:]`` re-admits it and undoes the exclusion the loop's own
        guard performed. The sibling id-carrying arm has no such hole because it applies
        both exclusions at the top of its loop.

        Two symptoms share that cause. A trailing ``done`` reaches the bounded response and
        ``_prepare_messages`` drops it while rendering, so a page whose only row is that
        ``done`` renders EMPTY and replaces the transcript. A trailing answered
        ``permission`` is instead re-ordered after every persisted row.

        The sibling test above puts the extra row in the MIDDLE, where the later match
        advances ``start`` PAST it -- so the exclusion it observes comes from ``start``
        overshooting, not from any filter, and it cannot see this defect. The whole body
        list is asserted rather than mere membership, because membership alone cannot see a
        mis-ordering.
        """
        import json
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._idless("user", "run it", "2026-08-20T04:00:00+00:00")
        stop = self._idless("assistant", "stopped", "2026-08-20T04:00:01+00:00")
        done = self._idless("done", "unowed-marker", "2026-08-20T04:00:02+00:00")
        queued = self._idless("queued", "queued-marker", "2026-08-20T04:00:02+00:00")
        answered = self._idless(
            "permission",
            "answered-approval",
            "2026-08-20T04:00:02+00:00",
            cls=json.dumps({"resolved": "approve"}),
        )

        for extra, body in (
            (done, "unowed-marker"),
            (queued, "queued-marker"),
            (answered, "answered-approval"),
        ):
            window = [q, stop, extra]
            assert window[-1] is extra, (
                "fixture must place the unowed row LAST, else a later match advances "
                "start past it and the assertion passes without a filter"
            )
            slot = SimpleNamespace(messages=window, _disk_older_count=0)
            out = _append_unflushed_tail(slot, [q, stop])  # type: ignore[arg-type]
            bodies = [m["content"] for m in out]
            assert body not in bodies, (
                f"trailing {extra['role']} row {body!r} was re-admitted by the unfiltered "
                f"tail slice; got {bodies}"
            )
            assert bodies == ["run it", "stopped"], (
                f"the bounded page must hold exactly the persisted rows in order; got " f"{bodies}"
            )

    def test_trailing_owed_rows_survive_the_idless_tail_filter(self):
        """Negative control: the tail filter must not over-narrow into a second drop.

        ``_UNOWED_WINDOW_ROLES`` is ``{done, queued}`` -- ``chunk`` and ``streaming`` are
        subtracted out, and a still-PENDING ``permission`` has not been dispositioned. All
        three are genuinely owed: a disk read carries no transient row, so they can never be
        matched, and dropping them loses a live streamed answer or an approval the user has
        yet to answer. That is the opposite defect from the leak, and it is the one an
        over-eager filter actually produces.
        """
        import json
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        q = self._idless("user", "run it", "2026-08-20T04:10:00+00:00")
        stop = self._idless("assistant", "stopped", "2026-08-20T04:10:01+00:00")
        pending = self._idless(
            "permission",
            "pending-approval",
            "2026-08-20T04:10:02+00:00",
            cls=json.dumps({}),
        )
        chunk = self._idless("chunk", "partial an", "2026-08-20T04:10:02+00:00")

        slot = SimpleNamespace(messages=[q, stop, pending], _disk_older_count=0)
        out = _append_unflushed_tail(slot, [q, stop])  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]
        assert bodies == ["run it", "stopped", "pending-approval"], (
            f"a still-pending permission is owed and must survive the tail filter; got " f"{bodies}"
        )

        slot = SimpleNamespace(messages=[q, stop, chunk], _disk_older_count=0)
        out = _append_unflushed_tail(slot, [q, stop])  # type: ignore[arg-type]
        bodies = [m["content"] for m in out]
        assert (
            bodies.count("partial an") == 1
        ), f"a trailing live chunk is owed and must be emitted exactly once; got {bodies}"
        assert bodies == [
            "run it",
            "stopped",
            "partial an",
        ], f"the owed chunk must keep its trailing window position; got {bodies}"


class TestHistoryResumeInterruptedTurnMarker:
    """The History resume endpoint shares the startup restore's marker handling."""

    @staticmethod
    def _saved_state(tmp_path, monkeypatch, name: str, rows: list[tuple], generation: int):
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot(name)
        for row in rows:
            slot.append(*row[:3], meta=row[3] if len(row) > 3 else None)
        slot._turn_in_flight_generation = generation
        assert _save_slot_to_history(state, slot, force=True)
        del state._slots[slot.key]
        return state

    @pytest.mark.asyncio
    async def test_resume_reconciles_partial_assistant_and_tool_tail(self, tmp_path, monkeypatch):
        state = self._saved_state(
            tmp_path,
            monkeypatch,
            "resume-interrupted",
            [
                ("user", "inspect it", "msg msg-u"),
                ("assistant", "partial result", "msg msg-a"),
                ("tool", "read complete", "tool", {"done": True}),
            ],
            generation=18,
        )

        async with TestClient(TestServer(_make_app(state))) as client:
            response = await client.post(
                "/api/chat/slots/resume-interrupted/resume",
                json={"key": "dashboard:resume-interrupted"},
            )
            payload = await response.json()

        assert response.status == 200
        restored = state.get_slot("resume-interrupted")
        assert restored is not None
        assert restored.to_dict()["interrupted"] is True
        assert restored.messages[-1]["meta"]["kind"] == "gateway_restart_interruption"
        # The paging cursor counts the recovered row, so the next older page
        # starts where this response's rows end.
        assert payload["total"] == 4
        assert payload["messages"][-1]["role"] == "error"
        assert payload["next_before"] == 0
        assert payload["has_more"] is False

    @pytest.mark.asyncio
    async def test_resume_preserves_a_deliberate_stop(self, tmp_path, monkeypatch):
        state = self._saved_state(
            tmp_path,
            monkeypatch,
            "resume-stopped",
            [
                ("user", "inspect it", "msg msg-u"),
                ("system", "Stopped", json.dumps({"kind": "stop_event"})),
            ],
            generation=19,
        )

        async with TestClient(TestServer(_make_app(state))) as client:
            response = await client.post(
                "/api/chat/slots/resume-stopped/resume",
                json={"key": "dashboard:resume-stopped"},
            )
            payload = await response.json()

        assert response.status == 200
        restored = state.get_slot("resume-stopped")
        assert restored is not None
        assert restored.to_dict()["interrupted"] is False
        assert all(
            (m.get("meta") or {}).get("kind") != "gateway_restart_interruption"
            for m in restored.messages
        )
        assert payload["total"] == 2

    @pytest.mark.asyncio
    async def test_resume_ignores_a_malformed_boolean_marker(self, tmp_path, monkeypatch):
        state = self._saved_state(
            tmp_path,
            monkeypatch,
            "resume-malformed",
            [("user", "hi", "msg msg-u"), ("assistant", "complete", "msg msg-a")],
            generation=0,
        )
        assert state.conversation_log is not None
        state.conversation_log.update_metadata(
            "dashboard:resume-malformed", {"turn_in_flight_generation": True}
        )

        async with TestClient(TestServer(_make_app(state))) as client:
            response = await client.post(
                "/api/chat/slots/resume-malformed/resume",
                json={"key": "dashboard:resume-malformed"},
            )

        assert response.status == 200
        restored = state.get_slot("resume-malformed")
        assert restored is not None
        assert restored.to_dict()["interrupted"] is False
        assert all(
            (m.get("meta") or {}).get("kind") != "gateway_restart_interruption"
            for m in restored.messages
        )


def _long_turn(n_chunks: int, finished: asyncio.Event):
    """A turn that streams *n_chunks* rows of about 1 KB through the real ``append``."""

    async def fake_run_chat(st, sl, msg, *, _directive_user_origin):
        for i in range(n_chunks):
            sl.append("chunk", f"{i:06d}" + "x" * 1024, "chunk")
            if i % 16 == 0:
                await asyncio.sleep(0)
        sl.append("assistant", "final answer", "msg")
        sl.purge_chunks()  # the runner's turn end
        sl.append("done", "", "done")
        finished.set()

    return fake_run_chat


async def _post_and_never_read(client, slot_key: str):
    """A client that sends a whole ``POST /api/chat`` and then never reads its socket."""
    import socket

    host, port = client.server.host, client.server.port
    loop = asyncio.get_running_loop()
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    raw.setblocking(False)
    await loop.sock_connect(raw, (host, port))
    body = json.dumps({"message": "hello", "slot": slot_key}).encode()
    head = (
        f"POST /api/chat HTTP/1.1\r\nHost: {host}:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode()
    await loop.sock_sendall(raw, head + body)
    return raw


@pytest.mark.asyncio
class TestApiChatStalledReader:
    """An SSE reader that stops reading is cut off, so it stops holding its slot."""

    async def test_a_stalled_reader_is_cut_off_and_releases_the_slot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.config.loader.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._SSE_WRITE_DEADLINE_SECS", 0.5, raising=False
        )
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        broadcast: list[str] = []
        slot._on_message = lambda _key, msg: broadcast.append(str(msg.get("content", "")))
        finished = asyncio.Event()
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._run_chat", _long_turn(8000, finished)
        )

        async with TestClient(TestServer(_make_app(state))) as client:
            raw = await _post_and_never_read(client, "s1")
            try:
                await asyncio.wait_for(finished.wait(), 60)
                await async_wait_until(  # the write deadline cuts the stalled reader off
                    lambda: slot._pending_consumers == 0,
                    describe=lambda: f"pending_consumers={slot._pending_consumers}",
                )
                consumers, reader = slot._pending_consumers, slot._has_reader
                held = [m for m in slot._pending if m.get("role") == "chunk"]
                # A row from outside the turn, as a cron or sub-agent notice adds one.
                slot.append("assistant", "a later notice", "msg")
                # The next turn on the slot, sent from a dashboard tab.
                finished.clear()
                sent = await client.post("/api/chat?ws=1", json={"message": "again", "slot": "s1"})
                assert (await sent.json())["ok"] is True
                await asyncio.wait_for(finished.wait(), 60)
                held_next = [m for m in slot._pending if m.get("role") == "chunk"]
            finally:
                raw.close()

        noticed = "a later notice" in broadcast
        assert (consumers, reader, len(held), noticed, len(held_next)) == (0, False, 0, True, 0), (
            f"a stalled SSE reader still holds its slot after the turn: "
            f"pending_consumers={consumers}, _has_reader={reader}, {len(held)} chunk rows "
            f"({sum(len(m['content']) for m in held)} bytes) held, a later row broadcast to other "
            f"tabs={noticed}; after the next turn {len(held_next)} chunk rows "
            f"({sum(len(m['content']) for m in held_next)} bytes) stay held"
        )

    async def test_a_reading_client_gets_the_whole_turn(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.config.loader.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        finished = asyncio.Event()
        monkeypatch.setattr(
            "kiro_crew.dashboard.chat_handlers._run_chat", _long_turn(8000, finished)
        )

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat", json={"message": "hello", "slot": "s1"}, timeout=None
            )
            body = await resp.read()
            await asyncio.wait_for(finished.wait(), 60)

        assert body.endswith(b"data: [DONE]\n\n")
        assert all(f"{i:06d}x".encode() in body for i in (0, 3999, 7999))
        assert slot._pending_consumers == 0 and slot._has_reader is False
        assert not [m for m in slot._pending if m.get("role") == "chunk"]
