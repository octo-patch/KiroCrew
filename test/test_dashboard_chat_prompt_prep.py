"""Dashboard chat: slash commands the dashboard refuses before dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from chat_test_helpers import _make_state

# ── Slash command tests ──


class TestBlockedSlashCommands:
    """Tests for _BLOCKED_SLASH_COMMANDS blocking dangerous commands."""

    def test_quit_is_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/quit" in _BLOCKED_SLASH_COMMANDS

    def test_exit_is_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/exit" in _BLOCKED_SLASH_COMMANDS

    def test_q_is_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/q" in _BLOCKED_SLASH_COMMANDS

    def test_editor_is_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/editor" in _BLOCKED_SLASH_COMMANDS

    def test_chat_is_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/chat" in _BLOCKED_SLASH_COMMANDS

    def test_paste_is_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/paste" in _BLOCKED_SLASH_COMMANDS

    def test_reply_is_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/reply" in _BLOCKED_SLASH_COMMANDS

    def test_tangent_is_blocked(self):
        # kiro-cli's terminal checkpoint toggle (Ctrl+T); the dashboard's
        # session model and /side cover its purpose, so it is inert here.
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/tangent" in _BLOCKED_SLASH_COMMANDS

    def test_compact_is_not_blocked(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS

        assert "/compact" not in _BLOCKED_SLASH_COMMANDS

    def test_blocked_is_subset_of_slash(self):
        from kiro_crew.dashboard.chat import _BLOCKED_SLASH_COMMANDS, _SLASH_COMMANDS

        assert _BLOCKED_SLASH_COMMANDS.issubset(_SLASH_COMMANDS)

    @pytest.mark.asyncio
    async def test_blocked_command_returns_warning_no_session(self, tmp_path, monkeypatch):
        """Posting /quit should add warning to slot and never acquire a session."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("s1")

        from kiro_crew.dashboard.chat import _run_chat

        await _run_chat(state, slot, "/quit")

        # Should have the warning message
        texts = [m["content"] for m in slot.messages if m.get("role") == "assistant"]
        assert any("not available in the dashboard" in t for t in texts)
        # Should never have called get_or_create (no session acquired)
        state.sessions.get_or_create.assert_not_called()
