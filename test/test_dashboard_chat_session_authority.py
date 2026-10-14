"""Dashboard chat: which copy of a session is authoritative — the in-memory window over disk."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_app, _make_state

# ── Default view uses in-memory (not stale disk) ──


class TestInMemoryAuthority:
    @pytest.mark.asyncio
    async def test_default_view_shows_current_messages(self, tmp_path, monkeypatch):
        """Default slot detail should return in-memory messages, not stale disk."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        log = state.conversation_log
        # Stale disk data
        log.append("dashboard:s1", "user", "old question")
        log.append("dashboard:s1", "assistant", "old answer")

        # Active slot with different messages
        slot = state.get_or_create_slot("s1")
        slot.append("user", "new question")
        slot.append("tool", "✅ running")
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/s1")
            data = await resp.json()
            # Should show in-memory (2 msgs), not disk (2 different msgs)
            assert data["total"] == 2
            assert data["messages"][0]["content"] == "new question"
            assert data["messages"][1]["content"] == "✅ running"

    @pytest.mark.asyncio
    async def test_full_load_prepends_older_disk_messages(self, tmp_path, monkeypatch):
        """No-limit path prepends older disk messages when restore truncated."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        log = state.conversation_log
        # Simulate: 8 messages on disk total (5 older + 3 recent)
        for i in range(8):
            log.append("dashboard:s2", "user", f"msg {i}")
        # Slot has only the last 3 in memory (simulating truncated restore)
        slot = state.get_or_create_slot("s2")
        slot.append("user", "msg 5")
        slot.append("user", "msg 6")
        slot.append("user", "msg 7")
        slot.drain()
        # Flag that restore truncated older messages
        slot._disk_older_count = 5

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/s2")
            data = await resp.json()
            assert data["total"] == 8  # 5 older + 3 recent
            assert data["has_more"] is False
            assert data["messages"][0]["content"] == "msg 0"
            assert data["messages"][4]["content"] == "msg 4"
            assert data["messages"][5]["content"] == "msg 5"

    @pytest.mark.asyncio
    async def test_legacy_pagination_with_limit(self, tmp_path, monkeypatch):
        """Legacy limit-based pagination reads from chained disk."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        log = state.conversation_log
        for i in range(10):
            log.append("dashboard:s3", "user", f"msg {i}")
        slot = state.get_or_create_slot("s3")  # noqa: F841

        async with TestClient(TestServer(_make_app(state))) as client:
            # limit=3 returns last 3, has_more=True
            resp = await client.get("/api/chat/slots/s3?limit=3")
            data = await resp.json()
            assert data["total"] == 10
            assert data["has_more"] is True
            assert len(data["messages"]) == 3
            assert data["messages"][-1]["content"] == "msg 9"

            # limit=3&before=5 returns msgs 2-4
            resp = await client.get("/api/chat/slots/s3?limit=3&before=5")
            data = await resp.json()
            assert data["has_more"] is True  # msgs 0, 1 still older
            assert [m["content"] for m in data["messages"]] == ["msg 2", "msg 3", "msg 4"]

            # before=2 returns last 2 older
            resp = await client.get("/api/chat/slots/s3?limit=100&before=2")
            data = await resp.json()
            assert data["has_more"] is False
            assert [m["content"] for m in data["messages"]] == ["msg 0", "msg 1"]

    def test_append_only_preserves_full_disk_history(self, tmp_path, monkeypatch):
        """Save keeps ALL on-disk messages; nothing dropped or archived.

        After a restart the slot holds only a recent window (the frozen prefix
        counted by _disk_older_count is OLDER on disk) while the JSONL still has
        the full history. Saving a new turn must preserve the frozen prefix
        byte-for-byte — no overwrite, no truncation, no archive.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        # 8 messages on disk total (the pre-restart full history).
        for i in range(8):
            log.append("dashboard:s4", "user", f"old msg {i}")
        # Restore truncated to the last 3 in memory; _disk_window_len mirrors
        # what restore sets (those 3 are already the on-disk window tail).
        # Mirror PRODUCTION restore, which re-appends each window message WITH its
        # persisted ``ts`` (chat_persistence restore uses
        # ``slot.append(..., ts=m.get("ts", ""))``). Preserving the ts is what lets
        # a steady save recognise the on-disk window-region copies as its OWN (a
        # ts-match) rather than as fresh-ts duplicates — so neither duplicating nor
        # archiving them.
        slot = state.get_or_create_slot("s4")
        disk_tail = log.read_messages("dashboard:s4")[5:]
        for m in disk_tail:
            slot.append("user", m["content"], ts=m.get("ts", ""))
        slot.drain()
        slot._resumed_count = len(slot.messages)
        slot._disk_window_len = len(slot.messages)
        slot._disk_older_count = 5
        slot.append("user", "brand new msg")
        slot.drain()

        _save_slot_to_history(state, slot)

        # Nothing archived — the frozen prefix is never dropped.
        assert not (tmp_path / "archive").exists()
        # All 8 original messages + the new one are on disk, in order.
        disk = log.read_messages("dashboard:s4")
        contents = [m["content"] for m in disk]
        assert contents == [f"old msg {i}" for i in range(8)] + ["brand new msg"]

    def test_append_only_colliding_ts_no_duplicate(self, tmp_path, monkeypatch):
        """Colliding timestamps must not duplicate a genuine message on disk.

        A coarse system clock (notably Windows' ~15ms tick) can stamp a burst of
        rapid appends with an IDENTICAL ``datetime.now().isoformat()``. The
        append-safe save must still match each on-disk window-region line to its
        own window entry one-for-one — never mis-classifying a real window line
        as a phantom "foreign append" and duplicating it.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        # Freeze history.py's clock so every on-disk append shares ONE ts,
        # reproducing the coarse-clock collision deterministically on any OS.
        import datetime as _dt

        from kiro_crew.dashboard.chat import _save_slot_to_history

        _FROZEN = _dt.datetime(2026, 7, 25, 0, 0, 0, tzinfo=_dt.timezone.utc)

        class _FrozenDateTime:
            @classmethod
            def now(cls, tz=None):
                return _FROZEN

        monkeypatch.setattr("kiro_crew.history.datetime", _FrozenDateTime)

        state = _make_state(tmp_path)
        log = state.conversation_log
        # 8 messages on disk, ALL sharing the frozen ts.
        for i in range(8):
            log.append("dashboard:s5c", "user", f"old msg {i}")
        disk = log.read_messages("dashboard:s5c")
        # Every on-disk ts is identical — the exact condition that broke matching.
        assert len({m.get("ts") for m in disk}) == 1

        # Restore truncated to the last 3, re-appending WITH the persisted ts
        # (mirrors production restore) — so those window entries also collide.
        slot = state.get_or_create_slot("s5c")
        for m in disk[5:]:
            slot.append("user", m["content"], ts=m.get("ts", ""))
        slot.drain()
        slot._resumed_count = len(slot.messages)
        slot._disk_window_len = len(slot.messages)
        slot._disk_older_count = 5
        slot.append("user", "brand new msg")
        slot.drain()

        _save_slot_to_history(state, slot)

        # No phantom foreign append, nothing archived, full history preserved.
        assert not (tmp_path / "archive").exists()
        contents = [m["content"] for m in log.read_messages("dashboard:s5c")]
        assert contents == [f"old msg {i}" for i in range(8)] + ["brand new msg"]

    def test_append_only_colliding_ts_edit_preserves_foreign(self, tmp_path, monkeypatch):
        """Colliding ts + in-place edit + foreign append must NOT drop the
        foreign line.

        The on-disk window region holds, sharing ONE coarse-clock ts: unchanged
        ``A``, an acknowledged cross-process foreign append ``X``, and the
        pre-edit copy of ``B`` — in that adversarial order (foreign BEFORE the
        pre-edit copy). Memory holds ``A`` and the EDITED ``B'``. A greedy
        ts-only match would consume ``X`` as if it were ``B``'s in-place edit and
        silently DROP the acknowledged foreign append. The ambiguity guard must
        instead preserve ``X`` (favouring a rare stale duplicate over
        irreversible data loss).
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        import json as _json

        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        t = "2026-07-25T00:00:00+00:00"  # single colliding timestamp

        def _line(content: str) -> str:
            return _json.dumps({"role": "user", "content": content, "ts": t}) + "\n"

        # Craft the adversarial on-disk order directly: A, foreign X, pre-edit B.
        path = log._path("dashboard:s5e")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps({"_type": "metadata", "created_at": t, "last_consolidated": 0})
            + "\n"
            + _line("msg A")
            + _line("msg X (foreign)")
            + _line("msg B"),
            encoding="utf-8",
        )

        # Window: A unchanged, B edited in place (content changes, ts preserved).
        slot = state.get_or_create_slot("s5e")
        slot.append("user", "msg A", ts=t)
        slot.append("user", "msg B (edited in place)", ts=t)
        slot.drain()
        slot._resumed_count = len(slot.messages)
        slot._disk_window_len = len(slot.messages)
        slot._dirty = True  # an in-place edit marks the slot dirty

        _save_slot_to_history(state, slot)

        contents = [m["content"] for m in log.read_messages("dashboard:s5e")]
        # The acknowledged foreign append survived (no data loss) ...
        assert "msg X (foreign)" in contents, "foreign append was dropped (data loss)"
        # ... and the edited window content is present.
        assert "msg B (edited in place)" in contents

    def test_append_only_no_duplicate_on_resave(self, tmp_path, monkeypatch):
        """Re-saving without new messages must not duplicate the tail on disk."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        for i in range(4):
            log.append("dashboard:s6", "user", f"msg {i}")
        slot = state.get_or_create_slot("s6")
        for i in range(4):
            slot.append("user", f"msg {i}")
        slot.drain()
        slot._resumed_count = len(slot.messages)
        slot._disk_window_len = len(slot.messages)

        slot.append("assistant", "reply A")
        slot.drain()
        _save_slot_to_history(state, slot)
        # A redundant save (force) must not duplicate the already-written tail.
        _save_slot_to_history(state, slot, force=True)

        disk = log.read_messages("dashboard:s6")
        contents = [m["content"] for m in disk]
        assert contents == [f"msg {i}" for i in range(4)] + ["reply A"]

    def test_save_steady_state_does_not_archive(self, tmp_path, monkeypatch):
        """A normal append (slot is a superset of disk) archives nothing."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s5")
        slot.append("user", "first")
        slot.drain()
        _save_slot_to_history(state, slot)
        slot.append("assistant", "reply")
        slot.drain()
        _save_slot_to_history(state, slot)

        assert not (tmp_path / "archive").exists()

    def test_append_only_does_not_merge_unrelated_session(self, tmp_path, monkeypatch):
        """Append-only writes ONLY to the slot's own file — never an unrelated one.

        Guards the regression the reporter flagged: a second, unrelated session
        must not get merged into this slot's history. Append-only touches a
        single session file, so a sibling file is left completely untouched.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        # Two independent on-disk sessions (no shared tab_id).
        for i in range(3):
            log.append("dashboard:sessA", "user", f"A{i}")
        for i in range(2):
            log.append("dashboard:sessB", "user", f"B{i}")
        b_before = log.read_messages("dashboard:sessB")

        slot = state.get_or_create_slot("sessA")
        for i in range(3):
            slot.append("user", f"A{i}")
        slot.drain()
        slot._resumed_count = len(slot.messages)
        slot._disk_window_len = len(slot.messages)
        slot._tab_id = ""  # no chaining
        slot.append("user", "A-new")
        slot.drain()
        _save_slot_to_history(state, slot)

        # sessA got its new turn; sessB is byte-for-byte untouched.
        assert [m["content"] for m in log.read_messages("dashboard:sessA")] == [
            "A0",
            "A1",
            "A2",
            "A-new",
        ]
        assert log.read_messages("dashboard:sessB") == b_before

    def test_rewrite_path_archives_dropped_tail(self, tmp_path, monkeypatch):
        """An explicit snapshot save (rewrite, e.g. rewind) archives dropped msgs."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        slot = state.get_or_create_slot("s7")
        for c in ("q1", "a1", "q2", "a2"):
            slot.append("user" if c.startswith("q") else "assistant", c)
        slot.drain()
        _save_slot_to_history(state, slot)  # append-only establishes the file
        assert not (tmp_path / "archive").exists()

        # Rewind-style: truncate to first turn and rewrite via explicit snapshot.
        snapshot = [dict(m) for m in slot.messages[:1]]
        _save_slot_to_history(state, slot, snapshot)

        # Disk now holds only the kept prefix; dropped tail is archived.
        disk = log.read_messages("dashboard:s7")
        assert [m["content"] for m in disk] == ["q1"]
        archives = list((tmp_path / "archive").glob("dashboard_s7__*.jsonl"))
        assert len(archives) == 1
        content = archives[0].read_text(encoding="utf-8")
        for dropped in ("a1", "q2", "a2"):
            assert dropped in content

    def test_rewrite_preserves_frozen_prefix(self, tmp_path, monkeypatch):
        """A rewrite (rewind) with a frozen prefix keeps the prefix, drops only the tail.

        With _disk_older_count > 0 the file is frozen_prefix + window. A rewrite
        that truncates the window must leave the frozen prefix byte-for-byte and
        archive only the dropped window tail — never the older history.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        # 6 messages on disk; only the last 2 are the in-memory window.
        for i in range(6):
            log.append("dashboard:s11", "user", f"old {i}")
        slot = state.get_or_create_slot("s11")
        slot.append("user", "old 4")
        slot.append("assistant", "old 5")
        slot.drain()
        slot._resumed_count = len(slot.messages)
        slot._disk_window_len = len(slot.messages)
        slot._disk_older_count = 4  # old 0..3 are the frozen prefix

        # Rewind-style truncate to just the first window message + rewrite.
        snapshot = [dict(m) for m in slot.messages[:1]]
        _save_slot_to_history(state, slot, snapshot)

        disk = log.read_messages("dashboard:s11")
        # Frozen prefix preserved + kept window head; dropped tail archived.
        assert [m["content"] for m in disk] == ["old 0", "old 1", "old 2", "old 3", "old 4"]
        archives = list((tmp_path / "archive").glob("dashboard_s11__*.jsonl"))
        assert len(archives) == 1
        content = archives[0].read_text(encoding="utf-8")
        assert "old 5" in content
        # The frozen prefix must NOT be archived.
        for frozen in ("old 0", "old 1", "old 2", "old 3"):
            assert frozen not in content

    def test_flush_then_finalized_reply_persists(self, tmp_path, monkeypatch):
        """#1: a mid-turn flush past a stop_event must not lose the later reply.

        Repro of the boundary-asymmetry bug: a stop happens mid-turn, the 5s
        flush persists the window ending in the stop_event, then _flush_segment
        finalizes the assistant reply and moves the stop_event AFTER it. The
        finalized reply must end up on disk (the old position-counter model
        committed past the stop_event and dropped the later assistant line).
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        import json

        from kiro_crew.dashboard.chat import _flush_segment, _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        slot = state.get_or_create_slot("s8")
        slot.append("user", "do a thing")
        # Streaming chunk + a stop_event interleaved with this segment.
        slot.append("chunk", "partial output")
        stop_cls = json.dumps({"kind": "stop_event", "id": "stop-1", "state": "stopping"})
        slot.append("system", stop_cls, cls=stop_cls)
        slot.drain()

        # 5s flush lands here — window ends in chunk + stop_event.
        _save_slot_to_history(state, slot)

        # _flush_segment finalizes the assistant reply and re-orders the
        # stop_event to land AFTER it.
        _flush_segment(state, slot, "final answer", broadcast=False)
        slot.drain()
        _save_slot_to_history(state, slot)

        disk = log.read_messages("dashboard:s8")
        contents = [m["content"] for m in disk]
        # The finalized reply is on disk, after the user prompt; chunk dropped.
        assert "final answer" in contents
        assert "do a thing" in contents
        assert "partial output" not in contents
        # Stop card persists after the finalized assistant reply.
        assert contents.index("final answer") < contents.index(stop_cls)

    def test_inplace_stop_resolution_persists(self, tmp_path, monkeypatch):
        """#2: an in-place edit of an already-flushed message must persist.

        The stop_event message is flushed as "stopping"; later it is mutated in
        place to "stopped" (mirrors _resolve_stop_event). The re-serialized
        window must carry the resolution to disk — the old append-only model
        only wrote new tail messages and dropped the in-place edit.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        import json

        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        slot = state.get_or_create_slot("s9")
        slot.append("user", "go")
        stopping = json.dumps({"kind": "stop_event", "id": "stop-x", "state": "stopping"})
        slot.append("system", stopping, cls=stopping)
        slot.drain()
        _save_slot_to_history(state, slot)  # flushed as "stopping"
        # Model a resumed slot: the window equals the resumed count and the
        # in-place edit adds NO new message — exercises the resumed-guard path.
        slot._resumed_count = len(slot.messages)
        slot._disk_window_len = len(slot.messages)
        slot._dirty = False

        # Resolve in place (same object already on disk at index < end).
        stopped = json.dumps({"kind": "stop_event", "id": "stop-x", "state": "stopped"})
        for m in slot.messages:
            if m.get("role") == "system":
                m["cls"] = stopped
                m["content"] = stopped
                slot._dirty = True  # _resolve_stop_event sets this
                break
        _save_slot_to_history(state, slot)

        disk = log.read_messages("dashboard:s9")
        sys_msgs = [m for m in disk if m.get("role") == "system"]
        assert len(sys_msgs) == 1
        assert json.loads(sys_msgs[0]["cls"])["state"] == "stopped"

    def test_failed_rewrite_does_not_strand_or_lose(self, tmp_path, monkeypatch):
        """#3: a failed inline rewrite must not lose the dropped tail or strand state.

        rewind/regenerate truncate the window then save with a snapshot. If that
        save raises, _pending_rewrite stays set so the next (flush) save still
        takes the archive-safe rewrite path — the dropped tail is archived, not
        silently overwritten, and the kept prefix is correct on disk.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        from kiro_crew.dashboard import chat_persistence
        from kiro_crew.dashboard.chat import _save_slot_to_history

        state = _make_state(tmp_path)
        log = state.conversation_log
        slot = state.get_or_create_slot("s10")
        for c in ("q1", "a1", "q2", "a2"):
            slot.append("user" if c.startswith("q") else "assistant", c)
        slot.drain()
        _save_slot_to_history(state, slot)

        # Simulate rewind: truncate window to first turn, mark pending rewrite,
        # and have the inline rewrite save blow up inside atomic_write.
        del slot.messages[1:]
        slot._pending_rewrite = True
        orig_atomic = chat_persistence.atomic_write
        calls = {"n": 0}

        def _boom(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk full")
            return orig_atomic(*a, **k)

        monkeypatch.setattr(chat_persistence, "atomic_write", _boom)
        snapshot = [dict(m) for m in slot.messages]
        with pytest.raises(OSError):
            _save_slot_to_history(state, slot, snapshot)
        # Flag still set → flush retries the archive-safe rewrite (2nd write OK).
        assert slot._pending_rewrite is True
        _save_slot_to_history(state, slot)
        assert slot._pending_rewrite is False

        disk = log.read_messages("dashboard:s10")
        assert [m["content"] for m in disk] == ["q1"]
        archives = list((tmp_path / "archive").glob("dashboard_s10__*.jsonl"))
        assert len(archives) >= 1
        content = "".join(a.read_text(encoding="utf-8") for a in archives)
        for dropped in ("a1", "q2", "a2"):
            assert dropped in content
