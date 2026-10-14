"""Dashboard chat: matching the unflushed tail a page owes against the disk window."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_app, _make_state


class TestSlotDetailPaginationUnflushedTail:
    """TestSlotDetailPagination: matching the unflushed tail against the disk window."""

    @pytest.mark.asyncio
    async def test_transient_window_row_does_not_duplicate_the_tail(self, tmp_path, monkeypatch):
        """A transient row in the window must not re-append rows already on disk.

        A save drops transient roles, so the disk read holds only persisted rows
        while the window holds both. Sizing the un-flushed tail by subtracting the
        two lengths therefore counts each transient row as one missing message and
        reaches that many rows too far back — rows the disk read already returned.

        Here everything persisted is on disk, so the correct tail is empty. The
        window carries one ``done`` row, which ``_prepare_messages`` drops from the
        response, so a duplicated row is directly visible in the content sequence.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("dup")
        log = state.conversation_log
        for i in range(4):
            log.append("dashboard:dup", "user", f"msg {i}")
        slot.append("user", "msg 0")
        slot.append("user", "msg 1")
        slot.append("done", "")
        slot.append("user", "msg 2")
        slot.append("user", "msg 3")
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/dup?limit=10")
            assert resp.status == 200
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        assert contents == [
            f"msg {i}" for i in range(4)
        ], "un-flushed tail sized by a length subtraction re-appended a persisted row"
        assert data["total"] == 4

    @pytest.mark.asyncio
    async def test_genuinely_unflushed_tail_is_still_appended_once(self, tmp_path, monkeypatch):
        """Control for the test above: a real un-flushed tail must still arrive.

        Only the first two window rows reached disk, so the last two are owed to the
        client. A transient row sits between them, so a fix that merely stopped
        appending would pass the duplication test and fail this one.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("owed")
        log = state.conversation_log
        for i in range(2):
            log.append("dashboard:owed", "user", f"msg {i}")
        slot.append("user", "msg 0")
        slot.append("user", "msg 1")
        slot.append("done", "")
        slot.append("user", "msg 2")
        slot.append("user", "msg 3")
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/owed?limit=10")
            assert resp.status == 200
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        assert contents == [f"msg {i}" for i in range(4)], "un-flushed tail was dropped"

    @pytest.mark.asyncio
    async def test_foreign_disk_row_does_not_consume_the_unflushed_tail(
        self, tmp_path, monkeypatch
    ):
        """A disk row absent from the window must not consume an owed turn.

        This is the OVER-count direction. A writer other than this slot's own save
        appends to the same transcript, so the chained disk read grows without any
        window row having been flushed. Sizing the boundary as
        ``len(all_msgs) - _disk_older_count`` therefore counts that foreign row as
        one more persisted window row, walks one row too far, and drops the
        genuinely un-flushed turn from a response the client uses as a replacement.
        Under-count fails safe with an empty tail; over-count loses a message.
        """
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("overcount")
        for i in range(3):
            slot.append("user", f"msg {i}")
        slot.drain()
        # Persist through the real save path, so the disk rows carry the window's
        # own message ids rather than a shape only a test helper produces.
        _save_slot_to_history(state, slot, force=True)
        # A writer that does not mirror into the window appends to the transcript.
        state.conversation_log.append("dashboard:overcount", "assistant", "foreign row")
        # A new turn arrives in memory and has not reached disk.
        slot.append("user", "new turn")
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/overcount?limit=10")
            assert resp.status == 200
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        assert "new turn" in contents, f"over-count dropped the owed turn; got {contents}"

    @pytest.mark.asyncio
    async def test_non_string_message_id_does_not_crash_the_bounded_read(
        self, tmp_path, monkeypatch
    ):
        """A caller-supplied non-string ``meta.mid`` must not turn the read into a 500.

        ``meta`` on an inbound message comes from the HTTP caller and is checked
        only for being a dict, so its values keep whatever type arrived. A truthy
        non-string id is preserved rather than replaced, reaches disk, and is then
        hashed by the boundary match, where a list raises ``TypeError``.
        """
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", AsyncMock())
        state = _make_state(tmp_path)
        state.get_or_create_slot("listmid")

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "hi",
                    "slot": "listmid",
                    "meta": {"mid": ["not-a-string"]},
                },
            )
            assert resp.status == 200
            slot = state.get_or_create_slot("listmid")
            slot.drain()
            # The id must be on disk as well as in the window: the boundary match
            # hashes both sides.
            _save_slot_to_history(state, slot, force=True)
            assert slot.messages[-1]["meta"]["mid"] == [
                "not-a-string"
            ], "fixture no longer reproduces a non-string id"

            resp = await client.get("/api/chat/slots/listmid?limit=10")
            assert resp.status == 200, await resp.text()

    @pytest.mark.asyncio
    async def test_foreign_row_does_not_consume_the_tail_of_an_id_less_session(
        self, tmp_path, monkeypatch
    ):
        """The same over-count, on a session whose disk rows carry no message ids.

        The id match cannot help here: a session persisted before ids existed has
        none on disk, so the boundary falls back to counting. Sizing it as
        ``len(all_msgs) - _disk_older_count`` measures the whole file rather than
        this slot's own persisted rows, so a row appended by any other writer is
        counted as one more flushed window row and the owed turn is dropped.

        ``_disk_window_len`` is how many window rows are on disk, which is the
        quantity the subtraction was approximating, and a foreign append does not
        move it.
        """
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("legacy")
        log = state.conversation_log
        # ConversationLog.append writes role/content/ts only, so these rows reach
        # disk with no ids -- what a pre-id session looks like.
        for i in range(3):
            log.append("dashboard:legacy", "user", f"msg {i}")
        for i in range(3):
            slot.append("user", f"msg {i}")
        slot.drain()
        # Mirror what the restore path leaves behind for such a session.
        slot._disk_older_count = 0
        slot._disk_window_len = len(slot.messages)
        slot._dirty = False
        # A writer that does not mirror into the window appends to the transcript.
        log.append("dashboard:legacy", "assistant", "foreign row")
        slot.append("user", "new turn")
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/legacy?limit=10")
            assert resp.status == 200
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        assert "new turn" in contents, f"over-count dropped the owed turn; got {contents}"
        assert contents.count("msg 2") == 1, f"boundary walked short and duplicated; got {contents}"

    def test_id_less_boundary_matches_a_row_redacted_on_load(self):
        """A restored row keeps its ``ts`` verbatim but not always its content.

        Restore redacts non-user content before it enters the window while the disk
        row stays raw, so matching on content alone ends the run at that row and
        re-appends everything from there — rows the disk read already returned.

        The pair is taken from the REAL redactor rather than written by hand: an
        invented pair passes for the wrong reason, because a stamp match would carry
        it even when the transform would not.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail
        from kiro_crew.security import redact_credentials, redact_exfiltration_urls

        raw = "AKIAIOSFODNN7EXAMPLE"
        loaded, _ = redact_exfiltration_urls(raw)
        loaded, _ = redact_credentials(loaded)
        assert loaded != raw, "fixture no longer exercises a real redaction"

        all_msgs = [
            {"role": "user", "content": "ask", "ts": "t1"},
            {"role": "assistant", "content": raw, "ts": "t2"},
        ]
        slot = SimpleNamespace(
            messages=[
                {"role": "user", "content": "ask", "ts": "t1"},
                {"role": "assistant", "content": loaded, "ts": "t2"},
                {"role": "user", "content": "new turn", "ts": "t3"},
            ],
            _disk_older_count=0,
        )

        out = _append_unflushed_tail(slot, all_msgs)  # type: ignore[arg-type]

        assert [m["content"] for m in out] == [
            "ask",
            raw,
            "new turn",
        ], "a row whose content was redacted on load was treated as un-flushed"

    def test_shared_stamp_foreign_row_does_not_swallow_an_unflushed_row(self):
        """A foreign row sharing a ``ts`` must not be accepted as the window's own.

        On a coarse clock two writers flooring off the same previous row both emit
        ``previous + 1µs`` (``history.py:1179-1219``), so a foreign disk row and an
        un-flushed window row of the SAME role can carry an identical stamp while
        holding different messages. Accepting a stamp match on its own then treats
        the foreign row as the window row, the boundary advances past the un-flushed
        one, and the bounded response omits it — from a payload the client uses to
        REPLACE its transcript, so the message is simply gone from view.

        Content is the discriminator: these two rows differ by more than the load
        redaction, so nothing may match them.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        collided = "2026-08-18T12:00:00.000001+00:00"
        all_msgs = [
            {"role": "user", "content": "q", "ts": "2026-08-18T12:00:00+00:00"},
            {"role": "assistant", "content": "another writer's row", "ts": collided},
        ]
        slot = SimpleNamespace(
            messages=[
                {"role": "user", "content": "q", "ts": "2026-08-18T12:00:00+00:00"},
                {"role": "assistant", "content": "owed reply", "ts": collided},
            ],
            _disk_older_count=0,
        )
        assert all_msgs[1]["ts"] == slot.messages[1]["ts"], "fixture lost the collision"
        assert all_msgs[1]["role"] == slot.messages[1]["role"], "fixture lost same-role"

        out = _append_unflushed_tail(slot, all_msgs)  # type: ignore[arg-type]

        assert "owed reply" in [
            m["content"] for m in out
        ], f"a shared stamp swallowed the un-flushed row; got {[m['content'] for m in out]}"

    @pytest.mark.asyncio
    async def test_repeated_caller_supplied_id_does_not_hide_an_unflushed_row(
        self, tmp_path, monkeypatch
    ):
        """A caller may repeat ``meta.mid``, and a set membership test over-reaches.

        ``meta`` on an inbound message is caller-supplied and an id is minted only
        when one is *absent*, so a client can post the same id twice. The first row
        reaches disk; the second is still only in the window. Testing membership
        against a *set* of disk ids matches both window rows, so the boundary walks
        past the row that was never persisted and the bounded read omits it — the
        silent-loss direction. One disk row is enough to reproduce it; two disk rows
        sharing an id are not required.
        """
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers._run_chat", AsyncMock())
        state = _make_state(tmp_path)
        state.get_or_create_slot("dupmid")
        dup = "m-deadbeefdeadbeef"

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat",
                json={"message": "first", "slot": "dupmid", "meta": {"mid": dup}},
            )
            assert resp.status == 200
            slot = state.get_or_create_slot("dupmid")
            slot.drain()
            # Only the first row reaches disk, so the id is on disk exactly once.
            _save_slot_to_history(state, slot, force=True)

            resp = await client.post(
                "/api/chat",
                json={"message": "second", "slot": "dupmid", "meta": {"mid": dup}},
            )
            assert resp.status == 200
            slot.drain()
            mids = [m["meta"].get("mid") for m in slot.messages if isinstance(m.get("meta"), dict)]
            assert mids.count(dup) == 2, f"fixture no longer reproduces a repeated id; got {mids}"

            resp = await client.get("/api/chat/slots/dupmid?limit=10")
            assert resp.status == 200, await resp.text()
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        assert "second" in contents, f"a repeated id hid the un-flushed row; got {contents}"
        assert contents.count("first") == 1, f"a row was duplicated; got {contents}"

    def test_frozen_prefix_id_does_not_fund_a_window_match(self):
        """An id occurrence in the frozen prefix must not count as a flushed window row.

        ``all_msgs[:_disk_older_count]`` is the frozen prefix — on-disk rows OLDER
        than the window, so none of them is in ``slot.messages``. Counting ids over
        the whole disk read lets such an occurrence fund a consumption for a window
        row that was never flushed, and the boundary then walks past it. Only the
        on-disk window region, ``all_msgs[_disk_older_count:]``, may fund a match.

        Reachable because a caller-supplied ``mid`` is preserved rather than re-minted
        (``state.py:1801-1803``), which is what lets a redelivered row carry an id
        whose original has since been paged into the prefix.
        """
        from types import SimpleNamespace

        from kiro_crew.dashboard.chat_handlers import _append_unflushed_tail

        dup = "m-deadbeefdeadbeef"
        all_msgs = [
            {"role": "user", "content": "paged out", "ts": "t0", "meta": {"mid": dup}},
            {"role": "user", "content": "on disk", "ts": "t1", "meta": {"mid": "m-window1"}},
        ]
        slot = SimpleNamespace(
            messages=[
                {"role": "user", "content": "on disk", "ts": "t1", "meta": {"mid": "m-window1"}},
                {"role": "user", "content": "owed turn", "ts": "t2", "meta": {"mid": dup}},
            ],
            _disk_older_count=1,
        )
        # Guard the fixture: with no frozen prefix this case cannot arise at all and
        # the test would pass against the defect.
        assert slot._disk_older_count > 0, "fixture establishes no frozen prefix"

        out = _append_unflushed_tail(slot, all_msgs)  # type: ignore[arg-type]

        assert [m["content"] for m in out] == [
            "paged out",
            "on disk",
            "owed turn",
        ], "a frozen-prefix id funded a match and hid the owed turn"

    @pytest.mark.asyncio
    async def test_mixed_id_and_id_less_disk_window_does_not_duplicate_an_injection(
        self, tmp_path, monkeypatch
    ):
        """A disk window holding BOTH id-carrying and id-less rows must not duplicate.

        The dual-write injectors stamp both copies of an injection with one id, so
        a NEW injection does not produce this state — but transcripts written
        before the append path accepted an id hold exactly it, as does any caller
        that passes no id: earlier saved rows WITH ids plus a durable row WITHOUT
        one. The id-less ``append_if_absent`` call below is that legacy writer
        shape, and it must keep working unmigrated.

        Selecting id matching because SOME row carries an id then applies it to a row
        that structurally cannot match, so the injection is treated as un-flushed and
        appended a second time. Id matching is only sound when EVERY row in the disk
        window carries one; a mixed window belongs on the ordered path, which matches
        the fields both writers do record.

        Driven through the real writers rather than a hand-built mixed window, so it
        cannot encode a row shape neither writer emits.
        """
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("mixed")
        key = "dashboard:mixed"

        slot.append("user", "q1")
        slot.append("assistant", "a1")
        slot.drain()
        # Real save: these disk rows carry the window's own ids.
        _save_slot_to_history(state, slot, force=True)

        # The injector shape: window copy (id minted) plus durable copy (no meta).
        slot.append("assistant", "injected result")
        slot.drain()
        state.conversation_log.append_if_absent(key, "assistant", "injected result")

        disk = state.conversation_log.read_messages_chained(key)
        with_id = [
            m
            for m in disk
            if isinstance(m.get("meta"), dict) and isinstance(m["meta"].get("mid"), str)
        ]
        without_id = [
            m
            for m in disk
            if not (isinstance(m.get("meta"), dict) and isinstance(m["meta"].get("mid"), str))
        ]
        assert with_id and without_id, (
            f"fixture did not produce a MIXED disk window; "
            f"with_id={len(with_id)} without_id={len(without_id)}"
        )

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/mixed?limit=10")
            assert resp.status == 200, await resp.text()
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        assert (
            contents.count("injected result") == 1
        ), f"the durable injection was appended twice; got {contents}"

    @pytest.mark.asyncio
    async def test_durable_injection_carrying_the_window_rows_id_matches_by_identity(
        self, tmp_path, monkeypatch
    ):
        """Both copies of an injection carry ONE id, so the identity walk matches.

        The dual-write shape: the window copy is appended through ``slot.append``,
        which mints ``meta.mid`` and returns the row; the durable copy passes that
        same id to ``append_if_absent``. The on-disk window region then holds ONLY
        id-carrying rows, so the bounded read reconciles by identity rather than a
        body heuristic, and the injection appears exactly once.

        Guarded on the all-id premise: if the durable copy ever stops carrying the
        id, the fixture degrades to the mixed shape and the guard names that,
        instead of the count assertion passing by way of the ordered fallback.
        """
        from kiro_crew.dashboard.chat_persistence import _save_slot_to_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("dualid")
        key = "dashboard:dualid"

        slot.append("user", "q1")
        slot.append("assistant", "a1")
        slot.drain()
        # Real save: these disk rows carry the window's own ids.
        _save_slot_to_history(state, slot, force=True)

        # The injector shape: one id minted in the window, the SAME id persisted
        # on the durable copy.
        window_row = slot.append("assistant", "injected result")
        slot.drain()
        mid = window_row["meta"]["mid"]
        assert (
            state.conversation_log.append_if_absent(key, "assistant", "injected result", mid=mid)
            is True
        )

        disk = state.conversation_log.read_messages_chained(key)
        region_mids = [
            m["meta"].get("mid") if isinstance(m.get("meta"), dict) else None for m in disk
        ]
        assert all(
            isinstance(x, str) and x for x in region_mids
        ), f"the durable copy dropped the id — the region is mixed, not all-id: {region_mids}"
        assert mid in region_mids, "the durable copy's id is not the window row's id"

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/dualid?limit=10")
            assert resp.status == 200, await resp.text()
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        assert (
            contents.count("injected result") == 1
        ), f"the identity walk failed to match the durable copy; got {contents}"

    @pytest.mark.asyncio
    async def test_hydrated_slot_over_an_all_id_transcript_is_not_served_twice(
        self, tmp_path, monkeypatch
    ):
        """Hydration must keep the disk rows' ids, or an all-id region doubles.

        With durable injection copies id-carrying, a cron transcript's window
        region can be ALL-id — the condition that selects the identity walk.
        ``hydrate_slot_from_history`` re-appends those rows into a fresh slot;
        if it dropped their ``meta``, every hydrated row would be re-minted a
        fresh id, no window id would match the disk, and the walk would mark
        the WHOLE history owed — a bounded read then serves every row twice
        until the next flush rewrites the file.
        """
        from kiro_crew.dashboard.cron_inject import hydrate_slot_from_history

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        key = "dashboard:cronhyd"

        # An all-id transcript: what a session looks like once every writer
        # (slot save and id-carrying durable appends) stamps meta.mid.
        state.conversation_log.append(key, "user", "q1", mid="m-disk-0000000001")
        state.conversation_log.append(key, "assistant", "a1", mid="m-disk-0000000002")
        state.conversation_log.append(key, "assistant", "cron result", mid="m-disk-0000000003")

        slot = state.get_or_create_slot("cronhyd")
        hydrate_slot_from_history(slot, state.conversation_log.read_messages(key))
        slot.drain()

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/cronhyd?limit=10")
            assert resp.status == 200, await resp.text()
            data = await resp.json()

        contents = [m["content"] for m in data["messages"]]
        for body in ("q1", "a1", "cron result"):
            assert contents.count(body) == 1, f"a hydrated row was served twice; got {contents}"
