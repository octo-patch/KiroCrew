"""opencode user steer as a second ``session/prompt`` (``ACP_BACKENDS_STEER_CONCURRENT_PROMPT``).

opencode 1.19.0 has no steer verb. A second ``session/prompt`` sent while a turn
runs writes the user message and joins the running loop, which reads it before
its next step; both requests are answered at the same idle. These tests drive a
real ``AcpClient`` read loop over a scripted stdout and pin what the client may
report: consumed only when the steer's answer comes with the turn's own clean
end, and otherwise nothing, so the caller queues the text.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp import client as acp_client
from kiro_crew.acp.client import AcpClient
from kiro_crew.acp.types import (
    EVENT_COMPLETE,
    EVENT_STEER_CONSUMED,
    EVENT_TEXT_CHUNK,
    EVENT_TOOL_CALL,
    METHOD_CANCEL,
    METHOD_PROMPT,
)
from kiro_crew.agent_sdk import backend_cards
from kiro_crew.agent_sdk.backends import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_KIRO,
    ACP_BACKEND_OPENCODE,
    ACP_BACKENDS_STEER,
    ACP_BACKENDS_STEER_CONCURRENT_PROMPT,
)
from kiro_crew.providers.acp import AcpProvider

_SID = "ses_probe"
_STEER = "end your answer with PINEAPPLE"
_WRAPPED = f"<user_message>\n{_STEER}\n</user_message>"


def _frame(payload: dict[str, Any]) -> bytes:
    return (json.dumps({"jsonrpc": "2.0", **payload}) + "\n").encode()


def _update(update: dict[str, Any]) -> bytes:
    return _frame({"method": "session/update", "params": {"sessionId": _SID, "update": update}})


def _chunk(text: str) -> bytes:
    return _update(
        {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}
    )


def _tool_call() -> bytes:
    return _update(
        {"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "bash", "status": "pending"}
    )


def _answer(rid: int, stop: str = "end_turn") -> bytes:
    return _frame({"id": rid, "result": {"stopReason": stop, "usage": {"totalTokens": 7}}})


class _Harness:
    """An opencode ``AcpClient`` whose stdout is scripted and whose writes are captured.

    ``on_prompt`` is called with the id of each ``session/prompt`` written and
    returns the frames the fake harness answers with.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
        self.client._session_id = _SID
        self.client.ensure_ready = AsyncMock()  # type: ignore[method-assign]
        self.client._send_response = AsyncMock()  # type: ignore[method-assign]
        self.reader = asyncio.StreamReader()
        proc = MagicMock()
        proc.stdout = self.reader
        proc.stdin = MagicMock()
        proc.returncode = None
        self.client._process = proc
        self.prompts: list[tuple[int, dict]] = []
        self.cancels = 0
        self.on_prompt: Callable[[int], list[bytes]] = lambda rid: []

        async def _write(data: bytes, request_id: int, method: str) -> None:
            frame = json.loads(data)
            if frame["method"] == METHOD_PROMPT:
                self.prompts.append((request_id, frame["params"]))
                for raw in self.on_prompt(request_id):
                    self.reader.feed_data(raw)

        async def _cancel() -> None:
            self.cancels += 1
            for raw in self.on_cancel():
                self.reader.feed_data(raw)

        self.on_cancel: Callable[[], list[bytes]] = lambda: []
        self.client._write_request_bounded = _write  # type: ignore[method-assign]
        self.client._write_steer_cancel = _cancel  # type: ignore[method-assign]
        monkeypatch.setattr(acp_client, "_CONCURRENT_STEER_GRACE_SECS", 0.2)
        monkeypatch.setattr(acp_client, "_CONCURRENT_STEER_CANCEL_SECS", 0.5)

    async def run_turn(self, on_event: Callable[[Any], Any] | None = None) -> list[Any]:
        events = []
        async for event in self.client.stream_events("first", timeout=30.0):
            events.append(event)
            if on_event is not None:
                await on_event(event)
            if event.kind == EVENT_COMPLETE:
                break
        return events


def _kinds(events: list[Any]) -> list[str]:
    return [e.kind for e in events]


# ── membership and what each driver advertises ──


def test_opencode_is_the_only_member() -> None:
    assert ACP_BACKENDS_STEER_CONCURRENT_PROMPT == frozenset({ACP_BACKEND_OPENCODE})
    assert not ACP_BACKENDS_STEER_CONCURRENT_PROMPT & ACP_BACKENDS_STEER


@pytest.mark.parametrize(
    "backend,steer,refusal,loss",
    [
        (ACP_BACKEND_OPENCODE, True, False, True),
        (ACP_BACKEND_KIRO, True, True, False),
        (ACP_BACKEND_CLAUDE, False, False, True),
    ],
)
def test_the_client_answers_steer_from_the_tables(
    tmp_path: Path, backend: str, steer: bool, refusal: bool, loss: bool
) -> None:
    client = AcpClient(work_dir=tmp_path, acp_backend=backend)
    assert client.supports_steer is steer
    assert client.supports_refusal_steer is refusal
    assert client.steer_needs_loss_recovery is loss


def test_the_provider_wrapper_refuses_an_opencode_steer(tmp_path: Path) -> None:
    """Only the dashboard composer keeps the pending entry a lost steer needs."""
    provider = AcpProvider.__new__(AcpProvider)
    provider._client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
    assert provider.supports_steer is False
    assert asyncio.run(provider.steer("hello")) is False


def test_the_card_line_is_on_for_opencode() -> None:
    for backend in (ACP_BACKEND_OPENCODE, ACP_BACKEND_CODEX, ACP_BACKEND_KIRO):
        card = backend_cards.card_for(backend)
        line = {entry.id: entry.available for entry in card.capabilities}
        assert line[backend_cards.LINE_MID_TURN_STEER] is True, backend


# ── a steer the running turn read ──


@pytest.mark.asyncio
async def test_a_steer_answered_with_the_turn_is_reported_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []

    def _on_prompt(rid: int) -> list[bytes]:
        if not turn_id:
            turn_id.append(rid)
            return [_tool_call()]
        # The steer joined the running loop: one answer to both, at one idle.
        return [_chunk("done PINEAPPLE"), _answer(turn_id[0]), _answer(rid)]

    h.on_prompt = _on_prompt
    steered: list[bool] = []

    async def _steer_on_tool(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL and not steered:
            steered.append(await h.client.steer(_STEER))

    events = await h.run_turn(_steer_on_tool)

    assert steered == [True]
    steer_params = h.prompts[1][1]
    assert steer_params == {"sessionId": _SID, "prompt": [{"type": "text", "text": _STEER}]}
    consumed = [e for e in events if e.kind == EVENT_STEER_CONSUMED]
    assert [e.text for e in consumed] == [_WRAPPED]
    assert _kinds(events).index(EVENT_STEER_CONSUMED) < _kinds(events).index(EVENT_COMPLETE)
    assert h.cancels == 0


@pytest.mark.asyncio
async def test_a_steer_answer_read_before_the_turn_terminal_still_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """opencode orders the two answers of one idle in no particular way."""
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []

    def _on_prompt(rid: int) -> list[bytes]:
        if not turn_id:
            turn_id.append(rid)
            return [_tool_call()]
        return [_chunk("ok"), _answer(rid), _answer(turn_id[0])]

    h.on_prompt = _on_prompt

    async def _steer(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL:
            await h.client.steer(_STEER)

    events = await h.run_turn(_steer)
    assert [e.text for e in events if e.kind == EVENT_STEER_CONSUMED] == [_WRAPPED]


# ── a steer the turn did not read ──


@pytest.mark.asyncio
async def test_a_steer_whose_turn_ends_first_is_cancelled_and_not_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn ended before the steer landed: it runs a loop of its own.

    That loop's output comes before the steer's answer. Nothing is reported
    consumed, the loop is cancelled, and none of its text reaches this turn.
    """
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []
    steer_id: list[int] = []

    def _on_prompt(rid: int) -> list[bytes]:
        if not turn_id:
            turn_id.append(rid)
            return [_tool_call()]
        steer_id.append(rid)
        return [_answer(turn_id[0]), _chunk("its own loop")]

    h.on_prompt = _on_prompt
    h.on_cancel = lambda: [_answer(steer_id[0], "cancelled")]

    async def _steer(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL:
            await h.client.steer(_STEER)

    events = await h.run_turn(_steer)

    assert EVENT_STEER_CONSUMED not in _kinds(events)
    assert h.cancels == 1
    assert all(e.text != "its own loop" for e in events if e.kind == EVENT_TEXT_CHUNK)
    assert h.client._orphan_steers == set()


@pytest.mark.asyncio
async def test_a_steer_answered_after_another_loops_output_is_not_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []

    def _on_prompt(rid: int) -> list[bytes]:
        if not turn_id:
            turn_id.append(rid)
            return [_tool_call()]
        return [_answer(turn_id[0]), _chunk("its own loop"), _answer(rid)]

    h.on_prompt = _on_prompt

    async def _steer(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL:
            await h.client.steer(_STEER)

    events = await h.run_turn(_steer)
    assert EVENT_STEER_CONSUMED not in _kinds(events)
    assert h.cancels == 0


@pytest.mark.asyncio
async def test_a_turn_with_a_denied_approval_reports_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """opencode's loop stops on a denial, before it reads what came after it."""
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []

    def _on_prompt(rid: int) -> list[bytes]:
        if not turn_id:
            turn_id.append(rid)
            return [_tool_call()]
        return [_answer(turn_id[0]), _answer(rid)]

    h.on_prompt = _on_prompt

    async def _steer_then_deny(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL:
            await h.client.steer(_STEER)
            await h.client.reject_tool("perm-1")

    events = await h.run_turn(_steer_then_deny)
    assert EVENT_STEER_CONSUMED not in _kinds(events)


@pytest.mark.asyncio
async def test_a_refused_steer_is_not_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that refuses a second prompt answers it with an error at once."""
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []

    def _on_prompt(rid: int) -> list[bytes]:
        if not turn_id:
            turn_id.append(rid)
            return [_tool_call()]
        refusal = _frame(
            {"id": rid, "error": {"code": -32603, "message": "already has an active prompt"}}
        )
        return [refusal, _chunk("ok"), _answer(turn_id[0])]

    h.on_prompt = _on_prompt

    async def _steer(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL:
            await h.client.steer(_STEER)

    events = await h.run_turn(_steer)
    assert EVENT_STEER_CONSUMED not in _kinds(events)
    assert _kinds(events)[-1] == EVENT_COMPLETE


# ── when nothing is sent ──


@pytest.mark.asyncio
async def test_no_steer_is_sent_without_a_running_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _Harness(tmp_path, monkeypatch)
    assert await h.client.steer(_STEER) is False
    assert h.prompts == []


@pytest.mark.asyncio
async def test_a_steer_past_the_bound_is_not_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []

    def _on_prompt(rid: int) -> list[bytes]:
        turn_id.append(rid)
        return [_tool_call()] if len(turn_id) == 1 else []

    h.on_prompt = _on_prompt
    results: list[bool] = []

    async def _steer(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL:
            results.append(await h.client.steer("x" * (acp_client._MAX_CONCURRENT_STEER_CHARS + 1)))
            h.reader.feed_data(_answer(turn_id[0]))

    await h.run_turn(_steer)
    assert results == [False]
    assert len(h.prompts) == 1


@pytest.mark.asyncio
async def test_a_steer_written_as_its_turn_ends_is_cancelled_before_the_next_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn ended during the write: False now, and its loop is stopped later.

    Its frames are read and dropped before the next prompt goes out, so the next
    turn never shows the steer's own loop as its answer.
    """
    h = _Harness(tmp_path, monkeypatch)
    turn_id: list[int] = []
    steer_id: list[int] = []
    results: list[bool] = []

    def _on_prompt(rid: int) -> list[bytes]:
        if not turn_id:
            turn_id.append(rid)
            return [_tool_call()]
        if not steer_id:
            steer_id.append(rid)
            # The turn ends while the steer frame is being written.
            h.client._turn_done.set()
            return []
        return [_chunk("second answer"), _answer(rid)]

    h.on_prompt = _on_prompt
    h.on_cancel = lambda: [_answer(steer_id[0], "cancelled")]

    async def _steer(event: Any) -> None:
        if event.kind == EVENT_TOOL_CALL and not results:
            results.append(await h.client.steer(_STEER))
            h.reader.feed_data(_answer(turn_id[0]))

    await h.run_turn(_steer)
    assert results == [False]
    assert h.client._orphan_steers == {steer_id[0]}

    # The steer's own loop streams after the turn is over, with no reader.
    h.reader.feed_data(_chunk("its own loop"))
    second = await h.run_turn()
    assert h.cancels == 1
    texts = [e.text for e in second if e.kind == EVENT_TEXT_CHUNK]
    assert texts == ["second answer"]
    assert h.client._orphan_steers == set()


@pytest.mark.asyncio
async def test_the_steer_cancel_is_a_session_cancel_notification(tmp_path: Path) -> None:
    client = AcpClient(work_dir=tmp_path, acp_backend=ACP_BACKEND_OPENCODE)
    client._session_id = _SID
    proc = MagicMock()
    proc.returncode = None
    proc.stdin = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.transport.get_write_buffer_size.return_value = 0
    client._process = proc

    await client._write_steer_cancel()

    written = b"".join(call.args[0] for call in proc.stdin.write.call_args_list)
    assert json.loads(written) == {
        "jsonrpc": "2.0",
        "method": METHOD_CANCEL,
        "params": {"sessionId": _SID},
    }
