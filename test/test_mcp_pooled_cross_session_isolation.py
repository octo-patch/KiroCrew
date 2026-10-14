"""One shared backend, two callers: nothing crosses between them.

A default install now stubs every managed Kiro Crew server and shares its backend
across sessions. That is safe only if each server tells its callers apart by the
caller block gatewayd injects per call, never by its own process. So each test
here starts ONE real server process through its production serve entry, with the
process environment holding a third, decoy identity (what a backend spawned from
gatewayd's environment would see), and sends it ``tools/call`` frames from two
sessions in turn, each with its own ``_meta`` caller block.

The only seam replaced is the loopback HTTP transport at the bottom of the
request helpers, which records each request and answers it. Everything above it
is production: the stdio loop parses the caller block, the per-session
``managedToolPolicy`` cache keys on it, and the server builds its headers and
body from it. For every server the tests prove:

* every request a call makes carries that caller's key and token, never the
  other caller's and never the decoy's;
* a tool one session's policy excludes is refused for that session and still
  served to the other, both before and after (the policy cache is per caller).

``kirocrew-cron`` additionally proves its two caller-derived defaults: the
delivery channel and the Slack reply thread come from each caller's own block,
and a caller whose block names no channel gets none, not the decoy's.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from kiro_crew import mcp_shared
from kiro_crew.mcp_caller import CALLER_META_KEY
from kiro_crew.mcp_discovery import _MANAGED_SERVER_TOOL_MODULES

SRC = str(Path(mcp_shared.__file__).resolve().parents[1])

#: The decoy: what a backend spawned from gatewayd's environment holds. It names
#: no caller, so seeing any of it on the wire is a cross-session leak.
_DECOY_ENV = {
    "KIROCREW_SESSION_KEY": "decoy:gatewayd-env",
    "KIROCREW_CHANNEL_ID": "C-DECOY",
    "KIROCREW_APPROVAL_MODE": "auto",
}
_DECOY_VALUES = ("decoy:gatewayd-env", "C-DECOY")

#: Runs one server through its real serve entry. Only the loopback transport is
#: replaced: it appends each request to a log and answers it.
_BOOT = textwrap.dedent("""
    import importlib, json, os, sys
    from kiro_crew import mcp_core, mcp_shared

    module_name, entry, log_path = sys.argv[1:4]
    policy = json.loads(os.environ["ISOLATION_TEST_POLICY"])

    class _Resp:
        def __init__(self, body):
            self._body = json.dumps(body).encode()
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    def _transport(req, timeout=None, unix_socket_path=None, **_kw):
        headers = {k.lower(): v for k, v in req.header_items()}
        body = req.data.decode() if req.data else ""
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": req.full_url, "headers": headers, "body": body}) + "\\n")
        if req.full_url.endswith("/api/session-tool-policy"):
            return _Resp({"exclude": policy.get(headers.get("x-session-key", ""), [])})
        return _Resp({"result": "ok", "ok": True, "text": "ok"})

    mcp_core._API = "http://127.0.0.1:9"
    mcp_core._API_UNIX_SOCKET = ""
    mcp_core._api_urlopen = _transport
    mcp_core._internal_secret = lambda: "test-secret"
    mcp_shared.loopback_urlopen = _transport
    mcp_shared.resolve_client_port_src = lambda _p: (9, "test")
    mcp_shared.read_local_secret = lambda *_a, **_k: "test-secret"
    module = importlib.import_module(module_name)
    if module_name == "kiro_crew.mcp_computer":
        module.loopback_urlopen = _transport
        module.enable_state.is_enabled = lambda: True
    getattr(module, entry)()
    """)

#: Server -> (serve entry, a tool whose call reaches the gateway, its arguments).
_CASES: dict[str, tuple[str, str, dict]] = {
    "kirocrew-core": ("run_mcp_core_server", "learn_list", {}),
    "kirocrew-cron": ("run_mcp_server", "cron_list", {}),
    "kirocrew-computer": ("run_mcp_server", "computer_list_apps", {}),
    "kirocrew-dashboard": ("run_mcp_server", "chat_folder_tree", {}),
    "kirocrew-work": ("run_mcp_server", "work_brief", {}),
    "kirocrew-crew-log": ("run_mcp_server", "crew_log_list", {}),
    "kirocrew-debug": ("run_mcp_server", "debug_gateway", {}),
    "kirocrew-panel": ("run_mcp_server", "panel_templates", {}),
    "kirocrew-guide": ("run_mcp_server", "guide_list_actions", {}),
    "kirocrew-mate": ("run_mcp_server", "update_self", {"display_name": "Mate"}),
}

#: Bound on one answer. It separates "answered" from "hung", not fast from slow.
_ANSWER_DEADLINE = 60.0


@dataclass(frozen=True)
class _Caller:
    key: str
    token: str
    channel: str = ""

    def meta(self) -> dict:
        block = {"schemaVersion": 1, "sessionKey": self.key, "sessionToken": self.token}
        block["sessionType"] = self.key.partition(":")[0]
        if self.channel:
            block["channelId"] = self.channel
        return {CALLER_META_KEY: block}


_A = _Caller("dashboard:tab-alpha", "tok-alpha")
_B = _Caller("slack:1700000000.000200", "tok-bravo", channel="C-BRAVO")


class _SharedBackend:
    """One real server process, fed frames from several callers in turn."""

    def __init__(self, tmp_path: Path, server: str, policy: dict[str, list[str]]) -> None:
        entry = _CASES[server][0]
        self.log = tmp_path / "requests.jsonl"
        self.log.touch()
        home = tmp_path / "home"
        home.mkdir()
        # The platform's own variables (Windows needs several to start Python at
        # all), minus every Kiro Crew and interpreter setting of the test run.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.upper().startswith(("KIROCREW_", "PYTHON", "MC_MCP_"))
        }
        env.update(
            {
                "HOME": str(tmp_path),
                "USERPROFILE": str(tmp_path),
                "KIROCREW_HOME": str(home),
                "PYTHONPATH": SRC,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUTF8": "1",
                "ISOLATION_TEST_POLICY": json.dumps(policy),
                **_DECOY_ENV,
            }
        )
        self.stderr_path = tmp_path / "server.stderr"
        self._stderr = open(self.stderr_path, "wb")
        self.p = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _BOOT,
                _MANAGED_SERVER_TOOL_MODULES[server],
                entry,
                str(self.log),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            env=env,
            cwd=str(tmp_path),
        )
        self.rid = 0
        self._lines: queue.Queue[bytes] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _stderr_tail(self) -> str:
        self._stderr.flush()
        return self.stderr_path.read_bytes()[-4000:].decode("utf-8", "replace")

    def _pump(self) -> None:
        assert self.p.stdout is not None
        for line in self.p.stdout:
            self._lines.put(line)
        self._lines.put(b"")

    def call(self, caller: _Caller, tool: str, args: dict) -> tuple[dict, list[dict]]:
        """One ``tools/call`` as *caller*; its response and the requests it made."""
        seen = len(self.log.read_text(encoding="utf-8").splitlines())
        self.rid += 1
        frame = {
            "jsonrpc": "2.0",
            "id": self.rid,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args, "_meta": caller.meta()},
        }
        assert self.p.stdin is not None
        self.p.stdin.write((json.dumps(frame) + "\n").encode())
        self.p.stdin.flush()
        while True:
            line = self._lines.get(timeout=_ANSWER_DEADLINE)
            assert line, f"server closed stdout; stderr:\n{self._stderr_tail()}"
            msg = json.loads(line)
            if msg.get("id") == self.rid:
                break
        requests = [
            json.loads(row) for row in self.log.read_text(encoding="utf-8").splitlines()[seen:]
        ]
        return msg, requests

    def close(self) -> None:
        assert self.p.stdin is not None
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        finally:
            if self.p.poll() is None:
                self.p.kill()
                self.p.wait(timeout=30)
            self._stderr.close()


def _text(msg: dict) -> str:
    result = msg.get("result") or {}
    return "".join(c.get("text", "") for c in result.get("content", []))


def _assert_attributed_to(caller: _Caller, other: _Caller, requests: list[dict]) -> None:
    assert requests, f"{caller.key}'s call reached no gateway request"
    for req in requests:
        headers = req["headers"]
        assert headers.get("x-session-key") == caller.key, req["url"]
        assert headers.get("x-session-token") == caller.token, req["url"]
        wire = json.dumps(req)
        for foreign in (other.key, other.token, *_DECOY_VALUES):
            assert foreign not in wire, f"{foreign!r} reached {caller.key}'s {req['url']}"


@pytest.mark.parametrize("server", sorted(_CASES))
def test_two_callers_on_one_backend_never_cross(server: str, tmp_path: Path) -> None:
    _entry, tool, args = _CASES[server]
    # Bravo's operator excluded the tool; Alpha's did not.
    backend = _SharedBackend(tmp_path, server, {_B.key: [tool]})
    try:
        for caller, other, served in ((_A, _B, True), (_B, _A, False)) * 2:
            msg, requests = backend.call(caller, tool, args)
            policy_reads = [r for r in requests if r["url"].endswith("/session-tool-policy")]
            tool_requests = [r for r in requests if r not in policy_reads]
            # Each policy read names its own caller; the cache never answers one
            # session with another's policy.
            for read in policy_reads:
                assert read["headers"].get("x-session-key") == caller.key
            if served:
                _assert_attributed_to(caller, other, tool_requests)
            else:
                assert tool_requests == [], f"{tool} ran for {caller.key} despite its policy"
                assert "not available" in _text(msg) or msg.get("error"), msg
    finally:
        backend.close()


def test_cron_defaults_come_from_each_callers_own_block(tmp_path: Path) -> None:
    """``cron_add`` with no channel: each caller's own channel and thread, or none.

    Bravo is a Slack thread session, so its job defaults to its channel and replies
    into its thread. Alpha is a dashboard tab whose block names no channel, so its
    job gets no channel and no thread -- in particular not the decoy channel in the
    backend's environment.
    """
    backend = _SharedBackend(tmp_path, "kirocrew-cron", {})
    add = {"message": "ping", "every": 3600}
    try:
        for _ in range(2):
            _msg, a_reqs = backend.call(_A, "cron_add", add)
            _msg, b_reqs = backend.call(_B, "cron_add", add)
            [a_post] = [r for r in a_reqs if r["url"].endswith("/api/crons/tools")]
            [b_post] = [r for r in b_reqs if r["url"].endswith("/api/crons/tools")]
            a_args = json.loads(a_post["body"])["arguments"]
            b_args = json.loads(b_post["body"])["arguments"]
            assert "channel" not in a_args and "thread_ts" not in a_args, a_args
            assert b_args["channel"] == "C-BRAVO"
            assert b_args["thread_ts"] == "1700000000.000200"
            _assert_attributed_to(_A, _B, [a_post])
            _assert_attributed_to(_B, _A, [b_post])
    finally:
        backend.close()


def test_a_pooled_spawn_names_its_caller_as_parent(tmp_path: Path) -> None:
    """A pooled cron's ``spawn_run`` auto-approves through its own session policy.

    The admission gate's ``parent_trusted`` reads the approval policy of the
    request's ``parent_session``, which the cron dispatch set to "auto" for the
    cron's own key. So on a shared backend the request must name the caller
    block's key, and must not carry the backend env's approval mode (the decoy
    is "auto", which would approve every caller's subagents).
    """
    cron = _Caller("cron:job-alpha", "tok-cron")
    backend = _SharedBackend(tmp_path, "kirocrew-core", {})
    try:
        for caller, other in ((cron, _B), (_B, cron)):
            _msg, requests = backend.call(
                caller, "spawn_run", {"task": "summarize", "solo_reason": "bulk_data"}
            )
            [post] = [r for r in requests if r["url"].endswith("/api/spawn")]
            body = json.loads(post["body"])
            assert body["parent_session"] == caller.key
            assert "approval_mode" not in body
            _assert_attributed_to(caller, other, [post])
    finally:
        backend.close()
