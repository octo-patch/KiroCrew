"""Each server in the default roster keeps its per-session fallback.

Phase 1 stubs all ten managed servers by default, so these two existing paths now
run for every default install. Both are pinned for each of the ten:

* A stub that cannot be served by gatewayd (daemon down, a fallback-eligible
  refusal) execs the server's real managed invocation itself, and keeps the
  session token, because the real ``_spawns_own_control_plane`` vetting accepts
  it. Without the token a fallback server refuses every call
  ``identity_unattested``.
* A crash-looping backend trips the circuit breaker, and the daemon answers
  ``capacity`` with no fallback authorisation: the stub keeps its transport and
  answers with a typed error rather than forking a process the host budget never
  sees. A session still reaches its server again once the cooldown elapses.

Named ``test_mcp_gateway_stub_*`` so the macOS job's glob selects it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiro_crew.config.integration_sections import DEFAULT_STUB_ROSTER
from kiro_crew.mcp_gateway import gatewayd as gw
from kiro_crew.mcp_gateway import stub as stub_mod
from kiro_crew.mcp_gateway.breaker import FAST_DEATH_LIMIT, OPEN_COOLDOWN_SECS, CircuitBreaker
from kiro_crew.mcp_gateway.claim import STUB_SESSION_TOKEN_ENV
from kiro_crew.mcp_gateway.pool import BackendUnavailable

_TOKEN = "tok-default-roster"


class _Exec(Exception):
    """Raised in place of ``execvpe`` so the fallback's argv and env can be read."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        super().__init__(argv[0])
        self.argv = argv
        self.env = env


@pytest.mark.parametrize("server", DEFAULT_STUB_ROSTER)
def test_a_fallback_execs_the_real_server_with_its_token(
    server: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kiro_crew.agent import managed_mcp_spec_entry
    from kiro_crew.agent_materialization import managed_mcp

    # ``kirocrew-computer`` is spawned only where its spec gate is open (a
    # supported driver and the keystone enable). Open it here: a closed gate
    # correctly refuses the token, and this test is about a server that runs.
    monkeypatch.setattr(managed_mcp, "_mcp_spec_gate_open", lambda _name, _spec: True)
    entry = managed_mcp_spec_entry(server, include_opt_in=True)
    assert entry is not None, server
    from kiro_crew.platform_compat import _UTF8_PROCESS_ENV

    monkeypatch.setenv(STUB_SESSION_TOKEN_ENV, _TOKEN)
    # A session's stub env: no loader or interpreter overlay beyond Kiro Crew's
    # own pinned UTF-8 keys (the test runner sets others, which the vetting
    # rightly refuses).
    for key in list(stub_mod.os.environ):
        if key.upper().startswith(("LD_", "DYLD_", "PYTHON")) and key not in _UTF8_PROCESS_ENV:
            monkeypatch.delenv(key, raising=False)
    for key, value in _UTF8_PROCESS_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    def _capture(_file: str, argv: list[str], env: dict[str, str]) -> Any:
        raise _Exec(list(argv), dict(env))

    monkeypatch.setattr(stub_mod.os, "execvpe", _capture)
    monkeypatch.setattr(stub_mod.platform_compat, "IS_WINDOWS", False)
    args = stub_mod._parse_args(
        [
            "--server",
            server,
            "--agent",
            "kirocrew",
            "--target-command",
            entry["command"],
            "--target-args",
            "|".join(entry["args"]),
            "--work-dir",
            str(tmp_path),
        ]
    )
    with pytest.raises(_Exec) as caught:
        stub_mod.fallback_exec(args)
    assert caught.value.argv == [entry["command"], *entry["args"]]
    assert caught.value.env.get(STUB_SESSION_TOKEN_ENV) == _TOKEN, (
        f"{server}'s per-session fallback lost the session token, so every call "
        "it serves would be refused identity_unattested"
    )


@pytest.mark.parametrize("server", DEFAULT_STUB_ROSTER)
def test_a_tripped_breaker_answers_capacity_and_reopens(server: str) -> None:
    class _Clock:
        t = 1000.0

        def __call__(self) -> float:
            return self.t

    clock = _Clock()
    breaker = CircuitBreaker(now_fn=clock)
    key = f"pool:{server}"
    for _ in range(FAST_DEATH_LIMIT):
        breaker.record_death(key, uptime_secs=0.1)
    assert breaker.allow(key) is False

    verdict = gw._classify_rejection(
        BackendUnavailable(f"circuit breaker OPEN for server {server!r}"), exclusive=False
    )
    assert verdict is not None
    assert (verdict.cls, verdict.fallback) == ("capacity", False)
    assert "fallback" not in verdict.frame("breaker open")

    clock.t += OPEN_COOLDOWN_SECS + 1
    assert breaker.allow(key) is True
