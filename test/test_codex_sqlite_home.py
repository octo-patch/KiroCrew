"""Each codex runtime gets its own SQLite home.

Every ``codex app-server`` opens the same SQLite files by default and they lock
each other out: Codex Desktop plus a Crew runtime fail new sessions with
``database is locked``. The harness names the need (``SpawnPlan.private_state_env``)
and the runtime points the variable at its own per-process scratch directory. Two
seams, asserted separately: what the harness asks for, and what the runtime does
with the ask.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kiro_crew import agent as agent_mod
from kiro_crew import sandbox as sandbox_mod
from kiro_crew.acp import client as client_mod
from kiro_crew.acp import runtime as runtime_mod
from kiro_crew.acp.codex_sqlite import seed_private_state
from kiro_crew.acp.harness import SpawnContext
from kiro_crew.acp.harness import codex as codex_mod
from kiro_crew.acp.harness import harness_for
from kiro_crew.acp.harness.base import SpawnPlan
from kiro_crew.acp.harness.codex import (
    _CODEX_HOME_ENV,
    _read_codex_mcp_server_names,
    _seed_bg_disable_mcp,
)
from kiro_crew.acp.types import ACP_BACKEND_CODEX, ACP_BACKEND_KIRO

SQLITE_HOME = "CODEX_SQLITE_HOME"


def _ctx(tmp_path: Path, environ: dict[str, str] | None = None) -> SpawnContext:
    return SpawnContext(
        agent="a", work_dir=str(tmp_path), model=None, environ=environ or {}, home=tmp_path
    )


@pytest.fixture
def codex_spawn_stubbed(monkeypatch):
    """The codex spawn's disk and sandbox work, pinned so only the plan is exercised."""
    monkeypatch.setattr(
        client_mod, "_resolve_codex_acp_bin", lambda: (["/n/node", "/p/index.js"], "/s")
    )
    monkeypatch.setattr(codex_mod, "resolve_spawn_masks", AsyncMock(return_value=((), ())))
    monkeypatch.setattr(codex_mod, "_sandbox_wrapper_generations", lambda mode: 0)


@pytest.fixture
def kiro_spawn_stubbed(monkeypatch):
    """The kiro spawn's binary search and pre-spawn gates, all answering "go"."""

    async def _bin(*, environ, home):
        return "/pinned/kiro-cli"

    monkeypatch.setattr(client_mod, "_resolve_kiro_bin_for_spawn", _bin)
    monkeypatch.setattr(agent_mod, "ensure_agent_materialized", lambda agent: None)
    monkeypatch.setattr(agent_mod, "require_fork_governance", lambda agent, work_dir: None)
    monkeypatch.setattr(
        sandbox_mod, "delegated_workspace_exposes_sealed_target", lambda work_dir: ""
    )


# ── The harness names the need ──


@pytest.mark.asyncio
async def test_codex_asks_for_a_private_sqlite_home(codex_spawn_stubbed, tmp_path):
    plan = await harness_for(ACP_BACKEND_CODEX).resolve_spawn(_ctx(tmp_path))
    assert plan.private_state_env == SQLITE_HOME
    assert plan.private_state_seed is seed_private_state


@pytest.mark.asyncio
async def test_codex_leaves_an_operator_set_sqlite_home_alone(codex_spawn_stubbed, tmp_path):
    """The operator chose that location; the variable reaches the child as set."""
    ctx = _ctx(tmp_path, environ={SQLITE_HOME: "/operator/sqlite"})
    plan = await harness_for(ACP_BACKEND_CODEX).resolve_spawn(ctx)
    assert plan.private_state_env is None


@pytest.mark.asyncio
async def test_kiro_asks_for_no_private_state(kiro_spawn_stubbed, tmp_path):
    """kiro-cli tolerates concurrent processes: the Kiro path gains nothing (H13)."""
    plan = await harness_for(ACP_BACKEND_KIRO).resolve_spawn(
        dataclasses.replace(_ctx(tmp_path), model="m")
    )
    assert plan.private_state_env is None
    assert plan.private_state_seed is None


# ── The runtime answers it with its scratch directory ──


def test_the_variable_points_at_the_scratch_dir(tmp_path):
    env: dict[str, str] = {"PATH": "/bin"}
    runtime_mod._point_private_state_at_scratch(env, SQLITE_HOME, tmp_path / "scratch")
    assert env[SQLITE_HOME] == str(tmp_path / "scratch")
    assert env["PATH"] == "/bin"


def test_a_value_already_in_the_env_is_kept(tmp_path):
    """An operator's or an ``extra_env`` value names a location they chose."""
    env = {SQLITE_HOME: "/operator/sqlite"}
    runtime_mod._point_private_state_at_scratch(env, SQLITE_HOME, tmp_path / "scratch")
    assert env[SQLITE_HOME] == "/operator/sqlite"


def test_an_empty_value_is_not_a_choice(tmp_path):
    env = {SQLITE_HOME: ""}
    runtime_mod._point_private_state_at_scratch(env, SQLITE_HOME, tmp_path / "scratch")
    assert env[SQLITE_HOME] == str(tmp_path / "scratch")


def test_a_host_that_asks_for_nothing_gets_nothing(tmp_path):
    env: dict[str, str] = {}
    runtime_mod._point_private_state_at_scratch(env, None, tmp_path / "scratch")
    assert env == {}


def test_no_scratch_sets_nothing_and_warns_once(caplog):
    """No directory to offer: the child gets the host's shared default, said once."""
    env: dict[str, str] = {}
    with caplog.at_level(logging.WARNING, logger=runtime_mod.logger.name):
        runtime_mod._point_private_state_at_scratch(env, SQLITE_HOME, None)
    assert SQLITE_HOME not in env
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert SQLITE_HOME in warnings[0].getMessage()


# ── The runtime lets the host seed that directory first ──


def _seeding_plan(calls: list, env_var: str | None = SQLITE_HOME) -> SpawnPlan:
    return SpawnPlan(
        argv=["codex"],
        private_state_env=env_var,
        private_state_seed=lambda env, scratch: calls.append((dict(env), scratch)),
    )


def test_the_seed_runs_with_the_env_and_scratch(tmp_path):
    calls: list = []
    env = {"CODEX_HOME": "/h"}
    runtime_mod._seed_private_state(env, _seeding_plan(calls), tmp_path)
    assert calls == [({"CODEX_HOME": "/h"}, tmp_path)]


@pytest.mark.parametrize(
    "env, env_var, has_scratch",
    [
        ({SQLITE_HOME: "/operator/sqlite"}, SQLITE_HOME, True),
        ({}, None, True),
        ({}, SQLITE_HOME, False),
    ],
    ids=["operator-location", "no-private-state", "no-scratch"],
)
def test_the_seed_is_skipped(tmp_path, env, env_var, has_scratch):
    calls: list = []
    scratch = tmp_path if has_scratch else None
    runtime_mod._seed_private_state(env, _seeding_plan(calls, env_var), scratch)
    assert calls == []


def test_a_plan_without_a_seed_is_a_no_op(tmp_path):
    plan = SpawnPlan(argv=["codex"], private_state_env=SQLITE_HOME)
    runtime_mod._seed_private_state({}, plan, tmp_path)


# ── Background sessions: per-server MCP disable via CODEX_CONFIG ──

_BACKGROUND = "kirocrew-lite"  # session.BACKGROUND_AGENT


def _bg_ctx(tmp_path: Path, environ: dict[str, str] | None = None) -> SpawnContext:
    return SpawnContext(
        agent=_BACKGROUND, work_dir=str(tmp_path), model=None, environ=environ or {}, home=tmp_path
    )


@pytest.mark.asyncio
async def test_background_codex_asks_for_sqlite_home_with_bg_seed(
    codex_spawn_stubbed, tmp_path
):
    """Background sessions use CODEX_SQLITE_HOME isolation and the MCP-disable seed."""
    plan = await harness_for(ACP_BACKEND_CODEX).resolve_spawn(_bg_ctx(tmp_path))
    assert plan.private_state_env == SQLITE_HOME
    assert plan.private_state_seed is _seed_bg_disable_mcp


@pytest.mark.asyncio
async def test_foreground_codex_still_uses_standard_seed(codex_spawn_stubbed, tmp_path):
    """Foreground sessions keep the standard sqlite seed; CODEX_CONFIG unchanged."""
    plan = await harness_for(ACP_BACKEND_CODEX).resolve_spawn(_ctx(tmp_path))
    assert plan.private_state_env == SQLITE_HOME
    assert plan.private_state_seed is seed_private_state


@pytest.mark.asyncio
async def test_background_codex_with_operator_sqlite_home_asks_for_nothing(
    codex_spawn_stubbed, tmp_path
):
    """Operator-set CODEX_SQLITE_HOME: private_state_env is None for both paths."""
    ctx = _bg_ctx(tmp_path, environ={SQLITE_HOME: "/operator/sqlite"})
    plan = await harness_for(ACP_BACKEND_CODEX).resolve_spawn(ctx)
    assert plan.private_state_env is None


# ── _read_codex_mcp_server_names ──

def test_read_mcp_server_names_section_headers(tmp_path):
    """Standard [mcp_servers.<name>] headers are extracted."""
    (tmp_path / "config.toml").write_text(
        "[mcp_servers.my-server]\ncommand = 'npx'\n\n[mcp_servers.another]\ncommand = 'python'\n",
        encoding="utf-8",
    )
    assert _read_codex_mcp_server_names(tmp_path / "config.toml") == ["my-server", "another"]


def test_read_mcp_server_names_dotted_key(tmp_path):
    """Dotted top-level keys mcp_servers.<name> = {...} are also extracted."""
    (tmp_path / "config.toml").write_text(
        'mcp_servers.evil2 = { command = "/bin/sh" }\n', encoding="utf-8"
    )
    names = _read_codex_mcp_server_names(tmp_path / "config.toml")
    assert "evil2" in names


def test_read_mcp_server_names_dotted_in_name(tmp_path):
    """Server names containing dots are preserved as-is."""
    (tmp_path / "config.toml").write_text(
        "[mcp_servers.my.server]\ncommand = 'npx'\n", encoding="utf-8"
    )
    assert "my.server" in _read_codex_mcp_server_names(tmp_path / "config.toml")


def test_read_mcp_server_names_empty_config(tmp_path):
    """A config with no mcp_servers returns an empty list."""
    (tmp_path / "config.toml").write_text('model = "gpt-4o"\n', encoding="utf-8")
    assert _read_codex_mcp_server_names(tmp_path / "config.toml") == []


def test_read_mcp_server_names_missing_file(tmp_path):
    """A missing config.toml returns an empty list without raising."""
    assert _read_codex_mcp_server_names(tmp_path / "config.toml") == []


# ── _seed_bg_disable_mcp ──

def test_seed_bg_sets_codex_config_with_disabled_servers(tmp_path):
    """CODEX_CONFIG is set with enabled:false for each configured server."""
    real = tmp_path / "codex_home"
    real.mkdir()
    (real / "config.toml").write_text(
        "[mcp_servers.server1]\ncommand = 'npx'\n[mcp_servers.server2]\ncommand = 'python'\n",
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    env: dict[str, str] = {_CODEX_HOME_ENV: str(real)}
    _seed_bg_disable_mcp(env, scratch)

    import json

    config = json.loads(env["CODEX_CONFIG"])
    assert config["mcp_servers"]["server1"]["enabled"] is False
    assert config["mcp_servers"]["server2"]["enabled"] is False


def test_seed_bg_no_codex_config_when_no_servers(tmp_path):
    """No CODEX_CONFIG is set when config.toml has no mcp_servers."""
    real = tmp_path / "codex_home"
    real.mkdir()
    (real / "config.toml").write_text('model = "gpt-4o"\n', encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    env: dict[str, str] = {_CODEX_HOME_ENV: str(real)}
    _seed_bg_disable_mcp(env, scratch)

    assert "CODEX_CONFIG" not in env


def test_seed_bg_no_codex_config_when_missing_file(tmp_path):
    """No CODEX_CONFIG is set when config.toml is absent."""
    real = tmp_path / "codex_home"
    real.mkdir()  # no config.toml
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    env: dict[str, str] = {_CODEX_HOME_ENV: str(real)}
    _seed_bg_disable_mcp(env, scratch)

    assert "CODEX_CONFIG" not in env


def test_seed_bg_preserves_existing_codex_config_other_keys(tmp_path):
    """Existing CODEX_CONFIG keys that are not mcp_servers are preserved."""
    real = tmp_path / "codex_home"
    real.mkdir()
    (real / "config.toml").write_text("[mcp_servers.srv]\ncommand='x'\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    import json

    existing = json.dumps({"model": "custom-model"})
    env: dict[str, str] = {_CODEX_HOME_ENV: str(real), "CODEX_CONFIG": existing}
    _seed_bg_disable_mcp(env, scratch)

    config = json.loads(env["CODEX_CONFIG"])
    assert config["model"] == "custom-model"
    assert config["mcp_servers"]["srv"]["enabled"] is False


def test_seed_bg_does_not_override_existing_mcp_servers_in_codex_config(tmp_path):
    """An existing mcp_servers key in CODEX_CONFIG is left untouched."""
    real = tmp_path / "codex_home"
    real.mkdir()
    (real / "config.toml").write_text("[mcp_servers.srv]\ncommand='x'\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    import json

    existing = json.dumps({"mcp_servers": {"custom-srv": {"enabled": True}}})
    env: dict[str, str] = {_CODEX_HOME_ENV: str(real), "CODEX_CONFIG": existing}
    _seed_bg_disable_mcp(env, scratch)

    config = json.loads(env["CODEX_CONFIG"])
    # Operator-set mcp_servers is preserved; config.toml names are NOT injected
    assert "custom-srv" in config["mcp_servers"]
    assert "srv" not in config["mcp_servers"]


def test_seed_bg_server_names_with_dots_preserved(tmp_path):
    """Server names containing dots survive JSON round-trip correctly."""
    real = tmp_path / "codex_home"
    real.mkdir()
    (real / "config.toml").write_text("[mcp_servers.my.server]\ncommand='x'\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    import json

    env: dict[str, str] = {_CODEX_HOME_ENV: str(real)}
    _seed_bg_disable_mcp(env, scratch)

    config = json.loads(env["CODEX_CONFIG"])
    # The dotted name is preserved as a JSON key, not split on dots
    assert config["mcp_servers"].get("my.server", {}).get("enabled") is False
