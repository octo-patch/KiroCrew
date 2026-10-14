"""The bytes every agent-spec writer puts on disk, frozen across the materialization split.

``kiro_crew.agent`` delegates to the owners under ``kiro_crew.agent_materialization``,
and that move is only behaviour-preserving if every spec file a rebuild writes comes out
byte-for-byte as it did before, together with the audit records the rebuild emits and
the sidecar bookkeeping it leaves behind. The existing suites pin individual fields; this
module pins the whole output of one real rebuild per scenario, so a field an extraction
dropped, reordered or re-typed is a red here even when no field-level test names it.

Each scenario drives :func:`kiro_crew.agent.rebuild_agent_config` against the SHIPPED
``defaults.json``, prompts and managed-server registry, in a private agents directory,
with only the machine-specific inputs pinned: the ``kirocrew`` launcher path, the
installed kiro-cli version, and the SEL writer (recorded, not written). Everything the
rebuild writes is read back, the scratch paths are replaced by stable placeholders, and
the result is compared against a SHA-256 digest recorded before the split. A mismatch
prints the normalized content that differs, so the drift is readable from the failure.

The goldens carry POSIX paths and exec bits, so these run off Windows; the same writers
run on Windows through the field-level suites.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from kiro_crew import agent, agent_state
from kiro_crew.kiro_cli import SPEC_PERMISSIONS_MIN_VERSION

#: One path segment under a normalized root, with the separator run before it: a
#: Windows spec spells ``<TMP>\\bin\\kirocrew`` where POSIX spells ``<TMP>/bin/kirocrew``.
_UNDER_ROOT = re.compile(r"(<TMP>|<HOME>|<PKG>)((?:\\+[^\\\"\s]+)+)")
_SEPARATORS = re.compile(r"\\+")


class _SelRecorder:
    """Stands in for ``sel()``: records each audit call instead of writing it."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def log_api_access(self, **fields: Any) -> None:
        self._events.append({"api": fields})

    def log(self, event: Any) -> None:
        self._events.append(
            {
                "event": {
                    "event_type": event.event_type,
                    "operation": event.operation,
                    "outcome": event.outcome,
                    "source": event.source,
                    "resources": event.resources,
                    "error": getattr(event, "error", None),
                }
            }
        )


class _Materialized:
    """One rebuild's full output, normalized for comparison."""

    def __init__(
        self, files: dict[str, str], events: list[Any], state: str, unrefreshed: list[str]
    ) -> None:
        self.files = files
        self.events = events
        self.state = state
        self.unrefreshed = unrefreshed

    def digests(self) -> dict[str, Any]:
        return {
            "files": {name: _sha(text) for name, text in sorted(self.files.items())},
            "events": _sha(json.dumps(self.events, sort_keys=True)),
            "state": _sha(self.state),
            "unrefreshed": self.unrefreshed,
        }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Rig:
    """A private agents directory plus the pinned machine-specific inputs."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.agents = tmp_path / "agents"
        self.agents.mkdir()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        self.bin = bindir / "kirocrew"
        self.bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.bin.chmod(0o755)
        self.home = Path(os.environ["KIROCREW_HOME"])
        self.kiro_mcp = tmp_path / "kiro-global-mcp.json"
        self.hooks_dir = tmp_path / "hooks"
        self.hooks_dir.mkdir()
        self.events: list[dict[str, Any]] = []
        monkeypatch.setattr(agent, "KIRO_AGENTS_DIR", self.agents)
        monkeypatch.setattr(agent, "_KIROCREW_BIN", str(self.bin))
        monkeypatch.setattr(agent, "_KIRO_MCP_JSON", self.kiro_mcp)
        monkeypatch.setattr(agent, "_DEFAULT_KIRO_HOOKS_DIR", self.hooks_dir)
        monkeypatch.setattr(agent, "sel", lambda: _SelRecorder(self.events))
        monkeypatch.setattr(
            "kiro_crew.apps.bridges._mcp_json_path", lambda: self.agents / "kirocrew.json"
        )
        monkeypatch.setattr(
            "kiro_crew.kiro_cli.installed_kiro_cli_version",
            lambda: SPEC_PERMISSIONS_MIN_VERSION,
        )

    def executable(self, name: str) -> Path:
        path = self.tmp / "bin" / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def config(self, data: dict[str, Any]) -> None:
        self.write_json(self.home / "config.json", data)

    def normalize(self, text: str) -> str:
        """Replace this run's scratch and home roots with labels, in every spelling.

        A root is spelled as-is, JSON-escaped once (inside a spec file) or twice
        (inside a JSON value an event records). A path under a root then keeps the
        host's separator, so it is folded to ``/``: the goldens are the same bytes
        on every platform. ``<PKG>`` is the installed ``kiro_crew`` package, which
        the assistant prompt names as the packaged docs index: it is wherever this
        checkout lives, so it is labelled like the scratch roots.
        """
        package = Path(agent.__file__).resolve().parent
        bases = {
            self.tmp: "<TMP>",
            self.tmp.resolve(): "<TMP>",
            self.home: "<HOME>",
            self.home.resolve(): "<HOME>",
            package: "<PKG>",
        }
        # A spec may also write a root forward-slashed (``Path.as_posix()``, as a
        # ``skill://`` resource does), which on Windows differs from ``str()``.
        roots = {str(p): label for p, label in bases.items()}
        roots.update({p.as_posix(): label for p, label in bases.items()})
        spellings: dict[str, str] = {}
        for root, label in roots.items():
            once = json.dumps(root)[1:-1]
            for spelled in (root, once, json.dumps(once)[1:-1]):
                spellings[spelled] = label
        for spelling in sorted(spellings, key=len, reverse=True):
            text = text.replace(spelling, spellings[spelling])
        return _UNDER_ROOT.sub(lambda m: m.group(1) + _SEPARATORS.sub("/", m.group(2)), text)

    def snapshot(self) -> _Materialized:
        files = {
            p.name: self.normalize(p.read_text(encoding="utf-8"))
            for p in sorted(self.agents.iterdir())
            if p.is_file() and not p.name.startswith(".")
        }
        state_path = agent_state._state_path()
        state = state_path.read_text(encoding="utf-8") if state_path.is_file() else ""
        if state:
            # The worker's mirror bookkeeping records the default spec's file identity
            # and content fingerprint, both of which carry this run's scratch paths. What
            # is contractual is that they describe the default spec now on disk.
            parsed = json.loads(state)
            # The dashboard-author ownership digest is the SHA-256 of the installed spec's
            # bytes, which carry this run's scratch paths (the kirocrew-core launcher). What
            # is contractual is that it equals the digest of the dashboard-author spec now on
            # disk -- so fold it to a stable label when it does, exactly as the worker's
            # mirror bookkeeping below is folded.
            da_file = self.agents / "kirocrew-dashboard-author.json"
            da_digest = None
            if da_file.is_file():
                da_digest = agent_state.spec_digest(json.loads(da_file.read_text(encoding="utf-8")))
            for entry in parsed.values():
                if not isinstance(entry, dict):
                    continue
                if da_digest is not None and entry.get("managed_digest") == da_digest:
                    entry["managed_digest"] = "<DASHBOARD-AUTHOR-DIGEST>"
                if entry.get("mirrored_stat") == agent.default_spec_identity():
                    entry["mirrored_stat"] = "<DEFAULT-SPEC-IDENTITY>"
                if entry.get("mirrored_from") == agent.default_spec_fingerprint():
                    entry["mirrored_from"] = "<DEFAULT-SPEC-FINGERPRINT>"
            state = json.dumps(parsed, indent=2, sort_keys=True)
        events = json.loads(self.normalize(json.dumps(self.events, sort_keys=True, default=str)))
        unrefreshed = sorted(agent._fork_refresh_failed)
        return _Materialized(files, events, self.normalize(state), unrefreshed)


# ── scenarios ────────────────────────────────────────────────────────────────


def _fresh(rig: _Rig) -> dict[str, Any]:
    """A first install: no spec on disk, no MCP sources, no user config."""
    return {}


def _customized(rig: _Rig) -> dict[str, Any]:
    """An existing spec a user has customized, with every MCP source populated."""
    tool = rig.executable("some-mcp")
    rig.write_json(
        rig.agents / "kirocrew.json",
        {
            "name": "kirocrew",
            "description": "customized",
            "model": "claude-opus-4.6-1m",
            "prompt": "file:///somewhere/else/prompt.md",
            "tools": ["fs_read", "@kirocrew-cron", "@kirocrew-core", "@user-srv", "@gone/tool"],
            "allowedTools": ["fs_read", "@kirocrew-core", "@user-srv/do_it", "@gone/tool"],
            "resources": [],
            "toolsSettings": {
                "execute_bash": {
                    "deniedCommands": ["rm -rf /"],
                    "autoAllowReadonly": True,
                    "allowedCommands": ["ls"],
                },
                "subagent": {
                    "availableAgents": ["kirocrew-worker", "review-*"],
                    "trustedAgents": ["kirocrew-worker"],
                },
                "fs_write": {"allowedPaths": ["~/work"]},
            },
            "mcpServers": {
                "kirocrew-cron": {
                    "command": "/stale/kirocrew",
                    "args": ["mcp-cron"],
                    "timeout": 90000,
                    "url": "http://stale",
                    "env": {"FOO": "bar", "HOME": "/elsewhere", "PATH": "/x"},
                    "autoApprove": ["cron_list"],
                },
                "user-srv": {"command": str(tool), "args": ["--serve"], "disabledTools": ["x"]},
            },
            "hooks": {"preToolUse": [{"command": "/bin/true"}]},
            "unknownTopLevel": {"kept": True},
        },
    )
    rig.write_json(
        rig.kiro_mcp,
        {
            "mcpServers": {
                "global-srv": {"command": str(tool), "args": ["g"], "timeout": 5},
                "npm:@scope/pkg": {"command": str(tool), "args": ["scoped"]},
                "missing-bin": {"command": "definitely-not-on-path-b08", "args": []},
                "no-command": {"args": ["x"]},
                "muted-srv": {"command": str(tool), "disabled": True},
                "remote-srv": {
                    "url": "https://mcp.example.test/mcp",
                    "oauth": {"scopes": ["read"], "clientId": "cid"},
                },
            }
        },
    )
    rig.write_json(
        rig.home / "mcp.json",
        {
            "mcpServers": {
                "store-srv": {"command": str(tool), "args": ["store"], "env": {"A": "1"}},
                "global-srv": {"env": {"B": "2"}},
            }
        },
    )
    rig.write_json(rig.home / "agent.json", {"toolsSettings": {"custom_tool": {"k": "v"}}})
    return {}


def _governed(rig: _Rig) -> dict[str, Any]:
    """The customized install under a ceiling that denies some auto-approvals."""
    _customized(rig)
    return {
        "may_auto_approve": lambda ref: ref
        not in {"fs_read", "@kirocrew-core", "@global-srv", "@kirocrew-core/select_crew"}
    }


def _clean_over_customized(rig: _Rig) -> dict[str, Any]:
    """A ``--clean`` rebuild over the customized install."""
    _customized(rig)
    return {"clean": True}


def _user_hooks(rig: _Rig) -> dict[str, Any]:
    """Explicit hooks in both spec shapes plus an autoimported script."""
    guard = rig.executable("guard.sh")
    script = rig.hooks_dir / "audit-post.sh"
    script.write_text("#!/bin/sh\n# matcher: fs_write\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    off = rig.hooks_dir / "off-pre.sh"
    off.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    off.chmod(0o755)
    rig.config(
        {
            "agent": {
                "kiro_hooks": [
                    {
                        "name": "guard",
                        "trigger": "PreToolUse",
                        "matcher": "execute_bash",
                        "action": {"type": "command", "command": str(guard)},
                    },
                    {
                        "trigger": "PostFileSave",
                        "action": {"type": "command", "command": str(guard)},
                    },
                    {
                        "trigger": "Stop",
                        "enabled": False,
                        "action": {"type": "command", "command": str(off)},
                    },
                    {"trigger": "nope", "action": {"type": "command", "command": "x"}},
                ],
                "kiro_hooks_autoimport": True,
            }
        }
    )
    return {}


def _object_hooks(rig: _Rig) -> dict[str, Any]:
    """The object-of-arrays hook shape, with the rejections it audits."""
    guard = rig.executable("guard2.sh")
    rig.config(
        {
            "agent": {
                "kiro_hooks": {
                    "preToolUse": [
                        {"command": str(guard), "matcher": "fs_*"},
                        {"command": str(guard), "matcher": "fs_*"},
                        {"command": "relative.sh"},
                        {"matcher": "x"},
                    ],
                    "fileEdited": [{"command": str(guard)}],
                    "bogusEvent": [{"command": str(guard)}],
                    "stop": "not-a-list",
                },
                "kiro_hooks_autoimport": False,
            }
        }
    )
    return {}


def _registry_mode(rig: _Rig) -> dict[str, Any]:
    """An install the operator declared registry-governed."""
    rig.config({"agent": {"mcp_registry_mode": True, "model": "claude-sonnet-4.5"}})
    return {}


def _forks(rig: _Rig) -> dict[str, Any]:
    """Two private template copies: one corroborated by a crew binding, one orphaned."""
    from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig

    cfg = KiroCrewConfig()
    cfg.agents = {"my-crew": KiroCrewAgentConfig(kiro_agent="my-crew")}
    cfg.save()
    for name in ("my-crew", "orphan-crew"):
        rig.write_json(
            rig.agents / f"{name}.json",
            {
                "name": name,
                "prompt": "file:///old-home/.kiro/crew/prompt.md",
                "tools": ["fs_read", "@kirocrew-core"],
                "allowedTools": ["fs_read", "@kirocrew-core", 7],
                "toolsSettings": {
                    "execute_bash": {"deniedCommands": ["rm"]},
                    "subagent": {"availableAgents": "not-a-list"},
                },
                "mcpServers": {"kirocrew-core": {"command": "/old", "autoApprove": ["x"]}},
                "hooks": {"old": "hook"},
            },
        )
        agent_state.set_fork_info(name, forked_from="kirocrew", private_to=name)
    return {"may_auto_approve": lambda ref: ref != "@kirocrew-core"}


SCENARIOS: dict[str, Callable[[_Rig], dict[str, Any]]] = {
    "fresh": _fresh,
    "customized": _customized,
    "governed": _governed,
    "clean_over_customized": _clean_over_customized,
    "user_hooks": _user_hooks,
    "object_hooks": _object_hooks,
    "registry_mode": _registry_mode,
    "forks": _forks,
}


def materialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str) -> _Materialized:
    """Run one scenario's rebuild in a private rig and return its normalized output."""
    rig = _Rig(tmp_path, monkeypatch)
    options = SCENARIOS[scenario](rig)
    if "may_auto_approve" in options:
        monkeypatch.setattr(agent, "_may_auto_approve", options["may_auto_approve"])
    agent.rebuild_agent_config(clean=options.get("clean", False))
    return rig.snapshot()


#: Digests recorded from the pre-split ``kiro_crew.agent``. See the module docstring.
GOLDEN: dict[str, dict[str, Any]] = {
    "clean_over_customized": {
        "events": "5ed66a55abd0152e3532cd31239cb5732cb171f58a57dc249138978a572ebe22",
        "files": {
            "kirocrew-conductor.json": "ee61435c8ba20be2657b4a891b4f46662e4f84281a83b88702077c98492eabab",
            "kirocrew-dashboard-author.json": "cbe7acbded8bd0255c466d21fd8e990014452f9b3cd4afb9b74d5ee6354b3366",
            "kirocrew-dashboard-manager.json": "4a58ddb1be44ec985dcd81001e85b52f7a2f1845923dcf6777a65f6b940755d4",
            "kirocrew-guest.json": "3ac87f33f4968a07c6f3dc45b29922d7002d38e93caabefa5005a7a1794fce38",
            "kirocrew-heartbeat.json": "7a212fc757669b2be5d1b141d58bac4bafc3dd9e19e206a68994f20f4c4f6ed4",
            "kirocrew-knowledge.json": "efcde26b5961417a7c9ed665ee20038461b5283b74099423cb8ee474400335f5",
            "kirocrew-ledger-conductor.json": "2732fbe5454ef51e918d56c7cc17881189701d00f83be1d881540a0317572316",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "9aa87b292612bc6ab53a62a5b0885ff56857eb1c3e3b7a34ae1a7c57942d4287",
            "kirocrew-research.json": "e6f3ae6eaa577c909b7d28a9d3637efb8e75548312641b497d4c50418c9130b1",
            "kirocrew-security-conductor.json": "c34378da19d5b5bddd8509347237821b4c71260db425c2a5397af913b5b7033c",
            "kirocrew-team-lead.json": "30d98937a92444a308f281d31f9e69ded5384ddf6e3e17c2c573b21b933dbcc4",
            "kirocrew-worker.json": "5c824440e50f511151df7a3ed2b287ca551a94e7f0973b48794c7d8be7aed77a",
            "kirocrew.json": "0545d2bc893070765f73d54d303c653454036b40472b357195e3c7800e0fc47f",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "state": "d704db8fc2311b53ecec035ace4d3145224e00ab26bbedeb0a2e6977a7c16aac",
        "unrefreshed": [],
    },
    "customized": {
        "events": "d3b12ca2886a167d45036c2cbbdceedef3d9e0773a50e7dcb49e534b327f38a3",
        "files": {
            "kirocrew-conductor.json": "ee61435c8ba20be2657b4a891b4f46662e4f84281a83b88702077c98492eabab",
            "kirocrew-dashboard-author.json": "cbe7acbded8bd0255c466d21fd8e990014452f9b3cd4afb9b74d5ee6354b3366",
            "kirocrew-dashboard-manager.json": "4a58ddb1be44ec985dcd81001e85b52f7a2f1845923dcf6777a65f6b940755d4",
            "kirocrew-guest.json": "3ac87f33f4968a07c6f3dc45b29922d7002d38e93caabefa5005a7a1794fce38",
            "kirocrew-heartbeat.json": "7a212fc757669b2be5d1b141d58bac4bafc3dd9e19e206a68994f20f4c4f6ed4",
            "kirocrew-knowledge.json": "efcde26b5961417a7c9ed665ee20038461b5283b74099423cb8ee474400335f5",
            "kirocrew-ledger-conductor.json": "2732fbe5454ef51e918d56c7cc17881189701d00f83be1d881540a0317572316",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "9aa87b292612bc6ab53a62a5b0885ff56857eb1c3e3b7a34ae1a7c57942d4287",
            "kirocrew-research.json": "e6f3ae6eaa577c909b7d28a9d3637efb8e75548312641b497d4c50418c9130b1",
            "kirocrew-security-conductor.json": "c34378da19d5b5bddd8509347237821b4c71260db425c2a5397af913b5b7033c",
            "kirocrew-team-lead.json": "30d98937a92444a308f281d31f9e69ded5384ddf6e3e17c2c573b21b933dbcc4",
            "kirocrew-worker.json": "e27321e54e679f2c719a0f4e415ecf69bfe15d134eac4aa3ca5b49c9f7391913",
            "kirocrew.json": "af6f3e40088950ae733c6139e7cd1180829cb841b94f9c86517fd03150803ab2",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "state": "939d5770910a05fc49ac6fff3da064f2ff08347839f39e1eadebc5fa964eb6b5",
        "unrefreshed": [],
    },
    "forks": {
        "events": "7a9e490c9a63e01a04e66830418c74f0f9c5eaebc810788018028983651c2057",
        "files": {
            "kirocrew-conductor.json": "9fe578f5e4261a781b47d213696e90a53bc890161619a6d1a0b9a9e15217e7ae",
            "kirocrew-dashboard-author.json": "359ad2ebfe08f8a558f0323f5eabf2a734d11ca4ed27c2bcfd7c2a49de328941",
            "kirocrew-dashboard-manager.json": "2c9c153b17e53f27da156c4b4cb2df19c9dfd8fe0e4c07076db786680038bd9f",
            "kirocrew-guest.json": "3ac87f33f4968a07c6f3dc45b29922d7002d38e93caabefa5005a7a1794fce38",
            "kirocrew-heartbeat.json": "7a212fc757669b2be5d1b141d58bac4bafc3dd9e19e206a68994f20f4c4f6ed4",
            "kirocrew-knowledge.json": "efcde26b5961417a7c9ed665ee20038461b5283b74099423cb8ee474400335f5",
            "kirocrew-ledger-conductor.json": "f91c1c4ed58942631585c05a9618a79ba3265076e29ad8951e6c6e2b71ea1927",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "259a08a152e0ead5017ec9e9ab3fee2c862fde14c64a45b13f2f4119562ded86",
            "kirocrew-research.json": "226891438f47950133f9fde98d5d7877321a2af2072ea47a4183337dc15d471b",
            "kirocrew-security-conductor.json": "bf89752990cd1655f9b4ec5487d7c98c131e5667ef79a388217f32f370a0ab51",
            "kirocrew-team-lead.json": "95bd94e7a99e17f33957008f5ee057c6495a1dcf15042d9d26522504a990913c",
            "kirocrew-worker.json": "195459eebf3af73eeb3acfc49cd4776d94770daf6ee31c6037b488e0f797eb87",
            "kirocrew.json": "51a88e33b99ba23a454f5f7274a1a89dbd10a4902fa69c8d68f2200e3d34611a",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "my-crew.json": "a041707d1f911fd7eaae2ca2afbe4304e67d03bb66dc5a3cabd65c8c55538309",
            "orphan-crew.json": "30c576d8c4eb514bdbb5139402df6588504cc92cfef8b580ec2e16bc98f74056",
        },
        "state": "cbb4cfdf2675dd8efeed9a8b77fc63ce287172f38fe4db9fad37ecb6654ed485",
        "unrefreshed": ["orphan-crew"],
    },
    "fresh": {
        "events": "256cc2390d115b4cdad4591c48f143cf649e553f333d0c515352200f1396a25b",
        "files": {
            "kirocrew-conductor.json": "9fe578f5e4261a781b47d213696e90a53bc890161619a6d1a0b9a9e15217e7ae",
            "kirocrew-dashboard-author.json": "359ad2ebfe08f8a558f0323f5eabf2a734d11ca4ed27c2bcfd7c2a49de328941",
            "kirocrew-dashboard-manager.json": "4a58ddb1be44ec985dcd81001e85b52f7a2f1845923dcf6777a65f6b940755d4",
            "kirocrew-guest.json": "3ac87f33f4968a07c6f3dc45b29922d7002d38e93caabefa5005a7a1794fce38",
            "kirocrew-heartbeat.json": "7a212fc757669b2be5d1b141d58bac4bafc3dd9e19e206a68994f20f4c4f6ed4",
            "kirocrew-knowledge.json": "efcde26b5961417a7c9ed665ee20038461b5283b74099423cb8ee474400335f5",
            "kirocrew-ledger-conductor.json": "f91c1c4ed58942631585c05a9618a79ba3265076e29ad8951e6c6e2b71ea1927",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "259a08a152e0ead5017ec9e9ab3fee2c862fde14c64a45b13f2f4119562ded86",
            "kirocrew-research.json": "f4ad1ea7fce492420f5e93905ff38ee0ef97cf362089e56aa8a51862c6266921",
            "kirocrew-security-conductor.json": "bf89752990cd1655f9b4ec5487d7c98c131e5667ef79a388217f32f370a0ab51",
            "kirocrew-team-lead.json": "d107737e48f7fe6cea45183834193d38e2b4ea4c06644710cb140e5ff342d25d",
            "kirocrew-worker.json": "dece852776a382a478d0ef8740e6898e81485280c65ab6148188ae6e0fbc67fc",
            "kirocrew.json": "2a9d8e606266cd13145fc340ea4df6cd0effa6759522484e03f24b4a5164d48a",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "state": "d704db8fc2311b53ecec035ace4d3145224e00ab26bbedeb0a2e6977a7c16aac",
        "unrefreshed": [],
    },
    "governed": {
        "events": "3b6d82386a6155f3647159e3a4045aa8af0e6bb7ac78acbdacc3470edbcce1f5",
        "files": {
            "kirocrew-conductor.json": "5042600ab9161e752af8b6a1fc9e03fcdcc7a295c0c16da34a1a653c5d7cd40e",
            "kirocrew-dashboard-author.json": "cbe7acbded8bd0255c466d21fd8e990014452f9b3cd4afb9b74d5ee6354b3366",
            "kirocrew-dashboard-manager.json": "4a58ddb1be44ec985dcd81001e85b52f7a2f1845923dcf6777a65f6b940755d4",
            "kirocrew-guest.json": "3ac87f33f4968a07c6f3dc45b29922d7002d38e93caabefa5005a7a1794fce38",
            "kirocrew-heartbeat.json": "7a212fc757669b2be5d1b141d58bac4bafc3dd9e19e206a68994f20f4c4f6ed4",
            "kirocrew-knowledge.json": "efcde26b5961417a7c9ed665ee20038461b5283b74099423cb8ee474400335f5",
            "kirocrew-ledger-conductor.json": "49c6b37558712cc070bde29dcf4117b3f87ab70f2a0eefee78cdf07831201a77",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "9aa87b292612bc6ab53a62a5b0885ff56857eb1c3e3b7a34ae1a7c57942d4287",
            "kirocrew-research.json": "0cb7038ce95c848b4fd6625a85cf9eccd0ef87224ce2f8ad65c0389990b5c7bb",
            "kirocrew-security-conductor.json": "c34378da19d5b5bddd8509347237821b4c71260db425c2a5397af913b5b7033c",
            "kirocrew-team-lead.json": "7ea1cc3c67180f247dcca0500f976a027ed345353241d569bc9748609b314912",
            "kirocrew-worker.json": "76646b8dea3161c731f21bb6c875c93586ef33cfaa168e14da089280c45247f1",
            "kirocrew.json": "ea820579bd16a99cd4402d92f284803c24ede1a66609d5a7fa33ce0d0b7fd332",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "state": "939d5770910a05fc49ac6fff3da064f2ff08347839f39e1eadebc5fa964eb6b5",
        "unrefreshed": [],
    },
    "object_hooks": {
        "events": "6a9c5563feb9cbc42dd4056b60626a35fc577b0ae982b3739731f2cb2e5ff48f",
        "files": {
            "kirocrew-conductor.json": "715c1d248ff12ea61bbf6b60f4b9a2540a972d747a1bcfa6c99f707f9e9b5f1a",
            "kirocrew-dashboard-author.json": "816d5e5dd9988c2122f0c2363110deeb3f6384a41400e478ee617e9dfbe4fef3",
            "kirocrew-dashboard-manager.json": "4a58ddb1be44ec985dcd81001e85b52f7a2f1845923dcf6777a65f6b940755d4",
            "kirocrew-guest.json": "3ac87f33f4968a07c6f3dc45b29922d7002d38e93caabefa5005a7a1794fce38",
            "kirocrew-heartbeat.json": "7a212fc757669b2be5d1b141d58bac4bafc3dd9e19e206a68994f20f4c4f6ed4",
            "kirocrew-knowledge.json": "efcde26b5961417a7c9ed665ee20038461b5283b74099423cb8ee474400335f5",
            "kirocrew-ledger-conductor.json": "023aad61467efb27b62f07ca6a4ddc606bec85b588736fe41da0c52205f9a91c",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "8f993db1e97034c09237e656f2469c97bcf5dfb86faa94d039f385b9dc4f0ab8",
            "kirocrew-research.json": "7666a438bbdfc5d547687d5b473d04471aaddf9757194e6227669d9e86182980",
            "kirocrew-security-conductor.json": "45567a2b9cc93fd50bc75bca810eaefc8088802ce3095b852615f5dd17d448d2",
            "kirocrew-team-lead.json": "52834fcbe49cc86c1b969eb4323ad752ab4907332f08dc2829345a33438ce390",
            "kirocrew-worker.json": "a7dacb132fc875a34a6eddd8e0a05b9dfe134b02ce72cf279cc8b7a987d52126",
            "kirocrew.json": "d06402ad60ce55f8e9d161e2b73608c3548c876798e309298854c68311c7b984",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "state": "d704db8fc2311b53ecec035ace4d3145224e00ab26bbedeb0a2e6977a7c16aac",
        "unrefreshed": [],
    },
    "registry_mode": {
        "events": "256cc2390d115b4cdad4591c48f143cf649e553f333d0c515352200f1396a25b",
        "files": {
            "kirocrew-conductor.json": "8a510fbbcd72f87d0e0361588124d0130475d5d0652fee6955aabce4102cce83",
            "kirocrew-dashboard-author.json": "d44d7f9bbb91f12cd6107609ca7c80b7bf81d6656bf884192e6c97241d39821d",
            "kirocrew-dashboard-manager.json": "3d2b380b27b82d7bbb1555ce698248fffe27541da0dc832a968fd225fac2809d",
            "kirocrew-guest.json": "2423a7b447fbcedec2a64ab54a89d181cb2357456c8ddcfc189dc2afe3525780",
            "kirocrew-heartbeat.json": "6dbd5042238c4b0565f250dd4e235f0f77b01f7d7e6091a127a29ec25e183cc3",
            "kirocrew-knowledge.json": "5275c0f70b6b42581c9c9841a572c16673b3a5ede1317936f4d4d870e2a883a0",
            "kirocrew-ledger-conductor.json": "053206841474a0d0e6765329133eaf93196016acdd8903ff3a0af49645f3eb7f",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "15b7e86caacf64e851bf4d00b97df22f52d47dfc2fe5ea3718695cca763b805d",
            "kirocrew-research.json": "4f6418d8353ac284af564ffe464e7e4852f0af510ba05c424387a7f0f6b9d961",
            "kirocrew-security-conductor.json": "28a7df9a998ded3541af9f2f3f7ebd265cba8485be5a2c32d44d213012b82e5a",
            "kirocrew-team-lead.json": "0f3c4c977b9cb9a23f32ae633c37aa246f3a332894c98974417684ce5fed1bff",
            "kirocrew-worker.json": "7e31f09295908481f417580988969e9c3cc680f87afe48b0bbcc77d8783bd967",
            "kirocrew.json": "12e07680a871ddd02394c6e6699840774c9d3165379d5bd8a8ffd792a2c69025",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "state": "d704db8fc2311b53ecec035ace4d3145224e00ab26bbedeb0a2e6977a7c16aac",
        "unrefreshed": [],
    },
    "user_hooks": {
        "events": "77662b39a7247eafdf1d85553b71b74003d9d00ed27103c8b1132ac919684462",
        "files": {
            "kirocrew-conductor.json": "b617788add291023460a9db36f2468cd97fc95654c7ba44d09dc3ed65a7789e7",
            "kirocrew-dashboard-author.json": "d867f2989cb8864b75c88ebfb052c6131ebe1ab123a9020329273edb296cd499",
            "kirocrew-dashboard-manager.json": "4a58ddb1be44ec985dcd81001e85b52f7a2f1845923dcf6777a65f6b940755d4",
            "kirocrew-guest.json": "3ac87f33f4968a07c6f3dc45b29922d7002d38e93caabefa5005a7a1794fce38",
            "kirocrew-heartbeat.json": "7a212fc757669b2be5d1b141d58bac4bafc3dd9e19e206a68994f20f4c4f6ed4",
            "kirocrew-knowledge.json": "efcde26b5961417a7c9ed665ee20038461b5283b74099423cb8ee474400335f5",
            "kirocrew-ledger-conductor.json": "f85032d1c8ea31b5ca73946e9ee37f22fc53f86894615ec07341915ad885dff5",
            "kirocrew-lite.json": "c24c577f0102ce20ccdd15f28e045888356e3a36db3e5646d86f02f62ca38612",
            "kirocrew-pipeline-conductor.json": "d14e919951362329ef1d5960a402b30037ee5d95dfe5db604da5a4c047b21fbc",
            "kirocrew-research.json": "7db3afde79a24d50e572046bc40531b845cfa17972dc2e0c408dc3be648f769a",
            "kirocrew-security-conductor.json": "fb87477b31545ea61cdad0ec04e3d5de4924e5abbce3609b53e2f77cb79c40d2",
            "kirocrew-team-lead.json": "524b6ecc3d9c8c0ed59e0e34be281e9bedf385f09de19f3868c7e2663496f08b",
            "kirocrew-worker.json": "1cd3d559b82a7a8f0f2600527bb23e33b877ddd0a5e3fe703179930656c6422b",
            "kirocrew.json": "84b13649c37e5c140cc04e09f29c7305a2b935356ceff07630e144f6898f5876",
            "kirocrew.lock": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "state": "d704db8fc2311b53ecec035ace4d3145224e00ab26bbedeb0a2e6977a7c16aac",
        "unrefreshed": [],
    },
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_written_spec_matches_the_pre_split_bytes(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = materialize(tmp_path, monkeypatch, scenario)
    expected = GOLDEN[scenario]
    digests = got.digests()
    assert sorted(digests["files"]) == sorted(expected["files"]), "a spec file appeared or vanished"
    # The goldens are CI's bytes: a developer host resolves different binaries and
    # gates, so they cannot be regenerated locally. A mismatch therefore prints
    # every digest this run produced, so one CI round is enough to re-pin a
    # deliberate change.
    regen = f"this run's digests for {scenario!r}:\n" + json.dumps(
        digests, indent=1, sort_keys=True
    )
    for name, digest in expected["files"].items():
        assert (
            digests["files"][name] == digest
        ), f"{scenario}: {name} no longer matches the pre-split bytes:\n{got.files[name]}\n{regen}"
    assert digests["events"] == expected["events"], (
        f"{scenario}: the audit record sequence changed:\n"
        + json.dumps(got.events, indent=1, sort_keys=True)
        + f"\n{regen}"
    )
    assert (
        digests["state"] == expected["state"]
    ), f"{scenario}: the agent-state sidecar changed:\n{got.state}\n{regen}"
    assert digests["unrefreshed"] == expected["unrefreshed"], "the fork refresh verdict changed"
