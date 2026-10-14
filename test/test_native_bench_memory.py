"""The RSS harness transports library paths as data, with explicit stdin opt-in."""

import importlib.util
import io
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "native" / "kc_native" / "bench_memory.py"
_LIBRARIES = ["space dir/kc_native.so", "line\nbreak.so", "库.so", "--child=native"]


@pytest.fixture
def bench(monkeypatch):
    name = "_test_native_bench_memory"
    spec = importlib.util.spec_from_file_location(name, _BENCH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "sys",
        SimpleNamespace(
            executable=sys.executable,
            stdin=SimpleNamespace(read=Mock(side_effect=AssertionError("unexpected stdin read"))),
        ),
    )
    return module


@pytest.mark.parametrize("library", _LIBRARIES)
def test_parent_keeps_library_out_of_argv(bench, library, monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [str(_BENCH), f"--library={library}", "--out", str(output)])
    monkeypatch.setattr(
        bench, "tempfile", SimpleNamespace(TemporaryDirectory=lambda **kw: nullcontext(tmp_path))
    )
    monkeypatch.setattr(
        bench,
        "random",
        SimpleNamespace(Random=lambda seed: SimpleNamespace(randbytes=lambda n: b"a")),
    )
    launches = []

    def run(cmd, **kwargs):
        assert library not in cmd
        assert cmd == [
            sys.executable,
            str(_BENCH),
            "--child",
            cmd[3],
            "--threads",
            cmd[5],
            "--blob",
            str(tmp_path / "blob.z"),
            "--library-stdin",
        ]
        assert kwargs == dict(
            input=json.dumps(str(Path(library))),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=180,
        )
        launches.append((cmd[3], int(cmd[5])))
        return SimpleNamespace(stdout='{"before": {}, "after": {}}')

    monkeypatch.setattr(bench, "subprocess", SimpleNamespace(run=run))
    bench.main()
    assert Counter(launches) == Counter(
        {(arm, threads): 18 for arm in ("python", "native") for threads in (1, 4)}
    )
    assert launches[:8] == [
        ("python", 1),
        ("native", 1),
        ("python", 4),
        ("native", 4),
        ("native", 1),
        ("python", 1),
        ("native", 4),
        ("python", 4),
    ]
    rows = json.loads(output.read_text())
    assert [(row["arm"], row["threads"]) for row in rows] == launches
    bench.sys.stdin.read.assert_not_called()


@pytest.mark.parametrize("library", _LIBRARIES)
@pytest.mark.parametrize("arm", ["python", "native"])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("stdin_mode", [False, True])
def test_child_passes_exact_path_to_loader(bench, library, arm, threads, stdin_mode, monkeypatch):
    class LoaderReached(Exception):
        pass

    loader = Mock(side_effect=LoaderReached)
    monkeypatch.setattr(
        bench, "importlib", SimpleNamespace(util=SimpleNamespace(spec_from_file_location=loader))
    )
    mode = ["--library-stdin"] if stdin_mode else [f"--library={library}"]
    monkeypatch.setattr(
        sys, "argv", [str(_BENCH), "--child", arm, "--threads", str(threads), *mode]
    )
    if stdin_mode:
        monkeypatch.setattr(bench.sys, "stdin", io.StringIO(json.dumps(library)))
    with pytest.raises(LoaderReached):
        bench.main()
    loader.assert_called_once_with("kc_native", Path(library))
    if not stdin_mode:
        bench.sys.stdin.read.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--child", "python"],
        ["--child", "native"],
        ["--library"],
        ["--child", "native", "--library"],
        ["--library-stdin"],
        ["--library=x.so", "--library-stdin"],
        ["--child", "python", "--library=x.so", "--library-stdin"],
        ["--child", "native", "--library-stdin", "--library=x.so"],
    ],
)
def test_missing_or_conflicting_modes_fail_without_reading(bench, argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(_BENCH), *argv])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
    assert "--library" in capsys.readouterr().err
    bench.sys.stdin.read.assert_not_called()
