from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import experiments.ctgan_orchestrator_tmux as tmux_mod


def test_default_session_name_uses_hostname_and_index(monkeypatch):
    monkeypatch.setattr(tmux_mod.socket, "gethostname", lambda: "node-a.local")

    session = tmux_mod.build_session_name(index=2)

    assert session == "ctgan-orch-node-a-02"


def test_log_and_metadata_paths_live_under_orchestrator_logs(tmp_path):
    log_path = tmux_mod.log_path_for_session(
        output_root=tmp_path / "results",
        session_name="ctgan-orch-node-a-02",
    )
    meta_path = tmux_mod.metadata_path_for_session(
        output_root=tmp_path / "results",
        session_name="ctgan-orch-node-a-02",
    )

    assert log_path == tmp_path / "results" / "orchestrator_logs" / "ctgan-orch-node-a-02.log"
    assert meta_path == tmp_path / "results" / "orchestrator_logs" / "ctgan-orch-node-a-02.json"


def test_build_worker_argv_preserves_orchestrator_contract():
    argv = tmux_mod.build_worker_argv(
        manifest_path=Path("/repo/experiments/ctgan_orchestrator_manifest.json"),
        worksheet="CTGAN",
        output_root=Path("/repo/experiments/results"),
        heartbeat_seconds=90,
    )

    assert argv == [
        tmux_mod.sys.executable,
        "experiments/ctgan_orchestrator.py",
        "--manifest",
        "/repo/experiments/ctgan_orchestrator_manifest.json",
        "--worksheet",
        "CTGAN",
        "--output-root",
        "/repo/experiments/results",
        "--heartbeat-seconds",
        "90",
    ]


def test_build_tmux_launch_command_wraps_worker_with_tee(tmp_path):
    command = tmux_mod.build_tmux_launch_command(
        session_name="ctgan-orch-node-a-02",
        worker_argv=["python", "experiments/ctgan_orchestrator.py", "--worksheet", "CTGAN"],
        log_path=tmp_path / "results" / "orchestrator_logs" / "ctgan-orch-node-a-02.log",
        cwd=tmp_path,
    )

    assert command[:5] == ["tmux", "new-session", "-d", "-s", "ctgan-orch-node-a-02"]
    assert "tee -a" in command[-1]
    assert "experiments/ctgan_orchestrator.py" in command[-1]


def test_launch_fails_when_tmux_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(tmux_mod.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="tmux"):
        tmux_mod.launch_session(
            manifest_path=tmp_path / "manifest.json",
            worksheet="CTGAN",
            output_root=tmp_path / "results",
            index=1,
        )


def test_launch_invokes_tmux_and_returns_session_info(monkeypatch, tmp_path):
    calls: list[tuple[list[str], bool, Path | None, bool, bool]] = []
    monkeypatch.setattr(tmux_mod.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(tmux_mod.socket, "gethostname", lambda: "node-a.local")

    def fake_run(argv, *, check=False, cwd=None, capture_output=False, text=False):
        calls.append((list(argv), check, cwd, capture_output, text))
        if argv[:2] == ["tmux", "has-session"]:
            return type("Result", (), {"stdout": "", "stderr": "", "returncode": 1})()
        return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(tmux_mod.subprocess, "run", fake_run)

    result = tmux_mod.launch_session(
        manifest_path=tmp_path / "manifest.json",
        worksheet="CTGAN",
        output_root=tmp_path / "results",
        index=1,
        heartbeat_seconds=75,
    )

    assert result.session_name == "ctgan-orch-node-a-01"
    assert result.log_path == tmp_path / "results" / "orchestrator_logs" / "ctgan-orch-node-a-01.log"
    assert result.metadata_path == tmp_path / "results" / "orchestrator_logs" / "ctgan-orch-node-a-01.json"
    assert list(result.attach_command) == ["tmux", "attach", "-t", "ctgan-orch-node-a-01"]
    assert calls[0][0] == ["tmux", "has-session", "-t", "ctgan-orch-node-a-01"]
    assert calls[1][0][:5] == ["tmux", "new-session", "-d", "-s", "ctgan-orch-node-a-01"]
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["session"] == "ctgan-orch-node-a-01"
    assert payload["worksheet"] == "CTGAN"
    assert payload["argv"][-1] == "75"


def test_attach_command_invokes_tmux_attach(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(tmux_mod.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(
        tmux_mod.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(list(argv)) or type("Result", (), {"stdout": "", "stderr": "", "returncode": 0})(),
    )

    tmux_mod.attach_session("ctgan-orch-node-a-01")

    assert calls == [["tmux", "attach", "-t", "ctgan-orch-node-a-01"]]


def test_list_sessions_reads_tmux_ls_and_filters_ctgan_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(tmux_mod.shutil, "which", lambda name: "/usr/bin/tmux")
    metadata_path = tmux_mod.metadata_path_for_session(
        output_root=tmp_path / "results",
        session_name="ctgan-orch-node-a-01",
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "session": "ctgan-orch-node-a-01",
                "hostname": "node-a",
                "started_at": "2026-04-02T03:00:00Z",
                "argv": ["python", "experiments/ctgan_orchestrator.py"],
                "log_path": str(
                    tmux_mod.log_path_for_session(
                        output_root=tmp_path / "results",
                        session_name="ctgan-orch-node-a-01",
                    )
                ),
                "manifest": "/repo/experiments/ctgan_orchestrator_manifest.json",
                "worksheet": "CTGAN",
            }
        ),
        encoding="utf-8",
    )

    def fake_run(argv, *, check=False, cwd=None, capture_output=False, text=False):
        assert argv == ["tmux", "ls"]
        return type(
            "Result",
            (),
            {
                "stdout": "ctgan-orch-node-a-01: 1 windows (created Thu Apr  2 03:00:00 2026)\nother-session: 1 windows\n",
                "stderr": "",
                "returncode": 0,
            },
        )()

    monkeypatch.setattr(tmux_mod.subprocess, "run", fake_run)

    sessions = tmux_mod.list_sessions(output_root=tmp_path / "results")

    assert len(sessions) == 1
    assert sessions[0].session_name == "ctgan-orch-node-a-01"
    assert sessions[0].worksheet == "CTGAN"


def test_tail_uses_metadata_to_find_log_file(tmp_path):
    log_path = tmux_mod.log_path_for_session(
        output_root=tmp_path / "results",
        session_name="ctgan-orch-node-a-01",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    metadata_path = tmux_mod.metadata_path_for_session(
        output_root=tmp_path / "results",
        session_name="ctgan-orch-node-a-01",
    )
    metadata_path.write_text(
        json.dumps({"log_path": str(log_path)}),
        encoding="utf-8",
    )

    tailed = tmux_mod.tail_log(
        session_name="ctgan-orch-node-a-01",
        output_root=tmp_path / "results",
        lines=2,
    )

    assert tailed == "line2\nline3"


def test_tmux_launcher_help_runs_as_script():
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "experiments/ctgan_orchestrator_tmux.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
