import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import experiments.ctgan_orchestrator as orchestrator_mod


def write_orchestrator_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "label": "adult",
                        "dataset_id": "openml_adult",
                        "target_col": "class",
                        "id_col": None,
                    }
                ],
                "encodings": [
                    {
                        "label": "one-hot",
                        "encoding_id": "one_hot_representation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _payload(
    status,
    *,
    run_id=None,
    owner=None,
    stage=None,
    heartbeat_at=None,
    started_at=None,
    finished_at=None,
    note="",
):
    return json.dumps(
        {
            "v": 1,
            "status": status,
            "run_id": run_id,
            "owner": owner,
            "started_at": started_at,
            "heartbeat_at": heartbeat_at,
            "finished_at": finished_at,
            "stage": stage,
            "note": note,
        }
    )


class FakeSheetsClient:
    def __init__(self, *, reread_value=None):
        self.write_calls = []
        self._raw_write_calls = []
        self.last_payload = None
        self._reread_value = reread_value

    @classmethod
    def single_not_started_cell(cls):
        return cls()

    @classmethod
    def ambiguous_claim(cls):
        return cls(
            reread_value=_payload(
                "in-progress",
                run_id="someone-else",
                owner="other-host:999:0",
                stage="launching",
                started_at="2026-04-02T00:00:00Z",
                heartbeat_at="2026-04-02T00:00:00Z",
            )
        )

    def read_matrix(self):
        cell_value = " " if not self._raw_write_calls else self._raw_write_calls[-1]
        return {
            "dataset_headers": ["adult"],
            "encoding_headers": ["one-hot"],
            "cell_values": {"B2": cell_value},
        }

    def read_cell(self, coord):
        if self._reread_value is not None and self._raw_write_calls:
            return self._reread_value
        return " " if not self._raw_write_calls else self._raw_write_calls[-1]

    def write_cell(self, coord, payload):
        if isinstance(payload, str):
            raw_payload = payload
        elif hasattr(payload, "model_dump"):
            raw_payload = json.dumps(payload.model_dump())
        else:
            raw_payload = json.dumps(payload.__dict__)
        parsed_payload = json.loads(raw_payload)
        self.last_payload = SimpleNamespace(**parsed_payload)
        self._raw_write_calls.append(raw_payload)
        self.write_calls.append(self.last_payload)


def fake_success_process(*args, **kwargs):
    stdout = io.StringIO('{"event":"progress","stage":"tuning","message":"started"}\n')
    return SimpleNamespace(
        stdout=stdout,
        poll=lambda: 0,
        wait=lambda timeout=None: 0,
        terminate=lambda: None,
        kill=lambda: None,
        returncode=0,
    )


def fake_long_running_process(*args, **kwargs):
    state = {"polls": 0}

    def poll():
        state["polls"] += 1
        return None if state["polls"] < 3 else 0

    return SimpleNamespace(
        stdout=io.StringIO(""),
        poll=poll,
        wait=lambda timeout=None: 0,
        terminate=lambda: None,
        kill=lambda: None,
        returncode=0,
    )


def fake_failing_process_with_oom_output(*args, **kwargs):
    stdout = io.StringIO("RuntimeError: MPS backend out of memory\n")
    return SimpleNamespace(
        stdout=stdout,
        poll=lambda: 1,
        wait=lambda timeout=None: 1,
        terminate=lambda: None,
        kill=lambda: None,
        returncode=1,
    )


def fake_signal_terminated_process(*args, **kwargs):
    stdout = io.StringIO("")
    return SimpleNamespace(
        stdout=stdout,
        poll=lambda: -9,
        wait=lambda timeout=None: -9,
        terminate=lambda: None,
        kill=lambda: None,
        returncode=-9,
    )


def fake_buffered_progress_process(*args, **kwargs):
    read_fd, write_fd = os.pipe()
    stdout = os.fdopen(read_fd, "r", encoding="utf-8")
    with os.fdopen(write_fd, "w", encoding="utf-8") as writer:
        writer.write('{"event":"progress","stage":"tuning","message":"started"}\n')
        writer.write('{"event":"progress","stage":"crossval","message":"running cross-validation"}\n')
        writer.flush()
    return SimpleNamespace(
        stdout=stdout,
        poll=lambda: 0,
        wait=lambda timeout=None: 0,
        terminate=lambda: None,
        kill=lambda: None,
        returncode=0,
    )


def test_dry_run_reports_candidate_without_writing(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    fake_sheets = FakeSheetsClient.single_not_started_cell()
    out = orchestrator_mod.run_once(
        sheets=fake_sheets,
        manifest_path=manifest_path,
        worksheet_name="CTGAN",
        dry_run=True,
    )
    assert out.claimed_coord == "B2"
    assert fake_sheets.write_calls == []


def test_orchestrator_claims_runs_and_marks_done(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    fake_sheets = FakeSheetsClient.single_not_started_cell()
    monkeypatch.setattr(orchestrator_mod, "spawn_runner", fake_success_process)

    out = orchestrator_mod.run_once(
        sheets=fake_sheets,
        manifest_path=manifest_path,
        worksheet_name="CTGAN",
        dry_run=False,
    )

    assert out.exit_code == 0
    assert fake_sheets.last_payload.status == "done"


def test_orchestrator_exits_non_zero_on_ambiguous_claim(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    fake_sheets = FakeSheetsClient.ambiguous_claim()
    out = orchestrator_mod.run_once(
        sheets=fake_sheets,
        manifest_path=manifest_path,
        worksheet_name="CTGAN",
        dry_run=False,
    )
    assert out.exit_code != 0


def test_orchestrator_updates_heartbeat_while_process_is_alive(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    fake_sheets = FakeSheetsClient.single_not_started_cell()
    monkeypatch.setattr(orchestrator_mod, "spawn_runner", fake_long_running_process)
    out = orchestrator_mod.run_once(
        sheets=fake_sheets,
        manifest_path=manifest_path,
        worksheet_name="CTGAN",
        dry_run=False,
        heartbeat_seconds=0,
    )
    assert out.exit_code == 0
    assert any(payload.status == "in-progress" for payload in fake_sheets.write_calls)


def test_orchestrator_does_not_miss_buffered_progress_lines(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    fake_sheets = FakeSheetsClient.single_not_started_cell()
    monkeypatch.setattr(orchestrator_mod, "spawn_runner", fake_buffered_progress_process)

    out = orchestrator_mod.run_once(
        sheets=fake_sheets,
        manifest_path=manifest_path,
        worksheet_name="CTGAN",
        dry_run=False,
    )

    assert out.exit_code == 0
    assert any(
        payload.status == "in-progress" and payload.stage == "crossval"
        for payload in fake_sheets.write_calls
    )


def test_read_available_line_drains_text_buffered_pipe():
    read_fd, write_fd = os.pipe()
    raw_stream = os.fdopen(read_fd, "rb")
    stream = io.TextIOWrapper(raw_stream, encoding="utf-8", newline="")
    writer = os.fdopen(write_fd, "wb", buffering=0)
    try:
        writer.write(b'{"event":"progress","stage":"tuning","message":"started"}\n')
        writer.write(b'{"event":"progress","stage":"crossval","message":"running cross-validation"}\n')
        writer.flush()

        first = orchestrator_mod._read_available_line(stream, timeout_seconds=0)
        second = orchestrator_mod._read_available_line(stream, timeout_seconds=0)
    finally:
        writer.close()

    assert first is not None and '"stage":"tuning"' in first
    assert second is not None and '"stage":"crossval"' in second


def test_orchestrator_classifies_out_of_memory_failures(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    fake_sheets = FakeSheetsClient.single_not_started_cell()
    monkeypatch.setattr(orchestrator_mod, "spawn_runner", fake_failing_process_with_oom_output)

    out = orchestrator_mod.run_once(
        sheets=fake_sheets,
        manifest_path=manifest_path,
        worksheet_name="CTGAN",
        dry_run=False,
    )

    assert out.exit_code != 0
    assert fake_sheets.last_payload.status == "failed"
    assert "resource_exhausted" in fake_sheets.last_payload.note
    assert "out of memory" in fake_sheets.last_payload.note


def test_orchestrator_classifies_signal_terminated_failures(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    fake_sheets = FakeSheetsClient.single_not_started_cell()
    monkeypatch.setattr(orchestrator_mod, "spawn_runner", fake_signal_terminated_process)

    out = orchestrator_mod.run_once(
        sheets=fake_sheets,
        manifest_path=manifest_path,
        worksheet_name="CTGAN",
        dry_run=False,
    )

    assert out.exit_code != 0
    assert fake_sheets.last_payload.status == "failed"
    assert "terminated by signal 9" in fake_sheets.last_payload.note


def test_build_runner_argv_passes_common_best_params_file_skip_tuning_and_device(tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    best_params_file = tmp_path / "best_params.json"
    best_params_file.write_text("{}", encoding="utf-8")
    dataset = SimpleNamespace(dataset_id="openml_adult", label="adult")
    encoding = SimpleNamespace(encoding_id="one_hot_representation", label="one-hot")

    argv = orchestrator_mod._build_runner_argv(
        manifest_path=manifest_path,
        dataset=dataset,
        encoding=encoding,
        output_root=tmp_path / "results",
        best_params_file=best_params_file,
        skip_tuning=True,
        device="cuda",
    )

    assert "--best-params-file" in argv
    assert str(best_params_file.resolve()) in argv
    assert "--skip-tuning" in argv
    assert argv[-2:] == ["--device", "cuda"]


def test_orchestrator_main_passes_best_params_file_skip_tuning_and_device(monkeypatch, tmp_path):
    manifest_path = write_orchestrator_manifest(tmp_path)
    best_params_file = tmp_path / "best_params.json"
    best_params_file.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    @dataclass(frozen=True)
    class DummySheetsConfig:
        worksheet_name: str = "CTGAN"

    def fake_from_env():
        return DummySheetsConfig()

    def fake_sheets_client(config):
        return object()

    def fake_run_once(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(exit_code=0)

    monkeypatch.setattr(orchestrator_mod.SheetsConfig, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(orchestrator_mod, "SheetsClient", fake_sheets_client)
    monkeypatch.setattr(orchestrator_mod, "run_once", fake_run_once)

    exit_code = orchestrator_mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--worksheet",
            "CTGAN",
            "--best-params-file",
            str(best_params_file),
            "--skip-tuning",
            "--device",
            "cuda",
        ]
    )

    assert exit_code == 0
    assert captured["best_params_file"] == best_params_file.resolve()
    assert captured["skip_tuning"] is True
    assert captured["device"] == "cuda"


def test_orchestrator_cli_help_runs_as_script():
    project_root = orchestrator_mod.Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "experiments/ctgan_orchestrator.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
