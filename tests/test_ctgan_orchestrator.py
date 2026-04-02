import io
import json
import subprocess
import sys
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
