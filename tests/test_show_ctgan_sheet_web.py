from __future__ import annotations

import os
import sys
import threading
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
from urllib.request import urlopen

import show_ctgan_sheet_web


class ShowCtganSheetWebTests(unittest.TestCase):
    def test_load_snapshot_does_not_import_full_orchestrator(self) -> None:
        from experiments.ctgan.orchestrator_staff.ctgan_manifest import (
            CtganManifest,
            DatasetEntry,
            EncodingEntry,
        )

        sys.modules.pop("experiments.ctgan.orchestrator_staff.ctgan_orchestrator", None)
        manifest = CtganManifest(
            datasets=(
                DatasetEntry(
                    label="dataset-a",
                    dataset_id="dataset_a",
                    target_col="target",
                    id_col=None,
                    csv_path=Path("dataset_a.csv"),
                ),
            ),
            encodings=(EncodingEntry(label="encoding-a", encoding_id="encoding_a"),),
        )

        snapshot = show_ctgan_sheet_web._load_snapshot(
            [["", "dataset-a"], ["encoding-a", ""]],
            manifest=manifest,
        )

        self.assertEqual(snapshot.dataset_headers, ("dataset-a",))
        self.assertEqual(snapshot.encoding_headers, ("encoding-a",))
        self.assertEqual(snapshot.cell_values, {"B2": ""})
        self.assertNotIn("experiments.ctgan.orchestrator_staff.ctgan_orchestrator", sys.modules)

    def test_main_uses_railway_port_env_when_no_cli_port_is_given(self) -> None:
        attempts: list[tuple[str, int]] = []

        class FakeApp:
            def __init__(self, **kwargs):
                pass

            def make_handler(self):
                return object

        class FakeServer:
            def __init__(self, address, handler):
                attempts.append(address)
                self.server_address = address

            def serve_forever(self):
                return None

        with (
            patch.dict(os.environ, {"PORT": "34567"}, clear=False),
            patch.object(show_ctgan_sheet_web, "App", FakeApp),
            patch.object(show_ctgan_sheet_web, "ThreadingHTTPServer", FakeServer),
        ):
            self.assertEqual(show_ctgan_sheet_web.main(["--no-sheets"]), 0)

        self.assertEqual(attempts, [("0.0.0.0", 34567)])

    def test_repair_invalid_ca_bundle_env_uses_certifi_bundle(self) -> None:
        import certifi

        missing = "/path/that/does/not/exist/ca-certificates.crt"
        with patch.dict(
            os.environ,
            {"SSL_CERT_FILE": missing, "REQUESTS_CA_BUNDLE": missing},
            clear=False,
        ):
            show_ctgan_sheet_web._repair_invalid_ca_bundle_env()

            self.assertEqual(os.environ["SSL_CERT_FILE"], certifi.where())
            self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], certifi.where())

    def test_main_resolves_default_paths_relative_to_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                captured = _run_main_with_fake_server()
            finally:
                os.chdir(old_cwd)

        repo_root = Path(show_ctgan_sheet_web.__file__).resolve().parent
        self.assertEqual(
            captured["manifest_path"],
            repo_root / "experiments/ctgan/orchestrator_staff/ctgan_orchestrator_manifest.json",
        )
        self.assertEqual(
            captured["output_root"],
            repo_root / "experiments/results/recomputed_ctgan",
        )

    def test_main_falls_back_when_default_port_is_busy(self) -> None:
        attempts: list[tuple[str, int]] = []

        class FakeApp:
            def __init__(self, **kwargs):
                pass

            def make_handler(self):
                return object

        class FakeServer:
            def __init__(self, address, handler):
                attempts.append(address)
                if address[1] == 8765:
                    raise OSError(48, "Address already in use")

            def serve_forever(self):
                return None

        with (
            patch.object(show_ctgan_sheet_web, "App", FakeApp),
            patch.object(show_ctgan_sheet_web, "ThreadingHTTPServer", FakeServer),
        ):
            self.assertEqual(show_ctgan_sheet_web.main(["--no-sheets"]), 0)

        self.assertEqual(attempts, [("127.0.0.1", 8765), ("127.0.0.1", 8766)])

    def test_main_does_not_fall_back_when_port_is_explicit(self) -> None:
        class FakeApp:
            def __init__(self, **kwargs):
                pass

            def make_handler(self):
                return object

        class FakeServer:
            def __init__(self, address, handler):
                raise OSError(48, "Address already in use")

        with (
            patch.object(show_ctgan_sheet_web, "App", FakeApp),
            patch.object(show_ctgan_sheet_web, "ThreadingHTTPServer", FakeServer),
        ):
            with self.assertRaises(OSError):
                show_ctgan_sheet_web.main(["--no-sheets", "--port", "8765"])

    def test_favicon_request_returns_no_content(self) -> None:
        app = show_ctgan_sheet_web.App(
            manifest_path=show_ctgan_sheet_web.DEFAULT_MANIFEST,
            worksheet_name=None,
            results_worksheet="",
            output_root=show_ctgan_sheet_web.DEFAULT_OUTPUT_ROOT,
            refresh_seconds=30,
            no_sheets=True,
        )
        server = show_ctgan_sheet_web.ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            response = urlopen(f"http://{host}:{port}/favicon.ico", timeout=5)
            self.assertEqual(response.status, 204)
            self.assertEqual(response.read(), b"")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_app_results_payload_serves_cache_and_starts_background_drive_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            app = show_ctgan_sheet_web.App(
                manifest_path=show_ctgan_sheet_web.DEFAULT_MANIFEST,
                worksheet_name=None,
                results_worksheet="Results",
                output_root=output_root,
                refresh_seconds=30,
                no_sheets=False,
            )

            with (
                patch.object(app, "_maybe_start_drive_sync") as start_sync,
                patch.object(show_ctgan_sheet_web, "build_results_payload", return_value={"ok": True}) as builder,
            ):
                self.assertEqual(app.build_results_payload()["ok"], True)

        start_sync.assert_called_once()
        builder.assert_called_once_with(
            manifest_path=show_ctgan_sheet_web.DEFAULT_MANIFEST,
            results_worksheet="Results",
            output_root=output_root,
            sync_drive=False,
            rank_excluded_dataset_ids=[],
        )

    def test_results_payload_passes_skipped_datasets_as_rank_exclusions(self) -> None:
        from experiments.ctgan.orchestrator_staff.ctgan_manifest import (
            CtganManifest,
            DatasetEntry,
            EncodingEntry,
        )

        manifest = CtganManifest(
            datasets=(
                DatasetEntry(
                    label="dataset-a",
                    dataset_id="dataset_a",
                    target_col="target",
                    id_col=None,
                    csv_path=Path("dataset_a.csv"),
                ),
                DatasetEntry(
                    label="dataset-b",
                    dataset_id="dataset_b",
                    target_col="target",
                    id_col=None,
                    csv_path=Path("dataset_b.csv"),
                ),
            ),
            encodings=(EncodingEntry(label="encoding-a", encoding_id="encoding_a"),),
        )
        status_payload = {
            "rows": [
                {
                    "cells": [
                        {"dataset": "dataset-a", "status": "done", "effective_status": "done"},
                        {"dataset": "dataset-b", "status": "skipped", "effective_status": "skipped"},
                    ]
                }
            ]
        }

        self.assertEqual(
            show_ctgan_sheet_web.skipped_dataset_ids_from_status_payload(status_payload, manifest=manifest),
            ["dataset_b"],
        )

    def test_drive_sync_state_persists_progress_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = show_ctgan_sheet_web.DriveSyncState(Path(tmp) / "sync.sqlite3")
            state.start_run()
            state.record_cell_start(
                dataset_id="dataset-a",
                encoding_id="encoding-a",
                folder_url="https://drive.google.com/drive/folders/folder-a",
            )
            state.record_cell_finish(
                dataset_id="dataset-a",
                encoding_id="encoding-a",
                status="ok",
                downloaded_files=3,
                error="",
            )
            state.finish_run(status="ok", error="")

            reopened = show_ctgan_sheet_web.DriveSyncState(Path(tmp) / "sync.sqlite3")
            summary = reopened.summary()

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["total_downloaded_files"], 3)
        self.assertEqual(summary["cells"][0]["dataset_id"], "dataset-a")
        self.assertEqual(summary["cells"][0]["status"], "ok")

    def test_drive_sync_includes_fold_detail_artifacts(self) -> None:
        self.assertTrue(show_ctgan_sheet_web._is_drive_artifact_path("fold_details/fold_2.json"))
        self.assertFalse(show_ctgan_sheet_web._is_drive_artifact_path("fold_details/notes.txt"))

    def test_results_payload_hides_local_legacy_aggregate_without_results_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "experiments" / "ctgan" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "label": "ghost-dataset",
                                "dataset_id": "ghost_dataset",
                                "target_col": "target",
                                "id_col": None,
                            }
                        ],
                        "encodings": [
                            {
                                "label": "legacy-encoding",
                                "encoding_id": "legacy_encoding",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "results" / "ghost_dataset" / "legacy_encoding"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "metrics" / "aggregate.json").write_text(
                json.dumps(
                    {
                        "distribution": {
                            "wasserstein_mean": {"mean": 1.0, "std": 0.1},
                            "marginal_kl_mean": {"mean": 2.0, "std": 0.2},
                            "corr_frobenius_transformed": {"mean": 3.0, "std": 0.3},
                        },
                        "tstr": {
                            "task_type": "classification",
                            "metrics": {
                                "f1_weighted_pct_diff": {"mean": 4.0, "std": 0.4},
                                "f1_weighted_real": {"mean": 0.9},
                                "f1_weighted_synth": {"mean": 0.8},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                show_ctgan_sheet_web,
                "_sheets_read",
                return_value=([["Dataset", "Categorical representation"]], "Results", "sheet", "url"),
            ):
                payload = show_ctgan_sheet_web.build_results_payload(
                    manifest_path=manifest_path,
                    results_worksheet="Results",
                    output_root=root / "results",
                    sync_drive=False,
                )

        cell = payload["cells"][0]
        self.assertFalse(cell["has_data"])
        self.assertIsNone(cell["wd_mean"])
        self.assertIsNone(cell["utility_mean"])


def _run_main_with_fake_server() -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def make_handler(self):
            return object

    class FakeServer:
        def __init__(self, address, handler):
            captured["address"] = address
            captured["handler"] = handler

        def serve_forever(self):
            return None

    with (
        patch.object(show_ctgan_sheet_web, "App", FakeApp),
        patch.object(show_ctgan_sheet_web, "ThreadingHTTPServer", FakeServer),
    ):
        assert show_ctgan_sheet_web.main(["--no-sheets", "--host", "127.0.0.1", "--port", "0"]) == 0
    return captured


if __name__ == "__main__":
    unittest.main()
