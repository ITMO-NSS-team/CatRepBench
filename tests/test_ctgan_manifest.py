import csv
import json
from pathlib import Path

import pytest

from experiments.ctgan_manifest import load_ctgan_manifest
from genbench.transforms.categorical import list_registered_representations


def test_load_manifest_resolves_dataset_label_without_source_prefix(tmp_path: Path):
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

    manifest = load_ctgan_manifest(manifest_path, project_root=tmp_path)
    dataset = manifest.resolve_dataset_label(" adult ")

    assert dataset.dataset_id == "openml_adult"
    assert dataset.target_col == "class"
    assert dataset.csv_path == tmp_path / "datasets" / "raw" / "openml_adult.csv"


def test_load_manifest_resolve_dataset_label_is_case_sensitive(tmp_path: Path):
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

    manifest = load_ctgan_manifest(manifest_path, project_root=tmp_path)

    with pytest.raises(ValueError, match="Unknown dataset label"):
        manifest.resolve_dataset_label("Adult")


def test_load_manifest_rejects_duplicate_dataset_labels(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "class", "id_col": None},
                    {"label": "adult", "dataset_id": "uci_adult", "target_col": "target", "id_col": None},
                ],
                "encodings": [{"label": "one-hot", "encoding_id": "one_hot_representation"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_ctgan_manifest(manifest_path, project_root=tmp_path)


def test_load_manifest_resolves_encoding_label(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "class", "id_col": None}
                ],
                "encodings": [
                    {"label": "one-hot", "encoding_id": "one_hot_representation"}
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_ctgan_manifest(manifest_path, project_root=tmp_path)
    encoding = manifest.resolve_encoding_label(" one-hot ")

    assert encoding.encoding_id == "one_hot_representation"


def test_load_manifest_resolve_encoding_label_is_case_sensitive(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "class", "id_col": None}
                ],
                "encodings": [
                    {"label": "one-hot", "encoding_id": "one_hot_representation"}
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_ctgan_manifest(manifest_path, project_root=tmp_path)

    with pytest.raises(ValueError, match="Unknown encoding label"):
        manifest.resolve_encoding_label("One-Hot")


def test_load_manifest_rejects_empty_target_col(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "", "id_col": None},
                ],
                "encodings": [{"label": "one-hot", "encoding_id": "one_hot_representation"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_ctgan_manifest(manifest_path, project_root=tmp_path)


def test_load_manifest_rejects_duplicate_dataset_ids(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "class", "id_col": None},
                    {"label": "bank-marketing", "dataset_id": "openml_adult", "target_col": "y", "id_col": None},
                ],
                "encodings": [{"label": "one-hot", "encoding_id": "one_hot_representation"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_ctgan_manifest(manifest_path, project_root=tmp_path)


def test_load_manifest_rejects_duplicate_encoding_ids(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "class", "id_col": None},
                ],
                "encodings": [
                    {"label": "one-hot", "encoding_id": "one_hot_representation"},
                    {"label": "one-hot-2", "encoding_id": "one_hot_representation"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_ctgan_manifest(manifest_path, project_root=tmp_path)


def test_load_manifest_rejects_duplicate_encoding_labels_after_trim(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "class", "id_col": None},
                ],
                "encodings": [
                    {"label": "one-hot", "encoding_id": "one_hot_representation"},
                    {"label": " one-hot ", "encoding_id": "ordinal_representation"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_ctgan_manifest(manifest_path, project_root=tmp_path)


def test_load_manifest_rejects_labels_that_only_differ_by_spaces(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"label": "adult", "dataset_id": "openml_adult", "target_col": "class", "id_col": None},
                    {"label": " adult ", "dataset_id": "uci_adult", "target_col": "target", "id_col": None},
                ],
                "encodings": [{"label": "one-hot", "encoding_id": "one_hot_representation"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_ctgan_manifest(manifest_path, project_root=tmp_path)


def test_repository_manifest_covers_canonical_datasets_registered_encodings_and_runner_metadata():
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_ctgan_manifest(
        project_root / "experiments" / "ctgan_orchestrator_manifest.json",
        project_root=project_root,
    )

    dataset_ids = {entry.dataset_id for entry in manifest.datasets}
    expected_dataset_ids = {
        "openml_MagicTelescope",
        "openml_PhishingWebsites",
        "openml_adult",
        "openml_bank-marketing",
        "openml_conference_attendance",
        "openml_connect-4",
        "openml_default-of-credit-card-clients",
        "openml_eucalyptus",
        "openml_letter",
        "openml_nursery",
        "uci_Covertype",
        "uci_Credit_Approval",
        "uci_Forest_Fires",
        "uci_HTRU2",
        "uci_Heart_Disease",
        "uci_Mammographic_Mass",
        "uci_Online_Shoppers_Purchasing_Intention_Dataset",
        "uci_Seoul_Bike_Sharing_Demand",
        "uci_Soybean__Large_",
        "uci_Statlog__German_Credit_Data_",
        "uci_Student_Performance",
    }
    encoding_ids = {entry.encoding_id for entry in manifest.encodings}

    assert dataset_ids == expected_dataset_ids
    assert encoding_ids == set(list_registered_representations())

    for entry in manifest.datasets:
        assert entry.csv_path == project_root / "datasets" / "raw" / f"{entry.dataset_id}.csv"


def test_repository_manifest_matches_downloaded_dataset_headers_when_raw_csvs_are_present():
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_ctgan_manifest(
        project_root / "experiments" / "ctgan_orchestrator_manifest.json",
        project_root=project_root,
    )
    missing_paths = [entry.csv_path for entry in manifest.datasets if not entry.csv_path.exists()]
    if missing_paths:
        pytest.skip("downloaded datasets/raw CSVs are not present in this checkout")

    for entry in manifest.datasets:
        with entry.csv_path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))

        assert entry.target_col in header
        assert entry.id_col is None or entry.id_col in header
