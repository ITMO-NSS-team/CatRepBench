from experiments.ctgan.ctgan_monitor import build_monitor_payload, render_monitor_html


def test_build_monitor_payload_defaults_legacy_rows_to_ctgan():
    payload = build_monitor_payload(
        [
            {
                "dataset_id": "openml_adult",
                "dataset_label": "adult",
                "encoding_id": "one_hot_representation",
                "encoding_label": "one-hot",
                "status": "done",
            }
        ]
    )
    assert payload["model_order"] == ["ctgan"]
    assert payload["model_labels"] == {"ctgan": "CTGAN"}
    assert payload["selected_model_id"] == "ctgan"
    assert payload["cells"][0]["model_id"] == "ctgan"
    assert payload["cells"][0]["model_label"] == "CTGAN"


def test_build_monitor_payload_keeps_multiple_models():
    payload = build_monitor_payload(
        [
            {"model_id": "ctgan", "dataset_id": "d1", "encoding_id": "e1", "status": "done"},
            {"model_id": "tvae", "dataset_id": "d1", "encoding_id": "e1", "status": "done"},
        ],
        selected_model_id="tvae",
    )
    assert payload["model_order"] == ["ctgan", "tvae"]
    assert payload["selected_model_id"] == "tvae"
    assert payload["cells"][1]["model_label"] == "TVAE"


def test_render_monitor_html_contains_model_selector_for_multiple_models():
    payload = build_monitor_payload(
        [
            {"model_id": "ctgan", "dataset_id": "d1", "encoding_id": "e1", "status": "done"},
            {"model_id": "tvae", "dataset_id": "d1", "encoding_id": "e1", "status": "done"},
        ]
    )
    html = render_monitor_html(payload)
    assert "<select" in html
    assert "TVAE" in html
    assert "CTGAN" in html
