from __future__ import annotations

import html
import json
from typing import Any

from experiments.ctgan.experiment_models import get_experiment_model


def _model_label(model_id: str) -> str:
    return get_experiment_model(model_id).display_name


def build_monitor_payload(
    cells: list[dict[str, Any]],
    *,
    selected_model_id: str | None = None,
) -> dict[str, Any]:
    normalized_cells: list[dict[str, Any]] = []
    model_order: list[str] = []
    model_labels: dict[str, str] = {}

    for cell in cells:
        model_id = str(cell.get("model_id") or "ctgan").strip().lower()
        label = _model_label(model_id)
        if model_id not in model_labels:
            model_order.append(model_id)
            model_labels[model_id] = label
        normalized = dict(cell)
        normalized["model_id"] = model_id
        normalized["model_label"] = label
        normalized_cells.append(normalized)

    selected = selected_model_id or (model_order[0] if model_order else "ctgan")
    selected = selected.strip().lower()
    if selected not in model_labels:
        selected = model_order[0] if model_order else "ctgan"

    return {
        "model_order": model_order,
        "model_labels": model_labels,
        "selected_model_id": selected,
        "cells": normalized_cells,
    }


def render_monitor_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    options = "".join(
        f'<option value="{html.escape(model_id)}">{html.escape(payload["model_labels"][model_id])}</option>'
        for model_id in payload.get("model_order", [])
    )
    selector = (
        f'<label>Model <select id="model-select">{options}</select></label>'
        if len(payload.get("model_order", [])) > 1
        else (
            f'<strong>{html.escape(payload["model_labels"].get(payload.get("selected_model_id", "ctgan"), "CTGAN"))}</strong>'
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CatRepBench Monitor</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #334155; padding: 8px; text-align: left; }}
    select {{ background: #111827; color: #e2e8f0; border: 1px solid #475569; padding: 6px; }}
  </style>
</head>
<body>
  <h1>CatRepBench Monitor</h1>
  <div>{selector}</div>
  <table>
    <thead><tr><th>Model</th><th>Dataset</th><th>Encoding</th><th>Status</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <script>
    const payload = {payload_json};
    const rows = document.getElementById("rows");
    const selector = document.getElementById("model-select");
    function render() {{
      const modelId = selector ? selector.value : payload.selected_model_id;
      rows.innerHTML = "";
      for (const cell of payload.cells.filter((item) => item.model_id === modelId)) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{cell.model_label}}</td><td>${{cell.dataset_label || cell.dataset_id || ""}}</td><td>${{cell.encoding_label || cell.encoding_id || ""}}</td><td>${{cell.status || ""}}</td>`;
        rows.appendChild(tr);
      }}
    }}
    if (selector) selector.value = payload.selected_model_id;
    if (selector) selector.addEventListener("change", render);
    render();
  </script>
</body>
</html>"""
