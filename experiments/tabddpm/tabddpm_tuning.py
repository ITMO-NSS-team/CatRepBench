"""Runner-facing TabDDPM tuning/runtime-estimate adapters.

The shared CTGAN/TVAE runner expects every model to expose a
``select_best_params(...)`` and ``estimate_runtime(...)`` with a fixed keyword
signature. This module adapts the existing Optuna tuner in
``experiments/tabddpm_tuning.py`` to that contract (same shape as
``experiments/tvae/tvae_tuning.py``). The Optuna objective minimizes
``wasserstein_mean`` — the common objective across models.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from experiments.tabddpm.tabddpm_common import (
    DEFAULT_TABDDPM_NUM_STEPS,
    build_tabddpm_kwargs,
)
from experiments.tabddpm_tuning import (
    _build_holdout,
    _score_synthetic,
    tune_tabddpm,
)
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout
from genbench.generative.tabddpm.tabddpm import TabDDPMGenerative


def select_tabddpm_best_params(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    dataset: str,
    encoding_method: str,
    task_type: Optional[str] = None,
    output_dir: Path | str | None = None,
    device: str = "cuda",
    progress_callback: Optional[Callable[[str], None]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Tune TabDDPM and return the runner's best-params payload.

    Mirrors ``select_tvae_best_params``: returns
    ``{"best_params", "best_value", "best_source"}``.
    """
    result = tune_tabddpm(
        df=df,
        schema=schema,
        dataset=dataset,
        encoding_method=encoding_method,
        task_type=task_type,
        output_dir=output_dir,
        device=device,
        progress_callback=progress_callback,
        **extra,
    )
    return {
        "best_params": result.best_params,
        "best_value": result.best_value,
        "best_source": result.best_source,
    }


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip()).strip("_") or "unknown"


def estimate_tabddpm_runtime(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    dataset: str,
    encoding_method: str,
    sample_epochs: int = 10,
    projected_epochs: int = DEFAULT_TABDDPM_NUM_STEPS,
    projected_total_runs: int = 35,
    task_type: Optional[str] = None,
    output_dir: Path | str | None = None,
    output_root: Path | str = Path("experiments/optuna_results"),
    device: str = "cuda",
    progress_callback: Optional[Callable[[str], None]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Time a short TabDDPM fit and project full-pipeline runtime.

    TabDDPM trains for a fixed number of optimizer *steps*; the runner's
    ``*_epochs`` arguments are interpreted here as step counts.
    """
    if sample_epochs <= 0:
        raise ValueError("sample_epochs must be > 0.")
    if projected_epochs <= 0:
        raise ValueError("projected_epochs must be > 0.")
    if projected_total_runs <= 0:
        raise ValueError("projected_total_runs must be > 0.")
    if sample_epochs > projected_epochs:
        raise ValueError("sample_epochs must be <= projected_epochs.")

    cfg = SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=5)
    train_df, val_df, transformed_schema, preprocessing_meta, pipeline = _build_holdout(
        df=df,
        schema=schema,
        encoding_method=encoding_method,
        task_type=task_type,
        holdout_cfg=cfg,
    )

    model_kwargs = build_tabddpm_kwargs({}, epochs=int(sample_epochs), device=device)
    model = TabDDPMGenerative(**model_kwargs)

    if progress_callback is not None:
        progress_callback(
            f"estimating runtime | sampled steps {sample_epochs}/{projected_epochs}"
        )

    fit_started_at = time.monotonic()
    model.fit(train_df, transformed_schema, source_schema=schema)
    fit_seconds_sampled = float(time.monotonic() - fit_started_at)

    post_fit_started_at = time.monotonic()
    synth_df = model.sample(len(val_df))
    score, details = _score_synthetic(
        val_processed=val_df,
        synth_processed=synth_df,
        schema_raw=schema,
        pipeline=pipeline,
    )
    post_fit_seconds = float(time.monotonic() - post_fit_started_at)

    projected_fit_seconds = fit_seconds_sampled * (
        float(projected_epochs) / float(sample_epochs)
    )
    projected_trial_seconds = projected_fit_seconds + post_fit_seconds
    projected_full_pipeline_seconds = projected_trial_seconds * float(projected_total_runs)

    out_dir = Path(output_dir) if output_dir is not None else (
        Path(output_root) / "tabddpm_runtime_estimate" / _slug(dataset) / _slug(encoding_method)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    payload: Dict[str, Any] = {
        "mode": "runtime_estimate",
        "dataset": dataset,
        "encoding_method": encoding_method,
        "sample_steps": int(sample_epochs),
        "projected_steps": int(projected_epochs),
        "projected_total_runs": int(projected_total_runs),
        "objective_metric": "wasserstein_mean",
        "objective_score_sampled": float(score),
        "objective_details": details,
        "fit_seconds_sampled": float(fit_seconds_sampled),
        "post_fit_seconds": float(post_fit_seconds),
        "projected_fit_seconds": float(projected_fit_seconds),
        "projected_trial_seconds": float(projected_trial_seconds),
        "projected_full_pipeline_seconds": float(projected_full_pipeline_seconds),
        "projected_full_pipeline_hours": float(projected_full_pipeline_seconds / 3600.0),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in payload.items()},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    if progress_callback is not None:
        progress_callback(
            f"estimating runtime | projected full pipeline {projected_full_pipeline_seconds:.0f}s"
        )

    payload["summary_path"] = str(summary_path)
    payload["output_dir"] = str(out_dir)
    return payload
