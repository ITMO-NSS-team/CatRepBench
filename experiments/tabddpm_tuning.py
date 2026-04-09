"""
TabDDPM Optuna tuning module.

Primary entrypoint:
    tune_tabddpm(...)

Available tune_tabddpm flags:
- df (pd.DataFrame): Full tabular dataset (features + target).
- schema (TabularSchema): Original dataset schema.
- dataset (str): Dataset name for artifacts path and metadata.
- encoding_method (str): Representation id. Must be one of
  `genbench.transforms.categorical.list_registered_representations()`.
- n_trials (int, default=30): Optuna trial budget.
- epochs (int, default=100): TabDDPM training epochs per trial.
- seed (int, default=42): Seed for Optuna sampler and trial reproducibility.
- task_type (Optional[str], default=None): "classification" or "regression".
  If None, inferred from target dtype/cardinality.
- holdout_cfg (Optional[SplitConfigHoldout], default=None): Optional custom
  tuning split config. If None, uses fixed 80/20 split with random_seed=5.
- output_root (Path | str, default=Path("experiments/optuna_results")):
  Root output directory.
- save_model (bool, default=False): Save fitted TabDDPM artifacts for best
params.
- timeout_seconds (Optional[int], default=None): Optional Optuna timeout.
- device (str, default="cuda"): "cpu" or "cuda".

Outputs:
- experiments/optuna_results/tabddpm/<dataset>/<encoding_method>/
- summary.json, trials.csv, best_params.json (+ model_artifacts/ when
save_model=True)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

from genbench.data.datamodule import TabularDataModule
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout
from genbench.evaluation.distribution.wasserstein import \
    WassersteinDistanceMetric
from genbench.evaluation.pipeline.single_run import \
    DistributionEvaluationPipeline
from genbench.transforms.categorical import (
    CategoricalRepresentationTransform,
    list_registered_representations,
)
from genbench.transforms.continuous import ContinuousStandardScaler
from genbench.transforms.missing import DropMissingRows
from genbench.transforms.pipeline import TransformPipeline
from genbench.transforms.target import TargetTypePreprocessor, \
    infer_is_regression_target
from genbench.generative.tabddpm.tabddpm import TabDDPMGenerative


@dataclass(frozen=True)
class TabDDPMTuningResult:
    dataset: str
    encoding_method: str
    study_name: str
    output_dir: Path
    best_value: float
    best_params: Dict[str, Any]
    best_source: str
    n_trials: int
    epochs: int
    duration_seconds: float
    summary_path: Path
    trials_path: Path
    best_params_path: Path
    model_artifacts_dir: Optional[Path] = None


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip()).strip(
        "_") or "unknown"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)


def _validate_encoding_method(encoding_method: str) -> str:
    supported = list_registered_representations()
    if encoding_method not in supported:
        raise ValueError(
            "Unsupported encoding_method. "
            f"Use one of {supported}. Got: '{encoding_method}'."
        )
    return encoding_method


def _build_preprocess_pipeline(
        schema: TabularSchema,
        encoding_method: str,
        task_type: Optional[str],
) -> tuple[TransformPipeline, Optional[str]]:
    representation_name = _validate_encoding_method(encoding_method)
    transforms = [DropMissingRows(), ContinuousStandardScaler()]
    if schema.categorical_cols:
        transforms.append(
            CategoricalRepresentationTransform(
                representation_name=representation_name,
            )
        )
    else:
        representation_name = None
    transforms.append(
        TargetTypePreprocessor(
            task_type=task_type,
        )
    )
    return TransformPipeline(transforms=transforms), representation_name


def _suggest_mlp_layers(trial: optuna.Trial) -> List[int]:
    """Suggests a list of layer dimensions for the MLP in TabDDPM."""

    def suggest_dim(name: str) -> int:
        t = trial.suggest_int(name, 7, 10)
        return 2 ** t

    min_n_layers, max_n_layers = 1, 4
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)

    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []

    d_layers = d_first + d_middle + d_last
    return d_layers


def _suggest_tabddpm_params(
        trial: optuna.Trial,
        *,
        epochs: int,
        device: str,
) -> Dict[str, Any]:
    """Sample hyperparameters for TabDDPM."""

    num_timesteps = trial.suggest_categorical("num_timesteps", [100, 1000])
    batch_size = trial.suggest_categorical("batch_size", [256, 4096])
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    d_layers = _suggest_mlp_layers(trial)

    weight_decay = 0.0
    gaussian_loss_type = "mse"
    dim_t = 256
    scheduler = "cosine"
    dropout = 0.0

    return {
        "num_timesteps": num_timesteps,
        "num_epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dim_t": dim_t,
        "d_layers": d_layers,
        "dropout": dropout,
        "scheduler": scheduler,
        "gaussian_loss_type": gaussian_loss_type,
        "device": device,
    }


def _score_synthetic(
        *,
        val_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        schema: TabularSchema,
) -> tuple[float, Dict[str, float]]:
    dist_pipeline = DistributionEvaluationPipeline(
        metrics=[WassersteinDistanceMetric()])
    dist_scores = dist_pipeline.evaluate(real=val_df, synth=synth_df,
                                         schema=schema).scores

    wd = float(dist_scores.get("wasserstein_mean", np.nan))
    details: Dict[str, float] = {
        "objective_score": wd,
        "wasserstein_mean": wd,
    }
    return wd, details


def _build_holdout(
        *,
        df: pd.DataFrame,
        schema: TabularSchema,
        encoding_method: str,
        task_type: Optional[str],
        holdout_cfg: SplitConfigHoldout,
) -> tuple[pd.DataFrame, pd.DataFrame, TabularSchema, Dict[str, Any]]:
    pipeline, representation_name = _build_preprocess_pipeline(
        schema=schema,
        encoding_method=encoding_method,
        task_type=task_type,
    )
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline,
                           validate=True)
    dm.prepare_holdout(holdout_cfg)
    holdout = dm.get_holdout()
    train_df, val_df = holdout.train, holdout.val

    target_transform: Optional[TargetTypePreprocessor] = None
    if holdout.transforms is not None and hasattr(holdout.transforms,
                                                  "transforms"):
        for tr in holdout.transforms.transforms:  # type: ignore[union-attr]
            if isinstance(tr, TargetTypePreprocessor):
                target_transform = tr
                break

    target_state: Dict[str, Any] = {}
    if target_transform is not None:
        target_state = target_transform.get_state().params

    inferred_target_mode = False
    if "is_regression" in target_state and target_state[
        "is_regression"] is not None:
        is_regression = bool(target_state["is_regression"])
    elif (schema.target_col is not None and schema.target_col in
          train_df.columns):
        is_regression = infer_is_regression_target(
            train_df[schema.target_col],
            task_type=task_type,
        )
        inferred_target_mode = True
    else:
        is_regression = False

    transformed_schema = TabularSchema.infer_from_dataframe(
        train_df,
        target_col=schema.target_col,
        id_col=schema.id_col,
    )
    transformed_schema.validate(val_df)
    discrete_threshold = int(target_state.get("discrete_unique_threshold", 20))
    target_encoded = bool(target_state.get("did_encode", False))
    target_scaled = bool(target_state.get("did_scale", False))
    preprocessing_meta: Dict[str, Any] = {
        "representation_name": representation_name,
        "target_processing": {
            "is_regression": bool(is_regression),
            "source": "target_type_preprocessor" if not inferred_target_mode
            else "inferred_fallback",
            "target_encoded": target_encoded,
            "target_scaled": target_scaled,
            "target_mean": float(target_state.get("target_mean",
                                                  0.0)) if target_scaled
            else None,
            "target_std": float(target_state.get("target_std",
                                                 1.0)) if target_scaled else
            None,
            "target_classes": list(
                target_state.get("classes", [])) if target_encoded else [],
        },
        "discrete_unique_threshold": discrete_threshold,
    }
    return train_df, val_df, transformed_schema, preprocessing_meta


def tune_tabddpm(
        *,
        df: pd.DataFrame,
        schema: TabularSchema,
        dataset: str,
        encoding_method: str,
        n_trials: int = 30,
        epochs: int = 100,
        seed: int = 42,
        task_type: Optional[str] = None,
        holdout_cfg: Optional[SplitConfigHoldout] = None,
        output_root: Path | str = Path("experiments/optuna_results"),
        save_model: bool = False,
        timeout_seconds: Optional[int] = None,
        device: str = "cuda",
) -> TabDDPMTuningResult:
    """
    Fine-tune TabDDPM with Optuna using the Wasserstein distance as objective.

    Saves results to:
      experiments/optuna_results/tabddpm/<dataset>/<encoding_method>
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0.")
    if epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    encoding_method = _validate_encoding_method(encoding_method)

    output_dir = _ensure_dir(
        Path(output_root) / "tabddpm" / _slug(dataset) / _slug(
            encoding_method))
    cfg = holdout_cfg or SplitConfigHoldout(val_size=0.2, shuffle=True,
                                            random_seed=5)
    train_df, val_df, transformed_schema, preprocessing_meta = _build_holdout(
        df=df,
        schema=schema,
        encoding_method=encoding_method,
        task_type=task_type,
        holdout_cfg=cfg,
    )
    is_regression = bool(
        preprocessing_meta["target_processing"]["is_regression"])

    study_name = f"tabddpm_{_slug(dataset)}_{_slug(encoding_method)}"
    storage_uri = f"sqlite:///{(output_dir / 'study.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage_uri,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_tabddpm_params(trial, epochs=epochs, device=device)
        model = TabDDPMGenerative(**params)
        try:
            model.fit(train_df, transformed_schema, source_schema=schema)
            synth_df = model.sample(len(val_df))
            score, details = _score_synthetic(
                val_df=val_df,
                synth_df=synth_df,
                schema=transformed_schema,
            )
            if not np.isfinite(score):
                raise optuna.TrialPruned("Non‑finite score.")
            trial.set_user_attr("objective_metric", "wasserstein_mean")
            trial.set_user_attr("sample_size", int(len(val_df)))
            trial.set_user_attr("details", details)
            return float(score)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            raise optuna.TrialPruned(f"Trial failed: {exc}") from exc

    started_at = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        n_jobs=1,
        show_progress_bar=False,
    )
    duration_seconds = time.time() - started_at

    if study.best_trial is None:
        raise RuntimeError("No successful Optuna trials.")

    full_best_params = _suggest_tabddpm_params(
        trial=optuna.trial.FixedTrial(dict(study.best_trial.params)),
        epochs=int(epochs),
        device=device,
    )

    best_value = float(study.best_value)
    best_source = "stage1"

    trials_path = output_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    best_params_path = output_dir / "best_params.json"
    _save_json(best_params_path,
               {"best_params": full_best_params, "best_value": best_value,
                "best_source": best_source})

    model_artifacts_dir: Optional[Path] = None
    if save_model:
        best_model = TabDDPMGenerative(**full_best_params)
        best_model.fit(train_df, transformed_schema, source_schema=schema)
        model_artifacts_dir = _ensure_dir(output_dir / "model_artifacts")
        best_model.save_artifacts(model_artifacts_dir)

    summary_path = output_dir / "summary.json"
    summary_payload: Dict[str, Any] = {
        "dataset": dataset,
        "encoding_method": encoding_method,
        "study_name": study_name,
        "storage_uri": storage_uri,
        "best_source": best_source,
        "best_value": best_value,
        "best_params": full_best_params,
        "n_trials": int(n_trials),
        "epochs": int(epochs),
        "seed": int(seed),
        "task_type": "regression" if is_regression else "classification",
        "objective_metric": "wasserstein_mean",
        "objective_direction": "minimize",
        "holdout": {
            "val_size": float(cfg.val_size),
            "shuffle": bool(cfg.shuffle),
            "random_seed": int(cfg.random_seed),
        },
        "preprocessing": preprocessing_meta,
        "duration_seconds": float(duration_seconds),
        "output_dir": output_dir,
        "trials_path": trials_path,
        "best_params_path": best_params_path,
        "model_artifacts_dir": model_artifacts_dir,
    }
    _save_json(summary_path, summary_payload)

    return TabDDPMTuningResult(
        dataset=dataset,
        encoding_method=encoding_method,
        study_name=study_name,
        output_dir=output_dir,
        best_value=best_value,
        best_params=full_best_params,
        best_source=best_source,
        n_trials=n_trials,
        epochs=epochs,
        duration_seconds=duration_seconds,
        summary_path=summary_path,
        trials_path=trials_path,
        best_params_path=best_params_path,
        model_artifacts_dir=model_artifacts_dir,
    )


def tune_tabddpm_and_return_params(**kwargs: Any) -> Dict[str, Any]:
    """
    Helper for external callers that need only the returned params + summary
    path.
    """
    result = tune_tabddpm(**kwargs)
    return {
        "best_params": result.best_params,
        "best_value": result.best_value,
        "best_source": result.best_source,
        "summary_path": str(result.summary_path),
    }
