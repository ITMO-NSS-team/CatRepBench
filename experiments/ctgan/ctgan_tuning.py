"""
CTGAN Optuna tuning module.

Primary entrypoint:
    tune_ctgan(...)

Available tune_ctgan flags:
- df (pd.DataFrame): Full tabular dataset (features + target).
- schema (TabularSchema): Original dataset schema.
- dataset (str): Dataset name for artifacts path and metadata.
- encoding_method (str): Representation id. Must be one of
  `genbench.transforms.categorical.list_registered_representations()`.
- n_trials (int, default=30): Optuna trial budget.
- epochs (int, default=50): CTGAN epochs per trial.
- seed (int, default=42): Seed for Optuna sampler and trial reproducibility.
- task_type (Optional[str], default=None): "classification" or "regression".
  If None, inferred from target dtype/cardinality.
- holdout_cfg (Optional[SplitConfigHoldout], default=None): Optional custom
  tuning split config. If None, uses fixed 80/20 split with random_seed=5.
- discrete_cols (Optional[Sequence[str]], default=None): Explicit CTGAN
  discrete columns in transformed space. If None, inferred from transformed schema.
- output_root (Path | str, default=Path("experiments/optuna_results")):
  Root output directory.
- output_dir (Path | str | None, default=None): Optional explicit artifact
  directory. When provided, artifacts are written directly there.
- save_model (bool, default=False): Save fitted CTGAN artifacts for best params.
- timeout_seconds (Optional[int], default=None): Optional Optuna timeout.
- device (str, default="cuda"): "cpu" or "cuda".

Outputs:
- explicit `output_dir`, when provided
- otherwise: experiments/optuna_results/ctgan/<dataset>/<encoding_method>/
- summary.json, trials.csv, best_params.json (+ model_artifacts/ when save_model=True)
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, TextIO

import numpy as np
import optuna
import pandas as pd

from experiments.ctgan.ctgan_common import (
    DEFAULT_CTGAN_EPOCHS,
    build_ctgan_kwargs,
    build_preprocess_pipeline,
    default_discrete_cols,
)
from genbench.data.datamodule import TabularDataModule
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout
from genbench.evaluation.distribution.wasserstein import WassersteinDistanceMetric
from genbench.evaluation.pipeline.single_run import DistributionEvaluationPipeline
from genbench.generative.ctgan.ctgan import CtganGenerative
from genbench.transforms.target import TargetTypePreprocessor, infer_is_regression_target

@dataclass(frozen=True)
class CtganTuningResult:
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


@dataclass(frozen=True)
class CtganFitProgress:
    current_step: int
    total_steps: int
    fraction: float
    display_text: str


_CTGAN_FIT_PROGRESS_RE = re.compile(
    r"^(?P<display>Gen\.\s*\([^)]*\)\s*\|\s*Discrim\.\s*\([^)]*\):.*?\|\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)\s*\[[^\]]+\])$"
)
_PROGRESS_EMIT_INTERVAL_SECONDS = 10.0


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip()).strip("_") or "unknown"


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


def _parse_ctgan_fit_progress_line(raw_line: str) -> CtganFitProgress | None:
    line = raw_line.strip()
    if not line:
        return None
    match = _CTGAN_FIT_PROGRESS_RE.match(line)
    if match is None:
        return None
    current_step = int(match.group("current"))
    total_steps = int(match.group("total"))
    if total_steps <= 0:
        return None
    fraction = min(max(float(current_step) / float(total_steps), 0.0), 1.0)
    return CtganFitProgress(
        current_step=current_step,
        total_steps=total_steps,
        fraction=fraction,
        display_text=match.group("display"),
    )


def _format_eta_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "calculating"
    total_seconds = max(int(round(seconds)), 0)
    if total_seconds < 60:
        return f"{total_seconds}s"
    total_minutes, seconds_part = divmod(total_seconds, 60)
    if total_minutes < 60:
        return f"{total_minutes}m {seconds_part:02d}s"
    total_hours, minutes_part = divmod(total_minutes, 60)
    if total_hours < 24:
        return f"{total_hours}h {minutes_part:02d}m"
    total_days, hours_part = divmod(total_hours, 24)
    return f"{total_days}d {hours_part}h"


def _estimate_tuning_eta_seconds(
    *,
    elapsed_seconds: float,
    total_trials: int,
    completed_trials: int,
    current_trial_fraction: float,
) -> float | None:
    if total_trials <= 0:
        return None
    completed_units = float(completed_trials) + min(max(float(current_trial_fraction), 0.0), 1.0)
    if completed_units <= 0.0:
        return None
    remaining_units = max(float(total_trials) - completed_units, 0.0)
    return (float(elapsed_seconds) / completed_units) * remaining_units


def _format_tuning_progress_message(
    *,
    current_trial: int,
    total_trials: int,
    fit_progress: CtganFitProgress | None,
    eta_seconds: float | None,
    total_fit_steps: int | None = None,
) -> str:
    message = f"trial {current_trial}/{total_trials}"
    if fit_progress is not None:
        message = f"{message} | {fit_progress.display_text}"
    elif total_fit_steps is not None and total_fit_steps > 0:
        message = f"{message} | 0/{total_fit_steps} | waiting for first fit update"
    message = f"{message} | tuning eta {_format_eta_seconds(eta_seconds)}"
    return message


class _FitOutputCapture:
    def __init__(self, on_line: Callable[[str], None], *, passthrough: TextIO) -> None:
        self._on_line = on_line
        self._passthrough = passthrough
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._passthrough.write(text)
        self._buffer += text
        self._drain_buffer()
        return len(text)

    def flush(self) -> None:
        self._passthrough.flush()

    def finish(self) -> None:
        if self._buffer:
            self._on_line(self._buffer)
            self._buffer = ""

    def _drain_buffer(self) -> None:
        start = 0
        for index, char in enumerate(self._buffer):
            if char not in {"\r", "\n"}:
                continue
            chunk = self._buffer[start:index]
            if chunk:
                self._on_line(chunk)
            start = index + 1
        if start:
            self._buffer = self._buffer[start:]


def _fit_model_with_progress_capture(
    *,
    model: CtganGenerative,
    train_df: pd.DataFrame,
    transformed_schema: TabularSchema,
    total_trials: int,
    current_trial: int,
    completed_trials: int,
    tuning_started_at: float,
    progress_callback: Callable[[str], None] | None,
    total_fit_steps: int | None,
) -> None:
    if progress_callback is None:
        model.fit(train_df, transformed_schema)
        return

    last_emitted_at: float | None = None
    last_message: str | None = None
    fit_progress_emitted = False

    def maybe_emit(message: str, *, force: bool = False, fit_progress: bool = False) -> None:
        nonlocal fit_progress_emitted, last_emitted_at, last_message

        now = time.monotonic()
        if not force and message == last_message:
            return
        if (
            not force
            and last_emitted_at is not None
            and (now - last_emitted_at) < _PROGRESS_EMIT_INTERVAL_SECONDS
            and not (fit_progress and not fit_progress_emitted)
        ):
            return
        progress_callback(message)
        last_emitted_at = now
        last_message = message
        if fit_progress:
            fit_progress_emitted = True

    maybe_emit(
        _format_tuning_progress_message(
            current_trial=current_trial,
            total_trials=total_trials,
            fit_progress=None,
            eta_seconds=_estimate_tuning_eta_seconds(
                elapsed_seconds=time.monotonic() - tuning_started_at,
                total_trials=total_trials,
                completed_trials=completed_trials,
                current_trial_fraction=0.0,
            ),
            total_fit_steps=total_fit_steps,
        ),
        force=True,
    )

    def on_line(raw_line: str) -> None:
        progress = _parse_ctgan_fit_progress_line(raw_line)
        if progress is None:
            return
        message = _format_tuning_progress_message(
            current_trial=current_trial,
            total_trials=total_trials,
            fit_progress=progress,
            eta_seconds=_estimate_tuning_eta_seconds(
                elapsed_seconds=time.monotonic() - tuning_started_at,
                total_trials=total_trials,
                completed_trials=completed_trials,
                current_trial_fraction=progress.fraction,
            ),
        )
        maybe_emit(message, fit_progress=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    capture_stdout = _FitOutputCapture(on_line, passthrough=original_stdout)
    capture_stderr = _FitOutputCapture(on_line, passthrough=original_stderr)
    try:
        with contextlib.redirect_stdout(capture_stdout), contextlib.redirect_stderr(capture_stderr):
            model.fit(train_df, transformed_schema)
    finally:
        capture_stdout.finish()
        capture_stderr.finish()


def _suggest_ctgan_params(
    trial: optuna.Trial,
    *,
    epochs: int,
    device: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    embedding_dim = int(trial.suggest_categorical("embedding_dim", [128, 256]))
    gen_dim = int(trial.suggest_categorical("gen_dim", [256, 512]))
    disc_dim = int(trial.suggest_categorical("disc_dim", [256, 512]))
    batch_size = int(trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]))
    discriminator_steps = int(trial.suggest_int("discriminator_steps", 1, 5))
    generator_lr = float(trial.suggest_float("generator_lr", 1e-4, 2e-3, log=True))
    lr_ratio = float(trial.suggest_float("lr_ratio", 0.7, 1.5, log=True))

    return build_ctgan_kwargs(
        {
            "embedding_dim": embedding_dim,
            "gen_dim": gen_dim,
            "disc_dim": disc_dim,
            "batch_size": batch_size,
            "discriminator_steps": discriminator_steps,
            "generator_lr": generator_lr,
            "lr_ratio": lr_ratio,
        },
        epochs=epochs,
        device=device,
        verbose=verbose,
    )


def _score_synthetic(
    *,
    val_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    schema: TabularSchema,
) -> tuple[float, Dict[str, float]]:
    dist_pipeline = DistributionEvaluationPipeline(metrics=[WassersteinDistanceMetric()])
    dist_scores = dist_pipeline.evaluate(real=val_df, synth=synth_df, schema=schema).scores

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
    pipeline, representation_name = build_preprocess_pipeline(
        schema=schema,
        encoding_method=encoding_method,
        task_type=task_type,
    )
    dm = TabularDataModule(
        df=df,
        schema=schema,
        transforms=pipeline,
        unseen_category_policy="move_to_train",
        validate=True,
    )
    dm.prepare_holdout(holdout_cfg)
    holdout = dm.get_holdout()
    train_df, val_df = holdout.train, holdout.val

    target_transform: Optional[TargetTypePreprocessor] = None
    if holdout.transforms is not None and hasattr(holdout.transforms, "transforms"):
        for tr in holdout.transforms.transforms:  # type: ignore[union-attr]
            if isinstance(tr, TargetTypePreprocessor):
                target_transform = tr
                break

    target_state: Dict[str, Any] = {}
    if target_transform is not None:
        target_state = target_transform.get_state().params

    inferred_target_mode = False
    if "is_regression" in target_state and target_state["is_regression"] is not None:
        is_regression = bool(target_state["is_regression"])
    elif schema.target_col is not None and schema.target_col in train_df.columns:
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
            "source": "target_type_preprocessor" if not inferred_target_mode else "inferred_fallback",
            "target_encoded": target_encoded,
            "target_scaled": target_scaled,
            "target_mean": float(target_state.get("target_mean", 0.0)) if target_scaled else None,
            "target_std": float(target_state.get("target_std", 1.0)) if target_scaled else None,
            "target_classes": list(target_state.get("classes", [])) if target_encoded else [],
        },
        "discrete_unique_threshold": discrete_threshold,
    }
    return train_df, val_df, transformed_schema, preprocessing_meta


def tune_ctgan(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    dataset: str,
    encoding_method: str,
    n_trials: int = 30,
    epochs: int = DEFAULT_CTGAN_EPOCHS,
    seed: int = 42,
    task_type: Optional[str] = None,
    holdout_cfg: Optional[SplitConfigHoldout] = None,
    discrete_cols: Optional[Sequence[str]] = None,
    output_root: Path | str = Path("experiments/optuna_results"),
    output_dir: Path | str | None = None,
    save_model: bool = False,
    timeout_seconds: Optional[int] = None,
    device: str = "cuda",
    progress_callback: Callable[[str], None] | None = None,
) -> CtganTuningResult:
    """
    Finetune CTGAN with Optuna using only project wrappers/classes/metrics.

    Saves results to:
      output_dir, when provided
      otherwise experiments/optuna_results/ctgan/<dataset>/<encoding_method>
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0.")
    if epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    cfg = holdout_cfg or SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=5)
    train_df, val_df, transformed_schema, preprocessing_meta = _build_holdout(
        df=df,
        schema=schema,
        encoding_method=encoding_method,
        task_type=task_type,
        holdout_cfg=cfg,
    )
    is_regression = bool(preprocessing_meta["target_processing"]["is_regression"])
    output_dir = _ensure_dir(
        Path(output_dir)
        if output_dir is not None
        else Path(output_root) / "ctgan" / _slug(dataset) / _slug(encoding_method)
    )

    used_discrete_cols = (
        [c for c in discrete_cols if c in train_df.columns]
        if discrete_cols is not None
        else default_discrete_cols(
            schema=transformed_schema,
            train_df=train_df,
            include_target_for_classification=(not is_regression),
        )
    )

    study_name = f"ctgan_{_slug(dataset)}_{_slug(encoding_method)}"
    storage_uri = f"sqlite:///{(output_dir / 'study.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage_uri,
        load_if_exists=True,
    )

    started_at = time.monotonic()
    started_trials = 0
    completed_trials = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal started_trials, completed_trials

        started_trials += 1
        current_trial = started_trials
        params = _suggest_ctgan_params(
            trial,
            epochs=epochs,
            device=device,
            verbose=(progress_callback is not None),
        )
        model = CtganGenerative(discrete_cols=used_discrete_cols, ctgan_kwargs=params)
        try:
            _fit_model_with_progress_capture(
                model=model,
                train_df=train_df,
                transformed_schema=transformed_schema,
                total_trials=int(n_trials),
                current_trial=current_trial,
                completed_trials=completed_trials,
                tuning_started_at=started_at,
                progress_callback=progress_callback,
                total_fit_steps=int(params["epochs"]),
            )
            synth_df = model.sample(len(val_df))
            score, details = _score_synthetic(
                val_df=val_df,
                synth_df=synth_df,
                schema=transformed_schema,
            )
            if not np.isfinite(score):
                raise optuna.TrialPruned("Non-finite score.")
            trial.set_user_attr("objective_metric", "wasserstein_mean")
            trial.set_user_attr("sample_size", int(len(val_df)))
            trial.set_user_attr("details", details)
            return float(score)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            raise optuna.TrialPruned(f"Trial failed: {exc}") from exc
        finally:
            completed_trials += 1

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        n_jobs=1,
        show_progress_bar=False,
    )
    duration_seconds = time.monotonic() - started_at

    if study.best_trial is None:
        raise RuntimeError("No successful Optuna trials.")

    best_params = dict(study.best_trial.params)
    best_value = float(study.best_value)
    best_source = "stage1"
    best_model: Optional[CtganGenerative] = None

    trials_path = output_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    best_params_path = output_dir / "best_params.json"
    _save_json(best_params_path, {"best_params": best_params, "best_value": best_value, "best_source": best_source})

    model_artifacts_dir: Optional[Path] = None
    if save_model:
        if best_model is None:
            final_kwargs = _suggest_ctgan_params(
                trial=optuna.trial.FixedTrial(best_params),
                epochs=int(epochs),
                device=device,
            )
            best_model = CtganGenerative(discrete_cols=used_discrete_cols, ctgan_kwargs=final_kwargs)
            best_model.fit(train_df, transformed_schema)
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
        "best_params": best_params,
        "n_trials": int(n_trials),
        "epochs": int(epochs),
        "seed": int(seed),
        "task_type": "regression" if is_regression else "classification",
        "used_discrete_cols": used_discrete_cols,
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

    return CtganTuningResult(
        dataset=dataset,
        encoding_method=encoding_method,
        study_name=study_name,
        output_dir=output_dir,
        best_value=best_value,
        best_params=best_params,
        best_source=best_source,
        n_trials=n_trials,
        epochs=epochs,
        duration_seconds=duration_seconds,
        summary_path=summary_path,
        trials_path=trials_path,
        best_params_path=best_params_path,
        model_artifacts_dir=model_artifacts_dir,
    )


def tune_ctgan_and_return_params(**kwargs: Any) -> Dict[str, Any]:
    """
    Helper for external callers that need only the returned params + summary path.
    """
    result = tune_ctgan(**kwargs)
    return {
        **select_ctgan_best_params_from_result(result),
        "summary_path": str(result.summary_path),
    }


def select_ctgan_best_params_from_result(result: CtganTuningResult) -> Dict[str, Any]:
    return {
        "best_params": result.best_params,
        "best_value": result.best_value,
        "best_source": result.best_source,
    }


def select_ctgan_best_params(**kwargs: Any) -> Dict[str, Any]:
    """
    Run the tuning stage and return only the selected hyperparameters contract.
    """
    result = tune_ctgan(**kwargs)
    return select_ctgan_best_params_from_result(result)
