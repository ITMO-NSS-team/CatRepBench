"""
CTGAN Optuna tuning module.

Primary entrypoint:
    tune_ctgan(...)

Parameters (tune_ctgan):
- df (pd.DataFrame): Input tabular dataset (features + target).
- schema (TabularSchema): Project schema object. `schema.target_col` must be set.
- dataset (str): Dataset name used in output path and metadata.
- encoding_method (str): Encoding label used in output path and metadata.
- n_trials (int, default=30): Number of Optuna trials.
- epochs (int, default=50): CTGAN epochs per stage-1 trial.
- finetune_epochs (Optional[int], default=300): Epochs for stage-2 top-k refinement.
- finetune_top_k (int, default=5): Number of top completed stage-1 trials to refine.
- seed (int, default=42): Base random seed for reproducibility.
- task_type (Optional[str], default=None): "classification" or "regression". If None, inferred.
- holdout_cfg (Optional[SplitConfigHoldout], default=None): Holdout split config.
- discrete_cols (Optional[Sequence[str]], default=None): Explicit discrete columns passed to CTGAN wrapper.
- output_root (Path | str, default=Path("experiments/optuna_results")): Root output directory.
- save_model (bool, default=False): Save best model artifacts via CtganGenerative.save_artifacts().
- timeout_seconds (Optional[int], default=None): Optional Optuna optimize timeout.
- dist_weight (float, default=0.1): Weight for distribution penalty in objective.
- device (str, default="cuda"): "cpu" or "cuda"; propagated to CTGAN wrapper kwargs.

Returns:
- CtganTuningResult with best params/score and generated artifact paths.

Outputs are saved under:
- experiments/optuna_results/ctgan/<dataset>/<encoding_method>/
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import optuna
import pandas as pd

from genbench.data.datamodule import TabularDataModule
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout
from genbench.evaluation.distribution.corr_frobenius import CorrelationFrobeniusMetric
from genbench.evaluation.distribution.marginal_kl import MarginalKLDivergenceMetric
from genbench.evaluation.distribution.wasserstein import WassersteinDistanceMetric
from genbench.evaluation.pipeline.single_run import DistributionEvaluationPipeline
from genbench.evaluation.utility.tstr_catboost import TSTRCatBoostEvaluator
from genbench.generative.ctgan.ctgan import CtganGenerative


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
    finetune_epochs: Optional[int]
    duration_seconds: float
    summary_path: Path
    trials_path: Path
    best_params_path: Path
    model_artifacts_dir: Optional[Path] = None


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


def _infer_is_regression(schema: TabularSchema, df: pd.DataFrame, task_type: Optional[str]) -> bool:
    if task_type is not None:
        normalized = task_type.strip().lower()
        if normalized not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'.")
        return normalized == "regression"

    if schema.target_col is None:
        return False

    y = df[schema.target_col]
    return bool(pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20)


def _default_discrete_cols(
    schema: TabularSchema,
    train_df: pd.DataFrame,
    *,
    include_target_for_classification: bool,
) -> list[str]:
    cols = [c for c in list(schema.categorical_cols) + list(schema.discrete_cols) if c in train_df.columns]
    if include_target_for_classification and schema.target_col and schema.target_col in train_df.columns:
        if schema.target_col not in cols:
            cols.append(schema.target_col)
    return cols


def _suggest_ctgan_params(
    trial: optuna.Trial,
    *,
    epochs: int,
    device: str,
) -> Dict[str, Any]:
    embedding_dim = int(trial.suggest_categorical("embedding_dim", [128, 256]))
    gen_dim = int(trial.suggest_categorical("gen_dim", [256, 512]))
    disc_dim = int(trial.suggest_categorical("disc_dim", [256, 512]))
    batch_size = int(trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]))
    discriminator_steps = int(trial.suggest_int("discriminator_steps", 1, 5))
    generator_lr = float(trial.suggest_float("generator_lr", 1e-4, 2e-3, log=True))
    lr_ratio = float(trial.suggest_float("lr_ratio", 0.7, 1.5, log=True))

    return {
        "epochs": int(epochs),
        "embedding_dim": embedding_dim,
        "generator_dim": (gen_dim, gen_dim),
        "discriminator_dim": (disc_dim, disc_dim),
        "batch_size": batch_size,
        "discriminator_steps": discriminator_steps,
        "generator_lr": generator_lr,
        "discriminator_lr": generator_lr * lr_ratio,
        "pac": 1,
        "verbose": False,
        "cuda": device == "cuda",
    }


def _score_synthetic(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    schema: TabularSchema,
    is_regression: bool,
    random_seed: int,
    dist_weight: float,
) -> tuple[float, Dict[str, float]]:
    dist_pipeline = DistributionEvaluationPipeline(
        metrics=[
            WassersteinDistanceMetric(),
            MarginalKLDivergenceMetric(),
            CorrelationFrobeniusMetric(),
        ]
    )
    dist_scores = dist_pipeline.evaluate(real=train_df, synth=synth_df, schema=schema).scores

    dist_values = [
        float(dist_scores.get("wasserstein_mean", 0.0)),
        float(dist_scores.get("marginal_kl_mean", 0.0)),
        float(dist_scores.get("corr_frobenius", 0.0)),
    ]
    dist_penalty = float(np.mean(dist_values))

    utility_score: Optional[float] = None
    if is_regression and schema.target_col is not None:
        utility_result = TSTRCatBoostEvaluator(random_seed=random_seed).evaluate(
            train_real=train_df,
            test_real=val_df,
            synth_train=synth_df,
            schema=schema,
        )
        utility_score = float(utility_result.scores["r2_synth"])
        final_score = utility_score - dist_weight * dist_penalty
    else:
        # For classification tasks use distribution-only objective with maximize direction.
        final_score = -dist_penalty

    details: Dict[str, float] = {
        "objective_score": float(final_score),
        "dist_penalty": dist_penalty,
        "wasserstein_mean": float(dist_scores.get("wasserstein_mean", 0.0)),
        "marginal_kl_mean": float(dist_scores.get("marginal_kl_mean", 0.0)),
        "corr_frobenius": float(dist_scores.get("corr_frobenius", 0.0)),
    }
    if utility_score is not None:
        details["utility_r2_synth"] = utility_score
    return float(final_score), details


def _build_holdout(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    holdout_cfg: SplitConfigHoldout,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dm = TabularDataModule(df=df, schema=schema, transforms=None, validate=True)
    dm.prepare_holdout(holdout_cfg)
    holdout = dm.get_holdout()
    return holdout.train, holdout.val


def tune_ctgan(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    dataset: str,
    encoding_method: str,
    n_trials: int = 30,
    epochs: int = 50,
    finetune_epochs: Optional[int] = 300,
    finetune_top_k: int = 5,
    seed: int = 42,
    task_type: Optional[str] = None,
    holdout_cfg: Optional[SplitConfigHoldout] = None,
    discrete_cols: Optional[Sequence[str]] = None,
    output_root: Path | str = Path("experiments/optuna_results"),
    save_model: bool = False,
    timeout_seconds: Optional[int] = None,
    dist_weight: float = 0.1,
    device: str = "cuda",
) -> CtganTuningResult:
    """
    Finetune CTGAN with Optuna using only project wrappers/classes/metrics.

    Saves results to:
      experiments/optuna_results/ctgan/<dataset>/<encoding_method>
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0.")
    if epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")

    output_dir = _ensure_dir(Path(output_root) / "ctgan" / _slug(dataset) / _slug(encoding_method))
    cfg = holdout_cfg or SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=seed)
    train_df, val_df = _build_holdout(df=df, schema=schema, holdout_cfg=cfg)
    is_regression = _infer_is_regression(schema, train_df, task_type)

    used_discrete_cols = (
        [c for c in discrete_cols if c in train_df.columns]
        if discrete_cols is not None
        else _default_discrete_cols(
            schema=schema,
            train_df=train_df,
            include_target_for_classification=(not is_regression),
        )
    )

    study_name = f"ctgan_{_slug(dataset)}_{_slug(encoding_method)}"
    storage_uri = f"sqlite:///{(output_dir / 'study.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage_uri,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_seed = seed + trial.number
        params = _suggest_ctgan_params(trial, epochs=epochs, device=device)
        model = CtganGenerative(discrete_cols=used_discrete_cols, ctgan_kwargs=params)
        try:
            model.fit(train_df, schema)
            synth_df = model.sample(len(train_df))
            score, details = _score_synthetic(
                train_df=train_df,
                val_df=val_df,
                synth_df=synth_df,
                schema=schema,
                is_regression=is_regression,
                random_seed=trial_seed,
                dist_weight=dist_weight,
            )
            if not np.isfinite(score):
                raise optuna.TrialPruned("Non-finite score.")
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

    stage1_best_params = dict(study.best_trial.params)
    stage1_best_value = float(study.best_value)
    best_params = stage1_best_params
    best_value = stage1_best_value
    best_source = "stage1"
    best_model: Optional[CtganGenerative] = None
    finetune_best_value: Optional[float] = None

    if finetune_epochs and finetune_top_k > 0:
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        complete_trials.sort(key=lambda t: float(t.value), reverse=True)
        for rank, trial in enumerate(complete_trials[: int(finetune_top_k)]):
            trial_seed = seed + 10_000 + rank
            params = _suggest_ctgan_params(
                trial=optuna.trial.FixedTrial(trial.params),
                epochs=int(finetune_epochs),
                device=device,
            )
            model = CtganGenerative(discrete_cols=used_discrete_cols, ctgan_kwargs=params)
            try:
                model.fit(train_df, schema)
                synth_df = model.sample(len(train_df))
                score, _ = _score_synthetic(
                    train_df=train_df,
                    val_df=val_df,
                    synth_df=synth_df,
                    schema=schema,
                    is_regression=is_regression,
                    random_seed=trial_seed,
                    dist_weight=dist_weight,
                )
            except Exception:
                continue
            if not np.isfinite(score):
                continue
            if finetune_best_value is None or float(score) > finetune_best_value:
                finetune_best_value = float(score)
                best_value = float(score)
                best_params = dict(trial.params)
                best_source = "finetune"
                best_model = model

    trials_path = output_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    best_params_path = output_dir / "best_params.json"
    _save_json(best_params_path, {"best_params": best_params, "best_value": best_value, "best_source": best_source})

    model_artifacts_dir: Optional[Path] = None
    if save_model:
        if best_model is None:
            final_epochs = int(finetune_epochs) if best_source == "finetune" and finetune_epochs else int(epochs)
            final_kwargs = _suggest_ctgan_params(
                trial=optuna.trial.FixedTrial(best_params),
                epochs=final_epochs,
                device=device,
            )
            best_model = CtganGenerative(discrete_cols=used_discrete_cols, ctgan_kwargs=final_kwargs)
            best_model.fit(train_df, schema)
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
        "stage1_best_value": stage1_best_value,
        "finetune_best_value": finetune_best_value,
        "best_params": best_params,
        "n_trials": int(n_trials),
        "epochs": int(epochs),
        "finetune_epochs": int(finetune_epochs) if finetune_epochs else None,
        "finetune_top_k": int(finetune_top_k),
        "seed": int(seed),
        "task_type": "regression" if is_regression else "classification",
        "used_discrete_cols": used_discrete_cols,
        "dist_weight": float(dist_weight),
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
        finetune_epochs=finetune_epochs,
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
        "best_params": result.best_params,
        "best_value": result.best_value,
        "best_source": result.best_source,
        "summary_path": str(result.summary_path),
    }
