from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ctgan.ctgan_common import (
    DEFAULT_CTGAN_EPOCHS,
    build_ctgan_kwargs,
    build_preprocess_pipeline,
    default_discrete_cols,
)
from experiments.ctgan.orchestrator_staff.ctgan_manifest import load_ctgan_manifest
from experiments.ctgan.ctgan_tuning import estimate_ctgan_runtime, select_ctgan_best_params
from genbench.data.datamodule import TabularDataModule
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout, SplitConfigKFold
from genbench.evaluation.distribution.corr_frobenius import CorrelationFrobeniusMetric
from genbench.evaluation.distribution.marginal_kl import MarginalKLDivergenceMetric
from genbench.evaluation.distribution.wasserstein import WassersteinDistanceMetric
from genbench.evaluation.pipeline.single_run import DistributionEvaluationPipeline
from genbench.generative.ctgan.ctgan import CtganGenerative
from genbench.transforms.target import infer_is_regression_target as _infer_is_regression_target


def tstr_catboost(**kwargs: Any) -> dict[str, float]:
    from genbench.evaluation.utility.tstr_catboost import tstr_catboost as _tstr_catboost

    return _tstr_catboost(**kwargs)


def infer_is_regression_target(*args: Any, **kwargs: Any) -> bool:
    return _infer_is_regression_target(*args, **kwargs)


@dataclass(frozen=True)
class FullCtganExperimentResult:
    output_dir: Path
    summary_path: Path
    aggregate_metrics_path: Path


@dataclass(frozen=True)
class PreparedFoldData:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_transformed: pd.DataFrame
    test_transformed: pd.DataFrame
    transformed_schema: TabularSchema
    transforms: Any | None


def _resolve_manifest_encoding(manifest: Any, *, encoding_method: str) -> Any:
    for entry in manifest.encodings:
        if getattr(entry, "encoding_id", None) == encoding_method:
            return manifest.resolve_encoding_label(entry.label)

    available = [getattr(entry, "encoding_id", None) for entry in getattr(manifest, "encodings", ())]
    raise ValueError(
        f"encoding_method mismatch: {encoding_method!r} is not declared in manifest encodings {available!r}"
    )


def _infer_project_root(manifest_path: Path) -> Path:
    for candidate in manifest_path.parents:
        if (candidate / "datasets" / "raw").exists() or (candidate / "genbench").exists():
            return candidate
    return manifest_path.parent


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
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(payload), handle, ensure_ascii=False, indent=2)


def _load_best_params_file(path: Path | str) -> dict[str, Any]:
    best_params_path = Path(path).resolve()
    with best_params_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("best_params_file must contain a JSON object.")

    best_params = payload.get("best_params", payload)
    if not isinstance(best_params, dict):
        raise ValueError("best_params_file must provide a JSON object in 'best_params' or at top level.")

    return {str(key): value for key, value in best_params.items()}


def _emit_progress(
    *,
    stage: str,
    message: str,
    progress_stream: TextIO | None,
    progress_format: str,
    dataset_id: str,
    encoding_method: str,
) -> None:
    if progress_format != "jsonl":
        raise ValueError("progress_format must be 'jsonl'.")
    stream = progress_stream or sys.stdout
    payload = {
        "event": "progress",
        "stage": stage,
        "message": message,
        "dataset_id": dataset_id,
        "encoding_method": encoding_method,
    }
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def _aggregate_numeric_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not records:
        return {}

    aggregate: dict[str, dict[str, float]] = {}
    keys = {key for record in records for key, value in record.items() if isinstance(value, (int, float))}
    for key in sorted(keys):
        values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
        if not values:
            continue
        if len(values) == 1:
            std = 0.0
        else:
            std = float(pd.Series(values, dtype=float).std(ddof=0))
        aggregate[key] = {
            "mean": float(pd.Series(values, dtype=float).mean()),
            "std": std,
        }
    return aggregate


def _cap_dataframe_rows(df: pd.DataFrame, *, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        return df.reset_index(drop=True)
    if max_rows <= 1:
        raise ValueError("max_rows must be > 1.")
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=42).sort_index().reset_index(drop=True)


def _prepare_fold_data(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    is_regression: bool | None,
    split_cfg: SplitConfigKFold,
    fold_id: int,
) -> PreparedFoldData:
    task_type = None
    if is_regression is True:
        task_type = "regression"
    elif is_regression is False:
        task_type = "classification"

    pipeline, _ = build_preprocess_pipeline(
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
    dm.prepare_kfold(split_cfg)
    fold = dm.get_fold(fold_id)
    if fold.train_raw is None or fold.test_raw is None:
        raise RuntimeError("FoldData must include raw train/test data.")
    train_raw = fold.train_raw.reset_index(drop=True)
    test_raw = fold.test_raw.reset_index(drop=True)
    train_df = fold.train.reset_index(drop=True)
    test_df = fold.test.reset_index(drop=True)
    transformed_schema = TabularSchema.infer_from_dataframe(
        train_df,
        target_col=schema.target_col,
        id_col=schema.id_col,
    )
    transformed_schema.validate(test_df)
    return PreparedFoldData(
        train_raw=train_raw,
        test_raw=test_raw,
        train_transformed=train_df,
        test_transformed=test_df,
        transformed_schema=transformed_schema,
        transforms=fold.transforms,
    )


def _prepare_holdout_data(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    is_regression: bool | None,
) -> PreparedFoldData:
    task_type = None
    if is_regression is True:
        task_type = "regression"
    elif is_regression is False:
        task_type = "classification"

    pipeline, _ = build_preprocess_pipeline(
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
    dm.prepare_holdout(SplitConfigHoldout(val_size=0.2, shuffle=True, random_seed=42))
    holdout = dm.get_holdout()
    if holdout.train_raw is None or holdout.val_raw is None:
        raise RuntimeError("HoldoutData must include raw train/val data.")
    train_raw = holdout.train_raw.reset_index(drop=True)
    test_raw = holdout.val_raw.reset_index(drop=True)
    train_df = holdout.train.reset_index(drop=True)
    test_df = holdout.val.reset_index(drop=True)
    transformed_schema = TabularSchema.infer_from_dataframe(
        train_df,
        target_col=schema.target_col,
        id_col=schema.id_col,
    )
    transformed_schema.validate(test_df)
    return PreparedFoldData(
        train_raw=train_raw,
        test_raw=test_raw,
        train_transformed=train_df,
        test_transformed=test_df,
        transformed_schema=transformed_schema,
        transforms=holdout.transforms,
    )


def _task_type_from_flag(is_regression: bool | None) -> str | None:
    if is_regression is True:
        return "regression"
    if is_regression is False:
        return "classification"
    return None


def _compute_distribution_scores(
    *,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    transformed_schema: TabularSchema,
) -> dict[str, float]:
    dist_pipeline = DistributionEvaluationPipeline(
        metrics=[
            WassersteinDistanceMetric(),
            MarginalKLDivergenceMetric(),
            CorrelationFrobeniusMetric(),
        ]
    )
    scores = dist_pipeline.evaluate(real=test_df, synth=synth_df, schema=transformed_schema).scores
    return {
        "wasserstein_mean": float(scores["wasserstein_mean"]),
        "marginal_kl_mean": float(scores["marginal_kl_mean"]),
        "corr_frobenius_transformed": float(scores["corr_frobenius"]),
    }


def _compute_original_space_corr(
    *,
    test_raw: pd.DataFrame,
    synth_df: pd.DataFrame,
    schema: TabularSchema,
    transforms: Any | None,
) -> tuple[float | None, str]:
    if transforms is None:
        return float(CorrelationFrobeniusMetric().compute(real=test_raw, synth=synth_df, schema=schema)), "ok"

    try:
        synth_original = transforms.inverse_transform(synth_df).reset_index(drop=True)
    except (NotImplementedError, RuntimeError, ValueError, KeyError):
        return None, "unsupported_not_invertible"

    value = float(CorrelationFrobeniusMetric().compute(real=test_raw, synth=synth_original, schema=schema))
    return value, "ok"


def _compute_tstr_scores(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    transformed_schema: TabularSchema,
    is_regression: bool | None,
) -> dict[str, Any]:
    if transformed_schema.target_col is None:
        return {"status": "unsupported_no_target"}

    scores = tstr_catboost(
        train_real=train_df,
        test_real=test_df,
        synth_train=synth_df,
        schema=transformed_schema,
        task_type=_task_type_from_flag(is_regression),
    )
    return {"status": "ok", **scores}


def _save_model_artifacts(model: Any, artifacts_dir: Path, split_id: int) -> None:
    """Save model checkpoint and loss history CSV for one fold."""
    fold_dir = _ensure_dir(artifacts_dir / f"fold_{split_id}")
    try:
        model.save_artifacts(fold_dir)
    except Exception:
        pass

    try:
        loss_history = model.get_loss_history()
        if loss_history:
            rows = [
                {"epoch": i, "generator_loss": g, "discriminator_loss": d}
                for i, (g, d) in enumerate(
                    zip(loss_history["generator_loss"], loss_history["discriminator_loss"])
                )
            ]
            pd.DataFrame(rows).to_csv(fold_dir / "loss_history.csv", index=False)
    except Exception:
        pass


def _evaluate_prepared_split(
    *,
    split_id: int,
    split_data: PreparedFoldData,
    schema: TabularSchema,
    best_params: dict[str, Any],
    device: str,
    is_regression: bool | None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    ctgan_kwargs = build_ctgan_kwargs(best_params, epochs=DEFAULT_CTGAN_EPOCHS, device=device)
    discrete_cols = default_discrete_cols(
        schema=split_data.transformed_schema,
        train_df=split_data.train_transformed,
        include_target_for_classification=(is_regression is False),
    )
    model = CtganGenerative(discrete_cols=discrete_cols, ctgan_kwargs=ctgan_kwargs)
    model.fit(split_data.train_transformed, split_data.transformed_schema)

    if artifacts_dir is not None:
        _save_model_artifacts(model, artifacts_dir, split_id)

    synth_df = model.sample(len(split_data.train_transformed)).reset_index(drop=True)

    distribution_scores = _compute_distribution_scores(
        test_df=split_data.test_transformed,
        synth_df=synth_df,
        transformed_schema=split_data.transformed_schema,
    )
    corr_original_value, corr_original_status = _compute_original_space_corr(
        test_raw=split_data.test_raw,
        synth_df=synth_df,
        schema=schema,
        transforms=split_data.transforms,
    )
    distribution_scores["corr_frobenius_original"] = corr_original_value
    distribution_scores["corr_frobenius_original_status"] = corr_original_status

    tstr_scores = _compute_tstr_scores(
        train_df=split_data.train_transformed,
        test_df=split_data.test_transformed,
        synth_df=synth_df,
        transformed_schema=split_data.transformed_schema,
        is_regression=is_regression,
    )

    return {
        "fold_id": int(split_id),
        "n_train": int(len(split_data.train_transformed)),
        "n_test": int(len(split_data.test_transformed)),
        "distribution": distribution_scores,
        "utility": tstr_scores,
    }


def _run_crossval_fold(
    *,
    fold_id: int,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    best_params: dict[str, Any],
    device: str,
    is_regression: bool | None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    split_cfg = SplitConfigKFold(n_splits=5, shuffle=True, random_seed=42)
    fold_data = _prepare_fold_data(
        df=df,
        schema=schema,
        encoding_method=encoding_method,
        is_regression=is_regression,
        split_cfg=split_cfg,
        fold_id=fold_id,
    )
    return _evaluate_prepared_split(
        split_id=fold_id,
        split_data=fold_data,
        schema=schema,
        best_params=best_params,
        device=device,
        is_regression=is_regression,
        artifacts_dir=artifacts_dir,
    )


def _run_holdout_split(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    best_params: dict[str, Any],
    device: str,
    is_regression: bool | None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    holdout_data = _prepare_holdout_data(
        df=df,
        schema=schema,
        encoding_method=encoding_method,
        is_regression=is_regression,
    )
    return _evaluate_prepared_split(
        split_id=0,
        split_data=holdout_data,
        schema=schema,
        best_params=best_params,
        device=device,
        is_regression=is_regression,
        artifacts_dir=artifacts_dir,
    )


def run_ctgan_runtime_estimate(
    *,
    manifest_path: Path | str,
    dataset_id: str,
    dataset_label: str,
    encoding_method: str,
    output_root: Path | str = Path("experiments/results"),
    progress_stream: TextIO | None = None,
    progress_format: str = "jsonl",
    device: str = "cuda",
    max_rows: int | None = None,
    estimate_sample_epochs: int = 10,
    estimate_total_epochs: int = DEFAULT_CTGAN_EPOCHS,
    estimate_total_runs: int = 35,
):
    manifest_path = Path(manifest_path).resolve()
    project_root = _infer_project_root(manifest_path)
    output_dir = _ensure_dir(Path(output_root) / "ctgan" / dataset_id / encoding_method / "runtime_estimate")

    _emit_progress(
        stage="launching",
        message="loading manifest and dataset",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )

    manifest = load_ctgan_manifest(manifest_path, project_root=project_root)
    dataset = manifest.resolve_dataset_label(dataset_label)
    encoding = _resolve_manifest_encoding(manifest, encoding_method=encoding_method)
    if dataset.dataset_id != dataset_id:
        raise ValueError(
            f"dataset_id mismatch for label {dataset_label!r}: "
            f"expected {dataset.dataset_id!r}, got {dataset_id!r}"
        )
    if encoding.encoding_id != encoding_method:
        raise ValueError(
            f"encoding_method mismatch: expected {encoding.encoding_id!r}, got {encoding_method!r}"
        )

    df = pd.read_csv(project_root / "datasets" / "raw" / f"{dataset_id}.csv")
    df = _cap_dataframe_rows(df, max_rows=max_rows)
    schema = TabularSchema.infer_from_dataframe(
        df,
        target_col=dataset.target_col,
        id_col=dataset.id_col,
    )

    is_regression: bool | None = None
    if dataset.target_col is not None:
        is_regression = bool(infer_is_regression_target(df[dataset.target_col]))

    def emit_estimate_progress(message: str) -> None:
        _emit_progress(
            stage="tuning",
            message=message,
            progress_stream=progress_stream,
            progress_format=progress_format,
            dataset_id=dataset_id,
            encoding_method=encoding_method,
        )

    emit_estimate_progress("estimating representative ctgan runtime")
    result = estimate_ctgan_runtime(
        df=df,
        schema=schema,
        dataset=dataset.label,
        encoding_method=encoding_method,
        task_type=_task_type_from_flag(is_regression),
        output_dir=output_dir,
        device=device,
        sample_epochs=estimate_sample_epochs,
        projected_epochs=estimate_total_epochs,
        projected_total_runs=estimate_total_runs,
        progress_callback=emit_estimate_progress,
    )

    _emit_progress(
        stage="saving",
        message="writing estimate summary",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )
    return result


def run_full_ctgan_experiment(
    *,
    manifest_path: Path | str,
    dataset_id: str,
    dataset_label: str,
    encoding_method: str,
    output_root: Path | str = Path("experiments/results"),
    progress_stream: TextIO | None = None,
    progress_format: str = "jsonl",
    best_params_file: Path | str | None = None,
    skip_tuning: bool = False,
    device: str = "cuda",
    poster_fast: bool = False,
    max_rows: int | None = None,
) -> FullCtganExperimentResult:
    manifest_path = Path(manifest_path).resolve()
    project_root = _infer_project_root(manifest_path)
    run_dir = _ensure_dir(Path(output_root) / "ctgan" / dataset_id / encoding_method)

    if skip_tuning and best_params_file is None:
        raise ValueError("skip_tuning requires best_params_file.")

    _emit_progress(
        stage="launching",
        message="loading manifest and dataset",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )

    manifest = load_ctgan_manifest(manifest_path, project_root=project_root)
    dataset = manifest.resolve_dataset_label(dataset_label)
    encoding = _resolve_manifest_encoding(manifest, encoding_method=encoding_method)
    if dataset.dataset_id != dataset_id:
        raise ValueError(
            f"dataset_id mismatch for label {dataset_label!r}: "
            f"expected {dataset.dataset_id!r}, got {dataset_id!r}"
        )
    if encoding.encoding_id != encoding_method:
        raise ValueError(
            f"encoding_method mismatch: expected {encoding.encoding_id!r}, got {encoding_method!r}"
        )

    effective_max_rows = 10_000 if poster_fast and max_rows is None else max_rows
    df = pd.read_csv(project_root / "datasets" / "raw" / f"{dataset_id}.csv")
    df = _cap_dataframe_rows(df, max_rows=effective_max_rows if poster_fast else None)
    schema = TabularSchema.infer_from_dataframe(
        df,
        target_col=dataset.target_col,
        id_col=dataset.id_col,
    )

    is_regression: bool | None = None
    if dataset.target_col is not None:
        is_regression = bool(infer_is_regression_target(df[dataset.target_col]))

    tuning_output_dir = run_dir / "tuning"
    if best_params_file is not None:
        _emit_progress(
            stage="launching",
            message="loading provided best params",
            progress_stream=progress_stream,
            progress_format=progress_format,
            dataset_id=dataset_id,
            encoding_method=encoding_method,
        )
        tuning_result = {
            "best_params": _load_best_params_file(best_params_file),
            "best_value": None,
            "best_source": "provided_file",
        }
    else:
        def emit_tuning_progress(message: str) -> None:
            _emit_progress(
                stage="tuning",
                message=message,
                progress_stream=progress_stream,
                progress_format=progress_format,
                dataset_id=dataset_id,
                encoding_method=encoding_method,
            )

        _emit_progress(
            stage="tuning",
            message="tuning ctgan",
            progress_stream=progress_stream,
            progress_format=progress_format,
            dataset_id=dataset_id,
            encoding_method=encoding_method,
        )
        tuning_result = select_ctgan_best_params(
            df=df,
            schema=schema,
            dataset=dataset.label,
            encoding_method=encoding_method,
            task_type=_task_type_from_flag(is_regression),
            output_dir=tuning_output_dir,
            device=device,
            progress_callback=emit_tuning_progress,
        )

    _emit_progress(
        stage="crossval",
        message="running poster-fast holdout" if poster_fast else "running cross-validation",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )
    per_fold_dir = _ensure_dir(run_dir / "crossval" / "per_fold")
    artifacts_dir = _ensure_dir(run_dir / "artifacts")
    fold_results: list[dict[str, Any]] = []
    if poster_fast:
        fold_payload = _run_holdout_split(
            df=df,
            schema=schema,
            encoding_method=encoding_method,
            best_params=dict(tuning_result["best_params"]),
            device=device,
            is_regression=is_regression,
            artifacts_dir=artifacts_dir,
        )
        fold_results.append(fold_payload)
        _save_json(per_fold_dir / "fold_0.json", fold_payload)
        n_folds = 1
    else:
        for fold_id in range(5):
            fold_payload = _run_crossval_fold(
                fold_id=fold_id,
                df=df,
                schema=schema,
                encoding_method=encoding_method,
                best_params=dict(tuning_result["best_params"]),
                device=device,
                is_regression=is_regression,
                artifacts_dir=artifacts_dir,
            )
            fold_results.append(fold_payload)
            _save_json(per_fold_dir / f"fold_{fold_id}.json", fold_payload)
        n_folds = 5

    _emit_progress(
        stage="metrics",
        message="aggregating metrics",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )
    if schema.target_col is None:
        _emit_progress(
            stage="metrics",
            message="skipping tstr utility: no target column in schema",
            progress_stream=progress_stream,
            progress_format=progress_format,
            dataset_id=dataset_id,
            encoding_method=encoding_method,
        )
    distribution_records = [fold["distribution"] for fold in fold_results]
    utility_records = [
        {key: value for key, value in fold["utility"].items() if key not in {"status", "task_type"}}
        for fold in fold_results
        if fold["utility"].get("status") == "ok"
    ]
    utility_status = fold_results[0]["utility"]["status"] if fold_results else "unsupported_no_target"
    utility_task_type = (
        fold_results[0]["utility"].get("task_type")
        if fold_results and fold_results[0]["utility"].get("status") == "ok"
        else None
    )
    aggregate_payload = {
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "encoding_method": encoding_method,
        "n_folds": n_folds,
        "distribution": _aggregate_numeric_records(distribution_records),
        "tstr": {
            "status": utility_status,
            "task_type": utility_task_type,
            "metrics": _aggregate_numeric_records(utility_records),
        },
    }
    aggregate_metrics_path = run_dir / "metrics" / "aggregate.json"
    _save_json(aggregate_metrics_path, aggregate_payload)

    _emit_progress(
        stage="saving",
        message="writing run summary",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )
    summary_path = run_dir / "run_summary.json"
    _save_json(
        summary_path,
        {
            "dataset_id": dataset_id,
            "dataset_label": dataset.label,
            "encoding_label": encoding.label,
            "encoding_method": encoding_method,
            "output_dir": run_dir,
            "schema": {
                "continuous_cols": schema.continuous_cols,
                "discrete_cols": schema.discrete_cols,
                "categorical_cols": schema.categorical_cols,
                "target_col": schema.target_col,
                "id_col": schema.id_col,
            },
            "poster_fast": {
                "enabled": poster_fast,
                "max_rows": effective_max_rows if poster_fast else None,
                "effective_rows": int(len(df)),
            },
            "tuning": {
                "output_dir": tuning_output_dir,
                "summary_path": tuning_output_dir / "summary.json",
                "best_params": dict(tuning_result["best_params"]),
                "best_value": tuning_result["best_value"],
                "best_source": tuning_result["best_source"],
                "best_params_file": str(Path(best_params_file).resolve()) if best_params_file is not None else None,
            },
            "crossval": {
                "n_folds": n_folds,
                "per_fold_dir": per_fold_dir,
            },
            "metrics_path": aggregate_metrics_path,
            "tstr": aggregate_payload["tstr"],
        },
    )

    # ------------------------------------------------------------------
    # Optional: upload artifacts to Google Drive and record metrics in
    # the Results sheet.  Silently skipped if Drive env vars are absent.
    # ------------------------------------------------------------------
    _maybe_upload_to_drive(
        run_dir=run_dir,
        aggregate_metrics_path=aggregate_metrics_path,
        dataset_id=dataset_id,
        dataset_label=dataset_label,
        encoding_method=encoding_method,
        encoding_label=encoding.label,
        progress_stream=progress_stream,
        progress_format=progress_format,
    )

    return FullCtganExperimentResult(
        output_dir=run_dir,
        summary_path=summary_path,
        aggregate_metrics_path=aggregate_metrics_path,
    )


def _load_dotenv() -> None:
    """Load .env from the project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            return


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Run the full CTGAN experiment pipeline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--encoding-method", required=True)
    parser.add_argument("--output-root", default=str(Path("experiments/results")))
    parser.add_argument("--progress-format", default="jsonl")
    parser.add_argument("--best-params-file")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--poster-fast", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--estimate-runtime", action="store_true")
    parser.add_argument("--estimate-sample-epochs", type=int, default=10)
    parser.add_argument("--estimate-total-epochs", type=int, default=DEFAULT_CTGAN_EPOCHS)
    parser.add_argument("--estimate-total-runs", type=int, default=35)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    common_kwargs = {
        "manifest_path": args.manifest,
        "dataset_id": args.dataset_id,
        "dataset_label": args.dataset_label,
        "encoding_method": args.encoding_method,
        "output_root": args.output_root,
        "progress_stream": sys.stdout,
        "progress_format": args.progress_format,
        "device": args.device,
        "max_rows": args.max_rows,
    }
    if args.estimate_runtime:
        run_ctgan_runtime_estimate(
            **common_kwargs,
            estimate_sample_epochs=args.estimate_sample_epochs,
            estimate_total_epochs=args.estimate_total_epochs,
            estimate_total_runs=args.estimate_total_runs,
        )
    else:
        run_full_ctgan_experiment(
            **common_kwargs,
            best_params_file=Path(args.best_params_file).resolve() if args.best_params_file else None,
            skip_tuning=args.skip_tuning,
            poster_fast=args.poster_fast,
        )
    return 0


def _maybe_upload_to_drive(
    *,
    run_dir: Path,
    aggregate_metrics_path: Path,
    dataset_id: str,
    dataset_label: str,
    encoding_method: str,
    encoding_label: str,
    progress_stream: Any,
    progress_format: str,
) -> None:
    """Upload artifacts to Google Drive and write a row in the Results sheet.

    Silently no-ops when CATREPBENCH_GDRIVE_RESULTS_FOLDER_ID is not set,
    so existing pipelines without Drive configuration are unaffected.
    """
    try:
        from experiments.ctgan.orchestrator_staff.ctgan_drive import (
            DriveConfig,
            DriveClient,
            upload_experiment_artifacts,
            write_results_row,
        )
        from experiments.ctgan.orchestrator_staff.ctgan_sheets import SheetsConfig
    except ImportError:
        return

    if not DriveConfig.is_configured():
        return

    _emit_progress(
        stage="saving",
        message="uploading artifacts to Google Drive",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )

    try:
        drive_config = DriveConfig.from_env()
        drive_client = DriveClient(drive_config)

        folder_url = upload_experiment_artifacts(
            drive_client=drive_client,
            run_dir=run_dir,
            model_name="CTGAN",
            dataset_id=dataset_id,
            encoding_method=encoding_method,
        )

        sheets_config = SheetsConfig.from_env()
        write_results_row(
            sheets_config=sheets_config,
            aggregate_metrics_path=aggregate_metrics_path,
            model_name="CTGAN",
            dataset_label=dataset_label,
            encoding_label=encoding_label,
            folder_url=folder_url,
            results_worksheet_name=os.getenv(
                "CATREPBENCH_GDRIVE_RESULTS_WORKSHEET", "Results"
            ),
        )

        _emit_progress(
            stage="saving",
            message=f"Drive upload complete: {folder_url}",
            progress_stream=progress_stream,
            progress_format=progress_format,
            dataset_id=dataset_id,
            encoding_method=encoding_method,
        )
    except Exception as exc:  # noqa: BLE001
        # Drive upload is best-effort — do not crash the experiment
        print(f"[ctgan_drive] WARNING: Drive upload failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
