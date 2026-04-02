from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ctgan_common import (
    build_ctgan_kwargs,
    build_preprocess_pipeline,
    default_discrete_cols,
)
from experiments.ctgan_manifest import load_ctgan_manifest
from experiments.ctgan_tuning import tune_ctgan
from genbench.data.datamodule import TabularDataModule
from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigKFold
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


def _resolve_manifest_encoding(manifest: Any, *, encoding_method: str) -> Any:
    for entry in manifest.encodings:
        if getattr(entry, "encoding_id", None) == encoding_method:
            return manifest.resolve_encoding_label(entry.label)

    available = [getattr(entry, "encoding_id", None) for entry in getattr(manifest, "encodings", ())]
    raise ValueError(
        f"encoding_method mismatch: {encoding_method!r} is not declared in manifest encodings {available!r}"
    )


def _infer_project_root(manifest_path: Path) -> Path:
    candidates = [manifest_path.parent, manifest_path.parent.parent]
    for candidate in candidates:
        if (candidate / "datasets" / "raw").exists() or (candidate / "genbench").exists():
            return candidate
    if manifest_path.parent.name == "experiments":
        return manifest_path.parent.parent
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


def _prepare_fold_data(
    *,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    is_regression: bool | None,
    split_cfg: SplitConfigKFold,
    fold_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, TabularSchema]:
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
    dm = TabularDataModule(df=df, schema=schema, transforms=pipeline, validate=True)
    dm.prepare_kfold(split_cfg)
    fold = dm.get_fold(fold_id)
    train_df = fold.train.reset_index(drop=True)
    test_df = fold.test.reset_index(drop=True)
    transformed_schema = TabularSchema.infer_from_dataframe(
        train_df,
        target_col=schema.target_col,
        id_col=schema.id_col,
    )
    transformed_schema.validate(test_df)
    return train_df, test_df, transformed_schema


def _run_crossval_fold(
    *,
    fold_id: int,
    df: pd.DataFrame,
    schema: TabularSchema,
    encoding_method: str,
    best_params: dict[str, Any],
    device: str,
    is_regression: bool | None,
) -> dict[str, Any]:
    split_cfg = SplitConfigKFold(n_splits=5, shuffle=True, random_seed=42)
    train_df, test_df, transformed_schema = _prepare_fold_data(
        df=df,
        schema=schema,
        encoding_method=encoding_method,
        is_regression=is_regression,
        split_cfg=split_cfg,
        fold_id=fold_id,
    )
    ctgan_kwargs = build_ctgan_kwargs(best_params, epochs=50, device=device)
    discrete_cols = default_discrete_cols(
        schema=transformed_schema,
        train_df=train_df,
        include_target_for_classification=(is_regression is False),
    )
    model = CtganGenerative(discrete_cols=discrete_cols, ctgan_kwargs=ctgan_kwargs)
    model.fit(train_df, transformed_schema)
    synth_df = model.sample(len(train_df)).reset_index(drop=True)

    dist_pipeline = DistributionEvaluationPipeline(metrics=[WassersteinDistanceMetric()])
    distribution_scores = {
        key: float(value)
        for key, value in dist_pipeline.evaluate(
            real=train_df,
            synth=synth_df,
            schema=transformed_schema,
        ).scores.items()
    }

    if schema.target_col is None:
        tstr_scores: dict[str, Any] = {"status": "unsupported_no_target"}
    elif is_regression is False:
        tstr_scores = {"status": "unsupported_classification"}
    else:
        tstr_scores = {"status": "ok"}
        tstr_scores.update(
            {
                key: float(value)
                for key, value in tstr_catboost(
                    train_real=train_df,
                    test_real=test_df,
                    synth_train=synth_df,
                    schema=transformed_schema,
                ).items()
            }
        )

    return {
        "fold_id": int(fold_id),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "distribution": distribution_scores,
        "tstr": tstr_scores,
    }


def run_full_ctgan_experiment(
    *,
    manifest_path: Path | str,
    dataset_id: str,
    dataset_label: str,
    encoding_method: str,
    output_root: Path | str = Path("experiments/results"),
    progress_stream: TextIO | None = None,
    progress_format: str = "jsonl",
    device: str = "cuda",
) -> FullCtganExperimentResult:
    manifest_path = Path(manifest_path).resolve()
    project_root = _infer_project_root(manifest_path)
    run_dir = _ensure_dir(Path(output_root) / "ctgan" / dataset_id / encoding_method)

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
    schema = TabularSchema.infer_from_dataframe(
        df,
        target_col=dataset.target_col,
        id_col=dataset.id_col,
    )

    is_regression: bool | None = None
    if dataset.target_col is not None:
        is_regression = bool(infer_is_regression_target(df[dataset.target_col]))

    _emit_progress(
        stage="tuning",
        message="tuning ctgan",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )
    tuning_result = tune_ctgan(
        df=df,
        schema=schema,
        dataset=dataset.label,
        encoding_method=encoding_method,
        output_dir=run_dir / "tuning",
        device=device,
    )

    _emit_progress(
        stage="crossval",
        message="running cross-validation",
        progress_stream=progress_stream,
        progress_format=progress_format,
        dataset_id=dataset_id,
        encoding_method=encoding_method,
    )
    per_fold_dir = _ensure_dir(run_dir / "crossval" / "per_fold")
    fold_results: list[dict[str, Any]] = []
    for fold_id in range(5):
        fold_payload = _run_crossval_fold(
            fold_id=fold_id,
            df=df,
            schema=schema,
            encoding_method=encoding_method,
            best_params=dict(tuning_result.best_params),
            device=device,
            is_regression=is_regression,
        )
        fold_results.append(fold_payload)
        _save_json(per_fold_dir / f"fold_{fold_id}.json", fold_payload)

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
    elif is_regression is False:
        _emit_progress(
            stage="metrics",
            message="skipping tstr utility: classification target unsupported",
            progress_stream=progress_stream,
            progress_format=progress_format,
            dataset_id=dataset_id,
            encoding_method=encoding_method,
        )
    distribution_records = [fold["distribution"] for fold in fold_results]
    tstr_records = [
        {key: value for key, value in fold["tstr"].items() if key != "status"}
        for fold in fold_results
        if fold["tstr"].get("status") == "ok"
    ]
    tstr_status = fold_results[0]["tstr"]["status"] if fold_results else "unsupported_no_target"
    aggregate_payload = {
        "dataset_id": dataset_id,
        "dataset_label": dataset_label,
        "encoding_method": encoding_method,
        "n_folds": 5,
        "distribution": _aggregate_numeric_records(distribution_records),
        "tstr": {"status": tstr_status, "metrics": _aggregate_numeric_records(tstr_records)},
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
            "tuning": {
                "output_dir": tuning_result.output_dir,
                "best_params": dict(tuning_result.best_params),
            },
            "crossval": {
                "n_folds": 5,
                "per_fold_dir": per_fold_dir,
            },
            "metrics_path": aggregate_metrics_path,
            "tstr": aggregate_payload["tstr"],
        },
    )

    return FullCtganExperimentResult(
        output_dir=run_dir,
        summary_path=summary_path,
        aggregate_metrics_path=aggregate_metrics_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full CTGAN experiment pipeline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--encoding-method", required=True)
    parser.add_argument("--output-root", default=str(Path("experiments/results")))
    parser.add_argument("--progress-format", default="jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    run_full_ctgan_experiment(
        manifest_path=args.manifest,
        dataset_id=args.dataset_id,
        dataset_label=args.dataset_label,
        encoding_method=args.encoding_method,
        output_root=args.output_root,
        progress_stream=sys.stdout,
        progress_format=args.progress_format,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
