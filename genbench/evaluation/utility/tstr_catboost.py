from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseEvaluator, EvaluationResult
from genbench.transforms.target import infer_is_regression_target


def _to_1d_prediction_series(preds: object, *, dtype: object | None = None) -> pd.Series:
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"Predictions must be 1-dimensional, got shape {arr.shape}.")
    return pd.Series(arr).astype(dtype if dtype is not None else None)


def _prepare_features(df: pd.DataFrame, schema: TabularSchema) -> pd.DataFrame:
    missing = [c for c in schema.all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    cols = schema.feature_cols + [
        schema.target_col] if schema.target_col else schema.feature_cols
    return df[cols].copy()


@dataclass
class TSTRCatBoostEvaluator(BaseEvaluator):
    """
    Train-on-Synthetic Test-on-Real utility evaluator with CatBoostRegressor.
    """

    name: str = "tstr_catboost"
    random_seed: int = 42
    task_type: Optional[str] = None

    def evaluate(
            self,
            train_real: pd.DataFrame,
            test_real: pd.DataFrame,
            synth_train: pd.DataFrame,
            schema: TabularSchema,
    ) -> EvaluationResult:
        scores = tstr_catboost(
            train_real=train_real,
            test_real=test_real,
            synth_train=synth_train,
            schema=schema,
            random_seed=self.random_seed,
            task_type=self.task_type,
        )
        return EvaluationResult(scores=scores, meta={"evaluator": self.name,
                                                     "random_seed": self.random_seed,
                                                     "task_type": scores.get("task_type")})


def tstr_catboost(
        train_real: pd.DataFrame,
        test_real: pd.DataFrame,
        synth_train: pd.DataFrame,
        schema: TabularSchema,
        random_seed: int = 42,
        task_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train-on-Synthetic Test-on-Real (TSTR) utility using CatBoostRegressor.
    Returns metrics and percentage deviations vs real baseline.
    Regression uses R2. Classification uses weighted F1.
    """
    if schema.target_col is None:
        raise ValueError("schema.target_col must be set for TSTR evaluation.")

    train_real = _prepare_features(train_real, schema)
    test_real = _prepare_features(test_real, schema)
    synth_train = _prepare_features(synth_train, schema)

    feature_cols = schema.feature_cols
    target_col = schema.target_col
    cat_features = [i for i, c in enumerate(feature_cols) if
                    c in schema.categorical_cols]
    resolved_task_type = task_type.strip().lower() if task_type is not None else None
    if resolved_task_type is not None and resolved_task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")
    if resolved_task_type is None:
        resolved_task_type = (
            "regression"
            if infer_is_regression_target(train_real[target_col])
            else "classification"
        )

    def _fit_and_eval_regression(train_df: pd.DataFrame) -> Dict[str, float]:
        train_pool = Pool(train_df[feature_cols], train_df[target_col],
                          cat_features=cat_features)
        test_pool = Pool(test_real[feature_cols], test_real[target_col],
                         cat_features=cat_features)
        model = CatBoostRegressor(random_seed=random_seed, verbose=False)
        model.fit(train_pool)
        preds = model.predict(test_pool)
        return {
            "r2": float(r2_score(test_real[target_col], preds)),
        }

    def _pct_diff(real_val: float, synth_val: float) -> float:
        if real_val == 0:
            return float("inf")
        return float(abs(real_val - synth_val) / abs(real_val) * 100.0)

    def _fit_and_eval_classification(train_df: pd.DataFrame) -> Dict[str, float]:
        train_pool = Pool(train_df[feature_cols], train_df[target_col], cat_features=cat_features)
        test_pool = Pool(test_real[feature_cols], test_real[target_col], cat_features=cat_features)
        model = CatBoostClassifier(random_seed=random_seed, verbose=False)
        model.fit(train_pool)
        preds = model.predict(test_pool)
        pred_series = _to_1d_prediction_series(
            preds,
            dtype=test_real[target_col].dtype if hasattr(test_real[target_col], "dtype") else None,
        )
        return {
            "f1_weighted": float(f1_score(test_real[target_col], pred_series, average="weighted")),
        }

    if resolved_task_type == "regression":
        real_scores = _fit_and_eval_regression(train_real)
        synth_scores = _fit_and_eval_regression(synth_train)
        return {
            "task_type": "regression",
            "r2_real": real_scores["r2"],
            "r2_synth": synth_scores["r2"],
            "r2_pct_diff": _pct_diff(real_scores["r2"], synth_scores["r2"]),
        }

    real_scores = _fit_and_eval_classification(train_real)
    synth_scores = _fit_and_eval_classification(synth_train)
    return {
        "task_type": "classification",
        "f1_weighted_real": real_scores["f1_weighted"],
        "f1_weighted_synth": synth_scores["f1_weighted"],
        "f1_weighted_pct_diff": _pct_diff(real_scores["f1_weighted"], synth_scores["f1_weighted"]),
    }
