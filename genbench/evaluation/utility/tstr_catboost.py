from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from catboost import CatBoostRegressor, Pool

from genbench.data.schema import TabularSchema
from genbench.evaluation.base import BaseEvaluator, EvaluationResult


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
        )
        return EvaluationResult(scores=scores, meta={"evaluator": self.name,
                                                     "random_seed": self.random_seed})


def tstr_catboost(
        train_real: pd.DataFrame,
        test_real: pd.DataFrame,
        synth_train: pd.DataFrame,
        schema: TabularSchema,
        random_seed: int = 42,
) -> Dict[str, float]:
    """
    Train-on-Synthetic Test-on-Real (TSTR) utility using CatBoostRegressor.
    Returns metrics and percentage deviations vs real baseline.
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

    def _fit_and_eval(train_df: pd.DataFrame) -> Dict[str, float]:
        train_pool = Pool(train_df[feature_cols], train_df[target_col],
                          cat_features=cat_features)
        test_pool = Pool(test_real[feature_cols], test_real[target_col],
                         cat_features=cat_features)
        model = CatBoostRegressor(random_seed=random_seed, verbose=False)
        model.fit(train_pool)
        preds = model.predict(test_pool)
        return {
            "r2": float(r2_score(test_real[target_col], preds)),
            "mape": float(
                mean_absolute_percentage_error(test_real[target_col], preds)),
        }

    real_scores = _fit_and_eval(train_real)
    synth_scores = _fit_and_eval(synth_train)

    def _pct_diff(real_val: float, synth_val: float) -> float:
        if real_val == 0:
            return float("inf")
        return float(abs(real_val - synth_val) / abs(real_val) * 100.0)

    return {
        "r2_real": real_scores["r2"],
        "mape_real": real_scores["mape"],
        "r2_synth": synth_scores["r2"],
        "mape_synth": synth_scores["mape"],
        "r2_pct_diff": _pct_diff(real_scores["r2"], synth_scores["r2"]),
        "mape_pct_diff": _pct_diff(real_scores["mape"], synth_scores["mape"]),
    }
