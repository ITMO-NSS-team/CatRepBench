from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tabpfgen import TabPFGen

from genbench.data.schema import TabularSchema
from genbench.generative.base import BaseGenerative, GenerativeState


@dataclass
class TabPFGenGenerative(BaseGenerative):
    """
    Thin wrapper around TabPFGen to comply with BaseGenerative protocol.
    """

    name: str = "tabpfgen"
    n_sgld_steps: int = 500
    sgld_step_size: float = 0.01
    sgld_noise_scale: float = 0.01
    device: str = "auto"
    balance_classes: bool = False
    use_quantiles: bool = True

    # fitted artifacts
    model_: Optional[TabPFGen] = None
    fitted_: bool = False
    task_type_: Optional[str] = None

    def requires_fit(self) -> bool:
        return True

    def is_conditional(self) -> bool:
        return False

    def fit(
            self, df: pd.DataFrame, schema: TabularSchema,
            source_schema: Optional[TabularSchema] = None
    ) -> "TabPFGenGenerative":
        if source_schema is None:
            source_schema = schema

        target_col = schema.target_col
        if target_col is None:
            raise ValueError("TabPFGen requires a target column.")

        # Separate features and target
        feature_cols = schema.feature_cols
        X = df[feature_cols].values
        y = df[target_col].values

        # Infer the task type
        self.task_type_ = _infer_task_type(y, schema, target_col)

        self.model_ = TabPFGen(
            n_sgld_steps=self.n_sgld_steps,
            sgld_step_size=self.sgld_step_size,
            sgld_noise_scale=self.sgld_noise_scale,
            device=self.device,
        )

        # Store data for later generation
        self._train_X = X
        self._train_y = y
        self._feature_cols = feature_cols
        self._target_col = target_col
        self.fitted_ = True
        return self

    def sample(self, n: int,
               conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if conditions is not None:
            raise NotImplementedError("Conditional sampling is not supported.")
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if self.task_type_ == "classification":
            X_synth, y_synth = self.model_.generate_classification(
                self._train_X, self._train_y, n_samples=n,
                balance_classes=self.balance_classes
            )
        elif self.task_type_ == "regression":
            X_synth, y_synth = self.model_.generate_regression(
                self._train_X, self._train_y, n_samples=n,
                use_quantiles=self.use_quantiles
            )
        else:
            raise RuntimeError(f"Unknown task type: {self.task_type_}")

        # Build a DataFrame
        synth_data = np.column_stack([X_synth, y_synth])
        return pd.DataFrame(synth_data,
                            columns=self._feature_cols + [self._target_col])

    def get_loss_history(self) -> Optional[Dict[str, list[float]]]:
        return None

    def get_state(self) -> GenerativeState:
        return GenerativeState(
            name=self.name,
            params={
                "n_sgld_steps": self.n_sgld_steps,
                "sgld_step_size": self.sgld_step_size,
                "sgld_noise_scale": self.sgld_noise_scale,
                "device": self.device,
                "balance_classes": self.balance_classes,
                "use_quantiles": self.use_quantiles,
            },
        )

    @classmethod
    def from_state(cls, state: GenerativeState) -> "TabPFGenGenerative":
        params = state.params or {}
        return cls(
            n_sgld_steps=params.get("n_sgld_steps", 500),
            sgld_step_size=params.get("sgld_step_size", 0.01),
            sgld_noise_scale=params.get("sgld_noise_scale", 0.01),
            device=params.get("device", "auto"),
            balance_classes=params.get("balance_classes", False),
            use_quantiles=params.get("use_quantiles", True),
        )

    def save_artifacts(self, path: Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Nothing to save: model is not fitted.")
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tabpfgen.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": self.model_,
                    "task_type": self.task_type_,
                    "train_X": self._train_X,
                    "train_y": self._train_y,
                    "feature_cols": self._feature_cols,
                    "target_col": self._target_col,
                    "fitted": self.fitted_,
                },
                f,
            )

    @classmethod
    def load_artifacts(cls, path: Path) -> "TabPFGenGenerative":
        path = path.resolve()
        bundle_path = path / "tabpfgen.pkl"
        if not bundle_path.exists():
            raise FileNotFoundError(f"tabpfgen.pkl not found in {path}")
        with open(bundle_path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj.model_ = payload.get("model")
        obj.task_type_ = payload.get("task_type")
        obj._train_X = payload.get("train_X")
        obj._train_y = payload.get("train_y")
        obj._feature_cols = payload.get("feature_cols", [])
        obj._target_col = payload.get("target_col")
        obj.fitted_ = bool(payload.get("fitted", False))
        return obj


def _infer_task_type(y: np.ndarray, schema: TabularSchema,
                     target_col: str) -> str:
    """Determine classification or regression based on schema and data."""
    if target_col in schema.categorical_cols:
        return "classification"
    if target_col in schema.discrete_cols:
        # If few unique values, treat as classification
        return "classification" if len(np.unique(y)) <= 20 else "regression"
    return "regression"
