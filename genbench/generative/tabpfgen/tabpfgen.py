from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.generative.base import BaseGenerative, GenerativeState


@dataclass
class TabPFGenGenerative(BaseGenerative):
    """
    Thin wrapper around sebhaan/TabPFGen to comply with BaseGenerative.

    TabPFGen runs SGLD on top of a pre-trained TabPFN — no per-dataset
    training. Hyperparameters control the sampler only.
    """

    name: str = "tabpfgen"
    n_sgld_steps: int = 1000
    sgld_step_size: float = 0.01
    sgld_noise_scale: float = 0.01
    device: str = "auto"
    balance_classes: bool = True
    use_quantiles: bool = True
    seed: Optional[int] = None

    # fitted artifacts
    model_: Any = None
    fitted_: bool = False
    task_type_: Optional[str] = None
    columns_: List[str] = field(default_factory=list)

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

        feature_cols = [c for c in schema.feature_cols if c != target_col]
        if not feature_cols:
            raise ValueError("No feature columns after removing target.")

        self.columns_ = list(df.columns)
        self._feature_cols = feature_cols
        self._target_col = target_col

        X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.task_type_ = _infer_task_type(
            df[target_col].to_numpy(), schema, target_col
        )

        if self.task_type_ == "regression":
            y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(
                dtype=np.float32
            )
            if np.isnan(y).any():
                raise ValueError("Regression target contains NaNs.")
        else:
            y = df[target_col].to_numpy(copy=True)

        try:
            from tabpfgen import TabPFGen
        except ImportError as exc:
            raise ImportError(
                "tabpfgen is required. Install via: pip install tabpfgen"
            ) from exc

        self.model_ = TabPFGen(
            n_sgld_steps=int(self.n_sgld_steps),
            sgld_step_size=float(self.sgld_step_size),
            sgld_noise_scale=float(self.sgld_noise_scale),
            device=str(self.device),
        )

        self._train_X = X
        self._train_y = y
        self.fitted_ = True
        return self

    def sample(self, n: int,
               conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if conditions is not None:
            raise NotImplementedError("Conditional sampling is not supported.")
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if n <= 0:
            raise ValueError("n must be positive.")

        if self.seed is not None:
            np.random.seed(int(self.seed))

        if self.task_type_ == "classification":
            X_synth, y_synth = self.model_.generate_classification(
                self._train_X, self._train_y, n_samples=int(n),
                balance_classes=bool(self.balance_classes),
            )
        elif self.task_type_ == "regression":
            X_synth, y_synth = self.model_.generate_regression(
                self._train_X, self._train_y, n_samples=int(n),
                use_quantiles=bool(self.use_quantiles),
            )
        else:
            raise RuntimeError(f"Unknown task type: {self.task_type_}")

        X_synth = np.asarray(X_synth)
        y_synth = np.asarray(y_synth)

        out = pd.DataFrame(index=range(len(X_synth)), columns=self.columns_)
        for j, c in enumerate(self._feature_cols):
            out[c] = X_synth[:, j]
        out[self._target_col] = y_synth
        return out[self.columns_]

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
                "seed": self.seed,
            },
        )

    @classmethod
    def from_state(cls, state: GenerativeState) -> "TabPFGenGenerative":
        params = state.params or {}
        return cls(
            n_sgld_steps=params.get("n_sgld_steps", 1000),
            sgld_step_size=params.get("sgld_step_size", 0.01),
            sgld_noise_scale=params.get("sgld_noise_scale", 0.01),
            device=params.get("device", "auto"),
            balance_classes=params.get("balance_classes", True),
            use_quantiles=params.get("use_quantiles", True),
            seed=params.get("seed", None),
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
                    "columns": self.columns_,
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
        obj.columns_ = payload.get("columns", [])
        obj.fitted_ = bool(payload.get("fitted", False))
        return obj


def _infer_task_type(y: np.ndarray, schema: TabularSchema,
                     target_col: str) -> str:
    """Determine classification or regression based on schema and data."""
    if target_col in schema.categorical_cols:
        return "classification"
    if target_col in schema.discrete_cols:
        return "classification" if len(np.unique(y)) <= 20 else "regression"
    return "regression"
