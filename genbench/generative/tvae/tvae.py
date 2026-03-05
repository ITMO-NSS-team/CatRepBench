from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.generative.base import BaseGenerative, GenerativeState

from ctgan import TVAE  # type: ignore



@dataclass
class TvaeGenerative(BaseGenerative):
    """
    Thin wrapper around TVAE to comply with BaseGenerative protocol.
    """

    name: str = "tvae"
    discrete_cols: Optional[List[str]] = None
    tvae_kwargs: Dict[str, Any] = field(default_factory=dict)

    # fitted artifacts
    model_: Any = None
    fitted_: bool = False
    used_discrete_cols_: List[str] = field(default_factory=list)

    def requires_fit(self) -> bool:
        return True

    def is_conditional(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "TvaeGenerative":
        if TVAE is None:
            raise ImportError("ctgan package with TVAE is required for TvaeGenerative.")

        if self.discrete_cols is None:
            candidate = list(schema.categorical_cols) + list(schema.discrete_cols)
            self.used_discrete_cols_ = [c for c in candidate if c in df.columns]
        else:
            self.used_discrete_cols_ = [c for c in self.discrete_cols if c in df.columns]

        self.model_ = TVAE(**self.tvae_kwargs)
        self.model_.fit(df, discrete_columns=self.used_discrete_cols_)
        self.fitted_ = True
        return self

    def sample(self, n: int, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if conditions is not None:
            raise NotImplementedError("TvaeGenerative does not expose conditional sampling.")
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.model_.sample(n)

    def get_loss_history(self) -> Optional[Dict[str, list[float]]]:
        if not self.fitted_ or self.model_ is None:
            return None
        loss_df = getattr(self.model_, "loss_values", None)
        if loss_df is None:
            return None
        try:
            # TVAE typically stores a DataFrame with columns ['Epoch', 'Loss'].
            if "Loss" in loss_df.columns:
                return {"loss": loss_df["Loss"].astype(float).tolist()}
            return None
        except Exception:
            return None

    def get_state(self) -> GenerativeState:
        return GenerativeState(
            name=self.name,
            params={
                "discrete_cols": self.discrete_cols,
                "tvae_kwargs": self.tvae_kwargs,
            },
        )

    @classmethod
    def from_state(cls, state: GenerativeState) -> "TvaeGenerative":
        params = state.params or {}
        return cls(
            discrete_cols=params.get("discrete_cols"),
            tvae_kwargs=params.get("tvae_kwargs", {}),
        )

    def save_artifacts(self, path: Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Nothing to save: model is not fitted.")
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tvae.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": self.model_,
                    "used_discrete_cols": self.used_discrete_cols_,
                    "fitted": self.fitted_,
                },
                f,
            )

    @classmethod
    def load_artifacts(cls, path: Path) -> "TvaeGenerative":
        path = path.resolve()
        bundle_path = path / "tvae.pkl"
        if not bundle_path.exists():
            raise FileNotFoundError(f"tvae.pkl not found in {path}")
        with open(bundle_path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj.model_ = payload.get("model")
        obj.used_discrete_cols_ = payload.get("used_discrete_cols", [])
        obj.fitted_ = bool(payload.get("fitted", obj.model_ is not None))
        return obj
