from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState


def _safe_str(x: object) -> str:
    """Stable string for categories (handles NaN/None)."""
    if x is None:
        return "__NONE__"
    if isinstance(x, float) and pd.isna(x):
        return "__NAN__"
    return str(x)


def _gumbel_softmax_sample(
        logits: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from Gumbel-Softmax distribution.

    Parameters
    ----------
    logits : np.ndarray
        Log probabilities for each category (shape: [n_categories])
    temperature : float
        Temperature parameter. Lower = harder (closer to one-hot),
        higher = softer (closer to uniform).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Soft one-hot vector (shape: [n_categories])
    """
    # Sample Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0, 1)
    u = rng.uniform(0, 1, size=logits.shape)
    u = np.clip(u, 1e-10, 1 - 1e-10)  # Avoid log(0)
    gumbel_noise = -np.log(-np.log(u))

    scaled = (logits + gumbel_noise) / temperature

    # Numerically stable softmax
    scaled_max = np.max(scaled)
    exp_scaled = np.exp(scaled - scaled_max)

    return exp_scaled / np.sum(exp_scaled)


@dataclass
class GumbelSoftmaxRepresentation:
    """
    Gumbel-Softmax representation for categorical columns.

    A differentiable approximation to one-hot encoding that allows gradients
    to flow through categorical choices.

    Output:
      - Keeps continuous + discrete columns as-is.
      - Replaces each categorical column with soft one-hot columns:
        "{col}__gs_0", "{col}__gs_1", etc.
      - Values are floats in [0, 1] that sum to 1 per column group.

    Options:
      - temperature: Softmax temperature (default: 1.0)
          - Lower values (e.g., 0.1) produce harder, more one-hot-like outputs
          - Higher values (e.g., 10.0) produce softer, more uniform outputs
      - unk_token: Token for unseen categories
      - include_unk: Whether to include UNK in vocabulary
      - drop_original_categoricals: Drop original categorical columns
      - seed: Random seed for reproducibility

    Inverse transform:
      - Recovers categorical by argmax over soft one-hot columns.
      - This is a "hard" reconstruction, losing the soft probabilities.
    """

    name: str = "gumbel_softmax_representation"
    temperature: float = 1.0
    unk_token: str = "__UNK__"
    include_unk: bool = True
    drop_original_categoricals: bool = True
    seed: Optional[int] = None

    # fitted state
    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(default_factory=dict)
    out_cols_: Dict[str, List[str]] = field(default_factory=dict)
    rng_: Optional[np.random.Generator] = field(default=None, repr=False)

    @staticmethod
    def requires_fit() -> bool:
        return True

    @staticmethod
    def is_invertible() -> bool:
        return True

    def _get_rng(self) -> np.random.Generator:
        """Get or create random number generator."""
        if self.rng_ is None:
            self.rng_ = np.random.default_rng(self.seed)
        return self.rng_

    def fit(
            self, df: pd.DataFrame, schema: TabularSchema
    ) -> "GumbelSoftmaxRepresentation":
        self.categorical_cols_ = list(schema.categorical_cols)

        vocab: Dict[str, List[str]] = {}
        out_cols: Dict[str, List[str]] = {}

        for c in self.categorical_cols_:
            s = df[c].map(_safe_str)
            cats = list(pd.unique(s.dropna()))
            cats = sorted(cats)  # deterministic ordering

            if self.include_unk and self.unk_token not in cats:
                cats = cats + [self.unk_token]

            vocab[c] = cats
            out_cols[c] = [f"{c}__gs_{i}" for i in range(len(cats))]

        self.vocab_ = vocab
        self.out_cols_ = out_cols
        self.fitted_ = True

        # Initialize RNG
        self._get_rng()

        return self

    def transform(
            self,
            df: pd.DataFrame,
            temperature: Optional[float] = None,
    ) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError(
                "GumbelSoftmaxRepresentation must be fitted before "
                "transform()."
            )

        temp = temperature if temperature is not None else self.temperature
        rng = self._get_rng()

        out = df.copy()

        for c in self.categorical_cols_:
            if c not in out.columns:
                raise KeyError(
                    f"Categorical column '{c}' not found in DataFrame.")

            s = out[c].map(_safe_str)

            # Map unseen categories to UNK
            if self.include_unk:
                known = set(self.vocab_[c])
                s = s.where(s.isin(known), other=self.unk_token)

            cats = self.vocab_[c]
            n_cats = len(cats)

            # Create uniform logits (all categories equally likely initially)
            # In practice, learned logits would be used in a neural network
            logits = np.zeros(n_cats)

            # Apply Gumbel-Softmax to each row
            soft_one_hot = np.zeros((len(s), n_cats))

            for idx, val in enumerate(s):
                if val in cats:
                    # Create logits where the correct category has higher logit
                    cat_logits = logits.copy()
                    cat_idx = cats.index(val)
                    cat_logits[cat_idx] = 10.0

                    soft_one_hot[idx] = _gumbel_softmax_sample(cat_logits,
                                                               temp, rng)
                else:
                    soft_one_hot[idx] = _gumbel_softmax_sample(logits, temp,
                                                               rng)

            # Create output columns
            for i, col_name in enumerate(self.out_cols_[c]):
                out[col_name] = soft_one_hot[:, i]

            if self.drop_original_categoricals:
                out = out.drop(columns=[c])

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError(
                "GumbelSoftmaxRepresentation must be fitted before "
                "inverse_transform()."
            )

        out = df.copy()

        for c in self.categorical_cols_:
            gs_cols = self.out_cols_[c]
            missing = [col for col in gs_cols if col not in out.columns]
            if missing:
                raise KeyError(
                    f"Missing Gumbel-Softmax columns for '{c}': {missing}")

            # Argmax to recover category
            mat = out[gs_cols].to_numpy()
            idx = mat.argmax(axis=1)

            cats = self.vocab_[c]
            recovered = [cats[i] if 0 <= i < len(cats) else self.unk_token for
                         i in idx]

            out[c] = recovered
            out = out.drop(columns=gs_cols)

        return out

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "temperature": self.temperature,
                "unk_token": self.unk_token,
                "include_unk": self.include_unk,
                "drop_original_categoricals": self.drop_original_categoricals,
                "seed": self.seed,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "vocab": self.vocab_,
                "out_cols": self.out_cols_,
            },
        )

    @classmethod
    def from_state(cls,
                   state: RepresentationState) -> \
            "GumbelSoftmaxRepresentation":
        obj = cls(
            temperature=float(state.params.get("temperature", 1.0)),
            unk_token=str(state.params.get("unk_token", "__UNK__")),
            include_unk=bool(state.params.get("include_unk", True)),
            drop_original_categoricals=bool(
                state.params.get("drop_original_categoricals", True)
            ),
            seed=state.params.get("seed"),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.vocab_ = dict(state.params.get("vocab", {}))
        obj.out_cols_ = dict(state.params.get("out_cols", {}))

        return obj
