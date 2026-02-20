from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import pandas as pd

from genbench.data.schema import TabularSchema


@dataclass
class GenerativeState:
    """
    Minimal serializable state for a generative model.
    """
    name: str
    params: Dict[str, Any]


@runtime_checkable
class BaseGenerative(Protocol):
    """
    Thin wrapper contract for pluggable tabular generative models.

    Expectations:
      - fit: train on already-preprocessed data (after TransformPipeline).
      - sample: return a DataFrame with the same columns/types seen at fit-time.
      - conditional sampling: optional; raise NotImplementedError if not supported.
      - get_state/from_state: lightweight (JSON-friendly) config; save_artifacts/load_artifacts:
        heavy weights/checkpoints.
    """

    name: str

    def requires_fit(self) -> bool:
        ...

    def is_conditional(self) -> bool:
        ...

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "BaseGenerative":
        ...

    def sample(self, n: int, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        ...

    def get_loss_history(self) -> Optional[Dict[str, list[float]]]:
        """
        Optional: return training loss traces for diagnostics/plotting.
        Expected format: dict[str, list[float]], e.g.
          {"generator_loss": [...], "discriminator_loss": [...]}
        Return None if not tracked.
        """
        ...

    def get_state(self) -> GenerativeState:
        ...

    @classmethod
    def from_state(cls, state: GenerativeState) -> "BaseGenerative":
        ...

    def save_artifacts(self, path: Path) -> None:
        """Persist heavy binary weights if present."""
        ...

    @classmethod
    def load_artifacts(cls, path: Path) -> "BaseGenerative":
        """Load weights previously saved via save_artifacts()."""
        ...
