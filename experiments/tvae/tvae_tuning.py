from __future__ import annotations

from typing import Any


def select_tvae_best_params(**kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("TVAE tuning is implemented in the TVAE tuning task.")


def estimate_tvae_runtime(**kwargs: Any) -> Any:
    raise NotImplementedError("TVAE runtime estimate is implemented in the TVAE tuning task.")
