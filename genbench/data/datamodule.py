from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .schema import TabularSchema
from .type_inference import is_categorical_like_dtype
from .splits import (
    SplitConfigKFold,
    SplitConfigHoldout,
    KFoldSplit,
    HoldoutSplit,
    make_kfold_splits,
    make_holdout_split,
)


@dataclass
class FoldData:
    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    # Fitted (fold-specific) transforms are returned for reproducibility / downstream decoding
    transforms: Optional[Any] = None


@dataclass
class HoldoutData:
    train: pd.DataFrame
    val: pd.DataFrame
    transforms: Optional[Any] = None


class TabularDataModule:
    """
    Mixed-type tabular data module (continuous + discrete + categorical).

    Provides TWO INDEPENDENT splitting protocols:
      - k-fold splits for experiments
      - single holdout split for tuning generative models

    Key sb-tabular design principles preserved:
      1) Schema validates input columns.
      2) All transforms, including missing handling, are applied split-wise.
      3) Fold-wise transforms are FIT on the corresponding train subset only, then
         applied to train/test (or train/val) subsets.

    Notes for mixed-type settings:
      - Do NOT encode categoricals here. Encoding belongs to `representations/`.
      - Transforms are responsible for:
          * missing value policy (drop/impute)
          * scaling / normalization for continuous features
          * optional canonicalization / rare-binning for categoricals (if you decide to do that as a transform)
      - The DataModule only guarantees "fit-on-train" semantics per split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: TabularSchema,
        transforms: Optional[Any] = None,
        reset_index: bool = True,
        # If True, also validates that df does not contain duplicated columns in schema groups, etc.
        validate: bool = True,
    ) -> None:
        self.schema = schema
        self.transforms = transforms

        if validate:
            schema.validate(df)

        df0 = df.copy()

        if reset_index:
            df0 = df0.reset_index(drop=True)

        self.df_clean = df0
        self.n_samples = len(df0)

        self._kfold_splits: Optional[list[KFoldSplit]] = None
        self._holdout_split: Optional[HoldoutSplit] = None

    # --------- K-FOLD (experiments) ---------

    def prepare_kfold(self, cfg: SplitConfigKFold) -> None:
        self._kfold_splits = make_kfold_splits(self.n_samples, cfg)

    def get_fold(self, fold_id: int) -> FoldData:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")

        if fold_id < 0 or fold_id >= len(self._kfold_splits):
            raise IndexError(f"fold_id={fold_id} out of range (n_folds={len(self._kfold_splits)})")

        fold = self._kfold_splits[fold_id]
        train_raw = self.df_clean.iloc[fold.train_idx].copy()
        test_raw = self.df_clean.iloc[fold.test_idx].copy()
        self._validate_no_unseen_categories(train_raw, test_raw, context=f"fold={fold_id} test")

        if self.transforms is None:
            return FoldData(fold_id=fold_id, train=train_raw, test=test_raw, transforms=None)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)  # fit only on fold-train
        train = pipe.transform(train_raw)
        test = pipe.transform(test_raw)

        self._validate_post_transform(train, context=f"fold={fold_id} train")
        self._validate_post_transform(test, context=f"fold={fold_id} test", reference_cols=list(train.columns))

        return FoldData(fold_id=fold_id, train=train, test=test, transforms=pipe)

    def get_all_folds(self) -> Dict[int, FoldData]:
        if self._kfold_splits is None:
            raise RuntimeError("K-fold splits are not prepared. Call prepare_kfold(cfg) first.")
        return {f.fold_id: self.get_fold(f.fold_id) for f in self._kfold_splits}

    # --------- HOLDOUT (tuning) ---------

    def prepare_holdout(self, cfg: SplitConfigHoldout) -> None:
        self._holdout_split = make_holdout_split(self.n_samples, cfg)

    def get_holdout(self) -> HoldoutData:
        if self._holdout_split is None:
            raise RuntimeError("Holdout split is not prepared. Call prepare_holdout(cfg) first.")

        sp = self._holdout_split
        train_raw = self.df_clean.iloc[sp.train_idx].copy()
        val_raw = self.df_clean.iloc[sp.val_idx].copy()
        self._validate_no_unseen_categories(train_raw, val_raw, context="holdout val")

        if self.transforms is None:
            return HoldoutData(train=train_raw, val=val_raw, transforms=None)

        pipe = self._clone_transforms(self.transforms)
        pipe.fit(train_raw, self.schema)  # fit only on holdout-train
        train = pipe.transform(train_raw)
        val = pipe.transform(val_raw)

        self._validate_post_transform(train, context="holdout train")
        self._validate_post_transform(val, context="holdout val", reference_cols=list(train.columns))

        return HoldoutData(train=train, val=val, transforms=pipe)

    # --------- Optional convenience ---------

    def get_clean_df(self) -> pd.DataFrame:
        """Return the validated DataFrame after optional index reset, before split-wise transforms."""
        return self.df_clean.copy()

    # --------- Internals ---------

    def _validate_post_transform(
        self,
        df: pd.DataFrame,
        context: str,
        reference_cols: Optional[list[str]] = None,
    ) -> None:
        """
        Lightweight checks after split-wise transforms.
        """
        required_cols = [c for c in (self.schema.id_col, self.schema.target_col) if c is not None]
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            raise ValueError(f"[{context}] Transforms removed required columns: {missing_required}")

        if reference_cols is not None:
            missing = [c for c in reference_cols if c not in df.columns]
            extra = [c for c in df.columns if c not in reference_cols]
            if missing or extra:
                raise ValueError(
                    f"[{context}] Transformed columns differ from train columns. "
                    f"Missing: {missing}. Extra: {extra}."
                )

    def _validate_no_unseen_categories(self, train_df: pd.DataFrame, other_df: pd.DataFrame, context: str) -> None:
        categorical_cols = list(self.schema.categorical_cols)
        if self.schema.target_col is not None and self.schema.target_col in train_df.columns:
            if is_categorical_like_dtype(train_df[self.schema.target_col]):
                categorical_cols.append(self.schema.target_col)

        for col in categorical_cols:
            if col not in train_df.columns or col not in other_df.columns:
                continue
            train_values = {value for value in pd.unique(train_df[col].dropna())}
            other_values = {value for value in pd.unique(other_df[col].dropna())}
            unseen_values = other_values - train_values
            if unseen_values:
                unseen_rendered = sorted(str(value) for value in unseen_values)
                raise ValueError(
                    f"[{context}] Found categories not present in train for column '{col}': "
                    f"{unseen_rendered}"
                )

    @staticmethod
    def _clone_transforms(transforms: Any) -> Any:
        """
        Best-effort cloning:
          - If pipeline supports get_state/from_state -> use it.
          - Else deepcopy.
        """
        if hasattr(transforms, "get_state") and hasattr(transforms.__class__, "from_state"):
            state = transforms.get_state()
            return transforms.__class__.from_state(state)  # type: ignore[attr-defined]
        import copy

        return copy.deepcopy(transforms)
