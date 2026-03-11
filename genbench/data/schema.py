from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class TabularSchema:
    """
    Schema for mixed-type tabular data.

    This schema supports three feature types:
      - continuous: float-like real-valued features
      - discrete: integer-valued (count/ordered) features
      - categorical: nominal categories (object/category/bool or explicitly declared)

    Notes / design goals:
      - This class is *only* about column typing/validation and safe inference.
      - It does NOT perform encoding. Encoding lives in `representations/`.
      - Validation is conservative: it checks column presence and rough type consistency.

    Parameters
    ----------
    continuous_cols:
        Names of continuous (float-like) features.
    discrete_cols:
        Names of discrete (integer-like) features (counts/ordered).
    categorical_cols:
        Names of categorical (nominal) features.
    target_col:
        Optional target column (may be any dtype; task-specific checks belong elsewhere).
    id_col:
        Optional row identifier column.
    """
    continuous_cols: List[str]
    discrete_cols: List[str]
    categorical_cols: List[str]
    target_col: Optional[str] = None
    id_col: Optional[str] = None

    # ---------------------------
    # Derived helpers
    # ---------------------------
    @property
    def feature_cols(self) -> List[str]:
        # Keep deterministic order: continuous -> discrete -> categorical
        return list(self.continuous_cols) + list(self.discrete_cols) + list(self.categorical_cols)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def all_cols(self) -> List[str]:
        cols: List[str] = []
        if self.id_col is not None:
            cols.append(self.id_col)
        cols.extend(self.feature_cols)
        if self.target_col is not None:
            cols.append(self.target_col)
        return cols

    @property
    def has_categorical(self) -> bool:
        return len(self.categorical_cols) > 0

    @property
    def has_discrete(self) -> bool:
        return len(self.discrete_cols) > 0

    @property
    def has_continuous(self) -> bool:
        return len(self.continuous_cols) > 0

    # ---------------------------
    # Validation
    # ---------------------------
    def validate(self, df: pd.DataFrame) -> None:
        # 1) Column presence
        missing = [c for c in self.all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # 2) No overlap / duplicates between groups
        groups: Dict[str, List[str]] = {
            "continuous_cols": list(self.continuous_cols),
            "discrete_cols": list(self.discrete_cols),
            "categorical_cols": list(self.categorical_cols),
        }
        # Duplicates within a group
        for name, cols in groups.items():
            dups = [c for c in set(cols) if cols.count(c) > 1]
            if dups:
                raise ValueError(f"Duplicate columns in {name}: {sorted(dups)}")

        # Overlaps across groups
        all_features = self.feature_cols
        overlaps = [c for c in set(all_features) if all_features.count(c) > 1]
        if overlaps:
            raise ValueError(
                "A column is assigned to multiple feature groups (continuous/discrete/categorical): "
                f"{sorted(overlaps)}"
            )

        # 3) Rough dtype checks (conservative)
        # Continuous: must be numeric. (Integers are allowed but discouraged; we warn via error message.)
        cont_bad: List[str] = []
        for c in self.continuous_cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                cont_bad.append(c)
        if cont_bad:
            raise TypeError(
                "Continuous columns must be numeric dtype (int/float). "
                f"Non-numeric continuous columns: {cont_bad}"
            )

        # Discrete: should be integer-like. We accept:
        #   - integer dtypes
        #   - numeric columns that are integer-valued on non-null entries
        disc_bad: List[str] = []
        for c in self.discrete_cols:
            s = df[c]
            if pd.api.types.is_integer_dtype(s):
                continue
            if pd.api.types.is_bool_dtype(s):
                # bool is typically better treated as categorical; but if user put it here, allow.
                continue
            if not pd.api.types.is_numeric_dtype(s):
                disc_bad.append(c)
                continue
            # numeric but not int dtype: check integer-valued (ignoring NaNs)
            sn = s.dropna()
            if not sn.empty:
                # allow tiny float noise by exact integer check on pandas numeric
                if not (sn.astype("float64") % 1 == 0).all():
                    disc_bad.append(c)
        if disc_bad:
            raise TypeError(
                "Discrete columns must be integer-like (integer dtype or integer-valued numeric). "
                f"Bad discrete columns: {disc_bad}"
            )

        # Categorical: allow object/category/bool OR numeric (if user intentionally uses numeric category codes).
        # We do not hard-fail numeric categoricals (common in datasets), but we ensure they are not all unique IDs.
        # The "all unique" check is a common pitfall, but we only warn via ValueError with guidance.
        cat_suspicious: List[Tuple[str, float]] = []
        for c in self.categorical_cols:
            s = df[c]
            # Accept category/object/bool freely
            if (
                pd.api.types.is_object_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(s)
            ):
                continue

            # Numeric categorical codes are allowed; detect "looks like an ID" (very high uniqueness).
            if pd.api.types.is_numeric_dtype(s):
                sn = s.dropna()
                if sn.empty:
                    continue
                uniq_ratio = float(sn.nunique()) / float(len(sn))
                if uniq_ratio > 0.98 and len(sn) >= 50:
                    cat_suspicious.append((c, uniq_ratio))
                continue

            # Otherwise unknown dtype
            raise TypeError(
                f"Categorical column '{c}' has unsupported dtype '{s.dtype}'. "
                "Use object/category/bool or numeric codes."
            )

        if cat_suspicious:
            details = ", ".join([f"{c} (uniq_ratio={r:.3f})" for c, r in cat_suspicious])
            raise ValueError(
                "Some categorical columns look like unique identifiers (almost all values unique): "
                f"{details}. If these are true IDs, move them to id_col or drop them; "
                "if they are categorical codes, consider casting to 'category' or applying rare-binning."
            )

        # 4) Optional: id_col sanity
        if self.id_col is not None:
            sid = df[self.id_col]
            if sid.dropna().empty:
                raise ValueError(f"id_col '{self.id_col}' is entirely NaN/empty.")
            # If it's intended as identifier, uniqueness is expected; do not enforce, but flag if constant.
            if sid.nunique(dropna=True) <= 1:
                raise ValueError(f"id_col '{self.id_col}' has <= 1 unique non-null value; looks constant.")

    # ---------------------------
    # Inference
    # ---------------------------
    @classmethod
    def infer_from_dataframe(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        id_col: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
        # Optional manual overrides
        continuous_cols: Optional[Sequence[str]] = None,
        discrete_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        # Heuristics knobs
        treat_bool_as_categorical: bool = True,
        discrete_max_unique: int = 20,
        categorical_numeric_max_unique: int = 200,
        categorical_unique_ratio_id_threshold: float = 0.98,
        drop_unused: bool = False,
    ) -> "TabularSchema":
        """
        Infer a mixed-type schema from a DataFrame.

        Strategy (when explicit cols are not provided):
          - Start from feature_cols (or all non-(id/target) columns).
          - Assign:
              * categorical: object/category; bool optionally
              * discrete: integer dtypes OR "few unique integer-like numeric"
              * continuous: remaining numeric
          - Numeric columns with very high uniqueness can be treated as continuous by default,
            but if explicitly specified as categorical we'll allow (and validate).

        Parameters
        ----------
        feature_cols:
            If provided, only these columns are considered features; otherwise all except id/target.
        continuous_cols / discrete_cols / categorical_cols:
            Explicit assignments. If given, they override inference for those columns.
        treat_bool_as_categorical:
            If True, bool columns are inferred as categorical (recommended).
        discrete_max_unique:
            For integer/integer-like columns: if nunique <= this, infer as discrete; otherwise treat as continuous-like.
            (Useful when integer columns are actually continuous measured values.)
        categorical_numeric_max_unique:
            Numeric columns with nunique <= this can be inferred as categorical *only if* explicitly requested
            OR if they are non-float integer-like but not too many unique values.
        drop_unused:
            If True, silently drop columns not in feature_cols/explicit lists. If False, ignore them.

        Returns
        -------
        TabularSchema
        """
        cols = list(df.columns)

        if id_col is not None and id_col not in cols:
            raise ValueError(f"id_col='{id_col}' not found in df.columns")
        if target_col is not None and target_col not in cols:
            raise ValueError(f"target_col='{target_col}' not found in df.columns")

        if feature_cols is None:
            exclude = {c for c in (id_col, target_col) if c is not None}
            feats = [c for c in cols if c not in exclude]
        else:
            feats = list(feature_cols)
            missing_feats = [c for c in feats if c not in cols]
            if missing_feats:
                raise ValueError(f"feature_cols contains columns not in df: {missing_feats}")

        feats_set = set(feats)

        # Explicit overrides
        cont_exp = set(continuous_cols or [])
        disc_exp = set(discrete_cols or [])
        cat_exp = set(categorical_cols or [])

        # Ensure explicit columns belong to feats (unless drop_unused=False and feature_cols=None, still ok)
        for name, exp in [("continuous_cols", cont_exp), ("discrete_cols", disc_exp), ("categorical_cols", cat_exp)]:
            extra = [c for c in exp if c not in feats_set]
            if extra and feature_cols is not None:
                raise ValueError(f"{name} contains columns not present in feature_cols: {extra}")

        # Start with explicit assignments
        cont: List[str] = [c for c in feats if c in cont_exp]
        disc: List[str] = [c for c in feats if c in disc_exp]
        cat: List[str] = [c for c in feats if c in cat_exp]

        assigned = set(cont) | set(disc) | set(cat)

        # Infer remaining
        for c in feats:
            if c in assigned:
                continue

            s = df[c]

            # Categorical by dtype
            if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s):
                cat.append(c)
                continue

            if pd.api.types.is_bool_dtype(s):
                if treat_bool_as_categorical:
                    cat.append(c)
                else:
                    disc.append(c)  # alternative: treat as discrete {0,1}
                continue

            # Numeric types
            if pd.api.types.is_numeric_dtype(s):
                sn = s.dropna()
                if sn.empty:
                    # If all missing, default to continuous (won't matter; downstream missing handler should drop/impute)
                    cont.append(c)
                    continue

                nunique = int(sn.nunique())
                uniq_ratio = float(nunique) / float(len(sn))

                # Integer dtype: could be discrete or continuous-like measurement stored as int.
                if pd.api.types.is_integer_dtype(s):
                    if nunique <= discrete_max_unique:
                        disc.append(c)
                    else:
                        # High unique integer columns often behave like continuous
                        cont.append(c)
                    continue

                # Non-integer numeric: check if integer-valued
                is_integer_valued = (sn.astype("float64") % 1 == 0).all()

                if is_integer_valued:
                    # integer-like float column (e.g., 1.0, 2.0, ...)
                    if nunique <= discrete_max_unique:
                        disc.append(c)
                    else:
                        cont.append(c)
                    continue

                # True float-valued -> continuous
                cont.append(c)
                continue

            # Fallback: unknown dtype -> categorical (but it's safer to require explicit handling)
            raise TypeError(
                f"Cannot infer type for column '{c}' with dtype '{s.dtype}'. "
                "Please cast it or specify categorical_cols/discrete_cols/continuous_cols explicitly."
            )

        # If user didn't specify feature_cols and drop_unused=True, we simply ignore unused.
        # (No-op here since we only built schema from feats.)

        schema = cls(
            continuous_cols=cont,
            discrete_cols=disc,
            categorical_cols=cat,
            target_col=target_col,
            id_col=id_col,
        )
        schema.validate(df)
        return schema
