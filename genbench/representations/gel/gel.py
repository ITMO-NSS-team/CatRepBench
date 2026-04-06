from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.data.type_inference import infer_feature_type
from genbench.representations.base import RepresentationState


def _safe_str(x: object) -> str:
    if x is None:
        return "__NONE__"
    if isinstance(x, float) and pd.isna(x):
        return "__NAN__"
    return str(x)


def _pad_vectors(mat: np.ndarray, width: int) -> np.ndarray:
    if mat.shape[1] >= width:
        return mat[:, :width]
    if mat.shape[0] == 0:
        return np.zeros((0, width), dtype=float)
    out = np.zeros((mat.shape[0], width), dtype=float)
    out[:, : mat.shape[1]] = mat
    return out


@dataclass
class GELRepresentation:
    """
    Practical GEL-style spectral embedding for categorical tokens.

    This implementation stays inside the project's common preprocessing pipeline:
      - continuous features are handled by ContinuousStandardScaler,
      - discrete features pass through unchanged,
      - categorical features are replaced by dense GEL vectors.

    We learn embeddings for column/category tokens from a token co-occurrence graph.
    When a classification-style target is available, the graph is weighted per class
    using class priors to keep the embedding supervised, similar in spirit to the
    class-partitioned instance representation described in the GEL paper.
    """

    name: str = "gel_representation"
    embedding_dim: int = 4
    supervision: Literal["auto", "supervised", "unsupervised"] = "auto"
    include_unk: bool = True
    unk_token: str = "__UNK__"
    drop_original_categoricals: bool = True
    eps: float = 1e-12

    fitted_: bool = False
    categorical_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(default_factory=dict)
    out_cols_: Dict[str, List[str]] = field(default_factory=dict)
    embeddings_: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    supervision_used_: Literal["supervised", "unsupervised"] = "unsupervised"
    target_classes_: List[str] = field(default_factory=list)

    def requires_fit(self) -> bool:
        return True

    def is_invertible(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame, schema: TabularSchema) -> "GELRepresentation":
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")

        self.categorical_cols_ = list(schema.categorical_cols)
        self.out_cols_ = {
            col: [f"{col}__gel_{idx}" for idx in range(self.embedding_dim)]
            for col in self.categorical_cols_
        }

        if not self.categorical_cols_:
            self.vocab_ = {}
            self.embeddings_ = {}
            self.target_classes_ = []
            self.supervision_used_ = "unsupervised"
            self.fitted_ = True
            return self

        vocab_known: Dict[str, List[str]] = {}
        token_specs: List[Tuple[str, str]] = []
        token_index: Dict[Tuple[str, str], int] = {}
        offset = 0

        for col in self.categorical_cols_:
            safe_values = df[col].map(_safe_str)
            categories = sorted(str(v) for v in pd.unique(safe_values))
            vocab_known[col] = categories
            for cat in categories:
                token_specs.append((col, cat))
                token_index[(col, cat)] = offset
                offset += 1

        token_count = len(token_specs)
        n_rows = len(df)
        token_matrix = np.zeros((n_rows, token_count), dtype=float)
        row_indices = np.arange(n_rows)

        for col in self.categorical_cols_:
            safe_values = df[col].map(_safe_str)
            col_token_indices = safe_values.map(lambda v: token_index[(col, v)]).to_numpy(dtype=int)
            token_matrix[row_indices, col_token_indices] = 1.0

        labels, supervision_used = self._resolve_labels(df=df, schema=schema)
        self.supervision_used_ = supervision_used
        self.target_classes_ = sorted(set(labels.tolist())) if labels is not None else []

        similarity = self._build_similarity_matrix(token_matrix=token_matrix, labels=labels)
        token_embeddings = self._spectral_embedding(similarity)

        vocab: Dict[str, List[str]] = {}
        embeddings: Dict[str, Dict[str, List[float]]] = {}
        for col in self.categorical_cols_:
            col_embeddings: Dict[str, List[float]] = {}
            vectors: List[np.ndarray] = []

            for cat in vocab_known[col]:
                vec = token_embeddings[token_index[(col, cat)]]
                col_embeddings[cat] = [float(v) for v in vec.tolist()]
                vectors.append(vec)

            vocab[col] = list(vocab_known[col])
            if self.include_unk and self.unk_token not in col_embeddings:
                if vectors:
                    unk_vec = np.mean(np.vstack(vectors), axis=0)
                else:
                    unk_vec = np.zeros(self.embedding_dim, dtype=float)
                col_embeddings[self.unk_token] = [float(v) for v in unk_vec.tolist()]
                vocab[col].append(self.unk_token)

            embeddings[col] = col_embeddings

        self.vocab_ = vocab
        self.embeddings_ = embeddings
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("GELRepresentation must be fitted before transform().")

        out = df.copy()
        if not self.categorical_cols_:
            return out

        for col in self.categorical_cols_:
            if col not in out.columns:
                raise KeyError(f"Categorical column '{col}' not found in DataFrame.")

            mapping = self.embeddings_[col]
            safe_values = out[col].map(_safe_str)
            if self.include_unk:
                safe_values = safe_values.where(safe_values.isin(mapping), other=self.unk_token)
            else:
                unknown_values = sorted(set(safe_values.tolist()) - set(mapping.keys()))
                if unknown_values:
                    raise ValueError(
                        f"Unknown categories found in column '{col}': {unknown_values}. "
                        "Set include_unk=True to encode unseen values."
                    )

            if len(out):
                fallback = mapping.get(self.unk_token, [0.0] * self.embedding_dim)
                rows = np.asarray(
                    [mapping[value] if value in mapping else fallback for value in safe_values.tolist()],
                    dtype=float,
                )
            else:
                rows = np.zeros((0, self.embedding_dim), dtype=float)

            for idx, out_col in enumerate(self.out_cols_[col]):
                out[out_col] = rows[:, idx]

            if self.drop_original_categoricals:
                out = out.drop(columns=[col])

        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("GELRepresentation must be fitted before inverse_transform().")
        raise NotImplementedError("GELRepresentation is not invertible.")

    def get_state(self) -> RepresentationState:
        return RepresentationState(
            name=self.name,
            params={
                "embedding_dim": self.embedding_dim,
                "supervision": self.supervision,
                "include_unk": self.include_unk,
                "unk_token": self.unk_token,
                "drop_original_categoricals": self.drop_original_categoricals,
                "eps": self.eps,
                "fitted": self.fitted_,
                "categorical_cols": self.categorical_cols_,
                "vocab": self.vocab_,
                "out_cols": self.out_cols_,
                "embeddings": self.embeddings_,
                "supervision_used": self.supervision_used_,
                "target_classes": self.target_classes_,
            },
        )

    @classmethod
    def from_state(cls, state: RepresentationState) -> "GELRepresentation":
        obj = cls(
            embedding_dim=int(state.params.get("embedding_dim", 4)),
            supervision=str(state.params.get("supervision", "auto")),  # type: ignore[arg-type]
            include_unk=bool(state.params.get("include_unk", True)),
            unk_token=str(state.params.get("unk_token", "__UNK__")),
            drop_original_categoricals=bool(state.params.get("drop_original_categoricals", True)),
            eps=float(state.params.get("eps", 1e-12)),
        )
        obj.fitted_ = bool(state.params.get("fitted", False))
        obj.categorical_cols_ = list(state.params.get("categorical_cols", []))
        obj.vocab_ = {
            str(col): [str(v) for v in values]
            for col, values in dict(state.params.get("vocab", {})).items()
        }
        obj.out_cols_ = {
            str(col): [str(v) for v in values]
            for col, values in dict(state.params.get("out_cols", {})).items()
        }
        raw_embeddings = dict(state.params.get("embeddings", {}))
        obj.embeddings_ = {
            str(col): {
                str(cat): [float(v) for v in vec]
                for cat, vec in dict(col_map).items()
            }
            for col, col_map in raw_embeddings.items()
        }
        obj.supervision_used_ = str(
            state.params.get("supervision_used", "unsupervised")
        )  # type: ignore[assignment]
        obj.target_classes_ = [str(v) for v in list(state.params.get("target_classes", []))]
        return obj

    def _resolve_labels(
        self,
        *,
        df: pd.DataFrame,
        schema: TabularSchema,
    ) -> tuple[pd.Series | None, Literal["supervised", "unsupervised"]]:
        if self.supervision == "unsupervised":
            return None, "unsupervised"

        target_col = schema.target_col
        if target_col is None or target_col not in df.columns:
            if self.supervision == "supervised":
                raise ValueError(
                    "GELRepresentation(supervision='supervised') requires schema.target_col in the DataFrame."
                )
            return None, "unsupervised"

        target_series = df[target_col]
        target_type = infer_feature_type(
            target_series,
            discrete_max_unique=20,
            treat_bool_as_categorical=True,
        )
        if target_type == "continuous":
            if self.supervision == "supervised":
                raise ValueError(
                    "GELRepresentation(supervision='supervised') requires a classification-style target "
                    "(categorical or low-cardinality discrete), not a continuous target."
                )
            return None, "unsupervised"

        return target_series.map(_safe_str), "supervised"

    def _build_similarity_matrix(
        self,
        *,
        token_matrix: np.ndarray,
        labels: pd.Series | None,
    ) -> np.ndarray:
        _n_rows, token_count = token_matrix.shape
        if token_count == 0:
            return np.zeros((0, 0), dtype=float)

        if labels is None:
            cooccurrence = token_matrix.T @ token_matrix
            supervision_similarity = None
        else:
            cooccurrence = np.zeros((token_count, token_count), dtype=float)
            label_values = labels.to_numpy()
            unique_labels = sorted(set(label_values.tolist()))
            label_profile = np.zeros((token_count, len(unique_labels)), dtype=float)
            for label_idx, label in enumerate(unique_labels):
                mask = label_values == label
                if not np.any(mask):
                    continue
                prior = float(np.mean(mask))
                token_block = token_matrix[mask]
                cooccurrence += prior * (token_block.T @ token_block)
                label_profile[:, label_idx] = token_block.sum(axis=0)

            row_sums = np.maximum(label_profile.sum(axis=1, keepdims=True), self.eps)
            label_profile = label_profile / row_sums
            supervision_similarity = label_profile @ label_profile.T

        diagonal = np.diag(cooccurrence).astype(float)
        scale = np.sqrt(np.maximum(diagonal, self.eps))
        similarity = cooccurrence / np.outer(scale, scale)
        if supervision_similarity is not None:
            similarity = 0.5 * similarity + 0.5 * supervision_similarity
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        similarity = 0.5 * (similarity + similarity.T)
        return similarity

    def _spectral_embedding(self, similarity: np.ndarray) -> np.ndarray:
        token_count = similarity.shape[0]
        if token_count == 0:
            return np.zeros((0, self.embedding_dim), dtype=float)
        if token_count == 1:
            return np.ones((1, self.embedding_dim), dtype=float)

        values, vectors = np.linalg.eigh(similarity)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]

        positive = values > self.eps
        if not np.any(positive):
            return np.zeros((token_count, self.embedding_dim), dtype=float)

        vectors = vectors[:, positive]
        values = values[positive]

        for idx in range(vectors.shape[1]):
            pivot = int(np.argmax(np.abs(vectors[:, idx])))
            if vectors[pivot, idx] < 0:
                vectors[:, idx] *= -1.0

        embedding = vectors * np.sqrt(np.clip(values, a_min=0.0, a_max=None))
        return _pad_vectors(embedding, self.embedding_dim)

