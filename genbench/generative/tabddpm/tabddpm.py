from __future__ import annotations

import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch

from genbench.data.schema import TabularSchema
from genbench.generative.base import BaseGenerative, GenerativeState
from .diffusion import GaussianMultinomialDiffusion
from .modules import MLPDiffusion
from .utils import update_ema


def _infer_feature_types(
        source_schema: TabularSchema,
        processed_schema: TabularSchema,
) -> Dict[str, str]:
    """
    Infer feature types for TabDDPM based on source and processed schemas.

    Logic:
        - Continuous in source -> numerical (Gaussian diffusion)
        - Discrete in source -> numerical (Gaussian diffusion)
        - Categorical in source + continuous in processed -> numerical (
        Gaussian diffusion)
        - Categorical in source + discrete in processed -> categorical (
        Multinomial diffusion)

    Parameters
    ----------
    source_schema : TabularSchema
        Schema of the original dataset before preprocessing.
    processed_schema : TabularSchema
        Schema of the dataset after preprocessing/encoding.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping column names to their TabDDPM types:
        'numerical' for Gaussian diffusion, 'categorical' for Multinomial
        diffusion.
    """
    source_cont: Set[str] = set(source_schema.continuous_cols)
    source_disc: Set[str] = set(source_schema.discrete_cols)
    source_cat: Set[str] = set(source_schema.categorical_cols)

    processed_cont: Set[str] = set(processed_schema.continuous_cols)
    processed_disc: Set[str] = set(processed_schema.discrete_cols)
    processed_cat: Set[str] = set(processed_schema.categorical_cols)

    feature_types: Dict[str, str] = {}

    # All features from processed schema
    all_features = processed_schema.feature_cols

    for col in all_features:
        if col in source_cont:
            # Continuous in source -> numerical
            feature_types[col] = 'numerical'
        elif col in source_disc:
            # Discrete in source -> numerical
            feature_types[col] = 'numerical'
        elif col in source_cat:
            if col in processed_cont:
                # Categorical -> continuous
                feature_types[col] = 'numerical'
            elif col in processed_disc:
                # Categorical -> discrete
                # Treat as categorical for Multinomial diffusion
                feature_types[col] = 'categorical'
            elif col in processed_cat:
                # Still categorical
                feature_types[col] = 'categorical'
            else:
                raise ValueError(
                    f"Column '{col}' from source categorical not found in "
                    f"processed schema"
                )
        else:
            # Column not in source schema
            if col in processed_cont or col in processed_disc:
                feature_types[col] = 'numerical'
            else:
                feature_types[col] = 'categorical'

    return feature_types


@dataclass
class TabDDPMGenerative(BaseGenerative):
    """
    Thin wrapper around TabDDPM to comply with BaseGenerative protocol.

    TabDDPM is a diffusion-based generative model for tabular data that handles
    both continuous (Gaussian diffusion) and categorical (Multinomial
    diffusion)
    features.

    Parameters
    ----------
    name : str
        Name identifier for the model.
    num_timesteps : int
        Number of diffusion timesteps (default: 1000).
    num_epochs : int
        Number of training epochs (default: 100).
    batch_size : int
        Batch size for training (default: 1024).
    lr : float
        Learning rate (default: 0.002).
    weight_decay : float
        Weight decay for optimizer (default: 1e-4).
    dim_t : int
        Dimension of the time embedding (default: 128).
    d_layers : List[int]
        Dimensions of the MLP hidden layers (default: [256, 256, 256]).
    dropout : float
        Dropout rate (default: 0.0).
    scheduler : str
        Beta schedule: 'cosine' or 'linear' (default: 'cosine').
    gaussian_loss_type : str
        Loss type for Gaussian diffusion: 'mse' or 'kl' (default: 'mse').
    device : str
        Device to use: 'cuda' or 'cpu' (default: 'cuda' if available).
    """

    name: str = "tabddpm"
    num_timesteps: int = 1000
    num_epochs: int = 100
    batch_size: int = 1024
    lr: float = 0.002
    weight_decay: float = 1e-4
    dim_t: int = 128
    d_layers: List[int] = field(default_factory=lambda: [256, 256, 256])
    dropout: float = 0.0
    scheduler: str = "cosine"
    gaussian_loss_type: str = "mse"
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")

    # fitted artifacts
    model_: Any = None
    diffusion_: Any = None
    ema_model_: Any = None
    fitted_: bool = False

    # data info - feature columns
    num_numerical_features_: int = 0
    num_classes_: np.ndarray = field(default_factory=lambda: np.array([0]))
    numerical_cols_: List[str] = field(default_factory=list)
    categorical_cols_: List[str] = field(default_factory=list)
    column_order_: List[str] = field(default_factory=list)

    # target info - target is always modeled as part of X
    target_col_: Optional[str] = None

    loss_history_: List[Dict[str, float]] = field(default_factory=list)

    def requires_fit(self) -> bool:
        return True

    def is_conditional(self) -> bool:
        return False

    def _prepare_data(
            self,
            df: pd.DataFrame,
            source_schema: TabularSchema,
            processed_schema: TabularSchema,
    ) -> torch.Tensor:
        """
        Prepare data for TabDDPM.

        Fills class fields based on schema analysis and data, and converts
        DataFrame to tensor. Data should be already preprocessed.

        Target variable is modeled together with other features.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed training data.
        source_schema : TabularSchema
            Schema of the original dataset before preprocessing.
        processed_schema : TabularSchema
            Schema of the dataset after preprocessing/encoding.

        Returns
        -------
        torch.Tensor
            Tensor representation of the data.
        """
        feature_types = _infer_feature_types(source_schema, processed_schema)

        # Store target column info
        self.target_col_ = processed_schema.target_col

        # Separate into numerical and categorical based on inferred types
        self.numerical_cols_ = [
            col for col in processed_schema.feature_cols
            if feature_types.get(col) == 'numerical'
        ]
        self.categorical_cols_ = [
            col for col in processed_schema.feature_cols
            if feature_types.get(col) == 'categorical'
        ]

        # Column order: numerical first, then categorical
        self.column_order_ = self.numerical_cols_ + self.categorical_cols_
        self.num_numerical_features_ = len(self.numerical_cols_)

        # Process numerical features
        if self.numerical_cols_:
            num_data = df[self.numerical_cols_].values.astype(np.float32)
        else:
            num_data = np.zeros((len(df), 0), dtype=np.float32)

        # Process categorical features
        if self.categorical_cols_:
            cat_data = df[self.categorical_cols_].values.astype(np.int64)

            # Compute number of classes for each categorical feature
            num_classes_list = []
            for i, col in enumerate(self.categorical_cols_):
                col_data = cat_data[:, i]
                n_classes = int(col_data.max()) + 1
                num_classes_list.append(n_classes)

            self.num_classes_ = np.array(num_classes_list)
        else:
            cat_data = np.zeros((len(df), 0), dtype=np.int64)
            self.num_classes_ = np.array([0])

        # Concatenate numerical and categorical features
        X = np.hstack([num_data, cat_data])
        return torch.from_numpy(X).float()

    def fit(
            self,
            df: pd.DataFrame,
            schema: TabularSchema,
            source_schema: Optional[TabularSchema] = None,
    ) -> "TabDDPMGenerative":
        """
        Fit the TabDDPM model on the provided data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        schema : TabularSchema
            Schema describing the data types.
        source_schema : TabularSchema, optional
            Schema of the original dataset before preprocessing.
            If not provided, assumes schema is the same as source schema
            (no preprocessing was applied).

        Returns
        -------
        self : TabDDPMGenerative
            Fitted model instance.
        """
        if source_schema is None:
            source_schema = schema

        device = torch.device(self.device)

        # Prepare data
        X = self._prepare_data(df, source_schema, schema)
        X = X.to(device)

        # Build model
        d_in = X.shape[1]

        # Adjust d_in for one-hot encoding of categorical features
        if self.num_classes_[0] > 0:
            d_in = self.num_numerical_features_ + int(self.num_classes_.sum())

        model_params = {
            'd_in': d_in,
            'num_classes': 0,  # Always 0 - target modeled with features
            'is_y_cond': False,  # Always False - unconditional on y
            'rtdl_params': {
                'd_layers': self.d_layers,
                'dropout': self.dropout,
            },
            'dim_t': self.dim_t,
        }

        self.model_ = MLPDiffusion(**model_params)
        self.model_.to(device)

        # Create diffusion model
        self.diffusion_ = GaussianMultinomialDiffusion(
            num_classes=self.num_classes_,
            num_numerical_features=self.num_numerical_features_,
            denoise_fn=self.model_,
            num_timesteps=self.num_timesteps,
            gaussian_loss_type=self.gaussian_loss_type,
            multinomial_loss_type='vb_stochastic',
            scheduler=self.scheduler,
            device=device,
        )
        self.diffusion_.to(device)
        self.diffusion_.train()

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.diffusion_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # EMA model for better sampling
        self.ema_model_ = deepcopy(self.model_)
        for param in self.ema_model_.parameters():
            param.detach_()

        # Training loop
        n_steps = self.num_epochs * (len(df) // self.batch_size + 1)
        step = 0
        self.loss_history_ = []

        # Create dataloader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        for epoch in range(self.num_epochs):
            epoch_losses = []
            epoch_multi_losses = []
            epoch_gauss_losses = []

            for (batch_x,) in dataloader:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()
                # Empty out_dict since target is in X (regression mode)
                out_dict = {
                    'y': torch.zeros(batch_x.shape[0], dtype=torch.long,
                                     device=device)}
                loss_multi, loss_gauss = self.diffusion_.mixed_loss(
                    batch_x, out_dict
                )
                loss = loss_multi + loss_gauss
                loss.backward()
                optimizer.step()

                # Anneal learning rate
                frac_done = step / n_steps
                new_lr = self.lr * (1 - frac_done)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

                # Update EMA
                update_ema(
                    self.ema_model_.parameters(),
                    self.model_.parameters()
                )

                # Collect losses for averaging
                epoch_losses.append(loss.item())
                epoch_multi_losses.append(loss_multi.item())
                epoch_gauss_losses.append(loss_gauss.item())
                step += 1

            # Compute epoch averages
            avg_loss = np.mean(epoch_losses)
            avg_multi = np.mean(epoch_multi_losses)
            avg_gauss = np.mean(epoch_gauss_losses)

            self.loss_history_.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'multinomial_loss': avg_multi,
                'gaussian_loss': avg_gauss,
            })

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}, "
                    f"Loss: {avg_loss:.4f} "
                    f"(multi: {avg_multi:.4f}, gauss: {avg_gauss:.4f})"
                )

        # Use EMA model for sampling
        self.diffusion_._denoise_fn = self.ema_model_
        self.diffusion_.eval()
        self.fitted_ = True

        return self

    def sample(self, n: int,
               conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        conditions : pd.DataFrame, optional
        Ignored. Model generates all features including target jointly.

        Returns
        -------
        pd.DataFrame
            Generated synthetic data with target column if present.
        """
        if not self.fitted_ or self.diffusion_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        device = torch.device(self.device)

        # Dummy y_dist for unconditional sampling
        y_dist = torch.ones(1, device=device)

        # Sample from diffusion model
        with torch.no_grad():
            X_gen, _ = self.diffusion_.sample_all(
                num_samples=n,
                batch_size=min(self.batch_size, n),
                y_dist=y_dist,
                ddim=False,
            )

        # Convert to DataFrame
        X_gen_np = X_gen.cpu().numpy()
        df = self._build_dataframe(X_gen_np)

        return df

    def _build_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """
        Build DataFrame from generated tensor.

        Parameters
        ----------
        X : np.ndarray
            Generated data array.

        Returns
        -------
        pd.DataFrame
            DataFrame with proper column names.
        """
        data = {}
        n_num = self.num_numerical_features_
        n_cat = len(self.num_classes_) if self.num_classes_[0] > 0 else 0

        # Numerical features
        if n_num > 0:
            num_data = X[:, :n_num]
            for i, col in enumerate(self.column_order_[:n_num]):
                data[col] = num_data[:, i]

        # Categorical features
        if n_cat > 0:
            cat_data = X[:, n_num:n_num + n_cat]
            # Round categorical values to nearest integer and clip to valid
            # range
            cat_data = np.round(cat_data).astype(np.int64)
            for i, col in enumerate(self.column_order_[n_num:n_num + n_cat]):
                data[col] = np.clip(cat_data[:, i], 0,
                                    self.num_classes_[i] - 1)

        return pd.DataFrame(data)

    def get_loss_history(self) -> Optional[Dict[str, list]]:
        """
        Return training loss history.

        Returns
        -------
        Dict[str, list] or None
            Dictionary with loss history or None if not available.
        """
        if not self.loss_history_:
            return None

        return {
            'loss': [h['loss'] for h in self.loss_history_],
            'multinomial_loss': [
                h['multinomial_loss'] for h in self.loss_history_
            ],
            'gaussian_loss': [
                h['gaussian_loss'] for h in self.loss_history_
            ],
        }

    def get_state(self) -> GenerativeState:
        """
        Get the model state for serialization.

        Returns
        -------
        GenerativeState
            Serializable state object.
        """
        return GenerativeState(
            name=self.name,
            params={
                "num_timesteps": self.num_timesteps,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "dim_t": self.dim_t,
                "d_layers": self.d_layers,
                "dropout": self.dropout,
                "scheduler": self.scheduler,
                "gaussian_loss_type": self.gaussian_loss_type,
                "device": self.device,
            },
        )

    @classmethod
    def from_state(cls, state: GenerativeState) -> "TabDDPMGenerative":
        """
        Create a model instance from a state object.

        Parameters
        ----------
        state : GenerativeState
            State object with model parameters.

        Returns
        -------
        TabDDPMGenerative
            New model instance with restored parameters.
        """
        params = state.params or {}
        return cls(
            num_timesteps=params.get("num_timesteps", 1000),
            num_epochs=params.get("num_epochs", 100),
            batch_size=params.get("batch_size", 1024),
            lr=params.get("lr", 0.002),
            weight_decay=params.get("weight_decay", 1e-4),
            dim_t=params.get("dim_t", 128),
            d_layers=params.get("d_layers", [256, 256, 256]),
            dropout=params.get("dropout", 0.0),
            scheduler=params.get("scheduler", "cosine"),
            gaussian_loss_type=params.get("gaussian_loss_type", "mse"),
            device=params.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def save_artifacts(self, path: Path) -> None:
        """
        Save model artifacts to disk.

        Parameters
        ----------
        path : Path
            Directory to save artifacts.
        """
        if self.diffusion_ is None or self.model_ is None:
            raise RuntimeError("Nothing to save: model is not fitted.")

        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(
            self.model_.state_dict(),
            path / "model.pt"
        )

        # Save EMA model weights
        if self.ema_model_ is not None:
            torch.save(
                self.ema_model_.state_dict(),
                path / "model_ema.pt"
            )

        # Save metadata
        with open(path / "tabddpm_artifacts.pkl", "wb") as f:
            pickle.dump(
                {
                    "num_numerical_features": self.num_numerical_features_,
                    "num_classes": self.num_classes_,
                    "numerical_cols": self.numerical_cols_,
                    "categorical_cols": self.categorical_cols_,
                    "column_order": self.column_order_,
                    "target_col": self.target_col_,
                    "loss_history": self.loss_history_,
                    "fitted": self.fitted_,
                },
                f,
            )

    @classmethod
    def load_artifacts(cls, path: Path) -> "TabDDPMGenerative":
        """
        Load model artifacts from disk.

        Parameters
        ----------
        path : Path
            Directory containing saved artifacts.

        Returns
        -------
        TabDDPMGenerative
            Model instance with loaded artifacts.
        """
        path = path.resolve()

        # Load metadata
        artifacts_path = path / "tabddpm_artifacts.pkl"
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"tabddpm_artifacts.pkl not found in {path}"
            )

        with open(artifacts_path, "rb") as f:
            payload = pickle.load(f)

        obj = cls()
        obj.num_numerical_features_ = payload.get("num_numerical_features", 0)
        obj.num_classes_ = payload.get("num_classes", np.array([0]))
        obj.numerical_cols_ = payload.get("numerical_cols", [])
        obj.categorical_cols_ = payload.get("categorical_cols", [])
        obj.column_order_ = payload.get("column_order", [])
        obj.target_col_ = payload.get("target_col")
        obj.loss_history_ = payload.get("loss_history", [])
        obj.fitted_ = payload.get("fitted", False)

        # Load model weights
        device = torch.device(obj.device)
        model_path = path / "model_ema.pt"
        if not model_path.exists():
            model_path = path / "model.pt"

        if model_path.exists():
            # Rebuild model architecture
            d_in = obj.num_numerical_features_
            if obj.num_classes_[0] > 0:
                d_in += int(obj.num_classes_.sum())

            model_params = {
                'd_in': d_in,
                'num_classes': 0,  # Always 0
                'is_y_cond': False,  # Always False
                'rtdl_params': {
                    'd_layers': obj.d_layers,
                    'dropout': obj.dropout,
                },
                'dim_t': obj.dim_t,
            }

            obj.model_ = MLPDiffusion(**model_params)
            obj.model_.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            obj.model_.to(device)

            # Rebuild diffusion model
            obj.diffusion_ = GaussianMultinomialDiffusion(
                num_classes=obj.num_classes_,
                num_numerical_features=obj.num_numerical_features_,
                denoise_fn=obj.model_,
                num_timesteps=obj.num_timesteps,
                gaussian_loss_type=obj.gaussian_loss_type,
                scheduler=obj.scheduler,
                device=device,
            )
            obj.diffusion_.to(device)
            obj.diffusion_.eval()
            obj.ema_model_ = obj.model_

        return obj
