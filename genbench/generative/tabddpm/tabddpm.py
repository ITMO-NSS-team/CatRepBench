from __future__ import annotations

import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, StandardScaler, \
    OrdinalEncoder

from genbench.data.schema import TabularSchema
from genbench.generative.base import BaseGenerative, GenerativeState
from .diffusion import GaussianMultinomialDiffusion
from .modules import MLPDiffusion
from .utils import update_ema


@dataclass
class TabDdpmGenerative(BaseGenerative):
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
    fitted_: bool = False

    # preprocessing transformers
    num_transform_: Any = None
    cat_transform_: Any = None

    # data info
    num_numerical_features_: int = 0
    num_classes_: np.ndarray = field(default_factory=lambda: np.array([0]))
    column_order_: List[str] = field(default_factory=list)
    loss_history_: List[Dict[str, float]] = field(default_factory=list)

    def requires_fit(self) -> bool:
        return True

    def is_conditional(self) -> bool:
        return False

    def _preprocess_data(self, df: pd.DataFrame,
                         schema: TabularSchema) -> torch.Tensor:
        """
        Preprocess the data for TabDDPM.

        - Continuous features: quantile normalization to standard normal
        - Categorical features: ordinal encoding
        - Discrete features: kept as is (treated as numerical)
        """
        # Determine column order: continuous + discrete + categorical
        cont_cols = list(schema.continuous_cols)
        disc_cols = list(schema.discrete_cols)
        cat_cols = list(schema.categorical_cols)

        self.column_order_ = cont_cols + disc_cols + cat_cols
        self.num_numerical_features_ = len(cont_cols) + len(disc_cols)

        # Process numerical features (continuous + discrete)
        num_cols = cont_cols + disc_cols
        if num_cols:
            num_data = df[num_cols].values.astype(np.float32)

            # Fit QuantileTransformer for normalization
            n_quantiles = min(1000, max(10, len(df) // 30))
            self.num_transform_ = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=n_quantiles,
                random_state=0,
            )
            num_data = self.num_transform_.fit_transform(num_data)
        else:
            num_data = np.zeros((len(df), 0), dtype=np.float32)

        # Process categorical features
        if cat_cols:
            cat_data = df[cat_cols].values.astype(str)

            # Fit OrdinalEncoder
            self.cat_transform_ = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )
            cat_data = self.cat_transform_.fit_transform(cat_data)

            # Compute number of classes for each categorical feature
            # Handle unknown values (-1) by adding 1 to the max
            num_classes_list = []
            for i in range(cat_data.shape[1]):
                col_data = cat_data[:, i]
                # Shift values to be non-negative (handle -1 for unknown)
                min_val = col_data.min()
                if min_val < 0:
                    cat_data[:, i] = col_data - min_val
                n_classes = int(cat_data[:, i].max()) + 1
                num_classes_list.append(n_classes)

            self.num_classes_ = np.array(num_classes_list)
            cat_data = cat_data.astype(np.int64)
        else:
            cat_data = np.zeros((len(df), 0), dtype=np.int64)
            self.num_classes_ = np.array([0])

        # Concatenate numerical and categorical features
        X = np.hstack([num_data, cat_data])
        return torch.from_numpy(X).float()

    def _postprocess_data(self, X: np.ndarray) -> pd.DataFrame:
        """
        Postprocess generated data back to original format.
        """
        # Split numerical and categorical features
        num_data = X[:, :self.num_numerical_features_]
        cat_data = X[:, self.num_numerical_features_:]

        # Inverse transform numerical features
        if self.num_transform_ is not None and num_data.shape[1] > 0:
            num_data = self.num_transform_.inverse_transform(num_data)

        # Inverse transform categorical features
        if self.cat_transform_ is not None and cat_data.shape[1] > 0:
            # Round categorical values to nearest integer
            cat_data = np.round(cat_data).astype(np.int64)
            # Clip to valid range
            for i in range(cat_data.shape[1]):
                cat_data[:, i] = np.clip(cat_data[:, i], 0,
                                         self.num_classes_[i] - 1)
            cat_data = self.cat_transform_.inverse_transform(cat_data)

        # Build DataFrame
        n_num = self.num_numerical_features_
        n_cat = len(self.num_classes_) if self.num_classes_[0] > 0 else 0

        columns = self.column_order_
        data = {}

        if n_num > 0:
            for i, col in enumerate(columns[:n_num]):
                data[col] = num_data[:, i]

        if n_cat > 0:
            for i, col in enumerate(columns[n_num:n_num + n_cat]):
                data[col] = cat_data[:, i]

        return pd.DataFrame(data)

    def fit(self, df: pd.DataFrame,
            schema: TabularSchema) -> "TabDdpmGenerative":
        """
        Fit the TabDDPM model on the provided data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        schema : TabularSchema
            Schema describing the data types.

        Returns
        -------
        self : TabDdpmGenerative
            Fitted model instance.
        """
        device = torch.device(self.device)

        # Preprocess data
        X = self._preprocess_data(df, schema)
        X = X.to(device)

        # Create dummy labels (for unconditional generation)
        y = torch.zeros(X.shape[0], dtype=torch.long, device=device)

        # Build model
        d_in = X.shape[1]

        # Adjust d_in for one-hot encoding of categorical features
        d_in_adjusted = self.num_numerical_features_ + int(
            self.num_classes_.sum())
        if self.num_classes_[0] == 0:
            d_in_adjusted = self.num_numerical_features_

        model_params = {
            'd_in': d_in,
            'num_classes': 0,  # unconditional
            'is_y_cond': False,
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
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        for epoch in range(self.num_epochs):
            epoch_losses = []
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                out_dict = {'y': batch_y}
                loss_multi, loss_gauss = self.diffusion_.mixed_loss(batch_x,
                                                                    out_dict)
                loss = loss_multi + loss_gauss
                loss.backward()
                optimizer.step()

                # Anneal learning rate
                frac_done = step / n_steps
                new_lr = self.lr * (1 - frac_done)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

                # Update EMA
                update_ema(self.ema_model_.parameters(),
                           self.model_.parameters())

                epoch_losses.append(loss.item())
                step += 1

            avg_loss = np.mean(epoch_losses)
            self.loss_history_.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'multinomial_loss': loss_multi.item(),
                'gaussian_loss': loss_gauss.item(),
            })

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Loss: "
                    f"{avg_loss:.4f}")

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
            Not supported for TabDDPM (unconditional generation).

        Returns
        -------
        pd.DataFrame
            Generated synthetic data.
        """
        if conditions is not None:
            raise NotImplementedError(
                "TabDdpmGenerative does not support conditional sampling.")
        if not self.fitted_ or self.diffusion_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        device = torch.device(self.device)

        # Create dummy class distribution for unconditional sampling
        y_dist = torch.ones(1, device=device)  # Single class (unconditional)

        # Sample from diffusion model
        with torch.no_grad():
            X_gen, _ = self.diffusion_.sample_all(
                num_samples=n,
                batch_size=min(self.batch_size, n),
                y_dist=y_dist,
                ddim=False,
            )

        # Postprocess generated data
        X_gen_np = X_gen.numpy()
        return self._postprocess_data(X_gen_np)

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
            'multinomial_loss': [h['multinomial_loss'] for h in
                                 self.loss_history_],
            'gaussian_loss': [h['gaussian_loss'] for h in self.loss_history_],
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
    def from_state(cls, state: GenerativeState) -> "TabDdpmGenerative":
        """
        Create a model instance from a state object.

        Parameters
        ----------
        state : GenerativeState
            State object with model parameters.

        Returns
        -------
        TabDdpmGenerative
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
            device=params.get("device",
                              "cuda" if torch.cuda.is_available() else "cpu"),
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

        # Save preprocessing transformers and other artifacts
        with open(path / "tabddpm_artifacts.pkl", "wb") as f:
            pickle.dump(
                {
                    "num_transform": self.num_transform_,
                    "cat_transform": self.cat_transform_,
                    "num_numerical_features": self.num_numerical_features_,
                    "num_classes": self.num_classes_,
                    "column_order": self.column_order_,
                    "loss_history": self.loss_history_,
                    "fitted": self.fitted_,
                },
                f,
            )

    @classmethod
    def load_artifacts(cls, path: Path) -> "TabDdpmGenerative":
        """
        Load model artifacts from disk.

        Parameters
        ----------
        path : Path
            Directory containing saved artifacts.

        Returns
        -------
        TabDdpmGenerative
            Model instance with loaded artifacts.
        """
        path = path.resolve()

        # Load preprocessing artifacts
        artifacts_path = path / "tabddpm_artifacts.pkl"
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"tabddpm_artifacts.pkl not found in {path}")

        with open(artifacts_path, "rb") as f:
            payload = pickle.load(f)

        obj = cls()
        obj.num_transform_ = payload.get("num_transform")
        obj.cat_transform_ = payload.get("cat_transform")
        obj.num_numerical_features_ = payload.get("num_numerical_features", 0)
        obj.num_classes_ = payload.get("num_classes", np.array([0]))
        obj.column_order_ = payload.get("column_order", [])
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
                'num_classes': 0,
                'is_y_cond': False,
                'rtdl_params': {
                    'd_layers': obj.d_layers,
                    'dropout': obj.dropout,
                },
                'dim_t': obj.dim_t,
            }

            obj.model_ = MLPDiffusion(**model_params)
            obj.model_.load_state_dict(
                torch.load(model_path, map_location=device))
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
