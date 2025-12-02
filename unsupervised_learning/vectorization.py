import os

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

#os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")  # hard-disable CUDA in Numba
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # hide GPUs from CUDA runtime

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute as tsf_impute
from tensorflow.keras import layers, models, optimizers


class BaseVectorizer:

    """Vectorizers must expose: partial_fit(batch), transform(batch)."""
    def __init__(self):
        self.needs_two_pass = False
        self.batch_data_points = 20_000_000

    def partial_fit(self, x: np.ndarray):
        """Optional fit; no-op by default to support pure-transform vectorizers."""
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TSFreshVectorizer(BaseVectorizer):
    """
    Vectorize 1D windows using TSFresh.

    Input  : x.shape == (B, L), univariate windows.
    Output : np.ndarray with one feature vector per window.

    Parameters
    ----------
    fc_params : {'minimal', 'efficient', 'comprehensive'} or tsfresh settings instance
        Which predefined TSFresh feature set to use.
    impute : bool
        Whether to impute missing values in TSFresh output.
    n_jobs : int
        Number of parallel jobs for TSFresh (0/-1 means all cores).
    """

    def __init__(
        self,
        fc_params: str | object = "efficient",
        impute: bool = True,
        n_jobs: int = 0,
    ):
        super().__init__()
        self.impute = impute
        self.n_jobs = n_jobs
        self.batch_data_points = 1_500_000_000

        # Choose feature set
        if isinstance(fc_params, str):
            fc_params = fc_params.lower()
            if fc_params == "minimal":
                self.fc_params = MinimalFCParameters()
            elif fc_params == "efficient":
                self.fc_params = EfficientFCParameters()
            elif fc_params == "comprehensive":
                self.fc_params = ComprehensiveFCParameters()
            else:
                raise ValueError(
                    "Unknown fc_params string. Use 'minimal', 'efficient', "
                    "'comprehensive', or pass a custom settings object."
                )
        else:
            # Assume user passed a tsfresh settings object
            self.fc_params = fc_params

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(
                f"TSFreshVectorizer expects univariate windows shaped (B, L); got {x.shape}"
            )
        B, L = x.shape
        x = x.astype(np.float32)
        if B == 0:
            return np.empty((0, 0), dtype=float)

        # Long-format dataframe for tsfresh: columns (id, time, value)
        ids = np.repeat(np.arange(B, dtype=np.uint32), L)
        values = x.reshape(-1)

        df = pd.DataFrame(
            {"id": ids, "value": values},
            copy=False,
        )
        with threadpool_limits(limits=1):
            feats = extract_features(
                df,
                column_id="id",
                default_fc_parameters=self.fc_params,
                n_jobs=self.n_jobs,
                disable_progressbar=True,
            )

        if self.impute:
            tsf_impute(feats)

        # Ensure rows are in window order [0..B-1] and return as float ndarray
        feats = feats.sort_index()
        if feats.shape[0] != B:
            feats = feats.reindex(range(B), copy=False)
        feats = feats.to_numpy(dtype=np.float32)
        return feats


class AutoencoderVectorizer(BaseVectorizer):
    """
    Vectorizer using a 1D-CNN Autoencoder based on Chameleon CNN layers.
    Produces feature vectors consisting of:
       - reconstruction error
       - latent vector (optional)
    """

    def __init__(self, window_size, latent_dim=32, include_latent=True):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.include_latent = include_latent

        # ===========================
        # Encoder (CNN)
        # ===========================
        inp = layers.Input(shape=(window_size, 1))  # 1D signal

        x = layers.Conv1D(16, kernel_size=5, strides=2, padding="same", activation="relu")(inp)
        x = layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)

        encoded = layers.Dense(latent_dim, activation="relu")(x)

        # ===========================
        # Decoder (mirror of encoder)
        # ===========================
        x = layers.Dense((window_size // 8) * 64, activation="relu")(encoded)
        x = layers.Reshape((window_size // 8, 64))(x)

        x = layers.Conv1DTranspose(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv1DTranspose(16, kernel_size=5, strides=2, padding="same", activation="relu")(x)
        decoded = layers.Conv1DTranspose(1, kernel_size=5, strides=2, padding="same", activation="linear")(x)

        # ===========================
        # Build models
        # ===========================
        self.autoencoder = models.Model(inp, decoded)
        self.encoder = models.Model(inp, encoded)

        self.autoencoder.compile(
            loss="mse",
            optimizer=optimizers.Adam(learning_rate=1e-3)
        )

        self.needs_two_pass = True  # training required
        self._trained = False

    # ==================================================================
    # Training phase
    # ==================================================================
    def partial_fit(self, windows: np.ndarray):
        """
        Train autoencoder on window batches.
        windows shape: (batch, window_size)
        """
        windows = windows[..., np.newaxis]  # add channel dimension

        self.autoencoder.fit(
            windows,
            windows,
            epochs=3,
            batch_size=128,
            verbose=0
        )
        self._trained = True
        return self

    # ==================================================================
    # Inference phase
    # ==================================================================
    def transform(self, windows: np.ndarray) -> np.ndarray:
        """
        Converts windows into anomaly features:
          - reconstruction error
          - latent vector (optional)
        """
        if not self._trained:
            raise RuntimeError("AutoencoderVectorizer must be trained using partial_fit() before calling transform()!")

        windows_cnn = windows[..., np.newaxis]

        # Reconstruction
        recon = self.autoencoder.predict(windows_cnn, verbose=0)

        # Mean squared error per window
        errors = np.mean((windows_cnn - recon) ** 2, axis=(1, 2))
        errors = errors.reshape(-1, 1)

        if self.include_latent:
            latent = self.encoder.predict(windows_cnn, verbose=0)
            return latent

        return errors
