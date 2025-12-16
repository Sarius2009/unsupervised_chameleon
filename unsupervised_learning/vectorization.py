import os
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

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class BaseVectorizer:

    """Vectorizers must expose: partial_fit(batch), transform(batch)."""
    def __init__(self):
        self.needs_two_pass = False
        self.batch_data_points = 2_000_000

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


# ============================================================================
# Autoencoder Vectorizer (1D-CNN, Chameleon-style)
# ============================================================================
class AutoencoderVectorizer(BaseVectorizer):
    # ============================================================================
    # Convolutional Block (Chameleon style)
    # ============================================================================
    @staticmethod
    def _conv_block(x, filters):
        y = layers.Conv1D(filters, kernel_size=64, strides=1, padding="same")(x)
        #y = layers.BatchNormalization()(y)
        #y = layers.LayerNormalization()(y)
        y = layers.GroupNormalization(groups=8, axis=-1, epsilon=1e-5)(y)
        y = layers.ReLU()(y)
        return y

    # ============================================================================
    # Residual Block (two conv blocks + skip, with 1x1 Conv for shortcut if needed)
    # ============================================================================
    @classmethod
    def _residual_block(cls, x, filters):
        shortcut = x
        y = cls._conv_block(x, filters)
        y = cls._conv_block(y, filters)

        # Adjust shortcut channels if they differ
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, kernel_size=1, padding="same")(shortcut)

        return layers.Add()([shortcut, y])
    """
    1D-CNN Autoencoder using Chameleon CNN encoder.
    Produces feature vectors consisting of:
       - reconstruction error
       - latent vector (optional)
    """
    def __init__(
            self,
            window_size,
            include_errors: bool = False,
            subsample_fraction: float = 0.15,
            base_lr: float = 1e-3,
            lr_decay: float = 0.7,
    ):
        super().__init__()
        self.batch_data_points = 3_000_000
        self.window_size = window_size
        self.include_errors = include_errors

        self.subsample_fraction = subsample_fraction
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.current_epoch = 0

        with tf.device("/GPU:0"):
            inp = layers.Input(shape=(window_size, 1))

            # ==============================
            # ENCODER (Chameleon CNN)
            # ==============================
            x = self._conv_block(inp, 16)
            x = self._residual_block(x, 16)
            x = self._residual_block(x, 32)
            x = layers.Flatten()(x)
            latent = layers.Dense(32, activation="linear", name="latent")(x)

            # ==============================
            # DECODER (mirror structure)
            # ==============================
            x = layers.Dense(self.window_size * 32)(latent)
            x = layers.Reshape((self.window_size, 32))(x)
            x = self._residual_block(x, 32)
            x = self._residual_block(x, 16)
            decoded = layers.Conv1D(
                1,
                kernel_size=1,
                padding="same",
                activation="linear",
                name="decoded",
            )(x)

            # ==============================
            # BUILD MODELS
            # ==============================
            self.autoencoder = models.Model(inp, decoded)
            self.encoder = models.Model(inp, latent)
            self.autoencoder_both = models.Model(inp, [decoded, latent])

            self.autoencoder.compile(
                loss="mse",
                optimizer=optimizers.Adam(learning_rate=self.base_lr)
            )

        self.needs_two_pass = True
        self._trained = False

        self.total_error_sum = 0.0
        self.total_error_count = 0
        self.global_avg_error = 0.0


    # ================================================================
    # TRAINING
    # ================================================================
    def partial_fit(self, windows: np.ndarray):
        """
        One gradient step on a batch of windows.
        Applies subsampling if subsample_fraction < 1.0.
        windows: np.ndarray of shape (B, L)
        """
        if windows.size == 0:
            return self

        # --- Subsample a fraction of windows (approx. global fraction) ---
        if 0.0 < self.subsample_fraction < 1.0:
            n = windows.shape[0]
            k = max(1, int(np.ceil(n * self.subsample_fraction)))
            idx = np.random.choice(n, size=k, replace=False)
            windows = windows[idx]

        # Keras expects (B, L, 1)
        windows = windows[..., np.newaxis]

        with tf.device("/GPU:0"):
            windows_tf = tf.convert_to_tensor(windows, dtype=tf.float32)
            loss = self.autoencoder.train_on_batch(windows_tf, windows_tf)
        self.global_avg_error = loss
        self._trained = True
        return self


    # ================================================================
    # FEATURE EXTRACTION
    # ================================================================
    def transform(self, windows: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Vectorizer must be trained before transform()!")

        # (B, L, 1) on GPU
        windows = windows[..., np.newaxis].astype(np.float32)

        with tf.device("/GPU:0"):
            windows_tf = tf.convert_to_tensor(windows, dtype=tf.float32)

            # Forward pass on GPU
            recon_tf, latent_tf = self.autoencoder_both(windows_tf, training=False)

            # Per-window MSE over time and channel dims
            errors_tf = tf.reduce_mean(
                tf.square(windows_tf - recon_tf),
                axis=[1, 2],
                keepdims=True,  # (B, 1)
            )

        # Back to NumPy
        errors = errors_tf.numpy().astype(np.float32)
        latent = latent_tf.numpy().astype(np.float32)

        if self.include_errors:
            return np.concatenate([errors, latent], axis=1)

        mean_error = float(errors.mean())
        self.total_error_sum += mean_error
        self.total_error_count += 1
        self.global_avg_error = self.total_error_sum / self.total_error_count

        return latent

        # ================================================================
    # EPOCH HOOK (called from train_autoencoder)
    # ================================================================
    def on_epoch_start(self, epoch_idx: int):
        self.current_epoch = epoch_idx
        if self.lr_decay is None:
            return
        new_lr = self.base_lr * (self.lr_decay ** epoch_idx)
        self.autoencoder.optimizer.learning_rate = new_lr


    def on_epoch_end(self, epoch_idx: int):
        """
        Optional: reset running error stats at end of epoch.
        This mirrors what train_autoencoder currently does.
        """
        self.total_error_sum = 0.0
        self.total_error_count = 0
        self.global_avg_error = 0.0
