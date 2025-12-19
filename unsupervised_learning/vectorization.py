import os
from threadpoolctl import threadpool_limits
import json



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
        if B == 0:
            return np.empty((0, 0), dtype=float)

        # Long-format dataframe for tsfresh: columns (id, value)
        ids = np.repeat(np.arange(B, dtype=np.uint32), L)
        values = x.reshape(-1)

        df = pd.DataFrame(
            {"id": ids, "value": values},
            copy=False,
        )
        with threadpool_limits(limits=4):
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
       - reconstruction error (optional)
       - latent vector
    """
    def __init__(
            self,
            window_size,
            include_errors: bool = True,
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

        loss = self.autoencoder.train_on_batch(windows, windows)
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
        windows = windows[..., np.newaxis]


        # Forward pass on GPU
        recon_tf, latent_tf = self.autoencoder_both(windows, training=False)

        errors = tf.reduce_mean(tf.square(windows - recon_tf), axis=(1, 2), keepdims=False).numpy().astype(np.float32)
        errors = errors[:, None]

        latent = latent_tf.numpy().astype(np.float32)

        mean_error = float(errors.mean())
        self.total_error_sum += mean_error
        self.total_error_count += 1
        self.global_avg_error = self.total_error_sum / self.total_error_count

        if self.include_errors:
            return np.concatenate([errors, latent], axis=1)
        else:
            return latent

    # ================================================================
    # EPOCH HOOKS (called from train_autoencoder)
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

    def save_autoencoder(self, folder: str = 'ae_model/'):

        os.makedirs(folder, exist_ok=True)

        # Save the full model that outputs (decoded, latent)
        self.autoencoder_both.save(os.path.join(folder, "autoencoder_both.keras"), include_optimizer=False)

        # Save metadata needed to reconstruct the vectorizer consistently
        meta = {
            "window_size": int(self.window_size),
            "include_errors": bool(self.include_errors),
            "subsample_fraction": float(self.subsample_fraction),
            "base_lr": float(self.base_lr),
            "lr_decay": float(self.lr_decay),
            "current_epoch": int(self.current_epoch),
            "_trained": bool(self._trained),
        }
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f)

        # Optional: persist running stats (not required for transform correctness)
        stats = {
            "total_error_sum": float(self.total_error_sum),
            "total_error_count": int(self.total_error_count),
            "global_avg_error": float(self.global_avg_error),
        }
        with open(os.path.join(folder, "stats.json"), "w") as f:
            json.dump(stats, f)


    @classmethod
    def load_autoencoder(cls, folder: str = 'ae_model/'):
        with open(os.path.join(folder, "meta.json"), "r") as f:
            meta = json.load(f)

        v = cls(
            window_size=meta["window_size"],
            include_errors=meta["include_errors"],
            subsample_fraction=meta["subsample_fraction"],
            base_lr=meta["base_lr"],
            lr_decay=meta["lr_decay"],
        )

        # Load saved model
        v.autoencoder_both = tf.keras.models.load_model(os.path.join(folder, "autoencoder_both.keras"))

        # Recreate derived models from the loaded graph
        v.autoencoder = tf.keras.Model(v.autoencoder_both.input, v.autoencoder_both.outputs[0])
        v.encoder     = tf.keras.Model(v.autoencoder_both.input, v.autoencoder_both.outputs[1])

        # Mark trained state
        v._trained = bool(meta.get("_trained", True))
        v.current_epoch = int(meta.get("current_epoch", 0))

        # Optional stats restore
        stats_path = os.path.join(folder, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            v.total_error_sum = float(stats.get("total_error_sum", 0.0))
            v.total_error_count = int(stats.get("total_error_count", 0))
            v.global_avg_error = float(stats.get("global_avg_error", 0.0))

        return v

