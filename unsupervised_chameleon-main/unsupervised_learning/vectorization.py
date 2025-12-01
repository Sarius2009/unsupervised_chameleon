import numpy as np
from tensorflow.keras import layers, models, optimizers

# ======================================================================
# Base Vectorizer
# ======================================================================

class BaseVectorizer:
    """
    Base class for feature extraction from windows.
    Vectorizers must implement:
        - partial_fit()
        - transform()
    """
    def __init__(self):
        self.needs_two_pass = False

    def partial_fit(self, windows: np.ndarray):
        """Optional: used only by trainable vectorizers (e.g., Autoencoder)."""
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ======================================================================
# Autoencoder Vectorizer (1D-CNN Version)
# ======================================================================

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
            return np.concatenate([errors, latent], axis=1)

        return errors


# ======================================================================
# (Optional) Dummy Vectorizer for compatibility
# ======================================================================

class IdentityVectorizer(BaseVectorizer):
    """Simply returns the input windows unchanged."""
    def transform(self, windows: np.ndarray) -> np.ndarray:
        return windows
