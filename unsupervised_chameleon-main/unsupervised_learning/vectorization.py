import numpy as np
from tensorflow.keras import layers, models, optimizers

class BaseVectorizer:
    """Vectorizers must expose: partial_fit(batch), transform(batch)."""
    def __init__(self):
        self.needs_two_pass = False

    def partial_fit(self, x: np.ndarray):
        """Optional fit; no-op by default to support pure-transform vectorizers."""
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
class AutoencoderVectorizer(BaseVectorizer):
    """
    Vectorizer using a 1D Autoencoder.
    Produces feature vectors consisting of:
       - reconstruction error
       - (optional) latent vector
    """

    def __init__(self, window_size, latent_dim=16, include_latent=True):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.include_latent = include_latent

        # Autoencoder model
        inp = layers.Input(shape=(window_size,))

        encoded = layers.Dense(64, activation="relu")(inp)
        encoded = layers.Dense(latent_dim, activation="relu")(encoded)

        decoded = layers.Dense(64, activation="relu")(encoded)
        decoded = layers.Dense(window_size, activation="linear")(decoded)

        self.autoencoder = models.Model(inp, decoded)
        self.encoder = models.Model(inp, encoded)

        self.autoencoder.compile(
            loss="mse", optimizer=optimizers.Adam(learning_rate=1e-3)
        )

        self.needs_two_pass = True    # we MUST train once before inference
        self._trained = False

    def partial_fit(self, windows: np.ndarray):
        """
        Train autoencoder on window batches.
        windows shape: (batch, window_size)
        """
        self.autoencoder.fit(
            windows,
            windows,
            epochs=3,
            batch_size=128,
            verbose=0,
        )
        self._trained = True

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """
        Converts windows into anomaly features:
          - reconstruction error
          - latent vector (optional)
        """
        if not self._trained:
            raise RuntimeError("AutoencoderVectorizer must be trained first!")

        recon = self.autoencoder.predict(windows, verbose=0)
        errors = np.mean((windows - recon) ** 2, axis=1)
        errors = errors.reshape(-1, 1)

        if self.include_latent:
            latent = self.encoder.predict(windows, verbose=0)
            return np.concatenate([errors, latent], axis=1)

        return errors
