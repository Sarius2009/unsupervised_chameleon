import numpy as np
from tensorflow.keras import layers, models, optimizers

# ============================================================================
# Convolutional Block (Chameleon style)
# ============================================================================
def conv_block(x, filters):
    y = layers.Conv1D(filters, kernel_size=64, strides=1, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    return y

# ============================================================================
# Residual Block (two conv blocks + skip, with 1x1 Conv for shortcut if needed)
# ============================================================================
def residual_block(x, filters):
    shortcut = x
    y = conv_block(x, filters)
    y = conv_block(y, filters)

    # Adjust shortcut channels if they differ
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding="same")(shortcut)

    return layers.Add()([shortcut, y])

# ============================================================================
# Base Vectorizer
# ============================================================================
class BaseVectorizer:
    def __init__(self):
        self.needs_two_pass = False

    def partial_fit(self, windows):
        return self

    def transform(self, windows):
        raise NotImplementedError

# ============================================================================
# Autoencoder Vectorizer (1D-CNN, Chameleon-style)
# ============================================================================
class AutoencoderVectorizer(BaseVectorizer):
    """
    1D-CNN Autoencoder using Chameleon CNN encoder.
    Produces feature vectors consisting of:
       - reconstruction error
       - latent vector (optional)
    """
    def __init__(self, window_size, include_latent=True):
        super().__init__()
        self.window_size = window_size
        self.include_latent = include_latent

        inp = layers.Input(shape=(window_size, 1))

        # ==============================
        # ENCODER (Chameleon CNN)
        # ==============================
        x = conv_block(inp, 16)
        x = residual_block(x, 16)
        x = residual_block(x, 32)
        latent = layers.GlobalAveragePooling1D()(x)

        # ==============================
        # DECODER (mirror structure)
        # ==============================
        decoder_input = layers.Dense((window_size // 4) * 32, activation="relu")(latent)
        x = layers.Reshape((window_size // 4, 32))(decoder_input)

        x = layers.UpSampling1D(size=2)(x)
        x = conv_block(x, 16)
        x = layers.UpSampling1D(size=2)(x)
        x = conv_block(x, 16)
        decoded = layers.Conv1D(1, kernel_size=64, padding="same", activation="linear")(x)

        # ==============================
        # BUILD MODELS
        # ==============================
        self.autoencoder = models.Model(inp, decoded)
        self.encoder = models.Model(inp, latent)

        self.autoencoder.compile(
            loss="mse",
            optimizer=optimizers.Adam(learning_rate=1e-3)
        )

        self.needs_two_pass = True
        self._trained = False

    # ================================================================
    # TRAINING
    # ================================================================
    def partial_fit(self, windows):
        windows = windows[..., np.newaxis]  # add channel
        self.autoencoder.fit(
            windows, windows,
            epochs=3,
            batch_size=64,
            verbose=0
        )
        self._trained = True

    # ================================================================
    # FEATURE EXTRACTION
    # ================================================================
    def transform(self, windows):
        if not self._trained:
            raise RuntimeError("Vectorizer must be trained before transform()!")

        windows = windows[..., np.newaxis]
        recon = self.autoencoder.predict(windows, verbose=0)
        errors = np.mean((windows - recon)**2, axis=(1,2)).reshape(-1,1)

        if self.include_latent:
            latent = self.encoder.predict(windows, verbose=0)
            return np.concatenate([errors, latent], axis=1)

        return errors

# ============================================================================
# Optional Identity Vectorizer
# ============================================================================
class IdentityVectorizer(BaseVectorizer):
    """Returns input windows unchanged."""
    def transform(self, windows):
        return windows
