"""
Custom Loss Functions and Metrics for SST Prediction
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


# ---------------------------------------------------------------------------
# Masked losses in normalized (z-score) space, over ocean pixels only.
# Como o modelo prevê z-scores, uma perda uniforme em z pondera cada pixel
# igualmente em unidades de desvio-padrão (equivale a ponderação inversa à
# variância em °C: pixels de alta variabilidade contribuem menos ao gradiente).
# ---------------------------------------------------------------------------


class MaskedMSE:
    """Masked Mean Squared Error Loss for ocean-only pixels"""

    def __init__(self, ocean_mask: np.ndarray):
        """
        Initialize masked MSE loss

        Args:
            ocean_mask: Boolean array (H, W) indicating valid ocean pixels
        """
        self.ocean_mask = ocean_mask
        self.ocean_mask_tf = tf.constant(ocean_mask.astype("float32"))
        self.ocean_mask_tf = tf.reshape(
            self.ocean_mask_tf,
            (1, ocean_mask.shape[0], ocean_mask.shape[1], 1)
        )

    def get_config(self):
        """Return configuration for serialization"""
        return {
            'ocean_mask': self.ocean_mask.tolist()
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration"""
        return cls(np.array(config['ocean_mask']))

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute masked MSE loss

        Args:
            y_true: Ground truth (B, H, W, Hn)
            y_pred: Predictions (B, H, W, Hn)

        Returns:
            Scalar loss value
        """
        y_pred = tf.cast(y_pred, tf.float32)

        finite = tf.math.is_finite(y_true)  # (B, H, W, Hn)
        mask = tf.cast(finite, tf.float32) * self.ocean_mask_tf  # (1, H, W, 1) broadcasts

        diff = y_pred - tf.where(finite, y_true, 0.0)
        se = tf.square(diff)

        return tf.reduce_sum(se * mask) / (tf.reduce_sum(mask) + K.epsilon())


class MaskedMAE:
    """Masked Mean Absolute Error for ocean-only pixels"""

    def __init__(self, ocean_mask: np.ndarray):
        """
        Initialize masked MAE metric

        Args:
            ocean_mask: Boolean array (H, W) indicating valid ocean pixels
        """
        self.ocean_mask = ocean_mask
        self.ocean_mask_tf = tf.constant(ocean_mask.astype("float32"))
        self.ocean_mask_tf = tf.reshape(
            self.ocean_mask_tf,
            (1, ocean_mask.shape[0], ocean_mask.shape[1], 1)
        )

    def get_config(self):
        """Return configuration for serialization"""
        return {
            'ocean_mask': self.ocean_mask.tolist()
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration"""
        return cls(np.array(config['ocean_mask']))

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute masked MAE

        Args:
            y_true: Ground truth (B, H, W, Hn)
            y_pred: Predictions (B, H, W, Hn)

        Returns:
            Scalar MAE value
        """
        y_pred = tf.cast(y_pred, tf.float32)

        finite = tf.math.is_finite(y_true)
        mask = tf.cast(finite, tf.float32) * self.ocean_mask_tf

        diff = y_pred - tf.where(finite, y_true, 0.0)
        ae = tf.abs(diff)

        return tf.reduce_sum(ae * mask) / (tf.reduce_sum(mask) + K.epsilon())


class MaskedRMSE:
    """Masked Root Mean Squared Error for ocean-only pixels"""

    def __init__(self, ocean_mask: np.ndarray):
        """
        Initialize masked RMSE metric

        Args:
            ocean_mask: Boolean array (H, W) indicating valid ocean pixels
        """
        self.ocean_mask = ocean_mask
        self.masked_mse = MaskedMSE(ocean_mask)

    def get_config(self):
        """Return configuration for serialization"""
        return {
            'ocean_mask': self.ocean_mask.tolist()
        }

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config['ocean_mask']))

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse = self.masked_mse(y_true, y_pred)
        return tf.sqrt(mse)



def create_masked_metrics(ocean_mask: np.ndarray) -> dict:
    return {
        'masked_mse': MaskedMSE(ocean_mask),
        'masked_mae': MaskedMAE(ocean_mask),
        'masked_rmse': MaskedRMSE(ocean_mask),
    }

