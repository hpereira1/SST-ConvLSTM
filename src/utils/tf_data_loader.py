"""
TensorFlow Dataset Creation for SST Prediction
Handles efficient data loading with filtering and batching
"""

import numpy as np
import tensorflow as tf
from typing import Tuple


class SSTDatasetBuilder:
    """Builder for creating TensorFlow datasets for SST prediction"""
    
    def __init__(
        self,
        sst_train_norm: np.ndarray,
        sst_val_norm: np.ndarray,
        ocean_mask: np.ndarray,
        lookback: int,
        horizons: Tuple[int, ...],
        min_valid: float = 0.98
    ):
        """
        Initialize the dataset builder
        
        Args:
            sst_train_norm: Normalized training SST data (T, H, W)
            sst_val_norm: Normalized validation SST data (T, H, W)
            ocean_mask: Ocean mask (H, W)
            lookback: Number of past timesteps to use
            horizons: Future timesteps to predict
            min_valid: Minimum proportion of valid pixels required
        """
        self.sst_train_tf = tf.constant(sst_train_norm, dtype=tf.float32)
        self.sst_val_tf = tf.constant(sst_val_norm, dtype=tf.float32)
        self.ocean_mask = ocean_mask
        self.lookback = lookback
        self.horizons = tf.constant(horizons, dtype=tf.int32)
        self.min_valid = min_valid
        
        self.H, self.W = ocean_mask.shape
        self.max_horizon = max(horizons)
        
        self.ocean_bool = tf.constant(ocean_mask, dtype=tf.bool)
        self.ocean4_x = tf.reshape(self.ocean_bool, (1, self.H, self.W, 1))
        self.ocean_hw1 = tf.reshape(self.ocean_bool, (self.H, self.W, 1))
        self.num_ocean = tf.constant(float(np.sum(ocean_mask)), tf.float32)
        
    def count_valid_windows(self, split: str) -> int:
        """
        Count windows that pass the `min_valid` filter for the given split,
        using the same logic as `_valid_filter` but in numpy. Used to compute
        the exact `steps_per_epoch` instead of the 0.8 heuristic.

        Args:
            split: 'train' or 'val'

        Returns:
            Number of windows whose (X + Y) ocean-finite pixel ratio is
            >= self.min_valid.
        """
        if split == 'train':
            sst = self.sst_train_tf.numpy()
        elif split == 'val':
            sst = self.sst_val_tf.numpy()
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        T = sst.shape[0]
        horizons = self.horizons.numpy().tolist()
        max_h = max(horizons)
        n_ocean = float(self.num_ocean.numpy())
        total_possible = (self.lookback + len(horizons)) * n_ocean

        valid_indices = np.arange(self.lookback, T - max_h + 1, dtype=np.int32)
        passed = 0
        for idx in valid_indices:
            x = sst[idx - self.lookback:idx]
            y = np.stack([sst[idx + h - 1] for h in horizons])
            x_valid = (np.isfinite(x) & self.ocean_mask[None, :, :]).sum()
            y_valid = (np.isfinite(y) & self.ocean_mask[None, :, :]).sum()
            ratio = (x_valid + y_valid) / (total_possible + 1e-12)
            if ratio >= self.min_valid:
                passed += 1
        return int(passed)

    def _map_fn_train(self, index: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Map function for training data
        
        Args:
            index: Index of the sample
        
        Returns:
            Tuple of (X, Y) where X is (L, H, W, 1) and Y is (H, W, Hn)
        """
        index = tf.cast(index, tf.int32)
        x = self.sst_train_tf[index - self.lookback:index, :, :]  # (L, H, W)
        x = tf.expand_dims(x, -1)  # (L, H, W, 1)
        
        y_stack = tf.gather(self.sst_train_tf, index + self.horizons - 1)  # (Hn, H, W)
        y = tf.transpose(y_stack, [1, 2, 0])  # (H, W, Hn)
        
        return x, y
    
    def _map_fn_val(self, index: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Map function for validation data
        
        Args:
            index: Index of the sample
        
        Returns:
            Tuple of (X, Y) where X is (L, H, W, 1) and Y is (H, W, Hn)
        """
        index = tf.cast(index, tf.int32)
        x = self.sst_val_tf[index - self.lookback:index, :, :]
        x = tf.expand_dims(x, -1)
        
        y_stack = tf.gather(self.sst_val_tf, index + self.horizons - 1)
        y = tf.transpose(y_stack, [1, 2, 0])
        
        return x, y
    
    def _valid_filter(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Filter samples based on minimum valid pixel ratio
        
        Args:
            x: Input tensor (L, H, W, 1)
            y: Output tensor (H, W, Hn)
        
        Returns:
            Boolean indicating if sample meets minimum valid threshold
        """
        x_fin = tf.math.is_finite(x)
        y_fin = tf.math.is_finite(y)
        
        x_m = tf.logical_and(
            x_fin,
            tf.tile(self.ocean4_x, [self.lookback, 1, 1, 1])
        )
        y_m = tf.logical_and(y_fin, self.ocean_hw1)
        
        x_cnt = tf.reduce_sum(tf.cast(x_m, tf.float32))
        y_cnt = tf.reduce_sum(tf.cast(y_m, tf.float32))
        
        total_possible = (self.lookback + len(self.horizons)) * self.num_ocean
        ratio = (x_cnt + y_cnt) / (total_possible + 1e-12)
        
        return ratio >= tf.constant(self.min_valid, tf.float32)
    
    @staticmethod
    def _sanitize_batch(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Replace NaN/Inf with zeros in batch
        
        Args:
            x: Input batch
            y: Output batch
        
        Returns:
            Sanitized (x, y)
        """
        x = tf.where(tf.math.is_finite(x), x, 0.0)
        return x, y
    
    def create_train_dataset(self, batch_size: int = 8, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create training dataset
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        
        Returns:
            tf.data.Dataset for training
        """
        train_indices = np.arange(
            self.lookback,
            self.sst_train_tf.shape[0] - self.max_horizon + 1,
            dtype=np.int32
        )
        
        dataset = tf.data.Dataset.from_tensor_slices(train_indices)
        
        if shuffle:
            dataset = dataset.shuffle(len(train_indices))
        
        # Add repeat() BEFORE filter to ensure continuous data generation
        dataset = (dataset
                   .repeat()
                   .map(self._map_fn_train, num_parallel_calls=tf.data.AUTOTUNE)
                   .filter(self._valid_filter)
                   .batch(batch_size, drop_remainder=True)
                   .map(self._sanitize_batch, num_parallel_calls=tf.data.AUTOTUNE)
                   .prefetch(tf.data.AUTOTUNE))
        
        return dataset
    
    def create_val_dataset(self, batch_size: int = 8) -> tf.data.Dataset:
        """
        Create validation dataset
        
        Args:
            batch_size: Batch size
        
        Returns:
            tf.data.Dataset for validation
        """
        val_indices = np.arange(
            self.lookback,
            self.sst_val_tf.shape[0] - self.max_horizon + 1,
            dtype=np.int32
        )
        
        dataset = tf.data.Dataset.from_tensor_slices(val_indices)
        
        # Validation dataset does NOT repeat - it should run once per epoch
        dataset = (dataset
                   .map(self._map_fn_val, num_parallel_calls=tf.data.AUTOTUNE)
                   .filter(self._valid_filter)
                   .batch(batch_size, drop_remainder=False)
                   .map(self._sanitize_batch, num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .prefetch(tf.data.AUTOTUNE))
        
        return dataset

