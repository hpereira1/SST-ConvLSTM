
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path


class ModelCheckpointWithBest(keras.callbacks.Callback):
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_save_path: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_freq: int = 1
    ):

        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_path = model_save_path
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        
        if mode == 'min':
            self.best_value = np.inf
            self.is_better = lambda new, best: new < best
        else:
            self.best_value = -np.inf
            self.is_better = lambda new, best: new > best
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return

        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.keras"
            self.model.save(checkpoint_path)
            print(f"\nCheckpoint saved to {checkpoint_path}")
        
        # Save best model
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.model.save(self.model_save_path)
            print(f"\nBest model saved to {self.model_save_path} "
                  f"({self.monitor}: {current_value:.6f})")


class LoggingCallback(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEpoch {epoch + 1} Summary:")
        for key, value in logs.items():
            print(f"  {key}: {value:.6f}")

