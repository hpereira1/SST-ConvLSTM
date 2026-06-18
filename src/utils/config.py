
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    nc_path: str = "data/raw/ostia_sst_1996-2022_clean.nc"
    lookback: int = 14
    horizons: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    min_valid: float = 0.98
    train_slice: Tuple[str, str] = ("1996-06-01", "2016-12-31")
    val_slice: Tuple[str, str] = ("2017-01-01", "2019-12-31")
    test_slice: Tuple[str, str] = ("2020-01-01", "2022-05-31")
    ice_threshold: float = 0.0
    mask_mode: str = "majority"


@dataclass
class ModelConfig:
    base_filters: int = 32
    conv_kernel_size: Tuple[int, int] = (3, 3)
    dropout: float = 0.1
    recurrent_dropout: float = 0.1
    spatial_dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8  # batch 8 cabe em GPU de 4 GB de VRAM
    epochs: int = 100  # EarlyStopping (patience=10) will stop before this ceiling
    learning_rate: float = 3e-4
    clipnorm: float = 1.0
    use_mixed_precision: bool = True
    enable_memory_growth: bool = True


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    # Output paths
    model_save_path: str = "models/convlstm_sst.keras"
    checkpoint_dir: str = "models/checkpoints"
    tensorboard_dir: str = "logs/tensorboard"
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


default_config = Config()

