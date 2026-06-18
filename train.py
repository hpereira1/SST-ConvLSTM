"""
Sequential training script for SST model variants — TCC progression study.

Trains the progression variants in order, with identical hyperparameters
across the v0..v5 group so that the only difference there is the
architectural switch. The vbase variants are unregularized baselines:

    vbase         — 1× ConvLSTM2D, no BN, no dropout (minimal floor)
    vbase2camadas — 2× ConvLSTM2D, no BN, no dropout (isolates regularization)
    v0            — encoder only (baseline)
    v3            — encoder + multihead heads
    v4            — encoder + skip connection (residual learning)
    v5            — encoder + multihead + skip connection (= legacy v2 final)

Each variant produces its own run directory under runs/ with:
    convlstm_<variant>_masked_seed<seed>.keras
    convlstm_<variant>_masked_seed<seed>_history.json
    convlstm_<variant>_masked_seed<seed>_metadata.json
    convlstm_<variant>_masked_seed<seed>_training_log.csv
    norm_stats.npz
    checkpoints/

Usage:
    python train2.py                              # all 6 variants
    python train2.py --variants vbase             # only the minimal baseline
    python train2.py --variants vbase vbase2camadas  # both baselines
    python train2.py --variants v4 v5             # only v4 and v5
    python train2.py --epochs 50 --seed 42        # override hyperparams
"""

import os
import gc
import json
import time
import random
import platform
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

from src.utils.config import Config
from src.preprocessing.sst_preprocessing import load_and_preprocess_sst
from src.utils.tf_data_loader import SSTDatasetBuilder
from src.models.convlstm_model import build_convlstm_variant
from src.utils.losses import MaskedMSE, create_masked_metrics
from src.callbacks.custom_callbacks import ModelCheckpointWithBest


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[SEED] Random seed fixed: {seed}")


def setup_gpu(enable_memory_growth: bool = True, use_mixed_precision: bool = True):
    """
    Configure GPU settings

    Args:
        enable_memory_growth: Enable incremental GPU memory allocation
        use_mixed_precision: Use mixed precision training (float16)
    """
    if enable_memory_growth:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Enabled memory growth for {gpu}")
            except RuntimeError as e:
                print(f"Could not enable memory growth: {e}")

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision training enabled (float16)")


# --------------------------------------------------------------------------- #
# Variant taxonomy                                                            #
# --------------------------------------------------------------------------- #

# Variants trained without any dropout (baselines + Phase 1 *_clean +
# Phase 2 *_clean_nobn).
CLEAN_VARIANTS = (
    'vbase', 'vbase2camadas',
    'v0_clean', 'v3_clean', 'v4_clean', 'v5_clean',
    'v3_clean_nobn', 'v4_clean_nobn', 'v5_clean_nobn',
)

# Variants trained without BatchNormalization (Phase 1 *_nobn + Phase 2 *_clean_nobn).
NOBN_VARIANTS = (
    'v4_nobn', 'v5_nobn',
    'v3_clean_nobn', 'v4_clean_nobn', 'v5_clean_nobn',
)

# Complete list of variants accepted on the CLI and registered in
# convlstm_model.build_convlstm_variant.
ALL_VARIANTS = (
    'vbase', 'vbase2camadas',
    'v0', 'v3', 'v4', 'v5',
    'v0_clean', 'v3_clean', 'v4_clean', 'v5_clean',
    'v4_nobn', 'v5_nobn',
    'v3_clean_nobn', 'v4_clean_nobn', 'v5_clean_nobn',
)


# --------------------------------------------------------------------------- #
# Run-directory layout                                                        #
# --------------------------------------------------------------------------- #

def make_run_dir(variant: str, config: Config, seed: int) -> Path:
    """Build the run directory path following the convention used in runs/."""
    today = datetime.now().strftime("%Y%m%d")
    bf = config.model.base_filters
    bs = config.training.batch_size
    lb = config.data.lookback
    nh = len(config.data.horizons)
    name = (
        f"convlstm_{variant}_masked_bf{bf}_bs{bs}_lb{lb}_h{nh}"
        f"_seed{seed}_{today}"
    )
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def save_norm_stats(run_dir: Path, data: dict):
    """Save normalization statistics for inference/evaluation reuse."""
    np.savez(
        run_dir / "norm_stats.npz",
        mu=data['mean'].astype(np.float32),
        sigma=data['std'].astype(np.float32),
        ocean_mask=data['ocean_mask'].astype(bool),
    )


# --------------------------------------------------------------------------- #
# Metadata                                                                    #
# --------------------------------------------------------------------------- #

def save_metadata(
    run_dir: Path, variant: str, history, training_time: float,
    model, data, config: Config, seed: int, loss_name: str,
    valid_train: int, valid_val: int,
    eff_dropout: float, eff_recurrent_dropout: float, eff_spatial_dropout: float,
    eff_use_bn: bool,
):
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    gpus = tf.config.list_physical_devices('GPU')
    gpu_name = str(gpus[0].name) if gpus else 'CPU'

    val_losses = history.history['val_loss']
    best_epoch = int(np.argmin(val_losses)) + 1
    epochs_completed = len(val_losses)

    sigma_ocean = data['std'][data['ocean_mask']]
    trainable_params = int(np.sum(
        [np.prod(v.shape) for v in model.trainable_variables]
    ))

    metadata = {
        # Reproducibility
        'seed': seed,
        'git_commit': git_hash,
        'tensorflow_version': tf.__version__,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'gpu': gpu_name,
        # Performance
        'training_time_seconds': round(training_time, 2),
        'training_time_minutes': round(training_time / 60, 2),
        'avg_epoch_time_seconds': round(training_time / epochs_completed, 2),
        'trainable_params': trainable_params,
        # Convergence
        'epochs_completed': epochs_completed,
        'epochs_configured': config.training.epochs,
        'early_stopping_triggered': epochs_completed < config.training.epochs,
        'best_epoch': best_epoch,
        'best_val_loss': float(min(val_losses)),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(val_losses[-1]),
        # Dataset
        'ocean_pixels': int(np.sum(data['ocean_mask'])),
        'sigma_mean_celsius': float(np.mean(sigma_ocean)),
        'sigma_std_celsius': float(np.std(sigma_ocean)),
        'sigma_min_celsius': float(np.min(sigma_ocean)),
        'sigma_max_celsius': float(np.max(sigma_ocean)),
        'valid_train_windows': int(valid_train),
        'valid_val_windows': int(valid_val),
        # Configuration
        'loss_function': loss_name,
        'model_variant': variant,
        'base_filters': config.model.base_filters,
        'batch_size': config.training.batch_size,
        'learning_rate': config.training.learning_rate,
        'lookback': config.data.lookback,
        'horizons': list(config.data.horizons),
        'dropout': eff_dropout,
        'recurrent_dropout': eff_recurrent_dropout,
        'spatial_dropout': eff_spatial_dropout,
        'batch_normalization': eff_use_bn,
        'mixed_precision': config.training.use_mixed_precision,
        'clipnorm': config.training.clipnorm,
    }

    path = run_dir / f"convlstm_{variant}_masked_seed{seed}_metadata.json"
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {path}")


# --------------------------------------------------------------------------- #
# Per-variant training                                                        #
# --------------------------------------------------------------------------- #

def train_variant(
    variant: str, config: Config, data: dict, dataset_builder: SSTDatasetBuilder,
    valid_train: int, valid_val: int, seed: int,
):
    """Train a single variant. Independent of any prior run."""
    print("\n" + "#" * 80)
    print(f"# TRAINING VARIANT {variant.upper()}")
    print("#" * 80)

    # Reset Keras state so layer names don't collide across variants
    tf.keras.backend.clear_session()
    set_seed(seed)

    # Re-enable mixed precision after clear_session (global policy survives,
    # but explicit re-set is safe)
    if config.training.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    run_dir = make_run_dir(variant, config, seed)
    save_norm_stats(run_dir, data)

    # Re-create datasets (constants need to be in the fresh Keras session)
    train_dataset = dataset_builder.create_train_dataset(
        batch_size=config.training.batch_size, shuffle=True,
    )
    val_dataset = dataset_builder.create_val_dataset(
        batch_size=config.training.batch_size,
    )

    steps_per_epoch  = valid_train // config.training.batch_size
    validation_steps = None  # let Keras exhaust the val dataset

    # Variants without dropout regularization (vbase baselines + Phase 1 *_clean).
    # We force zero dropouts so the builder, the model, and the metadata all
    # agree on what was actually used.
    if variant in CLEAN_VARIANTS:
        eff_dropout = 0.0
        eff_recurrent_dropout = 0.0
        eff_spatial_dropout = 0.0
    else:
        eff_dropout = config.model.dropout
        eff_recurrent_dropout = config.model.recurrent_dropout
        eff_spatial_dropout = config.model.spatial_dropout

    # BatchNormalization is disabled for two groups:
    #  - Phase 1 *_nobn variants (ablation test for BN under mixed precision)
    #  - vbase / vbase2camadas baselines (raw ConvLSTM floor, no regularization)
    eff_use_bn = variant not in NOBN_VARIANTS and variant not in ('vbase', 'vbase2camadas')

    # vbase uses a single ConvLSTM2D layer; everything else uses 2.
    eff_n_layers = 1 if variant == 'vbase' else 2

    # Build model
    input_shape = (config.data.lookback, data['height'], data['width'], 1)
    n_horizons = len(config.data.horizons)
    model = build_convlstm_variant(
        input_shape=input_shape,
        n_horizons=n_horizons,
        variant=variant,
        base_filters=config.model.base_filters,
        kernel_size=config.model.conv_kernel_size,
        dropout=eff_dropout,
        recurrent_dropout=eff_recurrent_dropout,
        spatial_dropout=eff_spatial_dropout,
        use_bn=eff_use_bn,
        n_layers=eff_n_layers,
    )
    print(f"\n  Model: {model.name}")
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    print(f"  Trainable params: {trainable:,}")

    # Compile (always MaskedMSE for this study)
    loss_fn = MaskedMSE(data['ocean_mask'])
    loss_name = "MaskedMSE (z-score space)"
    metrics_dict = create_masked_metrics(data['ocean_mask'])
    model.compile(
        optimizer=Adam(
            learning_rate=config.training.learning_rate,
            clipnorm=config.training.clipnorm,
        ),
        loss=loss_fn,
        metrics=[metrics_dict['masked_mae'], metrics_dict['masked_rmse']],
    )

    # Paths for callbacks
    keras_path = str(run_dir / f"convlstm_{variant}_masked_seed{seed}.keras")
    csv_path   = str(run_dir / f"convlstm_{variant}_masked_seed{seed}_training_log.csv")

    callbacks = [
        TensorBoard(
            log_dir=str(run_dir / "tensorboard"),
            histogram_freq=1, write_graph=False,
        ),
        ModelCheckpointWithBest(
            checkpoint_dir=str(run_dir / "checkpoints"),
            model_save_path=keras_path,
            monitor='val_loss', mode='min', save_freq=5,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1,
        ),
        CSVLogger(csv_path, append=False),
    ]

    print(f"\n  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs (max): {config.training.epochs}")
    print(f"  Run directory: {run_dir}")
    print("=" * 80)

    t0 = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.training.epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )
    training_time = time.time() - t0

    # Save final model (overwrites best-saved file with final weights;
    # restore_best_weights=True in EarlyStopping means these are the best weights)
    model.save(keras_path)
    print(f"\n  Final model saved to: {keras_path}")

    # Save history JSON
    history_path = run_dir / f"convlstm_{variant}_masked_seed{seed}_history.json"
    with open(history_path, 'w') as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2,
        )
    print(f"  History saved to: {history_path}")

    # Metadata
    save_metadata(
        run_dir=run_dir, variant=variant, history=history,
        training_time=training_time, model=model, data=data,
        config=config, seed=seed, loss_name=loss_name,
        valid_train=valid_train, valid_val=valid_val,
        eff_dropout=eff_dropout,
        eff_recurrent_dropout=eff_recurrent_dropout,
        eff_spatial_dropout=eff_spatial_dropout,
        eff_use_bn=eff_use_bn,
    )

    print(f"\n  ✔ {variant} done in {training_time/60:.1f} min")
    print(f"    best_val_loss = {min(history.history['val_loss']):.6f}")

    # Free GPU memory before next variant
    del model, history
    gc.collect()
    tf.keras.backend.clear_session()


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Sequential training of ConvLSTM variants v0/v3/v4/v5"
    )
    parser.add_argument(
        '--variants', nargs='+',
        default=['vbase', 'vbase2camadas', 'v0', 'v3', 'v4', 'v5'],
        choices=list(ALL_VARIANTS),
        help=(
            "Variants to train, in order. "
            f"Accepted: {' '.join(ALL_VARIANTS)}. "
            "Default: vbase vbase2camadas v0 v3 v4 v5 (original progression)."
        ),
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--base-filters', type=int, default=None)
    parser.add_argument('--data-path', type=str, default=None)

    args = parser.parse_args()

    # Build config and apply overrides
    config = Config()
    if args.data_path:      config.data.nc_path = args.data_path
    if args.epochs:         config.training.epochs = args.epochs
    if args.batch_size:     config.training.batch_size = args.batch_size
    if args.learning_rate:  config.training.learning_rate = args.learning_rate
    if args.base_filters:   config.model.base_filters = args.base_filters

    print("=" * 80)
    print("Sequential variant training (TCC progression study)")
    print(f"  Variants:    {args.variants}")
    print(f"  Seed:        {args.seed}")
    print(f"  Epochs (cap): {config.training.epochs}")
    print(f"  Loss:        MaskedMSE (z-score space) — fixed for this study")
    print("=" * 80)

    setup_gpu(
        enable_memory_growth=config.training.enable_memory_growth,
        use_mixed_precision=config.training.use_mixed_precision,
    )

    # Load data ONCE — identical input for every variant
    print("\nLoading data (shared across variants)...")
    data = load_and_preprocess_sst(
        nc_path=config.data.nc_path,
        train_slice=slice(*config.data.train_slice),
        val_slice=slice(*config.data.val_slice),
        test_slice=slice(*config.data.test_slice),
        mask_mode=config.data.mask_mode,
        ice_threshold=config.data.ice_threshold,
    )
    dataset_builder = SSTDatasetBuilder(
        sst_train_norm=data['sst_train_norm'].values,
        sst_val_norm=data['sst_val_norm'].values,
        ocean_mask=data['ocean_mask'],
        lookback=config.data.lookback,
        horizons=config.data.horizons,
        min_valid=config.data.min_valid,
    )

    # Count valid windows ONCE — same for all variants
    print("\nCounting filter-passing windows (once, shared across variants)...")
    valid_train = dataset_builder.count_valid_windows('train')
    valid_val   = dataset_builder.count_valid_windows('val')
    print(f"  Train windows: {valid_train}  (steps/epoch = {valid_train // config.training.batch_size})")
    print(f"  Val windows:   {valid_val}  (val_steps/epoch = {valid_val // config.training.batch_size})")

    overall_start = time.time()
    for variant in args.variants:
        train_variant(
            variant=variant, config=config, data=data,
            dataset_builder=dataset_builder,
            valid_train=valid_train, valid_val=valid_val, seed=args.seed,
        )

    print("\n" + "=" * 80)
    print(f"All variants done in {(time.time() - overall_start)/60:.1f} min")
    print("=" * 80)


if __name__ == '__main__':
    main()
