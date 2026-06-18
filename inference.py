import tensorflow as tf
import numpy as np
import xarray as xr
from pathlib import Path
from copy import copy
import argparse
import matplotlib.pyplot as plt
import cmocean

from src.preprocessing.sst_preprocessing import (
    build_ocean_mask_bitmask,
    denormalize_predictions
)
from src.utils.losses import MaskedMSE, MaskedMAE, MaskedRMSE
from src.models.convlstm_model import LastFrameSlice


def load_model(model_path: str) -> tf.keras.Model:
    print(f"Loading model from: {model_path}")
    
    custom_objects = {
        'MaskedMSE': MaskedMSE,
        'MaskedMAE': MaskedMAE,
        'MaskedRMSE': MaskedRMSE,
        'LastFrameSlice': LastFrameSlice,
    }
    
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )
    print(f"Model loaded successfully: {model.name}")
    return model


def prepare_input_sequence(
    sst_data: np.ndarray,
    start_idx: int,
    lookback: int,
    mean: np.ndarray,
    std: np.ndarray,
    ocean_mask: np.ndarray
) -> np.ndarray:
    sequence = sst_data[start_idx - lookback:start_idx, :, :]

    normalized = (sequence - mean) / std
    normalized[:, ~ocean_mask] = np.nan
    normalized = np.nan_to_num(normalized, nan=0.0)
    normalized = normalized[np.newaxis, ..., np.newaxis]  # (1, L, H, W, 1)
    
    return normalized.astype(np.float32)


def predict_horizons(
    model: tf.keras.Model,
    input_sequence: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ocean_mask: np.ndarray,
    horizons: tuple
) -> dict:
    pred_norm = model.predict(input_sequence, verbose=0)  # (1, H, W, Hn)
    pred_norm = pred_norm[0]  # (H, W, Hn)

    predictions = {}
    for i, h in enumerate(horizons):
        pred_h = pred_norm[:, :, i]
        pred_denorm = pred_h * std + mean
        pred_denorm[~ocean_mask] = np.nan
        predictions[f'horizon_{h}'] = pred_denorm
    
    return predictions


def visualize_predictions(
    predictions: dict,
    ground_truth: dict = None,
    save_path: str = None,
    display_horizons: list = None,
    error_vlim: float = 3.0,
    title: str = None,
):
    if display_horizons is not None:
        keys = [f'horizon_{h}' for h in display_horizons if f'horizon_{h}' in predictions]
        predictions = {k: predictions[k] for k in keys}
        if ground_truth is not None:
            ground_truth = {k: ground_truth[k] for k in keys if k in ground_truth}

    n_horizons = len(predictions)

    if ground_truth is not None:
        # 3-column layout: GT | Pred | Error  (one row per horizon)
        fig, axes = plt.subplots(n_horizons, 3, figsize=(12, 4 * n_horizons))
        if n_horizons == 1:
            axes = axes[np.newaxis, :]

        col_titles = ['Ground Truth (°C)', 'Prediction (°C)', '|Erro| (°C)']
        horizon_keys = list(predictions.keys())

        sst_cmap = copy(plt.cm.RdBu_r)
        sst_cmap.set_bad('lightgrey')
        err_cmap = copy(cmocean.cm.amp)
        err_cmap.set_bad('lightgrey')

        for i, key in enumerate(horizon_keys):
            pred = predictions[key]
            gt   = ground_truth.get(key)
            err  = np.abs(pred - gt) if gt is not None else np.full_like(pred, np.nan)

            vmin = np.nanmin(gt) if gt is not None else np.nanmin(pred)
            vmax = np.nanmax(gt) if gt is not None else np.nanmax(pred)

            label = key.replace('_', ' ').title()
            im0 = axes[i, 0].imshow(gt,   cmap=sst_cmap, vmin=vmin, vmax=vmax, interpolation='bilinear', origin='lower')
            im1 = axes[i, 1].imshow(pred, cmap=sst_cmap, vmin=vmin, vmax=vmax, interpolation='bilinear', origin='lower')
            im2 = axes[i, 2].imshow(err,  cmap=err_cmap, vmin=0,    vmax=error_vlim, interpolation='bilinear', origin='lower')

            for col, im in enumerate([im0, im1, im2]):
                axes[i, col].set_title(f"{col_titles[col]} — {label}")
                axes[i, col].axis('off')
                plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
    else:
        # Original single-row prediction-only grid
        n_cols = min(4, n_horizons)
        n_rows = (n_horizons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        sst_cmap = copy(plt.cm.RdBu_r)
        sst_cmap.set_bad('lightgrey')

        axes_flat = axes.flatten()
        for idx, (key, pred) in enumerate(predictions.items()):
            ax = axes_flat[idx]
            im = ax.imshow(pred, cmap=sst_cmap, interpolation='bilinear', origin='lower')
            ax.set_title(key.replace('_', ' ').title())
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for idx in range(len(predictions), len(axes_flat)):
            axes_flat[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_random_samples(model, ds, sst, ocean_mask, mean, std, args):
    if args.output_dir is None:
        raise ValueError("--output-dir is required when --n-random > 0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = ds["time"].values
    test_mask = (times >= np.datetime64(args.test_start)) & (times <= np.datetime64(args.test_end))
    test_idx = np.where(test_mask)[0]
    if test_idx.size == 0:
        raise ValueError(f"No timesteps found in test period {args.test_start} → {args.test_end}")

    max_h = max(args.horizons)
    lo = max(int(test_idx[0]), args.lookback)
    hi = min(int(test_idx[-1]), sst.shape[0] - max_h)
    valid_indices = np.arange(lo, hi + 1, dtype=np.int64)
    if valid_indices.size < args.n_random:
        raise ValueError(
            f"Only {valid_indices.size} valid indices available in test period; "
            f"requested {args.n_random}."
        )

    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(valid_indices, size=args.n_random, replace=False))
    print(f"\nRunning inference on {args.n_random} random test samples (seed={args.seed})")
    print(f"Output dir: {out_dir}")

    sst_values = sst.values
    per_sample_rows = []
    horizon_errors = {h: {'mae': [], 'rmse': [], 'bias': []} for h in args.horizons}

    for n, t_idx in enumerate(chosen, start=1):
        date_str = np.datetime_as_string(times[t_idx], unit='D')
        input_seq = prepare_input_sequence(
            sst_values, int(t_idx), args.lookback, mean, std, ocean_mask
        )
        predictions = predict_horizons(
            model, input_seq, mean, std, ocean_mask, tuple(args.horizons)
        )

        ground_truth = {}
        sample_metrics = {}
        for h in args.horizons:
            gt = sst_values[int(t_idx) + h - 1].astype(np.float32).copy()
            gt[~ocean_mask] = np.nan
            ground_truth[f'horizon_{h}'] = gt

            pred = predictions[f'horizon_{h}']
            valid = np.isfinite(gt) & np.isfinite(pred)
            diff = pred[valid] - gt[valid]
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            bias = float(np.mean(diff))
            sample_metrics[h] = (mae, rmse, bias)
            horizon_errors[h]['mae'].append(mae)
            horizon_errors[h]['rmse'].append(rmse)
            horizon_errors[h]['bias'].append(bias)

        png_name = f"sample_{n:02d}_t{int(t_idx)}_{date_str}.png"
        visualize_predictions(
            predictions,
            ground_truth=ground_truth,
            save_path=str(out_dir / png_name),
            display_horizons=args.display_horizons,
            error_vlim=args.error_vlim,
            title=f"Amostra: {date_str}",
        )

        per_sample_rows.append((n, int(t_idx), date_str, sample_metrics))
        print(f"  [{n:2d}/{args.n_random}] t={int(t_idx)} ({date_str}) -> {png_name}")

    stats_path = out_dir / "inference_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Inference on {args.n_random} random test samples\n")
        f.write(f"Model:      {args.model}\n")
        f.write(f"Data:       {args.data}\n")
        f.write(f"Test period: {args.test_start} to {args.test_end}\n")
        f.write(f"Seed:       {args.seed}\n")
        f.write(f"Horizons:   {args.horizons}\n")
        f.write(f"Lookback:   {args.lookback}\n\n")

        f.write("=== Per-sample metrics (°C) ===\n")
        header = f"{'#':>3} {'idx':>5} {'date':<12}"
        for h in args.horizons:
            header += f" | h{h} MAE  RMSE  Bias"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for n, t_idx, date_str, sm in per_sample_rows:
            row = f"{n:>3} {t_idx:>5} {date_str:<12}"
            for h in args.horizons:
                mae, rmse, bias = sm[h]
                row += f" | {mae:5.3f} {rmse:5.3f} {bias:+5.3f}"
            f.write(row + "\n")

        f.write("\n=== Aggregate metrics across samples (°C) ===\n")
        f.write(f"{'Horizon':<8} {'MAE_mean':>10} {'MAE_std':>10} {'RMSE_mean':>10} {'RMSE_std':>10} {'Bias_mean':>10}\n")
        f.write("-" * 64 + "\n")
        for h in args.horizons:
            mae_arr = np.array(horizon_errors[h]['mae'])
            rmse_arr = np.array(horizon_errors[h]['rmse'])
            bias_arr = np.array(horizon_errors[h]['bias'])
            f.write(
                f"{str(h)+'-day':<8} "
                f"{mae_arr.mean():>10.4f} {mae_arr.std():>10.4f} "
                f"{rmse_arr.mean():>10.4f} {rmse_arr.std():>10.4f} "
                f"{bias_arr.mean():>+10.4f}\n"
            )

    print(f"\nStats saved to: {stats_path}")
    print("\n=== Aggregate metrics across samples (°C) ===")
    print(f"{'Horizon':<8} {'MAE_mean':>10} {'RMSE_mean':>10} {'Bias_mean':>10}")
    for h in args.horizons:
        mae_arr = np.array(horizon_errors[h]['mae'])
        rmse_arr = np.array(horizon_errors[h]['rmse'])
        bias_arr = np.array(horizon_errors[h]['bias'])
        print(f"{str(h)+'-day':<8} {mae_arr.mean():>10.4f} {rmse_arr.mean():>10.4f} {bias_arr.mean():>+10.4f}")


def main():
    parser = argparse.ArgumentParser(description="SST Prediction Inference")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help="Path to netCDF data file"
    )
    parser.add_argument(
        '--time-index',
        type=int,
        default=100,
        help="Time index for prediction (ignored when --n-random > 0)"
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=14,
        help="Number of lookback timesteps (must match training config)"
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Prediction horizons (must match training config)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.png',
        help="Output file for single-sample visualization"
    )
    parser.add_argument(
        '--train-start',
        type=str,
        default='1996-06-01',
        help="Training period start date for normalization stats (must match training config)"
    )
    parser.add_argument(
        '--train-end',
        type=str,
        default='2016-12-31',
        help="Training period end date for normalization stats (must match training config)"
    )
    parser.add_argument(
        '--n-random',
        type=int,
        default=0,
        help="If > 0, generate N random predictions inside the test period."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Output directory for random-sample mode (one PNG per sample + stats file)."
    )
    parser.add_argument(
        '--test-start',
        type=str,
        default='2020-01-01',
        help="Test period start date (random sampling bound)."
    )
    parser.add_argument(
        '--test-end',
        type=str,
        default='2022-05-31',
        help="Test period end date (random sampling bound)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducible sampling."
    )
    parser.add_argument(
        '--display-horizons',
        type=int,
        nargs='+',
        default=[1, 3, 5, 7],
        help="Horizons to plot (subset of --horizons). The full set is still used "
             "for metric computation; only the figure is filtered. Default: 1 3 5 7."
    )
    parser.add_argument(
        '--error-vlim',
        type=float,
        default=3.0,
        help="Upper bound (°C) for the |error| colorbar in figures. Same value "
             "across samples to keep figures visually comparable. Default: 3.0."
    )

    args = parser.parse_args()

    model = load_model(args.model)

    print(f"\nLoading data from: {args.data}")
    ds = xr.open_dataset(args.data, decode_cf=True)
    sst = ds["sst"]

    # Try to load saved normalization stats next to the model;
    # fall back to recomputing from the training slice if absent.
    stats_path = Path(args.model).parent / "norm_stats.npz"
    if stats_path.exists():
        print(f"Loading saved normalization stats from {stats_path}")
        s = np.load(stats_path, allow_pickle=False)
        mean = s["mu"].astype(np.float32)
        std = s["sigma"].astype(np.float32)
        ocean_mask = s["ocean_mask"].astype(bool)
        print(f"Ocean mask shape: {ocean_mask.shape} (from stats file)")
    else:
        print(f"No norm_stats.npz next to model — recomputing from NetCDF")
        ocean_mask = build_ocean_mask_bitmask(ds, mode="first", ice_threshold=0.0)
        print(f"Ocean mask shape: {ocean_mask.shape}")
        print(f"  Computing normalization from training period {args.train_start} → {args.train_end}")
        sst_train = sst.sel(time=slice(args.train_start, args.train_end))
        mean = sst_train.where(ocean_mask).mean("time").values.astype(np.float32)
        std = sst_train.where(ocean_mask).std("time").values.astype(np.float32)
        std = np.where((std < 1e-6) | ~np.isfinite(std), 1.0, std)

    if args.n_random > 0:
        run_random_samples(model, ds, sst, ocean_mask, mean, std, args)
        print("\nInference completed successfully!")
        return

    print(f"\nPreparing input sequence at time index {args.time_index}")
    input_seq = prepare_input_sequence(
        sst.values,
        args.time_index,
        args.lookback,
        mean,
        std,
        ocean_mask
    )
    print(f"Input shape: {input_seq.shape}")
    
    print("\nMaking predictions...")
    predictions = predict_horizons(
        model,
        input_seq,
        mean,
        std,
        ocean_mask,
        tuple(args.horizons)
    )

    print(f"Generated predictions for {len(predictions)} horizons")

    sst_values = sst.values  # (T, H, W)
    n_times = sst_values.shape[0]
    max_horizon = max(args.horizons)
    ground_truth = None

    if args.time_index + max_horizon <= n_times:
        print("\nFetching ground truth for comparison...")
        ground_truth = {}
        print(f"\n{'Horizon':<10} {'MAE (°C)':>10} {'RMSE (°C)':>10}")
        print("-" * 32)
        for h in args.horizons:
            gt_raw = sst_values[args.time_index + h - 1].astype(np.float32)  # (H, W), raw scale
            gt_denorm = gt_raw.copy()
            gt_denorm[~ocean_mask] = np.nan
            ground_truth[f'horizon_{h}'] = gt_denorm

            pred = predictions[f'horizon_{h}']
            valid = np.isfinite(gt_denorm) & np.isfinite(pred)
            diff = pred[valid] - gt_denorm[valid]
            mae  = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            print(f"{str(h)+'-day':<10} {mae:>10.4f} {rmse:>10.4f}")
    else:
        print(f"\nNote: time_index {args.time_index} + max horizon {max_horizon} exceeds "
              f"dataset length {n_times}; skipping ground truth comparison.")

    print("\nGenerating visualization...")
    single_date = np.datetime_as_string(ds["time"].values[args.time_index], unit='D')
    visualize_predictions(
        predictions,
        ground_truth=ground_truth,
        save_path=args.output,
        display_horizons=args.display_horizons,
        error_vlim=args.error_vlim,
        title=f"Amostra: {single_date}",
    )

    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()

