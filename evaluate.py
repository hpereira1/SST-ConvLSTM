
import argparse
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

import xarray as xr

from src.preprocessing.sst_preprocessing import load_and_preprocess_sst
from src.utils.losses import MaskedMSE, MaskedMAE, MaskedRMSE
from src.models.convlstm_model import LastFrameSlice
from src.utils.config import Config


def load_model(model_path: str) -> tf.keras.Model:
    custom_objects = {
        'MaskedMSE': MaskedMSE,
        'MaskedMAE': MaskedMAE,
        'MaskedRMSE': MaskedRMSE,
        'LastFrameSlice': LastFrameSlice,
    }
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    print(f"Model loaded: {model.name}")
    return model


def collect_predictions(
    model: tf.keras.Model,
    sst_norm: np.ndarray,
    lookback: int,
    horizons: tuple,
    batch_size: int = 4,
    max_batches: int = None,
) -> tuple:
    max_h = max(horizons)
    valid_indices = np.arange(lookback, sst_norm.shape[0] - max_h + 1, dtype=np.int32)

    all_y_true, all_y_pred, all_y_persist = [], [], []

    n_batches = (len(valid_indices) + batch_size - 1) // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    for b in range(n_batches):
        if b % 50 == 0:
            print(f"  batch {b}/{n_batches}", flush=True)

        batch_idx = valid_indices[b * batch_size: (b + 1) * batch_size]
        X_list, Y_list, P_list = [], [], []
        for idx in batch_idx:
            x = sst_norm[idx - lookback:idx]
            x = np.nan_to_num(x, nan=0.0)[..., np.newaxis]
            y = np.stack([sst_norm[idx + h - 1] for h in horizons], axis=-1)
            persist = sst_norm[idx - 1]
            X_list.append(x); Y_list.append(y); P_list.append(persist)

        X_batch = np.array(X_list, dtype=np.float32)
        Y_batch = np.array(Y_list, dtype=np.float32)
        P_batch = np.array(P_list, dtype=np.float32)

        preds = model.predict(X_batch, verbose=0)

        all_y_true.append(Y_batch)
        all_y_pred.append(preds)
        all_y_persist.append(P_batch)

    return (
        np.concatenate(all_y_true,    axis=0),
        np.concatenate(all_y_pred,    axis=0),
        np.concatenate(all_y_persist, axis=0),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persist: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ocean_mask: np.ndarray,
    horizons: tuple,
) -> list:
    mean_4d = mean[np.newaxis, :, :, np.newaxis]
    std_4d  = std[np.newaxis, :, :, np.newaxis]

    y_true_d = y_true * std_4d + mean_4d
    y_pred_d = y_pred * std_4d + mean_4d
    y_persist_d = y_persist * std[np.newaxis] + mean[np.newaxis]

    mask_4d = np.broadcast_to(
        ocean_mask[np.newaxis, :, :, np.newaxis],
        y_true_d.shape,
    )
    y_true_d[~mask_4d] = np.nan
    y_pred_d[~mask_4d] = np.nan
    y_persist_d = np.where(ocean_mask[np.newaxis, :, :], y_persist_d, np.nan)

    clim_2d = mean.astype(np.float64)  # (H, W)

    results = []
    for i, h in enumerate(horizons):
        t = y_true_d[:, :, :, i]
        p = y_pred_d[:, :, :, i]

        valid = np.isfinite(t) & np.isfinite(p)
        diff = p[valid] - t[valid]

        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae  = float(np.mean(np.abs(diff)))
        bias = float(np.mean(diff))

        diff_persist = y_persist_d[valid] - t[valid]
        persist_rmse = float(np.sqrt(np.mean(diff_persist ** 2)))
        ss = 1.0 - rmse / persist_rmse if persist_rmse > 0 else float('nan')

        # ACC: centered Pearson on training-climatology anomalies
        clim_b = clim_2d[np.newaxis, :, :]
        anom_t = (t - clim_b)[valid]
        anom_p = (p - clim_b)[valid]
        acc = float(np.corrcoef(anom_t, anom_p)[0, 1])

        # MAPE — guard against |y| < 0.5 °C (rare here, SST > 12 °C)
        mape_mask = valid & (np.abs(t) > 0.5)
        if np.any(mape_mask):
            mape = float(
                np.mean(np.abs((p[mape_mask] - t[mape_mask]) / t[mape_mask])) * 100.0
            )
        else:
            mape = float('nan')

        results.append(dict(
            horizon=h, rmse=rmse, mae=mae, bias=bias,
            acc=acc, mape=mape, ss=ss,
        ))

    return results


def print_stats_table(metrics: list) -> str:
    header = (
        f"{'Horizon':<10} {'RMSE':>8} {'MAE':>8} {'Bias':>8}"
        f" {'ACC':>8} {'MAPE%':>8} {'SS%':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for m in metrics:
        lines.append(
            f"{str(m['horizon'])+'-day':<10} "
            f"{m['rmse']:>8.4f} "
            f"{m['mae']:>8.4f} "
            f"{m['bias']:>+8.4f} "
            f"{m['acc']:>8.4f} "
            f"{m['mape']:>8.2f} "
            f"{m['ss']*100:>+7.2f}%"
        )
    table = "\n".join(lines)
    print(table)
    return table


def save_metrics_table_image(metrics: list, save_path: str):
    columns = ['Horizon', 'RMSE (°C)', 'MAE (°C)', 'Bias (°C)', 'ACC', 'MAPE (%)', 'SS (%)']
    cell_data = []
    for m in metrics:
        cell_data.append([
            f"{m['horizon']}-day",
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['bias']:+.4f}",
            f"{m['acc']:.4f}",
            f"{m['mape']:.2f}",
            f"{m['ss']*100:+.2f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(metrics) + 1.6))
    ax.axis('off')

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(len(metrics)):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(color)

    plt.title('Test Set Evaluation Metrics', fontsize=13, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Metrics table image saved to: {save_path}")


def save_skill_chart(metrics: list, save_path: str):
    h_arr   = [m['horizon']  for m in metrics]
    rmse    = [m['rmse']     for m in metrics]
    mae_arr = [m['mae']      for m in metrics]
    ss_pct  = [100.0 * m['ss'] for m in metrics]
    x = np.arange(len(h_arr))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, rmse,    'o-', color='#d62728', linewidth=2, markersize=8, label='RMSE')
    ax.plot(x, mae_arr, 's-', color='steelblue', linewidth=2, markersize=8, label='MAE')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in h_arr])
    ax.set_xlabel('Horizonte de previsão')
    ax.set_ylabel('Erro (°C)')
    ax.grid(axis='y', alpha=0.3)
    for xi, r, m in zip(x, rmse, mae_arr):
        ax.annotate(f"{r:.3f}", (xi, r), textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=8, color='#d62728')
        ax.annotate(f"{m:.3f}", (xi, m), textcoords='offset points', xytext=(0, -14),
                    ha='center', fontsize=8, color='steelblue')

    ax2 = ax.twinx()
    bars = ax2.bar(x, ss_pct, alpha=0.25, color='#17becf', width=0.55,
                   label='Skill Score (%)')
    ax2.set_ylabel('Skill Score vs persistência (%)', color='#17becf')
    ax2.tick_params(axis='y', colors='#17becf')
    ax2.set_ylim(0, max(ss_pct) * 1.45 if max(ss_pct) > 0 else 1.0)
    for xi, s in zip(x, ss_pct):
        ax2.annotate(f"{s:+.1f}%", (xi, s), textcoords='offset points', xytext=(0, 4),
                     ha='center', fontsize=8, color='#17becf')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.title('RMSE/MAE e Skill Score por horizonte')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Skill chart saved to: {save_path}")


def save_acc_chart(metrics: list, save_path: str):
    h_arr   = [m['horizon'] for m in metrics]
    acc_arr = [m['acc']     for m in metrics]
    x = np.arange(len(h_arr))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, acc_arr, 'o-', color='#9467bd', linewidth=2, markersize=9)
    for xi, a in zip(x, acc_arr):
        ax.annotate(f"{a:.4f}", (xi, a), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color='#9467bd')

    ax.axhline(1.0, color='grey', linewidth=0.7, linestyle='--', alpha=0.6,
               label='ACC = 1 (predição perfeita)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in h_arr])
    ax.set_xlabel('Horizonte de previsão')
    ax.set_ylabel('ACC')
    ax.set_title('Anomaly Correlation Coefficient (ACC) por horizonte')
    ax.set_ylim(min(acc_arr) - 0.01, 1.01)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ACC chart saved to: {save_path}")


def save_mape_chart(metrics: list, save_path: str):
    h_arr    = [m['horizon'] for m in metrics]
    mape_arr = [m['mape']    for m in metrics]
    x = np.arange(len(h_arr))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, mape_arr, color='#ff7f0e', alpha=0.85,
                  edgecolor='white', linewidth=0.4)
    for xi, v in zip(x, mape_arr):
        ax.text(xi, v + max(mape_arr) * 0.02, f"{v:.2f}%",
                ha='center', fontsize=9, color='#ff7f0e')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in h_arr])
    ax.set_xlabel('Horizonte de previsão')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Erro percentual absoluto médio (MAPE) por horizonte')
    ax.set_ylim(0, max(mape_arr) * 1.20)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"MAPE chart saved to: {save_path}")


def save_bias_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ocean_mask: np.ndarray,
    horizons: tuple,
    save_path: str,
    bias_vlim: float = 0.2,
):
    mean_4d = mean[np.newaxis, :, :, np.newaxis]
    std_4d  = std[np.newaxis, :, :, np.newaxis]

    y_true_d = y_true * std_4d + mean_4d
    y_pred_d = y_pred * std_4d + mean_4d

    diff = y_pred_d - y_true_d  # (N, H, W, Hn)
    bias_map = np.nanmean(diff, axis=0)  # (H, W, Hn)
    bias_map[~ocean_mask, :] = np.nan

    cmap = copy(plt.cm.RdBu_r)
    cmap.set_bad('lightgrey')

    n = len(horizons)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, h in enumerate(horizons):
        ax = axes_flat[i]
        bm = bias_map[:, :, i]
        im = ax.imshow(
            bm, cmap=cmap, vmin=-bias_vlim, vmax=bias_vlim,
            interpolation='bilinear', origin='lower',
        )
        ax.set_title(f"Bias h={h}d  (média {np.nanmean(bm):+.3f} °C)")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle('Bias espacial médio (pred − true, °C) por horizonte',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Bias map saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SST Test-Set Evaluation")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data',  type=str, default=None,
                        help="Path to netCDF data file (default: from config.py)")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Directory for all outputs (overrides individual --output-* paths)")
    parser.add_argument('--max-batches', type=int, default=None,
                        help="Limit number of batches (default: all test data)")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output-stats', type=str, default='evaluation_stats.txt')
    parser.add_argument('--output-table', type=str, default='metrics_table.png')
    parser.add_argument('--output-skill', type=str, default='skill_chart.png')
    parser.add_argument('--output-acc',   type=str, default='acc_chart.png')
    parser.add_argument('--output-mape',  type=str, default='mape_chart.png')
    parser.add_argument('--output-bias',  type=str, default='bias_map.png')
    parser.add_argument('--bias-vlim',    type=float, default=0.2,
                        help="Symmetric range (°C) for the bias map colorbar. Default: 0.2.")
    args = parser.parse_args()

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        args.output_stats = str(out / 'evaluation_stats.txt')
        args.output_table = str(out / 'metrics_table.png')
        args.output_skill = str(out / 'skill_chart.png')
        args.output_acc   = str(out / 'acc_chart.png')
        args.output_mape  = str(out / 'mape_chart.png')
        args.output_bias  = str(out / 'bias_map.png')

    cfg = Config()

    data_path  = args.data if args.data else cfg.data.nc_path
    stats_path = Path(args.model).parent / "norm_stats.npz"

    if stats_path.exists():
        print(f"\nLoading saved normalization stats from {stats_path}")
        s = np.load(stats_path, allow_pickle=False)
        mean = s["mu"].astype(np.float32)
        std  = s["sigma"].astype(np.float32)
        ocean_mask = s["ocean_mask"].astype(bool)

        print(f"Loading test slice from: {data_path}")
        ds = xr.open_dataset(data_path, decode_cf=True)
        sst_test = ds["sst"].sel(time=slice(*cfg.data.test_slice))
        arr = (sst_test.values - mean) / std
        arr[:, ~ocean_mask] = np.nan
        sst_test_norm = arr
    else:
        print(f"\nNo norm_stats.npz next to model — recomputing from NetCDF")
        print(f"Loading data from: {data_path}")
        data = load_and_preprocess_sst(
            nc_path=data_path,
            train_slice=slice(*cfg.data.train_slice),
            val_slice=slice(*cfg.data.val_slice),
            test_slice=slice(*cfg.data.test_slice),
            mask_mode=cfg.data.mask_mode,
            ice_threshold=cfg.data.ice_threshold,
        )
        sst_test_norm = data['sst_test_norm'].values
        ocean_mask    = data['ocean_mask']
        mean          = data['mean']
        std           = data['std']

    print(f"Test data shape: {sst_test_norm.shape}")
    print(f"Test period: {cfg.data.test_slice}")

    model = load_model(args.model)

    print(f"\nRunning predictions over test set...")
    y_true, y_pred, y_persist = collect_predictions(
        model,
        sst_test_norm,
        lookback=cfg.data.lookback,
        horizons=cfg.data.horizons,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(f"Collected {y_true.shape[0]} samples.")

    print("\nComputing metrics...")
    metrics = compute_metrics(
        y_true, y_pred, y_persist, mean, std, ocean_mask, cfg.data.horizons
    )

    print("\n--- Test Set Metrics ---")
    table = print_stats_table(metrics)
    Path(args.output_stats).write_text(table + "\n")
    print(f"Stats saved to: {args.output_stats}")

    save_metrics_table_image(metrics, args.output_table)
    save_skill_chart(metrics, args.output_skill)
    save_acc_chart(metrics, args.output_acc)
    save_mape_chart(metrics, args.output_mape)
    save_bias_map(
        y_true, y_pred, mean, std, ocean_mask, cfg.data.horizons,
        args.output_bias, bias_vlim=args.bias_vlim,
    )


if __name__ == '__main__':
    main()
