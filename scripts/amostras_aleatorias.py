
import sys
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean
from copy import copy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "amostras_aleatorias"
DISPLAY_HZ = (1, 3, 5, 7)

def resolve_nc():
    from src.utils.config import DataConfig
    cfg = DataConfig()
    for cand in (ROOT / cfg.nc_path, ROOT / "ostia_sst_1996-2022_clean.nc"):
        if cand.exists():
            return cand, cfg
    raise SystemExit("[erro] arquivo .nc não encontrado (raiz nem data/raw/)")


def discover_models():
    out = []
    for run_dir in sorted((ROOT / "runs").glob("variante_*")):
        if not run_dir.is_dir():
            continue
        desc = run_dir.name.split("variante_")[1]
        keras = next(run_dir.glob("*.keras"), None)
        if keras:
            out.append((desc, run_dir, keras))
    return out


def plot_sample(truth, pred, model, init_date, save_path, hz=DISPLAY_HZ):
    errs = np.abs(pred - truth)
    svmin = float(np.floor(np.nanmin([truth, pred]) * 2) / 2)
    svmax = float(np.ceil(np.nanmax([truth, pred]) * 2) / 2)
    emax = float(np.ceil(np.nanpercentile(errs, 99) * 10) / 10) or 0.1

    sst_cmap = copy(cmocean.cm.thermal); sst_cmap.set_bad("lightgrey")
    err_cmap = copy(cmocean.cm.amp);     err_cmap.set_bad("lightgrey")
    kw = dict(origin="lower", interpolation="bilinear")

    fig, axes = plt.subplots(len(hz), 3, figsize=(11, 3.1 * len(hz)))
    for r, h in enumerate(hz):
        a0, a1, a2 = axes[r]
        a0.imshow(truth[:, :, r], cmap=sst_cmap, vmin=svmin, vmax=svmax, **kw)
        im1 = a1.imshow(pred[:, :, r], cmap=sst_cmap, vmin=svmin, vmax=svmax, **kw)
        im2 = a2.imshow(errs[:, :, r], cmap=err_cmap, vmin=0, vmax=emax, **kw)
        for a in (a0, a1, a2):
            a.set_xticks([]); a.set_yticks([])
        a0.set_ylabel(f"$h = {h}$", fontsize=12)
        if r == 0:
            a0.set_title("Verdade (°C)"); a1.set_title("Predição (°C)")
            a2.set_title("|Erro| (°C)")
        fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04)
    fig.suptitle(f"{model} — início {init_date}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="3 amostras aleatórias por modelo (estilo cap4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=3, help="amostras por modelo")
    args = ap.parse_args()

    import tensorflow as tf
    from src.utils.losses import MaskedMSE, MaskedMAE, MaskedRMSE
    from src.models.convlstm_model import LastFrameSlice
    custom = {"MaskedMSE": MaskedMSE, "MaskedMAE": MaskedMAE,
              "MaskedRMSE": MaskedRMSE, "LastFrameSlice": LastFrameSlice}

    nc_path, cfg = resolve_nc()
    lookback = cfg.lookback
    horizons = list(cfg.horizons)
    max_h = max(horizons)
    sel = [horizons.index(h) for h in DISPLAY_HZ]

    print(f"Carregando teste de {nc_path.name} ...")
    ds = xr.open_dataset(nc_path, decode_cf=True)
    sst_test = ds["sst"].sel(time=slice(*cfg.test_slice))
    raw = sst_test.values.astype(np.float32)              # (T,H,W) em °C
    times = sst_test["time"].values
    T = raw.shape[0]
    valid_idx = np.arange(lookback, T - max_h + 1, dtype=np.int64)
    print(f"  teste: {raw.shape}, janelas válidas: {valid_idx.size}")

    models = discover_models()
    print(f"Modelos encontrados: {len(models)}")
    rng = np.random.default_rng(args.seed)

    for desc, run_dir, keras in models:
        s = np.load(run_dir / "norm_stats.npz", allow_pickle=False)
        mean = s["mu"].astype(np.float32)
        std = s["sigma"].astype(np.float32)
        ocean = s["ocean_mask"].astype(bool)
        bad = ~ocean

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(keras, custom_objects=custom, compile=False)

        chosen = np.sort(rng.choice(valid_idx, size=args.n, replace=False))
        out_dir = OUT_ROOT / desc
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{desc}:")

        for n, idx in enumerate(chosen, start=1):
            idx = int(idx)
            win = (raw[idx - lookback:idx] - mean) / std
            win[:, bad] = np.nan
            x = np.nan_to_num(win, nan=0.0)[np.newaxis, ..., np.newaxis].astype(np.float32)
            pred_norm = model.predict(x, verbose=0)[0]       # (H,W,nh)
            pred = pred_norm * std[:, :, None] + mean[:, :, None]
            pred[bad] = np.nan
            pred_sel = pred[:, :, sel]

            truth = np.stack([raw[idx + h - 1] for h in DISPLAY_HZ], axis=-1)
            truth[bad] = np.nan

            init = np.datetime_as_string(times[idx - 1], unit="D")
            y, mth, dd = init.split("-")
            init_fmt = f"{dd}/{mth}/{y}"
            fname = f"sample_{n:02d}_{init}.png"
            plot_sample(truth, pred_sel, desc, init_fmt, out_dir / fname)
            print(f"  [{n}/{args.n}] início {init} -> {fname}")

    print(f"\n[ok] figuras em {OUT_ROOT}")


if __name__ == "__main__":
    main()
