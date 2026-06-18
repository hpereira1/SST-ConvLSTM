
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean
from copy import copy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "overleaftcc" / "images" / "cap4"
CACHE = Path(__file__).with_suffix(".cache.npz")


MODELS = [
    ("1C",
     "runs/variante_1C/convlstm_vbase_masked_seed42.keras"),
    ("2C-BN-D-S-PH",
     "runs/variante_2C-BN-D-S-PH/convlstm_v5_masked_seed42.keras"),
    ("2C-PH",
     "runs/variante_2C-PH/convlstm_v3_clean_nobn_masked_seed42.keras"),
]
BEST_MODEL = "2C-BN-D-S-PH"   # amostra escolhida pela dinâmica deste modelo (h1->h7)
FORCE_INIT = "2021-02-20"     # data de início fixa (último frame observado); None = dinâmica

def collect_predictions(model, sst_norm, lookback, horizons, batch_size=16):
    """Janelas deslizantes; retorna (y_true, y_pred, y_persist) normalizados.
    Mesma lógica de evaluate.collect_predictions."""
    max_h = max(horizons)
    valid = np.arange(lookback, sst_norm.shape[0] - max_h + 1, dtype=np.int32)
    yt, yp, ype = [], [], []
    n_batches = (len(valid) + batch_size - 1) // batch_size
    for b in range(n_batches):
        bidx = valid[b * batch_size:(b + 1) * batch_size]
        X, Y, P = [], [], []
        for idx in bidx:
            x = np.nan_to_num(sst_norm[idx - lookback:idx], nan=0.0)[..., np.newaxis]
            y = np.stack([sst_norm[idx + h - 1] for h in horizons], axis=-1)
            X.append(x); Y.append(y); P.append(sst_norm[idx - 1])
        preds = model.predict(np.array(X, dtype=np.float32), verbose=0)
        yt.append(np.array(Y, dtype=np.float32)); yp.append(preds)
        ype.append(np.array(P, dtype=np.float32))
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(ype)


def compute_cache():
    """Roda a inferência uma vez e salva, no cache, só o que os plots precisam:
    mapas de RMSE por pixel (todos os horizontes) e os campos da amostra favorável."""
    import tensorflow as tf
    from src.preprocessing.sst_preprocessing import load_and_preprocess_sst
    from src.utils.losses import MaskedMSE, MaskedMAE, MaskedRMSE
    from src.models.convlstm_model import LastFrameSlice
    from src.utils.config import DataConfig

    custom = {"MaskedMSE": MaskedMSE, "MaskedMAE": MaskedMAE,
              "MaskedRMSE": MaskedRMSE, "LastFrameSlice": LastFrameSlice}
    cfg = DataConfig()
    horizons = list(cfg.horizons)

    print("Carregando dados de teste ...")
    data = load_and_preprocess_sst(
        nc_path=str(ROOT / cfg.nc_path),
        train_slice=slice(*cfg.train_slice), val_slice=slice(*cfg.val_slice),
        test_slice=slice(*cfg.test_slice), mask_mode=cfg.mask_mode,
        ice_threshold=cfg.ice_threshold,
    )
    sst_test = data["sst_test_norm"]
    sst_norm = sst_test.values
    times = sst_test["time"].values
    ocean_mask = data["ocean_mask"]
    mean, std = data["mean"], data["std"]
    bad = ~ocean_mask
    print(f"  teste: {sst_norm.shape}, oceano: {int(ocean_mask.sum())} px")

    def denorm(a):
        if a.ndim == 4:
            return a * std[None, :, :, None] + mean[None, :, :, None]
        return a * std[None] + mean[None]

    y_true_c = None; y_persist_c = None; preds = {}
    for name, rel in MODELS:
        print(f"Predizendo {name} ...")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(ROOT / rel, custom_objects=custom,
                                            compile=False)
        yt, yp, ype = collect_predictions(model, sst_norm, cfg.lookback, cfg.horizons)
        preds[name] = denorm(yp)
        if y_true_c is None:
            y_true_c = denorm(yt); y_persist_c = denorm(ype)

    y_true_c[:, bad, :] = np.nan
    y_persist_c[:, bad] = np.nan
    for name in preds:
        preds[name][:, bad, :] = np.nan

    names = [n for n, _ in MODELS]
    nh = len(horizons)
    H, W = ocean_mask.shape

    # RMSE agregado por horizonte (só para conferência impressa)
    rmse_curve = np.array([[
        np.sqrt(np.nanmean(((preds[n][:, :, :, i] - y_true_c[:, :, :, i])[:, ocean_mask]) ** 2))
        for i in range(nh)] for n in names])
    persist_curve = np.array([
        np.sqrt(np.nanmean(((y_persist_c - y_true_c[:, :, :, i])[:, ocean_mask]) ** 2))
        for i in range(nh)])

    # RMSE por pixel, todos os horizontes (plot escolhe o subconjunto)
    rmse_pix = np.full((len(names), nh, H, W), np.nan, np.float32)
    persist_pix = np.full((nh, H, W), np.nan, np.float32)
    for i in range(nh):
        for k, n in enumerate(names):
            d = preds[n][:, :, :, i] - y_true_c[:, :, :, i]
            m = np.sqrt(np.nanmean(d ** 2, axis=0)); m[bad] = np.nan
            rmse_pix[k, i] = m
        dp = y_persist_c - y_true_c[:, :, :, i]
        mp = np.sqrt(np.nanmean(dp ** 2, axis=0)); mp[bad] = np.nan
        persist_pix[i] = mp

    # Atividade (Bouallègue et al. 2023): desvio-padrão ESPACIAL da anomalia
    # (campo - climatologia) por amostra, média sobre as amostras; por horizonte.
    # (std espacial por campo isola a estrutura espacial; o std agrupado seria
    # dominado pela variação sazonal entre amostras.) RA = ACT_pred / ACT_obs.
    clim = mean[None]                                          # (1,H,W)
    act_o = np.zeros(nh); act_f = np.zeros((len(names), nh))
    for i in range(nh):
        ao = (y_true_c[:, :, :, i] - clim)[:, ocean_mask]     # (N, n_ocean)
        act_o[i] = np.nanmean(np.nanstd(ao, axis=1))
        for k, n in enumerate(names):
            af = (preds[n][:, :, :, i] - clim)[:, ocean_mask]
            act_f[k, i] = np.nanmean(np.nanstd(af, axis=1))
    ra = act_f / act_o[None, :]

    # seleção da amostra: MAIOR variação da predição entre h=1 e h=7 (2C-PH),
    # i.e. uma amostra dinâmica (a SST evoluiu no período) — evita figura estática.
    i1, i7 = horizons.index(1), horizons.index(7)

    def _rms(a):                       # RMS sobre o oceano, por amostra
        return np.sqrt(np.nanmean((a[:, ocean_mask]) ** 2, axis=1))

    pred_var = _rms(preds[BEST_MODEL][:, :, :, i7] - preds[BEST_MODEL][:, :, :, i1])
    truth_var = _rms(y_true_c[:, :, :, i7] - y_true_c[:, :, :, i1])
    h7_rmse = _rms(preds[BEST_MODEL][:, :, :, i7] - y_true_c[:, :, :, i7])
    order = np.argsort(pred_var)[::-1]
    print(f"\nTop amostras por variação da predição h1->h7 ({BEST_MODEL}):")
    print(f"{'pos':>5} {'alvo_h7':>12} {'var_pred':>9} {'var_truth':>10} {'rmse_h7':>8}")
    for p in order[:12]:
        dt = np.datetime_as_string(times[cfg.lookback + p + 7 - 1], unit="D")
        print(f"{p:>5} {dt:>12} {pred_var[p]:>9.3f} {truth_var[p]:>10.3f} {h7_rmse[p]:>8.3f}")
    best = int(order[0])
    if FORCE_INIT is not None:
        tgt = np.datetime64(FORCE_INIT)
        pos = np.where(times.astype("datetime64[D]") == tgt)[0]
        if len(pos) == 0:
            raise SystemExit(f"[erro] {FORCE_INIT} não está no conjunto de teste")
        best = int(pos[0]) - cfg.lookback + 1
        if not (0 <= best < preds[BEST_MODEL].shape[0]):
            raise SystemExit(f"[erro] {FORCE_INIT} sem janela completa (lookback/h=7)")
    idx = cfg.lookback + best
    sample_date = np.datetime_as_string(times[idx + 7 - 1], unit="D")
    sample_truth = y_true_c[best]                              # (H,W,nh)
    sample_preds = np.stack([preds[n][best] for n in names])   # (M,H,W,nh)

    np.savez_compressed(
        CACHE,
        names=np.array(names), horizons=np.array(horizons),
        rmse_pix=rmse_pix, persist_pix=persist_pix,
        sample_truth=sample_truth, sample_preds=sample_preds,
        sample_date=np.array(sample_date),
        sample_rmse=np.array(h7_rmse[best]),
        act_f=act_f, act_o=act_o, ra=ra,
    )
    print(f"\n[cache] {CACHE}")
    print("\nRMSE (°C) por horizonte — conferir h=1,h=7 com tab:resultados_h1h7:")
    print(f"{'modelo':<16}" + "".join(f"  h{h}" for h in horizons))
    for k, n in enumerate(names):
        print(f"{n:<16}" + "".join(f" {v:5.3f}" for v in rmse_curve[k]))
    print(f"{'Persistência':<16}" + "".join(f" {v:5.3f}" for v in persist_curve))
    print("\nAtividade relativa RA = ACT_f/ACT_o por horizonte (Bouallègue 2023):")
    print(f"{'modelo':<16}" + "".join(f"  h{h}" for h in horizons))
    for k, n in enumerate(names):
        print(f"{n:<16}" + "".join(f" {v:5.3f}" for v in ra[k]))
    print(f"{'ACT_obs (°C)':<16}" + "".join(f" {v:5.3f}" for v in act_o))
    print(f"\nAmostra dinâmica ({BEST_MODEL}): pos={best}, "
          f"var_pred={pred_var[best]:.3f}, RMSE h=7={h7_rmse[best]:.3f} °C, "
          f"alvo h=7={sample_date}")


def plot_rmse_pixel(d, model="2C-BN-D-S-PH", hz=(1, 3, 5, 7)):
    """RMSE por pixel de um único modelo (geografia do erro), h em colunas."""
    names = [str(n) for n in d["names"]]; horizons = list(d["horizons"])
    k = names.index(model)
    sel = [horizons.index(h) for h in hz]
    maps = d["rmse_pix"][k, sel]                            # (len(hz), H, W)
    vmax = float(np.nanpercentile(maps, 99))
    cmap = copy(cmocean.cm.amp); cmap.set_bad("lightgrey")

    fig, axes = plt.subplots(1, len(hz), figsize=(4.0 * len(hz), 4.2))
    for c, h in enumerate(hz):
        im = axes[c].imshow(maps[c], cmap=cmap, vmin=0, vmax=vmax,
                            origin="lower", interpolation="bilinear")
        axes[c].set_xticks([]); axes[c].set_yticks([])
        axes[c].set_title(f"$h = {h}$", fontsize=12)
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.02)
    cbar.set_label("RMSE por pixel (°C)")
    fig.savefig(OUT_DIR / "rmse_por_pixel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT_DIR / 'rmse_por_pixel.png'}  ({model})")


def plot_sample(d, model="2C-BN-D-S-PH", hz=(1, 3, 5, 7)):
    names = [str(n) for n in d["names"]]; horizons = list(d["horizons"])
    k = names.index(model)
    sel = [horizons.index(h) for h in hz]
    truth = d["sample_truth"][:, :, sel]                   # (H,W,len(hz))
    preds = d["sample_preds"][k][:, :, sel]                # (H,W,len(hz))
    errs = np.abs(preds - truth)
    # escalas automáticas a partir da própria amostra (adapta a qualquer amostra)
    svmin = float(np.floor(np.nanmin([truth, preds]) * 2) / 2)   # SST, nearest 0,5 °C
    svmax = float(np.ceil(np.nanmax([truth, preds]) * 2) / 2)
    emax = float(np.ceil(np.nanpercentile(errs, 99) * 10) / 10)  # erro, nearest 0,1 °C

    # data de início da previsão (último frame observado) = alvo de h=7 menos 7 dias
    init = np.datetime64(str(d["sample_date"])) - np.timedelta64(7, "D")
    y, m, dd = str(init).split("-")
    init_fmt = f"{dd}/{m}/{y}"

    sst_cmap = copy(cmocean.cm.thermal); sst_cmap.set_bad("lightgrey")
    err_cmap = copy(cmocean.cm.amp);     err_cmap.set_bad("lightgrey")
    kw = dict(origin="lower", interpolation="bilinear")

    fig, axes = plt.subplots(len(hz), 3, figsize=(11, 3.1 * len(hz)))
    for r, h in enumerate(hz):
        a0, a1, a2 = axes[r]
        a0.imshow(truth[:, :, r], cmap=sst_cmap, vmin=svmin, vmax=svmax, **kw)
        im1 = a1.imshow(preds[:, :, r], cmap=sst_cmap, vmin=svmin, vmax=svmax, **kw)
        im2 = a2.imshow(errs[:, :, r], cmap=err_cmap, vmin=0, vmax=emax, **kw)
        for a in (a0, a1, a2):
            a.set_xticks([]); a.set_yticks([])
        a0.set_ylabel(f"$h = {h}$", fontsize=12)
        if r == 0:
            a0.set_title("Verdade (°C)"); a1.set_title("Predição (°C)")
            a2.set_title("|Erro| (°C)")
        fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04)
    fig.savefig(OUT_DIR / "amostra_completo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT_DIR / 'amostra_completo.png'}  ({model}, início {init_fmt}, "
          f"RMSE h=7 {float(d['sample_rmse']):.3f} °C)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--recompute" in sys.argv or not CACHE.exists():
        compute_cache()
    d = dict(np.load(CACHE, allow_pickle=False))
    plot_rmse_pixel(d, model="2C-BN-D-S-PH", hz=(1, 3, 5, 7))
    plot_sample(d, model="2C-BN-D-S-PH", hz=(1, 3, 5, 7))


if __name__ == "__main__":
    main()