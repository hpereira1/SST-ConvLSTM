import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_CSV = ROOT / "docs" / "acc_ra_por_modelo.csv"
OUT_MD = ROOT / "docs" / "acc_ra_por_modelo.md"

CLIM_WINDOW = 61          # janela (dias) da suavização triangular da climatologia doy

MODELS = {
    "1C": "vbase",
    "2C": "vbase2camadas",
    "2C-BN": "v0_clean",
    "2C-PH": "v3_clean_nobn",
    "2C-S": "v4_clean_nobn",
    "2C-S-PH": "v5_clean_nobn",
    "2C-BN-PH": "v3_clean",
    "2C-BN-S": "v4_clean",
    "2C-BN-S-PH": "v5_clean",
    "2C-BN-D": "v0",
    "2C-BN-D-PH": "v3",
    "2C-BN-D-S": "v4",
    "2C-D-S": "v4_nobn",
    "2C-BN-D-S-PH": "v5",
}


def find_keras(desc):
    run = ROOT / "runs" / f"variante_{desc}"
    if not run.is_dir():
        return None
    ks = list(run.glob("*.keras"))
    return ks[0] if ks else None


def build_windows(sst_norm, lookback, horizons):
    max_h = max(horizons)
    valid = np.arange(lookback, sst_norm.shape[0] - max_h + 1, dtype=np.int32)
    X = np.stack([np.nan_to_num(sst_norm[idx - lookback:idx], nan=0.0)
                  for idx in valid])[..., np.newaxis].astype(np.float32)
    Y = np.stack([np.stack([sst_norm[idx + h - 1] for h in horizons], axis=-1)
                  for idx in valid]).astype(np.float32)
    return X, Y, valid


def wstd(x, w):
    ws = w.sum()
    m = (x * w).sum(axis=1, keepdims=True) / ws
    v = (((x - m) ** 2) * w).sum(axis=1) / ws
    return np.sqrt(v)


def main():
    import tensorflow as tf
    import xarray as xr
    import pandas as pd
    from src.preprocessing.sst_preprocessing import load_and_preprocess_sst
    from src.utils.losses import MaskedMSE, MaskedMAE, MaskedRMSE
    from src.models.convlstm_model import LastFrameSlice
    from src.utils.config import DataConfig

    custom = {"MaskedMSE": MaskedMSE, "MaskedMAE": MaskedMAE,
              "MaskedRMSE": MaskedRMSE, "LastFrameSlice": LastFrameSlice}
    cfg = DataConfig()
    horizons = list(cfg.horizons)
    nh = len(horizons)

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
    H, W = ocean_mask.shape

    X, Y, valid = build_windows(sst_norm, cfg.lookback, cfg.horizons)
    y_true_c = Y * std[None, :, :, None] + mean[None, :, :, None]   # °C
    N = X.shape[0]
    print(f"  janelas de teste: {N}, oceano: {int(ocean_mask.sum())} px")


    print("Construindo climatologia dia-do-ano (treino) ...")
    ds = xr.open_dataset(ROOT / cfg.nc_path, decode_cf=True)
    sst_train_raw = ds["sst"].sel(time=slice(*cfg.train_slice))
    arr_tr = sst_train_raw.values.astype(np.float32)               # (Ttr,H,W) — carrega 1x
    doy_tr = sst_train_raw["time"].dt.dayofyear.values
    doys = np.arange(1, 367)
    cv = np.full((366, H, W), np.nan, np.float32)                  # média por dia-do-ano
    for j, dd in enumerate(doys):
        sel = doy_tr == dd
        if sel.any():
            cv[j] = np.nanmean(arr_tr[sel], axis=0)
    ker = np.bartlett(CLIM_WINDOW + 2)[1:-1]; ker = ker / ker.sum()  # triangular
    half = CLIM_WINDOW // 2
    padded = np.concatenate([cv[-half:], cv, cv[:half]], axis=0)
    sm = np.zeros_like(cv)
    for kk in range(CLIM_WINDOW):
        sm += ker[kk] * padded[kk:kk + cv.shape[0]]
    clim_by_doy = np.full((367, H, W), np.nan, np.float32)
    for j, dd in enumerate(doys):
        clim_by_doy[int(dd)] = sm[j]

    # pesos de latitude cos(lat)
    lat = ds["latitude"].values.astype(np.float32)
    w2d = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (H, W))
    w_ocean = w2d[ocean_mask].astype(np.float64)                    # (n_ocean,)

    test_doy = pd.DatetimeIndex(times).dayofyear.values             # (T,)
    # anomalias da observação (independem do modelo) + ACT_obs por horizonte
    oa = {}; act_o = np.zeros(nh)
    for i in range(nh):
        doy_i = test_doy[cfg.lookback + i: cfg.lookback + i + N]
        c_all = clim_by_doy[doy_i]                                  # (N,H,W)
        oa[i] = (y_true_c[:, :, :, i] - c_all)[:, ocean_mask]       # (N,n_ocean)
        act_o[i] = np.nanmean(wstd(oa[i], w_ocean))

    # ---- loop dos modelos ----------------------------------------------------
    rows = []
    for desc, tag in MODELS.items():
        kpath = find_keras(desc)
        if kpath is None:
            print(f"[skip] {desc} ({tag}): .keras não encontrado")
            continue
        print(f"Predizendo {desc} ({tag}) ...")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(kpath, custom_objects=custom, compile=False)
        pred = model.predict(X, batch_size=16, verbose=0)
        pred_c = pred * std[None, :, :, None] + mean[None, :, :, None]

        for i, h in enumerate(horizons):
            doy_i = test_doy[cfg.lookback + i: cfg.lookback + i + N]
            c_all = clim_by_doy[doy_i]
            fa = (pred_c[:, :, :, i] - c_all)[:, ocean_mask]        # (N,n_ocean)
            # ACC (WB2): por amostra, não-centrado, pesado; média sobre amostras
            num = np.nansum(w_ocean * fa * oa[i], axis=1)
            den = np.sqrt(np.nansum(w_ocean * fa ** 2, axis=1) *
                          np.nansum(w_ocean * oa[i] ** 2, axis=1))
            acc = float(np.nanmean(num / den))
            # RA (Bouallègue): ACT_pred / ACT_obs
            ra = float(np.nanmean(wstd(fa, w_ocean)) / act_o[i])
            rows.append({"modelo": desc, "horizonte": h,
                         "ACC": round(acc, 4), "RA": round(ra, 4)})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[ok] {OUT_CSV}")

    order = [m for m in MODELS if m in df["modelo"].unique()]
    acc_w = df.pivot(index="modelo", columns="horizonte", values="ACC").reindex(order)
    ra_w = df.pivot(index="modelo", columns="horizonte", values="RA").reindex(order)

    def md_table(wide, title):
        hs = list(wide.columns)
        lines = [f"### {title}", "",
                 "| Modelo | " + " | ".join(f"h={h}" for h in hs) + " |",
                 "|" + "---|" * (len(hs) + 1)]
        for m in wide.index:
            lines.append(f"| {m} | " +
                         " | ".join(f"{wide.loc[m, h]:.4f}" for h in hs) + " |")
        return "\n".join(lines)

    md = ("# ACC (WeatherBench2) e RA (Bouallègue) por modelo e horizonte\n\n"
          "ACC: anomalia vs climatologia dia-do-ano (treino, janela 61d), por "
          "amostra, não-centrado, peso cos(lat) — Rasp et al. (2024).\n"
          "RA: razão de atividade (desvio-padrão espacial da anomalia), "
          "pred/obs — Ben Bouallègue et al. (2023). RA<1 = subativo.\n\n"
          + md_table(acc_w, "ACC") + "\n\n" + md_table(ra_w, "RA") + "\n")
    OUT_MD.write_text(md)
    print(f"[ok] {OUT_MD}")
    print("\n" + md_table(acc_w, "ACC"))
    print("\n" + md_table(ra_w, "RA"))


if __name__ == "__main__":
    main()