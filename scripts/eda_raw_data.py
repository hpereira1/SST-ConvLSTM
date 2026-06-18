
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime


RAW_DIR = Path("data/raw")
FILE_1 = RAW_DIR / "METOFFICE-GLO-SST-L4-REP-OBS-SST_1755837049160(1).nc"
FILE_2 = RAW_DIR / "METOFFICE-GLO-SST-L4-REP-OBS-SST_1755836916104(1).nc"

OUTPUT_FILE = Path("docs/eda_raw_data_report.md")


def sep(title: str) -> str:
    return f"\n{'=' * 72}\n  {title}\n{'=' * 72}\n"


def subsep(title: str) -> str:
    return f"\n--- {title} ---\n"


def concatenate_raw_files() -> xr.Dataset:
    """Abre os dois arquivos brutos, concatena no eixo temporal e
    remove eventuais instantes duplicados. Retorna um único Dataset
    ordenado cronologicamente."""
    lines = [sep("0. CONCATENAÇÃO DOS ARQUIVOS BRUTOS")]

    ds1 = xr.open_dataset(FILE_1)
    ds2 = xr.open_dataset(FILE_2)

    lines.append(f"Arquivo 1: {FILE_1.name}")
    lines.append(f"  Período: {ds1.time.values[0]} a {ds1.time.values[-1]}")
    lines.append(f"  Instantes: {ds1.sizes['time']}")

    lines.append(f"\nArquivo 2: {FILE_2.name}")
    lines.append(f"  Período: {ds2.time.values[0]} a {ds2.time.values[-1]}")
    lines.append(f"  Instantes: {ds2.sizes['time']}")

    # Verificar grades espaciais idênticas
    lat_equal = np.array_equal(ds1.latitude.values, ds2.latitude.values)
    lon_equal = np.array_equal(ds1.longitude.values, ds2.longitude.values)
    lines.append(f"\nGrades espaciais idênticas: lat={lat_equal}, lon={lon_equal}")

    # Concatenar
    ds = xr.concat([ds1, ds2], dim="time")
    lines.append(f"\nApós concatenação: {ds.sizes['time']} instantes")

    # Duplicatas
    times = ds.time.values
    _, unique_idx = np.unique(times, return_index=True)
    n_duplicates = len(times) - len(unique_idx)
    lines.append(f"Instantes duplicados encontrados: {n_duplicates}")

    if n_duplicates > 0:
        ds = ds.isel(time=np.sort(unique_idx))
        lines.append(f"Após remoção de duplicatas: {ds.sizes['time']} instantes")

    # Ordenar cronologicamente
    ds = ds.sortby("time")
    lines.append(f"Série final ordenada: {ds.time.values[0]} a {ds.time.values[-1]}")

    ds1.close()
    ds2.close()

    return ds, "\n".join(lines)


def describe_dataset(ds: xr.Dataset) -> str:
    lines = [sep("1. ESTRUTURA DO DATASET CONCATENADO")]

    lines.append(f"Dimensões: {dict(ds.sizes)}")
    lines.append(f"Variáveis: {list(ds.data_vars)}")
    lines.append(f"Coordenadas: {list(ds.coords)}")
    lines.append("")

    for var in ds.data_vars:
        v = ds[var]
        mem_mb = v.nbytes / 1e6
        lines.append(f"  {var}: dtype={v.dtype}, shape={v.shape}, {mem_mb:.1f} MB")

    total_mb = sum(ds[v].nbytes for v in ds.data_vars) / 1e6
    lines.append(f"\nMemória total (arrays): {total_mb:.1f} MB")

    lines.append(f"\nAtributos globais relevantes:")
    for key in ["title", "institution", "source", "Conventions"]:
        if key in ds.attrs:
            lines.append(f"  {key}: {ds.attrs[key]}")

    return "\n".join(lines)


def analyze_coordinates(ds: xr.Dataset) -> str:
    lines = [sep("2. COORDENADAS")]

    # --- Tempo ---
    time = ds.time.values
    lines.append(f"TEMPO:")
    lines.append(f"  Primeiro instante: {time[0]}")
    lines.append(f"  Último instante:   {time[-1]}")
    lines.append(f"  Total de instantes (T): {len(time)}")

    diffs = np.diff(time).astype("timedelta64[D]").astype(int)
    lines.append(f"  Diferença mín entre instantes: {diffs.min()} dia(s)")
    lines.append(f"  Diferença máx entre instantes: {diffs.max()} dia(s)")
    lines.append(f"  Diferença média:               {diffs.mean():.6f} dia(s)")

    gaps = np.where(diffs > 1)[0]
    if len(gaps) > 0:
        lines.append(f"  LACUNAS DETECTADAS: {len(gaps)}")
        for idx in gaps[:10]:
            lines.append(f"    Entre {time[idx]} e {time[idx+1]} — {diffs[idx]} dias")
    else:
        lines.append(f"  Lacunas temporais: NENHUMA (série diária contínua)")

    # --- Latitude ---
    lat = ds.latitude.values
    lat_diffs = np.diff(lat)
    lines.append(f"\nLATITUDE:")
    lines.append(f"  Min: {lat.min():.2f}°   Max: {lat.max():.2f}°")
    lines.append(f"  Pontos (H): {len(lat)}")
    lines.append(f"  Espaçamento: {lat_diffs.mean():.4f}° (min={lat_diffs.min():.4f}°, max={lat_diffs.max():.4f}°)")

    # --- Longitude ---
    lon = ds.longitude.values
    lon_diffs = np.diff(lon)
    lines.append(f"\nLONGITUDE:")
    lines.append(f"  Min: {lon.min():.2f}°   Max: {lon.max():.2f}°")
    lines.append(f"  Pontos (W): {len(lon)}")
    lines.append(f"  Espaçamento: {lon_diffs.mean():.4f}° (min={lon_diffs.min():.4f}°, max={lon_diffs.max():.4f}°)")

    lines.append(f"\nGRADE TOTAL: {len(lat)} x {len(lon)} = {len(lat) * len(lon)} pixels")
    lines.append(f"Resolução nominal: 0.05° ≈ {0.05 * 111:.1f} km")

    return "\n".join(lines)


def analyze_sst(sst: np.ndarray) -> str:
    lines = [sep("3. ANALYSED_SST — ESTATÍSTICAS GLOBAIS")]

    T, H, W = sst.shape
    total = sst.size

    lines.append(f"Shape: (T={T}, H={H}, W={W})")
    lines.append(f"dtype: {sst.dtype}")
    lines.append(f"Total de elementos: {total:,}")

    # --- NaN ---
    nan_mask = np.isnan(sst)
    nan_count = nan_mask.sum()
    nan_pct = 100.0 * nan_count / total
    valid_count = total - nan_count

    lines.append(f"\nCONTAGEM:")
    lines.append(f"  Elementos válidos (finitos): {valid_count:,} ({100.0 - nan_pct:.4f}%)")
    lines.append(f"  Elementos NaN:               {nan_count:,} ({nan_pct:.4f}%)")

    # --- Estatísticas em Kelvin ---
    valid = sst[~nan_mask]

    lines.append(f"\nESTATÍSTICAS (Kelvin):")
    lines.append(f"  Mínimo:          {valid.min():.4f} K")
    lines.append(f"  Máximo:          {valid.max():.4f} K")
    lines.append(f"  Amplitude:       {valid.max() - valid.min():.4f} K")
    lines.append(f"  Média:           {valid.mean():.4f} K")
    lines.append(f"  Mediana:         {np.median(valid):.4f} K")
    lines.append(f"  Desvio padrão:   {valid.std():.4f} K")
    lines.append(f"  Variância:       {valid.var():.4f} K²")

    # --- Referência em Celsius ---
    lines.append(f"\nREFERÊNCIA EM CELSIUS (K − 273.15, sem alterar o array):")
    lines.append(f"  Mínimo:          {valid.min() - 273.15:.4f} °C")
    lines.append(f"  Máximo:          {valid.max() - 273.15:.4f} °C")
    lines.append(f"  Amplitude:       {valid.max() - valid.min():.4f} °C")
    lines.append(f"  Média:           {valid.mean() - 273.15:.4f} °C")
    lines.append(f"  Mediana:         {np.median(valid) - 273.15:.4f} °C")
    lines.append(f"  Desvio padrão:   {valid.std():.4f} °C")

    return "\n".join(lines)


def analyze_sst_spatial(sst: np.ndarray) -> str:
    lines = [sep("4. ANALYSED_SST — VARIABILIDADE ESPACIAL (POR PIXEL)")]

    T, H, W = sst.shape

    with np.errstate(all="ignore"):
        mean_per_pixel = np.nanmean(sst, axis=0)  # (H, W)
        std_per_pixel = np.nanstd(sst, axis=0, ddof=0)   # (H, W), population std
        min_per_pixel = np.nanmin(sst, axis=0)     # (H, W)
        max_per_pixel = np.nanmax(sst, axis=0)     # (H, W)

    ocean = np.isfinite(mean_per_pixel)
    n_ocean = ocean.sum()
    n_land = (~ocean).sum()

    lines.append(f"Cálculo: média, desvio, mín e máx temporais por pixel (eixo T={T})")
    lines.append(f"  np.nanmean(sst, axis=0)  → média ignorando NaN")
    lines.append(f"  np.nanstd(sst, axis=0, ddof=0)  → desvio padrão populacional")
    lines.append(f"  np.nanmin / np.nanmax  → extremos por pixel")

    lines.append(f"\nCLASSIFICAÇÃO DE PIXELS:")
    lines.append(f"  Oceano (pelo menos 1 valor finito): {n_ocean} pixels ({100.0 * n_ocean / (H * W):.2f}%)")
    lines.append(f"  Terra/gelo (todos NaN):             {n_land} pixels ({100.0 * n_land / (H * W):.2f}%)")

    om = ocean  # alias
    lines.append(f"\nMÉDIA TEMPORAL POR PIXEL (apenas oceano, Kelvin / °C):")
    lines.append(f"  Min das médias:   {mean_per_pixel[om].min():.4f} K  ({mean_per_pixel[om].min() - 273.15:.4f} °C)")
    lines.append(f"  Max das médias:   {mean_per_pixel[om].max():.4f} K  ({mean_per_pixel[om].max() - 273.15:.4f} °C)")
    lines.append(f"  Média das médias: {mean_per_pixel[om].mean():.4f} K  ({mean_per_pixel[om].mean() - 273.15:.4f} °C)")

    lines.append(f"\nDESVIO PADRÃO TEMPORAL POR PIXEL (apenas oceano):")
    lines.append(f"  Min dos desvios:   {std_per_pixel[om].min():.4f} K")
    lines.append(f"  Max dos desvios:   {std_per_pixel[om].max():.4f} K")
    lines.append(f"  Média dos desvios: {std_per_pixel[om].mean():.4f} K")
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        count = (std_per_pixel[om] > threshold).sum()
        lines.append(f"  Pixels com σ > {threshold}: {count} ({100.0 * count / n_ocean:.1f}%)")

    lines.append(f"\nAMPLITUDE TÉRMICA POR PIXEL (max − min temporal, apenas oceano):")
    amp = max_per_pixel[om] - min_per_pixel[om]
    lines.append(f"  Min amplitude:   {amp.min():.4f} K")
    lines.append(f"  Max amplitude:   {amp.max():.4f} K")
    lines.append(f"  Média amplitude: {amp.mean():.4f} K")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Estrutura de NaN
# ---------------------------------------------------------------------------
def analyze_nan_structure(sst: np.ndarray) -> str:
    lines = [sep("5. ESTRUTURA DE NaN")]

    T, H, W = sst.shape
    nan_map = np.isnan(sst)

    # --- Por pixel (eixo temporal) ---
    nan_per_pixel = nan_map.sum(axis=0)  # (H, W)

    always_nan = (nan_per_pixel == T).sum()
    never_nan = (nan_per_pixel == 0).sum()
    partial = H * W - always_nan - never_nan

    lines.append(f"POR PIXEL (ao longo de T={T} instantes):")
    lines.append(f"  Sempre NaN (terra):              {always_nan} pixels")
    lines.append(f"  Nunca NaN (oceano sem falhas):    {never_nan} pixels")
    lines.append(f"  Parcialmente NaN (falhas pontuais): {partial} pixels")

    if partial > 0:
        partial_mask = (nan_per_pixel > 0) & (nan_per_pixel < T)
        partial_counts = nan_per_pixel[partial_mask]
        lines.append(f"    Nº de NaN nesses pixels — min: {partial_counts.min()}, max: {partial_counts.max()}, média: {partial_counts.mean():.2f}")

    # --- Por instante temporal ---
    nan_per_time = nan_map.sum(axis=(1, 2))  # (T,)

    lines.append(f"\nPOR INSTANTE TEMPORAL:")
    lines.append(f"  NaN por frame — min: {nan_per_time.min()}, max: {nan_per_time.max()}")
    lines.append(f"  NaN por frame — média: {nan_per_time.mean():.2f}, desvio: {nan_per_time.std():.4f}")

    baseline = nan_per_time.min()
    above = (nan_per_time > baseline).sum()
    lines.append(f"\n  Linha de base (NaN fixos = terra): {baseline} pixels/frame")
    lines.append(f"  Instantes com NaN acima da linha de base: {above} de {T} ({100.0 * above / T:.2f}%)")

    if above > 0:
        excess = nan_per_time[nan_per_time > baseline] - baseline
        lines.append(f"    Excesso — min: {excess.min()}, max: {excess.max()}, média: {excess.mean():.2f}")

    # --- Fração de NaN ---
    lines.append(f"\nFRAÇÃO DE NaN TOTAL:")
    total_nan = nan_map.sum()
    total_elem = sst.size
    lines.append(f"  {total_nan:,} de {total_elem:,} elementos ({100.0 * total_nan / total_elem:.4f}%)")
    lines.append(f"  NaN atribuíveis a terra (baseline × T): {baseline * T:,} ({100.0 * baseline * T / total_elem:.4f}%)")
    ocean_nan = total_nan - baseline * T
    lines.append(f"  NaN em pixels oceânicos:                {ocean_nan:,} ({100.0 * ocean_nan / total_elem:.6f}%)")

    return "\n".join(lines)


def analyze_mask(ds: xr.Dataset) -> str:
    lines = [sep("6. MASK (BITMASK)")]

    mask = ds["mask"].values  # (T, H, W)
    T, H, W = mask.shape

    lines.append(f"Shape: (T={T}, H={H}, W={W})")
    lines.append(f"dtype: {mask.dtype}")

    # Valores únicos em toda a série
    valid = mask[np.isfinite(mask)]
    unique_vals = np.unique(valid)
    lines.append(f"\nValores únicos em toda a série: {unique_vals}")

    # Distribuição global
    lines.append(f"\nDISTRIBUIÇÃO GLOBAL:")
    for val in unique_vals:
        count = (mask == val).sum()
        lines.append(f"  valor={val:.0f}: {count:,} ocorrências ({100.0 * count / mask.size:.2f}%)")

    # Distribuição em um frame
    mask_t0 = mask[0]
    lines.append(f"\nDISTRIBUIÇÃO NO PRIMEIRO INSTANTE (t=0):")
    for val in unique_vals:
        count = (mask_t0 == val).sum()
        lines.append(f"  valor={val:.0f}: {count} pixels ({100.0 * count / (H * W):.2f}%)")

    # Constância temporal
    lines.append(f"\nVARIAÇÃO TEMPORAL:")
    n_checked = min(T, 50)
    step = max(1, T // n_checked)
    n_varying = 0
    for t in range(0, T, step):
        diff = (mask[t] != mask[0]).sum()
        if diff > 0:
            n_varying += 1
            lines.append(f"  t={t}: {diff} pixels diferem de t=0")

    if n_varying == 0:
        lines.append(f"  Mask é CONSTANTE no tempo ({n_checked} instantes amostrados, passo={step})")
    else:
        lines.append(f"  {n_varying} instantes com diferenças detectadas")

    # Decomposição por bit
    lines.append(f"\nDECOMPOSIÇÃO POR BIT (OSTIA bitmask, primeiro instante):")
    bit_names = ["bit 0 — água aberta", "bit 1 — terra", "bit 2 — lago", "bit 3 — gelo marinho"]
    for bit_idx, name in enumerate(bit_names):
        bit_val = 2 ** bit_idx
        count = ((mask_t0.astype(int) & bit_val) > 0).sum()
        lines.append(f"  {name}: {count} pixels ({100.0 * count / (H * W):.2f}%)")

    return "\n".join(lines)



def analyze_sea_ice(ds: xr.Dataset) -> str:
    lines = [sep("7. SEA_ICE_FRACTION")]

    sif = ds["sea_ice_fraction"].values
    T, H, W = sif.shape

    lines.append(f"Shape: (T={T}, H={H}, W={W})")
    lines.append(f"dtype: {sif.dtype}")

    nan_count = np.isnan(sif).sum()
    nan_pct = 100.0 * nan_count / sif.size
    lines.append(f"\nNaN: {nan_count:,} ({nan_pct:.4f}%)")

    valid = sif[np.isfinite(sif)]
    lines.append(f"Válidos: {valid.size:,}")

    if valid.size > 0:
        unique = np.unique(valid)
        lines.append(f"  Valores únicos: {unique[:20]}")
        lines.append(f"  Min: {valid.min():.6f}")
        lines.append(f"  Max: {valid.max():.6f}")
        lines.append(f"  Média: {valid.mean():.6f}")

        nonzero = (valid > 0).sum()
        lines.append(f"  Valores > 0: {nonzero} ({100.0 * nonzero / valid.size:.6f}%)")

        if valid.max() == 0.0:
            lines.append(f"\n  CONCLUSÃO: sea_ice_fraction é ZERO em todo o domínio e período.")
            lines.append(f"  Região subtropical sem gelo marinho.")

    return "\n".join(lines)



def analyze_analysis_error(ds: xr.Dataset) -> str:
    lines = [sep("8. ANALYSIS_ERROR")]

    err = ds["analysis_error"].values
    T, H, W = err.shape

    lines.append(f"Shape: (T={T}, H={H}, W={W})")
    lines.append(f"dtype: {err.dtype}")

    nan_count = np.isnan(err).sum()
    nan_pct = 100.0 * nan_count / err.size
    lines.append(f"\nNaN: {nan_count:,} ({nan_pct:.4f}%)")

    valid = err[np.isfinite(err)]
    lines.append(f"Válidos: {valid.size:,}")

    if valid.size > 0:
        lines.append(f"\nESTATÍSTICAS:")
        lines.append(f"  Min:    {valid.min():.6f}")
        lines.append(f"  Max:    {valid.max():.6f}")
        lines.append(f"  Média:  {valid.mean():.6f}")
        lines.append(f"  Mediana: {np.median(valid):.6f}")
        lines.append(f"  Desvio: {valid.std():.6f}")

    return "\n".join(lines)


def summary(ds: xr.Dataset, sst: np.ndarray) -> str:
    lines = [sep("9. RESUMO PARA O TCC")]

    T, H, W = sst.shape
    time = ds.time.values
    lat = ds.latitude.values
    lon = ds.longitude.values

    nan_mask = np.isnan(sst)
    valid = sst[~nan_mask]
    nan_per_time = nan_mask.sum(axis=(1, 2))

    with np.errstate(all="ignore"):
        std_per_pixel = np.nanstd(sst, axis=0, ddof=0)
    ocean = np.isfinite(std_per_pixel)

    lines.append(f"Período:             {time[0]} a {time[-1]}")
    lines.append(f"Instantes (T):       {T}")
    lines.append(f"Lacunas temporais:   {'nenhuma' if np.diff(time).astype('timedelta64[D]').astype(int).max() == 1 else 'SIM'}")
    lines.append(f"Grade:               {H} × {W} = {H * W} pixels")
    lines.append(f"Resolução:           0.05° ≈ {0.05 * 111:.1f} km")
    lines.append(f"Latitude:            {lat.min():.2f}° a {lat.max():.2f}°")
    lines.append(f"Longitude:           {lon.min():.2f}° a {lon.max():.2f}°")
    lines.append(f"Pixels oceânicos:    {ocean.sum()} ({100.0 * ocean.sum() / (H * W):.2f}%)")
    lines.append(f"Pixels terra:        {(~ocean).sum()} ({100.0 * (~ocean).sum() / (H * W):.2f}%)")
    lines.append(f"")
    lines.append(f"Unidade da SST:      Kelvin (sem conversão)")
    lines.append(f"SST mín:             {valid.min():.4f} K ({valid.min() - 273.15:.4f} °C)")
    lines.append(f"SST máx:             {valid.max():.4f} K ({valid.max() - 273.15:.4f} °C)")
    lines.append(f"SST média:           {valid.mean():.4f} K ({valid.mean() - 273.15:.4f} °C)")
    lines.append(f"SST desvio padrão:   {valid.std():.4f} K")
    lines.append(f"")
    lines.append(f"NaN total:           {nan_mask.sum():,} de {sst.size:,} ({100.0 * nan_mask.sum() / sst.size:.4f}%)")
    lines.append(f"NaN por frame:       constante = {nan_per_time.min()} (min=max={nan_per_time.min() == nan_per_time.max()})")
    lines.append(f"NaN oceânicos:       {nan_mask.sum() - nan_per_time.min() * T}")
    lines.append(f"sea_ice_fraction:    {'toda zero' if ds['sea_ice_fraction'].values[np.isfinite(ds['sea_ice_fraction'].values)].max() == 0 else 'contém gelo'}")
    lines.append(f"Variáveis:           analysed_sst, analysis_error, mask, sea_ice_fraction")

    return "\n".join(lines)


def main():
    report = []
    report.append(f"ANÁLISE EXPLORATÓRIA DOS DADOS BRUTOS")
    report.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Script: scripts/eda_raw_data.py")

    # 0. Concatenar
    print("Concatenando arquivos brutos...")
    ds, concat_report = concatenate_raw_files()
    report.append(concat_report)

    # Carregar SST em memória uma vez (usado em várias análises)
    print("Carregando analysed_sst em memória...")
    sst = ds["analysed_sst"].values  # (T, H, W), float32, ~360 MB

    # 1. Estrutura
    print("1. Estrutura do dataset...")
    report.append(describe_dataset(ds))

    # 2. Coordenadas
    print("2. Coordenadas...")
    report.append(analyze_coordinates(ds))

    # 3. SST global
    print("3. Estatísticas globais da SST...")
    report.append(analyze_sst(sst))

    # 4. SST espacial
    print("4. Variabilidade espacial da SST...")
    report.append(analyze_sst_spatial(sst))

    # 5. NaN
    print("5. Estrutura de NaN...")
    report.append(analyze_nan_structure(sst))

    # 6. Mask
    print("6. Mask (bitmask)...")
    report.append(analyze_mask(ds))

    # 7. Sea ice
    print("7. Sea ice fraction...")
    report.append(analyze_sea_ice(ds))

    # 8. Analysis error
    print("8. Analysis error...")
    report.append(analyze_analysis_error(ds))

    # 9. Resumo
    print("9. Resumo...")
    report.append(summary(ds, sst))

    ds.close()

    # Saída
    full_report = "\n".join(report)
    print(full_report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(full_report)

    print(f"\nRelatório salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()