
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "acc_ra_por_modelo.csv"
OUT = ROOT / "overleaftcc" / "images" / "cap4"


MODELS = ["1C", "2C-BN-D-S-PH", "2C-PH"]


def curve(df, metric, ref, ref_label, ylabel, fname):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODELS:
        sub = df[df["modelo"] == m].sort_values("horizonte")
        ax.plot(sub["horizonte"], sub[metric], label=m)
    ax.axhline(ref, ls=":", color="red", alpha=0.7, label=ref_label)
    ax.set_xlabel("Horizonte de previsão (dias)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df["horizonte"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT / fname}")


def main():
    df = pd.read_csv(CSV)
    curve(df, "ACC", 0.6, "Limiar 0,6 (ECMWF)", "ACC", "acc_por_horizonte.png")
    curve(df, "RA", 1.0, "RA = 1 (atividade da observação)",
          "Atividade relativa (RA)", "ra_por_horizonte.png")


if __name__ == "__main__":
    main()
