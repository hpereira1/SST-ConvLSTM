
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "overleaftcc" / "images" / "cap4" / "rmse_por_horizonte.png"

HORIZONS = [1, 2, 3, 4, 5, 6, 7]

RMSE = {
    "1C":            [0.2460, 0.3701, 0.4575, 0.5154, 0.5634, 0.6041, 0.6389],
    "2C-BN-D-S-PH":  [0.2423, 0.3657, 0.4513, 0.5078, 0.5545, 0.5950, 0.6297],
    "2C-PH":         [0.2435, 0.3646, 0.4470, 0.5023, 0.5478, 0.5866, 0.6211],
}
SS = {  # em %
    "1C":            [3.78, 5.68, 6.73, 7.82, 8.63, 9.32, 10.02],
    "2C-BN-D-S-PH":  [5.22, 6.81, 8.00, 9.18, 10.06, 10.69, 11.31],
    "2C-PH":         [4.75, 7.08, 8.88, 10.17, 11.16, 11.96, 12.53],
}


def persistence_rmse():
    est = np.array([
        np.array(RMSE[m]) / (1.0 - np.array(SS[m]) / 100.0) for m in RMSE
    ])
    if est.std(axis=0).max() > 1e-3:
        print("aviso: persistência diverge entre modelos:", est.std(axis=0))
    return est.mean(axis=0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    persist = persistence_rmse()

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in RMSE:
        ax.plot(HORIZONS, RMSE[name], label=name)
    ax.plot(HORIZONS, persist, ls="--", color="gray", label="Persistência")
    ax.set_xlabel("Horizonte de previsão (dias)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {OUT}")
    print("persistência (°C):", " ".join(f"{v:.3f}" for v in persist))


if __name__ == "__main__":
    main()
