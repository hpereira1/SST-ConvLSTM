
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RUNS = Path(__file__).resolve().parent.parent / "runs"
OVERLEAF_IMG = Path(__file__).resolve().parent.parent / "overleaftcc" / "images" / "cap4"

STD_FIGSIZE = (6, 4)

PUB = {
    "vbase2camadas": dict(label="2C", ylim=(0.03, 0.12),
                          yticks=[0.03, 0.05, 0.07, 0.09, 0.11]),
}

MODELS = [
    ("vbase",          "variante_1C"),
    ("vbase2camadas",  "variante_2C"),
    ("v0",             "variante_2C-BN-D"),
    ("v3",             "variante_2C-BN-D-PH"),
    ("v4",             "variante_2C-BN-D-S"),
    ("v5",             "variante_2C-BN-D-S-PH"),
    ("v0_clean",       "variante_2C-BN"),
    ("v3_clean",       "variante_2C-BN-PH"),
    ("v4_clean",       "variante_2C-BN-S"),
    ("v5_clean",       "variante_2C-BN-S-PH"),
    ("v4_nobn",        "variante_2C-D-S"),
    ("v3_clean_nobn",  "variante_2C-PH"),
    ("v4_clean_nobn",  "variante_2C-S"),
    ("v5_clean_nobn",  "variante_2C-S-PH"),
]


def plot_one(name: str, run_dir: Path) -> Path:
    history_path = run_dir / f"convlstm_{name}_masked_seed42_history.json"
    h = json.loads(history_path.read_text())

    epochs = list(range(1, len(h["loss"]) + 1))
    train  = h["loss"]
    val    = h["val_loss"]
    best_epoch = val.index(min(val)) + 1
    best_val   = min(val)

    pub = PUB.get(name)
    if pub:
        lbl_train, lbl_val = "Treino", "Validação"
        bv = f"{best_val:.4f}".replace(".", ",")
        lbl_best = f"Melhor val (ép. {best_epoch}, {bv})"
        ylabel = "Perda (MaskedMSE em espaço $z$)"
    else:
        lbl_train, lbl_val = "Train loss", "Val loss"
        lbl_best = f"Best val (ep {best_epoch}, {best_val:.4f})"
        ylabel = "Loss (MaskedMSE em z-score)"

    fig, ax = plt.subplots(figsize=STD_FIGSIZE)
    ax.plot(epochs, train, "-",  color="#1f77b4", linewidth=1.4, label=lbl_train)
    ax.plot(epochs, val,   "-",  color="#d62728", linewidth=1.4, label=lbl_val)

    _scatter = dict(color="#d62728", s=50, zorder=5,
                    edgecolor="white", linewidth=1.2)
    if pub:
        # Valor do melhor resultado anotado junto ao ponto (não na legenda).
        ax.scatter([best_epoch], [best_val], **_scatter)
        ax.annotate(f"ép. {best_epoch} · {bv}",
                    xy=(best_epoch, best_val), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8.5, color="#d62728")
    else:
        ax.scatter([best_epoch], [best_val], label=lbl_best, **_scatter)

    ax.axvline(best_epoch, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.4)

    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.3)

    if pub:
        # Sem título embutido: a legenda (\caption) do LaTeX é o título (ABNT).
        ax.set_ylim(*pub["ylim"])
        ax.set_yticks(pub["yticks"])
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y:.2f}".replace(".", ",")))
    else:
        ax.set_title(f"Curvas de loss — modelo {name}  ({len(epochs)} épocas)",
                     fontsize=12, fontweight="bold")
        ax.set_yscale("log")

    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out = run_dir / "loss_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    if pub:
        OVERLEAF_IMG.mkdir(parents=True, exist_ok=True)
        plt.savefig(OVERLEAF_IMG / f"loss_curve_{pub['label']}.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    for name, run_name in MODELS:
        run_dir = RUNS / run_name
        if not run_dir.exists():
            print(f"  skip {name}: run dir {run_name} not found")
            continue
        out = plot_one(name, run_dir)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()