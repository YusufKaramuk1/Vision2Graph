"""Kademeli saldiri sonucu agin cokus (dayaniklilik) egrisini cizer."""
import matplotlib

matplotlib.use("Agg")  # penceresiz ortam: dosyaya cizim
import matplotlib.pyplot as plt


def draw_degradation_plot(simulation, output_path=None):
    """Kasitli ve rastsal saldiri LCR egrilerini tek grafikte cizer.

    Matplotlib figurunu dondurur; output_path verilirse ayrica PNG kaydeder.
    """
    targeted = simulation["targeted"]
    random_attack = simulation["random"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([s["fraction"] for s in targeted["curve"]],
            [s["lcr"] for s in targeted["curve"]],
            "o-", color="#d62728",
            label=f"Kasitli saldiri (R={targeted['robustness_index']})")
    ax.plot([s["fraction"] for s in random_attack["curve"]],
            [s["lcr"] for s in random_attack["curve"]],
            "s-", color="#1f77b4",
            label=f"Rastsal ariza (R={random_attack['robustness_index']})")

    ax.set_xlabel("Kaldirilan node orani")
    ax.set_ylabel("En buyuk bilesen orani (LCR)")
    ax.set_title("Yol Agi Dayaniklilik Egrisi")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120)
    return fig
