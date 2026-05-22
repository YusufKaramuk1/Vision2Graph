"""Kademeli saldiri sonucu agin cokus (dayaniklilik) egrisini cizer."""
import matplotlib

matplotlib.use("Agg")  # penceresiz ortam: dosyaya cizim
import matplotlib.pyplot as plt


def draw_degradation_plot(simulation, output_path=None):
    """Rastsal, statik kasitli ve adaptif saldiri LCR egrilerini cizer.

    Matplotlib figurunu dondurur; output_path verilirse ayrica PNG kaydeder.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # (anahtar, renk, marker, etiket) - varsa adaptif egri de cizilir
    series = [
        ("random", "#1f77b4", "s-", "Rastsal ariza"),
        ("targeted", "#ff7f0e", "o-", "Kasitli saldiri (statik)"),
        ("adaptive", "#d62728", "^-", "Kasitli saldiri (adaptif)"),
    ]
    for key, color, marker, label in series:
        if key not in simulation:
            continue
        curve = simulation[key]["curve"]
        ax.plot([s["fraction"] for s in curve], [s["lcr"] for s in curve],
                marker, color=color,
                label=f"{label} (R={simulation[key]['robustness_index']})")

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
