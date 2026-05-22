"""Resilience skor kartini cizer: bilesen bazli yatay bar grafigi."""
import matplotlib

matplotlib.use("Agg")  # penceresiz ortam: dosyaya cizim
import matplotlib.pyplot as plt

_LABELS = {
    "largest_component_ratio": "En buyuk bilesen",
    "avg_degree": "Ortalama derece",
    "bridge_ratio": "Kopru azligi",
    "articulation_ratio": "Kesim noktasi azligi",
    "targeted_robustness": "Saldiri dayanikliligi",
}


def _bar_color(value):
    """Bilesen skorunu trafik isigi rengine esler."""
    if value >= 0.60:
        return "#2ca02c"
    if value >= 0.35:
        return "#ff7f0e"
    return "#d62728"


def draw_resilience_card(resilience, output_path):
    """Resilience bilesenlerini ve toplam skoru tek grafikte cizip kaydeder."""
    components = resilience["components"]
    keys = list(components.keys())
    values = [components[key] for key in keys]
    labels = [_LABELS.get(key, key) for key in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(labels, values, color=[_bar_color(v) for v in values])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Bilesen skoru (0-1, yuksek = dayanikli)")
    ax.set_title(f"Resilience Skoru: {resilience['score']} / 100   "
                 f"({resilience['grade']} - {resilience['label']})")
    for index, value in enumerate(values):
        ax.text(min(value + 0.02, 0.95), index, f"{value:.2f}", va="center")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
