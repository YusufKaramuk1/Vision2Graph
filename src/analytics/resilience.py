"""Resilience score: tum metrikleri tek yorumlanabilir nota (0-100) toplar.

Bilesenler ve agirliklari config.yaml > resilience.weights altinda seffaf
tutulur. Her bilesen "yuksek = daha dayanikli" olacak sekilde 0-1'e normalize
edilir; kopru ve kesim noktasi oranlari kirilganlik gosterdigi icin tersi
(1 - oran) alinir. Agirlikli ortalama 100 ile carpilarak skora donusur.
"""
from src.analytics.worst_case import largest_component_ratio

DEFAULT_WEIGHTS = {
    "largest_component_ratio": 0.30,
    "avg_degree": 0.15,
    "bridge_ratio": 0.20,
    "articulation_ratio": 0.20,
    "targeted_robustness": 0.15,
}

# avg_degree 0-1'e olceklenirken referans aralik: derece 1 = sadece zincir uclari
# (kotu), derece 4 = izgara benzeri bol baglantili kavsaklar (iyi).
_DEGREE_LOW, _DEGREE_HIGH = 1.0, 4.0


def _grade(score):
    """0-100 skoru harf notu ve aciklamaya cevirir."""
    if score >= 80:
        return "A", "Cok dayanikli"
    if score >= 60:
        return "B", "Dayanikli"
    if score >= 40:
        return "C", "Orta"
    if score >= 20:
        return "D", "Kirilgan"
    return "E", "Cok kirilgan"


def resilience_score(graph, analysis, simulation, config=None):
    """Faz 4 ana giris noktasi: Faz 2-3 metriklerini tek skorda birlestirir.

    analysis: Faz 2 analyze() ciktisi (articulation/bridge sayilari).
    simulation: Faz 3 simulate() ciktisi (kasitli saldiri dayaniklilik indeksi).
    """
    weights = (config or {}).get("resilience", {}).get("weights") or DEFAULT_WEIGHTS
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    degrees = [d for _, d in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
    degree_score = min(1.0, max(0.0,
                       (avg_degree - _DEGREE_LOW) / (_DEGREE_HIGH - _DEGREE_LOW)))
    bridge_ratio = analysis["bridge_count"] / edge_count if edge_count else 1.0
    articulation_ratio = (analysis["articulation_count"] / node_count
                          if node_count else 1.0)

    # Bilesen anahtarlari config'deki agirlik anahtarlariyla eslesir; degerler
    # "yuksek = dayanikli" yonune cevrilmis 0-1 skorlardir.
    components = {
        "largest_component_ratio": round(largest_component_ratio(graph), 4),
        "avg_degree": round(degree_score, 4),
        "bridge_ratio": round(1.0 - bridge_ratio, 4),
        "articulation_ratio": round(1.0 - articulation_ratio, 4),
        "targeted_robustness": round(simulation["targeted"]["robustness_index"], 4),
    }

    total_weight = sum(weights.values()) or 1.0
    score = 100.0 * sum(weights[key] * value
                        for key, value in components.items()) / total_weight
    grade, label = _grade(score)

    return {
        "score": round(score, 1),
        "grade": grade,
        "label": label,
        "components": components,
        "weights": dict(weights),
        "raw": {
            "avg_degree": round(avg_degree, 3),
            "bridge_ratio": round(bridge_ratio, 4),
            "articulation_ratio": round(articulation_ratio, 4),
        },
    }
