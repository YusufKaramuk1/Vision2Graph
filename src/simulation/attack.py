"""Kademeli saldiri simulasyonu: ag node kaybettikce nasil cokegidigi.

Iki strateji karsilastirilir:
  - targeted: en kritik node'lar once kaldirilir (kasitli saldiri / oncelikli ariza)
  - random:   node'lar rastgele kaldirilir (dogal/rastsal ariza)

Saglam bir ag rastsal arizaya dayaniklidir ama kasitli saldiriya kirilgandir;
iki egri arasindaki fark agin kirilganlik profilini ortaya koyar.
"""
import random

from src.analytics.centrality import annotate_centrality
from src.analytics.criticality import node_criticality
from src.analytics.vulnerability import annotate_vulnerability
from src.simulation.closure import network_state


def _removal_order(graph, strategy, seed):
    """Node'larin kaldirilma sirasini stratejiye gore belirler."""
    nodes = list(graph.nodes())
    if strategy == "random":
        random.Random(seed).shuffle(nodes)
        return nodes

    # targeted: kritiklik skoruna gore azalan. Graph daha once analiz edilmisse
    # skorlar ozniteliklerde hazirdir; degilse burada hesaplanir.
    scores = {n: graph.nodes[n].get("criticality") for n in nodes}
    if any(score is None for score in scores.values()):
        annotate_centrality(graph)
        annotate_vulnerability(graph)
        scores = node_criticality(graph)
    return sorted(nodes, key=lambda n: scores[n], reverse=True)


def progressive_attack(graph, strategy="targeted", max_fraction=0.5, seed=0):
    """Node'lari sirayla kaldirir, her adimda ag durumunu kaydeder.

    max_fraction: node'larin en fazla bu orani kaldirilir; egrinin anlamli
    kismi bu araliktadir, tamamini kaldirmak gereksiz hesap yukudur.
    """
    order = _removal_order(graph, strategy, seed)
    limit = int(len(order) * max_fraction)
    total = graph.number_of_nodes()
    working = graph.copy()

    curve = [{"removed": 0, "fraction": 0.0, **network_state(working)}]
    for index, node in enumerate(order[:limit], start=1):
        working.remove_node(node)
        curve.append({"removed": index,
                      "fraction": round(index / total, 4),
                      **network_state(working)})
    return curve


def random_attack(graph, max_fraction=0.5, runs=5):
    """Birden cok rastgele kosunun ortalama egrisi (tek kosu gurultulu olur)."""
    curves = [progressive_attack(graph, "random", max_fraction, seed=run)
              for run in range(runs)]
    length = min(len(curve) for curve in curves)
    averaged = []
    for step in range(length):
        averaged.append({
            "removed": curves[0][step]["removed"],
            "fraction": curves[0][step]["fraction"],
            "lcr": round(sum(c[step]["lcr"] for c in curves) / runs, 4),
            "efficiency": round(sum(c[step]["efficiency"] for c in curves) / runs, 6),
        })
    return averaged


def robustness_index(curve):
    """LCR egrisi altindaki normalize alan (0-1): yuksek = dayanikli."""
    if len(curve) < 2:
        return 0.0
    area = sum(0.5 * (a["lcr"] + b["lcr"]) * (b["fraction"] - a["fraction"])
               for a, b in zip(curve, curve[1:]))
    span = curve[-1]["fraction"] - curve[0]["fraction"]
    return round(area / span, 4) if span > 0 else 0.0


def simulate(graph, config=None):
    """Faz 3 ana giris noktasi: kasitli ve rastsal saldiri egrilerini uretir.

    fragility_gap: rastsal ve kasitli dayaniklilik indeksleri arasindaki fark.
    Buyuk deger = ag kasitli saldiriya belirgin sekilde kirilgan.
    """
    settings = (config or {}).get("simulation", {})
    max_fraction = settings.get("max_fraction", 0.5)
    random_runs = settings.get("random_runs", 5)

    targeted = progressive_attack(graph, "targeted", max_fraction)
    random_curve = random_attack(graph, max_fraction, random_runs)
    targeted_r = robustness_index(targeted)
    random_r = robustness_index(random_curve)

    return {
        "max_fraction": max_fraction,
        "targeted": {"curve": targeted, "robustness_index": targeted_r},
        "random": {"curve": random_curve, "robustness_index": random_r},
        "fragility_gap": round(random_r - targeted_r, 4),
    }
