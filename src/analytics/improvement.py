"""Resilience iyilestirme onerisi: agi en cok guclendirecek yeni yol baglantisi.

Sistem aday yeni baglantilar dener, her birinin resilience skoruna katkisini
olcer ve en yuksek kazanci saglayanlari onerir. Boylece arac yalnizca analiz
eden degil, oneri sunan bir karar destek sistemine donusur.
"""
import math

import networkx as nx

from src.analytics.criticality import analyze
from src.analytics.resilience import resilience_score
from src.simulation.attack import simulate


def _distance(graph, a, b):
    """Iki kavsak arasi kus ucusu piksel mesafesi."""
    node_a, node_b = graph.nodes[a], graph.nodes[b]
    return math.hypot(node_a["x"] - node_b["x"], node_a["y"] - node_b["y"])


def _closest_pair(graph, nodes_a, nodes_b):
    """Iki node kumesi arasinda geometrik olarak en yakin (a, b) ikilisi."""
    best, best_distance = None, float("inf")
    for a in nodes_a:
        for b in nodes_b:
            distance = _distance(graph, a, b)
            if distance < best_distance:
                best, best_distance = (a, b), distance
    return best


def candidate_connections(graph, max_candidates=12):
    """Denenecek yeni baglanti adaylarini uretir.

    Oncelik bilesenler arasi en yakin node ciftleridir: parcali bir agi
    birlestirmek dayanikliligi en cok artiran mudahaledir. Ek olarak bilesen
    ici dusuk dereceli (cikmaz) kavsaklar, kopru/cevrim ekleyerek yedeklilik
    kazandiracak sekilde eslestirilir.
    """
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    candidates = []

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            pair = _closest_pair(graph, components[i], components[j])
            if pair:
                candidates.append(pair)

    for component in components:
        dead_ends = [n for n in component if graph.degree(n) <= 2]
        for node in dead_ends:
            others = [other for other in dead_ends
                      if other != node and not graph.has_edge(node, other)]
            if others:
                nearest = min(others, key=lambda o: _distance(graph, node, o))
                candidates.append((node, nearest))

    unique, seen = [], set()
    for u, v in candidates:
        key = tuple(sorted((u, v)))
        if key not in seen:
            seen.add(key)
            unique.append((u, v))
    return unique[:max_candidates]


def _network_resilience(graph, config):
    """analyze + simulate + resilience_score zincirini calistirip skoru doner."""
    analysis = analyze(graph, config)
    simulation = simulate(graph, config)
    return resilience_score(graph, analysis, simulation, config)


def suggest_improvements(graph, config, top_n=3):
    """En cok resilience kazandiran yeni baglantilari siralayarak onerir.

    Onerilen kenar, gercek bir skeleton yolu olmadigi icin duz cizgi (pts=None)
    olarak eklenir; uzunlugu iki kavsak arasi kus ucusu mesafedir.
    """
    base = _network_resilience(graph, config)
    evaluated = []
    for u, v in candidate_connections(graph):
        improved = graph.copy()
        length = _distance(graph, u, v)
        improved.add_edge(u, v, length=length, weight=length, pts=None)
        new = _network_resilience(improved, config)
        evaluated.append({
            "edge": [int(u), int(v)],
            "new_length": round(length, 1),
            "score_after": new["score"],
            "gain": round(new["score"] - base["score"], 1),
        })

    evaluated.sort(key=lambda item: item["gain"], reverse=True)
    return {
        "base_score": base["score"],
        "base_grade": base["grade"],
        "suggestions": evaluated[:top_n],
    }
