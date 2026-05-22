"""A-B rota analizi: iki kavsak arasi en kisa yol ve kapanmanin rotaya etkisi.

Kapanan bir kavsak/yolun ulasimi ne kadar etkiledigini, teknik bilmeyen birinin
de anlayacagi sekilde gosterir: rota uzadi mi, koptu mu, kac kat maliyet olustu.
"""
import math

import networkx as nx

from src.simulation.closure import apply_closure


def shortest_route(graph, source, target, weight="length"):
    """source -> target en kisa rotasi. Doner: (node_listesi, uzunluk) veya (None, None)."""
    if source == target:
        return [source], 0.0
    if not (graph.has_node(source) and graph.has_node(target)):
        return None, None
    try:
        length, path = nx.single_source_dijkstra(graph, source, target,
                                                  weight=weight)
        return path, round(float(length), 1)
    except nx.NetworkXNoPath:
        return None, None


def _euclidean(graph, a, b):
    """Iki kavsak arasi kus ucusu (duz cizgi) piksel mesafesi."""
    node_a, node_b = graph.nodes[a], graph.nodes[b]
    return math.hypot(node_a["x"] - node_b["x"], node_a["y"] - node_b["y"])


def route_impact(graph, source, target, closed_nodes=None, closed_edges=None):
    """Bir kapanmanin A-B rotasina etkisi: kapanma oncesi/sonrasi karsilastirma.

    detour: rota uzunlugunun kus ucusu mesafeye orani (1.0 = dumduz; buyukse
    rota o kadar dolambacli). extension_pct: kapanmanin rotayi uzatma yuzdesi.
    """
    before_path, before_length = shortest_route(graph, source, target)
    damaged = apply_closure(graph, closed_nodes, closed_edges)
    if damaged.has_node(source) and damaged.has_node(target):
        after_path, after_length = shortest_route(damaged, source, target)
    else:
        after_path, after_length = None, None

    straight = _euclidean(graph, source, target)
    result = {
        "source": int(source),
        "target": int(target),
        "before_path": before_path,
        "before_length": before_length,
        "after_path": after_path,
        "after_length": after_length,
        "reachable_before": before_path is not None,
        "reachable_after": after_path is not None,
        "straight_line": round(straight, 1),
        "detour_before": (round(before_length / straight, 2)
                          if before_length and straight > 0 else None),
        "detour_after": (round(after_length / straight, 2)
                         if after_length and straight > 0 else None),
    }
    if before_length and after_length and before_length > 0:
        result["extension_pct"] = round(
            100 * (after_length - before_length) / before_length, 1)
    else:
        result["extension_pct"] = 0.0 if result["reachable_after"] else None
    return result
