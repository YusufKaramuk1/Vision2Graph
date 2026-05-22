"""Kritiklik skoru: merkezilik ve yapisal kirilganligi tek skorda birlestirir.

Skor bilesenleri ve agirliklari bilincli olarak seffaf tutulur (juriye
savunulabilir olmasi icin) ve config.yaml uzerinden ayarlanabilir.
"""
from src.analytics.centrality import annotate_centrality
from src.analytics.vulnerability import annotate_vulnerability
from src.analytics.worst_case import worst_case_nodes

DEFAULT_NODE_WEIGHTS = {"betweenness": 0.50, "articulation": 0.35, "degree": 0.15}
DEFAULT_EDGE_WEIGHTS = {"betweenness": 0.60, "bridge": 0.40}


def _normalize(values):
    """Sozluk degerlerini min-max ile 0-1 araligina olcekler."""
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high - low < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def node_criticality(graph, weights=None):
    """Her node icin 0-1 kritiklik skoru.

    Bilesenler: betweenness (kac guzergah gecer), articulation (kaldirilinca
    ag parcalanir mi), degree (kavsak baglanti sayisi).
    """
    weights = weights or DEFAULT_NODE_WEIGHTS
    betweenness = _normalize(
        {n: graph.nodes[n].get("betweenness", 0.0) for n in graph.nodes()})
    degree = _normalize({n: graph.degree(n) for n in graph.nodes()})

    scores = {}
    for node in graph.nodes():
        articulation = 1.0 if graph.nodes[node].get("is_articulation") else 0.0
        scores[node] = (weights["betweenness"] * betweenness[node]
                        + weights["articulation"] * articulation
                        + weights["degree"] * degree[node])
    return scores


def edge_criticality(graph, weights=None):
    """Her kenar icin 0-1 kritiklik skoru (betweenness + kopru olma durumu)."""
    weights = weights or DEFAULT_EDGE_WEIGHTS
    betweenness = _normalize(
        {(u, v): graph[u][v].get("betweenness", 0.0) for u, v in graph.edges()})

    scores = {}
    for u, v in graph.edges():
        bridge = 1.0 if graph[u][v].get("is_bridge") else 0.0
        scores[(u, v)] = (weights["betweenness"] * betweenness[(u, v)]
                          + weights["bridge"] * bridge)
    return scores


def _rank_nodes(scores, top_k):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"node": int(n), "score": round(float(s), 4)} for n, s in ranked]


def _rank_edges(scores, top_k):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"edge": [int(u), int(v)], "score": round(float(s), 4)}
            for (u, v), s in ranked]


def analyze(graph, config=None):
    """Faz 2 ana giris noktasi.

    Grafigi yerinde analiz eder (centrality, kirilganlik ve kritiklik skorlarini
    node/edge ozniteliklerine yazar), worst-case simulasyonu calistirir ve
    raporlanabilir bir ozet sozlugu dondurur.
    """
    settings = (config or {}).get("analytics", {})
    criticality_cfg = settings.get("criticality", {})
    node_weights = criticality_cfg.get("node_weights") or DEFAULT_NODE_WEIGHTS
    edge_weights = criticality_cfg.get("edge_weights") or DEFAULT_EDGE_WEIGHTS
    top_k = settings.get("worst_case_top_k", 5)

    annotate_centrality(graph)
    points, bridges = annotate_vulnerability(graph)

    node_scores = node_criticality(graph, node_weights)
    edge_scores = edge_criticality(graph, edge_weights)
    for node, score in node_scores.items():
        graph.nodes[node]["criticality"] = score
    for (u, v), score in edge_scores.items():
        graph[u][v]["criticality"] = score

    return {
        "articulation_count": len(points),
        "bridge_count": len(bridges),
        "top_critical_nodes": _rank_nodes(node_scores, top_k),
        "top_critical_edges": _rank_edges(edge_scores, top_k),
        "worst_case": worst_case_nodes(graph, node_scores, top_k),
    }
