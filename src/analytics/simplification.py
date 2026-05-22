"""Topolojik sadelestirme: yol grafigini farkli detay seviyelerinde gosterir.

En dusuk kritiklikli yollar atilarak grafik daha genel bir 'omurga' seviyesine
indirgenir. Bu, TUBITAK basvurusundaki harita genellestirme fikrinin yol
katmanina uygulanmis halidir: ayrintidan vazgecip stratejik iskeleti korumak.

Sadelestirme yuzdelik tabanlidir: en kritik kenarlarin belirli bir orani
tutulur. Boylece grafik dagilimi ne olursa olsun seviyeler belirgin sekilde
ayrisir.
"""
import networkx as nx

# detay seviyesi -> (ad, tutulacak en kritik kenar orani)
DETAIL_LEVELS = {
    1: ("Tum yollar", 1.0),
    2: ("Ana ve onemli yollar", 0.6),
    3: ("Sadece kritik omurga", 0.3),
}


def simplify_graph(graph, keep_ratio):
    """En kritik kenarlarin keep_ratio oranini tutar, gerisini atar.

    Graph daha once analyze() ile islenmis olmalidir; kenar 'criticality'
    onceligi bu adimda yazilir. Kenar kalmayan kavsaklar da temizlenir.
    """
    simplified = graph.copy()
    ranked = sorted(simplified.edges(data=True),
                    key=lambda edge: edge[2].get("criticality", 0.0),
                    reverse=True)
    keep_count = int(round(len(ranked) * keep_ratio))
    for u, v, _ in ranked[keep_count:]:
        simplified.remove_edge(u, v)
    simplified.remove_nodes_from(list(nx.isolates(simplified)))
    return simplified


def _total_length(graph):
    return sum(data.get("length", 0.0) for _, _, data in graph.edges(data=True))


def generalization_levels(graph):
    """Uc detay seviyesi icin sadelestirilmis grafikleri ve istatistikleri uretir."""
    base_edges = graph.number_of_edges()
    base_length = _total_length(graph)
    levels = []
    for level, (name, keep_ratio) in sorted(DETAIL_LEVELS.items()):
        simplified = simplify_graph(graph, keep_ratio)
        edges = simplified.number_of_edges()
        length = _total_length(simplified)
        levels.append({
            "level": level,
            "name": name,
            "keep_ratio": keep_ratio,
            "graph": simplified,
            "node_count": simplified.number_of_nodes(),
            "edge_count": edges,
            "edge_ratio_pct": round(100 * edges / base_edges, 1) if base_edges else 0.0,
            "length_ratio_pct": (round(100 * length / base_length, 1)
                                 if base_length else 0.0),
        })
    return levels
