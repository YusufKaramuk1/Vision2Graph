"""Skeleton goruntusunu NetworkX yol grafigine donusturur ve temizler."""
import networkx as nx
import numpy as np
import sknw


def _edge_length(pts):
    """Kenar uzunlugu: skeleton boyunca gercek piksel mesafesi (duz cizgi degil)."""
    if pts is None or len(pts) < 2:
        return 0.0
    diffs = np.diff(np.asarray(pts, dtype=np.float64), axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def build_graph(skeleton, spur_length_threshold=15, min_component_length=30):
    raw = sknw.build_sknw(skeleton)
    graph = nx.Graph()

    for node_id, data in raw.nodes(data=True):
        row, col = data["o"]
        graph.add_node(node_id, x=float(col), y=float(row))

    for s, e, data in raw.edges(data=True):
        if s == e:
            continue  # self-loop'lari at
        pts = data.get("pts")  # skeleton yol cizgisi (gorsellestime icin saklanir)
        length = _edge_length(pts)
        if graph.has_edge(s, e):
            if length < graph[s][e]["length"]:
                graph[s][e].update(length=length, weight=length, pts=pts)
            continue
        graph.add_edge(s, e, length=length, weight=length, pts=pts)

    graph = _prune_spurs(graph, spur_length_threshold)
    graph = _drop_small_components(graph, min_component_length)
    return nx.convert_node_labels_to_integers(graph)


def _prune_spurs(graph, threshold):
    """Kisa sarkan (degree-1) kenarlari iteratif olarak budar."""
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes()):
            if graph.degree(node) != 1:
                continue
            neighbor = next(iter(graph.neighbors(node)))
            if graph[node][neighbor]["length"] < threshold:
                graph.remove_node(node)
                changed = True
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def _drop_small_components(graph, min_length):
    """Toplam uzunlugu esigin altinda kalan bilesenleri (skeleton gurultusu) eler.

    Filtre node sayisina degil uzunluga bakar: tek bir uzun yol segmenti
    yalnizca 2 node'dan olussa bile gecerli bir yoldur, atilmamalidir.
    """
    for component in list(nx.connected_components(graph)):
        sub = graph.subgraph(component)
        total = sum(d["length"] for _, _, d in sub.edges(data=True))
        if total < min_length:
            graph.remove_nodes_from(component)
    return graph


def basic_counts(graph):
    degrees = [d for _, d in graph.degree()]
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "component_count": nx.number_connected_components(graph),
        "average_degree": round(sum(degrees) / len(degrees), 3) if degrees else 0.0,
        "total_length": round(sum(d["length"] for _, _, d in graph.edges(data=True)), 1),
    }
