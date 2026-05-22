"""Yol grafigi uzerinde merkezilik (centrality) olcumleri."""
import networkx as nx


def node_betweenness(graph, weight="length"):
    """Her node icin betweenness centrality.

    Agirlik olarak kenar uzunlugu kullanilir: en kisa yollar gercek yol
    mesafesini takip eder, dolayisiyla skor "kac guzergah bu kavsaktan gecer"
    sorusunu fiziksel olarak anlamli sekilde yanitlar.
    """
    if graph.number_of_nodes() < 3:
        return {n: 0.0 for n in graph.nodes()}
    return nx.betweenness_centrality(graph, weight=weight, normalized=True)


def edge_betweenness(graph, weight="length"):
    """Her kenar icin betweenness centrality (kac en-kisa-yol o kenari kullanir)."""
    if graph.number_of_edges() == 0:
        return {}
    return nx.edge_betweenness_centrality(graph, weight=weight, normalized=True)


def annotate_centrality(graph, weight="length"):
    """Centrality skorlarini graph node/edge ozniteliklerine yazar."""
    node_scores = node_betweenness(graph, weight)
    edge_scores = edge_betweenness(graph, weight)
    for node, score in node_scores.items():
        graph.nodes[node]["betweenness"] = score
    for (u, v), score in edge_scores.items():
        graph[u][v]["betweenness"] = score
    return node_scores, edge_scores
