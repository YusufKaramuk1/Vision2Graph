"""Graph metrikleri, kritiklik, resilience ve rota analizinin testleri."""
from src.analytics.criticality import analyze, node_criticality
from src.analytics.explainability import explain_critical_nodes
from src.analytics.resilience import resilience_score
from src.analytics.routing import route_impact, shortest_route
from src.analytics.vulnerability import articulation_points, bridge_edges
from src.analytics.worst_case import largest_component_ratio
from src.simulation.attack import simulate
from src.topology.graph_builder import basic_counts


def test_basic_counts(sample_graph):
    counts = basic_counts(sample_graph)
    assert counts["node_count"] == 7
    assert counts["edge_count"] == 5
    assert counts["component_count"] == 2


def test_articulation_and_bridges(sample_graph):
    points = articulation_points(sample_graph)
    assert points == {1}  # sadece yildiz merkezi kesim noktasi
    # orman yapisinda her kenar koprudur
    assert len(bridge_edges(sample_graph)) == 5


def test_largest_component_ratio(sample_graph):
    # en buyuk bilesen 5 kavsak (0-1-2-3-4), toplam 7
    assert largest_component_ratio(sample_graph) == 5 / 7


def test_node_criticality_in_range(sample_graph):
    analyze(sample_graph, {})
    scores = node_criticality(sample_graph)
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    # merkez kavsak en kritik olmali
    assert max(scores, key=scores.get) == 1


def test_resilience_score_range(sample_graph):
    analysis = analyze(sample_graph, {})
    result = resilience_score(sample_graph, analysis,
                              simulate(sample_graph, {}), {})
    assert 0.0 <= result["score"] <= 100.0
    assert result["grade"] in {"A", "B", "C", "D", "E"}


def test_shortest_route(sample_graph):
    path, length = shortest_route(sample_graph, 0, 2)
    assert path == [0, 1, 2]
    assert length > 0
    # farkli bilesenler arasinda rota yoktur
    assert shortest_route(sample_graph, 0, 5) == (None, None)


def test_route_impact_cut(sample_graph):
    # merkez kavsak 1 kapaninca 0-2 baglantisi tamamen kopar
    impact = route_impact(sample_graph, 0, 2, closed_nodes=[1])
    assert impact["reachable_before"] is True
    assert impact["reachable_after"] is False


def test_explanations_produced(sample_graph):
    analysis = analyze(sample_graph, {})
    explanations = explain_critical_nodes(sample_graph, analysis)
    assert len(explanations) >= 1
    assert all(item["explanation"] for item in explanations)
