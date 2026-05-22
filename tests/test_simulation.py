"""Kapanma etki motoru ve kademeli saldiri simulasyonunun testleri."""
from src.simulation.attack import robustness_index, simulate
from src.simulation.closure import apply_closure, closure_impact


def test_apply_closure_does_not_mutate_input(sample_graph):
    damaged = apply_closure(sample_graph, closed_nodes=[1])
    assert not damaged.has_node(1)
    assert sample_graph.has_node(1)  # orijinal grafik degismemeli


def test_closure_increases_components(sample_graph):
    # merkez kavsagin kapanmasi agi parcalara boler
    impact = closure_impact(sample_graph, closed_nodes=[1])
    assert impact["component_increase"] > 0
    assert impact["isolated_nodes"] >= 0
    assert 0.0 <= impact["isolation_ratio_pct"] <= 100.0


def test_closure_leaf_node_low_impact(sample_graph):
    # yaprak kavsak (derece 1) kapaninca ag bolunmez
    impact = closure_impact(sample_graph, closed_nodes=[0])
    assert impact["component_increase"] == 0


def test_simulate_returns_three_curves(sample_graph):
    simulation = simulate(sample_graph, {})
    for strategy in ("random", "targeted", "adaptive"):
        assert strategy in simulation
        index = simulation[strategy]["robustness_index"]
        assert 0.0 <= index <= 1.0


def test_robustness_index_bounds():
    full = [{"fraction": 0.0, "lcr": 1.0}, {"fraction": 1.0, "lcr": 1.0}]
    assert robustness_index(full) == 1.0
    collapsed = [{"fraction": 0.0, "lcr": 0.0}, {"fraction": 1.0, "lcr": 0.0}]
    assert robustness_index(collapsed) == 0.0
