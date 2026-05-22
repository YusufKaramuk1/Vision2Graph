"""PDF raporun yonetici ozeti uretiminin testleri."""
from src.analytics.criticality import analyze
from src.analytics.explainability import explain_critical_nodes
from src.analytics.resilience import resilience_score
from src.reporting.pdf_report import _executive_summary
from src.simulation.attack import simulate
from src.topology.graph_builder import basic_counts


def test_executive_summary_text(sample_graph):
    analysis = analyze(sample_graph, {})
    analysis["explanations"] = explain_critical_nodes(sample_graph, analysis)
    simulation = simulate(sample_graph, {})
    resilience = resilience_score(sample_graph, analysis, simulation, {})

    summary = _executive_summary(basic_counts(sample_graph), analysis,
                                 simulation, resilience)
    assert isinstance(summary, str)
    assert len(summary) > 50
    # ozet ag boyutunu ve dayaniklilik skorunu icermeli
    assert "kavsak" in summary
    assert "dayaniklilik skoru" in summary
