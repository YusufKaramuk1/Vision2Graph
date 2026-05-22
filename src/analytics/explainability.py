"""Kritik node'lar icin insan-okuyabilir "neden kritik?" aciklamalari uretir.

Aciklama, Faz 2 metriklerini (betweenness, kesim noktasi, derece) node'un
kapatilmasi durumundaki somut sonucla (closure_impact) birlestirir. Amac,
juri "bu kavsak neden kritik?" diye sordugunda sistemin kendi icinden
savunulabilir bir cevap verebilmesidir.
"""
from src.simulation.closure import closure_impact

_HIGH_BETWEENNESS = 0.6        # normalize betweenness bu esigin ustunde "yuksek"
_DENSE_DEGREE = 4              # bu derece ve uzeri "yogun kavsak"
_NOTABLE_EFFICIENCY_LOSS = 5.0  # % - bu uzeri verim kaybi aciklamaya eklenir


def _normalized_betweenness(graph):
    """Node betweenness degerlerini 0-1 araligina olcekler (baglamsal 'yuksek' icin)."""
    values = {n: graph.nodes[n].get("betweenness", 0.0) for n in graph.nodes()}
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high - low < 1e-12:
        return {n: 0.0 for n in values}
    return {n: (v - low) / (high - low) for n, v in values.items()}


def explain_node(graph, node, betweenness_norm):
    """Tek bir node icin 'neden kritik' aciklamasi (Turkce cumle) uretir."""
    impact = closure_impact(graph, [node])

    reasons = []
    if betweenness_norm >= _HIGH_BETWEENNESS:
        reasons.append("agdaki en kisa guzergahlarin buyuk kismi bu kavsaktan gecer")
    if graph.nodes[node].get("is_articulation"):
        reasons.append("bir kesim noktasidir (tek basina agi bolen kavsak)")
    degree = graph.degree(node)
    if degree >= _DENSE_DEGREE:
        reasons.append(f"{degree} yolun birlestigi yogun bir kavsaktir")

    consequences = []
    if impact["component_increase"] > 0:
        consequences.append(
            f"ag {impact['before']['components']} parcadan "
            f"{impact['after']['components']} parcaya bolunur")
    if impact["isolated_nodes"] > 0:
        consequences.append(
            f"{impact['isolated_nodes']} kavsak ve yol agininin "
            f"%{impact['isolation_ratio_pct']}'i ana agdan kopar")
    if impact["efficiency_loss_pct"] >= _NOTABLE_EFFICIENCY_LOSS:
        consequences.append(
            f"ulasim verimliligi %{impact['efficiency_loss_pct']} duser")

    if reasons:
        text = f"Node {node} kritik: " + "; ".join(reasons) + "."
    else:
        text = f"Node {node} yapisal olarak orta onemdedir."
    if consequences:
        text += " Kapatildiginda " + ", ".join(consequences) + "."
    return text


def explain_critical_nodes(graph, analysis, top_n=5):
    """analyze() ciktisindaki en kritik node'lar icin aciklama listesi uretir."""
    betweenness_norm = _normalized_betweenness(graph)
    explanations = []
    for item in analysis["top_critical_nodes"][:top_n]:
        node = item["node"]
        explanations.append({
            "node": node,
            "score": item["score"],
            "explanation": explain_node(graph, node,
                                        betweenness_norm.get(node, 0.0)),
        })
    return explanations
