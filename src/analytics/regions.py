"""Agi mekansal alt bolgelere ayirip her bolgenin dayanikliligini olcer.

Genel resilience skoru bir ortalamadir; bu modul skorun arkasindaki bolgesel
zayifliklari ortaya cikarir. Ag 2x2 mekansal izgaraya bolunur, her bolgenin
kendi ic yol agi ayri puanlanir ve en kirilgan bolge isaretlenir - bu bolge
iyilestirme onceligidir.
"""
from src.analytics.criticality import analyze
from src.analytics.resilience import resilience_score
from src.simulation.attack import simulate

_REGION_NAMES = {
    (0, 0): "Kuzeybati", (1, 0): "Kuzeydogu",
    (0, 1): "Guneybati", (1, 1): "Guneydogu",
}


def assign_regions(graph):
    """Her node'u 2x2 mekansal izgaradaki bolgesine atar (bolge -> node listesi)."""
    xs = [data["x"] for _, data in graph.nodes(data=True)]
    ys = [data["y"] for _, data in graph.nodes(data=True)]
    if not xs:
        return {}
    mid_x = (min(xs) + max(xs)) / 2
    mid_y = (min(ys) + max(ys)) / 2
    assignment = {}
    for node, data in graph.nodes(data=True):
        col = 1 if data["x"] >= mid_x else 0
        row = 1 if data["y"] >= mid_y else 0
        assignment.setdefault(_REGION_NAMES[(col, row)], []).append(node)
    return assignment


def analyze_regions(graph, config, min_nodes=3):
    """Her mekansal bolge icin resilience skoru hesaplar, en zayifini isaretler.

    Bolge alt grafigi, sadece iki ucu da o bolgede olan kenarlari icerir;
    yani bolgenin kendi basina (izole) yol agini temsil eder.
    """
    regions = []
    for name, nodes in assign_regions(graph).items():
        if len(nodes) < min_nodes:
            regions.append({"region": name, "node_count": len(nodes),
                            "score": None, "grade": "-", "label": "-",
                            "nodes": list(nodes), "top_critical": []})
            continue
        sub = graph.subgraph(nodes).copy()
        analysis = analyze(sub, config)
        resilience = resilience_score(sub, analysis, simulate(sub, config), config)
        regions.append({
            "region": name,
            "node_count": len(nodes),
            "score": resilience["score"],
            "grade": resilience["grade"],
            "label": resilience["label"],
            "nodes": list(nodes),
            "top_critical": analysis["top_critical_nodes"][:3],
        })

    scored = [r for r in regions if r["score"] is not None]
    weakest = min(scored, key=lambda r: r["score"]) if scored else None
    return {"regions": regions, "weakest": weakest}
