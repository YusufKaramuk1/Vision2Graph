"""Worst-case analizi: en kritik node'lar kapatildiginda agin nasil cokegidigi.

Iki etki olcusu kullanilir:
  - LCR (largest component ratio): en buyuk bagli bilesendeki node oraninin
    dususu. Yuksek dusus = ag parcalandi, sehir bolundu.
  - Global verimlilik: node ciftleri arasi ortalama 1/mesafe. Dususu, hayatta
    kalan rotalarin ne kadar uzadigini gosterir.
"""
import networkx as nx


def largest_component_ratio(graph):
    """En buyuk bagli bilesendeki node sayisinin toplam node'a orani."""
    if graph.number_of_nodes() == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return largest / graph.number_of_nodes()


def global_efficiency(graph):
    """Agirlikli global verimlilik: tum node ciftleri arasi ortalama 1/mesafe.

    Ulasilamayan ciftler 0 katki verir; bu yuzden kopuk ag dusuk skor alir.
    """
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    total = 0.0
    for source, distances in nx.all_pairs_dijkstra_path_length(graph, weight="length"):
        for target, dist in distances.items():
            if source != target and dist > 0:
                total += 1.0 / dist
    return total / (n * (n - 1))


def baseline_metrics(graph):
    """Hicbir node kaldirilmadan once agin saglik gostergeleri."""
    return {
        "lcr": largest_component_ratio(graph),
        "efficiency": global_efficiency(graph),
        "components": nx.number_connected_components(graph),
    }


def worst_case_nodes(graph, criticality, top_k=5):
    """En kritik top_k node'u tek tek kaldirir, her birinin etkisini olcer.

    Sonuc, gercek (skor degil) etki olan LCR dususune gore siralanir: kritiklik
    skoru bir tahmindir, bu liste tahminin dogrulanmis halidir.
    """
    base = baseline_metrics(graph)
    ranked = sorted(criticality, key=criticality.get, reverse=True)[:top_k]

    impacts = []
    for node in ranked:
        damaged = graph.copy()
        damaged.remove_node(node)
        lcr = largest_component_ratio(damaged)
        efficiency = global_efficiency(damaged)
        eff_drop = (100 * (base["efficiency"] - efficiency) / base["efficiency"]
                    if base["efficiency"] > 0 else 0.0)
        impacts.append({
            "node": int(node),
            "criticality": round(float(criticality[node]), 4),
            "lcr_after": round(lcr, 4),
            "lcr_drop": round(base["lcr"] - lcr, 4),
            "efficiency_after": round(efficiency, 6),
            "efficiency_drop_pct": round(eff_drop, 2),
            "components_after": nx.number_connected_components(damaged),
        })

    # LCR dususu birincil etki olcusu; parcali aglarda cogu node ayni LCR
    # dususunu verir, bu yuzden verim kaybi ikincil ayirt edici olarak kullanilir.
    impacts.sort(key=lambda item: (item["lcr_drop"], item["efficiency_drop_pct"]),
                 reverse=True)
    return {
        "baseline_lcr": round(base["lcr"], 4),
        "baseline_efficiency": round(base["efficiency"], 6),
        "baseline_components": base["components"],
        "impacts": impacts,
    }
