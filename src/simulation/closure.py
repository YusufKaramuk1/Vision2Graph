"""Kapanma etki motoru (closure impact engine).

Bir yol kapanma senaryosunu (node/kenar kumesi) uygular ve zengin bir etki
nesnesi uretir. Bu nesne backlog'daki cesitli ozelliklerin ortak cekirdegidir:
what-if simulasyonu, once/sonra paneli, izolasyon maliyeti ve kritiklik
aciklamasi hep ayni closure_impact() ciktisini tuketir.

Faz 2'deki worst-case analizi tek node'un otomatik secimine bakar; buradaki
closure ise kullanicinin/senaryonun verdigi kumeyi kapatir.
"""
import networkx as nx

from src.analytics.worst_case import global_efficiency, largest_component_ratio


def apply_closure(graph, closed_nodes=None, closed_edges=None):
    """Verilen node ve kenarlari kapatilmis yeni bir graph dondurur (girdi degismez)."""
    damaged = graph.copy()
    if closed_edges:
        damaged.remove_edges_from(closed_edges)
    if closed_nodes:
        damaged.remove_nodes_from(closed_nodes)
    return damaged


def _total_length(graph):
    """Agdaki tum kenarlarin toplam yol uzunlugu."""
    return sum(data.get("length", 0.0) for _, _, data in graph.edges(data=True))


def _largest_component(graph):
    """Agin en buyuk bagli bileseni (node kumesi) - 'ana ag'."""
    components = list(nx.connected_components(graph))
    return max(components, key=len) if components else set()


def network_state(graph):
    """Agin anlik saglik gostergeleri (kademeli saldiri egrisi de bunu kullanir)."""
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "lcr": round(largest_component_ratio(graph), 4),
        "efficiency": round(global_efficiency(graph), 6),
    }


def _rich_state(graph):
    """network_state + toplam yol uzunlugu (once/sonra paneli icin)."""
    state = network_state(graph)
    state["total_length"] = round(_total_length(graph), 1)
    return state


def closure_impact(graph, closed_nodes=None, closed_edges=None):
    """Bir kapanma senaryosunun zengin once/sonra etki nesnesi.

    isolated_* alanlari, bu kapanma yuzunden ana agdan (en buyuk bilesen) YENI
    kopan kismi olcer: kapanma oncesi ana agda olan, kapanmadan sag cikan ama
    kapanma sonrasi artik ana agda olmayan node'lar. Boylece sayilar dogrudan
    bu senaryonun sorumlulugunu yansitir, zaten kopuk olan kisim disarida kalir.
    """
    closed_set = set(closed_nodes or [])
    damaged = apply_closure(graph, closed_nodes, closed_edges)

    before = _rich_state(graph)
    after = _rich_state(damaged)
    eff_loss = (100 * (before["efficiency"] - after["efficiency"])
                / before["efficiency"] if before["efficiency"] > 0 else 0.0)

    # Ana agi kimligiyle takip et: kapanma sonrasi "ana ag", eski ana agin
    # node'lariyla en cok ortusen bilesendir. Esit boyutlu bilesenlerde "en
    # buyuk" secimi kararsiz oldugu icin boyuta degil ortusmeye bakilir.
    main_before = _largest_component(graph)
    survivors = main_before - closed_set
    main_after = max(nx.connected_components(damaged),
                     key=lambda comp: len(comp & survivors), default=set())
    newly_isolated = survivors - main_after
    isolated_length = sum(data.get("length", 0.0)
                          for u, v, data in damaged.edges(data=True)
                          if u in newly_isolated and v in newly_isolated)
    isolation_ratio = (100 * isolated_length / before["total_length"]
                       if before["total_length"] > 0 else 0.0)

    return {
        "closed_nodes": sorted(int(n) for n in closed_set),
        "closed_edges": [[int(u), int(v)] for u, v in (closed_edges or [])],
        "before": before,
        "after": after,
        "lcr_drop": round(before["lcr"] - after["lcr"], 4),
        "component_increase": after["components"] - before["components"],
        "efficiency_loss_pct": round(eff_loss, 2),
        "isolated_nodes": len(newly_isolated),
        "isolated_length": round(isolated_length, 1),
        "isolation_ratio_pct": round(isolation_ratio, 2),
    }
