"""Belirli bir yol kapanma senaryosunu uygular ve etkisini olcer.

Faz 2'deki worst-case analizi tek node'un otomatik secimine bakar; buradaki
closure ise kullanicinin/senaryonun verdigi node ve kenar kumesini kapatip
once/sonra karsilastirmasi uretir (orn. "su 3 kavsak sel altinda kalirsa").
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


def network_state(graph):
    """Agin anlik saglik gostergeleri."""
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "lcr": round(largest_component_ratio(graph), 4),
        "efficiency": round(global_efficiency(graph), 6),
    }


def closure_impact(graph, closed_nodes=None, closed_edges=None):
    """Bir kapanma senaryosunun once/sonra karsilastirmasi."""
    before = network_state(graph)
    after = network_state(apply_closure(graph, closed_nodes, closed_edges))
    eff_loss = (100 * (before["efficiency"] - after["efficiency"]) / before["efficiency"]
                if before["efficiency"] > 0 else 0.0)
    return {
        "closed_nodes": sorted(int(n) for n in (closed_nodes or [])),
        "closed_edges": [[int(u), int(v)] for u, v in (closed_edges or [])],
        "before": before,
        "after": after,
        "lcr_drop": round(before["lcr"] - after["lcr"], 4),
        "efficiency_loss_pct": round(eff_loss, 2),
    }
