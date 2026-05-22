"""Afet senaryolari: yol agi uzerinde mekansal toplu kapanma kumeleri uretir.

Her senaryo, kapatilacak kavsaklarin listesini dondurur; etki olcumu closure
etki motoru (closure_impact) tarafindan yapilir. Gercek afet verisi olmadigi
icin bunlar mekansal varsayimlara dayali simulasyonlardir.
"""
import math

from src.analytics.regions import assign_regions


def earthquake_zone(graph, epicenter, radius):
    """Episantr kavsagina Euclidean uzakligi radius'u asmayan tum kavsaklar."""
    center = graph.nodes[epicenter]
    return [node for node, data in graph.nodes(data=True)
            if math.hypot(data["x"] - center["x"],
                          data["y"] - center["y"]) <= radius]


def flood_zone(graph, level_fraction):
    """Goruntunun alt level_fraction kismindaki kavsaklar (su basan alcak bolge).

    level_fraction 0.35 -> en alttaki %35'lik seritteki yollar su altinda kalir.
    Goruntu koordinatlarinda buyuk y degeri 'alt' (guney) tarafa karsilik gelir.
    """
    ys = [data["y"] for _, data in graph.nodes(data=True)]
    if not ys:
        return []
    low, high = min(ys), max(ys)
    threshold = high - (high - low) * level_fraction
    return [node for node, data in graph.nodes(data=True)
            if data["y"] >= threshold]


def regional_zone(graph, region_name):
    """Secili 2x2 mekansal bolgedeki tum kavsaklar (bolgesel kapanma)."""
    return assign_regions(graph).get(region_name, [])
