"""Kritiklik skorlarini isi haritasi (heatmap) renkleriyle goruntu uzerine cizer.

Renk: yesil = dusuk kritiklik, sari = orta, kirmizi = yuksek. Kopru kenarlar
kalin, kesim noktalari (articulation) beyaz halka ile vurgulanir.
"""
import cv2
import numpy as np


def _heat_color(value):
    """0-1 skoru BGR renge cevirir (yesil -> sari -> kirmizi)."""
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        red = int(255 * value / 0.5)
        green = 255
    else:
        red = 255
        green = int(255 * (1.0 - (value - 0.5) / 0.5))
    return (0, green, red)


def draw_criticality_overlay(base_image, graph, thickness=2, node_radius=5):
    canvas = (cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
              if base_image.ndim == 2 else base_image.copy())

    for u, v, data in graph.edges(data=True):
        color = _heat_color(data.get("criticality", 0.0))
        width = thickness + (2 if data.get("is_bridge") else 0)
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            poly = np.asarray(pts)[:, ::-1].astype(np.int32)
            cv2.polylines(canvas, [poly], False, color, width)
        else:
            p1 = (int(graph.nodes[u]["x"]), int(graph.nodes[u]["y"]))
            p2 = (int(graph.nodes[v]["x"]), int(graph.nodes[v]["y"]))
            cv2.line(canvas, p1, p2, color, width)

    for _, data in graph.nodes(data=True):
        center = (int(data["x"]), int(data["y"]))
        cv2.circle(canvas, center, node_radius,
                   _heat_color(data.get("criticality", 0.0)), -1)
        if data.get("is_articulation"):
            cv2.circle(canvas, center, node_radius + 3, (255, 255, 255), 1)

    return canvas
