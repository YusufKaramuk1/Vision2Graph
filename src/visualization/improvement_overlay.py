"""Onerilen yeni yol baglantisini mevcut graph uzerine cizer."""
import cv2
import numpy as np

_EXISTING = (150, 150, 150)  # gri - mevcut ag
_SUGGESTED = (0, 210, 0)     # parlak yesil - onerilen yeni baglanti


def _node_xy(graph, node):
    data = graph.nodes[node]
    return int(data["x"]), int(data["y"])


def draw_improvement_overlay(base_image, graph, suggested_edge):
    """Mevcut agi soluk, onerilen yeni baglantiyi yesil vurguyla cizer."""
    canvas = (cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
              if base_image.ndim == 2 else base_image.copy())

    for u, v, data in graph.edges(data=True):
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            poly = np.asarray(pts)[:, ::-1].astype(np.int32)
            cv2.polylines(canvas, [poly], False, _EXISTING, 2)
        else:
            cv2.line(canvas, _node_xy(graph, u), _node_xy(graph, v), _EXISTING, 2)

    for node in graph.nodes():
        cv2.circle(canvas, _node_xy(graph, node), 3, _EXISTING, -1)

    if suggested_edge:
        u, v = suggested_edge
        p1, p2 = _node_xy(graph, u), _node_xy(graph, v)
        cv2.line(canvas, p1, p2, _SUGGESTED, 3)
        for point in (p1, p2):
            cv2.circle(canvas, point, 7, _SUGGESTED, -1)
            cv2.circle(canvas, point, 7, (255, 255, 255), 2)

    return canvas
