"""Mekansal bolgeleri ve en kirilgan bolgeyi goruntu uzerine cizer."""
import cv2
import numpy as np

_FAINT = (110, 110, 110)   # gri - ag baglami
_GRID = (90, 90, 90)       # bolge ayirma cizgileri
_WEAK = (60, 60, 220)      # kirmizi - en kirilgan bolgenin kavsaklari
_OTHER = (160, 160, 160)   # gri - diger bolgelerin kavsaklari


def _node_xy(graph, node):
    data = graph.nodes[node]
    return int(data["x"]), int(data["y"])


def draw_regions_overlay(base_image, graph, weakest_nodes):
    """2x2 bolge izgarasini cizer, en kirilgan bolgenin kavsaklarini kirmizilar."""
    canvas = (cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
              if base_image.ndim == 2 else base_image.copy())

    for u, v, data in graph.edges(data=True):
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            poly = np.asarray(pts)[:, ::-1].astype(np.int32)
            cv2.polylines(canvas, [poly], False, _FAINT, 1)
        else:
            cv2.line(canvas, _node_xy(graph, u), _node_xy(graph, v), _FAINT, 1)

    xs = [data["x"] for _, data in graph.nodes(data=True)]
    ys = [data["y"] for _, data in graph.nodes(data=True)]
    if xs:
        height, width = canvas.shape[:2]
        mid_x = int((min(xs) + max(xs)) / 2)
        mid_y = int((min(ys) + max(ys)) / 2)
        cv2.line(canvas, (mid_x, 0), (mid_x, height), _GRID, 1)
        cv2.line(canvas, (0, mid_y), (width, mid_y), _GRID, 1)

    weak = set(weakest_nodes or [])
    for node in graph.nodes():
        center = _node_xy(graph, node)
        if node in weak:
            cv2.circle(canvas, center, 6, _WEAK, -1)
        else:
            cv2.circle(canvas, center, 3, _OTHER, -1)

    return canvas
