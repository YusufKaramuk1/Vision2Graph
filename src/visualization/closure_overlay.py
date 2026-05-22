"""Bir kapanma senaryosunu goruntu uzerine cizer.

Kapatilan kavsaklar siyah X ile, kapanma yuzunden ana agdan kopan yol ve
kavsaklar kirmizi ile, ayakta kalan ag soluk gri ile gosterilir.
"""
import cv2
import numpy as np

_CLOSED = (255, 0, 255)      # parlak magenta - siyah maskede de gorunur
_ISOLATED = (60, 60, 220)    # kirmizi (BGR) - ana agdan kopan kisim
_SURVIVING = (150, 150, 150) # gri - ayakta kalan ag


def _node_xy(graph, node):
    data = graph.nodes[node]
    return int(data["x"]), int(data["y"])


def draw_closure_overlay(base_image, graph, closed_nodes, isolated_nodes):
    """Kapanma senaryosunu orijinal graph uzerine cizip overlay dondurur."""
    canvas = (cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
              if base_image.ndim == 2 else base_image.copy())
    closed = set(closed_nodes or [])
    isolated = set(isolated_nodes or [])

    for u, v, data in graph.edges(data=True):
        if u in closed or v in closed:
            color, thickness = _CLOSED, 1
        elif u in isolated and v in isolated:
            color, thickness = _ISOLATED, 3
        else:
            color, thickness = _SURVIVING, 2
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            poly = np.asarray(pts)[:, ::-1].astype(np.int32)
            cv2.polylines(canvas, [poly], False, color, thickness)
        else:
            cv2.line(canvas, _node_xy(graph, u), _node_xy(graph, v),
                     color, thickness)

    for node in graph.nodes():
        center = _node_xy(graph, node)
        if node in closed:
            radius = 10
            cv2.line(canvas, (center[0] - radius, center[1] - radius),
                     (center[0] + radius, center[1] + radius), _CLOSED, 4)
            cv2.line(canvas, (center[0] - radius, center[1] + radius),
                     (center[0] + radius, center[1] - radius), _CLOSED, 4)
        elif node in isolated:
            cv2.circle(canvas, center, 5, _ISOLATED, -1)
        else:
            cv2.circle(canvas, center, 3, _SURVIVING, -1)

    return canvas
