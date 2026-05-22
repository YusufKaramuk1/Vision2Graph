"""A-B rotasini ve kapanma sonrasi alternatif rotayi goruntu uzerine cizer."""
import cv2
import numpy as np

_CONTEXT = (110, 110, 110)  # gri - ag baglami
_BEFORE = (255, 140, 40)    # mavi (BGR) - kapanma oncesi rota
_AFTER = (40, 150, 255)     # turuncu (BGR) - kapanma sonrasi rota
_SOURCE = (0, 200, 0)       # yesil - baslangic
_TARGET = (0, 0, 225)       # kirmizi - hedef


def _node_xy(graph, node):
    data = graph.nodes[node]
    return int(data["x"]), int(data["y"])


def _draw_path(canvas, graph, path, color, thickness):
    """Bir rotayi (node listesi) olusturan kenarlari cizer."""
    if not path or len(path) < 2:
        return
    for u, v in zip(path, path[1:]):
        data = graph.get_edge_data(u, v) or {}
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            poly = np.asarray(pts)[:, ::-1].astype(np.int32)
            cv2.polylines(canvas, [poly], False, color, thickness)
        else:
            cv2.line(canvas, _node_xy(graph, u), _node_xy(graph, v),
                     color, thickness)


def draw_route_overlay(base_image, graph, before_path, after_path, source, target):
    """Kapanma oncesi/sonrasi rotalari ve A-B uclarini overlay olarak cizer."""
    canvas = (cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
              if base_image.ndim == 2 else base_image.copy())

    for u, v, data in graph.edges(data=True):
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            poly = np.asarray(pts)[:, ::-1].astype(np.int32)
            cv2.polylines(canvas, [poly], False, _CONTEXT, 1)
        else:
            cv2.line(canvas, _node_xy(graph, u), _node_xy(graph, v), _CONTEXT, 1)

    _draw_path(canvas, graph, before_path, _BEFORE, 4)
    _draw_path(canvas, graph, after_path, _AFTER, 3)

    for node, color in ((source, _SOURCE), (target, _TARGET)):
        if graph.has_node(node):
            center = _node_xy(graph, node)
            cv2.circle(canvas, center, 9, color, -1)
            cv2.circle(canvas, center, 9, (255, 255, 255), 2)

    return canvas
