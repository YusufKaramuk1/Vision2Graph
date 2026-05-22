"""Yol grafigini goruntu uzerine overlay olarak cizer."""
import cv2
import numpy as np


def draw_graph_overlay(base_image, graph, edge_color=(0, 255, 255),
                       node_color=(0, 0, 255), thickness=2, node_radius=4):
    canvas = (cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
              if base_image.ndim == 2 else base_image.copy())

    for u, v, data in graph.edges(data=True):
        pts = data.get("pts")
        if pts is not None and len(pts) >= 2:
            # pts (row, col) -> cv2 (x, y); kenar gercek yol cizgisini takip eder
            poly = np.asarray(pts)[:, ::-1].astype(np.int32)
            cv2.polylines(canvas, [poly], False, edge_color, thickness)
        else:
            p1 = (int(graph.nodes[u]["x"]), int(graph.nodes[u]["y"]))
            p2 = (int(graph.nodes[v]["x"]), int(graph.nodes[v]["y"]))
            cv2.line(canvas, p1, p2, edge_color, thickness)

    for _, data in graph.nodes(data=True):
        cv2.circle(canvas, (int(data["x"]), int(data["y"])), node_radius, node_color, -1)

    return canvas
