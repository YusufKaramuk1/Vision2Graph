"""Kademeli saldiri sirasinda agin parcalanisini GIF animasyonu olarak uretir."""
import io

import cv2
from PIL import Image

from src.visualization.graph_overlay import draw_graph_overlay


def _frame(base_image, graph, step, width):
    """Tek bir adimin graph overlay'ini olceklenmis PIL Image olarak uretir."""
    overlay = draw_graph_overlay(base_image, graph)
    cv2.putText(overlay, f"Adim {step} - {graph.number_of_nodes()} kavsak",
                (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    height = int(overlay.shape[0] * width / overlay.shape[1])
    resized = cv2.resize(overlay, (width, height))
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def build_collapse_gif(base_image, graph, removal_order, max_steps=None,
                       frame_width=512, duration_ms=650):
    """Node'lar sirayla kaldirilirken her adimin overlay'ini GIF'e diziler.

    removal_order: kaldirilma sirasi (orn. kritiklik azalan). Doner: GIF baytlari.
    """
    steps = removal_order if max_steps is None else removal_order[:max_steps]
    working = graph.copy()
    frames = [_frame(base_image, working, 0, frame_width)]
    for index, node in enumerate(steps, start=1):
        if working.has_node(node):
            working.remove_node(node)
        frames.append(_frame(base_image, working, index, frame_width))

    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True,
                   append_images=frames[1:], duration=duration_ms, loop=0)
    return buffer.getvalue()
