"""Vision2Graph - Faz 1 Core Pipeline.

Goruntu/maske -> yol maskesi -> skeleton -> graph -> overlay + metrikler.

Kullanim:
    python main.py data/train/100034_sat.jpg
    python main.py data/train/100034_mask.png --mask
"""
import argparse
import os
from pathlib import Path

import cv2
import yaml

from src.inference.infer_road_mask import (infer_from_image, load_mask_file,
                                           resolve_device)
from src.topology.graph_builder import basic_counts, build_graph
from src.topology.skeletonizer import mask_to_skeleton
from src.visualization.graph_overlay import draw_graph_overlay

ROOT = Path(__file__).resolve().parent


def load_config(path="config.yaml"):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(output_dir):
    for sub in ("masks", "skeletons", "graphs", "reports", "simulations"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def run(args):
    cfg = load_config()
    output_dir = ROOT / cfg["paths"]["output_dir"]
    ensure_dirs(output_dir)
    name = Path(args.input).stem

    if args.mask:
        print("[1/4] Hazir yol maskesi yukleniyor ...")
        mask = load_mask_file(args.input)
        base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        device = resolve_device(cfg["inference"]["device"])
        print(f"[1/4] D-LinkNet ile yol maskesi cikariliyor (device={device}) ...")
        mask, base = infer_from_image(
            ROOT / cfg["paths"]["model_checkpoint"], args.input,
            device, cfg["inference"]["threshold"])

    cv2.imwrite(str(output_dir / "masks" / f"{name}_mask.png"), mask)

    print("[2/4] Skeleton cikariliyor ...")
    skeleton = mask_to_skeleton(mask, cfg["skeleton"]["min_object_size"],
                                cfg["skeleton"]["closing_kernel"])
    cv2.imwrite(str(output_dir / "skeletons" / f"{name}_skeleton.png"), skeleton * 255)

    print("[3/4] Graph olusturuluyor ...")
    graph = build_graph(skeleton, cfg["graph"]["spur_length_threshold"],
                        cfg["graph"]["min_component_length"])

    print("[4/4] Graph overlay ciziliyor ...")
    overlay = draw_graph_overlay(base, graph)
    cv2.imwrite(str(output_dir / "graphs" / f"{name}_graph.png"), overlay)

    print("\n=== Graph Ozeti ===")
    for key, value in basic_counts(graph).items():
        print(f"  {key:18s}: {value}")
    print(f"\nCiktilar: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Vision2Graph - Core Pipeline")
    parser.add_argument("input", help="Uydu goruntusu veya yol maskesi yolu")
    parser.add_argument("--mask", action="store_true",
                        help="Girdi zaten binary yol maskesi (model atlanir)")
    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error(f"Girdi bulunamadi: {args.input}")
    run(args)


if __name__ == "__main__":
    main()
