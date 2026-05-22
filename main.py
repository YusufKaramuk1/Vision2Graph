"""Vision2Graph - Faz 1 Core Pipeline.

Goruntu/maske -> yol maskesi -> skeleton -> graph -> overlay + metrikler.

Kullanim:
    python main.py data/train/100034_sat.jpg
    python main.py data/train/100034_mask.png --mask
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import yaml

from src.analytics.criticality import analyze
from src.inference.infer_road_mask import (infer_from_image, load_mask_file,
                                           resolve_device)
from src.topology.graph_builder import basic_counts, build_graph
from src.topology.skeletonizer import mask_to_skeleton
from src.visualization.criticality_overlay import draw_criticality_overlay
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
        print("[1/5] Hazir yol maskesi yukleniyor ...")
        mask = load_mask_file(args.input)
        base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        device = resolve_device(cfg["inference"]["device"])
        print(f"[1/5] D-LinkNet ile yol maskesi cikariliyor (device={device}) ...")
        mask, base = infer_from_image(
            ROOT / cfg["paths"]["model_checkpoint"], args.input,
            device, cfg["inference"]["threshold"])

    cv2.imwrite(str(output_dir / "masks" / f"{name}_mask.png"), mask)

    print("[2/5] Skeleton cikariliyor ...")
    skeleton = mask_to_skeleton(mask, cfg["skeleton"]["min_object_size"],
                                cfg["skeleton"]["closing_kernel"])
    cv2.imwrite(str(output_dir / "skeletons" / f"{name}_skeleton.png"), skeleton * 255)

    print("[3/5] Graph olusturuluyor ...")
    graph = build_graph(skeleton, cfg["graph"]["spur_length_threshold"],
                        cfg["graph"]["min_component_length"])

    print("[4/5] Graph overlay ciziliyor ...")
    overlay = draw_graph_overlay(base, graph)
    cv2.imwrite(str(output_dir / "graphs" / f"{name}_graph.png"), overlay)

    print("[5/5] Graph analizi (kritiklik + worst-case) ...")
    analysis = analyze(graph, cfg)
    crit_overlay = draw_criticality_overlay(base, graph)
    cv2.imwrite(str(output_dir / "graphs" / f"{name}_criticality.png"), crit_overlay)
    report_path = output_dir / "reports" / f"{name}_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print("\n=== Graph Ozeti ===")
    for key, value in basic_counts(graph).items():
        print(f"  {key:18s}: {value}")

    print_analysis(analysis)
    print(f"\nCiktilar: {output_dir}")


def print_analysis(analysis):
    """Faz 2 analiz sonucunu konsola ozetler."""
    print("\n=== Graph Analizi (Faz 2) ===")
    print(f"  articulation_points : {analysis['articulation_count']}")
    print(f"  bridge_edges        : {analysis['bridge_count']}")

    print("  En kritik node'lar:")
    for item in analysis["top_critical_nodes"]:
        print(f"    node {item['node']:>4}   skor={item['score']}")

    worst = analysis["worst_case"]
    print(f"  Worst-case (baseline LCR={worst['baseline_lcr']}, "
          f"bilesen={worst['baseline_components']}):")
    for impact in worst["impacts"][:3]:
        print(f"    node {impact['node']:>4}  LCR dususu={impact['lcr_drop']}  "
              f"verim kaybi=%{impact['efficiency_drop_pct']}  "
              f"bilesen={impact['components_after']}")


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
