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
from src.analytics.explainability import explain_critical_nodes
from src.analytics.resilience import resilience_score
from src.analytics.scale import format_length, meters_per_pixel
from src.inference.infer_road_mask import (infer_from_image, load_mask_file,
                                           resolve_device)
from src.reporting.pdf_report import build_pdf_report
from src.simulation.attack import simulate
from src.topology.graph_builder import basic_counts, build_graph
from src.topology.skeletonizer import mask_to_skeleton
from src.visualization.criticality_overlay import draw_criticality_overlay
from src.visualization.degradation_plot import draw_degradation_plot
from src.visualization.graph_overlay import draw_graph_overlay
from src.visualization.resilience_card import draw_resilience_card

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
        print("[1/8] Hazir yol maskesi yukleniyor ...")
        mask = load_mask_file(args.input)
        base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        device = resolve_device(cfg["inference"]["device"])
        print(f"[1/8] D-LinkNet ile yol maskesi cikariliyor (device={device}) ...")
        mask, base = infer_from_image(
            ROOT / cfg["paths"]["model_checkpoint"], args.input,
            device, cfg["inference"]["threshold"])

    cv2.imwrite(str(output_dir / "masks" / f"{name}_mask.png"), mask)

    print("[2/8] Skeleton cikariliyor ...")
    skeleton = mask_to_skeleton(mask, cfg["skeleton"]["min_object_size"],
                                cfg["skeleton"]["closing_kernel"])
    cv2.imwrite(str(output_dir / "skeletons" / f"{name}_skeleton.png"), skeleton * 255)

    print("[3/8] Graph olusturuluyor ...")
    graph = build_graph(skeleton, cfg["graph"]["spur_length_threshold"],
                        cfg["graph"]["min_component_length"])

    print("[4/8] Graph overlay ciziliyor ...")
    overlay = draw_graph_overlay(base, graph)
    cv2.imwrite(str(output_dir / "graphs" / f"{name}_graph.png"), overlay)

    print("[5/8] Graph analizi (kritiklik + worst-case) ...")
    analysis = analyze(graph, cfg)
    analysis["explanations"] = explain_critical_nodes(graph, analysis)
    crit_overlay = draw_criticality_overlay(base, graph)
    cv2.imwrite(str(output_dir / "graphs" / f"{name}_criticality.png"), crit_overlay)
    with open(output_dir / "reports" / f"{name}_analysis.json", "w",
              encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print("[6/8] Kapanma simulasyonu (kasitli vs rastsal saldiri) ...")
    simulation = simulate(graph, cfg)
    draw_degradation_plot(
        simulation, str(output_dir / "simulations" / f"{name}_degradation.png"))
    with open(output_dir / "simulations" / f"{name}_simulation.json", "w",
              encoding="utf-8") as f:
        json.dump(simulation, f, indent=2, ensure_ascii=False)

    print("[7/8] Resilience skoru hesaplaniyor ...")
    resilience = resilience_score(graph, analysis, simulation, cfg)
    draw_resilience_card(
        resilience, str(output_dir / "reports" / f"{name}_resilience.png"))
    with open(output_dir / "reports" / f"{name}_resilience.json", "w",
              encoding="utf-8") as f:
        json.dump(resilience, f, indent=2, ensure_ascii=False)

    counts = basic_counts(graph)
    mpp = meters_per_pixel(cfg)
    print("[8/8] PDF rapor olusturuluyor ...")
    report_path = build_pdf_report(output_dir, name, counts, analysis,
                                   simulation, resilience, mpp)

    print("\n=== Graph Ozeti ===")
    for key, value in counts.items():
        shown = format_length(value, mpp) if key == "total_length" else value
        print(f"  {key:18s}: {shown}")

    print_analysis(analysis)
    print_simulation(simulation)
    print_resilience(resilience)
    print(f"\nPDF rapor: {report_path}")
    print(f"Ciktilar: {output_dir}")


def print_analysis(analysis):
    """Faz 2 analiz sonucunu konsola ozetler."""
    print("\n=== Graph Analizi (Faz 2) ===")
    print(f"  articulation_points : {analysis['articulation_count']}")
    print(f"  bridge_edges        : {analysis['bridge_count']}")

    print("  En kritik node'lar:")
    for item in analysis["top_critical_nodes"]:
        print(f"    node {item['node']:>4}   skor={item['score']}")

    if analysis.get("explanations"):
        print("  Neden kritik:")
        for item in analysis["explanations"]:
            print(f"    - {item['explanation']}")

    worst = analysis["worst_case"]
    print(f"  Worst-case (baseline LCR={worst['baseline_lcr']}, "
          f"bilesen={worst['baseline_components']}):")
    for impact in worst["impacts"][:3]:
        print(f"    node {impact['node']:>4}  LCR dususu={impact['lcr_drop']}  "
              f"verim kaybi=%{impact['efficiency_drop_pct']}  "
              f"bilesen={impact['components_after']}")


def print_simulation(simulation):
    """Faz 3 kapanma simulasyonu sonucunu konsola ozetler."""
    print("\n=== Kapanma Simulasyonu (Faz 3) ===")
    print(f"  Dayaniklilik indeksi  -> kasitli : {simulation['targeted']['robustness_index']}")
    print(f"                           rastsal : {simulation['random']['robustness_index']}")
    print(f"  Kirilganlik farki (fragility_gap): {simulation['fragility_gap']}")


def print_resilience(resilience):
    """Faz 4 resilience skorunu konsola ozetler."""
    print("\n=== Resilience Skoru (Faz 4) ===")
    print(f"  SKOR: {resilience['score']} / 100   "
          f"[{resilience['grade']}] {resilience['label']}")
    print("  Bilesenler (0-1, yuksek = dayanikli):")
    for key, value in resilience["components"].items():
        print(f"    {key:24s}: {value}")


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
