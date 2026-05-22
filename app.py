"""Vision2Graph - Streamlit demo arayuzu.

Pipeline'i (maske -> skeleton -> graph -> analiz -> simulasyon -> resilience)
saran interaktif demo. Calistirmak icin:

    streamlit run app.py
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml

from src.analytics.criticality import analyze
from src.analytics.resilience import resilience_score
from src.inference.infer_road_mask import (load_model, predict_mask,
                                           resolve_device)
from src.simulation.attack import simulate
from src.simulation.closure import apply_closure, closure_impact
from src.topology.graph_builder import basic_counts, build_graph
from src.topology.skeletonizer import mask_to_skeleton
from src.visualization.closure_overlay import draw_closure_overlay
from src.visualization.criticality_overlay import draw_criticality_overlay
from src.visualization.degradation_plot import draw_degradation_plot
from src.visualization.graph_overlay import draw_graph_overlay
from src.visualization.resilience_card import draw_resilience_card

ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_model(checkpoint_path, device):
    """D-LinkNet modelini bir kez yukler; tekrar calistirmalarda yeniden kullanir."""
    return load_model(checkpoint_path, device)


@st.cache_data
def list_samples(data_dir):
    """data/train altindaki hazir ornek maske ve goruntu dosyalarini listeler."""
    folder = ROOT / data_dir
    if not folder.exists():
        return []
    files = sorted(folder.glob("*_mask.png")) + sorted(folder.glob("*_sat.jpg"))
    return [f.name for f in files]


def to_rgb(image):
    """cv2 BGR/gri goruntusunu Streamlit gosterimi icin RGB'ye cevirir."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_pipeline(file_bytes, is_mask, cfg):
    """Yuklenen dosyadan tum Vision2Graph cikti zincirini uretir."""
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    result = {}

    if is_mask:
        raw = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError("Maske cozulemedi - gecerli bir goruntu degil.")
        mask = (raw > 127).astype(np.uint8) * 255
        base = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Goruntu cozulemedi - gecerli bir goruntu degil.")
        device = resolve_device(cfg["inference"]["device"])
        model = get_model(str(ROOT / cfg["paths"]["model_checkpoint"]), device)
        mask = predict_mask(model, image, device, cfg["inference"]["threshold"])
        base = image

    result["source"] = base
    result["mask"] = mask

    skeleton = mask_to_skeleton(mask, cfg["skeleton"]["min_object_size"],
                                cfg["skeleton"]["closing_kernel"])
    result["skeleton"] = skeleton * 255

    graph = build_graph(skeleton, cfg["graph"]["spur_length_threshold"],
                        cfg["graph"]["min_component_length"])
    result["counts"] = basic_counts(graph)
    result["graph"] = graph  # what-if sekmesi grafik uzerinde calisir
    result["graph_overlay"] = draw_graph_overlay(base, graph)

    # analyze() kritiklik ozniteliklerini graph'a yazar; sonraki adimlar buna dayanir.
    result["analysis"] = analyze(graph, cfg)
    result["criticality_overlay"] = draw_criticality_overlay(base, graph)
    result["simulation"] = simulate(graph, cfg)
    result["resilience"] = resilience_score(graph, result["analysis"],
                                            result["simulation"], cfg)
    return result


def render_input_tab(result):
    col1, col2 = st.columns(2)
    col1.image(to_rgb(result["source"]), caption="Girdi", use_container_width=True)
    col2.image(to_rgb(result["mask"]), caption="Yol maskesi",
               use_container_width=True)


def render_graph_tab(result):
    col1, col2 = st.columns(2)
    col1.image(to_rgb(result["skeleton"]), caption="Skeleton",
               use_container_width=True)
    col2.image(to_rgb(result["graph_overlay"]), caption="Yol grafigi overlay",
               use_container_width=True)
    counts = result["counts"]
    for col, (key, value) in zip(st.columns(len(counts)), counts.items()):
        col.metric(key, value)


def render_criticality_tab(result):
    analysis = result["analysis"]
    st.image(to_rgb(result["criticality_overlay"]),
             caption="Kritiklik isi haritasi - kirmizi = yuksek, beyaz halka = kesim noktasi",
             use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Articulation points", analysis["articulation_count"])
    col2.metric("Bridge edges", analysis["bridge_count"])
    st.subheader("En kritik node'lar")
    st.dataframe(analysis["top_critical_nodes"], use_container_width=True)
    st.subheader("Worst-case: en kritik node'lar kapaninca")
    st.dataframe(analysis["worst_case"]["impacts"], use_container_width=True)


def render_simulation_tab(result):
    simulation = result["simulation"]
    fig = draw_degradation_plot(simulation)
    st.pyplot(fig)
    plt.close(fig)
    col1, col2, col3 = st.columns(3)
    col1.metric("Kasitli saldiri R", simulation["targeted"]["robustness_index"])
    col2.metric("Rastsal ariza R", simulation["random"]["robustness_index"])
    col3.metric("Kirilganlik farki", simulation["fragility_gap"])
    st.caption("R = LCR egrisi altindaki alan (0-1). Dusuk kasitli-R, agin "
               "oncelikli kavsak kayiplarina kirilgan oldugunu gosterir.")


def render_resilience_tab(result):
    resilience = result["resilience"]
    col1, col2 = st.columns([1, 2])
    col1.metric("Resilience skoru", f"{resilience['score']} / 100")
    col1.subheader(f"{resilience['grade']} - {resilience['label']}")
    fig = draw_resilience_card(resilience)
    col2.pyplot(fig)
    plt.close(fig)
    st.caption("Bilesen agirliklari (config.yaml): " + ", ".join(
        f"{key}={value}" for key, value in resilience["weights"].items()))


def render_whatif_tab(result, cfg):
    graph = result["graph"]
    st.write("Kapatilacak kavsak/yollari secin; sistem agi yeniden analiz eder "
             "ve kapanmanin etkisini gosterir.")

    col1, col2 = st.columns(2)
    closed_nodes = col1.multiselect("Kapatilacak kavsaklar (node)",
                                    sorted(graph.nodes()))
    edge_options = sorted({tuple(sorted(edge)) for edge in graph.edges()})
    closed_edges = col2.multiselect(
        "Kapatilacak yollar (kenar)", edge_options,
        format_func=lambda edge: f"{edge[0]} - {edge[1]}")

    if not closed_nodes and not closed_edges:
        st.info("Etki gormek icin en az bir kavsak veya yol secin.")
        return

    impact = closure_impact(graph, closed_nodes, closed_edges)
    before, after = impact["before"], impact["after"]

    st.subheader("Etki ozeti")
    cols = st.columns(4)
    cols[0].metric("Bilesen artisi", f"+{impact['component_increase']}")
    cols[1].metric("Verim kaybi", f"%{impact['efficiency_loss_pct']}")
    cols[2].metric("Izole kavsak", impact["isolated_nodes"])
    cols[3].metric("Kopan ag orani", f"%{impact['isolation_ratio_pct']}")
    if impact["isolated_nodes"]:
        st.warning(
            f"Bu kapanma {impact['isolated_nodes']} kavsagi ve "
            f"{impact['isolated_length']} px yolu (yol agininin "
            f"%{impact['isolation_ratio_pct']}'ini) ana agdan koparir.")

    st.subheader("Once / Sonra")
    before_col, after_col = st.columns(2)
    before_col.caption("KAPANMA ONCESI")
    before_col.metric("Bilesen sayisi", before["components"])
    before_col.metric("En buyuk bilesen orani", before["lcr"])
    before_col.metric("Toplam yol (px)", before["total_length"])
    after_col.caption("KAPANMA SONRASI")
    after_col.metric("Bilesen sayisi", after["components"],
                     delta=after["components"] - before["components"],
                     delta_color="inverse")
    after_col.metric("En buyuk bilesen orani", after["lcr"],
                     delta=round(after["lcr"] - before["lcr"], 4))
    after_col.metric("Toplam yol (px)", after["total_length"],
                     delta=round(after["total_length"] - before["total_length"], 1))

    st.subheader("Resilience skoru")
    if st.checkbox("Kapanma sonrasi resilience skorunu hesapla", value=True):
        with st.spinner("Ag yeniden degerlendiriliyor ..."):
            damaged = apply_closure(graph, closed_nodes, closed_edges)
            new_resilience = resilience_score(damaged, analyze(damaged, cfg),
                                              simulate(damaged, cfg), cfg)
        original = result["resilience"]
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Kapanma oncesi", f"{original['score']} / 100")
        res_col2.metric("Kapanma sonrasi", f"{new_resilience['score']} / 100",
                        delta=round(new_resilience["score"] - original["score"], 1))

    st.subheader("Gorsel")
    overlay = draw_closure_overlay(result["source"], graph,
                                   closed_nodes, impact["isolated_node_list"])
    st.image(to_rgb(overlay),
             caption="Siyah X: kapatilan kavsak  |  kirmizi: ana agdan kopan "
                     "yol/kavsak  |  gri: ayakta kalan ag",
             use_container_width=True)


def main():
    st.set_page_config(page_title="Vision2Graph", layout="wide")
    cfg = load_config()

    st.title("Vision2Graph")
    st.caption("Uydu goruntusunden yol agi cikarimi ve dayaniklilik analizi")

    with st.sidebar:
        st.header("Girdi")
        source = st.radio("Girdi kaynagi", ["Ornek veri", "Dosya yukle"])
        file_bytes, is_mask = None, False

        if source == "Ornek veri":
            samples = list_samples(cfg["paths"]["data_dir"])
            if samples:
                choice = st.selectbox("Ornek dosya", samples)
                file_bytes = (ROOT / cfg["paths"]["data_dir"] / choice).read_bytes()
                is_mask = "_mask" in choice
                st.caption("Maske" if is_mask
                           else "Uydu goruntusu - model calistirilacak")
            else:
                st.warning("data/train altinda ornek dosya bulunamadi.")
        else:
            is_mask = st.checkbox("Girdi zaten yol maskesi (model atlanir)",
                                  value=False)
            uploaded = st.file_uploader("Goruntu veya maske yukle",
                                        type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                file_bytes = uploaded.getvalue()

        run = st.button("Analizi calistir", type="primary",
                        use_container_width=True, disabled=file_bytes is None)
        st.divider()
        st.caption("Esikler ve skor agirliklari config.yaml uzerinden ayarlanir.")

    if run and file_bytes is not None:
        with st.spinner("Pipeline calisiyor ..."):
            try:
                st.session_state["result"] = run_pipeline(file_bytes, is_mask, cfg)
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Hata: {exc}")

    result = st.session_state.get("result")
    if result is None:
        st.info("Soldan bir goruntu/maske yukleyip 'Analizi calistir' deyin.")
        return

    tab_input, tab_graph, tab_crit, tab_sim, tab_res, tab_whatif = st.tabs(
        ["Girdi & Maske", "Yol Grafigi", "Kritiklik (Faz 2)",
         "Simulasyon (Faz 3)", "Resilience (Faz 4)", "What-if (Kapanma)"])
    with tab_input:
        render_input_tab(result)
    with tab_graph:
        render_graph_tab(result)
    with tab_crit:
        render_criticality_tab(result)
    with tab_sim:
        render_simulation_tab(result)
    with tab_res:
        render_resilience_tab(result)
    with tab_whatif:
        render_whatif_tab(result, cfg)


if __name__ == "__main__":
    main()
