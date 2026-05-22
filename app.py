"""Vision2Graph - Streamlit demo arayuzu.

Pipeline'i (maske -> skeleton -> graph -> analiz -> simulasyon -> resilience)
saran interaktif demo. Calistirmak icin:

    streamlit run app.py
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import yaml
from pyvis.network import Network

from src.analytics.criticality import analyze
from src.analytics.explainability import explain_critical_nodes
from src.analytics.improvement import suggest_improvements
from src.analytics.regions import analyze_regions
from src.analytics.resilience import resilience_score
from src.analytics.routing import route_impact
from src.analytics.scale import format_length, meters_per_pixel
from src.analytics.simplification import DETAIL_LEVELS, generalization_levels
from src.inference.infer_road_mask import (load_model, predict_mask,
                                           resolve_device)
from src.simulation.attack import simulate
from src.simulation.closure import apply_closure, closure_impact
from src.simulation.disaster import earthquake_zone, flood_zone, regional_zone
from src.topology.graph_builder import basic_counts, build_graph
from src.topology.skeletonizer import mask_to_skeleton
from src.visualization.closure_overlay import draw_closure_overlay
from src.visualization.collapse_animation import build_collapse_gif
from src.visualization.criticality_overlay import draw_criticality_overlay
from src.visualization.degradation_plot import draw_degradation_plot
from src.visualization.graph_overlay import draw_graph_overlay
from src.visualization.improvement_overlay import draw_improvement_overlay
from src.visualization.regions_overlay import draw_regions_overlay
from src.visualization.resilience_card import draw_resilience_card
from src.visualization.route_overlay import draw_route_overlay

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
    result["mpp"] = meters_per_pixel(cfg)  # piksel -> metre olcegi
    result["graph"] = graph  # what-if sekmesi grafik uzerinde calisir
    result["graph_overlay"] = draw_graph_overlay(base, graph)

    # analyze() kritiklik ozniteliklerini graph'a yazar; sonraki adimlar buna dayanir.
    result["analysis"] = analyze(graph, cfg)
    result["analysis"]["explanations"] = explain_critical_nodes(
        graph, result["analysis"])
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
        shown = (format_length(value, result["mpp"])
                 if key == "total_length" else value)
        col.metric(key, shown)


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
    if analysis.get("explanations"):
        st.subheader("Neden kritik?")
        for item in analysis["explanations"]:
            st.markdown(f"- {item['explanation']}")
    st.subheader("Worst-case: en kritik node'lar kapaninca")
    st.dataframe(analysis["worst_case"]["impacts"], use_container_width=True)


def render_simulation_tab(result):
    simulation = result["simulation"]
    fig = draw_degradation_plot(simulation)
    st.pyplot(fig)
    plt.close(fig)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rastsal ariza R", simulation["random"]["robustness_index"])
    col2.metric("Kasitli statik R", simulation["targeted"]["robustness_index"])
    col3.metric("Kasitli adaptif R", simulation["adaptive"]["robustness_index"])
    col4.metric("Kirilganlik farki", simulation["fragility_gap"])
    st.caption("R = LCR egrisi altindaki alan (0-1). Adaptif saldiri her adimda "
               "kritikligi yeniden hesaplar; genelde en dusuk R'yi (en sert "
               "cokusu) bu egri verir.")


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
    mpp = result["mpp"]

    st.subheader("Etki ozeti")
    cols = st.columns(4)
    cols[0].metric("Bilesen artisi", f"+{impact['component_increase']}")
    cols[1].metric("Verim kaybi", f"%{impact['efficiency_loss_pct']}")
    cols[2].metric("Izole kavsak", impact["isolated_nodes"])
    cols[3].metric("Kopan ag orani", f"%{impact['isolation_ratio_pct']}")
    if impact["isolated_nodes"]:
        st.warning(
            f"Bu kapanma {impact['isolated_nodes']} kavsagi ve "
            f"{format_length(impact['isolated_length'], mpp)} yolu (yol "
            f"agininin %{impact['isolation_ratio_pct']}'ini) ana agdan koparir.")

    st.subheader("Once / Sonra")
    before_col, after_col = st.columns(2)
    before_col.caption("KAPANMA ONCESI")
    before_col.metric("Bilesen sayisi", before["components"])
    before_col.metric("En buyuk bilesen orani", before["lcr"])
    before_col.metric("Toplam yol", format_length(before["total_length"], mpp))
    after_col.caption("KAPANMA SONRASI")
    after_col.metric("Bilesen sayisi", after["components"],
                     delta=after["components"] - before["components"],
                     delta_color="inverse")
    after_col.metric("En buyuk bilesen orani", after["lcr"],
                     delta=round(after["lcr"] - before["lcr"], 4))
    after_col.metric("Toplam yol", format_length(after["total_length"], mpp))

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


def render_route_tab(result):
    graph = result["graph"]
    nodes = sorted(graph.nodes())
    if len(nodes) < 2:
        st.info("Rota analizi icin grafikte en az 2 kavsak gerekir.")
        return

    # varsayilan hedef: baslangicla ayni bilesendeki bir kavsak (rota garanti)
    same_component = [n for n in nx.node_connected_component(graph, nodes[0])
                      if n != nodes[0]]
    default_target = same_component[-1] if same_component else nodes[-1]

    col1, col2 = st.columns(2)
    source = col1.selectbox("Baslangic kavsagi (A)", nodes, index=0)
    target = col2.selectbox("Hedef kavsagi (B)", nodes,
                            index=nodes.index(default_target))
    if source == target:
        st.info("Baslangic ve hedef ayni; farkli iki kavsak secin.")
        return

    closed_nodes = st.multiselect(
        "Kapatilacak kavsaklar (senaryo - bos birakilabilir)",
        [n for n in nodes if n not in (source, target)])

    impact = route_impact(graph, source, target, closed_nodes)
    if not impact["reachable_before"]:
        st.warning(f"Kavsak {source} ile {target} arasinda kapanma olmadan da "
                   f"yol yok - bu iki kavsak agin farkli parcalarinda.")
        return

    mpp = result["mpp"]
    st.subheader("Rota etkisi")
    cols = st.columns(4)
    cols[0].metric("Kapanma oncesi rota",
                   format_length(impact["before_length"], mpp))
    if impact["reachable_after"]:
        cols[1].metric("Kapanma sonrasi rota",
                       format_length(impact["after_length"], mpp))
        cols[2].metric("Rota uzamasi", f"%{impact['extension_pct']}")
        cols[3].metric("Detour faktoru", impact["detour_after"] or "-")
    else:
        cols[1].metric("Kapanma sonrasi rota", "KESILDI")
        cols[2].metric("Rota uzamasi", "sonsuz")
        cols[3].metric("Detour faktoru", "-")
        st.error(f"Bu kapanma {source}-{target} baglantisini tamamen koparir; "
                 f"alternatif rota yoktur.")

    if not closed_nodes:
        st.caption("Henuz kapanma secilmedi - gosterilen, mevcut en kisa rotadir.")

    st.subheader("Gorsel")
    overlay = draw_route_overlay(result["source"], graph, impact["before_path"],
                                 impact["after_path"], source, target)
    st.image(to_rgb(overlay),
             caption="Yesil: baslangic  |  kirmizi: hedef  |  mavi: kapanma "
                     "oncesi rota  |  turuncu: kapanma sonrasi rota",
             use_container_width=True)


def _demo_route_endpoints(graph, critical):
    """Demo icin, rotasi kritik node'dan gecen anlamli bir A-B cifti secer.

    Kritik node bir kesim noktasiysa, onu cikarinca bilesen parcalara ayrilir;
    farkli parcalardan birer node secince aralarindaki rota zorunlu olarak
    kritik node'dan gecer - kapanmanin etkisi boylece net gorulur.
    """
    component = nx.node_connected_component(graph, critical)
    pieces = sorted(nx.connected_components(graph.subgraph(component - {critical})),
                    key=len, reverse=True)
    if len(pieces) >= 2:
        return max(pieces[0], key=graph.degree), max(pieces[1], key=graph.degree)
    ordered = sorted(component)
    return (ordered[0], ordered[-1]) if len(ordered) >= 2 else (None, None)


def render_demo_tab(result):
    graph = result["graph"]
    analysis = result["analysis"]
    resilience = result["resilience"]
    st.write("Otomatik senaryo: sistem en kritik kavsagi bulur, kapatir ve "
             "aga + ornek bir rotaya etkisini adim adim gosterir.")

    if not analysis["top_critical_nodes"]:
        st.warning("Grafik bu senaryo icin yetersiz.")
        return
    critical = analysis["top_critical_nodes"][0]["node"]

    st.subheader("1. Agin genel durumu")
    cols = st.columns(2)
    cols[0].metric("Resilience skoru", f"{resilience['score']} / 100")
    cols[1].metric("En kritik kavsak", f"Node {critical}")
    st.markdown(f"**Ag sinifi:** {resilience['grade']} - {resilience['label']}")
    explanation = next((item["explanation"]
                        for item in analysis.get("explanations", [])
                        if item["node"] == critical), "")
    if explanation:
        st.info(explanation)

    st.subheader(f"2. Senaryo: Node {critical} kapaniyor")
    impact = closure_impact(graph, [critical])
    row1 = st.columns(2)
    row1[0].metric("Bilesen artisi", f"+{impact['component_increase']}")
    row1[1].metric("Verim kaybi", f"%{impact['efficiency_loss_pct']}")
    row2 = st.columns(2)
    row2[0].metric("Izole kavsak", impact["isolated_nodes"])
    row2[1].metric("Kopan ag orani", f"%{impact['isolation_ratio_pct']}")
    st.image(to_rgb(draw_closure_overlay(result["source"], graph, [critical],
                                         impact["isolated_node_list"])),
             caption="Magenta X: kapatilan kavsak  |  kirmizi: ana agdan kopan kisim",
             use_container_width=True)

    st.subheader("3. Ornek bir rotaya etkisi")
    source_node, target_node = _demo_route_endpoints(graph, critical)
    if source_node is None:
        st.caption("Bu grafikte uygun ornek rota bulunamadi.")
    else:
        route = route_impact(graph, source_node, target_node,
                             closed_nodes=[critical])
        rcols = st.columns(3)
        rcols[0].metric("Onceki rota",
                        format_length(route["before_length"], result["mpp"])
                        if route["reachable_before"] else "-")
        if route["reachable_after"]:
            rcols[1].metric("Sonraki rota",
                            format_length(route["after_length"], result["mpp"]))
            rcols[2].metric("Rota uzamasi", f"%{route['extension_pct']}")
        else:
            rcols[1].metric("Sonraki rota", "KESILDI")
            rcols[2].metric("Rota uzamasi", "sonsuz")
        st.image(to_rgb(draw_route_overlay(result["source"], graph,
                                           route["before_path"],
                                           route["after_path"],
                                           source_node, target_node)),
                 caption=f"Node {source_node} -> Node {target_node} rotasi "
                         f"(mavi: oncesi, turuncu: sonrasi)",
                 use_container_width=True)
        st.subheader("Sonuc")
        if route["reachable_after"]:
            st.success(f"Node {critical} kapandiginda ornek rota "
                       f"%{route['extension_pct']} uzamakta, ag "
                       f"'{resilience['label']}' seviyesinde kalmaktadir.")
        else:
            st.error(f"Node {critical} kapandiginda Node {source_node}-"
                     f"{target_node} baglantisi tamamen kopmaktadir; bu kavsak "
                     f"ag icin tekil bir hata noktasidir.")


def render_improvement_tab(result, cfg):
    graph = result["graph"]
    st.write("Sistem, aga eklenebilecek yeni yol baglantilarini dener ve "
             "dayanikliligi en cok artiracak olani onerir.")

    if st.button("Iyilestirme onerisi hesapla", type="primary"):
        with st.spinner("Aday baglantilar degerlendiriliyor ..."):
            st.session_state["improvement"] = suggest_improvements(graph, cfg)

    suggestion = st.session_state.get("improvement")
    if suggestion is None:
        st.info("Oneri uretmek icin yukaridaki butona basin.")
        return
    if not suggestion["suggestions"]:
        st.warning("Denenebilecek uygun yeni baglanti adayi bulunamadi.")
        return

    st.metric("Mevcut resilience skoru",
              f"{suggestion['base_score']} / 100 ({suggestion['base_grade']})")
    best = suggestion["suggestions"][0]
    if best["gain"] > 0:
        st.success(f"Onerilen baglanti: Node {best['edge'][0]} - Node "
                   f"{best['edge'][1]}. Eklenirse resilience skoru "
                   f"{suggestion['base_score']} -> {best['score_after']} "
                   f"(+{best['gain']} puan).")
    else:
        st.info("Denenen baglantilar skoru anlamli sekilde artirmadi; ag "
                "mevcut haliyle bu mudahalelere kapali.")

    st.subheader("Aday baglantilar")
    st.dataframe(suggestion["suggestions"], use_container_width=True)

    st.subheader("Onerilen baglanti")
    st.image(to_rgb(draw_improvement_overlay(result["source"], graph,
                                             best["edge"])),
             caption="Yesil: aga onerilen yeni yol baglantisi",
             use_container_width=True)


def _heat_hex(value):
    """0-1 kritiklik skorunu yesil->sari->kirmizi hex renge cevirir."""
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        red, green = int(510 * value), 200
    else:
        red, green = 255, int(200 * (1.0 - (value - 0.5) / 0.5))
    return f"#{red:02x}{green:02x}30"


def render_interactive_tab(result):
    graph = result["graph"]
    st.write("Tiklanabilir yol grafigi - bir kavsagin uzerine gelince ID, "
             "kritiklik, derece ve betweenness degerleri balonda gorunur. "
             "Renk ve boyut kritikligi yansitir (yesil dusuk, kirmizi yuksek). "
             "Grafik surukle/yakinlastir ile gezilebilir.")

    network = Network(height="600px", width="100%", bgcolor="#0e1117",
                      font_color="#fafafa", cdn_resources="in_line")
    network.toggle_physics(True)
    for node, data in graph.nodes(data=True):
        criticality = data.get("criticality", 0.0)
        tooltip = (f"Node {node} | kritiklik {round(criticality, 3)} | "
                   f"derece {graph.degree(node)} | "
                   f"betweenness {round(data.get('betweenness', 0.0), 3)}")
        if data.get("is_articulation"):
            tooltip += " | KESIM NOKTASI"
        network.add_node(int(node), label=str(node), title=tooltip,
                         color=_heat_hex(criticality),
                         size=10 + criticality * 22)
    for u, v, data in graph.edges(data=True):
        network.add_edge(int(u), int(v),
                         color="#d62728" if data.get("is_bridge") else "#888888")
    components.html(network.generate_html(), height=620)

    st.subheader("Kavsak detayi")
    node = st.selectbox("Incelenecek kavsak", sorted(graph.nodes()))
    data = graph.nodes[node]
    cols = st.columns(4)
    cols[0].metric("Kritiklik", round(data.get("criticality", 0.0), 3))
    cols[1].metric("Derece", graph.degree(node))
    cols[2].metric("Betweenness", round(data.get("betweenness", 0.0), 3))
    cols[3].metric("Kesim noktasi",
                   "Evet" if data.get("is_articulation") else "Hayir")


def render_regions_tab(result, cfg):
    graph = result["graph"]
    st.write("Ag 2x2 mekansal bolgeye ayrilir; her bolgenin kendi ic yol agi "
             "ayri puanlanir ve en kirilgan bolge belirlenir.")

    if st.button("Bolge analizini calistir", type="primary"):
        with st.spinner("Bolgeler degerlendiriliyor ..."):
            st.session_state["regions"] = analyze_regions(graph, cfg)

    region_data = st.session_state.get("regions")
    if region_data is None:
        st.info("Bolge analizi icin yukaridaki butona basin.")
        return

    st.subheader("Bolge skorlari")
    st.dataframe(
        [{"bolge": r["region"], "kavsak": r["node_count"],
          "skor": r["score"] if r["score"] is not None else "-",
          "sinif": r["grade"]} for r in region_data["regions"]],
        use_container_width=True)

    weakest = region_data["weakest"]
    if weakest is None:
        st.warning("Bolgeler ayri analiz icin yeterince buyuk degil.")
        return

    st.error(f"En kirilgan bolge: {weakest['region']} - skor "
             f"{weakest['score']}/100 ({weakest['label']}). Bu bolge ile ana "
             f"omurga arasinda ek baglanti kurmak iyilestirme onceligidir.")
    if weakest["top_critical"]:
        st.write("Bu bolgenin en kritik kavsaklari:")
        st.dataframe(weakest["top_critical"], use_container_width=True)

    st.subheader("Bolge haritasi")
    st.image(to_rgb(draw_regions_overlay(result["source"], graph,
                                         weakest["nodes"])),
             caption="Kirmizi: en kirilgan bolgenin kavsaklari  |  "
                     "cizgiler: 2x2 bolge siniri",
             use_container_width=True)


def _comparison_summary(res):
    """Bir analiz sonucundan karsilastirma metriklerini cikarir."""
    return {
        "Resilience skoru": res["resilience"]["score"],
        "Sinif": res["resilience"]["grade"],
        "Kavsak sayisi": res["counts"]["node_count"],
        "Yol sayisi": res["counts"]["edge_count"],
        "Bilesen sayisi": res["counts"]["component_count"],
        "Articulation point": res["analysis"]["articulation_count"],
        "Bridge edge": res["analysis"]["bridge_count"],
        "Kasitli saldiri R": res["simulation"]["targeted"]["robustness_index"],
        "Rastsal ariza R": res["simulation"]["random"]["robustness_index"],
    }


def render_comparison_tab(result, cfg):
    st.write("Mevcut analizi baska bir ornek yol agiyla karsilastirin.")
    samples = list_samples(cfg["paths"]["data_dir"])
    if not samples:
        st.info("Karsilastirma icin ornek veri bulunamadi.")
        return

    choice = st.selectbox("Karsilastirilacak ikinci ornek", samples)
    if st.button("Karsilastir", type="primary"):
        with st.spinner(f"{choice} analiz ediliyor ..."):
            try:
                data = (ROOT / cfg["paths"]["data_dir"] / choice).read_bytes()
                st.session_state["comparison"] = run_pipeline(
                    data, "_mask" in choice, cfg)
                st.session_state["comparison_name"] = choice
            except Exception as exc:
                st.session_state.pop("comparison", None)
                st.error(f"Hata: {exc}")

    second = st.session_state.get("comparison")
    if second is None:
        st.info("Bir ikinci ornek secip 'Karsilastir' butonuna basin.")
        return

    name_b = st.session_state.get("comparison_name", "Ikinci")
    summary_a = _comparison_summary(result)
    summary_b = _comparison_summary(second)

    st.subheader("Karsilastirma tablosu")
    st.dataframe(
        [{"Metrik": key, "A (mevcut)": summary_a[key],
          f"B ({name_b})": summary_b[key]} for key in summary_a],
        use_container_width=True)

    score_a, score_b = summary_a["Resilience skoru"], summary_b["Resilience skoru"]
    if abs(score_a - score_b) < 1.0:
        st.info("Iki yol agi benzer dayaniklilik seviyesindedir.")
    else:
        stronger = "A (mevcut)" if score_a > score_b else f"B ({name_b})"
        st.success(f"{stronger} daha dayanikli - resilience skoru "
                   f"{max(score_a, score_b)} / {min(score_a, score_b)}.")

    col_a, col_b = st.columns(2)
    col_a.caption("A (mevcut)")
    col_a.image(to_rgb(result["graph_overlay"]), use_container_width=True)
    col_b.caption(f"B ({name_b})")
    col_b.image(to_rgb(second["graph_overlay"]), use_container_width=True)


def render_collapse_tab(result):
    graph = result["graph"]
    st.write("Kademeli kasitli saldiri animasyonu: en kritik kavsaklar sirayla "
             "kaldirilirken agin adim adim parcalanisi GIF olarak uretilir.")

    node_count = graph.number_of_nodes()
    if node_count < 3:
        st.info("Animasyon icin grafikte en az 3 kavsak gerekir.")
        return

    max_steps = st.slider("Kac kavsak kaldirilsin", 3,
                          min(30, node_count), min(12, node_count))
    if st.button("Animasyonu olustur", type="primary"):
        order = sorted(graph.nodes(),
                       key=lambda n: graph.nodes[n].get("criticality", 0.0),
                       reverse=True)
        with st.spinner("GIF olusturuluyor ..."):
            st.session_state["collapse_gif"] = build_collapse_gif(
                result["source"], graph, order, max_steps=max_steps)

    gif = st.session_state.get("collapse_gif")
    if gif is None:
        st.info("Animasyon icin yukaridaki butona basin.")
        return
    st.image(gif, caption="En kritik kavsaklar sirayla kaldiriliyor",
             use_container_width=True)


def render_disaster_tab(result):
    graph = result["graph"]
    nodes = sorted(graph.nodes())
    st.write("Hazir afet senaryolari: secili bir alan toplu kapatilir ve yol "
             "agininin nasil etkilendigi olculur. Gercek afet verisi olmadigi "
             "icin bunlar mekansal varsayimlara dayali simulasyonlardir.")

    kind = st.radio("Senaryo", ["Deprem", "Sel", "Bolgesel kapanma"],
                    horizontal=True)
    if kind == "Deprem":
        epicenter = st.selectbox("Episantr kavsagi", nodes)
        radius = st.slider("Etki yaricapi (piksel)", 20, 400, 120, 10)
        closed = earthquake_zone(graph, epicenter, radius)
    elif kind == "Sel":
        level = st.slider("Su seviyesi (goruntunun alt yuzdesi)", 10, 80, 35, 5)
        closed = flood_zone(graph, level / 100)
    else:
        region = st.radio("Bolge", ["Kuzeybati", "Kuzeydogu",
                                    "Guneybati", "Guneydogu"], horizontal=True)
        closed = regional_zone(graph, region)

    if not closed:
        st.info("Bu senaryo hicbir kavsagi kapsamiyor; parametreleri degistirin.")
        return

    impact = closure_impact(graph, closed_nodes=closed)
    before, after = impact["before"], impact["after"]

    st.subheader(f"Etki - {len(closed)} kavsak kapandi")
    cols = st.columns(4)
    cols[0].metric("Bilesen artisi", f"+{impact['component_increase']}")
    cols[1].metric("Verim kaybi", f"%{impact['efficiency_loss_pct']}")
    cols[2].metric("Izole kavsak", impact["isolated_nodes"])
    cols[3].metric("Kopan ag orani", f"%{impact['isolation_ratio_pct']}")

    before_col, after_col = st.columns(2)
    before_col.caption("AFET ONCESI")
    before_col.metric("Bilesen sayisi", before["components"])
    before_col.metric("En buyuk bilesen orani", before["lcr"])
    after_col.caption("AFET SONRASI")
    after_col.metric("Bilesen sayisi", after["components"],
                     delta=after["components"] - before["components"],
                     delta_color="inverse")
    after_col.metric("En buyuk bilesen orani", after["lcr"],
                     delta=round(after["lcr"] - before["lcr"], 4))

    st.image(to_rgb(draw_closure_overlay(result["source"], graph, closed,
                                         impact["isolated_node_list"])),
             caption="Magenta X: kapanan kavsaklar  |  kirmizi: ana agdan kopan "
                     "kisim  |  gri: ayakta kalan ag",
             use_container_width=True)


def render_simplification_tab(result):
    graph = result["graph"]
    st.write("Yol grafigi farkli detay seviyelerinde sadelestirilir: dusuk "
             "kritiklikli yollar atilarak ag daha genel bir omurgaya indirgenir "
             "(harita genellestirme).")

    levels = generalization_levels(graph)
    st.subheader("Detay seviyeleri")
    st.dataframe(
        [{"seviye": lvl["level"], "ad": lvl["name"],
          "kavsak": lvl["node_count"], "yol": lvl["edge_count"],
          "yol %": lvl["edge_ratio_pct"], "uzunluk %": lvl["length_ratio_pct"]}
         for lvl in levels],
        use_container_width=True)

    choice = st.radio(
        "Goruntulenecek detay seviyesi", [lvl["level"] for lvl in levels],
        format_func=lambda lv: f"Seviye {lv} - {DETAIL_LEVELS[lv][0]}",
        horizontal=True)
    selected = next(lvl for lvl in levels if lvl["level"] == choice)
    st.image(to_rgb(draw_graph_overlay(result["source"], selected["graph"])),
             caption=f"Seviye {choice}: {selected['name']} - yollarin "
                     f"%{selected['edge_ratio_pct']}'i korundu",
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
        demo = st.button("Hizli demo - ilk ornekle calistir",
                         use_container_width=True)
        st.caption("Esikler ve skor agirliklari config.yaml uzerinden ayarlanir.")

    if demo:
        samples = list_samples(cfg["paths"]["data_dir"])
        if samples:
            file_bytes = (ROOT / cfg["paths"]["data_dir"] / samples[0]).read_bytes()
            is_mask = "_mask" in samples[0]
            run = True
        else:
            st.error("data/train altinda demo icin ornek dosya bulunamadi.")

    if run and file_bytes is not None:
        with st.spinner("Pipeline calisiyor ..."):
            try:
                st.session_state["result"] = run_pipeline(file_bytes, is_mask, cfg)
                # yeni analizde eski on-demand sonuclari temizle
                st.session_state.pop("improvement", None)
                st.session_state.pop("regions", None)
                st.session_state.pop("comparison", None)
                st.session_state.pop("collapse_gif", None)
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Hata: {exc}")

    result = st.session_state.get("result")
    if result is None:
        st.info("Soldan bir goruntu/maske yukleyip 'Analizi calistir' deyin.")
        return

    (tab_demo, tab_input, tab_graph, tab_crit, tab_sim, tab_res, tab_whatif,
     tab_route, tab_disaster, tab_collapse, tab_improve, tab_regions,
     tab_simplify, tab_compare, tab_interactive) = st.tabs(
        ["Demo Senaryo", "Girdi & Maske", "Yol Grafigi", "Kritiklik (Faz 2)",
         "Simulasyon (Faz 3)", "Resilience (Faz 4)", "What-if (Kapanma)",
         "A-B Rota", "Afet Senaryosu", "Cokus Animasyonu", "Iyilestirme",
         "Bolge Analizi", "Sadelestirme", "Karsilastirma", "Interaktif Graph"])
    with tab_demo:
        render_demo_tab(result)
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
    with tab_route:
        render_route_tab(result)
    with tab_disaster:
        render_disaster_tab(result)
    with tab_collapse:
        render_collapse_tab(result)
    with tab_improve:
        render_improvement_tab(result, cfg)
    with tab_regions:
        render_regions_tab(result, cfg)
    with tab_simplify:
        render_simplification_tab(result)
    with tab_compare:
        render_comparison_tab(result, cfg)
    with tab_interactive:
        render_interactive_tab(result)


if __name__ == "__main__":
    main()
