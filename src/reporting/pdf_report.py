"""Vision2Graph analiz sonuclarini tek bir PDF rapora derler.

Rapor metni bilincli olarak ASCII-Turkce yazilir: standart PDF fontlari
(Helvetica) s/g/i gibi Turkce'ye ozgu karakterleri desteklemez ve projenin
geri kalani da ayni yaklasimi kullanir.
"""
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from src.analytics.scale import format_length

_MARGIN = 2 * cm
_PAGE_WIDTH = A4[0] - 2 * _MARGIN  # kenar bosluklari dusulmus kullanilabilir genislik

_GRADE_COLORS = {
    "A": colors.HexColor("#2ca02c"),
    "B": colors.HexColor("#7cb342"),
    "C": colors.HexColor("#ff7f0e"),
    "D": colors.HexColor("#e8551a"),
    "E": colors.HexColor("#d62728"),
}
_HEADER_BG = colors.HexColor("#2c3e50")


def _styles():
    sheet = getSampleStyleSheet()
    sheet.add(ParagraphStyle("V2GTitle", parent=sheet["Title"], fontSize=20))
    sheet.add(ParagraphStyle("V2GSection", parent=sheet["Heading2"],
                             textColor=_HEADER_BG, spaceBefore=14))
    sheet.add(ParagraphStyle("V2GCaption", parent=sheet["Normal"], fontSize=8,
                             textColor=colors.grey, alignment=1))
    sheet.add(ParagraphStyle("V2GSummary", parent=sheet["Normal"], fontSize=10,
                             leading=15, alignment=4, spaceBefore=4))
    return sheet


def _fmt(value):
    """Tablo hucresi icin deger bicimleme."""
    if isinstance(value, list):
        return "-".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _scaled_image(path, max_width):
    """Goruntuyu en-boy oranini koruyarak verilen genislige sigdirir."""
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = max_width
    image.drawHeight = max_width * ratio
    image.hAlign = "CENTER"
    return image


def _table(data):
    """Basligi koyu, satirlari zebra desenli standart tablo."""
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f4f6")]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _records_table(rows):
    """Sozluk listesini (orn. en kritik node'lar) tabloya cevirir."""
    headers = list(rows[0].keys())
    body = [[_fmt(row[key]) for key in headers] for row in rows]
    return _table([headers] + body)


def _pairs_table(pairs):
    """(anahtar, deger) ciftlerini iki sutunlu tabloya cevirir."""
    return _table([["Metrik", "Deger"]] + [[k, _fmt(v)] for k, v in pairs])


def _score_box(resilience):
    """Resilience skorunu rapor basinda one cikaran renkli kutu."""
    text = (f"RESILIENCE SKORU:  {resilience['score']} / 100"
            f"      [{resilience['grade']}]  {resilience['label']}")
    box = Table([[text]], colWidths=[_PAGE_WIDTH])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1),
         _GRADE_COLORS.get(resilience["grade"], colors.grey)),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return box


_VERDICTS = {
    "A": "Ag genel olarak dayanikli kabul edilebilir; tekil kapanmalar "
         "ulasimi ciddi sekilde tehdit etmemektedir.",
    "B": "Ag makul bir dayaniklilik seviyesindedir; yine de en kritik "
         "kavsaklar izlenmelidir.",
    "C": "Ag orta duzeyde dayaniklidir; kritik kavsaklarda iyilestirme "
         "yapilmasi onerilir.",
    "D": "Ag kirilgandir; kritik kavsak ve kopru baglantilar oncelikli "
         "olarak guclendirilmelidir.",
    "E": "Ag cok kirilgandir; tekil hata noktalari ulasimi ciddi sekilde "
         "tehdit etmektedir.",
}


def _executive_summary(counts, analysis, simulation, resilience):
    """Analiz sayilarini duz Turkce bir yonetici ozeti paragrafina cevirir."""
    parts = [
        f"Incelenen yol agi {counts['node_count']} kavsak ve "
        f"{counts['edge_count']} yol baglantisindan olusmakta, "
        f"{counts['component_count']} ayri parca icermektedir.",
        f"Genel dayaniklilik skoru {resilience['score']}/100 olarak "
        f"hesaplanmis ve ag '{resilience['label']}' ({resilience['grade']}) "
        f"sinifina yerlestirilmistir.",
    ]

    explanations = analysis.get("explanations") or []
    if explanations:
        parts.append(explanations[0]["explanation"])
    elif analysis["top_critical_nodes"]:
        top = analysis["top_critical_nodes"][0]["node"]
        parts.append(f"Agin en kritik kavsagi Node {top} olarak belirlenmistir.")

    targeted = simulation["targeted"]["robustness_index"]
    random_index = simulation["random"]["robustness_index"]
    if simulation["fragility_gap"] > 0.1:
        parts.append(
            f"Kademeli saldiri analizi, agin kasitli saldiriya rastsal "
            f"arizadan belirgin sekilde daha kirilgan oldugunu gostermektedir "
            f"(dayaniklilik indeksi kasitli {targeted}, rastsal {random_index}).")
    else:
        parts.append(
            f"Ag, kasitli ve rastsal kapanmalara benzer dayaniklilik "
            f"gostermektedir (dayaniklilik indeksi kasitli {targeted}, "
            f"rastsal {random_index}).")

    parts.append(_VERDICTS.get(resilience["grade"], ""))
    return " ".join(part for part in parts if part)


def build_pdf_report(output_dir, name, counts, analysis, simulation, resilience,
                     meters_per_pixel=None):
    """Faz 6 ana giris noktasi: kaydedilmis cikti ve metrikleri PDF'e derler.

    Pipeline'in onceki adimlarinin overlay/grafik PNG'lerini ve metrik
    sozluklerini birlestirir. PDF reports/{name}_report.pdf'e yazilir.
    """
    output_dir = Path(output_dir)
    report_path = output_dir / "reports" / f"{name}_report.pdf"
    styles = _styles()
    story = []

    story.append(Paragraph("Vision2Graph - Yol Agi Dayaniklilik Raporu",
                            styles["V2GTitle"]))
    story.append(Paragraph(f"Analiz girdisi: {name} &nbsp; | &nbsp; "
                            f"Tarih: {date.today().isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(_score_box(resilience))

    story.append(Paragraph("Yonetici Ozeti", styles["V2GSection"]))
    story.append(Paragraph(
        _executive_summary(counts, analysis, simulation, resilience),
        styles["V2GSummary"]))

    story.append(Paragraph("1. Yol Agi Ozeti", styles["V2GSection"]))
    story.append(_scaled_image(output_dir / "graphs" / f"{name}_graph.png", 11 * cm))
    story.append(Paragraph("Goruntuden cikarilan yol grafigi", styles["V2GCaption"]))
    story.append(Spacer(1, 6))
    count_pairs = [
        (key, format_length(value, meters_per_pixel) if key == "total_length"
         else value)
        for key, value in counts.items()]
    story.append(_pairs_table(count_pairs))

    story.append(Paragraph("2. Kritiklik Analizi", styles["V2GSection"]))
    story.append(_scaled_image(
        output_dir / "graphs" / f"{name}_criticality.png", 11 * cm))
    story.append(Paragraph(
        "Kritiklik isi haritasi - kirmizi yuksek, beyaz halka kesim noktasi",
        styles["V2GCaption"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Articulation points: {analysis['articulation_count']} &nbsp; | &nbsp; "
        f"Bridge edges: {analysis['bridge_count']}", styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("En kritik node'lar", styles["Normal"]))
    story.append(_records_table(analysis["top_critical_nodes"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Worst-case: en kritik node'lar kapaninca",
                           styles["Normal"]))
    story.append(_records_table(analysis["worst_case"]["impacts"]))

    story.append(Paragraph("3. Kapanma Simulasyonu", styles["V2GSection"]))
    story.append(_scaled_image(
        output_dir / "simulations" / f"{name}_degradation.png", 14 * cm))
    story.append(Paragraph("Kasitli saldiri vs rastsal ariza dayaniklilik egrisi",
                           styles["V2GCaption"]))
    story.append(Spacer(1, 6))
    story.append(_pairs_table([
        ("Kasitli saldiri R", simulation["targeted"]["robustness_index"]),
        ("Rastsal ariza R", simulation["random"]["robustness_index"]),
        ("Kirilganlik farki", simulation["fragility_gap"]),
    ]))

    story.append(Paragraph("4. Resilience Skoru Detayi", styles["V2GSection"]))
    story.append(_scaled_image(
        output_dir / "reports" / f"{name}_resilience.png", 14 * cm))
    story.append(Spacer(1, 6))
    weights = resilience["weights"]
    breakdown = [["Bilesen", "Skor (0-1)", "Agirlik"]]
    for key, value in resilience["components"].items():
        breakdown.append([key, _fmt(value), _fmt(weights.get(key, "-"))])
    story.append(_table(breakdown))

    SimpleDocTemplate(
        str(report_path), pagesize=A4,
        leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_MARGIN, bottomMargin=_MARGIN,
        title=f"Vision2Graph Raporu - {name}").build(story)
    return report_path
