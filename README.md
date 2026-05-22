# Vision2Graph

**Satellite Road Network Intelligence & Resilience Simulator**
Uydu görüntülerinden yol ağı çıkarımı ve graph tabanlı dayanıklılık simülasyonu.

## Amaç

Vision2Graph, uydu/drone görüntülerinden çıkarılan yol ağlarını matematiksel
çizge (graph) yapısına dönüştürür; bu yapı üzerinden kritik bağlantıları tespit
eder, kapanma senaryolarını simüle eder ve ağ dayanıklılığını ölçer.

Projenin amacı yalnızca yolu *bulmak* değil, bulunan yol ağının **stratejik
önemini ve kırılgan noktalarını** analiz etmektir.

## Pipeline

```
Uydu/Drone Görüntüsü
  → Yol Maskesi (D-LinkNet34)
  → Skeleton
  → Graph (NetworkX)
  → Kritik Node/Edge Analizi
  → Kapanma Simülasyonu
  → Resilience Score
  → Rapor / Demo Arayüzü
```

## Kurulum

```bash
pip install -r requirements.txt
```

`models/log01_dink34.th` (D-LinkNet34 checkpoint) ve `data/` veri seti, boyutları
nedeniyle depoya dahil değildir; yerel olarak ilgili klasörlere yerleştirilmelidir.

## Kullanım

```bash
# Uydu görüntüsünden (model ile)
python main.py data/train/100034_sat.jpg

# Hazır yol maskesinden (model atlanır)
python main.py data/train/100034_mask.png --mask
```

Çıktılar `outputs/` altında üretilir: yol maskesi, skeleton ve graph overlay.

## Proje Yapısı

```
src/
  inference/      D-LinkNet modeli ve yol maskesi çıkarımı
  topology/       Skeletonization ve skeleton→graph dönüşümü
  analytics/      Graph metrikleri, kritiklik ve dayanıklılık analizi
  visualization/  Graph overlay ve etki haritaları
  reporting/      JSON / HTML raporlama
```

## Geliştirme Durumu

- [x] Faz 1 — Core Pipeline (görüntü → maske → skeleton → graph)
- [ ] Faz 2 — Graph Intelligence (kritik node/edge analizi)
- [ ] Faz 3 — Simulation Engine (kapanma simülasyonu, A-B rota etkisi)
- [ ] Faz 4 — Resilience Analysis (random vs targeted, dayanıklılık skoru)
- [ ] Faz 5 — Streamlit arayüzü
- [ ] Faz 6 — Raporlama ve sunum
