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

## Gelişmiş Analiz Özellikleri

Temel pipeline'a ek olarak Streamlit arayüzü 11 sekmede şu analizleri sunar:

- **Kritiklik & açıklama** — betweenness, kesim noktası, köprü tespiti; her kritik
  kavşak için "neden kritik?" açıklaması
- **Kapanma simülasyonu** — rastsal / statik kasıtlı / adaptif saldırı dayanıklılık
  eğrileri
- **What-if** — seçilen kavşak/yolu kapatıp ağ etkisini (izolasyon, önce/sonra)
  anlık görme
- **A-B rota etkisi** — bir kapanmanın iki nokta arası rotayı ne kadar uzattığı
  veya kopardığı
- **Resilience skoru** — beş metriği tek 0-100 nota toplama
- **İyileştirme önerisi** — dayanıklılığı en çok artıracak yeni yol bağlantısını
  bulma
- **Bölge analizi** — ağı mekansal bölgelere ayırıp en kırılgan bölgeyi belirleme
- **Afet senaryosu** — deprem / sel / bölgesel toplu kapanma simülasyonu
- **Çöküş animasyonu** — kademeli saldırının adım adım GIF görselleştirmesi
- **Topolojik sadeleştirme** — yol grafiğini farklı detay seviyelerinde
  genelleştirme (harita genelleştirme)
- **Karşılaştırmalı analiz** — iki yol ağının resilience metrikleriyle kıyası
- **İnteraktif graph** — sürükle/yakınlaştır, hover ile kavşak detayları
- **Demo senaryo** — tek tıkla otomatik sunum senaryosu

## Kurulum

```bash
pip install -r requirements.txt
```

`models/log01_dink34.th` (D-LinkNet34 checkpoint) ve `data/` veri seti, boyutları
nedeniyle depoya dahil değildir; yerel olarak ilgili klasörlere yerleştirilmelidir.

## Kullanım

### Komut satırı — tam pipeline

```bash
# Uydu görüntüsünden (model ile)
python main.py data/train/100034_sat.jpg

# Hazır yol maskesinden (model atlanır)
python main.py data/train/100034_mask.png --mask
```

8 adımlık pipeline çıktıları `outputs/` altında üretilir: yol maskesi, skeleton,
graph overlay, kritiklik ısı haritası, dayanıklılık eğrisi, resilience kartı,
JSON metrik raporları ve derlenmiş tek dosyalık PDF rapor.

### İnteraktif demo — Streamlit

```bash
streamlit run app.py
```

Tarayıcı arayüzünden görüntü/maske yüklenir veya hazır örnek veri seçilir; tüm
analiz sonuçları sekmeler halinde görüntülenir.

## Testler

```bash
pytest
```

`tests/` altındaki birim testleri elde üretilen sentetik bir yol grafiği
üzerinde çalışır; gerçek veri veya model checkpoint'i gerektirmez.

## Proje Yapısı

```
src/
  inference/      D-LinkNet modeli ve yol maskesi çıkarımı
  topology/       Skeletonization ve skeleton→graph dönüşümü
  analytics/      Kritiklik, resilience, açıklama, rota, iyileştirme, bölge analizi
  simulation/     Kapanma etki motoru ve kademeli saldırı simülasyonu
  visualization/  Graph overlay, etki haritaları ve grafikler
  reporting/      Analiz sonuçlarının PDF rapora derlenmesi
app.py            Streamlit demo arayüzü (11 sekme)
main.py           Uçtan uca komut satırı pipeline
```

## Geliştirme Durumu

- [x] Faz 1 — Core Pipeline (görüntü → maske → skeleton → graph)
- [x] Faz 2 — Graph Intelligence (kritik node/edge analizi, worst-case)
- [x] Faz 3 — Simulation Engine (kapanma simülasyonu, kasıtlı vs rastsal saldırı)
- [x] Faz 4 — Resilience Analysis (metrikleri tek dayanıklılık skoruna toplama)
- [x] Faz 5 — Streamlit arayüzü
- [x] Faz 6 — Raporlama ve sunum (PDF)
