# Vision2Graph — Kapsamlı Kılavuz

> **Uydu Görüntülerinden Yol Ağı Çıkarımı ve Dayanıklılık Analizi**
> Bitirme projesi · graph tabanlı karar destek sistemi

---

## İçindekiler

1. [Bir Bakışta](#1-bir-bakişta)
2. [Proje Nedir? (Amaç)](#2-proje-nedir-amaç)
3. [Nasıl Çalışır? (Pipeline)](#3-nasil-çalişir-pipeline)
4. [Özellikler — Tam Liste](#4-özellikler--tam-liste)
5. [Mimari / Klasör Yapısı](#5-mimari--klasör-yapısı)
6. [Yeni PC'de Kurulum](#6-yeni-pcde-kurulum)
7. [Test Rehberi (Adım Adım)](#7-test-rehberi-adim-adim)
8. [Çıktılar (Outputs)](#8-çıktılar-outputs)
9. [Konfigürasyon (config.yaml)](#9-konfigürasyon-configyaml)
10. [Teknik Detaylar](#10-teknik-detaylar)
11. [Bilinen Sınırlar](#11-bilinen-sınırlar)
12. [Sunum İçin Önerilen Akış](#12-sunum-için-önerilen-akiş)
13. [Sık Karşılaşılan Sorunlar](#13-sik-karşılaşılan-sorunlar)
14. [Kaynaklar](#14-kaynaklar)

---

## 1) Bir Bakışta

**Vision2Graph**, uydu/drone görüntülerinden yol ağı çıkarıp bu ağ üzerinde **stratejik dayanıklılık analizi** yapan bir araçtır. Yalnızca yolu *bulmaz*; bulunan yolun **kritik noktalarını tespit eder**, **kapanma senaryolarını simüle eder**, **iyileştirme önerileri sunar** ve **profesyonel PDF rapor** üretir.

### Bir cümleyle ne işe yarar?
> "Bu yol ağındaki en kritik kavşak hangisi, kapanırsa ne olur, nereye yeni bir bağlantı eklersek ağ daha dayanıklı hale gelir?" sorularına **görsel, sayısal ve metin tabanlı yanıt** verir.

### Yeteneklerin özeti

| Boyut | Yetenek |
|-------|---------|
| **Çıkarım** | D-LinkNet34 ile uydu görüntüsünden yol maskesi |
| **Topoloji** | Skeleton'lama + NetworkX grafiği oluşturma |
| **Analiz** | Centrality, kesim noktası, köprü, kritiklik skoru |
| **Simülasyon** | Rastsal arıza, kasıtlı statik & adaptif saldırı |
| **Karar Desteği** | İyileştirme önerisi, en kırılgan bölge tespiti |
| **Açıklanabilirlik** | Her kritik kavşak için düz Türkçe gerekçe |
| **Görselleştirme** | Isı haritası, eğri grafiği, çöküş GIF'i, interaktif graph |
| **Raporlama** | 4 sayfalık otomatik PDF (yönetici özetiyle) |
| **Arayüz** | 15 sekmeli Streamlit demo + CLI pipeline |

---

## 2) Proje Nedir? (Amaç)

### 2.1 Çözülen Problem

Uydu görüntülerinden yol çıkarımı yapan **birçok** model var (D-LinkNet, U-Net türevleri vs). Ancak çıkarılan yol katmanı genelde **sadece görsel** kalır — *"Bu yolun stratejik önemi nedir?"*, *"Hangi kavşak kritik?"*, *"Doğal afet vurursa ağ ne kadar dayanır?"* sorularını **kimse otomatik yanıtlamaz**.

Vision2Graph bu boşluğu doldurur: **yolun kendisini değil, yolun stratejik anlamını** analiz eder.

### 2.2 Hedef Kullanım Senaryoları

- **Afet yönetimi:** Bir bölgede sel/deprem olursa hangi yollar kopar, hangi bölgeler izole olur?
- **Şehir planlama:** Mevcut yol ağına nereye yeni bir bağlantı eklemek dayanıklılığı en çok artırır?
- **Lojistik / Acil müdahale:** İki nokta arası rota, kritik bir kavşak kapanırsa nasıl etkilenir?
- **Akademik analiz:** Bir bölgenin yol ağı rastsal arızaya mı yoksa kasıtlı saldırıya mı daha kırılgan?

### 2.3 Akademik Çerçeve

Bu çalışma, **TÜBİTAK harita genelleştirme** temasına yol katmanı odağında bir uyarlama olarak konumlanır. Mevcut sistem:

- **Graph tabanlı resilience analizi** literatürünü (Holme & Kim 2002, Latora & Marchiori 2001) yol ağına uygular,
- **Adaptif kasıtlı saldırı** (recalculated attack) gibi akademik konseptleri implement eder,
- **Topolojik genelleştirme** (kritiklik eşiğine göre detay seviyeleri) ile haritacılık genelleştirme fikriyle doğrudan bağ kurar.

---

## 3) Nasıl Çalışır? (Pipeline)

Sistem **8 adımlı** uçtan uca bir pipeline'dır. Her adım bir öncekinin çıktısını tüketir.

```
┌──────────────────────────────────────────────────────┐
│ GİRDİ:  Uydu görüntüsü (.jpg) veya yol maskesi (.png)│
└─────────────────────┬────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
   [1]  │  Yol Maskesi Çıkarımı     │  D-LinkNet34 (PyTorch)
        │  (veya hazır maske yükle) │  → outputs/masks/
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
   [2]  │  Skeletonization          │  skimage.morphology
        │  (yolları tek piksel iz)  │  → outputs/skeletons/
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
   [3]  │  Graph İnşası             │  sknw + NetworkX
        │  (kavşak=node, yol=edge)  │  (bellekte)
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
   [4]  │  Graph Overlay            │  cv2 çizim
        │  (yol haritasını renkle)  │  → outputs/graphs/
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
   [5]  │  Analiz (Kritiklik)       │  Faz 2
        │  · betweenness            │  → outputs/graphs/_criticality.png
        │  · articulation point     │  → outputs/reports/_analysis.json
        │  · bridge edges           │
        │  · worst-case kapanma     │
        │  · "Neden kritik?" cümle  │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
   [6]  │  Simülasyon (3 strateji)  │  Faz 3
        │  · rastsal arıza          │  → outputs/simulations/
        │  · kasıtlı statik saldırı │
        │  · kasıtlı adaptif saldırı│
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
   [7]  │  Resilience Skoru         │  Faz 4
        │  (5 metrik → 0-100 puan)  │  → outputs/reports/_resilience.{json,png}
        │  + harf notu (A-E)        │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
   [8]  │  PDF Rapor Derleme        │  ReportLab
        │  · Yönetici Özeti         │  → outputs/reports/_report.pdf
        │  · 4 bölüm + tablolar     │
        └───────────────────────────┘
```

### Pipeline adımlarının özeti

| Adım | Modül | Görev | Çıktı |
|------|-------|-------|-------|
| 1 | `inference/` | Maske çıkarma (model veya hazır) | `_mask.png` |
| 2 | `topology/skeletonizer.py` | Skeleton'lama | `_skeleton.png` |
| 3 | `topology/graph_builder.py` | Graph oluşturma | NetworkX nesnesi |
| 4 | `visualization/graph_overlay.py` | Yol haritası overlay | `_graph.png` |
| 5 | `analytics/criticality.py` + `explainability.py` | Kritiklik + açıklama | `_criticality.png`, `_analysis.json` |
| 6 | `simulation/attack.py` | 3 saldırı eğrisi | `_degradation.png`, `_simulation.json` |
| 7 | `analytics/resilience.py` | Tek skor (0-100) | `_resilience.{json,png}` |
| 8 | `reporting/pdf_report.py` | PDF rapor | `_report.pdf` |

---

## 4) Özellikler — Tam Liste

### 4.1 Komut Satırı (CLI) — `main.py`

```bash
python main.py data/train/100129_mask.png --mask         # hazır maske
python main.py data/train/100034_sat.jpg                 # model ile maske üret
```

8 adımı sırayla çalıştırır, ekrana her adımın özetini basar, sonunda tüm çıktıları `outputs/` altına kaydeder.

### 4.2 Streamlit Arayüzü — 15 Sekme

```bash
streamlit run app.py
```

Tarayıcı açılır. Soldaki sidebar'dan **örnek veri seç + "Analizi calistir"** veya **"Hizli demo - ilk ornekle calistir"** tek tıkla otomatik analiz başlatır.

| # | Sekme | Backlog | Ne Yapar |
|---|-------|---------|----------|
| 1 | **Demo Senaryo** | #6 | Tek tıkla otomatik 3 adımlı sunum senaryosu (en kritik kavşak → kapat → rotaya etki) |
| 2 | **Girdi & Maske** | — | Yüklenen görüntü + çıkarılan yol maskesi yan yana |
| 3 | **Yol Grafiği** | — | Skeleton + graph overlay + temel sayılar (km cinsinden) |
| 4 | **Kritiklik (Faz 2)** | #5 | Isı haritası + en kritik node tablosu + "Neden kritik?" açıklamaları + worst-case tablo |
| 5 | **Simülasyon (Faz 3)** | #11 | 3 saldırı eğrisi (rastsal / statik / **adaptif**) + R indeksleri |
| 6 | **Resilience (Faz 4)** | — | Tek skor (0-100) + harf notu + 5 bileşen bar grafiği |
| 7 | **What-if (Kapanma)** | #1, #4, #7 | Manuel kavşak/yol kapat → anlık etki + Önce/Sonra paneli + izolasyon maliyeti |
| 8 | **A-B Rota** | #2 | İki nokta arası rota + kapanma sonrası uzama/kesilme |
| 9 | **Afet Senaryosu** | #13 | Deprem (episantr+yarıçap) / Sel (su seviyesi) / Bölgesel kapanma |
| 10 | **Çöküş Animasyonu** | #17 | Kademeli saldırının adım adım GIF'i |
| 11 | **İyileştirme** | #8 | "Nereye yeni yol eklersek skor en çok artar?" |
| 12 | **Bölge Analizi** | #12 | Ağı 4 bölgeye böl, en kırılganı bul |
| 13 | **Sadeleştirme** | #15 | 3 detay seviyesinde graph genelleştirme (harita genelleştirme) |
| 14 | **Karşılaştırma** | #14 | İki yol ağını yan yana metrik karşılaştırma |
| 15 | **İnteraktif Graph** | #10 | pyvis ile sürüklenebilir, hover'da detay gösteren graph |

### 4.3 PDF Rapor

8. adımda otomatik üretilir. **4 sayfalık** profesyonel PDF:

```
SAYFA 1: Başlık + Renkli Resilience Skor Kutusu + Yönetici Özeti (düz Türkçe)
         + Bölüm 1 (Yol Ağı Özeti) + counts tablosu
SAYFA 2: Bölüm 2 (Kritiklik Analizi) + ısı haritası + Articulation/Bridge sayısı
         + En kritik node tablosu + Worst-case tablosu
SAYFA 3: Bölüm 3 (Kapanma Simülasyonu) + 3 eğrili grafik + simülasyon metrikleri
         + Bölüm 4 (Resilience Skor Detayı) + bileşen kartı
SAYFA 4: Resilience bileşen tablosu (ağırlıklarla)
```

### 4.4 Backlog Özet Tablosu (17 Özellik)

| # | Özellik | Durum |
|---|---------|-------|
| 1 | What-if interaktif kapanma | ✅ |
| 2 | A-B rota etki analizi | ✅ |
| 3 | PDF yönetici özeti | ✅ |
| 4 | Before / After karşılaştırma | ✅ (What-if sekmesinde) |
| 5 | Explainability ("Neden kritik?") | ✅ |
| 6 | Demo Mode | ✅ |
| 7 | İzolasyon Maliyeti | ✅ (What-if sekmesinde) |
| 8 | Resilience iyileştirme önerisi | ✅ |
| 9 | Metre cinsinden ölçüm | ✅ |
| 10 | İnteraktif graph (pyvis) | ✅ |
| 11 | Adaptif saldırı | ✅ |
| 12 | Bölge analizi (weakest link) | ✅ |
| 13 | Afet senaryoları | ✅ |
| 14 | Karşılaştırmalı analiz | ✅ |
| 15 | Topolojik sadeleştirme | ✅ |
| 16 | Pytest test suite | ✅ (15 test) |
| 17 | Çöküş animasyonu (GIF) | ✅ |
| 18 | OSM ile doğrulama | ⚠️ Future work (veri kısıtı, bkz. Bölüm 11) |

---

## 5) Mimari / Klasör Yapısı

```
Vision2Graph/
├── main.py                          # 8 adımlı CLI pipeline
├── app.py                           # Streamlit arayüzü (15 sekme)
├── config.yaml                      # Eşikler ve ağırlıklar
├── requirements.txt                 # Python bağımlılıkları
├── pytest.ini                       # Test yapılandırması
├── README.md                        # Kısa genel bakış
├── KILAVUZ.md                       # Bu döküman
│
├── src/
│   ├── inference/
│   │   ├── dlinknet_model.py        # D-LinkNet34 mimarisi
│   │   └── infer_road_mask.py       # Model yükleme + tahmin
│   │
│   ├── topology/
│   │   ├── skeletonizer.py          # Maske → skeleton
│   │   └── graph_builder.py         # Skeleton → NetworkX graph
│   │
│   ├── analytics/
│   │   ├── centrality.py            # Betweenness centrality
│   │   ├── vulnerability.py         # Articulation point, bridge
│   │   ├── criticality.py           # Kritiklik skoru (her node)
│   │   ├── worst_case.py            # En kötü senaryo + verimlilik
│   │   ├── resilience.py            # 5 metrik → tek skor
│   │   ├── explainability.py        # "Neden kritik?" cümle üretici
│   │   ├── routing.py               # A-B en kısa rota + impact
│   │   ├── improvement.py           # Yeni yol bağlantısı önerisi
│   │   ├── regions.py               # 2x2 mekansal bölge analizi
│   │   ├── simplification.py        # Topolojik genelleştirme
│   │   └── scale.py                 # Piksel ↔ metre dönüşümü
│   │
│   ├── simulation/
│   │   ├── closure.py               # Kapanma etki motoru
│   │   ├── attack.py                # 3 strateji (random/static/adaptive)
│   │   └── disaster.py              # Deprem / sel / bölgesel preset
│   │
│   ├── visualization/
│   │   ├── graph_overlay.py         # Yol grafiği renkli çizim
│   │   ├── criticality_overlay.py   # Kritiklik ısı haritası
│   │   ├── closure_overlay.py       # Kapanma + izolasyon görseli
│   │   ├── route_overlay.py         # Önce/sonra rota
│   │   ├── regions_overlay.py       # Bölge gridi + en kırılgan
│   │   ├── improvement_overlay.py   # Önerilen yeni bağlantı
│   │   ├── degradation_plot.py      # 3 eğrili dayanıklılık grafiği
│   │   ├── resilience_card.py       # Resilience bileşen barları
│   │   └── collapse_animation.py    # Çöküş GIF üreticisi
│   │
│   └── reporting/
│       └── pdf_report.py            # PDF rapor + yönetici özeti
│
├── tests/                           # 15 birim testi
│   ├── conftest.py                  # Sentetik graph fixture
│   ├── test_analytics.py
│   ├── test_simulation.py
│   └── test_reporting.py
│
├── data/                            # DEPOSUZ — yerel olarak konur
│   └── train/                       # DeepGlobe örnek maskeleri
│
├── models/                          # DEPOSUZ — yerel olarak konur
│   └── log01_dink34.th              # D-LinkNet checkpoint
│
└── outputs/                         # DEPOSUZ — pipeline çıktıları
    ├── masks/
    ├── skeletons/
    ├── graphs/
    ├── simulations/
    └── reports/
```

> **Not:** `data/`, `models/`, `outputs/` ve `.venv/` `.gitignore` ile dışlanmıştır. Yeni bir kurulumda bu klasörlerin elle yerleştirilmesi gerekir (bkz. Bölüm 6).

---

## 6) Yeni PC'de Kurulum

### 6.1 Önkoşullar

| Bileşen | Sürüm | Not |
|---------|-------|-----|
| **Python** | 3.10 veya 3.11 | 3.12+ önerilmez (torch uyumu) |
| **pip** | en az 23.x | `python -m pip install --upgrade pip` |
| **Disk alanı** | ~3 GB | torch + model + örnek veri |
| **RAM** | 4 GB+ | CPU modu için yeterli |
| **GPU** | İsteğe bağlı | yoksa CPU kullanılır |

### 6.2 Adım Adım Kurulum

**Adım 1 — Zip'i aç**
```
Vision2Graph.zip → masaüstüne çıkart
```

**Adım 2 — Terminal aç**
```cmd
cd Vision2Graph
```

**Adım 3 — Sanal ortam oluştur**
```cmd
python -m venv .venv
.venv\Scripts\activate
```
Linux/Mac için: `source .venv/bin/activate`

**Adım 4 — Bağımlılıkları kur**
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```
> Bu adım **5-10 dakika** sürebilir (torch ~700 MB). İnternet bağlantısı şart.

**Adım 5 — Veri ve model yerleştir**

Zip'le birlikte gelmeyen iki şey lokal olarak eklenmelidir:

| Konum | İçerik | Nereden |
|-------|--------|---------|
| `data/train/` | `*_mask.png` ve `*_sat.jpg` DeepGlobe örnekleri | DeepGlobe Road Extraction Challenge veri seti |
| `models/log01_dink34.th` | D-LinkNet34 checkpoint | Orijinal D-LinkNet repo'su (releases) |

> **Eğer arkadaşa ayrıca veri/model gönderdiysek:** o klasörleri proje köküne kopyala, klasör isimleri yukarıdaki gibi olsun.

**Adım 6 — Test çalıştır**
```cmd
pytest
```
Beklenen: `15 passed in ~1s`. **Bu adım modeli ya da gerçek veriyi gerektirmez** — sentetik graph üzerinde çalışır.

**Adım 7 — Pipeline'ı dene**

Maske ile (hızlı, modelsiz):
```cmd
python main.py data/train/100129_mask.png --mask
```

Veya tam uydu görüntüsünden (model gerekli):
```cmd
python main.py data/train/100034_sat.jpg
```

**Adım 8 — Streamlit arayüzünü aç**
```cmd
streamlit run app.py
```
Tarayıcı otomatik açılır → `http://localhost:8501`

---

## 7) Test Rehberi (Adım Adım)

### 7.1 Hızlı Test (3 dakika)

| Adım | Komut / Aksiyon | Beklenen |
|------|-----------------|----------|
| 1 | `pytest` | `15 passed` |
| 2 | `python main.py data/train/100129_mask.png --mask` | 8 adım + PDF yolu |
| 3 | `outputs/reports/100129_mask_report.pdf` aç | 4 sayfa PDF |
| 4 | `streamlit run app.py` | tarayıcı açılır |
| 5 | "Hizli demo" butonu | 15 sekme + Demo Senaryo açılır |

### 7.2 Tam Test Walkthrough (15 sekme)

> Sidebar'dan "Hizli demo - ilk ornekle calistir" butonuna bas, sonra sırayla sekmelere gez.

#### Sekme 1 — Demo Senaryo
3 bölümlü otomatik senaryo: ağ durumu → kritik node kapatma → rota etkisi → sonuç kutusu. Tek bakışta bütün hikâye.

#### Sekme 2 — Girdi & Maske
Yan yana iki resim: yüklenen görüntü + işlenmiş yol maskesi.

#### Sekme 3 — Yol Grafiği
Solda skeleton (tek piksel iz), sağda renkli graph overlay. Altta 5 metrik (node, edge, component, derece, **km cinsinden uzunluk**).

#### Sekme 4 — Kritiklik (Faz 2)
- Isı haritası: yeşil = düşük, kırmızı = yüksek kritik
- Articulation point + bridge sayıları
- En kritik 5 node tablosu
- **"Neden kritik?"** — her biri için düz Türkçe cümle
- Worst-case tablosu (her kritik node kapanırsa ne olur)

#### Sekme 5 — Simülasyon (Faz 3)
- **3 eğrili grafik:** mavi=rastsal, turuncu=statik kasıtlı, **kırmızı=adaptif** kasıtlı
- 4 metrik: 3 R indeksi + kırılganlık farkı
- Adaptif eğri her zaman en sert (en küçük R)

#### Sekme 6 — Resilience (Faz 4)
- Solda büyük "Resilience skoru: X / 100"
- Harf notu (A-E) + etiket (Çok dayanıklı / Dayanıklı / Orta / Kırılgan / Çok kırılgan)
- Sağda yatay bar grafiği (5 bileşen, renkli)

#### Sekme 7 — What-if (Kapanma)
**Aksiyon:** Multiselect'ten bir kavşak seç (en kritiği veya rastgele).
**Çıktı:**
- Etki özeti (4 metrik): bileşen artışı, verim kaybı, izole kavşak, kopan ağ oranı
- Önce/Sonra paneli (3'lü metrik karşılaştırması)
- Resilience skoru kapanma öncesi vs sonrası
- Görsel: magenta X = kapatılan, kırmızı = kopan kısım

#### Sekme 8 — A-B Rota
**Aksiyon:** İki kavşak seç (A başlangıç, B hedef). İsteğe bağlı: kapatılacak kavşak ekle.
**Çıktı:** Eski rota uzunluğu, yeni rota uzunluğu, **detour faktörü**, uzama yüzdesi. KESİLDİ durumunda kırmızı uyarı.

#### Sekme 9 — Afet Senaryosu
**3 senaryo:**
- **Deprem:** episantr + yarıçap → o alandaki tüm kavşaklar kapanır
- **Sel:** su seviyesi sliderı → görüntünün altındaki yüzde kapanır
- **Bölgesel:** 4 bölgeden biri komple kapanır

#### Sekme 10 — Çöküş Animasyonu
**Aksiyon:** Kaç kavşak kaldırılacak (slider 3-30) → "Animasyonu oluştur" butonu.
**Çıktı:** Ağın adım adım yok olduğunu gösteren animasyonlu GIF, sol üstte adım sayacı.

#### Sekme 11 — İyileştirme
**Aksiyon:** "İyileştirme önerisi hesapla" butonu (~5 sn).
**Çıktı:**
- Mevcut skor + önerilen yeni bağlantı (örn. *"Node 9 → Node 6 eklenirse skor 35 → 45 (+10)"*)
- Aday bağlantılar tablosu (her birinin getirisi)
- Görsel: önerilen bağlantı yeşil çizgi

#### Sekme 12 — Bölge Analizi
**Aksiyon:** "Bölge analizini çalıştır" butonu (~5 sn).
**Çıktı:**
- 4 bölgenin (Kuzeybatı/Kuzeydoğu/Güneybatı/Güneydoğu) ayrı resilience skorları
- En kırılgan bölge kırmızı uyarıyla
- Harita: 2×2 grid + en kırılgan bölgenin node'ları kırmızı

#### Sekme 13 — Sadeleştirme
**3 detay seviyesi:**
- **Seviye 1:** Tüm yollar (%100)
- **Seviye 2:** Ana + önemli yollar (%60)
- **Seviye 3:** Sadece kritik omurga (%30)

Radio ile seçim → o seviyenin overlay'i görünür.

#### Sekme 14 — Karşılaştırma
**Aksiyon:** İkinci bir örnek seç (selectbox) → "Karşılaştır" butonu.
**Çıktı:** 9 satırlık karşılaştırma tablosu + hangi ağın daha dayanıklı olduğu verdict'i + iki graph yan yana.

#### Sekme 15 — İnteraktif Graph
- Sürüklenebilir, zoom'lanabilir node-link diyagramı
- Bir node'un üstüne hover → tooltip: ID, kritiklik, derece, betweenness
- Altta selectbox ile kavşak detayı

---

## 8) Çıktılar (Outputs)

`main.py` çalıştığında `outputs/` altında şu klasörler oluşur:

```
outputs/
├── masks/           {ad}_mask.png            (binary yol maskesi)
├── skeletons/       {ad}_skeleton.png        (tek piksel iz)
├── graphs/          {ad}_graph.png           (renkli graph overlay)
│                    {ad}_criticality.png     (ısı haritası)
├── simulations/     {ad}_degradation.png     (3 eğrili saldırı grafiği)
│                    {ad}_simulation.json     (her adımın sayıları)
└── reports/         {ad}_analysis.json       (Faz 2 analiz sonucu)
                     {ad}_resilience.json     (skor + bileşenler)
                     {ad}_resilience.png      (bar grafiği)
                     {ad}_report.pdf          (derlenmiş 4 sayfalık rapor)
```

> Tüm JSON'lar `ensure_ascii=False` + UTF-8 ile yazılır — Türkçe metinler düzgün korunur.

---

## 9) Konfigürasyon (config.yaml)

Tüm eşikler ve ağırlıklar `config.yaml` üzerinden ayarlanır. **Hiçbir şey kod içinde sabit değil** — jüriye savunulabilir olması için her parametre dışarıdan kontrol edilebilir.

```yaml
paths:
  model_checkpoint: models/log01_dink34.th
  data_dir: data/train
  output_dir: outputs

inference:
  device: auto          # auto | cpu | cuda
  threshold: 0.5        # sigmoid eşiği

skeleton:
  min_object_size: 64   # küçük lekeleri ele
  closing_kernel: 3     # morfolojik closing

graph:
  spur_length_threshold: 15   # kısa sarkan kenarları buda
  min_component_length: 30    # küçük bileşenleri at

scale:
  meters_per_pixel: 0.5       # null = piksel; sayı = metre

analytics:
  criticality:
    node_weights:
      betweenness: 0.50
      articulation: 0.35
      degree: 0.15
    edge_weights:
      betweenness: 0.60
      bridge: 0.40
  worst_case_top_k: 5

simulation:
  max_fraction: 0.5      # node'ların en fazla %50'si kaldırılır
  random_runs: 5         # rastsal eğri kaç koşunun ortalaması

resilience:
  weights:
    largest_component_ratio: 0.30
    avg_degree: 0.15
    bridge_ratio: 0.20
    articulation_ratio: 0.20
    targeted_robustness: 0.15
```

---

## 10) Teknik Detaylar

### 10.1 Kritik Metrikler

| Metrik | Anlamı | Aralık |
|--------|--------|--------|
| **Betweenness centrality** | Bu node'dan kaç en kısa yol geçer | 0-1 (normalize) |
| **Articulation point** | Kaldırılırsa ağ bileşene bölünür mü | 0 / 1 |
| **Bridge edge** | Kaldırılırsa bileşen ikiye bölünür mü | 0 / 1 |
| **LCR (Largest Component Ratio)** | En büyük bileşendeki node'ların oranı | 0-1 |
| **Global efficiency** | Tüm node çiftleri arası ortalama 1/uzaklık | 0-1 |
| **Robustness Index (R)** | LCR eğrisi altındaki normalize alan | 0-1 |
| **Detour factor** | Rota uzunluğu / kuş uçuşu mesafe | ≥ 1 |

### 10.2 Resilience Skor Formülü

```
score = 100 × Σ ( weight_i × component_i )

components:
  LCR                        → kullan olduğu gibi
  avg_degree                 → (derece - 1) / (4 - 1), [0,1]'e kelep
  bridge_ratio               → 1 - bridge_oranı
  articulation_ratio         → 1 - articulation_oranı
  targeted_robustness        → R (statik kasıtlı saldırı)
```

| Skor | Harf | Etiket |
|------|------|--------|
| 80-100 | A | Çok dayanıklı |
| 60-80 | B | Dayanıklı |
| 40-60 | C | Orta |
| 20-40 | D | Kırılgan |
| 0-20 | E | Çok kırılgan |

### 10.3 Saldırı Stratejileri (Faz 3)

| Strateji | Mantık | Kullanım |
|----------|--------|----------|
| **Rastsal arıza** | Her adımda rastgele node sil (5 koşu ortalaması) | Dogal arıza modeli |
| **Kasıtlı statik** | En kritik node listesi baştan hesaplanır, sırayla silinir | Saldırgan ön bilgi varsayımı |
| **Kasıtlı adaptif** | Her silmeden sonra kritiklik **yeniden hesaplanır** | Akademik altın standart — en sert |

**Beklenen sıralama:** R_adaptif < R_statik < R_rastsal (sayı küçükse ağ o saldırıya daha çok çöker).

### 10.4 Kullanılan Kütüphaneler

| Amaç | Kütüphane |
|------|-----------|
| Derin öğrenme | `torch`, `torchvision` |
| Görüntü işleme | `opencv-python`, `scikit-image` |
| Graph algoritmaları | `networkx`, `sknw` |
| Görselleştirme | `matplotlib`, `pyvis`, `Pillow` |
| Web arayüz | `streamlit` |
| PDF | `reportlab` |
| Konfigürasyon | `pyyaml` |
| Test | `pytest` |

---

## 11) Bilinen Sınırlar

### 11.1 DeepGlobe Veri Kısıtı

DeepGlobe uydu karoları **coğrafi koordinat içermez** — anonimleştirilmiş 1024×1024 piksel kareler. Bu yüzden:

- `meters_per_pixel: 0.5` yaklaşık bir değerdir (DeepGlobe rapor edilen çözünürlük).
- **OSM (OpenStreetMap) doğrulama yapılmamıştır** — koordinatsız kareyi gerçek dünyadaki bir bölgeyle eşleştirmek mümkün değil. Bu yüzden #18 backlog maddesi **future work** olarak bırakılmıştır. (Alternatif: ground-truth maskeleri ile piksel IoU — eklenebilir).

### 11.2 Model Boyutu

D-LinkNet34 checkpoint'i ~85 MB. Bu nedenle:
- GitHub deposuna dahil edilmedi.
- Zip ile arkadaşa ayrıca gönderilmelidir.
- Sadece "maske ile" çalıştırırken model gerekmez (`--mask` flag).

### 11.3 Performans Notları

| İşlem | Süre (30 node'lu graph) | Süre (300 node'lu graph) |
|-------|--------------------------|---------------------------|
| Pipeline tam (CLI) | ~3 sn | ~10 sn |
| Streamlit ilk yükleme | ~3 sn | ~15 sn |
| What-if (canlı) | < 1 sn | ~2 sn |
| İyileştirme (12 aday) | ~5 sn | ~20 sn |
| Çöküş GIF (12 frame) | ~3 sn | ~10 sn |

> Uydu görüntüsünden çıkarım CPU'da ~20 sn sürer. GPU varsa anlık.

---

## 12) Sunum İçin Önerilen Akış

Jüriye **15 dakikalık sunum** önerisi:

```
00:00-02:00  Problemi anlat — "Yolu bulmak değil, anlamlandırmak"
02:00-03:00  Streamlit'i aç + "Hizli demo" butonuna bas
03:00-05:00  Demo Senaryo sekmesini gez (otomatik 3 adım)
05:00-08:00  Kritiklik tab + "Neden kritik?" açıklamaları (savunulabilirlik)
08:00-10:00  Simülasyon — 3 eğrili grafik (adaptif neden daha sert?)
10:00-12:00  What-if veya Afet senaryosu — interaktif demo
12:00-14:00  İyileştirme önerisi — karar destek vurgusu
14:00-15:00  PDF raporu aç + Yönetici Özeti'ni göster
```

**Önemli güçlü yanlar:**
- "Sadece grafik çiziyor" iddiasına karşı: **otomatik metin açıklaması** + **somut öneri**
- "Sayılar ne anlama geliyor" sorusuna karşı: **km cinsinden** çıktılar + **harf notu**
- "Sahte mi" şüphesine karşı: **3 farklı saldırı stratejisinin tutarlı sıralaması** + **15 birim testi**
- "Pratik mi" sorusuna karşı: **deprem/sel preset'leri** + **iyileştirme önerisi**

---

## 13) Sık Karşılaşılan Sorunlar

### Sorun 1 — `torch` DLL hatası (Windows)
```
ImportError: DLL load failed while importing _C
```
**Çözüm:** Microsoft Visual C++ Redistributable yükle ([buradan](https://aka.ms/vs/17/release/vc_redist.x64.exe)).

### Sorun 2 — Streamlit 8501 portu meşgul
```
Port 8501 is already in use
```
**Çözüm:** Farklı port ile çalıştır:
```
streamlit run app.py --server.port 8502
```

### Sorun 3 — Model checkpoint eksik
```
FileNotFoundError: models/log01_dink34.th
```
**Çözüm 1:** Model dosyasını `models/` altına yerleştir.
**Çözüm 2:** Maske moduyla çalıştır (model gerekmez):
```
python main.py data/train/100129_mask.png --mask
```

### Sorun 4 — Sample dosya bulunamadı
```
data/train altında ornek dosya bulunamadi
```
**Çözüm:** `data/train/` klasörüne en az bir `*_mask.png` veya `*_sat.jpg` koy.

### Sorun 5 — PDF açılmıyor / bozuk
**Çözüm:** main.py'ı yeniden çalıştır, `outputs/reports/` altındaki eski PDF'i sil, yeniden üretilsin.

### Sorun 6 — pyvis interaktif graph siyah görünüyor
**Çözüm:** İlk açıldığında fizik motoru node'ları yerleştirirken birkaç saniye sürer. 5 saniye bekle veya graph'ı sürükleyerek hareketlendir.

---

## 14) Kaynaklar

- **GitHub deposu:** https://github.com/YusufKaramuk1/Vision2Graph
- **D-LinkNet modeli:** Zhou, L. et al. (2018), *D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction* (CVPR Workshop)
- **DeepGlobe veri seti:** https://competitions.codalab.org/competitions/18467
- **NetworkX:** https://networkx.org/
- **Resilience metrikleri:** Latora & Marchiori (2001), *Efficient Behavior of Small-World Networks*

---

## Hızlı Referans Kartı

```
KURULUM:        python -m venv .venv && .venv\Scripts\activate
                pip install -r requirements.txt

CLI PIPELINE:   python main.py data/train/100129_mask.png --mask
STREAMLIT:      streamlit run app.py
TESTLER:        pytest

ÖNEMLİ DOSYA:   config.yaml         (tüm parametreler)
                outputs/reports/*.pdf (otomatik raporlar)
                KILAVUZ.md          (bu döküman)

ZIP'E DAHİL ET:  Vision2Graph/
                + models/log01_dink34.th      (~85 MB, ayrı)
                + data/train/*.png/jpg        (örnekler, ayrı)
```

---

*Vision2Graph — bitirme projesi, 2026.
Tüm kaynak kodu MIT lisansı altındadır.
Kapsamlı kılavuz: v1.0*
