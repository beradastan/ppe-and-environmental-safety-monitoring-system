# Factory Safety Monitoring System

Endüstriyel ortamlarda gerçek zamanlı KKD (Kişisel Koruyucu Donanım) denetimi ve yangın tespiti yapan yapay zeka destekli güvenlik izleme sistemi.

## İçindekiler

- [Özellikler](#özellikler)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Nasıl Çalışır](#nasıl-çalışır)
- [API Referansı](#api-referansı)
- [Yapılandırma](#yapılandırma)
- [Testler](#testler)
- [Sorun Giderme](#sorun-giderme)

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| **Çift Tespit Modu** | Crop-based ve scene-based PPE analizi, benchmark ile seçilebilir |
| **Kişi Takibi** | YOLO11n + ByteTrack ile stabil track ID ve kayıp kimlik yeniden atama |
| **PPE Tespiti** | Baret, yelek ve maske için ayrı fine-tuned YOLOv8 modelleri |
| **Temporal Oylama** | Deque tabanlı oylama ile geçici yanlış pozitif azaltma |
| **Yangın/Duman Tespiti** | Alan büyümesi analizi ile güvenilir yangın alarmı |
| **Event Sistemi** | Durum makinesi tabanlı olay yönetimi (new → active → resolved) |
| **LLM Raporlama** | Ollama (qwen3:8b) ile otomatik günlük/haftalık/aylık güvenlik raporları |
| **Web Dashboard** | React 18 + Vite, Socket.IO ile gerçek zamanlı güncelleme |
| **REST API** | Flask Blueprint mimarisi, 15+ endpoint |
| **Çift Depolama** | PostgreSQL öncelikli, bağlantı yoksa dosya sistemi yedeği |
| **Kamera İzleme** | Çevrimdışı / donuk / karanlık kare tespiti ve bildirim |

---

## Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                     │
│   Dashboard · AlertHistory · CameraSetup · Reports       │
└────────────────────┬────────────────────────────────────┘
                     │ Socket.IO + REST
┌────────────────────▼────────────────────────────────────┐
│                  Backend (Flask)                         │
│  routes/events · routes/reports · routes/pipeline        │
│  routes/settings · watcher · database                    │
└────────────────────┬────────────────────────────────────┘
                     │ subprocess
┌────────────────────▼────────────────────────────────────┐
│                 Pipeline (CV)                            │
│  run_live_video → ppe_processor → event_manager          │
│  fire_smoke_detector · camera_monitor · visualizer       │
└─────────────────────────────────────────────────────────┘
```

### Proje Yapısı

```
.
├── pipeline/                   # Tespit pipeline'ı
│   ├── run_live_video.py       # Giriş noktası — kamera döngüsü
│   ├── config.py               # Sabitler ve model yolları
│   ├── ppe_processor.py        # PPEProcessor (crop/scene dispatch)
│   ├── ppe_detector.py         # Yardımcı fonksiyonlar: oylama, crop
│   ├── event_manager.py        # Olay durum makinesi
│   ├── event_io.py             # Olay kaydetme / kapatma / bildirim
│   ├── fire_smoke_detector.py  # Yangın/duman alan analizi
│   ├── camera_monitor.py       # Donuk/karanlık kare tespiti
│   ├── tracking_identity.py    # TrackReattacher — kayıp ID yeniden atama
│   └── visualizer.py           # HUD çizim yardımcıları
│
├── backend/                    # Flask + Socket.IO API
│   ├── app.py                  # Uygulama fabrikası, Blueprint kayıt
│   ├── config_manager.py       # Yapılandırma yükleme, DB/dosya geçişi
│   ├── extensions.py           # SocketIO singleton
│   ├── event_reader.py         # Dosya sistemi event okuyucu
│   ├── watcher.py              # Dosya sistemi değişiklik izleyici
│   ├── routes/
│   │   ├── events.py           # GET/POST /api/events, stats, görseller
│   │   ├── reports.py          # Raporlar, LLM özeti, CSV/PDF export
│   │   ├── pipeline.py         # Pipeline başlat/durdur, stream
│   │   └── settings.py         # GET/PUT /api/config
│   └── database/               # PostgreSQL bağlayıcı (isteğe bağlı)
│
├── llm/                        # LLM raporlama modülü
│   ├── llm_coordinator.py      # Ollama istemcisi
│   └── safety_report_agent.py  # Rapor üretim mantığı
│
├── frontend/                   # React 18 + Vite dashboard
│   └── src/
│       ├── pages/              # Dashboard, AlertHistory, CameraSetup, Reports, Settings
│       └── components/         # EventCard, Timeline, Sidebar, Navbar...
│
├── config/
│   └── bytetrack.yaml          # ByteTrack izleyici parametreleri
├── models/                     # YOLO ağırlık dosyaları (.pt)
├── results/                    # Kaydedilen olay verileri
├── tests/                      # pytest test suite
├── config.yaml                 # Ana yapılandırma
├── requirements.txt
└── start.py                    # Tek komutla başlatıcı
```

---

## Kurulum

### Gereksinimler

- Python 3.10+
- Node.js 18+
- CUDA 11.8+ *(isteğe bağlı, GPU hızlandırma)*
- PostgreSQL 14+ *(isteğe bağlı)*
- Ollama *(isteğe bağlı, LLM raporlama)*

### 1. Python Ortamı

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

### 3. Yapılandırma

`config.yaml` dosyasını düzenleyin:

```yaml
database:
  enabled: false        # PostgreSQL yoksa false

llm:
  enabled: true
  model: qwen3:8b       # ollama pull qwen3:8b

models:
  device: cuda          # veya cpu
  person_model: models/person_agent_scene_vinayakstyle_best.pt
  crop:
    helmet_model: models/bera/crophelmet_agent_final_best.pt
    vest_model:   models/bera/cropvest_agent_final_best.pt
    mask_model:   models/bera/cropmask_agent_final_best.pt
```

---

## Çalıştırma

### Tek Komutla (Önerilen)

```bash
python start.py
```

Backend `http://localhost:5050`, frontend `http://localhost:5173` adresinde başlar.

### Manuel

```bash
# Backend
python -m backend.app

# Frontend (ayrı terminal)
cd frontend && npm run dev
```

### Pipeline (Canlı Kamera)

```bash
# Crop modu, varsayılan kamera
python pipeline/run_live_video.py --mode crop --camera 0

# Scene modu, görüntü penceresi ile
python pipeline/run_live_video.py --mode scene --display

# Kamera ID ve bölge bilgisiyle
python pipeline/run_live_video.py --camera 1 --camera-id cam_01 --zone "Üretim Hattı A"

# CPU zorla
python pipeline/run_live_video.py --device cpu
```

Web arayüzünden başlatmak için **CameraSetup** sayfasını kullanın.

---

## Nasıl Çalışır

### Tespit Pipeline'ı

Her video karesinde:

1. **Kişi tespiti** — YOLO11n + ByteTrack, stabil track ID üretir
2. **PPE analizi** — Her kişi için baret/yelek/maske crop'u alınır, ayrı modelle sınıflandırılır
3. **Temporal oylama** — Son N frame üzerinden majority vote ile kararlı ihlal kararı
4. **Yangın tespiti** — Ayrı model + alan büyümesi filtresi ile yanlış pozitif azaltma
5. **Event yönetimi** — Durum makinesi ile olaylar oluşturulur, güncellenir, kapatılır

### Event Durum Makinesi

```
idle
 └─ Alarm (2 ardışık frame) ──→ new    ──→ Dosya kaydedilir
                                  │
                          Alarm sürüyor
                          ├─ Değişim yok ──→ active   (kayıt yok)
                          └─ Değişim var ──→ update   ──→ Dosya kaydedilir
                          
                          Alarm kayboldu (resolved_confirm_sec)
                                  └─────────────────→ resolved ──→ Dosya kaydedilir
```

| Durum | Kayıt | Açıklama |
|---|---|---|
| `idle` | — | Alarm yok |
| `new` | ✅ | Yeni olay açıldı |
| `active` | — | Olay sürüyor, değişim yok |
| `update` | ✅ | İhlal sayısı veya tipi değişti |
| `resolved` | ✅ | Olay kapandı |

### Olay Dosya Yapısı

```
results/
  evt_0042/
    evt_0042_new.jpg          # Başlangıç ekran görüntüsü
    evt_0042_new.json         # Olay verisi (JSON)
    evt_0042_update_01.json   # Güncelleme (ihlal değişti)
    evt_0042_resolved.json    # Kapanış kaydı
```

**Örnek JSON:**
```json
{
  "event_id": "evt_0042",
  "event_status": "new",
  "timestamp": "2026-06-13T10:30:00.000000",
  "camera_id": "cam_01",
  "zone": "Üretim Hattı A",
  "violations": ["no_helmet", "no_vest"],
  "person_count": 3,
  "duration_sec": 0.0
}
```

### Socket.IO Olayları

| Olay | Yön | Açıklama |
|---|---|---|
| `new_event` | Server → Client | Yeni ihlal olayı |
| `event_closed` | Server → Client | Olay kapandı |
| `camera_status` | Server → Client | Kamera durumu değişti |
| `report_llm_ready` | Server → Client | LLM raporu hazır |
| `report_llm_error` | Server → Client | LLM raporu başarısız |

---

## API Referansı

### Olaylar

| Yöntem | Uç Nokta | Açıklama |
|---|---|---|
| GET | `/api/events` | Olay listesi (filtre: tarih, tür, durum) |
| GET | `/api/events/<id>` | Olay detayı ve zaman çizelgesi |
| POST | `/api/events/<id>/note` | Olaya not ekle |
| PATCH | `/api/events/<id>/close` | Olayı manuel kapat |
| PATCH | `/api/events/<id>/false-positive` | Yanlış pozitif işaretle |
| GET | `/api/stats` | Aktif alarmlar, bugünkü ihlaller, toplam |

### Raporlar

| Yöntem | Uç Nokta | Açıklama |
|---|---|---|
| GET | `/api/reports` | Periyodik raporlar (daily/weekly/monthly) |
| POST | `/api/reports/summary/llm` | LLM ile rapor özeti üret (async) |
| GET | `/api/reports/export/csv` | CSV dışa aktarım |

### Pipeline

| Yöntem | Uç Nokta | Açıklama |
|---|---|---|
| GET | `/api/pipeline/status` | Pipeline çalışıyor mu? |
| POST | `/api/pipeline/start` | Pipeline başlat (`source`, `camera_id`, `zone`, `mode`) |
| POST | `/api/pipeline/stop` | Pipeline durdur |
| POST | `/api/pipeline/camera-status` | Kamera durum bildirimi |
| GET | `/api/stream/frame` | Canlı kamera karesi (JPEG) |

### Yapılandırma

| Yöntem | Uç Nokta | Açıklama |
|---|---|---|
| GET | `/api/config` | Mevcut yapılandırma |
| PUT | `/api/config` | Yapılandırma güncelle |

---

## Yapılandırma

`config.yaml` veya `/api/config` endpoint'i üzerinden değiştirilebilir.

### Tespit Parametreleri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `use_helmet` | `true` | Baret kontrolü |
| `use_vest` | `true` | Yelek kontrolü |
| `use_mask` | `true` | Maske kontrolü |
| `use_fire` | `true` | Yangın tespiti |
| `person_conf` | `0.25` | Kişi tespiti güven eşiği |
| `crop_helmet_conf` | `0.20` | Crop modu baret eşiği |
| `crop_vest_conf` | `0.30` | Crop modu yelek eşiği |
| `crop_mask_conf` | `0.25` | Crop modu maske eşiği |
| `scene_helmet_conf` | `0.25` | Scene modu baret eşiği |
| `scene_vest_conf` | `0.30` | Scene modu yelek eşiği |
| `temporal_window` | `20` | Crop oylama pencere boyutu (frame) |
| `scene_temporal_window` | `30` | Scene oylama pencere boyutu (frame) |
| `resolved_confirm_sec` | `5` | Olay kapanma bekleme süresi (sn) |

### Yangın Filtreleri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `fire_conf` | `0.50` | Yangın güven eşiği |
| `fire_min_area_ratio` | `0.027` | Minimum alan / frame oranı |
| `fire_growth_factor` | `1.5` | Alan büyüme hızı çarpanı |
| `fire_growth_window` | `10` | Büyüme analiz penceresi (frame) |

### Kamera İzleme

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `cam_freeze_frames` | `60` | Donuk kare sayısı eşiği |
| `cam_freeze_diff` | `0.002` | Kare fark eşiği |
| `cam_dark_frames` | `60` | Karanlık kare sayısı eşiği |
| `cam_dark_thresh` | `0.03` | Parlaklık eşiği |

---

## LLM Raporlama

```bash
# Ollama kur: https://ollama.ai
ollama pull qwen3:8b
ollama serve
```

`config.yaml` içinde `llm.enabled: true` ile etkinleştirin. Backend otomatik olarak günlük, haftalık ve aylık güvenlik raporları üretir. Raporlar web dashboard üzerinden görüntülenebilir ve CSV/PDF olarak dışa aktarılabilir.

---

## Testler

Backend çalışırken:

```bash
python tests/test_api.py
```

API uçtan uca testi 15+ endpoint'i kontrol eder ve sonuçları raporlar.

---

## Sorun Giderme

**Backend başlamıyor:**
```bash
python -m backend.app
```

**CUDA bellek hatası:**
```bash
python pipeline/run_live_video.py --device cpu
```

**Veritabanı bağlantı hatası:**
`config.yaml` içinde `database.enabled: false` yapın — sistem otomatik olarak dosya sistemine geçer.

**Kamera açılamıyor:**
`--camera` argümanıyla doğru indeksi belirtin (`0`, `1`, `2`...).

**LLM yanıt vermiyor:**
```bash
# Ollama çalışıyor mu?
curl http://localhost:11434/api/tags

# config.yaml içinde devre dışı bırakın
llm:
  enabled: false
```

---

## Lisans

Akademik amaçlar için geliştirilmiş mezuniyet projesidir.
