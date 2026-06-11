# Fabrika İş Güvenliği Tespit Sistemi

Endüstriyel ortamlarda gerçek zamanlı KKD (Kişisel Koruyucu Donanım) ve yangın tespiti yapan yapay zeka destekli güvenlik izleme sistemi. YOLO11n kişi takibi, ByteTrack, çift tespit modu (crop/scene), temporal oylama ve LLM tabanlı periyodik raporlama içerir.

## Özellikler

- **Çift Tespit Modu**: Kırpma tabanlı (crop) ve sahne tabanlı (scene) PPE analizi
- **Kişi Takibi**: YOLO11n + ByteTrack ile stabil track ID yönetimi
- **PPE Tespiti**: Baret, yelek ve maske için ayrı YOLOv8 modelleri
- **Temporal Oylama**: Deque tabanlı geçici yanlış pozitif azaltma
- **Yangın/Duman Tespiti**: Alan büyümesi analizi ile güvenilir alarm
- **LLM Raporlama**: Ollama (qwen3:8b) ile otomatik günlük/haftalık/aylık güvenlik raporları
- **Web Arayüzü**: React 18 + Vite dashboard, Socket.IO gerçek zamanlı güncelleme
- **REST API**: 22 endpoint, Flask Blueprint mimarisi
- **Çift Depolama**: PostgreSQL veritabanı, bağlantı yoksa dosya sistemi yedeği

## Sistem Gereksinimleri

- Python 3.10+
- CUDA 11.8+ (isteğe bağlı, GPU hızlandırma için)
- Node.js 18+ (frontend için)
- PostgreSQL 14+ (isteğe bağlı)
- Ollama (LLM raporlama için isteğe bağlı)

## Kurulum

### 1. Python Ortamı

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

### 3. Yapılandırma

`config.yaml` dosyasında gerekli ayarları yapın:

```yaml
backend:
  host: 0.0.0.0
  port: 5050

database:
  enabled: false   # PostgreSQL yoksa false bırakın

llm:
  enabled: true
  model: qwen3:8b  # ollama pull qwen3:8b

models:
  device: cuda     # veya cpu
  person_model: models/person_agent_scene_vinayakstyle_best.pt
  crop:
    helmet_model: models/bera/crophelmet_agent_final_best.pt
    vest_model:   models/bera/cropvest_agent_final_best.pt
    mask_model:   models/bera/cropmask_agent_final_best.pt
  scene:
    helmet_model: models/vinayak_trained_byBera/helmet_agent_final_best.pt
    vest_model:   models/vinayak_trained_byBera/vest_agent_final_best.pt
    mask_model:   models/mask_agent_scene_200ep_yolov8m_best.pt
```

## Çalıştırma

### Tek Komutla (Backend + Frontend)

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
# Crop modu (varsayılan) — kamera 0
python pipeline/run_live_video.py

# Scene modu
python pipeline/run_live_video.py --mode scene

# Görüntü penceresi ile
python pipeline/run_live_video.py --display

# Belirli kamera ve bölge
python pipeline/run_live_video.py --camera 1 --camera-id cam_01 --zone "Üretim Hattı A"
```

## Proje Yapısı

```
.
├── pipeline/                   # Tespit pipeline'ı (9 modül)
│   ├── run_live_video.py       # Giriş noktası — kamera döngüsü
│   ├── config.py               # Sabitler ve model yolları
│   ├── ppe_processor.py        # PPEProcessor (crop/scene dispatch)
│   ├── ppe_detector.py         # Saf fonksiyonlar: oylama, crop, sahne atama
│   ├── event_manager.py        # Olay durum makinesi
│   ├── event_io.py             # Olay kaydetme / kapatma / bildirim
│   ├── fire_smoke_detector.py  # Yangın/duman alan analizi
│   ├── camera_monitor.py       # Donuk/karanlık kare tespiti
│   ├── tracking_identity.py    # TrackReattacher — kayıp ID yeniden atama
│   └── visualizer.py           # Çizim yardımcıları
│
├── backend/                    # Flask + Socket.IO API
│   ├── app.py                  # Uygulama fabrikası, Blueprint kayıt
│   ├── config_manager.py       # Yapılandırma yükleme, DB/dosya geçişi
│   ├── extensions.py           # SocketIO singleton
│   ├── event_reader.py         # Dosya sistemi event okuyucu
│   ├── routes/
│   │   ├── events.py           # GET/POST /api/events, stats, görseller
│   │   ├── reports.py          # Raporlar, LLM özeti, CSV/PDF dışa aktarım
│   │   ├── pipeline.py         # Pipeline başlat/durdur, stream
│   │   └── settings.py         # GET/PUT /api/config
│   └── database/               # PostgreSQL bağlayıcı (isteğe bağlı)
│
├── frontend/                   # React 18 + Vite dashboard
│   └── src/
│
├── config/
│   └── bytetrack.yaml          # ByteTrack izleyici yapılandırması
├── models/                     # YOLO ağırlık dosyaları (.pt)
├── results/                    # Kaydedilen olay verileri
├── tests/
│   ├── test_api.py             # Backend API uçtan uca testi (27 kontrol)
│   ├── test_pipeline_crop.py   # Crop modu tespit doğrulaması
│   └── test_pipeline_scene.py  # Scene modu tespit doğrulaması
├── config.yaml                 # Ana yapılandırma
├── requirements.txt
└── start.py                    # Tek komutla başlatıcı
```

## Testler

Backend çalışırken:

```bash
# API uçtan uca testi
python tests/test_api.py

# Crop modu pipeline testi (3 video, 200 frame)
python tests/test_pipeline_crop.py --max-frames 200

# Scene modu pipeline testi
python tests/test_pipeline_scene.py --max-frames 200
```

Test videoları `test/` dizininde olmalı: `nohat_test.mp4`, `novest_test.mp4`, `noppe_test.mp4`

## API Uç Noktaları

| Yöntem | Uç Nokta | Açıklama |
|--------|----------|----------|
| GET | `/api/events` | Olayları listele (filtre: tarih, tür, durum) |
| GET | `/api/events/<id>` | Olay zaman çizelgesi ve notlar |
| POST | `/api/events/<id>/note` | Not ekle |
| PATCH | `/api/events/<id>/close` | Olayı kapat |
| PATCH | `/api/events/<id>/false-positive` | Yanlış pozitif işaretle |
| GET | `/api/stats` | Aktif alarmlar, bugünkü ihlaller, toplam |
| GET | `/api/reports` | Periyodik raporlar (daily/weekly/monthly) |
| POST | `/api/reports/summary/llm` | LLM ile rapor özeti oluştur |
| GET | `/api/reports/export/csv` | CSV dışa aktarım |
| GET | `/api/pipeline/status` | Pipeline çalışma durumu |
| POST | `/api/pipeline/start` | Pipeline başlat |
| POST | `/api/pipeline/stop` | Pipeline durdur |
| GET | `/api/stream/frame` | MJPEG kare akışı |
| GET | `/api/config` | Mevcut yapılandırma |
| PUT | `/api/config` | Yapılandırma güncelle |

## Yapılandırma Parametreleri

`/api/config` veya `config.yaml` → `ppe_pipeline` bölümü üzerinden değiştirilebilir:

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `use_helmet` | `true` | Baret kontrolü aktif |
| `use_vest` | `true` | Yelek kontrolü aktif |
| `use_mask` | `true` | Maske kontrolü aktif |
| `use_fire` | `true` | Yangın tespiti aktif |
| `crop_helmet_conf` | `0.20` | Crop modu baret güven eşiği |
| `crop_vest_conf` | `0.30` | Crop modu yelek güven eşiği |
| `scene_helmet_conf` | `0.25` | Scene modu baret güven eşiği |
| `temporal_window` | `20` | Crop modu oylama pencere boyutu |
| `scene_temporal_window` | `30` | Scene modu oylama pencere boyutu |
| `person_conf` | `0.25` | Kişi tespiti güven eşiği |

## LLM Raporlama

Ollama kurulumu ve model indirme:

```bash
# https://ollama.ai adresinden Ollama'yı kur
ollama pull qwen3:8b
ollama serve
```

`config.yaml` içinde `llm.enabled: true` ayarlandığında backend otomatik olarak günlük, haftalık ve aylık güvenlik raporları oluşturur.

## Sorun Giderme

**Backend bağlantı hatası:**
```bash
# Backend çalışıyor mu kontrol et
python -m backend.app
```

**CUDA bellek hatası:**
```bash
# CPU moduna geç
python pipeline/run_live_video.py --device cpu
```

**ByteTrack dosyası bulunamadı:**
`config/bytetrack.yaml` dosyasının mevcut olduğundan emin olun.

**Veritabanı bağlantısı kurulamıyor:**
`config.yaml` içinde `database.enabled: false` yapın — sistem otomatik olarak dosya sistemine geçer.

## Lisans

Akademik amaçlar için hazırlanmış mezuniyet projesidir.
