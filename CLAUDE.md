# Factory Safety — CLAUDE.md (unified-detection branch)

## Proje Özeti
Fabrika içi iş güvenliği izleme sistemi. YOLO + ByteTrack ile PPE (baret/yelek/maske) ve yangın tespiti yapıp ihlalleri kaydeder, raporlar ve web arayüzü üzerinden gösterir.

Bu branch: **unified detection** — frontend'den `crop` veya `scene` modu seçilir, `run_live_video.py --mode crop|scene` ile pipeline başlar.

## Teknoloji Yığını
- **Backend:** Flask + Flask-SocketIO, Python 3.9 (`.venv`), `from __future__ import annotations` gerekli
- **Frontend:** React + Vite (`frontend/`), port 5173/5174
- **DB:** PostgreSQL — host: localhost:5432, db: ppe_db, user: postgres, password: 1234
- **Detection:** `run_live_video.py` — `--mode crop|scene` ile seçilebilir, ByteTrack + TrackReattacher
- **LLM:** Ollama — model `config.yaml` `llm.model`'dan okunur (varsayılan qwen3:8b), her iki modda da etkin
- **GPU:** RTX 3060 6GB

## Çalıştırma
```bash
# Backend (port 5050)
python -m backend.app

# Frontend
cd frontend && npm run dev

# Detection pipeline — crop modu (varsayılan)
python run_live_video.py --mode crop --camera 0 --camera-id cam_01 --zone "Üretim Hattı A" --display

# Detection pipeline — scene modu
python run_live_video.py --mode scene --camera 0 --camera-id cam_01 --zone "Üretim Hattı A" --display
```

## Detection Mimarileri

### Crop-Based (`--mode crop`)
```
frame
 └─ person_model.track()         → kişi bbox + ByteTrack ID
     └─ TrackReattacher.update() → stable_pid
         └─ crop_ppe(frame, bbox, "helmet") → baş bölgesi kırpılır
         └─ helmet_model.predict(crop)      → kırpık üzerinde tespit
         └─ crop_ppe(frame, bbox, "vest")   → gövde bölgesi kırpılır
         └─ vest_model.predict(crop)        → kırpık üzerinde tespit
         └─ (mask için de aynı)
         └─ _validate_ppe_scored()          → geometrik doğrulama
         └─ _global_assign_ppe()            → one-to-one greedy atama
```

### Scene-Based (`--mode scene`)
```
frame
 └─ person_model.track()         → kişi bbox + ByteTrack ID
     └─ TrackReattacher.update() → stable_pid
         └─ _scene_dets(helmet_model, frame) → tam kareye tespit
         └─ _best_scene(dets, person_box)    → _inside_frac >= 0.40 olan en iyi eşleşme
         └─ (vest ve mask için de aynı)
```

**Her iki modda ortak:** person tracking, fire detection, temporal voting, event state machine, save_event, LLM, DB yazımı, kamera izleme.

## Kesinleşmiş Model Dosyaları (git'e dahil değil)
```
models/person_agent_scene_vinayakstyle_best.pt   — kişi tespiti (her iki modda ortak)
models/bera/
  fire_smoke_other_agent_final_best.pt           — yangın/duman (her iki modda ortak)
  crophelmet_agent_final_best.pt                 — baret (crop modu)
  cropvest_agent_final_best.pt                   — yelek (crop modu, compare_vest.py ile seçildi)
  cropmask_agent_final_best.pt                   — maske (crop modu)
models/vinayak_trained_byBera/
  helmet_agent_final_best.pt                     — baret (scene modu)
  vest_agent_final_best.pt                       — yelek (scene modu)
models/
  mask_agent_scene_200ep_yolov8m_best.pt         — maske (scene modu)
```

## config.yaml Model Yapısı
```yaml
models:
  device: cuda
  person_model: ...    # ortak
  fire_model: ...      # ortak
  crop:
    helmet_model: ...
    vest_model: ...
    mask_model: ...
    mask_imgsz: 640
  scene:
    helmet_model: ...
    vest_model: ...
    mask_model: ...
```

## Proje Yapısı
```
backend/
  app.py              — Flask API + Socket.IO (port 5050)
  event_reader.py     — results/ dizininden event okuma (DB kapalıyken)
  watcher.py          — results/ dosya izleyici → DB writer
  database/
    connection.py     — psycopg2 bağlantısı
    reader.py         — DB okuma (events, timeline, notes, summary)
    writer.py         — DB yazma (write_event, close_event, resolve_event, update_llm_report, save_llm_report)
    schema.sql        — Tablo tanımları
  reports/
    services.py       — EventAnalyticsService + ReportSummaryService
llm/
  safety_report_agent.py  — SafetyReportAgent (model config'den, /think strip)
run_live_video.py     — Birleşik detection pipeline (--mode crop|scene)
event_manager.py      — Event state machine: idle→new→active→closed
tracking_identity.py  — TrackReattacher: çok-sinyal stabil kişi kimliği
track_reattacher.py   — tracking_identity için geriye uyumlu re-export
config.yaml           — Tüm ayarlar (detection thresholds, DB, backend, LLM, model paths)
frontend/src/
  pages/
    Dashboard.jsx     — İstatistik + grafik
    AlertHistory.jsx  — Event listesi + timeline (sol: sidebar, sağ: detay)
    Reports.jsx       — Periyodik raporlar + LLM raporu (async via Socket.IO)
    CameraSetup.jsx   — Kamera seçimi + detection modu seçimi + pipeline başlatma
    Settings.jsx      — PPE konfig
  components/
    Navbar.jsx, Sidebar.jsx, EventCard.jsx, MainPanel.jsx
    TimelineStep.jsx, PipelineControl.jsx
  api.js              — Tüm API çağrıları
  socket.js           — Socket.IO istemcisi
```

## Veritabanı Şeması (Ana Tablolar)
```sql
events          — event_id, event_status, camera_id, zone, signature(jsonb), persons(jsonb), ...
event_timeline  — her status geçişi için satır (ts, recorded_at, image_filename, ...)
event_notes     — kullanıcı notları
llm_reports     — kayıtlı periyodik LLM raporları (period, report_date, llm_text)
```

## API Endpoint'leri
```
GET  /api/events                      — filtrelenmiş event listesi
GET  /api/events/<id>                 — timeline + notlar
POST /api/events/<id>/note            — not ekle
PATCH /api/events/<id>/close          — event kapat (frontend veya pipeline callback)
PATCH /api/events/<id>/resolve        — event kapat (alternatif, /close ile aynı işlev)
PATCH /api/events/<id>/llm            — per-alarm LLM raporu güncelle
GET  /api/stats                       — dashboard istatistikleri
GET  /api/reports                     — grafik verisi (daily/weekly/monthly)
GET  /api/reports/summary             — risk skoru, trend, lokasyon dağılımı (DB gerekli)
POST /api/reports/summary/llm         — LLM raporu üret (async, Socket.IO ile döner)
GET  /api/reports/saved               — kayıtlı LLM raporları listesi
GET  /api/reports/saved/<id>          — tek kayıtlı rapor
GET  /api/config                      — PPE pipeline konfig
PUT  /api/config                      — PPE pipeline konfig güncelle
GET  /api/pipeline/status             — pipeline durumu {running, source, camera_id, zone, mode}
POST /api/pipeline/start              — pipeline başlat {source, camera_id, zone, mode}
POST /api/pipeline/stop               — pipeline durdur
GET  /api/pipeline/browse             — Windows dosya seçici (video)
POST /api/pipeline/camera-status      — kamera durum bildirimi → camera_status socket eventi
GET  /api/images/<event_id>/<fname>   — event fotoğrafı
```

## Socket.IO Eventleri
```
new_alert         → yeni ihlal eventi
event_closed      → event kapandı (/close veya /resolve)
llm_updated       → per-alarm LLM raporu güncellendi {event_id, llm_report}
report_llm_ready  → periyodik LLM raporu hazır {period, date, llm_text}
report_llm_error  → LLM raporu başarısız
camera_status     → kamera durumu {status: online|offline|frozen|dark, camera_id, zone}
```

## Event Status Akışı
`new → active → closed`

## TrackReattacher
`tracking_identity.py` — ByteTrack ID değişince (kısa oklüzyon sonrası) aynı kişiyi tanır.
- Sinyaller: merkez mesafesi (0.40), bbox alanı (0.25), en-boy oranı (0.15), zaman (0.15), PPE imzası (0.05)
- `reattacher.update(detections)` → `{raw_tid: stable_pid}` mapping döner
- `stable_pid` tüm state tracking ve event yazımında kullanılır

## Kamera İzleme
- **Freeze:** ardışık frame farkı < `cam_freeze_diff` eşiği → `cam_freeze_frames` kare sonra tetiklenir
- **Dark:** ortalama parlaklık < `cam_dark_thresh` → `cam_dark_frames` kare sonra tetiklenir
- **Offline:** video/kamera akışı kesilirse anında tetiklenir
- Hepsi → `POST /api/pipeline/camera-status` → `camera_status` socket eventi → `App.jsx` global banner

## Yangın Filtresi (her iki modda aynı)
- `fire_min_area_ratio`: yangın alanı / frame alanı bu oranın altındaysa görmezden gelinir
- `fire_growth_factor` + `fire_growth_window`: yangın büyümüyorsa bastırılır
- Smoke (duman) filtrelenmez
- Fire inference her 5 frame'de bir çalışır (`FIRE_INFER_EVERY = 5`)

## LLM Entegrasyonu
- **Per-alarm LLM:** `save_event()` içinde, `llm.enabled=true` ise async thread başlatılır → Ollama çağrılır → `PATCH /api/events/<id>/llm` ile DB'ye yazılır → `llm_updated` socket eventi
- **Periyodik rapor:** `POST /api/reports/summary/llm` → `SafetyReportAgent` (llm/safety_report_agent.py) → `report_llm_ready` socket eventi
- Her iki LLM akışı da `config.yaml llm:` bölümünden model/url/timeout okur

## Önemli Kararlar
- **"resolved" → "closed"** — tüm codebase'de event_status "closed" kullanılır
- **`/close` ve `/resolve`** — ikisi de aynı işlevi yapar (status=closed, repeat_count günceller)
- **`repeat_count`** — sadece aktif ihlal varken artar; "confirming_resolved" aşamasında artmaz; event kapanırken DB'ye final değer yazılır
- **`duration_sec`** — `_close_event()` artık duration_sec kabul eder ve PATCH body'e ekler; `/close` endpoint DB'ye yazar; hem normal kapanışta hem video-sonu temizliğinde geçirilir
- **camera_id + zone** — CLI `--camera-id` `--zone`, DB'ye yazılıyor, rapor lokasyon analizinde kullanılıyor
- **LLM async** — POST hemen döner, sonuç socket ile gelir
- **DB kapalıyken** — dosya sistemi (results/) fallback çalışır ama summary endpoint'leri 503 döner
- **device: cuda** — GPU öncelikli, CUDA yoksa otomatik CPU'ya düşer
- **Vest modeli (crop)** — `compare_vest.py` karşılaştırmasında `cropvest_agent_final_best.pt` seçildi

## Frontend Tema Sistemi
- **Light/dark toggle** — `App.jsx`'te `theme` state'i, `localStorage` ile persist; `toggleTheme()` senkron olarak `document.documentElement.setAttribute('data-theme', ...)` çağırır (grafik renkleri aynı frame'de güncellenir)
- **CSS değişkenleri** — `global.css` `:root` dark varsayılan, `[data-theme="light"]` override; tüm 16 CSS dosyası `var(--bg)`, `var(--surface)` vb. kullanıyor
- **Navbar toggle butonu** — `Navbar.jsx` `theme` + `onToggleTheme` prop alır; dark modda güneş ikonu, light modda ay ikonu gösterir; Navbar kendisi her iki modda da koyu kalır (`--bg-nav` override edilmez)
- **PPE renkleri (kesinleşmiş):** baret sarı `#ffd740`, yelek turuncu `#ff8c40`, maske mavi `#66bbff`, yangın kırmızı `#ff5f5f` — `SignatureSummary.css`, `Dashboard.jsx` `DIST_COLORS`, `Reports.jsx` `COLORS` hepsi bu değerleri kullanıyor
- **Grafik tema stilleri** — `Dashboard.jsx` ve `Reports.jsx`'te `CHART_STYLES = { dark: {...}, light: {...} }` objesi; `theme` prop ile seçilir; tick rengi, tooltip arkaplan/kenarlık, grid, legend renkleri tema ile değişir

## Düzeltilen Buglar
- **`services.py` SyntaxError** — `generate_daily/weekly/monthly_summary` içindeki 3 adet `except` satırında literal `\n` karakteri vardı (gerçek newline değil); Python import sırasında SyntaxError'a yol açıyordu → `/api/reports/summary` 500 döndürüyordu → Reports sayfası boş görünüyordu. Gerçek newline ile değiştirildi.
- **`api.js _post()` BASE eksikliği** — `_get` ve `_put` `BASE + path` kullanırken `_post` sadece `path` kullanıyordu. `BASE + path` yapıldı.
- **Reports `undefined%`** — `comparison.change_percent !== null` (strict) `undefined`'ı geçiriyordu; `!= null` (loose) yapıldı.
- **`duration_sec` DB'ye yazılmıyordu** — `_close_event()` sadece `repeat_count` geçiriyordu; `writer.close_event()` SQL'de `duration_sec` yoktu. Dört dosyada düzeltildi: `run_live_video.py` (`_close_event` fonksiyonu + çağrı noktaları), `backend/app.py` (`/close` endpoint), `backend/database/writer.py` (`close_event` SQL).

## config.yaml Bölümleri
`database`, `backend`, `detection`, `ppe_pipeline`, `models` (crop/scene alt bölümleri dahil), `event_manager`, `llm`, `results_keep_events`

## Kesinleşmiş Pipeline Parametreleri (benchmark ile doğrulandı)
| Parametre | Değer | Kaynak |
|-----------|-------|--------|
| `PPE_INFER_EVERY` | 4 | benchmark_skip.py — skip=4: 29.4 FPS (+133% vs skip=1), doğruluk kaybı yok |
| `temporal_window` | 20 | benchmark_temporal.py — window=20 elbow; üstünde V-viol monoton düşüyor |
| `helmet_conf` | 0.20 | benchmark_conf.py — 0.15+ plato, 0.20 seçildi |
| `vest_conf` | 0.30 | benchmark_conf.py — 0.30'da violation_rate artar, known_rate kaybı kabul edilebilir |
| `mask_conf` | 0.25 | benchmark_conf.py — tüm değerlerde viol=%100, 0.25 gürültü filtresi için optimal |

`PPE_INFER_EVERY` config.yaml'da yok — `run_live_video.py` içinde hardcoded sabit olarak tanımlı.

## Mezuniyet Tezi — TEZ_TASLAK.md
Tez taslağı bu repo kökünde `TEZ_TASLAK.md` dosyasında tutulmaktadır (~1811 satır, Mayıs 2026).

**Tamamlanan bölümler:**
```
ÖZET + ABSTRACT
SİMGELER VE KISALTMALAR   (31 kısaltma, alfabetik)
1. GİRİŞ
2. YÖNTEM (2.1–2.8)
3. BULGULAR (3.1–3.7)
   3.4.4 PPE_INFER_EVERY optimizasyonu — benchmark_skip.py sonuçları
   3.4.5 Conf eşiği optimizasyonu     — benchmark_conf.py sonuçları
   3.4.6 temporal_window optimizasyonu — benchmark_temporal.py sonuçları
   3.7   Literatür karşılaştırması     — Nath 2020, Wu 2019 vs SafetyMonitor
4. SONUÇ  (optimizasyon + literatür karşılaştırma alt başlıkları dahil)
KAYNAKLAR  [1]–[13] APA formatı
EK-1  Fizibilite Raporu (7 alt bölüm: problem, teknik, operasyonel, süre, maliyet, risk, sonuç)
EK-2  Proje Dizin Yapısı
EK-3  config.yaml tam içeriği
EK-4  Veritabanı Şeması (schema.sql)
EK-5  API Endpoint Referans Tablosu
EK-6  Kesinleşmiş Model Dosyaları
```

**Kullanıcının kendi yazması gereken bölümler (içerik eksik):**
- `TEŞEKKÜR` — kişisel
- `İÇİNDEKİLER` — Word'de otomatik oluşturulur
- `ÖZGEÇMİŞ` — kişisel

**Benchmark betikleri** (`scripts/` dizini):
- `benchmark_skip.py` — PPE_INFER_EVERY taraması (1,2,3,4,6,8), 250 frame, 11 video
- `benchmark_conf.py` — helmet/vest/mask conf bağımsız taraması, 200 frame, 4 video
- `benchmark_temporal.py` — temporal_window taraması [5,10,15,20,30,40,50], 300 frame, 4 video
- Sonuçlar: `runs/benchmarks/{skip,conf,temporal}/` altında CSV olarak kaydedildi
