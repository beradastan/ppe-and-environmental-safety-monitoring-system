# Factory Safety — CLAUDE.md (crop-based branch)

## Proje Özeti
Fabrika içi iş güvenliği izleme sistemi. YOLO + ByteTrack ile PPE (baret/yelek/maske) ve yangın tespiti yapıp ihlalleri kaydeder, raporlar ve web arayüzü üzerinden gösterir.

Bu branch: **crop-based detection** — kişi kırpılır, ayrı PPE modeli her crop için çalıştırılır.  
Diğer branch: `feature/scene-based-ppe-detection` — tam kare PPE tespiti, `_inside_frac` eşleştirmesi.

**İki branch arasındaki tek fark detection mimarisidir** — backend, frontend, LLM mantığı her ikisinde de aynı.

## Teknoloji Yığını
- **Backend:** Flask + Flask-SocketIO, Python 3.9 (`.venv`), `from __future__ import annotations` gerekli
- **Frontend:** React + Vite (`frontend/`), port 5173/5174
- **DB:** PostgreSQL — host: localhost:5432, db: ppe_db, user: postgres, password: 1234
- **Detection:** `run_live_video.py` — YOLO crop-based PPE pipeline, ByteTrack + TrackReattacher
- **LLM:** Ollama — model `config.yaml` `llm.model`'dan okunur (varsayılan qwen3:8b), periyodik özet raporlar için
- **GPU:** RTX 3060 6GB

## Çalıştırma
```bash
# Backend (port 5050)
python -m backend.app

# Frontend
cd frontend && npm run dev

# Detection pipeline
python run_live_video.py --camera 0 --camera-id cam_01 --zone "Üretim Hattı A" --display
```

## Detection Mimarisi — Crop-Based
```
frame
 └─ person_model.track()         → kişi bbox + ByteTrack ID
     └─ TrackReattacher.update() → stable_pid (oklüzyon geçince aynı ID)
         └─ crop_ppe(frame, bbox, "helmet") → kişi başlık bölgesi kırpılır
         └─ helmet_model.predict(crop)      → kırpık üzerinde tespit
         └─ crop_ppe(frame, bbox, "vest")   → kişi gövde bölgesi kırpılır
         └─ vest_model.predict(crop)        → kırpık üzerinde tespit
         └─ (mask için de aynı)
         └─ _global_assign_ppe()            → one-to-one kişi-PPE atama
```

## Kesinleşmiş Model Dosyaları (git'e dahil değil)
```
models/person_agent_scene_vinayakstyle_best.pt  — kişi tespiti (her iki branch'te aynı)
models/bera/
  crophelmet_agent_final_best.pt               — baret tespiti (crop)
  cropvest_agent_final_best.pt                 — yelek tespiti (crop, compare_vest.py ile seçildi)
  cropmask_agent_final_best.pt                 — maske tespiti (crop)
  fire_smoke_other_agent_final_best.pt         — yangın/duman (her iki branch'te aynı)
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
    writer.py         — DB yazma (write_event, close_event, resolve_event, update_llm_report)
    schema.sql        — Tablo tanımları
  reports/
    services.py       — EventAnalyticsService + ReportSummaryService
llm/
  safety_report_agent.py  — SafetyReportAgent (model config'den, /think strip)
run_live_video.py     — Ana detection pipeline (crop-based)
event_manager.py      — Event state machine: idle→new→active→closed
tracking_identity.py  — TrackReattacher: çok-sinyal stabil kişi kimliği
track_reattacher.py   — tracking_identity için geriye uyumlu re-export
config.yaml           — Tüm ayarlar (detection thresholds, DB, backend, LLM)
frontend/src/
  pages/
    Dashboard.jsx     — İstatistik + grafik
    AlertHistory.jsx  — Event listesi + timeline (sol: sidebar, sağ: detay)
    Reports.jsx       — Periyodik raporlar + LLM raporu (async via Socket.IO)
    CameraSetup.jsx   — Kamera seçimi + pipeline başlatma
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
PATCH /api/events/<id>/close          — event kapat (frontend)
PATCH /api/events/<id>/resolve        — event kapat (pipeline callback)
PATCH /api/events/<id>/llm            — per-alarm LLM raporu güncelle
GET  /api/stats                       — dashboard istatistikleri
GET  /api/reports                     — grafik verisi (daily/weekly/monthly)
GET  /api/reports/summary             — risk skoru, trend, lokasyon dağılımı (DB gerekli)
POST /api/reports/summary/llm         — LLM raporu üret (async, Socket.IO ile döner)
GET  /api/reports/saved               — kayıtlı LLM raporları listesi
GET  /api/reports/saved/<id>          — tek kayıtlı rapor
GET  /api/config                      — PPE pipeline konfig
PUT  /api/config                      — PPE pipeline konfig güncelle
GET  /api/pipeline/status             — pipeline çalışıyor mu {running, source, camera_id, zone}
POST /api/pipeline/start              — pipeline başlat {source, camera_id, zone}
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

## Yangın Filtresi
- `fire_min_area_ratio`: yangın alanı / frame alanı bu oranın altındaysa görmezden gelinir
- `fire_growth_factor` + `fire_growth_window`: yangın büyümüyorsa (eski/kontrollü ateş) bastırılır
- Smoke (duman) filtrelenmez

## Önemli Kararlar
- **"resolved" → "closed"** — tüm codebase'de renamed
- **camera_id + zone** — CLI `--camera-id` `--zone`, DB'ye yazılıyor, rapor lokasyon analizinde kullanılıyor
- **LLM config'den okunur** — `SafetyReportAgent` artık `config.yaml` `llm:` bölümünü kullanır
- **LLM async** — POST hemen döner, sonuç `report_llm_ready` socket eventi ile gelir
- **StatusBadge** — sadece `new` + görülmemiş eventlerde "YENİ" göster
- **DB kapalıyken** — dosya sistemi (results/) fallback çalışır ama summary endpoint'leri 503 döner
- **Yangın filtresi** — `fire_min_area_ratio` (boyut) + `fire_growth_factor/window` (büyüme) ile küçük/kontrollü ateşler elenir
- **device: cuda** — GPU öncelikli, CUDA yoksa otomatik CPU'ya düşer
- **Vest modeli** — `compare_vest.py` karşılaştırmasında `cropvest_agent_final_best.pt` seçildi (false positive azaldı)

## config.yaml Bölümleri
`database`, `backend`, `detection`, `ppe_pipeline`, `models`, `event_manager`, `llm`, `results_keep_events`
