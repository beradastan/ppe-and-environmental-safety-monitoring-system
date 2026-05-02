# Factory Safety — CLAUDE.md

## Proje Özeti
Fabrika içi iş güvenliği izleme sistemi. YOLO + ByteTrack ile PPE (baret/yelek/maske) ve yangın tespiti yapıp ihlalleri kaydeder, raporlar ve web arayüzü üzerinden gösterir.

## Teknoloji Yığını
- **Backend:** Flask + Flask-SocketIO, Python 3.9 (`.venv`), `from __future__ import annotations` gerekli
- **Frontend:** React + Vite (`frontend/`), port 5173/5174
- **DB:** PostgreSQL — host: localhost:5432, db: ppe_db, user: postgres, password: 1234
- **Detection:** `run_live_video.py` — YOLO crop-based PPE pipeline, ByteTrack
- **LLM:** Ollama + qwen3:8b (periyodik özet raporlar için)
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

## Proje Yapısı
```
backend/
  app.py              — Flask API + Socket.IO (port 5050)
  event_reader.py     — results/ dizininden event okuma (DB kapalıyken)
  event_manager.py    — Event state machine: idle→new→active→closed
  watcher.py          — results/ dosya izleyici → DB writer
  database/
    connection.py     — psycopg2 bağlantısı
    reader.py         — DB okuma (events, timeline, notes, summary)
    writer.py         — DB yazma (write_event, close_event, write_note)
    schema.sql        — Tablo tanımları
  reports/
    services.py       — EventAnalyticsService + ReportSummaryService
llm/
  safety_report_agent.py  — SafetyReportAgent (qwen3:8b, /think prefix)
run_live_video.py     — Ana detection pipeline
event_manager.py      — State machine (proje kökünde)
config.yaml           — Tüm ayarlar (detection thresholds, DB, backend)
frontend/src/
  pages/
    Dashboard.jsx     — İstatistik + grafik
    AlertHistory.jsx  — Event listesi + timeline (sol: sidebar, sağ: detay)
    Reports.jsx       — Periyodik raporlar + LLM raporu (async via Socket.IO)
    CameraSetup.jsx   — Kamera seçimi + pipeline başlatma
    Settings.jsx      — PPE konfig
  components/
    Navbar.jsx        — Dashboard / Alarmlar / Raporlar / Kamera / Ayarlar
    Sidebar.jsx       — Event kartları listesi
    EventCard.jsx     — Tek event özet kartı
    MainPanel.jsx     — Timeline + notlar
    TimelineStep.jsx  — Tek timeline adımı
    PipelineControl.jsx — Settings sayfasındaki pipeline kontrolü
  api.js              — Tüm API çağrıları
  socket.js           — Socket.IO istemcisi
```

## Veritabanı Şeması (Ana Tablolar)
```sql
events          — event_id, event_status, camera_id, zone, signature(jsonb), persons(jsonb), ...
event_timeline  — her status geçişi için satır
event_notes     — kullanıcı notları
```

## API Endpoint'leri
```
GET  /api/events                      — filtrelenmiş event listesi
GET  /api/events/<id>                 — timeline + notlar
POST /api/events/<id>/note            — not ekle
PATCH /api/events/<id>/close          — event kapat
GET  /api/stats                       — dashboard istatistikleri
GET  /api/reports                     — grafik verisi (daily/weekly/monthly)
GET  /api/reports/summary             — risk skoru, trend, lokasyon dağılımı (DB gerekli)
POST /api/reports/summary/llm         — LLM raporu üret (async, Socket.IO ile döner)
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
event_closed      → event kapandı
report_llm_ready  → periyodik LLM raporu hazır {period, date, llm_text}
report_llm_error  → LLM raporu başarısız
camera_status     → kamera durumu {status: online|offline|frozen|dark, camera_id, zone}
```

## Event Status Akışı
`new → active → closed` (update ara durumu var ama frontend'e yansımıyor)

## Önemli Kararlar
- **Sahne bazlı LLM kaldırıldı** — sadece periyodik özet raporlar (daily/weekly/monthly) var
- **"resolved" → "closed"** — tüm codebase'de renamed
- **camera_id + zone** — CLI argümanları `--camera-id` `--zone`, DB'ye yazılıyor, rapor lokasyon analizinde kullanılıyor
- **StatusBadge** — sadece `new` + görülmemiş eventlerde "YENİ" göster
- **LLM async** — POST hemen döner, sonuç `report_llm_ready` socket eventi ile gelir
- **DB kapalıyken** — dosya sistemi (results/) fallback çalışır ama summary endpoint'leri 503 döner
- **Yangın filtresi** — `fire_min_area_ratio` (boyut) + `fire_growth_factor/window` (büyüme) ile küçük/kontrollü ateşler elenir; smoke filtrelenmez
- **Kamera izleme** — offline/frozen/dark durumları `camera_status` socket eventi ile frontend'e iletilir; `App.jsx`'te global banner gösterilir

## Model Dosyaları (git'e dahil değil)
```
models/person_agent_scene_vinayakstyle_best.pt
models/bera/crophelmet_agent_final_best.pt
models/bera/vest_agent_final_best.pt
models/bera/cropmask_agent_final_best.pt
models/bera/fire_smoke_other_agent_final_best.pt
```

## config.yaml Yapısı
database, backend, detection, ppe_pipeline, models, event_manager, llm, results_keep_events bölümleri var.
