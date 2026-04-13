# 🎉 Event-Based Factory Safety System - TAMAMLANDI

## 📌 Yapılan İşler

### ✅ Sistem Mimarisi
1. **EventManager** (event_manager.py)
   - AlarmSignature: Anlık tehlike özeti (helmet, vest, fire)
   - ActiveEvent: Aktif olay bilgisi (event_id, repeat_count, duration, status)
   - State Machine: idle → new → active → update → resolved
   - Alarm benzerliği: Aynı alarmları aynı event'te tutar

2. **LLM Coordinator** (llm/llm_coordinator.py)
   - Event context desteği
   - Minimal format (hızlı + halüsinasyon az)
   - Event bilgisi ile birlikte rapor üretimi

3. **Canlı Video Pipeline** (run_live_video.py)
   - Saniyede 1 kez CNN çalıştırma
   - Event-bazlı kayıt (frame-bazlı değil)
   - Ekran gösterimi (event_id, status, repeat_count)
   - Offline test modu

### ✅ Dosya Yapısı
```
results/
  evt_0001/
    evt_0001_start.jpg           # Screenshot
    evt_0001_start.txt           # Text rapor
    evt_0001_start.json          # JSON veri
    evt_0001_update_01.jpg       # Güncelleme
    evt_0001_update_01.txt
    evt_0001_update_01.json
    evt_0001_resolved.txt        # Kapanış
    evt_0001_resolved.json
```

### ✅ Dokumentasyon
- **EVENT_SYSTEM_GUIDE.md**: Detaylı teknik dokümantasyon
- **LIVE_VIDEO_GUIDE.md**: Hızlı başlangıç rehberi
- **README.md**: Proje genel bilgileri

### ✅ Cleanup
- Eski/çakışan dokümantasyon dosyaları silindi
- Eski run_with_llm.py silindi
- Proje yapısı temiz ve odaklanmış hale getirildi

---

## 🚀 Kullanım

### Offline Modda (Hızlı Test)
```bash
python run_live_video.py --offline --video test/images.jpg
```

### Gerçek Modda
```bash
# Kameradan
python run_live_video.py

# Video dosyasından
python run_live_video.py --video test/ppe_test1.mp4
```

---

## 🎯 Event Lifecycle Örneği

```
Saniye 0-1: Alarm görüldü (1 frame)
  → Debounce counter = 1

Saniye 1-2: Aynı alarm (2 frames)
  → Debounce counter = 2 → EVENT AÇIL (status=new)
  → evt_0001_start.jpg/txt/json KAYIT
  → LLM raporu üretilir

Saniye 2-5: Aynı alarm (tekrar ediyor)
  → repeat_count = 4
  → status = active
  → KAYIT YOK (değişiklik yok)

Saniye 5-6: Alarm değişti (sayı artmış)
  → status = update
  → evt_0001_update_01.jpg/txt/json KAYIT
  → LLM raporu güncellenir

Saniye 6-16: Alarm bitmiş (10 saniye timeout)
  → status = resolved
  → evt_0001_resolved.txt/json KAYIT
  → Event kapatılır, idle durumuna dön
```

---

## 📊 Console Çıktısı

```
=================================================================
  FACTORY SAFETY — CANLŞ VIDEO + EVENT-BAZLI YÖNETİM
=================================================================
  Kaynak  : Video: test/ppe_test1.mp4
  LLM     : [Ollama / mistral]
  Timeout : 10 saniye
  İşleme  : saniyede 1 kez

[1/4] CNN Ajanları yükleniyor...
  ✓ HelmetAgent
  ✓ VestAgent
  ✓ FireAgent

[2/4] LLM + Event Manager başlatılıyor...
  ✓ LLMCoordinator
  ✓ EventManager (timeout=10s)

[3/4] Video kaynağı açılıyor...
  ✓ Video dosyası: .../test/ppe_test1.mp4

[4/4] Video işleniyor (ESC = çıkış)...

  Frame 30 | Events: 0
  Frame 60 | Events: 1
  Frame 90 | Events: 1
  Frame 120 | Events: 2

=================================================================
  ÖZET
=================================================================
  Toplam frame     : 614
  Event sayısı     : 3
  Sonuçlar         : results/
=================================================================
```

---

## 💾 GitHub Entegrasyonu

✅ Repository: https://github.com/beradastan/final_project
✅ Private olarak ayarlandı
✅ 2 commit yapıldı:
- Initial commit: 46 dosya
- Event-based upgrade: Cleanup + Documentation

---

## 🔧 Teknik Detaylar

### AlarmSignature
```python
@dataclass
class AlarmSignature:
    helmet_violation_count: int      # Baretsiz kişi
    vest_violation_count: int        # Yeleksiz kişi
    fire_detected: bool              # Yangın var mı?
    fire_confidence: float           # 0.0-1.0
```

### Alarm Benzerliği
```
Aynı event kabul şartları:
✓ helmet count farkı ≤ 1
✓ vest count farkı ≤ 1
✓ fire_detected aynı
✓ fire_confidence farkı ≤ 0.05
```

### Event State Machine
```
idle → new → active ↔ update → resolved
```

---

## ⚙️ Ayarlanabilir Parametreler

**timeout_sec**: Event kapanmadan önce bekleme süresi
- Varsayılan: 10 saniye
- Değiştirme: `event_manager.py` → EventManager init

**process_interval_sec**: CNN çalıştırma aralığı
- Varsayılan: 1 saniye (saniyede 1 kez)
- Değiştirme: `run_live_video.py` → process_interval_sec

**LLM Model**: Ollama model seçimi
- Varsayılan: mistral
- Seçenekler: mistral, llama2, neural-chat, vb.

---

## 📈 Sistem Performansı

- **Video FPS**: 30 FPS (standart)
- **CNN İşleme**: 1 FPS (saniyede 1 kez)
- **Bellek Kullanımı**: ~500MB (model yükleme dahil)
- **LLM Çağrısı**: Sadece event değiştiğinde (hızlı)

---

## 🔮 Gelecek Geliştirmeler

- [ ] Web dashboard (event timeline görselleştirme)
- [ ] Database (SQLite/PostgreSQL) entegrasyonu
- [ ] Alert sistemi (email/SMS/Slack)
- [ ] Multi-zone support (bölge bazlı alarm)
- [ ] Machine learning confidence tuning
- [ ] Export (CSV/Excel) desteği
- [ ] Real-time metrics (WebSocket)

---

## 📚 Dokümantasyon

1. **LIVE_VIDEO_GUIDE.md** ← Başla buradan! (Hızlı başlangıç)
2. **EVENT_SYSTEM_GUIDE.md** ← Teknik detaylar
3. **README.md** ← Genel proje bilgileri
4. **event_manager.py** ← Source code (AlarmSignature, ActiveEvent)
5. **run_live_video.py** ← Ana canlı video loop

---

## ✨ Özellikler Özeti

| Feature | Durum | Notlar |
|---------|-------|--------|
| Event-based kayıt | ✅ | Frame-bazlı değil |
| Alarm benzerliği | ✅ | Toleranslı karşılaştırma |
| LLM entegrasyonu | ✅ | Event context ile |
| Canlı video | ✅ | Saniyede 1 kez CNN |
| Offline test | ✅ | Mock LLM |
| JSON/TXT export | ✅ | Her event için |
| Screenshot kayıt | ✅ | start/update/resolved |
| Timeout mekanizması | ✅ | Ayarlanabilir |
| Console logger | ✅ | Real-time output |

---

## 🎬 Hemen Başla!

```bash
# 1. Offline modda test et
python run_live_video.py --offline --video test/images.jpg

# 2. Sonuçları kontrol et
ls results/evt_0001/

# 3. JSON'ı oku
cat results/evt_0001/evt_0001_start.json

# 4. Gerçek video ile çalıştır
python run_live_video.py --video test/ppe_test1.mp4
```

---

**📧 Sorular? → LIVE_VIDEO_GUIDE.md'ye bak!**
**🔧 Teknik detay? → EVENT_SYSTEM_GUIDE.md'ye bak!**
**💻 Kod? → event_manager.py / run_live_video.py'yi aç!**

---

*Event-based Factory Safety System v1.0 - Hazır Üretim Yönetimi* 🚀

