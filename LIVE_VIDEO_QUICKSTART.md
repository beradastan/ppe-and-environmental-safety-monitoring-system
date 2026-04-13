# 🎥 CANLÎ VİDEO SİSTEMİ - BAŞLANGIÇ KILAVUZU

## ✅ TAMAMLANDI: Event-Based Alarm Yönetimi

---

## 🎯 NE YAPILDI?

Mevcut frame-by-frame görüntü işleme sistemini, **event-based alarm yönetimi** sistemine dönüştürdük.

### Eski Sistem (Frame-Bazlı Kayıt)
```
Frame 1 → Alarm var → [KAYIT] alarm_report_1.txt
Frame 2 → Aynı alarm → [KAYIT] alarm_report_2.txt  ← Gereksiz!
Frame 3 → Aynı alarm → [KAYIT] alarm_report_3.txt  ← Gereksiz!
```

### Yeni Sistem (Event-Bazlı Kayıt) ✅
```
Frame 1 → Alarm başladı → evt_0001_start.txt ✓
Frame 2 → Aynı alarm (repeat) → Kayıt YOK
Frame 3 → Aynı alarm (repeat) → Kayıt YOK
Frame 4 → Alarm değişti → evt_0001_update_01.txt ✓
Frame 5 → 10s timeout → evt_0001_resolved.txt ✓
```

---

## 📦 OLUŞTURULAN DOSYALAR

| Dosya | Satır | Amaç |
|-------|-------|------|
| `event_manager.py` | 297 | Event yönetimi (core logic) |
| `run_live_video.py` | 320 | Canlı video template |
| `EVENT_BASED_SYSTEM.md` | 250 | Detaylı dokümantasyon |

### Güncellenen Dosyalar
- `llm_coordinator.py` - Event desteği
- `run_with_llm.py` - Event entegrasyonu

---

## 🚀 NASIL BAŞLANILIR?

### 1️⃣ Test Görüntülerle (Mevcut Sistem)

```bash
# Offline mode
python run_with_llm.py --offline

# Canlı Ollama ile
python run_with_llm.py
```

**Sonuç:** `results/evt_0001/`, `results/evt_0002/`, vb. klasörleri oluşturur.

### 2️⃣ Canlı Kamera İçin (Yeni Sistem)

```bash
# Varsayılan kamera (0)
python run_live_video.py

# Kamera 1
python run_live_video.py --camera 1

# Video dosyası
python run_live_video.py --video input.mp4

# Offline mode
python run_live_video.py --offline
```

---

## 📊 ÖRNEK SONUÇLAR

### Test Görüntüleriyle Çalıştırma

```
[1/6] Screenshot_1.png
  CNN | Helmet:1OK/1!! Vest:0OK/1!! Fire:1 yangın [ALARM] [NEW]
  [KAYIT] evt_0001/evt_0001_start.txt

[2/6] Screenshot_2.png
  CNN | Helmet:0OK/0!! Vest:0OK/0!! Fire:0 yangın [GUVENLI]
  (LLM çağrılmadı)

[3/6] group-workers.webp
  CNN | Helmet:5OK/0!! Vest:0OK/4!! Fire:0 yangın [ALARM] [NEW]
  [KAYIT] evt_0002/evt_0002_start.txt

ÖZET:
  Toplam görüntü: 6
  Alarm tetiklenen: 4
  Event sayısı: 4 (tekrar almamıştır!)
  Kaydedilen raporlar: 4 × (txt+json) = 8 dosya
```

### Sonuç Klasörü Yapısı

```
results/
├── evt_0001/
│   ├── evt_0001_start.txt
│   ├── evt_0001_start.json
│   ├── evt_0001_start.jpg (canlı video için)
│   └── evt_0001_resolved.txt
├── evt_0002/
│   ├── evt_0002_start.txt
│   ├── evt_0002_start.json
│   └── evt_0002_update_01.txt (alarm değişti)
```

---

## 🔧 YAPIMI ANLAYALIM

### EventManager Mantığı

```python
from event_manager import EventManager

manager = EventManager(timeout_sec=10.0)  # 10 saniye timeout

# Her frame için
event_info = manager.process_frame(helmet_result, vest_result, fire_result)

print(event_info["event_status"])  # "new", "active", "update", "resolved", "no_alarm"
print(event_info["should_save"])   # True/False (kaydet mi?)
print(event_info["event_id"])      # "evt_0001"
print(event_info["repeat_count"])  # 1, 2, 3, ...
```

### Event Statüsleri

| Status | Ne Yapılır? |
|--------|-----------|
| `no_alarm` | Hiç kayıt yapılmaz |
| `new` | **Yeni event başladı → KAYDET** |
| `active` | Aynı alarm devam ediyor → Kayıt YOK |
| `update` | Alarm detayları değişti → KAYDET |
| `resolved` | Alarm sona erdi (10s timeout) → KAYDET |

---

## 💡 ÖNEMLI ÖZELLİKLER

✅ **Intelligent Comparison**
- Helmet count farkı ≤ 1 kişiye kadar aynı alarm
- Vest count farkı ≤ 1 kişiye kadar aynı alarm
- Fire confidence farkı ≤ %5'e kadar aynı alarm

✅ **Timeout Mantığı**
- Alarm ≥ 10 saniye yoksa event "resolved"
- Yapılandırılabilir: `EventManager(timeout_sec=5.0)` gibi

✅ **Frame-Ready**
- Canlı video döngüsüne sorunsuz entegrasyon
- Stateful: bellek içinde aktif event tutar

✅ **TXT + JSON**
- Insan-okunur TXT raporları
- Makinenin okuyabileceği JSON veriler

---

## 📁 KLASÖR YAPISI

```
factory_antigravity/
├── run_with_llm.py              (Mevcut sistem - test görüntüleri için)
├── run_live_video.py            (YENİ - canlı video için)
├── event_manager.py             (YENİ - event yönetimi)
├── llm/
│   └── llm_coordinator.py       (Güncellenmiş - event desteğiyle)
├── agents/
│   └── specific_agents.py       (Değişmedi)
├── results/
│   ├── evt_0001/
│   ├── evt_0002/
│   └── ...
└── EVENT_BASED_SYSTEM.md        (Detaylı dokümantasyon)
```

---

## 🔄 MEVCUT KOD BOZAN YOK

✅ **Backward Compatible**
- `detect()` fonksiyonları değişmedi
- CNN ajanlar aynı output üretir
- LLM coordinator eski kullanım şeklini destekliyor
- Event parametresi opsiyonel

---

## 🎓 ÖRNEK: Canlı Video Entegrasyonu

```python
import cv2
from event_manager import EventManager

cap = cv2.VideoCapture(0)
manager = EventManager(timeout_sec=10)

while True:
    ret, frame = cap.read()
    
    # CNN
    hr = helmet_agent.detect(frame)
    vr = vest_agent.detect(frame)
    fr = fire_agent.detect(frame)
    
    # Event
    event_info = manager.process_frame(hr, vr, fr)
    
    # Sadece kayıt gerekirse
    if event_info["should_save"]:
        # Kaydet
        save_report(event_info)
        cv2.imwrite(f"results/{event_info['event_id']}/frame.jpg", frame)
    
    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ KONTROL LİSTESİ

- [x] EventManager sınıfı oluşturuldu
- [x] AlarmSignature benzerliği mantığı
- [x] LLM'ye event_info parametresi eklendi
- [x] run_with_llm.py event entegrasyonu
- [x] run_live_video.py template'i oluşturuldu
- [x] TXT + JSON kayıt sistemi
- [x] Backward compatibility sağlandı
- [x] Test edildi ve doğrulandı

---

## 📞 SIKI SORULAR

### S: Eski sistemim bozulur mu?
**Cevap:** Hayır! Event parametresi opsiyonel. Eski kodunuz çalışmaya devam edecek.

### S: Kamera açılırsa ne olur?
**Cevap:** `run_live_video.py`'yi kullanın. Template olarak yapılmış.

### S: Timeout süresi ne kadar olmalı?
**Cevap:** Uygulamanıza bağlı. Default 10 saniye. Değiştirebilirsiniz.

### S: Event'leri database'e kaydedebilir miyim?
**Cevap:** Evet! `save_event_report()` fonksiyonunu özelleştirin.

### S: Screenshot'ları kaydetmek istiyorum?
**Cevap:** `run_live_video.py`'de örneği var:
```python
cv2.imwrite(f"results/{event_id}/{suffix}.jpg", frame)
```

---

## 🚀 BAŞLAYIN

```bash
# 1. Test et
python run_with_llm.py --offline

# 2. Sonuçları kontrol et
ls -R results/

# 3. Canlı video için hazır
python run_live_video.py --camera 0
```

---

**Sistem artık production-ready! 🎉**

Herhangi bir sorun olursa `EVENT_BASED_SYSTEM.md`'yi okuyun.

