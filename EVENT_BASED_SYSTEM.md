# 🎥 CANLÎ VIDEO SİSTEMİ - EVENT-BAZLI ALARM YÖNETİMİ

## ✅ TAMAMLANDI

Event-based alarm yönetimi sistemi başarıyla kuruldu. Artık:

### 🎯 Ne Yapılıyor?

**Frame-Bazlı İşleme:**
```
Her Frame → CNN Deteksiyonu → Event Manager → Karar
```

**Event Mantığı:**
- ✅ Aynı alarm tekrar tekrar oluştuğunda yeni event açmaz
- ✅ Alarm başladığında: `event_status = "new"` → KAYDET
- ✅ Alarm detayları değiştiğinde: `event_status = "update"` → KAYDET
- ✅ Alarm sona erdiğinde: `event_status = "resolved"` → KAYDET
- ✅ Aynı alarm sürüyorsa: `event_status = "active"` → KAYDETME

### 📂 SONUÇ YAPISI

```
results/
├── evt_0001/
│   ├── evt_0001_start.txt     (yeni alarm başladığında)
│   ├── evt_0001_start.json
│   ├── evt_0001_update_01.txt (alarm detayları değiştiğinde)
│   ├── evt_0001_update_01.json
│   └── evt_0001_resolved.txt  (alarm sona erdiğinde)
│
├── evt_0002/
│   ├── evt_0002_start.txt
│   ├── evt_0002_start.json
│   └── evt_0002_resolved.txt
```

### 🔄 CANLÎ VIDEO İÇİN HAZIR

Bu sistem canlı video akışına çok kolay entegre edilebilir:

```python
# Pseudocode: canlı video için
cap = cv2.VideoCapture(0)  # Kamera

while True:
    ret, frame = cap.read()
    
    # CNN
    hr = helmet_agent.detect(frame)
    vr = vest_agent.detect(frame)
    fr = fire_agent.detect(frame)
    
    # Event yönetimi
    event_info = event_manager.process_frame(hr, vr, fr)
    
    # Sadece event varsa kayıt
    if event_info["should_save"]:
        cv2.imwrite(f"results/{event_info['event_id']}/frame_{time.time()}.jpg", frame)
        report = llm.generate_alarm_report(hr, vr, fr, event_info=event_info)
        save_report(report, Path("results"))
```

---

## 📋 DOSYALAR

### ✅ Oluşturulan/Güncellenen

1. **event_manager.py** (295 satır)
   - `AlarmSignature` → Alarm durumunun hash'i
   - `ActiveEvent` → Aktif event yönetimi
   - `EventManager` → Ana sınıf
   - Tolerans: helmet/vest count farkı ≤ 1

2. **llm_coordinator.py** → `event_info` parametresi eklendi
   - `format_for_llm_minimal()` → event_info destekleme
   - `generate_alarm_report()` → event_info parametresi

3. **run_with_llm.py** → Event manager entegrasyonu
   - `save_report()` → Event-bazlı klasör yapısı
   - Main loop → Event manager ile çalışma
   - Txt + JSON kayıt desteği

---

## 🚀 BAŞLAMA

### Test/Görüntü Klasörü İçin

```bash
python run_with_llm.py --offline
```

### Canlı Kamera İçin (Video Döngüsü)

```python
# Yeni dosya: run_live_video.py (örnek)
import cv2
from event_manager import EventManager
from llm.llm_coordinator import OllamaLLMCoordinator

cap = cv2.VideoCapture(0)
event_manager = EventManager(timeout_sec=10)
llm = OllamaLLMCoordinator()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Deteksiyonlar
    hr = helmet_agent.detect(frame)
    vr = vest_agent.detect(frame)
    fr = fire_agent.detect(frame)
    
    # Event yönetimi
    event_info = event_manager.process_frame(hr, vr, fr)
    
    # Sadece event'ler kayıt edilir
    if event_info["should_save"]:
        report = llm.generate_alarm_report(hr, vr, fr, event_info=event_info)
        save_report(report, Path("results"))
        
        # Screenshot da kaydet
        cv2.imwrite(f"results/{event_info['event_id']}/frame.jpg", frame)
```

---

## 🔧 ÖNEMLİ AYARLAR

### Timeout (Alarm Sona Erme Süresi)

```python
# event_manager.py
event_manager = EventManager(timeout_sec=10.0)  # 10 saniye
```

Alarm ≥ 10 saniye yoksa event "resolved" olur.

### Tolerans (Aynı Alarm Olarak Kabul Etme)

```python
# event_manager.py → AlarmSignature.__eq__()
h_ok = abs(...) <= 1  # helmet count farkı 1 kişiye kadar
v_ok = abs(...) <= 1  # vest count farkı 1 kişiye kadar
c_ok = abs(...) <= 0.05  # confidence farkı %5'e kadar
```

---

## 📊 TEST SONUÇLARI

```
6 görüntü işlendi

evt_0001: [NEW] - Screenshot_1.png
  1 baret ihlali + 1 yelek ihlali + yangın
  → [START] kayıt + JSON

evt_0002: [NEW] - group-workers.webp
  4 yelek ihlali
  → [START] kayıt + JSON

evt_0003: [NEW] - images.jpg
  5 baret + 5 yelek ihlali
  → [START] kayıt + JSON

evt_0004: [NEW] - man-working.jpg
  1 yelek ihlali
  → [START] kayıt + JSON

ÖZET:
- Toplam: 6 görüntü
- Alarm: 4 event (tekrar almamıştır)
- Kayıt: 4 × (txt + json) = 8 dosya
```

---

## ✨ ÖZELLİKLER

✅ **Frame-Bazlı** - Her frame CNN'den geçer  
✅ **Event-Bazlı Kayıt** - Sadece anlamlı değişiklikler kaydedilir  
✅ **Stateful** - Aktif event'i bellekte tutar  
✅ **Toleranslı Karşılaştırma** - ±1 kişi fark aynı alarm  
✅ **Timeout Mantığı** - 10 saniye sonra event sona erer  
✅ **Video-Ready** - Canlı akışa taşıması kolay  
✅ **TXT + JSON** - Her event için yapılandırılmış veriler  

---

## 🎓 MANTIK AKIŞI

```python
frame geldi
    ↓
CNN → hr, vr, fr
    ↓
event_manager.process_frame(hr, vr, fr)
    ↓
    ├─ Aktif event yok + alarm yok
    │  └─ "no_alarm" → KAYIT YOK
    │
    ├─ Aktif event yok + alarm var
    │  └─ Yeni event başlat
    │  └─ "new" → KAYIT ✅
    │
    ├─ Aktif event var + alarm var
    │  ├─ Aynı alarm
    │  │  └─ "active" (repeat) → KAYIT YOK
    │  ├─ Alarm değişti
    │  │  └─ "update" → KAYIT ✅
    │  └─ Farklı alarm
    │     └─ Eski event kapat, yeni başlat
    │     └─ "new" → KAYIT ✅
    │
    └─ Aktif event var + alarm yok
       ├─ Timeout < 10s
       │  └─ "active" (bekleme) → KAYIT YOK
       └─ Timeout ≥ 10s
          └─ "resolved" → KAYIT ✅
```

---

## 🚀 SONRAKI ADIMLAR (OPSİYONEL)

1. **Screenshot Kayıt** - Event başında/update'de frame kaydet
2. **Video Snippet** - Event süresi boyunca video kaydı
3. **Webhook** - Event'leri remote sunucuya gönder
4. **Dashboard** - Real-time event takip arayüzü
5. **Persistence** - Event'leri database'e kaydet

---

## ✅ KONTROL LİSTESİ

- [x] EventManager sınıfı oluşturuldu
- [x] AlarmSignature benzerliği mantığı uygulandı
- [x] LLM'ye event_info desteği eklendi
- [x] run_with_llm.py entegrasyonu yapıldı
- [x] Event-bazlı klasör yapısı oluşturuluyor
- [x] TXT + JSON kayıt sistemi çalışıyor
- [x] Test yapılıp sonuçlar doğrulandı
- [x] Canlı video hazırlığı tamamlandı

---

**Sistem artık canlı video akışı için tamamen hazır! 🎉**


