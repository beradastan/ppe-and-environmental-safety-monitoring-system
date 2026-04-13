# Event-Based Factory Safety System

## 📋 Genel Bakış

Sistem, canlı video akışında güvenlik olaylarını **event-bazlı** olarak yönetir:

- **Frame işleme**: Saniyede 1 kez CNN agentleri çalışır
- **Alarm imzası**: Her frame için `AlarmSignature` oluşturulur
- **Event yönetimi**: Aynı alarmlar aynı event içinde gruplandırılır
- **Kayıt mantığı**: Sadece olay başladığında, değiştiğinde ve bittiğinde dosya oluşturulur
- **LLM raporu**: Event context'i ile birlikte oluşturulur

---

## 🔧 Mimari Bileşenler

### 1. **AlarmSignature** (event_manager.py)
```python
@dataclass
class AlarmSignature:
    helmet_violation_count: int     # Baretsiz kişi sayısı
    vest_violation_count: int       # Yeleksiz kişi sayısı
    fire_detected: bool             # Yangın tespit edildi mi?
    fire_confidence: float          # Yangın emin olma derecesi
```

**Amaç**: Bir andaki tehlike özeti. Alarm benzerliğini karşılaştırmak için kullanılır.

### 2. **ActiveEvent** (event_manager.py)
```python
@dataclass
class ActiveEvent:
    event_id: str                   # evt_0001, evt_0002 vb
    start_time: float               # Event başlama zamanı (timestamp)
    last_seen: float                # Son gözlenen zaman
    alarm_signature: AlarmSignature # Son alarm özeti
    repeat_count: int               # Kaç kez tekrarlandı?
    status: str                     # "new", "active", "update", "resolved"
    change_reason: str              # Değişim sebebi
```

**Amaç**: Aktif olay bilgilerini tutmak.

### 3. **EventManager** (event_manager.py)
Ana logic sınıfı. Her frame için:
1. CNN sonuçlarından AlarmSignature üretir
2. Aktif event varsa benzerlik kontrol eder
3. Yeni event açar, günceller veya kapatır
4. Çıktı: Kayıt gerekli mi, hangi statü?

**Key Method**: `process_frame(helmet_result, vest_result, fire_result)`

---

## 🎯 Event State Machine

```
idle (başlangıç)
  ↓
  ├─→ Alarm görüldü (1 frame) → Debounce sayacı +1
  │
  └─→ Alarm görüldü (2 frames) → EVENT AÇILSIN (status="new")
                                   ↓
                          ┌────────┴────────┐
                          ↓                 ↓
                    Alarm sürüyor     Alarm kayboldu
                       (aktif)        (3 frame temiz)
                          │                 │
                          ├─ Aynı alarm  ├─→ EVENT KAPANSYN
                          │   (repeat)      (status="resolved")
                          │                 
                          ├─ Alarm değişti
                          │   (update)
                          │
                          └─ Yeni alarm
                              (new event)
```

---

## 🔍 Alarm Benzerliği Mantığı

Aynı event mi karar vermek için:
```
1. helmet_violation_count farkı ≤ 1      ✓
2. vest_violation_count farkı ≤ 1        ✓
3. fire_detected değeri aynı mı?         ✓
4. fire_confidence farkı ≤ 0.05          ✓
```

Eğer tüm koşullar sağlanırsa **aynı event** kabul edilir.

**Update durumu**: Aynı event ama detaylarda değişiklik:
- Sayı artmış/azalmış
- Fire confidence değişmiş

---

## 📁 Dosya Yapısı ve Kayıt Kuralları

### Kaydedilen Dosyalar

Olay başladığında:
```
results/
  evt_0001/
    evt_0001_start.jpg              # Screenshot
    evt_0001_start.txt              # Text rapor
    evt_0001_start.json             # JSON veri
```

Olay güncellendiğinde:
```
results/
  evt_0001/
    evt_0001_start.jpg
    evt_0001_start.txt
    evt_0001_start.json
    evt_0001_update_01.jpg          # İlk güncelleme
    evt_0001_update_01.txt
    evt_0001_update_01.json
    evt_0001_update_02.jpg          # İkinci güncelleme
    evt_0001_update_02.txt
    evt_0001_update_02.json
```

Olay kapatıldığında:
```
results/
  evt_0001/
    ... (tüm start ve update dosyaları)
    evt_0001_resolved.txt           # Kapanış raporu
    evt_0001_resolved.json
```

### Kayıt Triggerı

| Status | Kayıt? | Açıklama |
|--------|--------|----------|
| idle | ❌ | Alarm yok, kaydedilmez |
| new | ✅ | Yeni olay açıldı, kaydedilir |
| active | ❌ | Olay sürüyor ama değişiklik yok, KAYDEDILMEZ |
| update | ✅ | Olay güncellenmiş, kaydedilir |
| resolved | ✅ | Olay kapandı, kaydedilir |

---

## 📊 JSON Örneği

### start.json
```json
{
  "event_id": "evt_0001",
  "event_status": "new",
  "timestamp": "2026-04-13T11:11:49.816903",
  "repeat_count": 1,
  "duration_sec": 0.0,
  "alarm": true,
  "llm_report": "5 kişi baret takmıyor. 5 kişi yelek takmıyor. Derhal tedbirler alınmalı.",
  "structured": "helmet_violation_count=5\nvest_violation_count=5\nfire_detected=no\nfire_confidence=0.00"
}
```

### update_01.json
```json
{
  "event_id": "evt_0001",
  "event_status": "update",
  "timestamp": "2026-04-13T11:11:52.123456",
  "repeat_count": 4,
  "duration_sec": 2.3,
  "alarm": true,
  "llm_report": "6 kişi baret takmıyor. 4 kişi yelek takmıyor. Yeleksiz personel sayısında azalış.",
  "structured": "helmet_violation_count=6\nvest_violation_count=4\nfire_detected=no\nfire_confidence=0.00"
}
```

### resolved.json
```json
{
  "event_id": "evt_0001",
  "event_status": "resolved",
  "timestamp": "2026-04-13T11:12:05.654321",
  "repeat_count": 15,
  "duration_sec": 15.8,
  "alarm": false,
  "llm_report": "Olay çözüldü. İş alanı güvenli duruma döndü.",
  "structured": "helmet_violation_count=0\nvest_violation_count=0\nfire_detected=no\nfire_confidence=0.00"
}
```

---

## 🚀 Kullanım

### Canlı Video Modu
```bash
python run_live_video.py
```
- Varsayılan kamera: 0 (WebCam)
- Timeout: 10 saniye (event kapanmadan önce bekleme süresi)

### Video Dosyasıyla
```bash
python run_live_video.py --video test/ppe_test1.mp4
```

### Offline Modda (Ollama olmadan)
```bash
python run_live_video.py --offline
```
- Mock LLM raporu kullanılır
- Hızlı test için ideal

### Kamera Seçme
```bash
python run_live_video.py --camera 1
```

---

## 📝 LLM Prompt Yapısı

LLM'ye şu bilgiler gönderilir (minimal format):

```
event_id=evt_0001
event_status=new
helmet_violation_count=5
vest_violation_count=5
fire_detected=no
fire_confidence=0.00
repeat_count=1
duration_sec=0.0
change_reason=initial_alarm
```

**Prompt**: "Sadece bu veriye dayanarak 1-2 kısa cümle yaz. Sayıları doğru kullan, yeni bilgi uydurma."

---

## ⚙️ Ayarlar (event_manager.py)

```python
manager = EventManager(
    timeout_sec=10.0,           # Event kapanmadan önce bekle
    event_id_prefix="evt"       # Event ID öneki
)
```

- **timeout_sec**: Şu an için hardcoded 10 saniye. Video ortasında alarm kaybolsa, 10 saniye beklene sonra kapatılır.
- **debounce**: 2 ardışık frame (2 saniye) alarm gereği event açılır

---

## 🔄 İş Akışı (run_live_video.py)

```
1. Video kaynağı aç
   ↓
2. Frame loop
   ├─ Her saniyede bir:
   │  ├─ CNN çalıştır (helmet, vest, fire)
   │  ├─ event_manager.process_frame(...)
   │  ├─ if should_save:
   │  │  ├─ LLM raporu üret
   │  │  ├─ Dosya kaydet (txt/json/jpg)
   │  └─ Ekranda event info göster
   └─ Next frame
   ↓
3. Video sonu
   ├─ Aktif event varsa kapla (resolved)
   └─ Dosya kaydet
```

---

## 📊 Sistem Performansı

Video sırasında görünen istatistikler:
- **Frame N**: N. frame işleniyor
- **Events: M**: Toplam olay sayısı

### Örnek Çıktı
```
  Frame 30  Events: 0
  Frame 60  Events: 1
  Frame 90  Events: 1
  Frame 120 Events: 2
```

---

## 🐛 Debugging

### Log çıktısı kontrol et
```bash
python run_live_video.py --offline 2>&1 | Out-String
```

### LLM raporu debug et
```python
from llm.llm_coordinator import OllamaLLMCoordinator

llm = OllamaLLMCoordinator(offline_mode=True)

# Alarm sonucu
hr = {"warning_count": 2}
vr = {"warning_count": 1}
fr = {"detection_count": 0}

report = llm.generate_alarm_report(hr, vr, fr)
print(report["report"])
```

---

## 🔗 İlgili Dosyalar

- **event_manager.py**: Event logic
- **llm/llm_coordinator.py**: LLM raporlama
- **run_live_video.py**: Ana canlı video loop
- **pipeline/**: Detection pipeline (CNN agents)
- **agents/**: Helmet, Vest, Fire detection agents

---

## ✅ Kontrol Listesi

- [x] Event state machine kuruldu
- [x] Alarm benzerliği mantığı
- [x] Event başlatma (new)
- [x] Event güncelleme (update)
- [x] Event kapatma (resolved)
- [x] Dosya kayıt sistemi (txt/json/jpg)
- [x] LLM event context desteği
- [x] Canlı video akışı
- [x] Offline test modu

---

## 🎯 Gelecek Geliştirmeler

- [ ] Web dashboard (event timeline)
- [ ] Database desteği (SQLite/PostgreSQL)
- [ ] Alert notification (email/SMS)
- [ ] Event replay sistemi
- [ ] Perimeter detection (bölge bazlı alarm)

