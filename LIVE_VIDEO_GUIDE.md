# 🎬 Canlı Video Sistemi - Hızlı Başlangıç

## ⚡ 1 Dakikalık Başlangıç

### Kurulum
```bash
# Python 3.9+ gerekli
pip install -r requirements.txt
```

### Offline Modda Test Et (Model yükleme beklemeden)
```bash
python run_live_video.py --offline --video test/images.jpg
```

### Gerçek Modda Çalıştır
```bash
# Kameradan
python run_live_video.py

# Video dosyasından
python run_live_video.py --video test/ppe_test1.mp4

# Ollama LLM aktif değilse offline şu şekilde:
python run_live_video.py --offline
```

---

## 📊 Ne Olduğunu Anlamak

### Sistem Neler Yapıyor?

1. **Her saniyede bir** CNN ajanları çalışır:
   - HelmetAgent → Kaç kişi baret takmıyor?
   - VestAgent → Kaç kişi yelek takmıyor?
   - FireAgent → Yangın var mı?

2. **Event yönetimi**:
   - Aynı alarm tekrar tekrar görülürse hep **aynı event** içinde tutuluyor
   - Alarm değişirse **update** oluşturulur
   - Alarm biterse **resolved** oluşturulur

3. **Kayıt sistemi**:
   - Sadece event değiştiğinde dosya oluşturulur
   - **frame bazlı değil, event bazlı kayıt**
   - Her event kendi klasörü altında: `results/evt_XXXX/`

4. **LLM raporu**:
   - Her event için Ollama/Mistral kullanarak kısa rapor üretilir
   - Event context'i ile birlikte gönderilir

---

## 🎯 Örnek Senaryo

### Sürü: 10 saniye

**Saniye 0-1**: 3 kişi baret takmıyor
```
→ EventManager: "Alarm var, debounce +1"
→ Dosya YOK (henüz 1 frame)
```

**Saniye 1-2**: 3 kişi baret takmıyor (benzer)
```
→ EventManager: "Debounce +2, şimdi event açılsın"
→ evt_0001_start.jpg, evt_0001_start.txt, evt_0001_start.json (KAYIT)
→ LLM: "3 kişi baret takmıyor. Derhal tedbirler alınmalı."
→ Ekran: EVENT: evt_0001 [NEW]
```

**Saniye 2-3**: 4 kişi baret takmıyor (değişti!)
```
→ EventManager: "Aynı event ama sayı değişti → UPDATE"
→ evt_0001_update_01.jpg, evt_0001_update_01.txt (KAYIT)
→ LLM: "4 kişi baret takmıyor. Artış tespit edildi."
→ Ekran: EVENT: evt_0001 [UPDATE], repeat_count=3
```

**Saniye 3-12**: 4 kişi baret takmıyor (değişim yok)
```
→ EventManager: "Aynı event, aynı alarm → ACTIVE (KAYIT YOK)"
→ Dosya oluşturulmaz, sadece repeat_count artır
→ Ekran: EVENT: evt_0001 [ACTIVE], repeat_count=10
```

**Saniye 12-13**: Temiz (alarm yok)
```
→ EventManager: "Alarm kayboldu, timeout başlasın"
→ Dosya YOK (henüz timeout değil)
→ Ekran: EVENT: evt_0001 [ACTIVE], bekleme durumu
```

**Saniye 13-23** (10 saniye timeout): Hala temiz
```
→ EventManager: "Timeout aşıldı → RESOLVED"
→ evt_0001_resolved.txt, evt_0001_resolved.json (KAYIT)
→ LLM: "Olay çözüldü. İş alanı güvenli."
→ Ekran: EVENT: evt_0001 [RESOLVED]
→ Sonra: idle durumuna dön
```

---

## 📂 Sonuçlar Klasörü

### Yapı
```
results/
  evt_0001/
    evt_0001_start.jpg
    evt_0001_start.txt
    evt_0001_start.json
    evt_0001_update_01.jpg
    evt_0001_update_01.txt
    evt_0001_update_01.json
    evt_0001_resolved.txt
    evt_0001_resolved.json
  evt_0002/
    ...
  evt_0003/
    ...
```

### Dosya Türleri

**start.txt** → Olay başladığında
```
============================================================
EVENT: evt_0001 [NEW]
Zaman: 2026-04-13T11:11:49.816903
Tekrar: 1
Süre: 0.0s
============================================================

--- CNN TESPİT VERİSİ ---
helmet_violation_count=5
vest_violation_count=5
fire_detected=no
fire_confidence=0.00

--- LLM GÜVENLİK RAPORU ---
5 kişi baret takmıyor. 5 kişi yelek takmıyor. Derhal tedbirler alınmalı.
```

**update_01.txt** → Olay güncellendiğinde
```
============================================================
EVENT: evt_0001 [UPDATE]
Zaman: 2026-04-13T11:11:52.123456
Tekrar: 4
Süre: 2.3s
============================================================

--- CNN TESPİT VERİSİ ---
helmet_violation_count=6
vest_violation_count=4
fire_detected=no
fire_confidence=0.00

--- LLM GÜVENLİK RAPORU ---
6 kişi baret takmıyor. 4 kişi yelek takmıyor. Baretsiz sayısında artış.
```

**resolved.json** → Olay bittiğinde
```json
{
  "event_id": "evt_0001",
  "event_status": "resolved",
  "timestamp": "2026-04-13T11:12:05.654321",
  "repeat_count": 15,
  "duration_sec": 15.8,
  "alarm": false,
  "llm_report": "Olay çözüldü. İş alanı güvenli duruma döndü.",
  "structured": "helmet_violation_count=0\nvest_violation_count=0\n..."
}
```

---

## 🎮 Kontroller

### Canlı Video Penceresinde
- **ESC**: Çık
- **Ctrl+C**: Durdur

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
2026-04-13 11:12:44,061 - LLM - INFO - LLM raporu isteniyor: frame_35
  [KAYIT] evt_0001/start

  Frame 90 | Events: 1
```

**Ne anlama geliyorsa?**
- **Frame 30**: 30 frame işlendi
- **Events: 1**: Şu ana kadar 1 olay
- **[KAYIT] evt_0001/start**: evt_0001 başladı ve dosya kaydedildi

---

## ⚙️ Ayarları Değiştirmek

### Event Timeout'unu Değiştir

**run_live_video.py** satırı ~167:
```python
event_manager = EventManager(timeout_sec=10.0)  # 10 saniye
```

10'u başka bir değere çevir (örn: 5.0)

### Debounce (alarm açılma eşiği)

Şu an hardcoded: 2 ardışık frame

**event_manager.py** güncellenebilir (gelecek sprint)

---

## 🐛 Sorun Giderme

### "Video açılamadı"
```bash
# test/ klasöründe dosya var mı kontrol et
ls test/
```

### "Ollama bağlantısı başarısız"
```bash
# Ollama sunucusu çalışıyor mı?
curl http://localhost:11434/api/tags

# Çalışmıyorsa offline modda test et
python run_live_video.py --offline
```

### "Modeller bulunamadı"
```bash
# models/ klasöründe şunlar var mı?
# - yihong.pt (helmet & vest)
# - fire_best.pt

# Yoksa FTP/Roboflow'dan download et
python config/config_loader.py
```

---

## 🎯 Ne Sonra?

1. **Canlı kameraya bağla**: `python run_live_video.py`
2. **LLM raporlarını özelleştir**: `llm/llm_coordinator.py` → Prompt'u düzenle
3. **Event timeout'unu ayarla**: İş alanınıza göre (10s ideal başlangıç)
4. **Dashboard ekle**: `results/` klasöründen raporları oku ve web göster

---

## 📖 Daha Fazla Bilgi

- **EVENT_SYSTEM_GUIDE.md**: Detaylı teknik dokümantasyon
- **README.md**: Proje genel bilgileri
- **event_manager.py**: Source code (AlarmSignature, ActiveEvent)
- **llm/llm_coordinator.py**: LLM prompt ve format

---

## 💡 İpuçları

1. **Offline modda başla** → Modelleri yüklemeden mantığı test et
2. **Gerçek videoda 5+ dakika çalıştır** → Event lifecycle'ı tamamını gör
3. **results/ klasörünü kontrol et** → JSON'ları okuyarak event detaylarını anla
4. **Prompt'u kustomize et** → LLM'nin halucinasyon yapmaması için sıkı tut

---

**Sorular mı? event_manager.py ve EVENT_SYSTEM_GUIDE.md'yi oku!** 🚀

