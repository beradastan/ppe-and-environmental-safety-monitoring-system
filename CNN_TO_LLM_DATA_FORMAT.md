# CNN Ajanlarından LLM'ye Veri Akışı Analizi

## 📊 Sistem Mimarisi

```
HelmetAgent (CNN)  ─┐
VestAgent (CNN)    ─┼──→ generate_alarm_report() ──→ LLM Coordinator ──→ Ollama/Mistral
FireAgent (CNN)    ─┘                                   (format_for_llm)
```

---

## 1️⃣ CNN AJANLARININ ÇIKTI FORMATI

### A. HelmetAgent (Baret Tespiti)

**Model:** `models/yihong.pt` (YOLO11n)

**Çıktı Dictionary Yapısı:**
```python
{
    "agent": "HelmetAgent",
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],           # Bounding box koordinatları
            "confidence": 0.95,                  # Güven skoru (0-1)
            "class_id": 0,                       # 0 = Hardhat (baret GİYMİŞ)
            "label": "Hardhat",                  # İnsan okunabilir etiket
            "agent": "HelmetAgent",
            "status": "safe"                     # durum: safe / warning
        },
        # ... En fazla 5 detection (confidence'a göre sıralanmış)
    ],
    "warnings": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.87,
            "class_id": 2,                       # 2 = NO-Hardhat (baretsiz) ⚠️
            "label": "NO-Hardhat",
            "agent": "HelmetAgent",
            "status": "warning"
        },
        # ... En fazla 5 warning
    ],
    "detection_count": 3,                        # Baret giymiş kişi sayısı
    "warning_count": 1,                          # Baretsiz kişi sayısı (ALARM TESTİ)
    "has_issue": True                            # warning_count > 0
}
```

**Sınıf Haritası:**
```
class_id 0 → "Hardhat"     (pozitif ✅)
class_id 2 → "NO-Hardhat"  (uyarı ⚠️)
```

---

### B. VestAgent (Güvenlik Yelegi Tespiti)

**Model:** `models/yihong.pt` (YOLO11n)

**Çıktı Dictionary Yapısı:**
```python
{
    "agent": "VestAgent",
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.92,
            "class_id": 7,                       # 7 = Safety Vest (yelek GİYMİŞ)
            "label": "Safety Vest",
            "agent": "VestAgent",
            "status": "safe"
        },
        # ... En fazla 5 detection
    ],
    "warnings": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.88,
            "class_id": 4,                       # 4 = NO-Safety Vest (yeleksiz) ⚠️
            "label": "NO-Safety Vest",
            "agent": "VestAgent",
            "status": "warning"
        },
        # ... En fazla 5 warning
    ],
    "detection_count": 2,                        # Yelek giymiş kişi sayısı
    "warning_count": 1,                          # Yeleksiz kişi sayısı (ALARM TESTİ)
    "has_issue": True
}
```

**Sınıf Haritası:**
```
class_id 7 → "Safety Vest"     (pozitif ✅)
class_id 4 → "NO-Safety Vest"  (uyarı ⚠️)
```

---

### C. FireAgent (Yangın Tespiti)

**Model:** `models/fire_best.pt` (YOLO)

**Çıktı Dictionary Yapısı:**
```python
{
    "agent": "FireAgent",
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.78,
            "class_id": 0,                       # 0 = fire (yangın) ⚠️
            "label": "fire",
            "agent": "FireAgent",
            "severity": "high"                   # "low", "medium", "high"
        }
        # SADECE 1 EN İYİ detection (NMS uygulanmış)
    ],
    "detection_count": 1,                        # Yangın deteksiyon sayısı (ALARM TESTİ)
    "alert": True                                # detection_count > 0
}
```

**Not:** 
- FireAgent sadece `class_id = 0` (fire) barındırır
- Smoke (class 1) sayılmaz
- En fazla 1 detection döner (NMS optimizasyonu)
- Severity hesaplaması:
  - area > 100000 → "high"
  - area > 50000 → "medium"
  - else → "low"

---

## 2️⃣ ALARM TÜREDİŞİ KRİTERİ

LLM çağırılır mı diye karar logic:

```python
alarm = (
    helmet_result.get("warning_count", 0) > 0   # Baretsiz biri var mı?
    or vest_result.get("warning_count", 0) > 0  # Yeleksiz biri var mı?
    or fire_result.get("detection_count", 0) > 0  # Yangın var mı?
)
```

**Alarm = True** → LLM çağırılır ve rapor üretilir  
**Alarm = False** → LLM çağırılmaz, sadece "Güvenli sahne" denir

---

## 3️⃣ LLM COORDINATOR'ÜN FORMAT_FOR_LLM() FONKSİYONU

CNN çıktılarını LLM'in anlayacağı Türkçe metne çevirme:

### Input:
```python
helmet_result = {
    "detection_count": 3,
    "warning_count": 1,
    "detections": [...],
    "warnings": [...]
}

vest_result = {
    "detection_count": 2,
    "warning_count": 1,
    "detections": [...],
    "warnings": [...]
}

fire_result = {
    "detection_count": 1,
    "detections": [{"confidence": 0.78}]
}

image_name = "fabrika_001.jpg"
```

### Çevirim Mantığı:

```python
h_safe = helmet_result.get("detection_count", 0)    # 3 kişi baret takmış
h_viol = helmet_result.get("warning_count", 0)      # 1 kişi baretsiz
v_safe = vest_result.get("detection_count", 0)      # 2 kişi yelek takmış  
v_viol = vest_result.get("warning_count", 0)        # 1 kişi yeleksiz
f_count = fire_result.get("detection_count", 0)     # 1 yangın tespit

# Yangın güven skoru
f_guven = ""
if f_count > 0:
    f_guven = f" (guven skoru: {fire_result['detections'][0]['confidence']:.2f})"
    # Örnek: " (guven skoru: 0.78)"

# İhlal listesi
ihlaller = []
if h_viol > 0:
    ihlaller.append("1 kisi BARETSIZ calistirildi")
if v_viol > 0:
    ihlaller.append("1 kisi GUVENLIK YELEKSIZ calistirildi")
if f_count > 0:
    ihlaller.append("YANGIN tespit edildi (guven skoru: 0.78)")
```

### Output (LLM'e gönderilen metin):

```
=== FABRIKA GUVENLIK TESPIT RAPORU ===
Zaman        : 2026-04-12 14:35:22
Goruntu      : fabrika_001.jpg

[BARET (HARDHAT) DURUMU]
  Uyumlu  (baret takmis)   : 3 kisi
  IHLAL   (baretsiz)       : 1 kisi <-- ALARM

[GUVENLIK YELEGI (VEST) DURUMU]
  Uyumlu  (yelek giymis)   : 2 kisi
  IHLAL   (yeleksiz)       : 1 kisi <-- ALARM

[YANGIN DURUMU]
  Yangin tespit edildi     : EVET (guven skoru: 0.78) <-- ALARM

[ALARM TETIKLENDI] : EVET
[IHLALLER]         :
  - 1 kisi BARETSIZ calistirildi
  - 1 kisi GUVENLIK YELEKSIZ calistirildi
  - YANGIN tespit edildi (guven skoru: 0.78)
```

---

## 4️⃣ LLM PROMPT YAPISI

Bu metin yukarıdaki struktureli rapor + sistem instruksiyonu şeklinde LLM'e gönderilir:

```python
prompt = (
    "Sen bir endüstriyel iş sağlığı ve güvenliği uzmanı yapay zekasın.\n"
    "Aşağıda bir fabrikada çalışan bilgisayarlı görü sisteminin tespit raporu yer almaktadır.\n"
    "Bu rapora dayanarak KISA bir güvenlik olay raporu yaz (3-5 cümle, Türkçe).\n"
    "YALNIZCA tespit edilen ihlallere odaklan. Net ve profesyonel ol.\n"
    "Ham sayıları tekrar etme — yorumla ve açıkla.\n"
    "Raporunu tek bir somut ve acil eylem önerisiyle bitir.\n\n"
    f"{structured}\n"
    "Güvenlik Raporu:"
)
```

### LLM Cevabı Örneği:

```
Fabrikanın A bölgesinde bir kişi koruyucu başlık olmadan çalışmaktadır. 
Ek olarak, B bölgesinde iki işçi güvenlik yeledni takmadan görevlerini 
sürdürmektedir. Bu durum yaralanma riskini önemli ölçüde arttırmaktadır. 
Derhal bu alanlar kapatılmalı ve eksik KKD malzemeleri sağlanmalıdır.
```

---

## 5️⃣ ÇIKTI RAPOR YAPISI

LLM coordinator `generate_alarm_report()` döndürür:

```python
{
    "alarm": True,                                    # Alarm tetiklendi mi?
    "llm_called": True,                               # LLM çağırıldı mı?
    "report": "Fabrikanın A bölgesinde...",           # LLM'in Türkçe raporu
    "structured": "=== FABRIKA GUVENLIK TESPIT RAPORU ===\n...", # Ham tespit verisi
    "image": "fabrika_001.jpg",
    "timestamp": "2026-04-12T14:35:22.123456"
}
```

---

## 📋 Veri Akışı Özeti

```
[CNN Çalıştırma]
    ↓
helmet_agent.detect(image) → helmet_result {detection_count, warning_count, detections, warnings}
vest_agent.detect(image)   → vest_result   {detection_count, warning_count, detections, warnings}
fire_agent.detect(image)   → fire_result   {detection_count, detections, severity}
    ↓
[Alarm Kontrolü]
    h_warning > 0 || v_warning > 0 || f_detection > 0?
    ↓
    EVET → LLM çağır | HAYIR → "Güvenli sahne"
    ↓
[format_for_llm() → Yapılandırılmış Metin]
    ↓
[Prompt + Sistem Instruksiyonu]
    ↓
[Ollama/Mistral → Türkçe Rapor]
    ↓
[Rapor Kaydı]
    results/ klasörüne txt dosyası
```

---

## 🔄 Değişiklik Noktaları

Eğer CNN ajanlarının çıktı formatını değiştirmek istersen:

1. **specific_agents.py** → `postprocess()` metodlarını düzenle
   - `detections` listesi yapısını değiştir
   - `detection_count` / `warning_count` hesaplamasını değiştir

2. **llm_coordinator.py** → `format_for_llm()` metodunu güncelle
   - CNN çıktılarını nasıl okuduğunu değiştir
   - LLM'e gönderilen metin formatını değiştir

3. **Uyarı:** Eğer warning/detection field isimleri değişirse, alarm kontrolü (`generate_alarm_report()` içindeki `alarm = ...` satırı) da güncellenmelidir.

---

## 🧪 Test Etme

```bash
# Offline (mock LLM) ile test
python run_with_llm.py --offline

# Gerçek Ollama ile test
python run_with_llm.py
```

results/ klasöründe üretilen raporları kontrol et.

