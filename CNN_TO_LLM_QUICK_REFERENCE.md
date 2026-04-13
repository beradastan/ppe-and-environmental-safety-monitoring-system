# CNN → LLM Veri Yapısı — QUICK REFERENCE

## 🎯 En Hızlı Referans

### CNN Çıktı Özeti

| Agent | Model | Classes | Detections | Warnings | Alarm Koşulu |
|-------|-------|---------|------------|----------|--------------|
| **HelmetAgent** | yihong.pt | class 0: Hardhat (✅)<br>class 2: NO-Hardhat (⚠️) | Baret giyenler | Baretsiz | `warning_count > 0` |
| **VestAgent** | yihong.pt | class 7: Safety Vest (✅)<br>class 4: NO-Safety Vest (⚠️) | Yelek giyen | Yeleksiz | `warning_count > 0` |
| **FireAgent** | fire_best.pt | class 0: fire (⚠️) | Yangın detection | - | `detection_count > 0` |

---

## 📦 Dictionary Yapıları

### HelmetAgent Output

```python
{
    "agent": "HelmetAgent",
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.95,
            "class_id": 0,
            "label": "Hardhat",
            "agent": "HelmetAgent",
            "status": "safe"
        },
        # ... max 5
    ],
    "warnings": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.87,
            "class_id": 2,
            "label": "NO-Hardhat",
            "agent": "HelmetAgent",
            "status": "warning"
        },
        # ... max 5
    ],
    "detection_count": 3,
    "warning_count": 1,
    "has_issue": True
}
```

### VestAgent Output

```python
{
    "agent": "VestAgent",
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.94,
            "class_id": 7,
            "label": "Safety Vest",
            "agent": "VestAgent",
            "status": "safe"
        },
        # ... max 5
    ],
    "warnings": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.89,
            "class_id": 4,
            "label": "NO-Safety Vest",
            "agent": "VestAgent",
            "status": "warning"
        },
        # ... max 5
    ],
    "detection_count": 2,
    "warning_count": 2,
    "has_issue": True
}
```

### FireAgent Output

```python
{
    "agent": "FireAgent",
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.78,
            "class_id": 0,
            "label": "fire",
            "agent": "FireAgent",
            "severity": "high"  # "low" | "medium" | "high"
        }
        # SADECE 1 (NMS applied)
    ],
    "detection_count": 1,
    "alert": True
}
```

---

## 🚨 Alarm Tetikleme

```python
alarm = (
    helmet_result.get("warning_count", 0) > 0    # Baretsiz?
    or vest_result.get("warning_count", 0) > 0   # Yeleksiz?
    or fire_result.get("detection_count", 0) > 0 # Yangın?
)

if alarm:
    # LLM çağrılır
    report = llm.generate_alarm_report(helmet_result, vest_result, fire_result)
else:
    # LLM çağrılmaz
    report = {
        "alarm": False,
        "llm_called": False,
        "report": "Sahne guvenli. LLM raporu uretilmedi."
    }
```

---

## 📝 LLM'ye Gönderilen Metin Formatı

```
=== FABRIKA GUVENLIK TESPIT RAPORU ===
Zaman        : 2026-04-12 14:35:22
Goruntu      : fabrika_001.jpg

[BARET (HARDHAT) DURUMU]
  Uyumlu  (baret takmis)   : {h_safe} kisi
  IHLAL   (baretsiz)       : {h_viol} kisi <-- ALARM (if h_viol > 0)

[GUVENLIK YELEGI (VEST) DURUMU]
  Uyumlu  (yelek giymis)   : {v_safe} kisi
  IHLAL   (yeleksiz)       : {v_viol} kisi <-- ALARM (if v_viol > 0)

[YANGIN DURUMU]
  Yangin tespit edildi     : {'EVET' + f_guven if f_count > 0 else 'HAYIR'} <-- ALARM (if f_count > 0)

[ALARM TETIKLENDI] : {'EVET' if ihlaller else 'HAYIR'}
[IHLALLER]         :
  - {ihlal listesi}
```

---

## 🤖 LLM Prompt Yapısı

```python
prompt = """
Sen bir endüstriyel iş sağlığı ve güvenliği uzmanı yapay zekasın.
Aşağıda bir fabrikada çalışan bilgisayarlı görü sisteminin tespit raporu yer almaktadır.
Bu rapora dayanarak KISA bir güvenlik olay raporu yaz (3-5 cümle, Türkçe).
YALNIZCA tespit edilen ihlallere odaklan. Net ve profesyonel ol.
Ham sayıları tekrar etme — yorumla ve açıkla.
Raporunu tek bir somut ve acil eylem önerisiyle bitir.

{structured_text_above}

Güvenlik Raporu:
"""
```

---

## 📊 Final Report (Output)

```python
{
    "alarm": True,                                      # Alarm tetiklendi mi?
    "llm_called": True,                                 # LLM çağırıldı mı?
    "report": "Fabrikanın A bölgesinde bir kişi...",   # LLM raporu
    "structured": "=== FABRIKA GUVENLIK...",           # Ham tespit verisi
    "image": "fabrika_001.jpg",                         # Görüntü adı
    "timestamp": "2026-04-12T14:35:22.123456"          # ISO timestamp
}
```

---

## 🔍 Field Açıklamaları

| Field | Kaynak | Tip | Açıklama |
|-------|--------|-----|----------|
| `bbox` | YOLO | `[x1, y1, x2, y2]` | Bounding box koordinatları (pixel) |
| `confidence` | YOLO | `0.0-1.0` | Güven skoru (yüksek = daha güvenilir) |
| `class_id` | YOLO | `int` | Sınıf ID (HelmetAgent: 0/2, VestAgent: 7/4, FireAgent: 0) |
| `label` | Agent | `str` | İnsan okunabilir sınıf adı |
| `agent` | Agent | `str` | Hangi ajan döndü |
| `status` | Agent | `"safe"/"warning"` | İyi mi / Kötü mü |
| `severity` | FireAgent | `"low"/"medium"/"high"` | Yangının şiddeti (bbox alanına göre) |
| `detection_count` | Agent | `int` | Pozitif tespitlerin sayısı |
| `warning_count` | Agent | `int` | Uyarı tespitlerin sayısı (⚠️) |
| `has_issue` | Agent | `bool` | `warning_count > 0` |
| `alert` | FireAgent | `bool` | `detection_count > 0` |

---

## 🎮 Pratik Kullanım

### 1. Deteksiyon Çalıştır

```python
from agents.specific_agents import HelmetAgent, VestAgent, FireAgent
import cv2

# Ajanları yükle
helmet = HelmetAgent(device="cpu")
vest = VestAgent(device="cpu")
fire = FireAgent(device="cpu")

# Görüntü oku
img = cv2.imread("fabrika_001.jpg")

# Deteksiyonlar
hr = helmet.detect(img)  # {"detection_count": 3, "warning_count": 1, ...}
vr = vest.detect(img)    # {"detection_count": 2, "warning_count": 1, ...}
fr = fire.detect(img)    # {"detection_count": 0, "alert": False, ...}
```

### 2. Alarm Kontrol Et

```python
alarm = (
    hr["warning_count"] > 0
    or vr["warning_count"] > 0
    or fr["detection_count"] > 0
)

print(f"Alarm: {alarm}")  # True/False
```

### 3. LLM Raporu Al

```python
from llm.llm_coordinator import OllamaLLMCoordinator

llm = OllamaLLMCoordinator(offline_mode=False)

report = llm.generate_alarm_report(
    helmet_result=hr,
    vest_result=vr,
    fire_result=fr,
    image_name="fabrika_001.jpg"
)

print(report["report"])  # LLM tarafından üretilen Türkçe metin
```

---

## ⚙️ Ayarlanabilir Parametreler

| Parametre | Varsayılan | Dosya | Açıklama |
|-----------|------------|-------|----------|
| `helmet_confidence_threshold` | 0.20 | specific_agents.py | Helmet tespiti için güven eşiği |
| `vest_confidence_threshold` | 0.25 | specific_agents.py | Vest tespiti için güven eşiği |
| `fire_confidence_threshold` | 0.5 | specific_agents.py | Fire tespiti için güven eşiği |
| `helmet_imgsz` | 640 | specific_agents.py | Helmet resolution |
| `vest_imgsz` | 640-1280 | specific_agents.py | Vest resolution (adaptive) |
| `fire_imgsz` | 640 | specific_agents.py | Fire resolution |
| `fire_nms_iou` | 0.3 | specific_agents.py | Fire NMS IoU threshold |
| `llm_model` | "mistral" | llm_coordinator.py | Ollama model adı |
| `llm_temperature` | 0.2 | llm_coordinator.py | LLM temperature (düşük = daha deterministik) |

---

## 🐛 Debug İpuçları

### Deteksiyon Sayıları Düşük

```python
# Confidence threshold'i düşür
helmet = HelmetAgent(confidence_threshold=0.15)
vest = VestAgent(confidence_threshold=0.20)
```

### Yanlış İhlal Tespiti

```python
# LLM'e gönderilen metni kontrol et
structured = llm.format_for_llm(hr, vr, fr)
print(structured)  # CNN verilerinin nasıl formatlandığını gör
```

### LLM Yanıt Vermiyor

```python
# Offline mode ile test et
llm = OllamaLLMCoordinator(offline_mode=True)
# Ollama servisi kontrol et: http://localhost:11434
```

---

## 📂 İlgili Dosyalar

| Dosya | Amaç |
|-------|------|
| `agents/specific_agents.py` | CNN ajanlarının `postprocess()` → CNN çıktı formatı |
| `llm/llm_coordinator.py` | `format_for_llm()` → Metin formatı, `generate_alarm_report()` → Prompt & alarm logic |
| `run_with_llm.py` | Complete pipeline kullanımı |

---

## 🚀 Başlangıç

```bash
# Offline test (LLM gerekmiyor)
python run_with_llm.py --offline

# Gerçek LLM (Ollama)
python run_with_llm.py

# Özel klasör
python run_with_llm.py path/to/images/
```

Sonuçlar → `results/` klasöründe `alarm_report_*.txt` dosyaları

