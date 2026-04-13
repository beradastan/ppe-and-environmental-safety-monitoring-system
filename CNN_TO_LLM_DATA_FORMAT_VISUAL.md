# CNN → LLM Veri Akışı — Visual Rehber

## 1. HELMET AGENT ÇIKTI DETAYLI ÖRNEK

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: fabrika_001.jpg (1920x1080)                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │ HelmetAgent.detect(img)   │
         │ model: yihong.pt (YOLO11n)│
         └───────────────┬───────────┘
                         │
         ┌───────────────▼─────────────────┐
         │ Preprocess:                     │
         │ - Resize → 640×640              │
         │ - CLAHE Contrast (LAB)          │
         └───────────────┬─────────────────┘
                         │
         ┌───────────────▼─────────────────┐
         │ YOLO İnference                  │
         │ conf_threshold: 0.20            │
         └───────────────┬─────────────────┘
                         │
         ┌───────────────▼─────────────────────────────────────┐
         │ Postprocess:                                        │
         │ - class 0 (Hardhat) → detections[] (status="safe") │
         │ - class 2 (NO-Hardhat) → warnings[] (status="warn")│
         │ - Confidence'a göre sırala, max 5 tut              │
         └───────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────────────┐
         │ ÇIKTI DICTIONARY:                                       │
         │                                                         │
         │ {                                                      │
         │   "agent": "HelmetAgent",                             │
         │   "detections": [                                     │
         │     {                                                 │
         │       "bbox": [123, 45, 234, 156],                   │
         │       "confidence": 0.95,                            │
         │       "class_id": 0,                                 │
         │       "label": "Hardhat",                            │
         │       "status": "safe"                               │
         │     },                                               │
         │     {                                                │
         │       "bbox": [456, 78, 567, 289],                  │
         │       "confidence": 0.92,                            │
         │       "class_id": 0,                                 │
         │       "label": "Hardhat",                            │
         │       "status": "safe"                               │
         │     },                                               │
         │     {                                                │
         │       "bbox": [789, 123, 890, 334],                │
         │       "confidence": 0.88,                            │
         │       "class_id": 0,                                 │
         │       "label": "Hardhat",                            │
         │       "status": "safe"                               │
         │     }                                                │
         │   ],                                                 │
         │   "warnings": [                                      │
         │     {                                                │
         │       "bbox": [1012, 200, 1123, 411],              │
         │       "confidence": 0.87,                            │
         │       "class_id": 2,           👈 ALARM TRIGGER     │
         │       "label": "NO-Hardhat",                         │
         │       "status": "warning"                            │
         │     }                                                │
         │   ],                                                 │
         │   "detection_count": 3,        👈 Baret giyenler    │
         │   "warning_count": 1,          👈 Baretsizler       │
         │   "has_issue": true            👈 alarm logic       │
         │ }                                                    │
         └──────────────────────────────────────────────────────┘
```

---

## 2. VEST AGENT ÇIKTI DETAYLI ÖRNEK

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: group_workers.webp (1500×1100)                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │ VestAgent.detect(img)     │
         │ model: yihong.pt (YOLO11n)│
         └───────────────┬───────────┘
                         │
         ┌───────────────▼──────────────────────────┐
         │ Preprocess:                              │
         │ - Resize → 1280×1280                     │
         │   (>1000px için adaptive resolution)    │
         │ - Basit resize (yihong kendi normalize) │
         └───────────────┬──────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────┐
         │ YOLO İnference (imgsz=1280)              │
         │ conf_threshold: 0.25                     │
         └───────────────┬──────────────────────────┘
                         │
         ┌───────────────▼─────────────────────────────────────┐
         │ Postprocess:                                        │
         │ - class 7 (Safety Vest) → detections[]             │
         │ - class 4 (NO-Safety Vest) → warnings[]            │
         │ - Confidence'a göre sırala, max 5 tut              │
         └───────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────────────────────────────┐
         │ ÇIKTI DICTIONARY:                                       │
         │                                                         │
         │ {                                                      │
         │   "agent": "VestAgent",                              │
         │   "detections": [                                    │
         │     {                                                │
         │       "bbox": [234, 156, 345, 367],                │
         │       "confidence": 0.94,                            │
         │       "class_id": 7,                                │
         │       "label": "Safety Vest",                       │
         │       "status": "safe"                               │
         │     },                                               │
         │     {                                                │
         │       "bbox": [456, 200, 567, 411],                │
         │       "confidence": 0.91,                            │
         │       "class_id": 7,                                │
         │       "label": "Safety Vest",                       │
         │       "status": "safe"                               │
         │     }                                                │
         │   ],                                                 │
         │   "warnings": [                                      │
         │     {                                                │
         │       "bbox": [678, 250, 789, 461],                │
         │       "confidence": 0.89,                            │
         │       "class_id": 4,           👈 ALARM TRIGGER     │
         │       "label": "NO-Safety Vest",                    │
         │       "status": "warning"                            │
         │     },                                               │
         │     {                                                │
         │       "bbox": [900, 300, 1011, 512],               │
         │       "confidence": 0.85,                            │
         │       "class_id": 4,           👈 ALARM TRIGGER     │
         │       "label": "NO-Safety Vest",                    │
         │       "status": "warning"                            │
         │     }                                                │
         │   ],                                                 │
         │   "detection_count": 2,        👈 Yelek giyen       │
         │   "warning_count": 2,          👈 Yeleksiz          │
         │   "has_issue": true            👈 alarm logic       │
         │ }                                                    │
         └──────────────────────────────────────────────────────┘
```

---

## 3. FIRE AGENT ÇIKTI DETAYLI ÖRNEK

```
┌──────────────────────────────────────────────────────────┐
│ INPUT: warehouse_fire.jpg (1280×960)                     │
└────────────────┬──────────────────────────────────────────┘
                 │
                 ▼
     ┌───────────────────────────┐
     │ FireAgent.detect(img)     │
     │ model: fire_best.pt       │
     └───────────────┬───────────┘
                     │
         ┌───────────▼─────────────────────────────────────┐
         │ Preprocess:                                     │
         │ - Resize → 640×640                              │
         │ - HSV: Kırmızı/Turuncu mask (yangın renkleri)  │
         │ - Morfolojik işlemler                          │
         │ - Contrast artırma (alpha=1.3)                 │
         └───────────┬─────────────────────────────────────┘
                     │
         ┌───────────▼─────────────────────────────────────┐
         │ YOLO İnference                                  │
         │ conf_threshold: 0.5 (yüksek → false positive ↓)│
         └───────────┬─────────────────────────────────────┘
                     │
         ┌───────────▼────────────────────────────────────────────┐
         │ Postprocess:                                          │
         │ - SADECE class 0 (fire) kabul et                     │
         │ - Smoke (class 1) HARİÇ TUTULUYOR                    │
         │ - NMS (iou_threshold=0.3) → 1 detection MAX          │
         │ - Severity: bbox_area'ye göre (low/med/high)        │
         └───────────┬────────────────────────────────────────────┘
                     │
         ┌───────────▼──────────────────────────────────────────────┐
         │ ÇIKTI DICTIONARY:                                       │
         │                                                         │
         │ {                                                      │
         │   "agent": "FireAgent",                              │
         │   "detections": [                                    │
         │     {                                                │
         │       "bbox": [150, 120, 480, 400],                │
         │       "confidence": 0.78,                            │
         │       "class_id": 0,            👈 ONLY FIRE        │
         │       "label": "fire",          👈 ALARM TRIGGER    │
         │       "agent": "FireAgent",                         │
         │       "severity": "high"                            │
         │                                                    │
         │   }                            👈 SADECE 1 (NMS)  │
         │   ],                                               │
         │   "detection_count": 1,        👈 Yangın var       │
         │   "alert": true                👈 alarm logic      │
         │ }                                                   │
         └──────────────────────────────────────────────────────┘

SEVERITY HESAPLAMASI:
  area = (x2 - x1) * (y2 - y1)
  
  area > 100,000  → severity = "high"    (büyük yangın)
  50,000-100,000  → severity = "medium"  (orta yangın)
  area < 50,000   → severity = "low"     (küçük yangın/alev)
```

---

## 4. ALARM KONTROLÜ MANTIK AKIŞI

```
┌────────────────────────────────────────────────┐
│ helmet_result, vest_result, fire_result        │
└────────────────┬───────────────────────────────┘
                 │
        ┌────────▼────────────────────────────────────┐
        │ alarm = (                                   │
        │   helmet_result["warning_count"] > 0   OR   │ 👈 Baretsiz var mı?
        │   vest_result["warning_count"] > 0     OR   │ 👈 Yeleksiz var mı?
        │   fire_result["detection_count"] > 0       │ 👈 Yangın var mı?
        │ )                                           │
        └────────┬────────────────────────────────────┘
                 │
        ┌────────▼────────────────┐
        │ alarm == True?          │
        ├─────┬──────────────────┤
        │YES  │      NO          │
        │     │                  │
        ▼     ▼                  ▼
    ┌─────┐  ┌──────────────────┐
    │LLM  │  │ Format: "Güvenli │
    │Call │  │ sahne. LLM       │
    └─┬───┘  │ raporu           │
      │      │ uretilmedi."     │
      │      └──────────────────┘
      │
      ▼
    format_for_llm() → Yapılandırılmış Metin
      │
      ├─→ h_safe  = helmet_result["detection_count"]
      ├─→ h_viol  = helmet_result["warning_count"]
      ├─→ v_safe  = vest_result["detection_count"]
      ├─→ v_viol  = vest_result["warning_count"]
      ├─→ f_count = fire_result["detection_count"]
      │
      ▼
    İHLALLER LİSTESİ:
    ├─→ h_viol > 0  → "X kişi baretsiz"
    ├─→ v_viol > 0  → "Y kişi yeleksiz"
    └─→ f_count > 0 → "Yangın (güven: Z%)"
      │
      ▼
    PROMPT:
    "=== FABRIKA GÜVENLIK TESPIT RAPORU ===\n
     Zaman: ...\n
     [BARET DURUMU]\n
       Uyumlu: h_safe\n
       İHLAL: h_viol\n
     [YELEK DURUMU]\n
       Uyumlu: v_safe\n
       İHLAL: v_viol\n
     [YANGIN DURUMU]\n
       Yangın: f_count\n
     [İHLALLER]\n
       - ...\n"
      │
      ▼
    generate_response() → Ollama/Mistral
      │
      ▼
    LLM OUTPUT: "Fabrikanın A bölgesinde..."
      │
      ▼
    RETURN: {
      "alarm": true,
      "llm_called": true,
      "report": "LLM raporu...",
      "structured": "format_for_llm() output",
      "image": "fabrika_001.jpg",
      "timestamp": "2026-04-12T14:35:22"
    }
```

---

## 5. COMPLETE DATA FLOW ÖRNEK

```
SAHNE: 3 işçi var — 1'i baretsiz, 1'i yeleksiz

┌──────────────────────────────────────────────────────────┐
│ fabrika_001.jpg → CNN Processing                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ 🎫 HelmetAgent:    2 OK / 1 WARNING  (baretsiz 1)      │
│ 🎫 VestAgent:      2 OK / 1 WARNING  (yeleksiz 1)      │
│ 🎫 FireAgent:      0 detections                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
            ALARM CHECK: true (h_viol>0 || v_viol>0)
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ format_for_llm()                                         │
├──────────────────────────────────────────────────────────┤
│ =================================================== ╮   │
│ === FABRIKA GUVENLIK TESPIT RAPORU ===             │   │
│ Zaman        : 2026-04-12 14:35:22                 │   │
│ Goruntu      : fabrika_001.jpg                     │   │
│                                                    │ ╶→ Bu metin
│ [BARET DURUMU]                                     │   │ LLM'e
│   Uyumlu  : 2 kisi                                │   │ gönderiliyor
│   IHLAL   : 1 kisi <-- ALARM                     │   │
│                                                    │   │
│ [YELEK DURUMU]                                     │   │
│   Uyumlu  : 2 kisi                                │   │
│   IHLAL   : 1 kisi <-- ALARM                     │   │
│                                                    │   │
│ [YANGIN]                                           │   │
│   Tespit : HAYIR                                   │   │
│                                                    │   │
│ [ALARM]   : EVET                                   │   │
│ [IHLALLER]:                                        │   │
│   - 1 kisi BARETSIZ calistirildi                  │   │
│   - 1 kisi GUVENLIK YELEKSIZ calistirildi         │   │
│ =================================================== ╯   │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ LLM Prompt                                               │
├──────────────────────────────────────────────────────────┤
│ Sen bir endüstriyel iş sağlığı uzmanı yapay                │
│ zekasın. Aşağıda fabrika tespit raporu yer                │
│ almaktadır. Kısa (3-5 cümle, Türkçe) bir rapor           │
│ yaz. Sadece ihlallere odaklan. Somut eylem               │
│ önerisiyle bitir.                                        │
│                                                          │
│ === FABRIKA GUVENLIK TESPIT RAPORU ===                   │
│ [yukarıdaki structured output]                           │
│                                                          │
│ Güvenlik Raporu:                                         │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Ollama/Mistral Generate                                  │
├──────────────────────────────────────────────────────────┤
│ RESPONSE:                                                │
│                                                          │
│ "Fabrikanın üretim alanında bir işçi koruyucu başlık     │
│  takmadan ve diğeri güvenlik yeledni takmadan           │
│  çalışmaktadır. Bu durumda baş ve göğüs yaralanması    │
│  riski bulunmaktadır. Acil şekilde tüm işçilere         │
│  eksik KKD (Kişisel Koruyucu Donanım) tamamlanmalı     │
│  ve iş durmalıdır."                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│ Final Report Dictionary                                  │
├──────────────────────────────────────────────────────────┤
│ {                                                        │
│   "alarm": true,                                         │
│   "llm_called": true,                                    │
│   "report": "Fabrikanın üretim alanında...",            │
│   "structured": "[format_for_llm output]",              │
│   "image": "fabrika_001.jpg",                           │
│   "timestamp": "2026-04-12T14:35:22.123456"             │
│ }                                                        │
│                                                          │
│ ─→ results/alarm_report_fabrika_001_20260412_143522.txt │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 🔧 Değişiklik Yapacaksan

### SENARYOLAR:

#### 1️⃣ CNN ÇIKTI FORMATINI DEĞİŞTİRMEK
👉 `agents/specific_agents.py` → `postprocess()` metodları
   - Dictionary key isimleri
   - Field yapısı

#### 2️⃣ LLM'E GÖNDERILEN METNİ DEĞİŞTİRMEK
👉 `llm/llm_coordinator.py` → `format_for_llm()` metodu
   - Yapılandırılmış rapor formatı
   - Başlık/başlık türü

#### 3️⃣ PROMPT TALİMATINI DEĞİŞTİRMEK
👉 `llm/llm_coordinator.py` → `generate_alarm_report()` metodu içindeki `prompt` stringi
   - LLM'ye verilen sistem instruksiyonu
   - Output beklentileri

#### 4️⃣ ALARM TRİGGER KRİTERLERİNİ DEĞİŞTİRMEK
👉 `llm/llm_coordinator.py` → `generate_alarm_report()` metodu içindeki `alarm = ...` satırı
   ```python
   # Örnek: Sadece yangın alarm olsun
   alarm = fire_result.get("detection_count", 0) > 0
   
   # Örnek: Çok sıkı kriter
   alarm = (
       helmet_result.get("warning_count", 0) > 0
       and vest_result.get("warning_count", 0) > 0
   )
   ```

