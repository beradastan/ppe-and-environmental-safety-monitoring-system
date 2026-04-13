# 🎯 HIZLI REFERANS KARTI

## 📌 ÖNEMLİ KOMUTLAR

```bash
# Test et
python test_models_simple.py

# Demo gör
python demo_agents.py

# Web kamerası
python video_processor.py

# System kontrol
python system_check.py

# LLM analizi
python llm_integration_example.py --image test.jpg
```

---

## 💻 3 SATIRLİ KODLAR

### Baret Tespiti
```python
from agents.specific_agents import HelmetAgent
import cv2
HelmetAgent().visualize_results(cv2.imread("test.jpg"), 
                                 HelmetAgent().detect(cv2.imread("test.jpg")))
```

### Yangın Tespiti
```python
from agents.specific_agents import FireAgent
import cv2
print(FireAgent().detect(cv2.imread("test.jpg")))
```

### Tüm Ajanlar
```python
from pipeline.detection_pipeline import DetectionPipeline
import cv2
print(DetectionPipeline().process_image(cv2.imread("test.jpg")))
```

---

## 📊 HIZLI BILGI

| Ajan | Model | Sınıflar | Hız |
|------|-------|----------|-----|
| Helmet | ppe_best.pt | 3 | 0.5s |
| Vest | ppe_best.pt | 3 | 0.5s |
| Fire | fire_best.pt | 2 | 0.4s |

---

## 🔧 AYARLAMALAR

```python
# Confidence
HelmetAgent(confidence_threshold=0.7)

# Device
HelmetAgent(device="cuda")

# Model
HelmetAgent(model_name="models/ppe_best.pt")
```

---

## 🎓 DOCLAR

- **Başlangıç:** QUICKSTART.md
- **Setup:** SETUP_GUIDE.md
- **API:** MODELS_USAGE_GUIDE.md
- **Durum:** PROJECT_STATUS.md
- **Index:** INDEX.md

---

## 🐛 SORUN GİDERME

| Sorun | Çözüm |
|-------|-------|
| Model not found | `ls models/` |
| Import error | `pip install -r requirements.txt` |
| CUDA out of memory | `device="cpu"` |
| Web kamera? | Device numarasını değiştir |

---

## 🟢 DURUM: HAZIR!

Başlangıç: `python test_models_simple.py`

---

**2026-03-26 | Hazır & Test Edildi**

