# Multi-CNN + Open-Source LLM Detection Sistemi

**Bitirme Projesi - Safety Detection with AI**

Endüstriyel ortamlarda güvenlik önlemleri (baret, yelek, yangın) tespiti yapan gelişmiş bir yapay zeka sistemi.

## 🎯 Proje Özellikleri

- **3 Ayrı CNN Ajanı**: YOLO v8 kullanarak
  - 🎩 **HelmetAgent**: Baret/Kask tespit
  - 👷 **VestAgent**: İş elbisesi/Yelek tespit  
  - 🔥 **FireAgent**: Yangın/Alevler tespit

- **LLM Entegrasyonu**: Ollama / Local LLM
  - Prompt-based karar verme
  - Deteksiyon sonuçlarını analiz
  - Akıllı yönlendirme

- **Ana Detection Pipeline**:
  - Görüntü işleme
  - Video işleme
  - Web kamerası canlı detection

- **Eğitim Desteği**: Kendi veri seti ile eğitim imkanı

## 📋 Sistem Gereksinimleri

- **Python**: 3.10 veya üzeri
- **GPU** (Opsiyonel): CUDA 11.8+ (NVIDIA)
- **Ollama**: Local LLM için (opsiyonel)

## 📊 TEST SONUÇLARI

**Test Tarihi:** 2026-03-28  
**Başarı Oranı:** 83.3% (15/18)

### Başarılı Testler
✅ images.jpg - Boş tespit (0/0/0)  
✅ man-working - 1 Helmet doğru  
✅ Screenshot_2 - Boş tespit (0/0/0)  
✅ ... ve 11 test daha

### Agent Thresholds
```
HelmetAgent:  0.30 (Model: Hexmon YOLO)
VestAgent:    0.25 (Model: Hexmon YOLO, Multi-scale)
FireAgent:    0.5  (Model: fire_best.pt)
```

**Detaylı rapor:** [FINAL_REPORT.md](FINAL_REPORT.md)

---

## 🚀 Kurulum

### 1. Virtual Environment Oluştur

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 3. Ollama Kurulumu (Opsiyonel ama Önerilir)

LLM özelliklerini kullanmak için:

```bash
# Windows
# https://ollama.ai adresinden indir ve kur

# Linux
curl https://ollama.ai/install.sh | sh

# Model indir ve çalıştır
ollama pull mistral
ollama serve
```

## 📁 Proje Yapısı

```
factory/
├── main.py                    # Ana uygulama
├── requirements.txt           # Bağımlılıklar
├── README.md                  # Bu dosya
│
├── agents/                    # CNN Ajanları
│   ├── __init__.py
│   ├── base_agent.py         # Temel ajan sınıfı
│   └── specific_agents.py    # HelmetAgent, VestAgent, FireAgent
│
├── llm/                       # LLM Entegrasyonu
│   ├── __init__.py
│   └── llm_coordinator.py    # Ollama bağlantısı
│
├── pipeline/                  # Ana Pipeline
│   ├── __init__.py
│   └── detection_pipeline.py # DetectionPipeline sınıfı
│
├── utils/                     # Utility Fonksiyonları
│   ├── __init__.py
│   └── training_utils.py     # Eğitim için hazırlık
│
├── dataset/                   # Veri Seti (ileride)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
└── results/                   # Sonuçlar
    └── *.jpg                  # Anotasyonlu görüntüler
```

## 🎓 Kullanım Örnekleri

### Sistem Test Et

```bash
python main.py --test
```

Çıktı:
```
SİSTEM TEST BAŞLAMAKTA...
1. Pipeline oluşturuluyor...
✓ Pipeline başarıyla oluşturuldu
2. Test görüntüsü oluşturuluyor...
✓ Test görüntüsü oluşturuldu
3. Ajanlar test ediliyor...
✓ Tüm ajanlar çalıştırıldı
...
```

### Görüntü İşle

```bash
# LLM analizi ile
python main.py --mode image --image_path ./test.jpg

# LLM olmadan (hızlı)
python main.py --mode image --image_path ./test.jpg --no-llm

# GPU kullan
python main.py --mode image --image_path ./test.jpg --device cuda
```

### Video İşle

```bash
# Video işle ve kaydet
python main.py --mode video --video_path ./test.mp4 --output ./output.mp4

# Her 5. frame'i işle (hız için)
python main.py --mode video --video_path ./test.mp4 --no-llm
```

### Web Kamerası Canlı Detection

```bash
# 30 saniye çalıştır
python main.py --mode webcam --duration 30

# 60 saniye
python main.py --mode webcam --duration 60
```

## 📊 Python API Kullanımı

### Basit Kullanım

```python
from pipeline.detection_pipeline import DetectionPipeline
import cv2

# Pipeline oluştur
pipeline = DetectionPipeline(use_llm=True, device="cpu")

# Görüntü yükle
image = cv2.imread("test.jpg")

# Detection yap
result = pipeline.process_image(image, use_llm_analysis=True)

# Sonuçları göster
for agent_name, detections in result["detections"].items():
    print(f"{agent_name}: {detections['detection_count']} tespit")
    for det in detections["detections"]:
        print(f"  - {det['label']}: {det['confidence']:.2%}")
```

### Spesifik Agent Kullanımı

```python
from agents.specific_agents import HelmetAgent, VestAgent, FireAgent
import cv2
import numpy as np

# Ajan oluştur
helmet_agent = HelmetAgent(confidence_threshold=0.5, device="cpu")

# Görüntü yükle
image = cv2.imread("test.jpg")

# Detection yap
results = helmet_agent.detect(image)

# Sonuçları göster
print(f"Tespit: {results['detection_count']}")
for det in results['detections']:
    bbox = det['bbox']
    confidence = det['confidence']
    print(f"  Baret bulundu: {confidence:.2%} güvenle konum: {bbox}")

# Anotasyonlu görüntü
annotated = helmet_agent.visualize_results(image, results, output_path="annotated.jpg")
```

### LLM ile Koordinasyon

```python
from llm.llm_coordinator import LLMCoordinator
from pipeline.detection_pipeline import DetectionPipeline
import cv2

# Pipeline ve LLM
pipeline = DetectionPipeline(use_llm=True)
llm = pipeline.llm_coordinator

# Deteksiyon yap
image = cv2.imread("test.jpg")
result = pipeline.process_image(image)

# LLM analizi
analysis = llm.coordinate_detections(result["detections"])
print("LLM Analizi:")
print(analysis["llm_analysis"])
```

## 🎓 Eğitim

### Veri Seti Hazırlama

```python
from utils.training_utils import DatasetPreparation, DataAugmentation

# Dataset hazırla
prep = DatasetPreparation(dataset_dir="./dataset")
prep.create_yolo_structure()

# data.yaml oluştur
prep.create_data_yaml(
    class_names=["helmet", "no-helmet", "vest", "no-vest", "fire", "smoke"]
)

# Görüntüleri böl
splits = prep.split_dataset(
    images_source_dir="./raw_images",
    train_ratio=0.7,
    val_ratio=0.15
)
```

### Veri Artırma

```python
from utils.training_utils import DataAugmentation
import cv2

aug = DataAugmentation()
image = cv2.imread("image.jpg")

# Çoklu augmentation
augmented_images = aug.augment_image(image, num_augmentations=5)

for i, aug_img in enumerate(augmented_images):
    cv2.imwrite(f"augmented_{i}.jpg", aug_img)
```

### YOLO Eğitimi

```python
from ultralytics import YOLO

# Model yükle
model = YOLO("yolov8m.pt")

# Eğitim başlat
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU 0
    patience=20
)

# Eğitilmiş modeli kullan
model = YOLO("runs/detect/train/weights/best.pt")
results = model("test.jpg")
```

## 📈 Sonuçlar ve İstatistikler

```python
# Pipeline istatistikleri
stats = pipeline.get_pipeline_statistics()

print(f"Toplam Frame: {stats['pipeline_stats']['total_frames']}")
print(f"Toplam Deteksiyon: {stats['pipeline_stats']['total_detections']}")
print(f"Ortalama Processing Time: {stats['avg_processing_time']:.3f}s")

# Ajan istatistikleri
helmet_stats = stats['helmet_agent']
print(f"HelmetAgent: {helmet_stats['statistics']['detections_made']} tespit")
```

## ⚙️ Ayarlar

### Güven Eşiği (Confidence Threshold)

```bash
# Daha yüksek eşik = daha az false positive
python main.py --mode image --image_path test.jpg --confidence 0.7

# Daha düşük eşik = daha fazla deteksiyon
python main.py --mode image --image_path test.jpg --confidence 0.3
```

### İşleme Frekansı (Video)

```bash
# Her 5. frame'i işle (daha hızlı)
python main.py --mode video --video_path test.mp4 --no-llm

# Kodda:
pipeline.process_video(
    "test.mp4",
    process_every_n_frames=10  # Her 10. frame
)
```

## 🔧 Sorun Giderme

### CUDA Hatası

```
RuntimeError: CUDA out of memory
```

**Çözüm**: Batch size'ı azalt veya CPU kullan

```bash
python main.py --mode image --image_path test.jpg --device cpu
```

### Ollama Bağlantısı Hatası

```
⚠ Ollama sunucusuna bağlanamadı
```

**Çözüm**: Ollama'yı başlat

```bash
# Yeni terminalden
ollama serve

# Modelini başlat
ollama pull mistral
```

### Model İndir Hatası

İlk çalıştırmada YOLO modelleri otomatik indirilir. İnternet bağlantısı gerekli.

## 📝 Logging

Log dosyası otomatik oluşturulur: `detection_pipeline.log`

## 🎯 Gelecek Geliştirmeler

- [ ] Custom YOLO modelleri eğitimi
- [ ] Gerçek zamanlı stats dashboard
- [ ] Web arayüzü
- [ ] Multi-GPU desteği
- [ ] ONNX model export
- [ ] TensorRT optimization

## 📚 Kaynaklar

- [YOLO v8 Dokümantasyon](https://docs.ultralytics.com/)
- [Ollama](https://ollama.ai/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

## 📞 İletişim ve Destek

Sorularınız ve önerileriniz için lütfen iletişim kurun.

## 📄 Lisans

Bu proje akademik amaçlar için hazırlanmıştır.

---

**Yapı Tarihi**: 2026-03-23  
**Sürüm**: 1.0.0

