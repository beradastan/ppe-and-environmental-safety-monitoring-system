"""
Eğitim Utilities - Kendi veri setiyle eğitim için hazırlık
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import yaml
import logging


class DatasetPreparation:
    """Kendi veri seti oluşturma ve hazırlama"""
    
    def __init__(self, dataset_dir: str = "./dataset"):
        """
        Args:
            dataset_dir: Veri seti klasörü
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("DatasetPreparation")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - DatasetPrep - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def create_yolo_structure(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        YOLO v8 için veri seti yapısı oluştur
        
        Yapı:
            dataset/
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── labels/
                ├── train/
                ├── val/
                └── test/
        
        Args:
            train_ratio: Eğitim seti oranı
            val_ratio: Validasyon seti oranı
        """
        try:
            # Klasör yapısı
            for split in ["train", "val", "test"]:
                (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            
            self.logger.info("✓ YOLO yapısı oluşturuldu")
            self._print_structure()
            
        except Exception as e:
            self.logger.error(f"Yapı oluşturma hatası: {str(e)}")
    
    def _print_structure(self):
        """Veri seti yapısını göster"""
        structure = """
Dataset Yapısı:
dataset/
├── images/
│   ├── train/        <- Eğitim görüntüleri
│   ├── val/          <- Validasyon görüntüleri
│   └── test/         <- Test görüntüleri
├── labels/
│   ├── train/        <- Eğitim annotations (YOLO format)
│   ├── val/          <- Validasyon annotations
│   └── test/         <- Test annotations
└── data.yaml         <- Dataset konfigürasyonu
        """
        self.logger.info(structure)
    
    def create_data_yaml(
        self,
        class_names: List[str],
        output_path: str = None
    ) -> str:
        """
        YOLO data.yaml dosyası oluştur
        
        Args:
            class_names: Sınıf isimleri
            output_path: Çıktı yolu
            
        Returns:
            Dosya yolu
        """
        try:
            output_path = output_path or str(self.dataset_dir / "data.yaml")
            
            data_yaml = {
                "path": str(self.dataset_dir.absolute()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(class_names),
                "names": class_names
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            
            self.logger.info(f"✓ data.yaml oluşturuldu: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"YAML oluşturma hatası: {str(e)}")
            return ""
    
    def coco_to_yolo(
        self,
        image_path: str,
        bbox: Tuple[float, float, float, float],
        class_id: int,
        image_width: int,
        image_height: int
    ) -> Tuple[float, float, float, float]:
        """
        COCO format bbox'ı YOLO format'a çevir
        
        COCO format: [x_min, y_min, width, height]
        YOLO format: [x_center, y_center, width_norm, height_norm] (0-1 aralığı)
        
        Args:
            image_path: Görüntü yolu
            bbox: COCO format bbox
            class_id: Sınıf ID'si
            image_width: Görüntü genişliği
            image_height: Görüntü yüksekliği
            
        Returns:
            YOLO format bbox
        """
        x_min, y_min, width, height = bbox
        
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        w_norm = width / image_width
        h_norm = height / image_height
        
        return (x_center, y_center, w_norm, h_norm)
    
    def create_annotation_file(
        self,
        label_path: str,
        detections: List[Dict]
    ) -> None:
        """
        YOLO format annotation dosyası oluştur
        
        Format:
            <class_id> <x_center> <y_center> <width> <height>
            ... (her satırda bir deteksiyon)
        
        Args:
            label_path: Çıktı label dosyası
            detections: Detection listesi
        """
        try:
            with open(label_path, 'w') as f:
                for det in detections:
                    class_id = det["class_id"]
                    x_center, y_center, w_norm, h_norm = det["bbox_yolo"]
                    f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")
            
        except Exception as e:
            self.logger.error(f"Annotation yazma hatası: {str(e)}")
    
    def split_dataset(
        self,
        images_source_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict:
        """
        Görüntüleri train/val/test'e böl
        
        Args:
            images_source_dir: Kaynak görüntüler klasörü
            train_ratio: Eğitim seti oranı
            val_ratio: Validasyon seti oranı
            
        Returns:
            Split istatistikleri
        """
        try:
            source_path = Path(images_source_dir)
            image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
            
            self.logger.info(f"Toplam görüntü: {len(image_files)}")
            
            # Shuffle
            np.random.shuffle(image_files)
            
            # Split points
            train_split = int(len(image_files) * train_ratio)
            val_split = int(len(image_files) * (train_ratio + val_ratio))
            
            train_files = image_files[:train_split]
            val_files = image_files[train_split:val_split]
            test_files = image_files[val_split:]
            
            # Dosyaları kopyala
            splits = {
                "train": (train_files, "train"),
                "val": (val_files, "val"),
                "test": (test_files, "test")
            }
            
            for split_name, (files, split_dir) in splits.items():
                for file in files:
                    dest = self.dataset_dir / "images" / split_dir / file.name
                    shutil.copy2(str(file), str(dest))
                
                self.logger.info(f"✓ {split_name}: {len(files)} görüntü")
            
            return {
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files)
            }
            
        except Exception as e:
            self.logger.error(f"Dataset split hatası: {str(e)}")
            return {}


class DataAugmentation:
    """Veri artırma (Data Augmentation)"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataAugmentation")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - Augment - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def random_rotation(self, image: np.ndarray, angle_range: Tuple[int, int] = (-15, 15)) -> np.ndarray:
        """Rastgele rotasyon"""
        angle = np.random.randint(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def random_flip(self, image: np.ndarray) -> np.ndarray:
        """Rastgele flip"""
        if np.random.random() > 0.5:
            return cv2.flip(image, 1)  # Yatay flip
        return image
    
    def random_brightness(self, image: np.ndarray, brightness_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """Rastgele parlaklık değişikliği"""
        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        return cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    def random_contrast(self, image: np.ndarray, contrast_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """Rastgele kontrast değişikliği"""
        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.convertScaleAbs(l, alpha=contrast_factor, beta=0)
        return cv2.merge([l, a, b])
    
    def augment_image(self, image: np.ndarray, num_augmentations: int = 3) -> List[np.ndarray]:
        """
        Bir görüntüyü çoklu şekilde artır
        
        Args:
            image: Orijinal görüntü
            num_augmentations: Kaç tane augmented görüntü oluştur
            
        Returns:
            Augmented görüntüler
        """
        augmented = [image]  # Orijinal ekle
        
        for _ in range(num_augmentations):
            aug = image.copy()
            aug = self.random_flip(aug)
            aug = self.random_rotation(aug)
            aug = self.random_brightness(aug)
            aug = self.random_contrast(aug)
            augmented.append(aug)
        
        return augmented


class TrainingGuide:
    """YOLO v8 eğitim rehberi"""
    
    @staticmethod
    def print_training_guide():
        """Eğitim kılavuzunu yazdır"""
        guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     YOLO v8 EĞITIM KAULAVU                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ADIM: DATASET HAZIRLIĞI
   ✓ Görüntüleri toplayın ve organize edin
   ✓ YOLO format annotation'ları oluşturun
   ✓ Dataset'i train/val/test'e bölün

2. ADIM: VERİ ARTIRMA (OPTİYONEL)
   from utils.training_utils import DataAugmentation
   aug = DataAugmentation()
   augmented = aug.augment_image(image, num_augmentations=5)

3. ADIM: EĞITIM YAPISI OLUŞTURMA
   from utils.training_utils import DatasetPreparation
   prep = DatasetPreparation()
   prep.create_yolo_structure()
   prep.create_data_yaml(["helmet", "no-helmet", "vest", "no-vest", "fire", "smoke"])

4. ADIM: EĞITIMI BAŞLATMA
   from ultralytics import YOLO
   
   # Önceden eğitilmiş model yükle
   model = YOLO("yolov8m.pt")
   
   # Eğitim başlat
   results = model.train(
       data="dataset/data.yaml",      # Dataset config
       epochs=100,                     # Epoch sayısı
       imgsz=640,                      # Görüntü boyutu
       batch=16,                       # Batch size
       device=0,                       # GPU ID (0) veya CPU ('cpu')
       patience=20,                    # Early stopping patience
       save=True,                      # Sonuçları kaydet
       project="runs/detect",          # Proje klasörü
       name="helmet_detector"          # Run adı
   )

5. ADIM: EĞITILMIŞ MODELI KULLANMA
   model = YOLO("runs/detect/helmet_detector/weights/best.pt")
   results = model(image)

6. ADIM: MODEL DEĞERLENDİRME
   metrics = model.val()
   print(f"mAP50: {metrics.box.map50}")
   print(f"mAP50-95: {metrics.box.map}")

⚠️ ÖNEMLİ İPUÇLARI:

• Batch Size:
  - GPU (NVIDIA A100): 128-256
  - GPU (RTX 3090): 64-128
  - GPU (RTX 2080): 32-64
  - CPU: 8-16

• Epoch Sayısı: Genellikle 50-200, dataset boyutuna bağlı

• Learning Rate: Default 0.01, küçük dataset için 0.001

• İmgsz: 640 (standart), 1280 (daha yüksek doğruluk)

• Augmentation: Ayar yapması gerekiyorsa config'i değiştir

• GPU Memory: CUDA out of memory hatası alırsanız batch size'ı azalt

║
║  KOMUT ÖRNEKLERI:
║  
║  # Komutsatırından eğitim
║  yolo detect train data=dataset/data.yaml epochs=100 imgsz=640
║  
║  # Python'dan eğitim
║  from ultralytics import YOLO
║  model = YOLO("yolov8m.pt")
║  model.train(data="dataset/data.yaml", epochs=100)
║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(guide)

