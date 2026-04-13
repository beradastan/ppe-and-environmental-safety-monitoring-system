# -*- coding: utf-8 -*-
"""
Hard Hat Detection Agent
Roboflow Hard Hat Workers dataset ile egitilmis
"""
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics kutuephanesi kuruluyor...")
    import subprocess
    subprocess.check_call(["pip", "install", "ultralytics"])
    from ultralytics import YOLO
    YOLO_AVAILABLE = True


class HardHatDetectionAgent:
    def __init__(self, model_path=None):
        """
        Args:
            model_path: YOLOv8 model dosyasinin yolu
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "hardhat" / "best.pt"
        
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
        
        self.class_names = {
            0: 'hard_hat',
            1: 'person',
            2: 'safety_gear'
        }
    
    def load_model(self):
        """Model yukle"""
        if self.model_path.exists():
            print(f"LOADING Model yukleniyor: {self.model_path.name}")
            try:
                self.model = YOLO(str(self.model_path))
                print(f"OK - Model basarıyla yuklendi")
            except Exception as e:
                print(f"ERROR Model yukleme hatası: {e}")
        else:
            print(f"WARNING Model bulunamadi: {self.model_path}")
            print(f"    Lutfen modeli kopyala veya egit")
    
    def detect(self, image_source, conf=0.5):
        """
        Deteksiyon yap
        
        Args:
            image_source: Gorsel dosya yolu veya numpy array
            conf: Confidence threshold
        
        Returns:
            dict: Deteksiyon sonuclari
        """
        if self.model is None:
            return {"error": "Model yuklenmedi", "detections": [], "count": 0}
        
        try:
            results = self.model(image_source, conf=conf, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    detection = {
                        'class': self.class_names.get(cls_id, 'unknown'),
                        'class_id': cls_id,
                        'confidence': confidence,
                        'bbox': bbox,
                        'x1': int(bbox[0]),
                        'y1': int(bbox[1]),
                        'x2': int(bbox[2]),
                        'y2': int(bbox[3])
                    }
                    detections.append(detection)
            
            return {
                'detections': detections,
                'count': len(detections),
                'has_hardhat': any(d['class'] == 'hard_hat' for d in detections),
                'has_person': any(d['class'] == 'person' for d in detections),
                'has_safety_gear': any(d['class'] == 'safety_gear' for d in detections)
            }
        except Exception as e:
            return {"error": str(e), "detections": [], "count": 0}
    
    def detect_and_visualize(self, image_path, output_path=None):
        """Deteksiyon yap ve gorsellesdir"""
        if self.model is None:
            return None
        
        try:
            results = self.model(str(image_path), conf=0.5, verbose=False)
            annotated = results[0].plot()
            
            if output_path:
                cv2.imwrite(str(output_path), annotated)
            
            return annotated
        except Exception as e:
            print(f"ERROR Gorsellestime hatası: {e}")
            return None
    
    def batch_detect(self, image_dir, conf=0.5):
        """Klasordeki tum goerselleri detekt et"""
        results = []
        
        image_dir = Path(image_dir)
        for image_path in list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")):
            detection = self.detect(str(image_path), conf=conf)
            results.append({
                'image': image_path.name,
                'detection': detection
            })
        
        return results
