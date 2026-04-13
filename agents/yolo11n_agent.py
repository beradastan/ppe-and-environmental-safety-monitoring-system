"""
YOLO11n Agent - Hugging Face Construction Hazard Detection Model
Vest ve Hardhat Tespit için optimize edilmiş agent
"""

import numpy as np
import cv2
from typing import Dict, Any
from agents.base_agent import BaseAgent


class YOLO11nAgent(BaseAgent):
    """
    YOLO11n Construction Hazard Detection Agent
    Model: yihong.pt (Hugging Face - yihong1120/Construction-Hazard-Detection)
    
    Sınıflar:
    - Class 0: Hardhat (Baret giymiş) ✓
    - Class 2: NO-Hardhat (Baretsiz) ⚠️
    - Class 4: NO-Safety Vest (Yeleksiz) ⚠️
    - Class 5: Person (Kişi) - arka plan
    - Class 7: Safety Vest (Yelek giymiş) ✓
    """
    
    def __init__(
        self,
        model_name: str = "models/yihong.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu"
    ):
        super().__init__(
            agent_name="YOLO11nAgent",
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Sınıf tanımlamaları
        self.class_names = {
            0: "Hardhat",           # Baret giymiş (pozitif)
            1: "Mask",              # Maske (dikkate almıyoruz)
            2: "NO-Hardhat",        # Baretsiz (uyarı)
            3: "NO-Mask",           # Maskesiz (dikkate almıyoruz)
            4: "NO-Safety Vest",    # Yeleksiz (uyarı)
            5: "Person",            # Kişi (arka plan)
            6: "Safety Cone",       # Güvenlik konisi (dikkate almıyoruz)
            7: "Safety Vest",       # Yelek giymiş (pozitif)
            8: "machinery",         # Makine (dikkate almıyoruz)
            9: "vehicle"            # Araç (dikkate almıyoruz)
        }
        
        # Hardhat ve Vest tespit için target sınıflar
        self.hardhat_target = [0]           # Hardhat giymiş
        self.hardhat_warning = [2]          # NO-Hardhat (uyarı)
        self.vest_target = [7]              # Safety Vest giymiş
        self.vest_warning = [4]             # NO-Safety Vest (uyarı)
        
        # Tüm hedef sınıflar (diğerleri göz ardı edilecek)
        self.target_classes = [0, 2, 4, 7]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        YOLO11n için ön işleme
        - Görüntü boyutu standardizasyonu
        - CLAHE contrast enhancement
        """
        # Standart boyuta resize (YOLO11n 640x640 kullanıyor)
        processed = cv2.resize(image, (640, 640))
        
        # Kontrast iyileştirme (CLAHE)
        if len(processed.shape) == 3:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def postprocess(self, results: Any) -> Dict:
        """
        YOLO11n sonuçlarını işle
        Hardhat ve Safety Vest tespitleri ayrı kategorilere ayır
        """
        hardhat_detections = []
        hardhat_warnings = []
        vest_detections = []
        vest_warnings = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Sadece hedef sınıfları işle
                    if class_id not in self.target_classes:
                        continue
                    
                    # Hardhat tespitleri
                    if class_id in self.hardhat_target:
                        hardhat_detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names.get(class_id, "unknown"),
                            "agent": self.agent_name,
                            "status": "safe",
                            "type": "hardhat_positive"
                        })
                    
                    # Hardhat uyarıları (baretsiz)
                    elif class_id in self.hardhat_warning:
                        hardhat_warnings.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names.get(class_id, "unknown"),
                            "agent": self.agent_name,
                            "status": "warning",
                            "type": "hardhat_warning"
                        })
                    
                    # Safety Vest tespitleri
                    elif class_id in self.vest_target:
                        vest_detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names.get(class_id, "unknown"),
                            "agent": self.agent_name,
                            "status": "safe",
                            "type": "vest_positive"
                        })
                    
                    # Safety Vest uyarıları (yeleksiz)
                    elif class_id in self.vest_warning:
                        vest_warnings.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names.get(class_id, "unknown"),
                            "agent": self.agent_name,
                            "status": "warning",
                            "type": "vest_warning"
                        })
        
        # Confidence'a göre sırala (en yüksek ilk)
        hardhat_detections = sorted(hardhat_detections, key=lambda x: x['confidence'], reverse=True)
        hardhat_warnings = sorted(hardhat_warnings, key=lambda x: x['confidence'], reverse=True)
        vest_detections = sorted(vest_detections, key=lambda x: x['confidence'], reverse=True)
        vest_warnings = sorted(vest_warnings, key=lambda x: x['confidence'], reverse=True)
        
        return {
            "agent": self.agent_name,
            "hardhat_detections": hardhat_detections,
            "hardhat_warnings": hardhat_warnings,
            "vest_detections": vest_detections,
            "vest_warnings": vest_warnings,
            "detections": hardhat_detections + vest_detections,  # Compatible with formatter
            "warnings": hardhat_warnings + vest_warnings,
            "detection_count": len(hardhat_detections) + len(vest_detections),
            "warning_count": len(hardhat_warnings) + len(vest_warnings),
            "has_issue": len(hardhat_warnings) + len(vest_warnings) > 0
        }
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Görüntüde tespit yap
        """
        # Ön işleme
        processed_image = self.preprocess(image)
        
        # Model inferansı
        results = self.model(processed_image, conf=self.confidence_threshold, verbose=False)
        
        # Son işleme
        result = self.postprocess(results)
        
        return result

