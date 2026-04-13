"""
Spesifik Detection Ajanları
- HelmetAgent: Baret/Kask tespit
- VestAgent: Yelek tespit
- FireAgent: Yangın/Alevler tespit
"""

import numpy as np
import cv2
from typing import Dict, Any
from agents.base_agent import BaseAgent


class HelmetAgent(BaseAgent):
    """
    Baret/Kask Tespit Ajanı
    Model: yihong.pt (YOLO11n — en iyi helmet/no-helmet performansı)
      class 0 → Hardhat      (baret GİYMİŞ  — pozitif)
      class 2 → NO-Hardhat   (baretsiz      — uyarı)
    """

    def __init__(
        self,
        model_name: str = "models/yihong.pt",
        confidence_threshold: float = 0.20,  # 0.25 → 0.20: kaçan hardhatleri yakalamak için
        device: str = "cpu"
    ):
        super().__init__(
            agent_name="HelmetAgent",
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.class_names = {
            0: "Hardhat",     # Baret giymiş (pozitif)
            2: "NO-Hardhat",  # Baretsiz     (uyarı)
        }
        self.target_classes  = [0]  # Baret giymiş
        self.warning_classes = [2]  # Baretsiz — alarm tetikler

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Baret tespiti için ön işleme: resize + CLAHE kontrast iyileştirme"""
        processed = cv2.resize(image, (640, 640))
        if len(processed.shape) == 3:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        return processed

    def postprocess(self, results: Any) -> Dict:
        """yihong.pt baret sonuçlarını işle"""
        detections = []
        warnings   = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id   = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    if class_id in self.target_classes:
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names[class_id],
                            "agent": self.agent_name,
                            "status": "safe",
                        })
                    elif class_id in self.warning_classes:
                        warnings.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names[class_id],
                            "agent": self.agent_name,
                            "status": "warning",
                        })

        return {
            "agent": self.agent_name,
            "detections": detections,
            "warnings":   warnings,
            "detection_count": len(detections),
            "warning_count":   len(warnings),
            "has_issue": len(warnings) > 0,
        }

    def detect(self, image: np.ndarray) -> Dict:
        """Baret tespiti — en güvenilir 5 tespiti döndür"""
        result = super().detect(image)
        result["detections"] = sorted(
            result["detections"], key=lambda x: x["confidence"], reverse=True
        )[:5]
        result["warnings"] = sorted(
            result["warnings"], key=lambda x: x["confidence"], reverse=True
        )[:5]
        result["detection_count"] = len(result["detections"])
        result["warning_count"]   = len(result["warnings"])
        return result


class VestAgent(BaseAgent):
    """
    Yelek Tespit Ajanı
    Model: yihong.pt (YOLO11n — hexmon'a göre vest tespitinde çok daha iyi)
      class 7 → Safety Vest    (yelek GİYMİŞ  — pozitif)
      class 4 → NO-Safety Vest (yeleksiz       — uyarı)
    """
    # NOT: hexmon_best.pt vest tespitinde yetersiz bulundu
    # (diagnose_results.txt — group-workers.webp: hexmon 0 vest, yihong 5 NO-vest tespit)
    
    
    def __init__(
        self,
        model_name: str = "models/yihong.pt",
        confidence_threshold: float = 0.25,  # adaptive imgsz ile büyük görsellerde no-vest yakalanıyor
        device: str = "cpu"
    ):
        super().__init__(
            agent_name="VestAgent",
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.class_names = {
            7: "Safety Vest",    # Yelek giymiş (pozitif)
            4: "NO-Safety Vest", # Yeleksiz     (uyarı)
        }
        self.target_classes  = [7]  # Yelek giymiş
        self.warning_classes = [4]  # Yeleksiz — alarm tetikler

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Yelek tespiti için ön işleme: sadece resize (yihong kendi normalize ediyor)"""
        return cv2.resize(image, (640, 640))

    def postprocess(self, results: Any) -> Dict:
        """yihong.pt yelek sonuçlarını işle"""
        detections = []
        warnings   = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id   = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    if class_id in self.target_classes:
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names[class_id],
                            "agent": self.agent_name,
                            "status": "safe",
                        })
                    elif class_id in self.warning_classes:
                        warnings.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": self.class_names[class_id],
                            "agent": self.agent_name,
                            "status": "warning",
                        })

        return {
            "agent": self.agent_name,
            "detections": detections,
            "warnings":   warnings,
            "detection_count": len(detections),
            "warning_count":   len(warnings),
            "has_issue": len(warnings) > 0,
        }

    def detect(self, image: np.ndarray) -> Dict:
        """
        Yelek tespiti — adaptive imgsz.
        Büyük görüntülerde (>1000px) imgsz=1280 kullan: küçük kişiler sıkıştırmayla kaçmasın.
        En güvenilir 5 tespiti döndür.
        """
        import time
        start_time = time.time()

        # Büyük görüntüler için daha yüksek çözünürlük — stock-photo gibi 1500×1100 görsellerde
        # 640×640'a sıkıştırma küçük kişileri kaçırıyor; 1280 bunu önler
        h, w = image.shape[:2]
        imgsz = 1280 if max(h, w) > 1000 else 640

        processed = self.preprocess(image)
        results   = self.model(processed, conf=self.confidence_threshold, imgsz=imgsz, verbose=False)
        result    = self.postprocess(results)

        # Confidence'a göre sırala, en fazla 5 tut
        result["detections"] = sorted(
            result["detections"], key=lambda x: x["confidence"], reverse=True
        )[:5]
        result["warnings"] = sorted(
            result["warnings"], key=lambda x: x["confidence"], reverse=True
        )[:5]
        result["detection_count"] = len(result["detections"])
        result["warning_count"]   = len(result["warnings"])

        self.detection_stats["total_frames"]   += 1
        self.detection_stats["detections_made"] += len(result["detections"])
        self.detection_stats["processing_time"]  = time.time() - start_time

        return result


class FireAgent(BaseAgent):
    """Yangın Tespit Ajanı - SADECE FIRE (class 0)"""
    
    def __init__(
        self,
        model_name: str = "models/fire_best.pt",
        confidence_threshold: float = 0.5,  # Daha yüksek threshold
        device: str = "cpu"
    ):
        super().__init__(
            agent_name="FireAgent",
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device
        )
        # SADECE fire (class 0) - smoke'u saymıyoruz
        self.class_names = {
            0: "fire"
        }
        self.target_classes = [0]  # SADECE 0
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Yangın tespiti için ön işleme
        - Kırmızı/turuncu renkler vurgulama
        - Kontrast artırma
        """
        # Standart boyuta resize
        processed = cv2.resize(image, (640, 640))
        
        # HSV renk uzayında kırmızı/turuncu renkler vurgulama
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        # Kırmızı ve turuncu aralığı (yangın renkleri)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([25, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Kontrast artırma
        processed = cv2.convertScaleAbs(processed, alpha=1.3, beta=20)
        
        return processed
    
    def postprocess(self, results: Any) -> Dict:
        """Yangın tespiti sonuçlarını işle - SADECE FIRE (class 0) ve NMS optimizasyonu"""
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            # Tüm fire detection'ları topla
            fire_detections_raw = []
            for box in boxes:
                class_id = int(box.cls[0])
                
                # SADECE fire (class 0)
                if class_id not in self.target_classes:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                
                fire_detections_raw.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": self.class_names.get(class_id, "unknown"),
                    "agent": self.agent_name,
                    "severity": self._calculate_severity(x1, y1, x2, y2)
                })
            
            # NMS (Non-Maximum Suppression) - çakışan detection'ları birleştir
            detections = self._apply_nms(fire_detections_raw, iou_threshold=0.3)
        
        return {
            "agent": self.agent_name,
            "detections": detections,
            "detection_count": len(detections),
            "alert": len(detections) > 0
        }
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """
        Non-Maximum Suppression - çakışan detection'ları birleştir
        Yangın tespiti: En yüksek confidence detection'ı tut
        """
        if not detections:
            return []
        
        # Confidence'a göre sırala (yüksek confidence önce)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Yangın tespiti: En fazla 1 detection
        return [detections[0]]
    
    def _calculate_iou(self, box1, box2):
        """IoU (Intersection over Union) hesapla"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_width = max(0, xi_max - xi_min)
        inter_height = max(0, yi_max - yi_min)
        inter_area = inter_width * inter_height
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        # IoU
        if union_area == 0:
            return 0
        return inter_area / union_area
    
    def _calculate_severity(self, x1: float, y1: float, x2: float, y2: float) -> str:
        """Yangının ciddiyetini hesapla (bbox boyutuna göre)"""
        area = (x2 - x1) * (y2 - y1)
        if area > 100000:
            return "high"
        elif area > 50000:
            return "medium"
        else:
            return "low"

# YOLO11n Agent - Yeni entegrasyon
from agents.yolo11n_agent import YOLO11nAgent

__all__ = ['HelmetAgent', 'VestAgent', 'FireAgent', 'YOLO11nAgent']
