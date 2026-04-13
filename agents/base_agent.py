"""
BaseAgent - Tüm CNN ajanlarının temel sınıfı
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import logging
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class BaseAgent(ABC):
    """Tüm detection ajanlarının temel sınıfı"""
    
    def __init__(
        self,
        agent_name: str,
        model_name: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Args:
            agent_name: Ajanın adı (HelmetAgent, VestAgent, FireAgent)
            model_name: Kullanılacak YOLO modeli
            confidence_threshold: Güven eşiği
            device: "cpu" veya "cuda"
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Logging ayarı
        self.logger = logging.getLogger(self.agent_name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Model yükleme
        self.model = None
        self.load_model()
        
        # İstatistikler
        self.detection_stats = {
            "total_frames": 0,
            "detections_made": 0,
            "avg_confidence": 0.0,
            "processing_time": 0.0
        }
    
    def load_model(self) -> None:
        """YOLO modelini yükle"""
        try:
            self.model = YOLO(self.model_name)
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
            self.logger.info(f"Model başarıyla yüklendi: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
            raise
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Görüntü ön işleme - alt sınıflar tarafından implement edilmeli"""
        pass
    
    @abstractmethod
    def postprocess(self, results: Any) -> Dict:
        """Sonuç işleme - alt sınıflar tarafından implement edilmeli"""
        pass
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Ana detection fonksiyonu
        
        Args:
            image: Input görüntü (numpy array)
            
        Returns:
            Detection sonuçları içeren dictionary
        """
        try:
            import time
            start_time = time.time()
            
            # Ön işleme
            processed_image = self.preprocess(image)
            
            # Detection
            results = self.model(processed_image, conf=self.confidence_threshold, verbose=False)
            
            # Sonuç işleme
            processed_results = self.postprocess(results)
            
            # İstatistikleri güncelle
            self.detection_stats["total_frames"] += 1
            self.detection_stats["processing_time"] = time.time() - start_time
            
            if processed_results["detections"]:
                self.detection_stats["detections_made"] += len(processed_results["detections"])
                confidences = [d["confidence"] for d in processed_results["detections"]]
                self.detection_stats["avg_confidence"] = np.mean(confidences)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Detection hatası: {str(e)}")
            return {
                "agent": self.agent_name,
                "detections": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_statistics(self) -> Dict:
        """Ajanın istatistiklerini döndür"""
        return {
            "agent": self.agent_name,
            "statistics": self.detection_stats.copy()
        }
    
    def reset_statistics(self) -> None:
        """İstatistikleri sıfırla"""
        self.detection_stats = {
            "total_frames": 0,
            "detections_made": 0,
            "avg_confidence": 0.0,
            "processing_time": 0.0
        }
    
    def visualize_results(self, image: np.ndarray, results: Dict, output_path: Optional[str] = None) -> np.ndarray:
        """
        Detection sonuçlarını görüntüle
        
        Args:
            image: Orijinal görüntü
            results: Detection sonuçları
            output_path: Çıktı resim yolu (opsiyonel)
            
        Returns:
            Anotasyonlu görüntü
        """
        annotated_image = image.copy()
        
        for detection in results.get("detections", []):
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            label = detection["label"]
            
            # Bbox çiz
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Label ve confidence yaz
            text = f"{label}: {confidence:.2f}"
            cv2.putText(
                annotated_image, text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            self.logger.info(f"Görüntü kaydedildi: {output_path}")
        
        return annotated_image

