# -*- coding: utf-8 -*-
"""
Hard Hat Detection Pipeline Modulu
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.hardhat_agent import HardHatDetectionAgent


class HardHatDetectionPipeline:
    """Hard Hat Detection pipeline'ı"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Model dosyası
        """
        print("INIT Hard Hat Detection Pipeline baslatiliyor...")
        
        self.detector = HardHatDetectionAgent(model_path)
        self.results_history: List[Dict] = []
        
        print("OK Pipeline hazir\n")
    
    def process_image(self, image_source: str, conf: float = 0.5) -> Dict[str, Any]:
        """
        Görsel işle
        
        Args:
            image_source: Görsel dosyası
            conf: Confidence threshold
        
        Returns:
            İşleme sonuçları
        """
        print(f"PROCESS Gorsel isleniyor: {image_source}")
        
        # Deteksiyon yap
        detection = self.detector.detect(image_source, conf=conf)
        
        # Pipeline sonucu oluştur
        result = {
            "timestamp": datetime.now().isoformat(),
            "image": str(image_source),
            "detection": detection,
            "processing_status": "SUCCESS" if not detection.get('error') else "ERROR",
            "summary": self._create_summary(detection)
        }
        
        # Geçmişe ekle
        self.results_history.append(result)
        
        print(f"OK Isleme tamamlandi\n")
        
        return result
    
    def process_batch(self, image_dir: str, conf: float = 0.5) -> List[Dict]:
        """
        Batch işleme
        
        Args:
            image_dir: Görsel klasörü
            conf: Confidence threshold
        
        Returns:
            İşleme sonuçları
        """
        print(f"BATCH Batch isleme basliyor: {image_dir}")
        
        image_dir = Path(image_dir)
        results = []
        
        for image_path in list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")):
            result = self.process_image(str(image_path), conf=conf)
            results.append(result)
        
        print(f"OK {len(results)} gorsel islendi\n")
        
        return results
    
    def process_video(self, video_path: str, conf: float = 0.5, 
                     frame_skip: int = 1) -> Dict[str, Any]:
        """
        Video işle (frame by frame)
        
        Args:
            video_path: Video dosyası
            conf: Confidence threshold
            frame_skip: Kaç frame'de bir deteksiyon yap
        
        Returns:
            Video işleme sonuçları
        """
        print(f"VIDEO Video isleniyor: {video_path} (skip={frame_skip})")
        
        try:
            import cv2
        except ImportError:
            print("ERROR cv2 kutuephanesi yok")
            return {"error": "cv2 kutuephanesi gerekli"}
        
        video_path = Path(video_path)
        if not video_path.exists():
            return {"error": f"Video bulunamadi: {video_path}"}
        
        cap = cv2.VideoCapture(str(video_path))
        
        frame_count = 0
        detection_count = 0
        detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % frame_skip == 0:
                # Deteksiyon yap
                results = self.detector.model(frame, conf=conf, verbose=False)
                
                if results[0].boxes:
                    detection_count += 1
                    detections.append({
                        "frame": frame_count,
                        "count": len(results[0].boxes),
                        "classes": [int(b.cls[0]) for b in results[0].boxes]
                    })
        
        cap.release()
        
        result = {
            "video": str(video_path),
            "total_frames": frame_count,
            "processed_frames": frame_count // frame_skip,
            "frames_with_detections": detection_count,
            "detections": detections,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results_history.append(result)
        
        print(f"OK Video islendi ({frame_count} frame, {detection_count} ile deteksiyon)\n")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """İstatistikleri al"""
        
        if not self.results_history:
            return {"error": "Henüz işlenmiş sonuç yok"}
        
        stats = {
            "total_processed": len(self.results_history),
            "successful": 0,
            "failed": 0,
            "total_detections": 0,
            "hardhat_count": 0,
            "person_count": 0,
            "safety_gear_count": 0,
            "confidence_avg": 0.0
        }
        
        all_confidences = []
        
        for result in self.results_history:
            if result.get('processing_status') == 'SUCCESS':
                stats['successful'] += 1
                
                detection = result.get('detection', {})
                detections_list = detection.get('detections', [])
                
                stats['total_detections'] += len(detections_list)
                stats['hardhat_count'] += 1 if detection.get('has_hardhat') else 0
                stats['person_count'] += 1 if detection.get('has_person') else 0
                stats['safety_gear_count'] += 1 if detection.get('has_safety_gear') else 0
                
                for det in detections_list:
                    all_confidences.append(det.get('confidence', 0))
            else:
                stats['failed'] += 1
        
        if all_confidences:
            stats['confidence_avg'] = sum(all_confidences) / len(all_confidences)
        
        return stats
    
    def export_results(self, output_path: str) -> bool:
        """Sonuçları dışa aktar"""
        
        print(f"EXPORT Sonuclar disa aktariliyor: {output_path}")
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "results": self.results_history,
                "statistics": self.get_statistics()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"OK Sonuclar kaydedildi: {output_path}\n")
            return True
            
        except Exception as e:
            print(f"ERROR Export hatası: {e}\n")
            return False
    
    def _create_summary(self, detection: Dict) -> str:
        """Deteksiyon özeti oluştur"""
        
        summary = {
            "total": detection.get('count', 0),
            "hardhat": "YES" if detection.get('has_hardhat') else "NO",
            "person": "YES" if detection.get('has_person') else "NO",
            "safety_gear": "YES" if detection.get('has_safety_gear') else "NO",
            "safe": "SAFE" if (detection.get('has_hardhat') and detection.get('has_person')) else "UNSAFE"
        }
        
        return summary
    
    def clear_history(self):
        """Geçmişi temizle"""
        self.results_history = []
        print("OK Gecmis temizlendi\n")


def demo():
    """Demo - Pipeline test"""
    
    print("=" * 70)
    print("DEMO HARD HAT DETECTION PIPELINE")
    print("=" * 70)
    print("")
    
    # Pipeline oluştur
    pipeline = HardHatDetectionPipeline()
    
    # Test görselü var mı kontrol et
    test_image = project_root / "test_image.jpg"
    
    if test_image.exists():
        print(f"TEST Gorsel bulundu: {test_image}\n")
        
        # İşle
        result = pipeline.process_image(str(test_image))
        
        print("RESULTS Sonuclar:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # İstatistikler
        stats = pipeline.get_statistics()
        print("\nSTATISTICS İstatistikler:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"INFO Test gorselü bulunamadi: {test_image}")
    
    print("\n" + "=" * 70)
    print("COMPLETE Demo tamamlandi")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo()

