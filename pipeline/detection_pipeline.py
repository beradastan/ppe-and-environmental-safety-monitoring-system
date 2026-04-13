"""
DetectionPipeline - Ana workflow ve koordinasyon
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from agents.specific_agents import HelmetAgent, VestAgent, FireAgent
from llm.llm_coordinator import OllamaLLMCoordinator as LLMCoordinator
from pipeline.alarm_manager import AlarmManager


class DetectionPipeline:
    """Ana detection pipeline - tüm ajanları koordine eder"""
    
    def __init__(
        self,
        use_llm: bool = True,
        ollama_url: str = "http://localhost:11434",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        save_results: bool = True,
        results_dir: str = "./results"
    ):
        """
        Args:
            use_llm: LLM coordinator kullanılsın mı?
            ollama_url: Ollama sunucu URL'i
            device: "cpu" veya "cuda"
            confidence_threshold: Detection güven eşiği
            save_results: Sonuçları kaydet
            results_dir: Sonuç klasörü
        """
        self.use_llm = use_llm
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        
        # Logging
        self.logger = logging.getLogger("DetectionPipeline")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - Pipeline - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Sonuç klasörünü oluştur
        if self.save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Ajanları başlat
        self.logger.info("Ajanlar başlatılıyor...")
        self.helmet_agent = HelmetAgent(
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.vest_agent = VestAgent(
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.fire_agent = FireAgent(
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # LLM Coordinator
        self.llm_coordinator = None
        if self.use_llm:
            self.logger.info("LLM Coordinator başlatılıyor...")
            self.llm_coordinator = LLMCoordinator(
                ollama_base_url=ollama_url
            )
            if not self.llm_coordinator.check_connection():
                self.logger.warning("LLM devre dışı bırakılıyor")
                self.llm_coordinator = None
                
        # Alarm Sistemi
        self.alarm_manager = AlarmManager(log_dir=results_dir)
        
        # İstatistikler
        self.pipeline_stats = {
            "total_frames": 0,
            "total_detections": 0,
            "processing_times": []
        }
        
        self.logger.info("Pipeline hazır!")
    
    def process_image(
        self,
        image: np.ndarray,
        use_llm_analysis: bool = True,
        save_annotated: bool = False
    ) -> Dict:
        """
        Tek bir görüntüyü işle
        
        Args:
            image: Input görüntü
            use_llm_analysis: LLM analizini kullan
            save_annotated: Anotasyonlu görüntüyü kaydet
            
        Returns:
            Toplam detection sonuçları
        """
        import time
        start_time = time.time()
        
        try:
            # Tüm ajanları çalıştır
            helmet_results = self.helmet_agent.detect(image)
            vest_results = self.vest_agent.detect(image)
            fire_results = self.fire_agent.detect(image)
            
            # Sonuçları birleştir
            all_results = {
                "HelmetAgent": helmet_results,
                "VestAgent": vest_results,
                "FireAgent": fire_results
            }
            
            # Anında Alarm Analizi (LLM beklemeden Python JSON bazlı kurallı check)
            alarm_status = self.alarm_manager.evaluate_detections(all_results)
            
            # LLM analizi (isteğe bağlı) ve Güvenlik Raporu
            llm_analysis = None
            if self.use_llm and self.llm_coordinator and use_llm_analysis:
                llm_analysis = self.llm_coordinator.coordinate_detections(all_results)
            
            # İstatistikleri güncelle
            self.pipeline_stats["total_frames"] += 1
            processing_time = time.time() - start_time
            self.pipeline_stats["processing_times"].append(processing_time)
            
            # Anotasyonlu görüntü kaydet
            if save_annotated and self.save_results:
                self._save_annotated_image(image, all_results)
            
            # Sonuç
            result = {
                "timestamp": datetime.now().isoformat(),
                "detections": all_results,
                "alarm_triggered": alarm_status,
                "llm_analysis": llm_analysis,
                "processing_time": processing_time,
                "frame_number": self.pipeline_stats["total_frames"]
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"İşleme hatası: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "frame_number": self.pipeline_stats["total_frames"]
            }
    
    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        process_every_n_frames: int = 1,
        use_llm_analysis: bool = False
    ) -> Dict:
        """
        Video dosyasını işle
        
        Args:
            video_path: Video dosyasının yolu
            output_video_path: Çıktı video yolu
            process_every_n_frames: Her N frame'de process et
            use_llm_analysis: LLM analizini kullan
            
        Returns:
            Video işleme sonuçları
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Video açılamadı: {video_path}")
            
            # Video özellikleri
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frame")
            
            # Video yazıcı (isteğe bağlı)
            out = None
            if output_video_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            frame_count = 0
            all_detections = []
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # İşleme frekansı kontrol et
                if frame_count % process_every_n_frames != 0:
                    continue
                
                # Frame'i işle
                result = self.process_image(
                    frame,
                    use_llm_analysis=use_llm_analysis,
                    save_annotated=False
                )
                
                all_detections.append(result)
                
                # Anotasyonlu frame
                annotated_frame = self._annotate_frame(frame, result)
                
                # Video'ya yaz
                if out:
                    out.write(annotated_frame)
                
                # Progress
                if frame_count % 30 == 0:
                    self.logger.info(f"İşlenen: {frame_count}/{total_frames}")
            
            cap.release()
            if out:
                out.release()
                self.logger.info(f"Video kaydedildi: {output_video_path}")
            
            return {
                "status": "success",
                "video_path": video_path,
                "output_path": output_video_path,
                "total_frames": frame_count,
                "detections": all_detections
            }
            
        except Exception as e:
            self.logger.error(f"Video işleme hatası: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_webcam(
        self,
        duration_seconds: int = 30,
        display: bool = True,
        use_llm_analysis: bool = False
    ) -> Dict:
        """
        Web kameradan canlı detection
        
        Args:
            duration_seconds: Kaç saniye çalıştır
            display: Sonuçları göster
            use_llm_analysis: LLM analizini kullan
            
        Returns:
            Detection sonuçları
        """
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise ValueError("Web kameraya erişilemedi")
            
            self.logger.info(f"Web kamera açıldı, {duration_seconds} saniye çalışacak...")
            
            all_detections = []
            frame_count = 0
            
            import time
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Frame'i işle
                result = self.process_image(
                    frame,
                    use_llm_analysis=use_llm_analysis,
                    save_annotated=False
                )
                
                all_detections.append(result)
                
                # Anotasyonlu frame
                annotated_frame = self._annotate_frame(frame, result)
                
                if display:
                    cv2.imshow("Detection Pipeline", annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            self.logger.info(f"Toplam işlenen frame: {frame_count}")
            
            return {
                "status": "success",
                "duration": duration_seconds,
                "total_frames": frame_count,
                "detections": all_detections
            }
            
        except Exception as e:
            self.logger.error(f"Web kamera hatası: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _annotate_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Frame'i detection sonuçlarıyla anotasyon yap"""
        annotated = frame.copy()
        
        detections = result.get("detections", {})
        
        for agent_name, agent_results in detections.items():
            for detection in agent_results.get("detections", []):
                x1, y1, x2, y2 = detection["bbox"]
                label = detection["label"]
                confidence = detection["confidence"]
                
                # Renk belirle
                if agent_name == "HelmetAgent":
                    color = (0, 255, 0)  # Yeşil
                elif agent_name == "VestAgent":
                    color = (255, 255, 0)  # Cyan
                else:  # FireAgent
                    color = (0, 0, 255)  # Kırmızı
                
                # Bbox çiz
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Text
                text = f"{label}: {confidence:.2f}"
                cv2.putText(annotated, text, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # LLM analizi metni
        llm_analysis = result.get("llm_analysis")
        if llm_analysis:
            status = llm_analysis.get("status", "")
            cv2.putText(annotated, f"LLM: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated
    
    def _save_annotated_image(self, image: np.ndarray, detections: Dict) -> None:
        """Anotasyonlu görüntüyü kaydet"""
        try:
            annotated = image.copy()
            
            for agent_name, result in detections.items():
                for detection in result.get("detections", []):
                    x1, y1, x2, y2 = detection["bbox"]
                    label = detection["label"]
                    confidence = detection["confidence"]
                    
                    if agent_name == "HelmetAgent":
                        color = (0, 255, 0)
                    elif agent_name == "VestAgent":
                        color = (255, 255, 0)
                    else:
                        color = (0, 0, 255)
                    
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(annotated, text, (int(x1), int(y1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = self.results_dir / filename
            cv2.imwrite(str(filepath), annotated)
            
        except Exception as e:
            self.logger.error(f"Görüntü kaydetme hatası: {str(e)}")
    
    def get_pipeline_statistics(self) -> Dict:
        """Pipeline istatistiklerini döndür"""
        return {
            "pipeline_stats": self.pipeline_stats,
            "helmet_agent": self.helmet_agent.get_statistics(),
            "vest_agent": self.vest_agent.get_statistics(),
            "fire_agent": self.fire_agent.get_statistics(),
            "avg_processing_time": (
                sum(self.pipeline_stats["processing_times"]) / 
                len(self.pipeline_stats["processing_times"])
                if self.pipeline_stats["processing_times"] else 0
            )
        }
    
    def save_results_to_json(self, results: Dict, filename: str = "results.json") -> str:
        """Sonuçları JSON dosyasına kaydet"""
        try:
            filepath = self.results_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Sonuçlar kaydedildi: {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"JSON kaydetme hatası: {str(e)}")
            return ""

