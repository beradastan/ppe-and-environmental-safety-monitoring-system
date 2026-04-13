"""
Detection Pipeline - Ana Workflow
Görüntü -> Ajanlar -> Formatter -> LLM -> Öneriler
"""

import cv2
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime

from agents.specific_agents import HelmetAgent, VestAgent, FireAgent, YOLO11nAgent
from pipeline.result_formatter import DetectionResultFormatter


class DetectionPipeline:
    """Ana detection pipeline - tüm bileşenleri birleştir"""
    
    def __init__(self, verbose: bool = True, use_yolo11n: bool = False):
        """
        Args:
            verbose: Detaylı log çıktısı?
            use_yolo11n: YOLO11n modelini kullan? (True) / Hexmon kullan? (False)
        """
        self.verbose = verbose
        self.use_yolo11n = use_yolo11n
        
        self.logger = logging.getLogger("DetectionPipeline")
        handler = logging.StreamHandler()
        level = logging.DEBUG if verbose else logging.INFO
        formatter = logging.Formatter('%(asctime)s - Pipeline - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(level)
        
        # Ajanları başlat
        self.logger.info("🚀 Detection Pipeline başlatılıyor...")
        
        if use_yolo11n:
            self.logger.info("📦 YOLO11n modeli kullanılıyor (Vest & Hardhat)")
            self.yolo11n_agent = YOLO11nAgent(confidence_threshold=0.25, device="cpu")
            self.helmet_agent = None
            self.vest_agent = None
        else:
            self.logger.info("📦 Hexmon modeli kullanılıyor")
            self.helmet_agent = HelmetAgent(confidence_threshold=0.30, device="cpu")
            self.vest_agent = VestAgent(confidence_threshold=0.25, device="cpu")
            self.yolo11n_agent = None
        
        self.fire_agent = FireAgent(confidence_threshold=0.5, device="cpu")
        
        # Formatter başlat
        self.formatter = DetectionResultFormatter()
        
        self.logger.info("✅ Pipeline hazır!")
    
    def process_image(
        self,
        image_path: str,
        save_results: bool = True,
        return_formatted: bool = True
    ) -> Dict[str, Any]:
        """
        Tek bir görüntüyü işle
        
        Args:
            image_path: Görüntü dosya yolu
            save_results: Sonuçları kaydet?
            return_formatted: Formatted JSON döndür?
            
        Returns:
            Formatted JSON ve meta bilgiler
        """
        
        self.logger.info(f"📸 Görüntü işleniyor: {image_path}")
        
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Görüntü yüklenemedi: {image_path}")
            
            # Detection yap
            self.logger.info("🔍 Detection başlıyor...")
            
            if self.use_yolo11n:
                # YOLO11n modeli kullanılıyor
                yolo11n_results = self.yolo11n_agent.detect(image)
                
                # Hardhat ve Vest sonuçlarını ayrı ayrı formatla
                helmet_results = {
                    "agent": "YOLO11nAgent",
                    "detections": yolo11n_results.get("hardhat_detections", []),
                    "warnings": yolo11n_results.get("hardhat_warnings", []),
                    "detection_count": len(yolo11n_results.get("hardhat_detections", [])),
                    "warning_count": len(yolo11n_results.get("hardhat_warnings", []))
                }
                
                vest_results = {
                    "agent": "YOLO11nAgent",
                    "detections": yolo11n_results.get("vest_detections", []),
                    "warnings": yolo11n_results.get("vest_warnings", []),
                    "detection_count": len(yolo11n_results.get("vest_detections", [])),
                    "warning_count": len(yolo11n_results.get("vest_warnings", []))
                }
            else:
                # Hexmon modeli kullanılıyor
                helmet_results = self.helmet_agent.detect(image)
                vest_results = self.vest_agent.detect(image)
            
            # Fire tespiti (her zaman)
            fire_results = self.fire_agent.detect(image)
            
            # Sonuçları formatla
            self.logger.info("📋 Sonuçlar formatlanıyor...")
            
            if return_formatted:
                formatted_results = self.formatter.format_results(
                    helmet_results,
                    vest_results,
                    fire_results
                )
            else:
                formatted_results = None
            
            # Final sonuç
            result = {
                "success": True,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "model_used": "YOLO11n" if self.use_yolo11n else "Hexmon",
                "raw_detections": {
                    "helmet": helmet_results,
                    "vest": vest_results,
                    "fire": fire_results
                },
                "formatted_results": formatted_results
            }
            
            # Sonuçları kaydet
            if save_results and formatted_results:
                self._save_results(image_path, formatted_results)
            
            self.logger.info(f"✅ İşlem tamamlandı")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Hata: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def process_batch(
        self,
        image_dir: str,
        extensions: list = ["*.jpg", "*.png", "*.webp"],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Bir klasördeki tüm görüntüleri işle
        
        Args:
            image_dir: Görüntü klasörü
            extensions: İşlenecek dosya uzantıları
            save_results: Sonuçları kaydet?
            
        Returns:
            İşlem özeti ve sonuçlar
        """
        
        self.logger.info(f"📂 Batch işlemi başlıyor: {image_dir}")
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            self.logger.error(f"Klasör bulunamadı: {image_dir}")
            return {"success": False, "error": "Klasör bulunamadı"}
        
        # Görüntüleri bul
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(ext))
        
        self.logger.info(f"📸 Bulunan görüntü: {len(image_files)}")
        
        results = {
            "success": True,
            "total_images": len(image_files),
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        # Her görüntüyü işle
        for image_path in image_files:
            result = self.process_image(
                str(image_path),
                save_results=save_results,
                return_formatted=True
            )
            
            results["results"].append(result)
            results["processed"] += 1
            
            if result.get("success"):
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        self.logger.info(f"✅ Batch işlemi tamamlandı")
        self.logger.info(f"   Toplam: {results['processed']}, Başarılı: {results['successful']}, Başarısız: {results['failed']}")
        
        return results
    
    def _save_results(self, image_path: str, formatted_results: Dict) -> None:
        """Sonuçları JSON dosyasına kaydet"""
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        image_name = Path(image_path).stem
        json_file = results_dir / f"pipeline_{image_name}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)
        
        self.logger.debug(f"💾 Sonuçlar kaydedildi: {json_file}")
    
    def get_llm_prompt(self, image_path: str) -> str:
        """Görüntü için LLM prompt oluştur"""
        
        result = self.process_image(image_path, save_results=False)
        
        if result['success']:
            helmet_results = result['raw_detections']['helmet']
            vest_results = result['raw_detections']['vest']
            fire_results = result['raw_detections']['fire']
            
            prompt = self.formatter.to_llm_prompt(
                helmet_results,
                vest_results,
                fire_results
            )
            return prompt
        else:
            return f"Hata: {result.get('error')}"


if __name__ == "__main__":
    # Örnek kullanım
    pipeline = DetectionPipeline(verbose=True)
    
    # Tek görüntü işle
    result = pipeline.process_image("test/group-workers-working-factory-260nw-2642536197.webp")
    
    if result['success'] and result['formatted_results']:
        print("\n📊 SUMMARY:")
        print(f"Status: {result['formatted_results']['summary']['safety_status']}")
        print(f"Risk Level: {result['formatted_results']['analysis']['risk_level']}")

