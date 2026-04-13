# -*- coding: utf-8 -*-
"""
Hard Hat Detection - LLM Coordinator Integration
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.hardhat_agent import HardHatDetectionAgent

class HardHatLLMAgent:
    """Hard Hat Detection ve LLM entegrasyonu"""
    
    def __init__(self, model_path: Optional[str] = None, llm_coordinator=None):
        """
        Args:
            model_path: Hard Hat model dosyası
            llm_coordinator: LLM Coordinator örneği
        """
        print("INIT Hard Hat LLM Agent baslatiliyor...")
        
        self.detector = HardHatDetectionAgent(model_path)
        self.llm = llm_coordinator
        
        print("OK Hard Hat LLM Agent hazir\n")
    
    def analyze_safety(self, image_source: str, conf: float = 0.5) -> Dict[str, Any]:
        """
        Güvenlik analizi yap
        
        Args:
            image_source: Görsel dosyası
            conf: Confidence threshold
        
        Returns:
            Güvenlik analizi sonuçları
        """
        print(f"ANALYZE Guvenlilk analizi yapiliyor: {image_source}")
        
        # Deteksiyon yap
        detection = self.detector.detect(image_source, conf=conf)
        
        if detection.get('error'):
            return {
                "status": "ERROR",
                "error": detection['error']
            }
        
        # Güvenlik analizi
        analysis = self._analyze_detections(detection)
        
        print(f"COMPLETE Analiz tamamlandi\n")
        
        return {
            "status": "SUCCESS",
            "detection": detection,
            "safety_analysis": analysis,
            "image": image_source
        }
    
    def _analyze_detections(self, detection: Dict) -> Dict[str, Any]:
        """Deteksiyon sonuçlarını güvenlik açısından analiz et"""
        
        analysis = {
            "safe": True,
            "warnings": [],
            "critical": [],
            "recommendations": []
        }
        
        # Temel kontroller
        if not detection.get('has_person'):
            analysis["recommendations"].append("Hiç kimse tespit edilmedi")
            return analysis
        
        # Hard Hat kontrolleri
        if detection.get('has_person') and not detection.get('has_hardhat'):
            analysis["safe"] = False
            analysis["critical"].append("Kask kullanilmamis - TEHLIKELI!")
            analysis["recommendations"].append("Insanlar kask kullanmali")
        
        # Safety Gear kontrolleri
        if detection.get('has_person') and not detection.get('has_safety_gear'):
            analysis["safe"] = False
            analysis["warnings"].append("Guvenlik giysisi eksik")
            analysis["recommendations"].append("Guvenlik giysisi giyilmeli")
        
        # Kombineli kontrol
        if detection.get('has_hardhat') and detection.get('has_safety_gear'):
            analysis["safe"] = True
            analysis["recommendations"].append("Tum guvenlik onlemleri uygulanmis")
        elif detection.get('has_hardhat'):
            analysis["warnings"].append("Kask var ama guvenlik giysisi yok")
        elif detection.get('has_safety_gear'):
            analysis["warnings"].append("Guvenlik giysisi var ama kask yok")
        
        return analysis
    
    def batch_analyze(self, image_dir: str, conf: float = 0.5) -> List[Dict]:
        """
        Klasordeki tum gorseleri analiz et
        
        Args:
            image_dir: Gorsel klasoru
            conf: Confidence threshold
        
        Returns:
            Analiz sonuçlari listesi
        """
        print(f"BATCH Klasor olusturuluyor: {image_dir}")
        
        image_dir = Path(image_dir)
        results = []
        
        for image_path in list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")):
            analysis = self.analyze_safety(str(image_path), conf=conf)
            results.append(analysis)
        
        print(f"OK {len(results)} gorsel analiz edildi\n")
        
        return results
    
    def generate_llm_report(self, analysis_results: List[Dict]) -> str:
        """
        LLM ile rapor olustur
        
        Args:
            analysis_results: Analiz sonuçlari
        
        Returns:
            LLM tarafından olusturulan rapor
        """
        if not self.llm:
            print("WARNING LLM coordinator hazir degil")
            return self._generate_default_report(analysis_results)
        
        print("GENERATE LLM raporu olusturuluyor...")
        
        summary = self._create_summary(analysis_results)
        
        try:
            prompt = f"""
Lütfen şu inşaat alanı güvenlik inspeksiyonu raporu için bir özet yaz:

{summary}

Raporun şunları içermesi gerekir:
1. Genel güvenlik durumu
2. Bulunulan sorunlar
3. Öneriler
"""
            
            report = self.llm.generate_response(prompt)
            print("OK Rapor olusturuldu\n")
            return report
            
        except Exception as e:
            print(f"WARNING LLM rapor hatası: {e}")
            return self._generate_default_report(analysis_results)
    
    def _create_summary(self, results: List[Dict]) -> str:
        """Analiz sonuçlarından özet oluştur"""
        
        total = len(results)
        safe_count = sum(1 for r in results if r.get('safety_analysis', {}).get('safe'))
        unsafe_count = total - safe_count
        
        critical_issues = []
        warnings = []
        
        for result in results:
            analysis = result.get('safety_analysis', {})
            if analysis.get('critical'):
                critical_issues.extend(analysis['critical'])
            if analysis.get('warnings'):
                warnings.extend(analysis['warnings'])
        
        summary = f"""
Güvenlik İnceleme Özeti:
- Toplam İncelenen: {total} görsel
- Güvenli: {safe_count}
- Güvensiz: {unsafe_count}

Kritik Sorunlar ({len(critical_issues)}):
"""
        for issue in critical_issues:
            summary += f"- {issue}\n"
        
        summary += f"\nUyarılar ({len(warnings)}):\n"
        for warning in warnings:
            summary += f"- {warning}\n"
        
        return summary
    
    def _generate_default_report(self, results: List[Dict]) -> str:
        """Varsayılan rapor oluştur (LLM olmadan)"""
        
        summary = self._create_summary(results)
        
        return f"""
INŞAAT ALANI GÜVENLIK İNCELEME RAPORU

{summary}

SONUÇ:
Lütfen tüm kritik sorunları hemen çözün.
Güvenlik önlemleri eksiksiz uygulanmalıdır.
"""


def demo():
    """Demo - Hard Hat LLM Agent'i test et"""
    
    print("=" * 70)
    print("DEMO HARD HAT LLM AGENT")
    print("=" * 70)
    print("")
    
    # Agent oluştur
    agent = HardHatLLMAgent()
    
    # Test görselü var mı kontrol et
    test_image = project_root / "test_image.jpg"
    
    if test_image.exists():
        print(f"TEST Gorsel bulundu: {test_image}\n")
        
        # Analiz yap
        result = agent.analyze_safety(str(test_image))
        
        print("RESULTS Sonuclar:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"INFO Test gorselü bulunamadi: {test_image}")
    
    print("\n" + "=" * 70)
    print("COMPLETE Demo tamamlandi")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo()

