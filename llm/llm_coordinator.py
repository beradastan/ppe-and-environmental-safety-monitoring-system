"""
LLMCoordinator - Ollama/LocalLLM entegrasyonu
Deteksiyon sonuçlarini LLM'e göndererek analiz yap
"""

import logging
import json
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime


class OllamaLLMCoordinator:
    """Ollama LLM kullanarak detection analizi ve öneriler
    
    ⚡ OPTİMİZE MODLAR:
    - use_minimal_format=True (DEFAULT) → Sadece ihlal özeti, hızlı
    - use_minimal_format=False → Detaylı format, daha açıklayıcı
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "mistral",
        timeout: int = 60,
        offline_mode: bool = False
    ):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.timeout = timeout
        self.offline_mode = offline_mode

        self.logger = logging.getLogger("OllamaLLMCoordinator")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - LLM - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        if not offline_mode:
            self.check_connection()
        else:
            self.logger.warning("OFFLINE MODE - Mock LLM kullaniliyor")

    # ── Baglanti & temel request ───────────────────────────────────

    def check_connection(self) -> bool:
        """Ollama sunucusuna baglanti kontrol et"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                self.logger.info(f"Ollama baglantisi basarili: {self.ollama_base_url}")
                return True
            self.logger.warning(f"Ollama baglanti hatasi: {response.status_code}")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"Ollama sunucusuna baglanamadi: {self.ollama_base_url}")
            return False
        except Exception as e:
            self.logger.error(f"Hata: {str(e)}")
            return False

    def generate_response(self, prompt: str, temperature: float = 0.2) -> str:
        """LLM'ye prompt göndererek yanit al."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            self.logger.error(f"LLM hata: {response.status_code}")
            return ""
        except Exception as e:
            self.logger.error(f"Prompt gonderme hatasi: {str(e)}")
            return ""

    # ── Formatlama — CNN ciktisi → LLM'in anlayacagi metin ────────

    def format_for_llm_minimal(self, helmet_result, vest_result, fire_result,
                              event_info: dict = None) -> str:
        """
        Minimal format - event bilgisi içerebilir.

        Args:
            event_info: event_manager'dan gelen dict
                {
                    "event_id": str,
                    "event_status": str,
                    "repeat_count": int,
                    "duration_sec": float,
                    "change_reason": str,
                    ...
                }
        """
        h_viol = helmet_result.get("warning_count", 0)
        v_viol = vest_result.get("warning_count", 0)
        f_count = fire_result.get("detection_count", 0)

        fire_conf = 0.0
        if f_count > 0 and fire_result.get("detections"):
            fire_conf = fire_result["detections"][0].get("confidence", 0.0)

        lines = [
            f"helmet_violation_count={h_viol}",
            f"vest_violation_count={v_viol}",
            f"fire_detected={'yes' if f_count > 0 else 'no'}",
            f"fire_confidence={fire_conf:.2f}"
        ]

        # Event bilgisi ekle
        if event_info:
            lines.insert(0, f"event_id={event_info.get('event_id', 'N/A')}")
            lines.insert(1, f"event_status={event_info.get('event_status', 'N/A')}")
            lines.append(f"repeat_count={event_info.get('repeat_count', 0)}")
            lines.append(f"duration_sec={event_info.get('duration_sec', 0):.1f}")
            if event_info.get('change_reason'):
                lines.append(f"change_reason={event_info['change_reason']}")

        return "\n".join(lines)

    def format_for_llm(
        self,
        helmet_result: Dict,
        vest_result: Dict,
        fire_result: Dict,
        image_name: str = "unknown",
    ) -> str:
        """
        CNN ajanlarinin ham sonuclarini LLM'in kolayca yorumlayabilecegi
        Turkce yapilandirilmis metin formatina cevirir.

        'detections' -> guvenli kisiler, 'warnings' -> ihlaller (alarm).
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        h_safe  = helmet_result.get("detection_count", 0)  # baret giyen
        h_viol  = helmet_result.get("warning_count", 0)    # baretsiz (ihlal)
        v_safe  = vest_result.get("detection_count", 0)    # yelek giyen
        v_viol  = vest_result.get("warning_count", 0)      # yeleksiz (ihlal)
        f_count = fire_result.get("detection_count", 0)

        f_guven = ""
        if f_count > 0 and fire_result.get("detections"):
            f_guven = f" (guven skoru: {fire_result['detections'][0]['confidence']:.2f})"

        ihlaller = []
        if h_viol > 0:
            ihlaller.append(f"{h_viol} kisi BARETSIZ calistirildi")
        if v_viol > 0:
            ihlaller.append(f"{v_viol} kisi GUVENLIK YELEKSIZ calistirildi")
        if f_count > 0:
            ihlaller.append(f"YANGIN tespit edildi{f_guven}")

        alarm_str   = "EVET" if ihlaller else "HAYIR"
        ihlal_str   = "\n  - ".join(ihlaller) if ihlaller else "Yok"

        return (
            f"=== FABRIKA GUVENLIK TESPIT RAPORU ===\n"
            f"Zaman        : {ts}\n"
            f"Goruntu      : {image_name}\n"
            f"\n"
            f"[BARET (HARDHAT) DURUMU]\n"
            f"  Uyumlu  (baret takmis)   : {h_safe} kisi\n"
            f"  IHLAL   (baretsiz)       : {h_viol} kisi"
            f"{' <-- ALARM' if h_viol > 0 else ''}\n"
            f"\n"
            f"[GUVENLIK YELEGI (VEST) DURUMU]\n"
            f"  Uyumlu  (yelek giymis)   : {v_safe} kisi\n"
            f"  IHLAL   (yeleksiz)       : {v_viol} kisi"
            f"{' <-- ALARM' if v_viol > 0 else ''}\n"
            f"\n"
            f"[YANGIN DURUMU]\n"
            f"  Yangin tespit edildi     : {'EVET' + f_guven if f_count > 0 else 'HAYIR'}"
            f"{' <-- ALARM' if f_count > 0 else ''}\n"
            f"\n"
            f"[ALARM TETIKLENDI] : {alarm_str}\n"
            f"[IHLALLER]         :\n"
            f"  - {ihlal_str}\n"
        )

    # ── Ana raporlama — sadece alarm=True durumunda LLM cagrisi ───

    def generate_alarm_report(
        self,
        helmet_result: Dict,
        vest_result: Dict,
        fire_result: Dict,
        image_name: str = "unknown",
        use_minimal_format: bool = True,
        event_info: dict = None,
    ) -> Dict[str, Any]:
        """
        ⚡ OPTİMİZE - Alarm durumunda MINIMAL veri ile LLM'e gönder
        
        - use_minimal_format=True (default) → Sadece ihlal özeti (hızlı)
        - use_minimal_format=False → Detaylı format (eski davranış)
        - event_info: event_manager'dan gelen dict (opsiyonel)

        Alarm yoksa LLM çağrılmaz (güvenli sahne).

        Returns dict:
          alarm      : bool
          llm_called : bool
          report     : str
          structured : str (minimal veya detaylı)
          image      : str
          timestamp  : str
          event_id   : str (event_info varsa)
          event_status : str (event_info varsa)
        """
        # Alarm kontrol
        alarm = (
            helmet_result.get("warning_count", 0) > 0
            or vest_result.get("warning_count", 0) > 0
            or fire_result.get("detection_count", 0) > 0
        )

        if not alarm:
            return {
                "alarm"      : False,
                "llm_called" : False,
                "report"     : "Sahne guvenli.",
                "structured" : "guvenli",
                "image"      : image_name,
                "timestamp"  : datetime.now().isoformat(),
                "event_id"   : event_info.get("event_id") if event_info else None,
                "event_status": event_info.get("event_status") if event_info else None,
            }

        # Alarm var → LLM'e minimal veri gönder
        if use_minimal_format:
            violations = self.format_for_llm_minimal(
                helmet_result, vest_result, fire_result, event_info=event_info
            )
            
            # ⚡ SIKI PROMPT - Halüsinasyon engelle, sadece faktual cevap
            prompt = (
                "Sen bir iş güvenliği olay raporlama asistanısın.\n"
                "Aşağıdaki veriler doğrulanmış CNN ihlal özetidir.\n"
                "Sadece bu verilere dayan.\n"
                "1 veya 2 kısa cümle yaz.\n"
                "Önce ihlali söyle, sonra tek bir acil aksiyon öner.\n"
                "Sayıları doğru kullan, yeni bilgi uydurma.\n\n"
                f"{violations}\n\n"
                "Kısa rapor:"
            )
            structured = violations
        else:
            # Detaylı format (eski davranış)
            structured = self.format_for_llm(
                helmet_result, vest_result, fire_result, image_name
            )
            prompt = (
                "Sen bir endüstriyel iş sağlığı uzmanısın.\n"
                "Aşağıdaki tespit raporuna dayanarak 3 cümlelik rapor yaz.\n"
                "Yalnızca ihlallere odaklan, net ve profesyonel ol.\n\n"
                f"{structured}\n"
                "Rapor:"
            )

        self.logger.info(f"LLM raporu isteniyor: {image_name}")

        if self.offline_mode:
            llm_text = self._mock_report(helmet_result, vest_result, fire_result)
        else:
            # ⚡ Temperature = 0.1 (çok düşük → deterministik, halüsinasyon az)
            llm_text = self.generate_response(prompt, temperature=0.1)

        if not llm_text:
            llm_text = "[LLM yanit veremedi — Ollama servisi kontrol ediniz]"

        result = {
            "alarm"      : True,
            "llm_called" : True,
            "report"     : llm_text.strip(),
            "structured" : structured,
            "image"      : image_name,
            "timestamp"  : datetime.now().isoformat(),
        }

        # Event bilgisi ekle
        if event_info:
            result["event_id"] = event_info.get("event_id")
            result["event_status"] = event_info.get("event_status")
            result["repeat_count"] = event_info.get("repeat_count")
            result["duration_sec"] = event_info.get("duration_sec")
            result["change_reason"] = event_info.get("change_reason")

        return result

    def _mock_report(self, helmet_result, vest_result, fire_result) -> str:
        """Offline/test modu icin mock rapor - ÇOOK KISA"""
        parts = []
        
        if helmet_result.get("warning_count", 0) > 0:
            n = helmet_result["warning_count"]
            parts.append(f"{n} kişi baret takmıyor.")
        
        if vest_result.get("warning_count", 0) > 0:
            n = vest_result["warning_count"]
            parts.append(f"{n} kişi yelek takmıyor.")
        
        if fire_result.get("detection_count", 0) > 0:
            parts.append("Yangın tespit edildi.")
        
        # Acil eylem
        if parts:
            parts.append("Derhal tedbirler alınmalı.")
        
        return " ".join(parts)

    # ── Eski metodlar (gerive donus uyumluluk) ────────────────────

    def coordinate_detections(self, detection_results: Dict) -> Dict:
        """[Eski] Coklu detection sonuclarindan hareketle LLM ile karar ver"""
        try:
            detection_summary = self._format_detection_summary(detection_results)
            prompt = (
                "Sen bir Endustriyel Is Sagligi ve Guvenligi Uzmani Yapay Zekasın.\n"
                "Asagidaki guvenlik tespiti sonuclarina dayanarak kisa, profesyonel bir rapor olustur:\n\n"
                f"{detection_summary}\n\n"
                "Raporunda: Genel Durum, Risk Seviyesi, Kritik Ihlaller, Acil Eylem."
            )
            response = self.generate_response(prompt, temperature=0.3)
            return {
                "timestamp"     : datetime.now().isoformat(),
                "coordinator"   : "LLMCoordinator",
                "llm_analysis"  : response,
                "raw_detections": detection_results,
                "status"        : "success"
            }
        except Exception as e:
            self.logger.error(f"Koordinasyon hatasi: {str(e)}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e), "status": "failed"}

    def _format_detection_summary(self, detection_results: Dict) -> str:
        """[Eski] Detection sonuclarini LLM icin formatla"""
        summary = "Detection Sonuclari:\n"
        for agent_name, result in detection_results.items():
            if isinstance(result, dict):
                detection_count = result.get("detection_count", 0)
                detections = result.get("detections", [])
                summary += f"\n{agent_name}:\n  - Toplam Tespit: {detection_count}\n"
                for i, det in enumerate(detections[:3]):
                    label = det.get("label", "unknown")
                    confidence = det.get("confidence", 0)
                    summary += f"  - {i+1}. {label} (Guven: {confidence:.2%})\n"
                if len(detections) > 3:
                    summary += f"  - ... ve {len(detections) - 3} daha tespit\n"
        return summary

    def analyze_detection(self, llm_prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """[Eski] Detection sonuclarini LLM'e göndererek analiz yap"""
        try:
            response = self.generate_response(llm_prompt)
            if response:
                return {"success": True, "analysis": response, "model": self.model_name,
                        "tokens_generated": len(response.split())}
            return {"success": False, "error": "Ollama yanit veremedi", "model": self.model_name}
        except Exception as e:
            self.logger.error(f"Analysis hatasi: {str(e)}")
            return {"success": False, "error": str(e), "model": self.model_name}

    def get_agent_instructions(self, detection_type: str) -> str:
        """[Eski] Belirli detection turu icin LLM'den talimat al"""
        try:
            prompt = f"Guvenlik gozetim sistemi: '{detection_type}' tespiti icin adimlar (max 100 kelime)."
            return self.generate_response(prompt, temperature=0.2)
        except Exception as e:
            self.logger.error(f"Talimat alinamadi: {str(e)}")
            return ""


class MockLLMCoordinator:
    """Test icin mock LLM coordinator"""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self.logger = logging.getLogger("MockLLMCoordinator")

    def generate_alarm_report(self, helmet_result, vest_result, fire_result,
                              image_name="unknown") -> Dict[str, Any]:
        """Mock alarm raporu (offline mode ile OllamaLLMCoordinator kullanir)"""
        coordinator = OllamaLLMCoordinator(offline_mode=True)
        return coordinator.generate_alarm_report(
            helmet_result, vest_result, fire_result, image_name
        )

    def analyze_detection(self, llm_prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """[Eski] Mock analiz"""
        mock_response = ("Fabrika guvenlik durumu degerlendirildi. "
                         "Eksik KKD tespiti yapilmistir. "
                         "Derhal denetim baslatilmalidir.")
        return {"success": True, "analysis": mock_response, "model": self.model_name, "is_mock": True}
