
import logging
import json
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

class OllamaLLMCoordinator:

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
            self.logger.warning("OFFLINE MODE — using mock LLM")

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                self.logger.info(f"Ollama connection successful: {self.ollama_base_url}")
                return True
            self.logger.warning(f"Ollama connection error: {response.status_code}")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"Cannot connect to Ollama: {self.ollama_base_url}")
            return False
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            return False

    def generate_response(self, prompt: str, temperature: float = 0.2) -> str:
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
            self.logger.error(f"LLM error: {response.status_code}")
            return ""
        except Exception as e:
            self.logger.error(f"Error sending prompt: {str(e)}")
            return ""

    def format_for_llm_minimal(self, helmet_result, vest_result, fire_result,
                              event_info: dict = None) -> str:
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
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        h_safe  = helmet_result.get("detection_count", 0)
        h_viol  = helmet_result.get("warning_count", 0)
        v_safe  = vest_result.get("detection_count", 0)
        v_viol  = vest_result.get("warning_count", 0)
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

    def generate_alarm_report(
        self,
        helmet_result: Dict,
        vest_result: Dict,
        fire_result: Dict,
        image_name: str = "unknown",
        use_minimal_format: bool = True,
        event_info: dict = None,
    ) -> Dict[str, Any]:
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

        if use_minimal_format:
            violations = self.format_for_llm_minimal(
                helmet_result, vest_result, fire_result, event_info=event_info
            )
            
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

        self.logger.info(f"Requesting LLM report: {image_name}")

        if self.offline_mode:
            llm_text = self._mock_report(helmet_result, vest_result, fire_result)
        else:
            llm_text = self.generate_response(prompt, temperature=0.1)

        if not llm_text:
            llm_text = "[LLM unavailable — check Ollama service]"

        result = {
            "alarm"      : True,
            "llm_called" : True,
            "report"     : llm_text.strip(),
            "structured" : structured,
            "image"      : image_name,
            "timestamp"  : datetime.now().isoformat(),
        }

        if event_info:
            result["event_id"] = event_info.get("event_id")
            result["event_status"] = event_info.get("event_status")
            result["repeat_count"] = event_info.get("repeat_count")
            result["duration_sec"] = event_info.get("duration_sec")
            result["change_reason"] = event_info.get("change_reason")

        return result

    def _mock_report(self, helmet_result, vest_result, fire_result) -> str:
        parts = []
        
        if helmet_result.get("warning_count", 0) > 0:
            n = helmet_result["warning_count"]
            parts.append(f"{n} kişi baret takmıyor.")
        
        if vest_result.get("warning_count", 0) > 0:
            n = vest_result["warning_count"]
            parts.append(f"{n} kişi yelek takmıyor.")
        
        if fire_result.get("detection_count", 0) > 0:
            parts.append("Yangın tespit edildi.")
        
        if parts:
            parts.append("Derhal tedbirler alınmalı.")
        
        return " ".join(parts)

