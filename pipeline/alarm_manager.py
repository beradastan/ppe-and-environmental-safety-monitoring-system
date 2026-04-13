"""
AlarmManager - Ajanlardan gelen JSON deteksiyon sonuçlarına göre anında alarm üretir
"""

import logging
from typing import Dict, List
import json
import winsound
from datetime import datetime
import os

class AlarmManager:
    """Deteksiyon JSON/Dict verisine bakarak kural bazlı alarm üretir"""
    
    def __init__(self, log_dir: str = "./results"):
        self.logger = logging.getLogger("AlarmManager")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - 🚨 ALARM - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.alarm_log_file = os.path.join(self.log_dir, "alarms.log")
        
    def evaluate_detections(self, detections: Dict) -> bool:
        """
        Gelen deteksiyon sonuçlarını (Dict) inceler ve kritik durum varsa alarm çalar.

        Kurallar:
          1. FireAgent  → herhangi bir yangan/alev bulduysa       → KRİTİK ALARM
          2. HelmetAgent → warnings listesinde NO-Hardhat varsa    → UYARI ALARMI
          3. VestAgent  → warnings listesinde NO-Safety Vest varsa → UYARI ALARMI

        Args:
            detections: Ajanların üretiti raw JSON/Dict
        Returns:
            Alarm tetiklendiyse True, aksi halde False
        """
        alarm_triggered = False
        reasons = []

        # 1. Yangın Kontrolü (En Yüksek Öncelik)
        if "FireAgent" in detections:
            fire_data = detections["FireAgent"]
            if isinstance(fire_data, dict) and fire_data.get("detection_count", 0) > 0:
                for det in fire_data.get("detections", []):
                    reasons.append(f"YANGIN TESPITİ! (Güven: {det.get('confidence', 0):.2f})")
                alarm_triggered = True

        # 2. Baret Kontrolü
        # Düzeltme: warnings listesi kontrol ediliyor (NO-Hardhat oraya yazılıyor)
        if "HelmetAgent" in detections:
            helmet_data = detections["HelmetAgent"]
            if isinstance(helmet_data, dict) and helmet_data.get("warning_count", 0) > 0:
                for warn in helmet_data.get("warnings", []):
                    reasons.append(f"BARETSIZ PERSONEL! (Güven: {warn.get('confidence', 0):.2f})")
                    alarm_triggered = True

        # 3. Yelek Kontrolü
        # Düzeltme: warnings listesi kontrol ediliyor (NO-Safety Vest oraya yazılıyor)
        if "VestAgent" in detections:
            vest_data = detections["VestAgent"]
            if isinstance(vest_data, dict) and vest_data.get("warning_count", 0) > 0:
                for warn in vest_data.get("warnings", []):
                    reasons.append(f"YELEKSİZ PERSONEL! (Güven: {warn.get('confidence', 0):.2f})")
                    alarm_triggered = True

        # Eğer alarm şartları oluştuysa tetikle
        if alarm_triggered:
            self._trigger_alarm(reasons)

        return alarm_triggered
        
    def _trigger_alarm(self, reasons: List[str]):
        """Alarm mekanizmalarını (Ses, Log, Konsol) çalıştırır"""
        alarm_message = " | ".join(reasons)
        
        # Konsola Bas
        self.logger.warning(f"Kritik Durum: {alarm_message}")
        
        # Dosyaya Logla
        with open(self.alarm_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] ALARM: {alarm_message}\n")
            
        # Sesli Uyarı (Windows - Kısa süreli Beep)
        try:
            # Frekans: 1000 Hz, Süre: 500 ms
            winsound.Beep(1000, 500)
        except Exception:
            pass  # Ses çalınamasa bile hata fırlatma
