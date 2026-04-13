"""
Detection Result Formatter - Ajanlardan LLM'e standardize JSON çevirme
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class DetectionResultFormatter:
    """Ajan çıktılarını LLM-friendly format'a çevir"""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
    
    def format_results(self, helmet_results: Dict, vest_results: Dict, fire_results: Dict) -> Dict:
        """
        Tüm ajan sonuçlarını LLM'in anlayabileceği format'a çevir
        
        Returns:
            {
                "timestamp": "2026-03-28T14:30:00",
                "summary": {
                    "total_objects": 10,
                    "safety_status": "CAUTION",
                    "alerts": []
                },
                "detections": {
                    "helmets": [...],
                    "vests": [...],
                    "fires": [...]
                },
                "analysis": {
                    "ppe_compliance": "80%",
                    "risk_level": "MEDIUM"
                }
            }
        """
        
        # Helmet işleme
        helmets = self._format_helmets(helmet_results)
        
        # Vest işleme
        vests = self._format_vests(vest_results)
        
        # Fire işleme
        fires = self._format_fires(fire_results)
        
        # Uyarıları topla
        alerts = self._generate_alerts(helmets, vests, fires)
        
        # Safety status belirle
        safety_status = self._determine_safety_status(helmets, vests, fires)
        
        # Final format
        result = {
            "timestamp": self.timestamp,
            "summary": {
                "total_detections": len(helmets) + len(vests) + len(fires),
                "safety_status": safety_status,
                "alert_count": len(alerts),
                "alerts": alerts
            },
            "detections": {
                "helmets": helmets,
                "vests": vests,
                "fires": fires
            },
            "analysis": {
                "people_with_helmet": len([h for h in helmets if h['status'] == 'safe']),
                "people_without_helmet": helmet_results.get('warning_count', 0),
                "people_with_vest": len([v for v in vests if v['status'] == 'safe']),
                "people_without_vest": vest_results.get('warning_count', 0),
                "fire_detected": len(fires) > 0,
                "risk_level": self._calculate_risk_level(helmets, vests, fires)
            }
        }
        
        return result
    
    def _format_helmets(self, results: Dict) -> List[Dict]:
        """Helmet sonuçlarını format'la"""
        formatted = []
        
        # Positif tespitler (giymiş)
        for det in results.get('detections', []):
            formatted.append({
                "type": "helmet",
                "status": "safe",
                "confidence": round(det['confidence'], 3),
                "location": {
                    "x1": round(det['bbox'][0], 1),
                    "y1": round(det['bbox'][1], 1),
                    "x2": round(det['bbox'][2], 1),
                    "y2": round(det['bbox'][3], 1)
                },
                "message": f"Kişi baret/kask giymiş (%{det['confidence']*100:.1f})"
            })
        
        # Uyarılar (giymiş değil)
        for warn in results.get('warnings', []):
            formatted.append({
                "type": "helmet",
                "status": "warning",
                "confidence": round(warn['confidence'], 3),
                "location": {
                    "x1": round(warn['bbox'][0], 1),
                    "y1": round(warn['bbox'][1], 1),
                    "x2": round(warn['bbox'][2], 1),
                    "y2": round(warn['bbox'][3], 1)
                },
                "message": f"⚠️ Kişi BARET/KASK GİYMEMİŞ! (%{warn['confidence']*100:.1f})"
            })
        
        return formatted
    
    def _format_vests(self, results: Dict) -> List[Dict]:
        """Vest sonuçlarını format'la"""
        formatted = []
        
        # Positif tespitler (giymiş)
        for det in results.get('detections', []):
            formatted.append({
                "type": "vest",
                "status": "safe",
                "confidence": round(det['confidence'], 3),
                "location": {
                    "x1": round(det['bbox'][0], 1),
                    "y1": round(det['bbox'][1], 1),
                    "x2": round(det['bbox'][2], 1),
                    "y2": round(det['bbox'][3], 1)
                },
                "message": f"Kişi iş elbisesi/yelek giymiş (%{det['confidence']*100:.1f})"
            })
        
        # Uyarılar (giymiş değil)
        for warn in results.get('warnings', []):
            formatted.append({
                "type": "vest",
                "status": "warning",
                "confidence": round(warn['confidence'], 3),
                "location": {
                    "x1": round(warn['bbox'][0], 1),
                    "y1": round(warn['bbox'][1], 1),
                    "x2": round(warn['bbox'][2], 1),
                    "y2": round(warn['bbox'][3], 1)
                },
                "message": f"⚠️ Kişi YELEK/İŞ ELBİSESİ GİYMEMİŞ! (%{warn['confidence']*100:.1f})"
            })
        
        return formatted
    
    def _format_fires(self, results: Dict) -> List[Dict]:
        """Fire sonuçlarını format'la"""
        formatted = []
        
        for det in results.get('detections', []):
            formatted.append({
                "type": "fire",
                "status": "danger",
                "confidence": round(det['confidence'], 3),
                "severity": det.get('severity', 'unknown'),
                "location": {
                    "x1": round(det['bbox'][0], 1),
                    "y1": round(det['bbox'][1], 1),
                    "x2": round(det['bbox'][2], 1),
                    "y2": round(det['bbox'][3], 1)
                },
                "message": f"🔥 YANGIN TESPİT EDİLDİ! Ciddiyet: {det.get('severity', 'bilinmiyor')} (%{det['confidence']*100:.1f})"
            })
        
        return formatted
    
    def _generate_alerts(self, helmets: List, vests: List, fires: List) -> List[str]:
        """Uyarıları oluştur"""
        alerts = []
        
        # Fire uyarısı
        danger_fires = [f for f in fires if f['status'] == 'danger']
        if danger_fires:
            alerts.append(f"🚨 KRİTİK: {len(danger_fires)} yangın tespiti!")
        
        # Helmet uyarısı
        unsafe_helmets = [h for h in helmets if h['status'] == 'warning']
        if unsafe_helmets:
            alerts.append(f"⚠️  {len(unsafe_helmets)} kişi baret/kask GİYMEMİŞ!")
        
        # Vest uyarısı
        unsafe_vests = [v for v in vests if v['status'] == 'warning']
        if unsafe_vests:
            alerts.append(f"⚠️  {len(unsafe_vests)} kişi yelek/iş elbisesi GİYMEMİŞ!")
        
        return alerts
    
    def _determine_safety_status(self, helmets: List, vests: List, fires: List) -> str:
        """Genel safety status'unu belirle"""
        
        # Fire varsa DANGER
        if any(f['status'] == 'danger' for f in fires):
            return "🔥 DANGER"
        
        # Uyarı varsa CAUTION
        if any(h['status'] == 'warning' for h in helmets) or \
           any(v['status'] == 'warning' for v in vests):
            return "⚠️  CAUTION"
        
        # Her şey iyi ise SAFE
        return "✅ SAFE"
    
    def _calculate_risk_level(self, helmets: List, vests: List, fires: List) -> str:
        """Risk seviyesini hesapla"""
        
        # Fire varsa HIGH risk
        if any(f['status'] == 'danger' for f in fires):
            return "CRITICAL"
        
        # Uyarı varsa MEDIUM risk
        unsafe_count = len([h for h in helmets if h['status'] == 'warning']) + \
                      len([v for v in vests if v['status'] == 'warning'])
        
        if unsafe_count > 2:
            return "HIGH"
        elif unsafe_count > 0:
            return "MEDIUM"
        
        return "LOW"
    
    def to_json(self, helmet_results: Dict, vest_results: Dict, fire_results: Dict) -> str:
        """JSON string olarak döndür"""
        result = self.format_results(helmet_results, vest_results, fire_results)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def to_llm_prompt(self, helmet_results: Dict, vest_results: Dict, fire_results: Dict) -> str:
        """LLM için prompt text oluştur"""
        result = self.format_results(helmet_results, vest_results, fire_results)
        
        prompt = f"""
Fabrika İş Güvenliği Analizi
=============================
Zaman: {result['timestamp']}

📊 ÖZET:
--------
Toplam Tespit: {result['summary']['total_detections']}
Durum: {result['summary']['safety_status']}
Uyarı Sayısı: {result['summary']['alert_count']}

"""
        
        if result['summary']['alerts']:
            prompt += "🚨 UYARILAR:\n"
            for alert in result['summary']['alerts']:
                prompt += f"  • {alert}\n"
            prompt += "\n"
        
        prompt += f"""📈 DETAYLI ANALİZ:
--------------------
✓ Baret/Kask Giymiş: {result['analysis']['people_with_helmet']}
✗ Baret/Kask GİYMEMİŞ: {result['analysis']['people_without_helmet']}
✓ Yelek/İş Elbisesi Giymiş: {result['analysis']['people_with_vest']}
✗ Yelek/İş Elbisesi GİYMEMİŞ: {result['analysis']['people_without_vest']}
🔥 Yangın Tespit Durumu: {"EVET - KRİTİK!" if result['analysis']['fire_detected'] else "Hayır - Güvenli"}

Risk Seviyesi: {result['analysis']['risk_level']}

Bu bilgileri analiz ederek ne yapılması gerektiğine dair önerilerde bulunabilir misin?
Güvenlik sorunları varsa ne gibi iyileştirmeler yapılmalıdır?
"""
        
        return prompt

