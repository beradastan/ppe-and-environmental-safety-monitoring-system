#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Manager - Config dosyasından ajanları yükle ve yönet
Kendi modellerinizi kullanmak için models/config.json dosyasını güncelle
"""

import json
from pathlib import Path
from typing import Dict, Optional
import logging

from agents.specific_agents import HelmetAgent, VestAgent, FireAgent


class AgentManager:
    """Ajanları config dosyasından yükle ve yönet"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Config dosyasının yolu (varsayılan: models/config.json)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "models" / "config.json"
        else:
            config_path = Path(config_path)
        
        self.config_path = config_path
        self.config = None
        self.agents = {}
        self.logger = logging.getLogger(__name__)
        
        # Config'i yükle
        self.load_config()
    
    def load_config(self) -> bool:
        """Config dosyasını yükle"""
        if not self.config_path.exists():
            self.logger.warning(f"Config dosyası bulunamadı: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.logger.info(f"Config yüklendi: {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Config yükleme hatası: {e}")
            return False
    
    def get_model_config(self, model_type: str) -> Optional[Dict]:
        """Model konfigürasyonunu getir"""
        if self.config is None:
            return None
        
        models = self.config.get('models', {})
        return models.get(model_type)
    
    def create_helmet_agent(self) -> Optional[HelmetAgent]:
        """Baret ajanını oluştur"""
        config = self.get_model_config('helmet')
        
        if config is None:
            self.logger.warning("Baret modeli config'inde bulunamadı")
            return None
        
        if not config.get('enabled', False):
            self.logger.info("Baret modeli devre dışı")
            return None
        
        try:
            model_path = config.get('path', 'yolov8m.pt')
            confidence = config.get('confidence', 0.5)
            device = config.get('device', 'cpu')
            
            agent = HelmetAgent(
                model_name=model_path,
                confidence_threshold=confidence,
                device=device
            )
            self.logger.info(f"✅ Baret ajanı oluşturuldu: {model_path}")
            self.agents['helmet'] = agent
            return agent
        
        except Exception as e:
            self.logger.error(f"Baret ajanı oluşturma hatası: {e}")
            return None
    
    def create_vest_agent(self) -> Optional[VestAgent]:
        """Yelek ajanını oluştur"""
        config = self.get_model_config('vest')
        
        if config is None:
            self.logger.warning("Yelek modeli config'inde bulunamadı")
            return None
        
        if not config.get('enabled', False):
            self.logger.info("Yelek modeli devre dışı")
            return None
        
        try:
            model_path = config.get('path', 'yolov8m.pt')
            confidence = config.get('confidence', 0.5)
            device = config.get('device', 'cpu')
            
            agent = VestAgent(
                model_name=model_path,
                confidence_threshold=confidence,
                device=device
            )
            self.logger.info(f"✅ Yelek ajanı oluşturuldu: {model_path}")
            self.agents['vest'] = agent
            return agent
        
        except Exception as e:
            self.logger.error(f"Yelek ajanı oluşturma hatası: {e}")
            return None
    
    def create_fire_agent(self) -> Optional[FireAgent]:
        """Yangın ajanını oluştur"""
        config = self.get_model_config('fire')
        
        if config is None:
            self.logger.warning("Yangın modeli config'inde bulunamadı")
            return None
        
        if not config.get('enabled', False):
            self.logger.info("Yangın modeli devre dışı")
            return None
        
        try:
            model_path = config.get('path', 'yolov8m.pt')
            confidence = config.get('confidence', 0.5)
            device = config.get('device', 'cpu')
            
            agent = FireAgent(
                model_name=model_path,
                confidence_threshold=confidence,
                device=device
            )
            self.logger.info(f"✅ Yangın ajanı oluşturuldu: {model_path}")
            self.agents['fire'] = agent
            return agent
        
        except Exception as e:
            self.logger.error(f"Yangın ajanı oluşturma hatası: {e}")
            return None
    
    def create_all_agents(self) -> Dict:
        """Tüm ajanları oluştur"""
        print("\n" + "="*60)
        print("🤖 AJANLAR OLUŞTURULUYOR")
        print("="*60 + "\n")
        
        self.create_helmet_agent()
        self.create_vest_agent()
        self.create_fire_agent()
        
        print(f"\n✅ Hazırlanan ajanlar: {len(self.agents)}")
        for name in self.agents:
            print(f"   - {name}")
        
        return self.agents
    
    def get_agent(self, agent_type: str):
        """Ajanı getir"""
        return self.agents.get(agent_type)
    
    def get_all_agents(self) -> Dict:
        """Tüm ajanları getir"""
        return self.agents


# Kullanım örneği
if __name__ == "__main__":
    import cv2
    import logging
    
    # Logging ayarı
    logging.basicConfig(level=logging.INFO)
    
    # Agent Manager oluştur
    manager = AgentManager()
    
    # Tüm ajanları oluştur
    agents = manager.create_all_agents()
    
    if len(agents) == 0:
        print("\n⚠️  Hiç ajan oluşturulamadı!")
        print("Lütfen models/config.json dosyasını kontrol et")
        exit(1)
    
    # Test görüntüsü yükle
    test_image_path = Path(__file__).parent.parent / "test_image.jpg"
    if not test_image_path.exists():
        print(f"\n⚠️  Test görüntüsü bulunamadı: {test_image_path}")
        exit(1)
    
    image = cv2.imread(str(test_image_path))
    
    # Her ajanı test et
    print("\n" + "="*60)
    print("🧪 AJANLAR TEST EDİLİYOR")
    print("="*60 + "\n")
    
    for agent_type, agent in agents.items():
        print(f"\n🔍 {agent_type.upper()} test ediliyor...")
        try:
            result = agent.detect(image)
            print(f"   ✅ Deteksiyon tamamlandı")
            print(f"   - Tespit sayısı: {result.get('count', len(result.get('detections', [])))}")
        except Exception as e:
            print(f"   ❌ Hata: {e}")
    
    print("\n✅ Test tamamlandı!")

