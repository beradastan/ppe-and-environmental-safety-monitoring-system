"""
Configuration Loader - Hexmon ayarlarını yükle
"""

import json
from pathlib import Path
from typing import Dict, Any
import logging

class ConfigLoader:
    """Hexmon konfigürasyonunu yükle ve manage et"""
    
    def __init__(self, config_path: str = "config/hexmon_config.json"):
        """
        Args:
            config_path: Hexmon config dosyasının yolu
        """
        self.config_path = Path(config_path)
        self.config = {}
        self.logger = logging.getLogger("ConfigLoader")
        
        if not self.config_path.exists():
            self.logger.error(f"Config dosyası bulunamadı: {config_path}")
            return
        
        self.load()
    
    def load(self) -> bool:
        """Config dosyasını yükle"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.logger.info(f"✓ Config yüklendi: {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Config yükleme hatası: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Iç içe key'lere erişim (dot notation)
        
        Örnek:
            config.get("models.hexmon.path")
            config.get("agents.HelmetAgent.confidence_threshold")
        """
        keys = key.split(".")
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_agent_config(self, agent_name: str) -> Dict:
        """Ajan için tam konfigürasyon al"""
        return self.get(f"agents.{agent_name}", {})
    
    def get_model_path(self, model_name: str) -> str:
        """Model yolunu al"""
        return self.get(f"models.{model_name}.path", "")
    
    def get_classes(self, model_name: str = "hexmon") -> Dict:
        """Model class'larını al"""
        return self.get(f"models.{model_name}.classes", {})
    
    def get_active_classes(self) -> Dict:
        """Aktif class'ları al (helmet ve vest)"""
        return self.get("models.hexmon.active_classes", {})
    
    def get_pipeline_config(self) -> Dict:
        """Pipeline konfigürasyonunu al"""
        return self.get("pipeline", {})
    
    def get_status(self) -> Dict:
        """Sistem durumunu al"""
        return self.get("status", {})
    
    def is_agent_active(self, agent_name: str) -> bool:
        """Ajan aktif mi?"""
        agent_key = agent_name.lower().replace("agent", "")
        status_key = f"{agent_key}_agent" if agent_name != "FireAgent" else "fire_agent"
        return self.get(f"status.{status_key}", "").upper() == "ACTIVE"
    
    def print_summary(self):
        """Konfigürasyonun özetini yazdır"""
        print("\n" + "=" * 70)
        print("HEXMON SYSTEM CONFIGURATION")
        print("=" * 70)
        
        # System
        print(f"\n📋 SYSTEM:")
        print(f"  Project: {self.get('system.project_name')}")
        print(f"  Version: {self.get('system.version')}")
        print(f"  Model Type: {self.get('system.model_type')}")
        
        # Models
        print(f"\n🤖 MODELS:")
        print(f"  Hexmon: {self.get('models.hexmon.path')} ({self.get('models.hexmon.size_mb')}MB)")
        print(f"  Fire: {self.get('models.fire.path')}")
        
        # Agents
        print(f"\n🔧 AGENTS:")
        for agent in ["HelmetAgent", "VestAgent", "FireAgent"]:
            status = "✅" if self.is_agent_active(agent) else "❌"
            threshold = self.get(f"agents.{agent}.confidence_threshold", "N/A")
            print(f"  {status} {agent}: threshold={threshold}")
        
        # Pipeline
        print(f"\n⚙️  PIPELINE:")
        pipeline = self.get_pipeline_config()
        print(f"  LLM Enabled: {pipeline.get('use_llm', False)}")
        print(f"  Device: {pipeline.get('device', 'cpu')}")
        print(f"  Confidence Threshold: {pipeline.get('confidence_threshold', 0.3)}")
        
        # Performance
        perf = self.get("performance_expectations", {})
        print(f"\n📊 PERFORMANCE EXPECTATIONS:")
        print(f"  Average: {perf.get('average_processing_time_per_frame', 'N/A')}")
        print(f"  Min: {perf.get('min_processing_time', 'N/A')}")
        print(f"  Max: {perf.get('max_processing_time', 'N/A')}")
        
        print("\n" + "=" * 70)


# Global config instance
_config_instance = None

def get_config(config_path: str = "config/hexmon_config.json") -> ConfigLoader:
    """Global config instance'ı al (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test
    config = ConfigLoader()
    
    # Örnekler
    print("\n📖 CONFIGURATION EXAMPLES:")
    print(f"\nHelmet agent threshold: {config.get('agents.HelmetAgent.confidence_threshold')}")
    print(f"Hexmon model path: {config.get('models.hexmon.path')}")
    print(f"Pipeline device: {config.get('pipeline.device')}")
    print(f"Fire agent active: {config.is_agent_active('FireAgent')}")
    
    # Özet
    config.print_summary()

