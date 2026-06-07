"""
Configuration Manager - Persistent settings storage
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "video_source": {
        "type": "file",  # "file" or "rtsp"
        "path": "./data/videos/sample.mp4",
    },
    "vlm": {
        "api_url": "https://api.openai.com/v1",
        "api_key": "",
        "model_name": "gpt-4o",
        "system_prompt": "",  # Custom system prompt (overrides default if set)
        "presets": {
            "OpenAI": {
                "api_url": "https://api.openai.com/v1",
                "api_key": "",
                "model_name": "gpt-4o",
            },
            "Local (Ollama)": {
                "api_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model_name": "qwen2-vl",
            },
            "SiliconFlow": {
                "api_url": "https://api.siliconflow.cn/v1",
                "api_key": "",
                "model_name": "Qwen/Qwen2-VL-72B-Instruct",
            }
        },  # Stored API presets
    },
    "embedding": {
        "provider": "api",  # "api" or "local"
        # API mode settings
        "api_url": "https://api.openai.com/v1",
        "api_key": "",
        "model_name": "text-embedding-3-small",
        # Local mode settings
        "local_model_path": "",  # Absolute path to local model (e.g., D:/models/Qwen3-VL-Embedding)
    },
    "analysis": {
        "interval": 1.0,
        "motion_threshold": 5.0,
    },
    "ocr": {
        "roi": [0.65, 0.85, 0.35, 0.15],  # x, y, w, h ratios (default: bottom-right corner)
    },
    "storage": {
    "max_saved_frames": 50,
    },
    "alarms": {
        "rules": {
            # Default Object Detection Alarms
            "PERSON": {
                "enabled": True, 
                "description": "检测到人", 
                "severity": "Info", 
                "icon": "👤",
                "keywords": "人, 男, 女, 孩, 老, person, man, woman, child"
            },
            "DOG": {
                "enabled": True, 
                "description": "检测到狗", 
                "severity": "Info", 
                "icon": "🐕",
                "keywords": "狗, 犬, dog, puppy"
            },
            "CAT": {
                "enabled": True, 
                "description": "检测到猫", 
                "severity": "Info", 
                "icon": "🐱",
                "keywords": "猫, cat, kitten"
            },
            "CAR": {
                "enabled": True, 
                "description": "检测到车辆", 
                "severity": "Info", 
                "icon": "🚗",
                "keywords": "车, SUV, 轿车, 卡车, car, truck, vehicle, van, bus, SUV"
            },
            "CHICKEN": {
                "enabled": True, 
                "description": "检测到鸡", 
                "severity": "Info", 
                "icon": "🐔",
                "keywords": "鸡, chicken, rooster, hen"
            },
            # Critical Safety Alarms
            "FALL": {
                "enabled": True, 
                "description": "检测到跌倒", 
                "severity": "Critical", 
                "icon": "⚠️",
                "keywords": "摔倒, 倒地, 跌倒, fall, collapse, ground"
            },
            "FIRE": {
                "enabled": True, 
                "description": "检测到火灾", 
                "severity": "Critical", 
                "icon": "🔥",
                "keywords": "火, 烟, fire, smoke, flame"
            },
            "VIOLENCE": {
                "enabled": True, 
                "description": "检测到暴力", 
                "severity": "High", 
                "icon": "👊",
                "keywords": "打架, 殴打, 暴力, fight, violence, hit, punch"
            },
            "INTRUSION": {
                "enabled": True, 
                "description": "检测到闯入", 
                "severity": "High", 
                "icon": "🚫",
                "keywords": "闯入, 入侵, 徘徊, intrusion, trespass, loiter"
            },
        }
    }
}


import copy

class ConfigManager:
    """Manages persistent configuration storage."""
    
    def __init__(self, config_path: str = "./config/settings.json"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self._load()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _load(self):
        """Load config from file and merge into defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                
                # Merge file config into defaults
                self._deep_update(self._config, file_config)
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                # Reset to defaults on corruption? Or just keep defaults
                self._config = copy.deepcopy(DEFAULT_CONFIG)
        else:
            self._save()
            logger.info(f"Created default config at {self.config_path}")
    
    def _save(self):
        """Save config to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation (e.g., 'vlm.api_url')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set a config value using dot notation."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self):
        """Explicitly save config."""
        self._save()
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire config dict."""
        return self._config.copy()
    
    def update_from_dict(self, data: Dict[str, Any]):
        """Update config from a flat dict."""
        for key, value in data.items():
            self.set(key, value)
        self._save()


# Global singleton
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
