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
    },
    "ocr": {
        "roi": [0.65, 0.85, 0.35, 0.15],  # x, y, w, h ratios (default: bottom-right corner)
    },
    "storage": {
        "max_saved_frames": 50,
    },
}


class ConfigManager:
    """Manages persistent configuration storage."""
    
    def __init__(self, config_path: str = "./config/settings.json"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        """Load config from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()
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
