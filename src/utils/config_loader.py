"""Configuration loader utility"""
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load global settings"""
        return self.load_yaml("settings.yaml")
    
    def load_videos(self) -> Dict[str, Any]:
        """Load video configuration"""
        return self.load_yaml("videos.yaml")
    
    def load_linguistic_mappings(self) -> Dict[str, Any]:
        """Load linguistic normalization mappings"""
        return self.load_yaml("linguistic_mappings.yaml")
