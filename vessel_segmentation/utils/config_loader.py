# utils/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

class ConfigValidator:
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, name: str):
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")

    @staticmethod
    def validate_positive(value: float, name: str):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def validate_config(config: Dict) -> None:
        """Validate configuration parameters"""
        try:
            # Preprocessing validation
            v = config['preprocessing']
            for spacing in v['target_spacing']:
                ConfigValidator.validate_positive(spacing, "target_spacing")
            
            # Vessel detection validation
            v = config['vessel_detection']
            ConfigValidator.validate_positive(v['scale_range']['min_vessel_size'], "min_vessel_size")
            ConfigValidator.validate_positive(v['scale_range']['max_vessel_size'], "max_vessel_size")
            if v['scale_range']['min_vessel_size'] >= v['scale_range']['max_vessel_size']:
                raise ValueError("min_vessel_size must be less than max_vessel_size")
            
            # Frangi filter validation
            f = v['frangi_filter']
            ConfigValidator.validate_positive(f['alpha'], "alpha")
            ConfigValidator.validate_positive(f['beta'], "beta")
            ConfigValidator.validate_positive(f['c'], "c")
            
            # Vessel refinement validation
            v = config['vessel_refinement']
            ConfigValidator.validate_range(v['vessel_connectivity']['angle_threshold'], 
                                        0, 180, "angle_threshold")
            ConfigValidator.validate_positive(v['vessel_connectivity']['radius_ratio_max'], 
                                           "radius_ratio_max")
            
            # Surface generation validation
            s = config['surface_generation']
            ConfigValidator.validate_positive(s['smoothing']['iterations'], "smoothing iterations")
            ConfigValidator.validate_range(s['smoothing']['relaxation_factor'], 
                                        0, 1, "relaxation_factor")
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration parameter: {str(e)}")

class ConfigLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            # Validate configuration
            ConfigValidator.validate_config(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save configuration to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise