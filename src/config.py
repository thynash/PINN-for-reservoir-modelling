"""
Configuration settings for the PINN tutorial system.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

# Ensure output directories exist
for directory in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    # Data processing
    "data": {
        "required_curves": ["GR", "RHOB", "NPHI", "RT"],  # Gamma ray, density, neutron porosity, resistivity
        "depth_unit": "FT",
        "min_depth_range": 100.0,  # Minimum depth range for valid wells
        "outlier_threshold": 3.0,  # Standard deviations for outlier detection
    },
    
    # Model architecture
    "model": {
        "input_dim": 4,
        "hidden_dims": [100, 100, 100],
        "output_dim": 2,
        "activation": "tanh",
        "dropout_rate": 0.0,
        "batch_normalization": False,
    },
    
    # Training parameters
    "training": {
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "num_epochs": 5000,
        "optimizer_switch_epoch": 3000,
        "lbfgs_iterations": 1000,
        "validation_split": 0.2,
        "validation_frequency": 100,
        "early_stopping_patience": 500,
    },
    
    # Physics parameters
    "physics": {
        "enable_physics_loss": True,
        "pde_weight": 1.0,
        "boundary_weight": 1.0,
        "data_weight": 1.0,
        "adaptive_weights": True,
    },
    
    # Visualization
    "visualization": {
        "figure_size": (12, 8),
        "dpi": 300,
        "style": "seaborn-v0_8",
        "color_palette": "viridis",
        "save_format": "png",
    },
    
    # Logging
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
    }
}


def get_config() -> Dict[str, Any]:
    """Get the default configuration."""
    return DEFAULT_CONFIG.copy()


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values."""
    def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    return deep_update(config, updates)


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    import json
    
    config = get_config()
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                file_config = json.load(f)
            elif filepath.endswith(('.yml', '.yaml')):
                try:
                    import yaml
                    file_config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                raise ValueError("Configuration file must be JSON or YAML format")
        
        config = update_config(config, file_config)
    
    return config