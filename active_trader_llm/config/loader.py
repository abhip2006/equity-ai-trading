"""
Configuration loader for YAML/JSON files.
"""

import yaml
import json
from pathlib import Path
from typing import Union
from .config_schema import Config


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return Config(**config_dict)


def save_config(config: Config, config_path: Union[str, Path]):
    """Save configuration to YAML file"""
    config_path = Path(config_path)

    with open(config_path, 'w') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config.dict(), f, default_flow_style=False)
        elif config_path.suffix == '.json':
            json.dump(config.dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
