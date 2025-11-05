"""
Configuration module for ActiveTrader-LLM.

Provides:
- Standard single-agent configuration (Config)
- Battle configuration for multi-model competitions (BattleConfig)
- Loaders with environment variable expansion and validation
"""

from .config_schema import Config
from .battle_config_schema import BattleConfig
from .loader import load_config, save_config, load_battle_config, save_battle_config

__all__ = [
    'Config',
    'BattleConfig',
    'load_config',
    'save_config',
    'load_battle_config',
    'save_battle_config',
]
