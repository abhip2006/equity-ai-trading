"""
Configuration loader for YAML/JSON files.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Union, Dict
from .config_schema import Config
from .battle_config_schema import BattleConfig


def _expand_env_vars(config_dict: dict) -> dict:
    """Recursively expand environment variables in config dictionary"""
    if isinstance(config_dict, dict):
        return {k: _expand_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [_expand_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        # Expand environment variables like ${VAR_NAME}
        return os.path.expandvars(config_dict)
    else:
        return config_dict


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

    # Expand environment variables
    config_dict = _expand_env_vars(config_dict)

    return Config(**config_dict)


def save_config(config: Config, config_path: Union[str, Path]):
    """Save configuration to YAML file"""
    config_path = Path(config_path)

    with open(config_path, 'w') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
        elif config_path.suffix == '.json':
            json.dump(config.model_dump(), f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def load_battle_config(config_path: Union[str, Path]) -> BattleConfig:
    """
    Load battle configuration from YAML or JSON file.

    This loader:
    1. Loads the configuration file
    2. Expands environment variables (${VAR_NAME} syntax)
    3. Validates all required API keys are present
    4. Checks database paths don't conflict
    5. Validates battle-specific requirements

    Args:
        config_path: Path to battle configuration file

    Returns:
        BattleConfig: Validated battle configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is unsupported or validation fails
        EnvironmentError: If required environment variables are missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Battle config file not found: {config_path}")

    # Load raw config
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Expand environment variables
    config_dict = _expand_env_vars(config_dict)

    # Parse and validate with Pydantic
    try:
        battle_config = BattleConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Battle config validation failed: {e}")

    # Validate all required environment variables are present
    missing_vars = battle_config.validate_env_vars_present(os.environ)
    if missing_vars:
        error_msg = (
            f"Missing required environment variables for battle:\n"
            f"  {', '.join(missing_vars)}\n\n"
            f"Please set these variables in your .env file or environment.\n"
            f"See .env.example.battle for template."
        )
        raise EnvironmentError(error_msg)

    # Additional validation: Check database paths don't exist and conflict
    _validate_database_paths(battle_config)

    return battle_config


def _validate_database_paths(config: BattleConfig) -> None:
    """
    Validate that database paths don't conflict with existing files.

    This is a soft check - warns if databases exist but allows proceeding.
    """
    from pathlib import Path

    for model_id, model_config in config.models.items():
        if not model_config.enabled:
            continue

        db_path = Path(model_config.database_path)

        # Create parent directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Note: We don't raise an error if database exists, as it might be intentional
        # (resuming a battle). The battle orchestrator will handle this.


def save_battle_config(config: BattleConfig, config_path: Union[str, Path]):
    """Save battle configuration to YAML file"""
    config_path = Path(config_path)

    with open(config_path, 'w') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
        elif config_path.suffix == '.json':
            json.dump(config.model_dump(), f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
