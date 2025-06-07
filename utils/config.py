"""
Configuration utilities for the project.
"""

import os
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def load_environment() -> Dict[str, str]:
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = [
        'API_KEY',
        'MODEL_PATH',
        'DEBUG'
    ]
    
    env_vars = {}
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
        env_vars[var] = value
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    return env_vars 