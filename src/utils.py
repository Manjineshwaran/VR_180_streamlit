import os
from pathlib import Path
import yaml
import re

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def substitute_env_vars(text):
    """Substitute environment variables in text like ${VAR:default}"""
    if isinstance(text, str):
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        return re.sub(r'\$\{([^:}]+):?([^}]*)\}', replace_var, text)
    return text

def load_config(path="config.yaml"):
    # Try to open the config file in the current directory
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # If not found, try the parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(parent_dir, path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    # Substitute environment variables in the config
    def process_config(obj):
        if isinstance(obj, dict):
            return {k: process_config(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_config(item) for item in obj]
        else:
            return substitute_env_vars(obj)
    
    return process_config(config)

def sorted_files(folder, exts=(".png", ".jpg", ".jpeg")):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
