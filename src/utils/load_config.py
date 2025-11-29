import yaml
import argparse
from pathlib import Path

def load_config(config_path: str):
    """
    Load YAML configuration file and return as a Python dict.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """
    Parse command-line arguments for train.py and others.
    Usage:
        python train.py --config src/config/params.yaml
    """
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/params.yaml",
        help="Path to params.yaml"
    )
    return parser.parse_args()
