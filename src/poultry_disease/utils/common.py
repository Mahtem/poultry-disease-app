import yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
import os


# Read YAML and return ConfigBox
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("yaml file is empty")

    except Exception as e:
        raise e


# Create directories
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"Created directory at: {path}")
            