import yaml
from pathlib import Path


def load_config(config_path="configs/config.yaml"):
    """Reads the YAML config file and returns it as a Python dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path.resolve()}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


class Config:
    """
    Wraps the config dictionary so you can access settings like:
        cfg = Config()
        cfg.data["target_column"]   --> "Weekly_Sales"
        cfg.forecasting["horizon"]  --> 12
    """
    def __init__(self, config_path="configs/config.yaml"):
        self._raw = load_config(config_path)
        for key, value in self._raw.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Config(project={self._raw['project']['name']})"