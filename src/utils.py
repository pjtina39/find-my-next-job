"""Shared utilities for the jev pipeline."""
import logging
from pathlib import Path
import yaml
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(logging.INFO)]
)
log = logging.getLogger("rich")


def params_show() -> dict:
    """Load params.yaml from the project root, bypassing DVC's monorepo namespacing."""
    params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)
