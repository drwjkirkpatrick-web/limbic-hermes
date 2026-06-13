"""
limbic_hermes/storage.py
========================
Persistence helpers so the limbic state can be read by dashboards, other
processes, and the browser via a small JSON state file.
"""
import json
import os
from pathlib import Path
from typing import Dict


def get_default_state_dir() -> Path:
    """Use Hermes's home if available; otherwise fall back to ~/.limbic_hermes."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(hermes_home) / "limbic_state"


def ensure_state_dir(state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def write_state_file(state: Dict, path: Path) -> None:
    """Atomic-ish write: write temp then rename."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)


def read_state_file(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def default_state_path(profile: str = "default") -> Path:
    return ensure_state_dir(get_default_state_dir()) / f"{profile}.json"
