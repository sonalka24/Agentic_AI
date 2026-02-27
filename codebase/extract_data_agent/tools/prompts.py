import json
from pathlib import Path


JSON_DIR = Path(__file__).resolve().parent.parent / "json"


def json_file_path(filename):
    """Build a path to a JSON resource under the package json directory."""
    return JSON_DIR / filename


def load_prompts(prompts_path=None):
    """Load prompt templates from JSON and merge with defaults."""
    path = Path(prompts_path) if prompts_path else json_file_path("prompts.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise ValueError(f"Could not load prompts JSON from {path}")
