"""Where things live, and how records are saved.

Private data (JDs, labels, predictions, CVs, feedback) goes in DATA_DIR,
which is outside git. Code never writes private data inside the repo.
"""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FMNJ_DATA_DIR", REPO_DIR / "private-data"))
KB_DIR = Path(os.environ.get("FMNJ_KB_DIR", REPO_DIR.parent / "career-knowledge-base"))


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"


def git_rev(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def path(*parts) -> Path:
    p = DATA_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(p: Path, record: dict, overwrite: bool = False) -> Path:
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} exists. Records are append-only.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return p


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def latest(folder: Path, pattern: str = "*.json") -> Path | None:
    files = sorted(folder.glob(pattern)) if folder.exists() else []
    return files[-1] if files else None
