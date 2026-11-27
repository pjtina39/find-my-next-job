"""Read-only loader for the career knowledge base repo.

We never write to the KB. Its git commit is the KB version.
"""
import json
from pathlib import Path

from . import store


def load(kb_dir: Path | None = None) -> dict:
    kb_dir = Path(kb_dir or store.KB_DIR)
    data = kb_dir / "data"
    if not data.exists():
        raise FileNotFoundError(f"No KB at {kb_dir}. Set FMNJ_KB_DIR.")
    kb = {name: json.loads((data / f"{name}.json").read_text())
          for name in ["experience", "projects", "skills", "evidence"]}
    kb["version"] = store.git_rev(kb_dir)
    check(kb)
    return kb


def check(kb: dict) -> None:
    """Fail fast if the KB is broken: unknown skill or role ids."""
    skills = {s["id"] for s in kb["skills"]}
    roles = {e["id"] for e in kb["experience"]}
    for ev in kb["evidence"]:
        bad = set(ev["skills"]) - skills
        if bad:
            raise ValueError(f"{ev['id']}: unknown skills {bad}")
        if ev["experience_id"] and ev["experience_id"] not in roles:
            raise ValueError(f"{ev['id']}: unknown role {ev['experience_id']}")


def summary(kb: dict) -> str:
    return (f"KB {kb['version']}: {len(kb['experience'])} roles, "
            f"{len(kb['evidence'])} evidence, {len(kb['skills'])} skills")
