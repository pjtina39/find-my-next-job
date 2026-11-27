"""Job ingestion and requirement extraction.

A job is saved as:
  jobs/<job_id>/jd.md      the original text, unchanged
  jobs/<job_id>/job.json   metadata + extracted requirements

Extraction is rule-based: split the JD into sections, turn each bullet
into a requirement, and split each requirement into short phrases.
No keyword lists: match.py maps phrases to KB skills with embeddings.
An LLM extractor can replace this later behind the same output.
"""
import re

from . import store

# Order matters: first match wins. "ignore" sections are dropped,
# "context" lines are kept but not scored.
SECTION_TYPES = [
    (r"about us|about the team|about the company|equal opportunit|benefits|^#", "ignore"),
    (r"job description|overview|summary", "context"),
    (r"responsib|what you.?ll do|the role|your role", "responsibility"),
    (r"preferred|nice to have|desirable|bonus|plus", "preferred"),
    (r"required|requirement|qualifications|what you.?ll need|must have|skills|experience|about you", "required"),
]


def ingest(job_id: str, jd_text: str, meta: dict) -> dict:
    """Save the raw JD and metadata. Refuses to overwrite an existing job."""
    jd_path = store.path("jobs", job_id, "jd.md")
    if jd_path.exists():
        raise FileExistsError(f"Job {job_id} already ingested.")
    jd_path.write_text(jd_text)
    job = {
        "job_id": job_id,
        "company": meta.get("company"),
        "title": meta.get("title"),
        "source_url": meta.get("source_url"),
        "requisition_id": meta.get("requisition_id"),
        "location": meta.get("location"),
        "source": meta.get("source", "manual"),
        "ingested_at": store.now(),
        "jd_file": "jd.md",
    }
    store.write_json(store.path("jobs", job_id, "job.json"), job)
    return job


def load_job(job_id: str) -> tuple[dict, str]:
    job = store.read_json(store.path("jobs", job_id, "job.json"))
    jd = store.path("jobs", job_id, "jd.md").read_text()
    return job, jd


def phrases(text: str) -> list[str]:
    """Split a requirement into short skill phrases, e.g.
    "Proficiency in Python, SQL, and data science tooling" ->
    ["Proficiency in Python", "SQL", "data science tooling"].
    match.py embeds these and maps them to KB skills."""
    parts = re.split(r",|;|\band\b|\bor\b|\bincluding\b", text)
    return [p.strip(" .") for p in parts if len(p.strip(" .")) > 2]


def section_type(heading: str) -> str | None:
    h = heading.lower()
    for pattern, kind in SECTION_TYPES:
        if re.search(pattern, h):
            return kind
    return None


def extract(jd: str, kb: dict | None = None) -> list[dict]:
    """Turn JD text into a list of requirements."""
    reqs, kind = [], "context"  # lines before the first known heading are context
    for raw in jd.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_bullet = bool(re.match(r"^([-*•·▪]|\d+[.)])\s+", line))
        text = re.sub(r"^([-*•·▪]|\d+[.)])\s+", "", line).strip()
        # a short line without a bullet is treated as a heading
        if not is_bullet and len(text) < 70 and not text.endswith("."):
            k = section_type(text)
            if k:
                kind = k
            continue
        if kind == "ignore" or len(text) < 25:
            continue
        reqs.append({
            "id": f"r{len(reqs) + 1:02d}",
            "text": text,
            "type": kind,
            "phrases": phrases(text),
            "years": years(text),
            "is_education": bool(re.search(r"\b(degree|bachelor|master|phd|bsc|msc)\b", text.lower())),
        })
    return reqs


def years(text: str) -> int | None:
    m = re.search(r"(\d+)\s*\+?\s*(?:or more\s*)?years?", text.lower())
    return int(m.group(1)) if m else None


def validate(reqs: list[dict]) -> list[str]:
    """Return problems a human should look at. Empty list = looks fine."""
    problems = []
    reqs = [r for r in reqs if r["type"] != "context"]
    if not reqs:
        problems.append("No requirements found. Check the JD format.")
    if not any(r["type"] == "required" for r in reqs):
        problems.append("No 'required' section detected.")
    for r in reqs:
        if not r["phrases"]:
            problems.append(f"{r['id']} has no phrases: {r['text'][:60]}")
    return problems
