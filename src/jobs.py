"""Job ingestion and requirement extraction.

A job is saved in data/ (store.JOBS_DIR) as:
  jd/<job_id>.txt   the original text, unchanged
  job.json          metadata for every job, keyed by job id
The job id is <company>_<role>=<uuid>, e.g.
  jp-morgan_applied-ai-ml-senior-associate=3f2a9c1e-...

Extraction is rule-based: split the JD into sections, turn each bullet
into a requirement, and split each requirement into short phrases.
No keyword lists: match.py maps phrases to KB skills with embeddings.
An LLM extractor can replace this later behind the same output.
"""
import re
import uuid

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
# lowercase words allowed inside a Title Case heading, e.g. "What you'll do"
SMALL_WORDS = {"a", "an", "and", "or", "of", "to", "the", "in", "with", "for",
               "you", "you'll", "you’ll", "we", "do", "need", "have"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def new_job_id(company: str, title: str) -> str:
    """<company>_<role>=<uuid>. Every ingested JD (manual or scraped) gets one."""
    return f"{slug(company)}_{slug(title)}={uuid.uuid4()}"


def jd_path(job_id: str):
    return store.JOBS_DIR / "jd" / f"{job_id}.txt"


def load_registry() -> dict:
    """job.json: job id -> metadata, for every ingested job."""
    p = store.JOBS_DIR / "job.json"
    return store.read_json(p) if p.exists() else {}


def ingest(job_id: str | None, jd_text: str, meta: dict) -> dict:
    """Save the raw JD and metadata. Refuses to overwrite an existing job.
    No job_id: one is made from company, title and a new uuid."""
    job_id = job_id or new_job_id(meta["company"], meta["title"])
    path = jd_path(job_id)
    if path.exists():
        raise FileExistsError(f"Job {job_id} already ingested.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jd_text)
    job = {
        "job_id": job_id,
        "company": meta.get("company"),
        "title": meta.get("title"),
        "source_url": meta.get("source_url"),
        "requisition_id": meta.get("requisition_id"),
        "location": meta.get("location"),
        "source": meta.get("source", "manual"),
        "ingested_at": store.now(),
        "jd_file": f"jd/{job_id}.txt",
    }
    registry = load_registry()
    registry[job_id] = job
    store.write_json(store.JOBS_DIR / "job.json", registry, overwrite=True)
    return job


def load_job(job_id: str) -> tuple[dict, str]:
    return load_registry()[job_id], jd_path(job_id).read_text()


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


def is_heading(text: str) -> bool:
    """A short line that names a known section and is written like a heading:
    'Required Qualifications, Capabilities, And Skills' or 'Responsibilities:'.
    'Strong analytical and communication skills' is a requirement, not a
    heading, even though it contains 'skills'."""
    if len(text) >= 70 or text.endswith(".") or not section_type(text):
        return False
    words = re.findall(r"[A-Za-z][\w'’]*", text)
    return text.endswith(":") or all(w[0].isupper() or w.lower() in SMALL_WORDS for w in words)


def extract(jd: str, kb: dict | None = None) -> list[dict]:
    """Turn JD text into a list of requirements."""
    reqs, kind = [], "context"  # lines before the first known heading are context
    for raw in jd.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_bullet = bool(re.match(r"^([-*•·▪]|\d+[.)])\s+", line))
        text = re.sub(r"^([-*•·▪]|\d+[.)])\s+", "", line).strip()
        if not is_bullet and is_heading(text):
            kind = section_type(text)
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
