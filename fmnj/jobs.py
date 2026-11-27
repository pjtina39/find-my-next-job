"""Job ingestion and requirement extraction.

A job is saved as:
  jobs/<job_id>/jd.md      the original text, unchanged
  jobs/<job_id>/job.json   metadata + extracted requirements

Extraction is rule-based: split the JD into sections, turn each bullet
into a requirement, and tag skills using KB aliases plus a few common
JD phrases. An LLM extractor can replace this later behind the same output.
"""
import re

from . import store

# Common JD wording -> KB skill id. KB aliases are added at runtime.
EXTRA_TERMS = {
    "machine learning": "ml", "ml ": "ml", "deep learning": "ml", "pytorch": "ml",
    "tensorflow": "ml", "scikit": "ml", "statistic": "ml",
    "large language model": "llm", "generative ai": "llm", "genai": "llm", "gen ai": "llm",
    "natural language": "nlp", "nlp": "nlp",
    "retrieval": "rag", "rag": "rag", "vector": "rag", "embedding": "rag",
    "agent": "agents", "agentic": "agents",
    "prompt": "prompting",
    "evaluat": "evaluation", "benchmark": "evaluation", "metric": "evaluation",
    "experiment": "experiment-design", "a/b": "experiment-design", "hypothesis": "experiment-design",
    "python": "python", "sql": "sql",
    "aws": "cloud", "cloud": "cloud", "azure": "cloud", "gcp": "cloud", "sagemaker": "cloud",
    "pipeline": "data-pipelines", "etl": "data-pipelines", "spark": "data-pipelines",
    "git": "version-control", "ci/cd": "cicd", "deploy": "cicd", "production": "cicd",
    "stakeholder": "stakeholder-communication", "communicat": "stakeholder-communication",
    "cross-functional": "stakeholder-communication", "partner with": "stakeholder-communication",
    "collaborat": "stakeholder-communication",
    "product": "product", "classif": "classification", "entity": "entity-resolution",
    "architect": "system-design", "system design": "system-design", "scal": "system-design",
    "analy": "data-analysis", "dashboard": "data-analysis",
}

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


def skill_terms(kb: dict) -> dict:
    terms = dict(EXTRA_TERMS)
    for s in kb["skills"]:
        for a in s["aliases"] + [s["name"]]:
            terms[a.lower()] = s["id"]
    return terms


def tag_skills(text: str, terms: dict) -> list[str]:
    t = " " + text.lower() + " "
    found = []
    for term, sid in terms.items():
        # short terms must match as whole words (avoid "r" in "for")
        hit = (re.search(rf"\b{re.escape(term.strip())}\b", t) if len(term.strip()) <= 3
               else term in t)
        if hit and sid not in found:
            found.append(sid)
    return found


def section_type(heading: str) -> str | None:
    h = heading.lower()
    for pattern, kind in SECTION_TYPES:
        if re.search(pattern, h):
            return kind
    return None


def extract(jd: str, kb: dict) -> list[dict]:
    """Turn JD text into a list of requirements."""
    terms = skill_terms(kb)
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
            "skills": tag_skills(text, terms),
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
        if not r["skills"] and not r["years"] and not r["is_education"]:
            problems.append(f"{r['id']} has no skill tags: {r['text'][:60]}")
    return problems
