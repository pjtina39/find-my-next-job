"""Evidence retrieval and qualification-fit scoring.

For each JD requirement we score every evidence record with:
  - skill overlap: share of the requirement's skills the evidence covers
  - text similarity: TF-IDF cosine between requirement and evidence text
We keep the top evidence (with ids and source refs) and label the
requirement matched / partial / unsupported / needs_human.

This module never reads labels. See evaluate.py for that.
"""
import math
import re
from collections import Counter
from datetime import date

METHOD_VERSION = "match-v0.1"
SKILL_WEIGHT, TEXT_WEIGHT = 0.6, 0.4
MATCHED, PARTIAL = 0.6, 0.3            # requirement thresholds
APPLY, CONSIDER = 0.6, 0.4             # overall recommendation thresholds
TYPE_WEIGHT = {"required": 1.0, "responsibility": 0.7, "preferred": 0.5, "context": 0.0}
TOP_K = 3

STOP = set("a an and the of to in for with on at by or as is are be we you our your this that "
           "will from using use across into who which their them it its have has".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class Tfidf:
    """Tiny TF-IDF + cosine. Good enough as a baseline; swap for embeddings later."""

    def __init__(self, docs: list[str]):
        self.docs = [Counter(tokens(d)) for d in docs]
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vecs = [self.vec(d) for d in self.docs]

    def vec(self, counts: Counter) -> dict:
        v = {t: c * self.idf.get(t, 0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def sims(self, query: str) -> list[float]:
        q = self.vec(Counter(tokens(query)))
        return [sum(q.get(t, 0) * x for t, x in d.items()) for d in self.vecs]


def evidence_text(ev: dict, kb: dict) -> str:
    names = {s["id"]: s["name"] for s in kb["skills"]}
    return ev["claim"] + " " + " ".join(names[s] for s in ev["skills"])


def relevant_years(kb: dict, today: date | None = None) -> float:
    """Years in roles that have ML/AI evidence. An estimate, not a claim."""
    today = today or date.today()
    ml = {s["id"] for s in kb["skills"] if s["category"] == "ml-ai"}
    roles = {ev["experience_id"] for ev in kb["evidence"] if set(ev["skills"]) & ml}
    total = 0.0
    for role in kb["experience"]:
        if role["id"] not in roles:
            continue
        y, m = map(int, role["start"].split("-")[:2]) if "-" in role["start"] else (int(role["start"]), 1)
        if role["end"] == "present":
            ey, em = today.year, today.month
        else:
            ey, em = map(int, role["end"].split("-")[:2])
        total += ((ey - y) * 12 + (em - m)) / 12
    return round(total, 1)


def score_requirement(req: dict, kb: dict, tfidf: Tfidf, years_have: float) -> dict:
    if req["is_education"]:
        return {**req, "status": "needs_human", "score": None, "evidence": [],
                "note": "Education is not modelled in the KB yet. Check by hand."}

    sims = tfidf.sims(req["text"])
    scored = []
    for ev, sim in zip(kb["evidence"], sims):
        cov = (len(set(req["skills"]) & set(ev["skills"])) / len(req["skills"])) if req["skills"] else 0.0
        s = SKILL_WEIGHT * cov + TEXT_WEIGHT * min(sim * 2.5, 1.0)
        scored.append((s, cov, sim, ev))
    scored.sort(key=lambda x: -x[0])
    top = [{"evidence_id": ev["id"], "score": round(s, 3), "skill_overlap": round(cov, 2),
            "text_sim": round(sim, 3), "source_ref": ev["source"]["ref"],
            "has_metrics": bool(ev["metrics"])}
           for s, cov, sim, ev in scored[:TOP_K] if s > 0]
    best = top[0]["score"] if top else 0.0

    note = ""
    if req["years"]:
        ratio = min(years_have / req["years"], 1.0)
        best = min(best, ratio) if top else ratio
        note = f"Needs {req['years']}+ years; KB shows about {years_have} years in ML/AI-related roles."

    status = "matched" if best >= MATCHED else "partial" if best >= PARTIAL else "unsupported"
    return {**req, "status": status, "score": round(best, 3), "evidence": top, "note": note}


def predict(job: dict, reqs: list[dict], kb: dict) -> dict:
    """Score all requirements and give an apply / consider / skip recommendation."""
    tfidf = Tfidf([evidence_text(ev, kb) for ev in kb["evidence"]])
    years_have = relevant_years(kb)
    reqs = [r for r in reqs if r["type"] != "context"]
    results = [score_requirement(r, kb, tfidf, years_have) for r in reqs]

    scored = [r for r in results if r["score"] is not None]
    wsum = sum(TYPE_WEIGHT[r["type"]] for r in scored) or 1.0
    fit = sum(TYPE_WEIGHT[r["type"]] * r["score"] for r in scored) / wsum

    skill_reqs = [r for r in scored if r["skills"]]
    year_reqs = [r for r in scored if r["years"]]
    matched = [r for r in scored if r["status"] == "matched"]
    dims = {
        "skills": avg(r["score"] for r in skill_reqs),
        "experience": avg(r["score"] for r in year_reqs),
        "achievements": avg(1.0 if r["evidence"] and r["evidence"][0]["has_metrics"] else 0.0
                            for r in matched),
        "qualifications": "needs_human" if any(r["is_education"] for r in results) else None,
        "seniority": "needs_human",
    }
    rec = "apply" if fit >= APPLY else "consider" if fit >= CONSIDER else "skip"
    counts = Counter(r["status"] for r in results)
    return {
        "job_id": job["job_id"],
        "method_version": METHOD_VERSION,
        "kb_version": kb["version"],
        "fit_score": round(fit, 3),
        "recommendation": rec,
        "dimensions": dims,
        "counts": dict(counts),
        "strengths": [r["text"] for r in sorted(matched, key=lambda r: -r["score"])[:5]],
        "gaps": [r["text"] for r in results if r["status"] in ("unsupported", "needs_human")],
        "requirements": results,
        "thresholds": {"matched": MATCHED, "partial": PARTIAL, "apply": APPLY, "consider": CONSIDER},
    }


def avg(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None
