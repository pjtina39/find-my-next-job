"""Evidence retrieval and qualification-fit scoring.

Embedding only (SBERT, see embed.py). No skill mapping, no keywords.

For each requirement and each evidence record:
    score = cosine(requirement sentence, evidence sentence)
We keep the top evidence (with ids and source refs) and label the
requirement matched / partial / unsupported / needs_human.

This module never reads labels. See evaluate.py for that.
"""
from collections import Counter
from datetime import date

from . import embed

METHOD_VERSION = "match-v0.5-embedding-only"
MATCHED, PARTIAL = 0.6, 0.3            # requirement thresholds (untuned for SBERT)
APPLY, CONSIDER = 0.6, 0.4             # overall recommendation thresholds
TYPE_WEIGHT = {"required": 1.0, "responsibility": 0.7, "preferred": 0.5, "context": 0.0}
TOP_K = 3


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


def score_requirement(req: dict, kb: dict, sims, years_have: float) -> dict:
    if req["is_education"]:
        return {**req, "status": "needs_human", "score": None, "evidence": [],
                "note": "Education is not modelled in the KB yet. Check by hand."}

    scored = sorted(((max(float(sim), 0.0), ev) for ev, sim in zip(kb["evidence"], sims)),
                    key=lambda x: -x[0])
    top = [{"evidence_id": ev["id"], "score": round(s, 3), "source_ref": ev["source"]["ref"],
            "has_metrics": bool(ev["metrics"])}
           for s, ev in scored[:TOP_K] if s > 0]
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
    years_have = relevant_years(kb)
    reqs = [dict(r) for r in reqs if r["type"] != "context"]
    ev_vecs = embed.encode([evidence_text(ev, kb) for ev in kb["evidence"]])
    req_vecs = embed.encode([r["text"] for r in reqs]) if reqs else ev_vecs[:0]
    sim_matrix = req_vecs @ ev_vecs.T   # rows = requirements, cols = evidence
    results = [score_requirement(r, kb, sim_matrix[i], years_have) for i, r in enumerate(reqs)]

    scored = [r for r in results if r["score"] is not None]
    wsum = sum(TYPE_WEIGHT[r["type"]] for r in scored) or 1.0
    fit = sum(TYPE_WEIGHT[r["type"]] * r["score"] for r in scored) / wsum

    year_reqs = [r for r in scored if r["years"]]
    matched = [r for r in scored if r["status"] == "matched"]
    dims = {
        "experience": avg(r["score"] for r in year_reqs),
        "achievements": avg(1.0 if r["evidence"] and r["evidence"][0]["has_metrics"] else 0.0
                            for r in matched),
        "qualifications": "needs_human" if any(r["is_education"] for r in results) else None,
        "seniority": "needs_human",
    }
    rec, missing_required = recommend(fit, results)
    counts = Counter(r["status"] for r in results)
    return {
        "job_id": job["job_id"],
        "method_version": METHOD_VERSION,
        "embed_model": embed.MODEL_NAME,
        "kb_version": kb["version"],
        "fit_score": round(fit, 3),
        "recommendation": rec,
        "missing_required": missing_required,
        "dimensions": dims,
        "counts": dict(counts),
        "strengths": [r["text"] for r in sorted(matched, key=lambda r: -r["score"])[:5]],
        "gaps": [r["text"] for r in results if r["status"] in ("unsupported", "needs_human")],
        "requirements": results,
        "thresholds": {"matched": MATCHED, "partial": PARTIAL, "apply": APPLY, "consider": CONSIDER},
    }


def recommend(fit: float, results: list[dict]) -> tuple[str, list[str]]:
    """apply / consider / skip from the fit score. A required item with no
    evidence caps the result at 'consider': no evidence is not proof of absence."""
    missing = [r["id"] for r in results if r["type"] == "required" and r["status"] == "unsupported"]
    rec = "apply" if fit >= APPLY else "consider" if fit >= CONSIDER else "skip"
    if rec == "apply" and missing:
        rec = "consider"
    return rec, missing


def avg(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None
