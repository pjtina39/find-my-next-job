"""Human decisions, document ratings, edits and approval.

Rubrics are in docs/rubrics.md. Ratings are 1-5.
A document cannot be approved while it has known factual errors,
whatever its overall score.
"""
import hashlib
from pathlib import Path

from . import store

RUBRICS = {
    "cv": ["factual_accuracy", "relevance", "evidence_selection", "impact", "writing_style", "overall"],
    "cover_letter": ["factual_accuracy", "relevance", "personalisation", "writing_style", "overall"],
}


def save_decision(job_id: str, prediction_id: str, decision: str, reason: str,
                  fit_rating: int | None, preference_rating: int | None) -> dict:
    """Final human apply/skip after seeing the prediction. Separate from the pre-inference label."""
    assert decision in ("apply", "skip")
    rec = {
        "decision_id": store.new_id("dec"),
        "job_id": job_id,
        "prediction_id": prediction_id,
        "decision": decision,
        "reason": reason,
        "qualification_fit_rating": fit_rating,     # 1-5, human view of fit
        "personal_preference_rating": preference_rating,  # 1-5, how much she wants it
        "decided_at": store.now(),
    }
    store.write_json(store.path("decisions", job_id, f"{rec['decision_id']}.json"), rec)
    return rec


def latest_decision(job_id: str) -> dict | None:
    p = store.latest(store.path("decisions", job_id, "x").parent)
    return store.read_json(p) if p else None


def save_feedback(job_id: str, gen_id: str, doc: str, ratings: dict, comment: str,
                  factual_errors: list[str], edited_file: str | None, approve: bool) -> dict:
    assert doc in RUBRICS, f"doc must be one of {list(RUBRICS)}"
    missing = [k for k in RUBRICS[doc] if k not in ratings]
    if missing:
        raise ValueError(f"Missing ratings: {missing}")
    if any(not 1 <= v <= 5 for v in ratings.values()):
        raise ValueError("Ratings must be 1-5.")

    gen_dir = store.path("generations", job_id, gen_id, "x").parent
    meta = store.read_json(gen_dir / "meta.json")
    original = (gen_dir / f"{doc}.md").read_text()

    fb_id = store.new_id("fb")
    edited_rel = None
    if edited_file:
        edited = Path(edited_file).read_text()
        edited_rel = f"{doc}.{fb_id}.edited.md"
        (gen_dir / edited_rel).write_text(edited)

    if approve and (factual_errors or ratings["factual_accuracy"] < 5) and not edited_file:
        raise ValueError("Factual errors must be fixed before approval. Pass an edited file.")

    rec = {
        "feedback_id": fb_id,
        "job_id": job_id,
        "generation_id": gen_id,
        "document": doc,
        "ratings": ratings,
        "comment": comment,
        "factual_errors": factual_errors,
        "original_sha256": sha(original),
        "edited_file": edited_rel,
        "preference_pair": ({"chosen": edited_rel, "rejected": f"{doc}.md"} if edited_rel else None),
        "approved": approve,
        "approved_file": (edited_rel or f"{doc}.md") if approve else None,
        "evidence_ids": meta["documents"][doc]["evidence_ids"],
        "kb_version": meta["kb_version"],
        "generator": meta["generator"],
        "model": meta["model"],
        "prompt_version": meta["prompt_version"],
        "created_at": store.now(),
    }
    store.write_json(store.path("feedback", job_id, f"{fb_id}.json"), rec)
    return rec


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
