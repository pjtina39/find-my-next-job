"""Command line for the V0.1 pipeline.

Order: ingest -> label -> match -> decide -> generate -> review
Run `python -m fmnj <command> -h` for options.
"""
import argparse
import json
import sys
from pathlib import Path

from . import evaluate, generate, jobs, kb as kbmod, match, review, store


def cmd_ingest(a):
    meta = {"company": a.company, "title": a.title, "source_url": a.url,
            "requisition_id": a.req_id, "location": a.location}
    job = jobs.ingest(a.job_id, Path(a.jd).read_text(), meta)
    print(f"Saved job {job['job_id']} to {store.path('jobs', a.job_id, 'x').parent}")


def cmd_label(a):
    evaluate.save_label(a.job_id, a.value, a.by, a.source)
    print(f"Label saved for {a.job_id}: {a.value}")


def cmd_match(a):
    if not evaluate.label_exists(a.job_id):
        sys.exit("Save the human label first (fmnj label). Labels must exist before inference.")
    kb = kbmod.load()
    job, jd = jobs.load_job(a.job_id)
    reqs = jobs.extract(jd, kb)
    problems = jobs.validate(reqs)
    store.write_json(store.path("jobs", a.job_id, "requirements.json"),
                     {"job_id": a.job_id, "extracted_at": store.now(), "method": "rules-v0.1",
                      "requirements": reqs, "problems": problems}, overwrite=True)

    pred = match.predict(job, reqs, kb)
    pred["prediction_id"] = store.new_id("pred")
    pred["created_at"] = store.now()
    pred["code_version"] = store.git_rev(store.REPO_DIR)
    pred["inputs"] = ["jobs/<job_id>/jd.md", "career-knowledge-base"]  # labels are never inputs
    store.write_json(store.path("predictions", a.job_id, f"{pred['prediction_id']}.json"), pred)

    cmp = evaluate.compare(pred)
    store.write_json(store.path("evaluations", a.job_id, f"{pred['prediction_id']}.json"), cmp)

    print(kbmod.summary(kb))
    print(f"{len(reqs)} requirements, {len(problems)} extraction warnings")
    print(f"Fit {pred['fit_score']} -> {pred['recommendation']}  counts={pred['counts']}")
    print(f"Label {cmp['label']}: strict agree={cmp['agree_strict']}, lenient agree={cmp['agree_lenient']}")
    print(f"Prediction {pred['prediction_id']}")


def latest_prediction(job_id):
    p = store.latest(store.path("predictions", job_id, "x").parent)
    if not p:
        sys.exit("No prediction yet. Run fmnj match first.")
    return store.read_json(p)


def cmd_decide(a):
    pred = latest_prediction(a.job_id)
    rec = review.save_decision(a.job_id, pred["prediction_id"], a.value, a.reason, a.fit, a.pref)
    print(f"Decision saved: {rec['decision']} ({rec['decision_id']})")


def cmd_generate(a):
    d = review.latest_decision(a.job_id)
    if not d or d["decision"] != "apply":
        sys.exit("Documents are only generated after a human 'apply' decision (fmnj decide).")
    kb = kbmod.load()
    job, jd = jobs.load_job(a.job_id)
    meta = generate.generate(job, jd, kb, latest_prediction(a.job_id),
                             "file" if a.from_file else a.llm, a.model, a.allow_sensitive,
                             a.from_file)
    print(f"Generation {meta['generation_id']}")
    for kind, doc in meta["documents"].items():
        print(f"  {kind}: {len(doc['evidence_ids'])} evidence ids, {len(doc['claim_flags'])} flags")
        for f in doc["claim_flags"]:
            print(f"    [{f['type']}] {f['text'][:90]}")


def cmd_review(a):
    ratings = {k: int(v) for k, v in (kv.split("=") for kv in a.ratings.split(","))}
    rec = review.save_feedback(a.job_id, a.gen_id, a.doc, ratings, a.comment or "",
                               a.error or [], a.edited, a.approve)
    print(f"Feedback saved: {rec['feedback_id']} approved={rec['approved']}")


def cmd_show(a):
    for kind in ["labels", "predictions", "evaluations", "decisions", "generations", "feedback"]:
        folder = store.DATA_DIR / kind
        items = sorted(p.name for p in folder.glob(f"{a.job_id}*")) if kind == "labels" else \
            sorted(p.name for p in (folder / a.job_id).glob("*")) if (folder / a.job_id).exists() else []
        print(f"{kind}: {items or '-'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fmnj", description="Find My Next Job V0.1")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="save a JD")
    s.add_argument("job_id"); s.add_argument("--jd", required=True, help="text file with the JD")
    s.add_argument("--company", required=True); s.add_argument("--title", required=True)
    s.add_argument("--url"); s.add_argument("--req-id"); s.add_argument("--location")
    s.set_defaults(f=cmd_ingest)

    s = sub.add_parser("label", help="save the human apply/skip label BEFORE matching")
    s.add_argument("job_id"); s.add_argument("value", choices=["apply", "skip"])
    s.add_argument("--by", required=True); s.add_argument("--source", required=True)
    s.set_defaults(f=cmd_label)

    s = sub.add_parser("match", help="extract requirements, score fit, compare with label")
    s.add_argument("job_id"); s.set_defaults(f=cmd_match)

    s = sub.add_parser("decide", help="final human apply/skip after seeing the match")
    s.add_argument("job_id"); s.add_argument("value", choices=["apply", "skip"])
    s.add_argument("--reason", default=""); s.add_argument("--fit", type=int, help="1-5")
    s.add_argument("--pref", type=int, help="1-5")
    s.set_defaults(f=cmd_decide)

    s = sub.add_parser("generate", help="write CV and cover letter")
    s.add_argument("job_id"); s.add_argument("--llm", choices=["anthropic", "openrouter"],
                                             help="default: extractive, no API")
    s.add_argument("--model", help="model id, required for openrouter; a label for --from-file")
    s.add_argument("--from-file", help="JSON with cv and cover_letter lines written elsewhere")
    s.add_argument("--allow-sensitive", action="store_true")
    s.set_defaults(f=cmd_generate)

    s = sub.add_parser("review", help="rate, edit and approve a document")
    s.add_argument("job_id"); s.add_argument("gen_id")
    s.add_argument("--doc", required=True, choices=list(review.RUBRICS))
    s.add_argument("--ratings", required=True, help="e.g. factual_accuracy=5,relevance=4,...")
    s.add_argument("--comment"); s.add_argument("--error", action="append", help="a factual error")
    s.add_argument("--edited", help="your edited version (markdown file)")
    s.add_argument("--approve", action="store_true")
    s.set_defaults(f=cmd_review)

    s = sub.add_parser("show", help="list saved records for a job")
    s.add_argument("job_id"); s.set_defaults(f=cmd_show)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
