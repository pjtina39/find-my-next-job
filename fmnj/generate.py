"""Tailored CV and cover letter, plus a claim check.

Every generated line carries the evidence ids it relies on. The claim
check flags any line whose numbers are not in the cited evidence, any
achievement line with no evidence, and any use of sensitive evidence.

Two generators:
  extractive (default): reuses evidence claims word for word. No API needed.
  llm: Claude rewrites from the same evidence pack. Needs ANTHROPIC_API_KEY.
"""
import json
import os
import re

from . import store

GENERATOR_VERSION = "extractive-v0.1"
PROMPT_VERSION = "cv-cl-prompt-v0.1"
MODEL = "claude-opus-5-5"
MAX_BULLETS_TOP_ROLE = 5
MAX_BULLETS_OTHER = 2


def relevance(prediction: dict) -> dict:
    """Evidence id -> best score across all requirements."""
    best = {}
    for r in prediction["requirements"]:
        for e in r["evidence"]:
            best[e["evidence_id"]] = max(best.get(e["evidence_id"], 0), e["score"])
    return best


def select_evidence(kb: dict, prediction: dict, allow_sensitive: bool = False) -> tuple[list, list]:
    rel = relevance(prediction)
    chosen, excluded = [], []
    for ev in kb["evidence"]:
        if ev["sensitivity"] and not allow_sensitive:
            excluded.append({"evidence_id": ev["id"], "sensitivity": ev["sensitivity"]})
            continue
        chosen.append({**ev, "relevance": rel.get(ev["id"], 0.0)})
    chosen.sort(key=lambda e: -e["relevance"])
    return chosen, excluded


def line(text, ev_ids=(), kind="claim"):
    return {"text": text, "evidence_ids": list(ev_ids), "kind": kind}


def extractive(job: dict, kb: dict, prediction: dict, evidence: list, name: str) -> dict:
    by_role = {}
    for ev in evidence:
        by_role.setdefault(ev["experience_id"] or ev.get("project_id"), []).append(ev)
    skill_name = {s["id"]: s["name"] for s in kb["skills"]}

    # skills: ones with relevant evidence first
    skill_rank = {}
    for ev in evidence:
        for s in ev["skills"]:
            skill_rank[s] = max(skill_rank.get(s, 0), ev["relevance"])
    top_skills = [skill_name[s] for s, _ in sorted(skill_rank.items(), key=lambda x: -x[1])]

    cv = [line(f"# {name}", kind="header"),
          line(f"Tailored for: {job['title']}, {job['company']}", kind="header"),
          line("## Summary", kind="header")]
    current = kb["experience"][0]
    summary_ids = [e["id"] for e in evidence[:3]]
    cv.append(line(f"{current['title']} at {current['org']}. Recent work covers "
                   f"{', '.join(top_skills[:4]).lower()}.", summary_ids, "summary"))

    cv.append(line("## Experience", kind="header"))
    for i, role in enumerate(kb["experience"]):
        cv.append(line(f"### {role['title']}, {role['org']} ({role['start']} to {role['end']})",
                       kind="role"))
        limit = MAX_BULLETS_TOP_ROLE if i < 2 else MAX_BULLETS_OTHER
        for ev in by_role.get(role["id"], [])[:limit]:
            cv.append(line(f"- {ev['claim']}", [ev["id"]]))

    projects = [p for p in kb["projects"] if p["id"] in by_role]
    if projects:
        cv.append(line("## Projects", kind="header"))
        for p in projects:
            cv.append(line(f"### {p['name']}", kind="role"))
            for ev in by_role[p["id"]][:2]:
                cv.append(line(f"- {ev['claim']}", [ev["id"]]))

    cv.append(line("## Skills", kind="header"))
    cv.append(line(", ".join(top_skills), kind="skills"))
    cv.append(line("## Education", kind="header"))
    cv.append(line("TODO: add by hand. Education is not in the knowledge base yet.", kind="todo"))

    top3 = [e for e in evidence if e["experience_id"]][:3]
    org = {r["id"]: r["org"] for r in kb["experience"]}
    cl = [line("Dear Hiring Team,", kind="text"),
          line(f"I am applying for the {job['title']} role at {job['company']}. "
               f"I am currently {current['title']} at {current['org']}.", kind="text")]
    for ev in top3:
        claim = ev["claim"][0].lower() + ev["claim"][1:]
        cl.append(line(f"At {org[ev['experience_id']]}, I {claim}", [ev["id"]]))
    cl.append(line("TODO: add one or two sentences on why this team and role. "
                   "The knowledge base has no record of your motivation.", kind="todo"))
    cl.append(line("Thank you for your time. I would welcome the chance to talk.", kind="text"))
    cl.append(line(f"Kind regards,\n{name}", kind="text"))
    return {"cv": cv, "cover_letter": cl}


LLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cv", "cover_letter"],
    "properties": {k: {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["text", "evidence_ids", "kind"],
        "properties": {"text": {"type": "string"},
                       "evidence_ids": {"type": "array", "items": {"type": "string"}},
                       "kind": {"type": "string"}}}} for k in ["cv", "cover_letter"]},
}

LLM_SYSTEM = """You write a tailored CV and cover letter from evidence records only.
Rules:
- Every claim about the candidate must come from the evidence. Do not add skills, numbers, titles or employers that are not there.
- You may shorten or reorder a claim. Never make it stronger.
- Give each line the evidence ids it uses. Headers and plain courtesy lines use an empty list.
- Use kind: header, role, summary, claim, skills, text or todo.
- If something the JD asks for is not in the evidence, do not cover it. Add a 'todo' line instead.
- Plain, direct British English. No buzzwords."""


def llm(job: dict, jd: str, kb: dict, evidence: list, name: str) -> dict:
    import anthropic  # optional dependency

    pack = [{"id": e["id"], "role": e["experience_id"], "project": e.get("project_id"),
             "claim": e["claim"], "metrics": e["metrics"]} for e in evidence]
    roles = [{k: r[k] for k in ["id", "org", "title", "start", "end"]} for r in kb["experience"]]
    user = (f"Candidate name: {name}\n\nJob: {job['title']} at {job['company']}\n\nJD:\n{jd}\n\n"
            f"Roles:\n{json.dumps(roles)}\n\nEvidence:\n{json.dumps(pack)}")
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=MODEL, max_tokens=32000, system=LLM_SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": LLM_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Model refused. Use the extractive generator.")
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text)


NUM = re.compile(r"\d[\d,.]*\s?%?|\d+(?:\.\d+)?\s?(?:days|hrs|hours|x)")


def check_claims(doc: list, kb: dict) -> list[dict]:
    """Return flags. Empty list = every number and claim traced to evidence."""
    evs = {e["id"]: e for e in kb["evidence"]}
    flags = []
    for i, ln in enumerate(doc):
        if ln["kind"] in ("header", "role", "todo"):
            if ln["kind"] == "todo":
                flags.append({"line": i, "type": "todo", "text": ln["text"]})
            continue
        unknown = [x for x in ln["evidence_ids"] if x not in evs]
        if unknown:
            flags.append({"line": i, "type": "unknown_evidence", "ids": unknown, "text": ln["text"]})
        cited = [evs[x] for x in ln["evidence_ids"] if x in evs]
        if ln["kind"] == "claim" and not cited:
            flags.append({"line": i, "type": "no_evidence", "text": ln["text"]})
        source = " ".join(e["claim"] + " " + e["source"]["quote"] + " " +
                          " ".join(m["value"] for m in e["metrics"]) for e in cited)
        for n in NUM.findall(ln["text"]):
            n = n.strip().rstrip(".,")
            if n and n not in source:
                flags.append({"line": i, "type": "unsupported_number", "value": n, "text": ln["text"]})
        for e in cited:
            if e["sensitivity"]:
                flags.append({"line": i, "type": "sensitive", "evidence_id": e["id"],
                              "sensitivity": e["sensitivity"], "text": ln["text"]})
    return flags


def to_markdown(doc: list) -> str:
    out = []
    for ln in doc:
        if ln["kind"] in ("header", "role") and out:
            out.append("")
        out.append(ln["text"])
    return "\n".join(out) + "\n"


def generate(job: dict, jd: str, kb: dict, prediction: dict, use_llm: bool = False,
             allow_sensitive: bool = False) -> dict:
    name = os.environ.get("FMNJ_CANDIDATE_NAME", "Candidate")
    evidence, excluded = select_evidence(kb, prediction, allow_sensitive)
    docs = llm(job, jd, kb, evidence, name) if use_llm else extractive(job, kb, prediction, evidence, name)

    gen_id = store.new_id("gen")
    folder = store.path("generations", job["job_id"], gen_id, "x").parent
    meta = {
        "generation_id": gen_id,
        "job_id": job["job_id"],
        "prediction_id": prediction["prediction_id"],
        "kb_version": kb["version"],
        "code_version": store.git_rev(store.REPO_DIR),
        "generator": "llm" if use_llm else GENERATOR_VERSION,
        "model": MODEL if use_llm else None,
        "prompt_version": PROMPT_VERSION if use_llm else None,
        "created_at": store.now(),
        "excluded_sensitive_evidence": excluded,
        "documents": {},
    }
    for kind, doc in docs.items():
        (folder / f"{kind}.md").write_text(to_markdown(doc))
        meta["documents"][kind] = {
            "file": f"{kind}.md",
            "lines": doc,
            "evidence_ids": sorted({x for ln in doc for x in ln["evidence_ids"]}),
            "claim_flags": check_claims(doc, kb),
        }
    store.write_json(folder / "meta.json", meta)
    return meta
