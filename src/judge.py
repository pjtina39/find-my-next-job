"""LLM judge: a second, independent match metric.

Send the JD and the CV to a model, ask one yes/no question, and read
P(yes) from the token log-probabilities of the first answer token.

  p_yes = P(Yes) / (P(Yes) + P(No))

Only the JD and CV go in. Never the label, decision or fit score.
Needs OPENROUTER_API_KEY and a model that returns logprobs.
"""
import json
import math
import os
import urllib.request

QUESTION = ("Based only on the CV and the job description, should this candidate be "
            "shortlisted for an interview? Answer with one word: Yes or No.")
VERSION = "judge-v0.1"


def cv_from_kb(kb: dict) -> str:
    """Plain-text CV from roles and non-sensitive evidence."""
    lines = []
    for role in kb["experience"]:
        lines.append(f"{role['title']}, {role['org']} ({role['start']} to {role['end']})")
        for ev in kb["evidence"]:
            if ev["experience_id"] == role["id"] and not ev["sensitivity"]:
                lines.append(f"- {ev['claim']}")
    for p in kb["projects"]:
        evs = [ev for ev in kb["evidence"] if ev.get("project_id") == p["id"] and not ev["sensitivity"]]
        if evs:
            lines.append(f"Project: {p['name']}")
            lines += [f"- {ev['claim']}" for ev in evs]
    return "\n".join(lines)


def call_openrouter(body: dict) -> dict:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def p_yes(top_logprobs: list[dict]) -> float | None:
    """Sum probability over Yes-like and No-like tokens, then normalise."""
    yes = no = 0.0
    for t in top_logprobs:
        word = t["token"].strip().lower()
        if word == "yes":
            yes += math.exp(t["logprob"])
        elif word == "no":
            no += math.exp(t["logprob"])
    return round(yes / (yes + no), 4) if yes + no > 0 else None


def judge(jd: str, cv: str, model: str, call=call_openrouter) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user",
                      "content": f"Job description:\n{jd}\n\nCV:\n{cv}\n\n{QUESTION}"}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
    }
    out = call(body)
    choice = out["choices"][0]
    answer = choice["message"]["content"]
    lp = (choice.get("logprobs") or {}).get("content") or []
    result = {"version": VERSION, "model": model, "question": QUESTION, "answer": answer,
              "p_yes": None, "note": ""}
    if not lp:
        result["note"] = "Model returned no logprobs. Pick a model/provider that supports them."
        return result
    result["p_yes"] = p_yes(lp[0]["top_logprobs"])
    if result["p_yes"] is None:
        result["note"] = "No Yes/No token in the top logprobs."
    return result
