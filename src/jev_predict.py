"""Jev judge: an optional match metric from TypeSafe's Jev model, via OpenRouter.

Jev gets the JD, the candidate's experience and skills as its state, and one
yes/no question: "does the candidate's experience and skill set align with
the requirements of the job description?". It returns the probability of yes.

Only the JD, experience and skills go in. Never the label, decision or fit score.
Needs OPENROUTER_API_KEY. The question is in prompts/jev_shortlist.json.
"""
import json
from pathlib import Path

from langchain_typesafe import Noul
from langchain_typesafe.types import NoulCriteria

from .jev_openrouter import OpenRouterJevClassifier
from .judge import cv_from_kb

VERSION = "jev-v0.1"
QUESTION_FILE = Path(__file__).parent / "prompts" / "jev_shortlist.json"


def load_question(path: Path = QUESTION_FILE) -> Noul:
    """Read the question (instructions + optional true/false criteria) from JSON."""
    config = json.loads(Path(path).read_text())
    criteria = config.get("criteria")
    return Noul(instructions=config["instructions"],
                criteria=NoulCriteria(**criteria) if criteria else None)


def profile_from_kb(kb: dict) -> tuple[str, str]:
    """Experience (roles + non-sensitive evidence) and skill names, as plain text."""
    return cv_from_kb(kb), ", ".join(s["name"] for s in kb["skills"])


def predict(jd: str, experience: str, skills: str, classifier=None) -> dict:
    """Ask Jev the shortlist question for one JD, experience and skills."""
    classifier = classifier or OpenRouterJevClassifier(timeout=60)
    question = load_question()
    response = classifier.invoke({
        "state": {"job_description": jd, "experience": experience, "skills": skills},
        "questions": {"shortlist": question},
    })
    usage = response.usage
    latency = getattr(response, "latency_s", None)
    return {
        "version": VERSION,
        "model": response.model,
        "question": question.instructions,
        "probability": round(response.nouls["shortlist"].noul, 4),
        "cost": getattr(response, "cost", None),
        "latency_s": round(latency, 2) if latency is not None else None,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
    }
