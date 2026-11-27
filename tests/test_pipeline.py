"""Tests run on a synthetic KB and JD only. No real career data here."""
import importlib
import json
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"
JOB = "test-job"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FMNJ_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FMNJ_KB_DIR", str(FIX / "kb"))
    monkeypatch.setenv("FMNJ_JOBS_DIR", str(tmp_path / "jobs"))
    from src import store
    importlib.reload(store)
    from src import cli
    return cli, tmp_path


def run(cli, *args):
    cli.main(list(args))


def ingest(cli):
    run(cli, "ingest", JOB, "--jd", str(FIX / "jd.txt"), "--company", "Test Co", "--title", "Test Role")


def test_extract_sections_and_skills(env):
    from src import jobs, kb
    reqs = jobs.extract((FIX / "jd.txt").read_text(), kb.load(FIX / "kb"))
    types = {r["type"] for r in reqs}
    assert {"responsibility", "required", "preferred"} <= types
    assert not any("fictional" in r["text"] for r in reqs), "About Us must be ignored"
    r = next(r for r in reqs if "3+ years" in r["text"])
    assert r["years"] == 3 and "SQL" in r["phrases"]
    assert any(r["is_education"] for r in reqs)


def test_unbulleted_short_lines_are_requirements_not_headings():
    from src import jobs
    jd = ("Required Qualifications, Capabilities, And Skills\n"
          "Strong analytical, problem-solving, and communication skills\n"
          "Preferred Qualifications:\n"
          "Advanced degree in a quantitative discipline\n"
          "Familiarity with cloud-based data and analytics platforms\n")
    reqs = jobs.extract(jd)
    assert [(r["type"], r["text"].split()[0]) for r in reqs] == [
        ("required", "Strong"), ("preferred", "Advanced"), ("preferred", "Familiarity")]


def test_phrases_map_to_kb_skills_by_embedding(env):
    from src import jobs, kb, match
    k = kb.load(FIX / "kb")
    reqs = jobs.extract((FIX / "jd.txt").read_text())
    match.map_skills(reqs, k)
    r = next(r for r in reqs if "3+ years" in r["text"])
    assert "sql" in r["skills"]
    kube = next(r for r in reqs if "Kubernetes" in r["text"])
    assert kube["skills"] == [], "unknown skills must stay unmapped"


def test_ingest_without_id_makes_company_role_uuid(env):
    cli, tmp = env
    run(cli, "ingest", "--jd", str(FIX / "jd.txt"), "--company", "JP Morgan", "--title", "Applied AI & ML")
    (jd,) = (tmp / "jobs" / "jd").iterdir()
    name, uid = jd.stem.split("=")
    company, role = name.split("_")
    assert (company, role) == ("jp-morgan", "applied-ai-ml") and len(uid) == 36
    registry = json.loads((tmp / "jobs" / "job.json").read_text())
    assert registry[jd.stem]["jd_file"] == f"jd/{jd.stem}.txt"


def test_match_needs_label_first(env):
    cli, _ = env
    ingest(cli)
    with pytest.raises(SystemExit):
        run(cli, "match", JOB)


def test_model_code_never_reads_labels():
    """Extraction, matching and generation must not touch labels."""
    for name in ["jobs.py", "match.py", "generate.py", "judge.py", "jev_judge.py"]:
        src = (Path(__file__).parent.parent / "src" / name).read_text()
        assert "import evaluate" not in src and "evaluate import" not in src and '"labels"' not in src, name


def test_full_pipeline(env):
    cli, tmp = env
    ingest(cli)
    run(cli, "label", JOB, "apply", "--by", "tester", "--source", "test")
    run(cli, "match", JOB)
    data = tmp / "data"
    pred = json.loads(next((data / "predictions" / JOB).glob("*.json")).read_text())
    assert pred["recommendation"] in ("apply", "consider", "skip")
    assert "label" not in json.dumps(pred["requirements"]).lower()
    kube = next(r for r in pred["requirements"] if "Kubernetes" in r["text"])
    assert kube["status"] == "unsupported"
    cmp = json.loads(next((data / "evaluations" / JOB).glob("*.json")).read_text())
    assert cmp["label"] == "apply"

    with pytest.raises(SystemExit):  # no decision yet
        run(cli, "generate", JOB)
    run(cli, "decide", JOB, "apply", "--fit", "4", "--pref", "5")
    run(cli, "generate", JOB)
    gen_dir = next((data / "generations" / JOB).iterdir())
    meta = json.loads((gen_dir / "meta.json").read_text())
    cv = (gen_dir / "cv.md").read_text()
    assert "named client" not in cv, "sensitive evidence must be excluded by default"
    assert meta["excluded_sensitive_evidence"][0]["evidence_id"] == "test-secret-client"
    flags = meta["documents"]["cv"]["claim_flags"]
    assert not [f for f in flags if f["type"] in ("unsupported_number", "no_evidence")]

    gid = meta["generation_id"]
    ratings = "factual_accuracy=3,relevance=4,evidence_selection=4,impact=3,writing_style=3,overall=3"
    with pytest.raises(ValueError):  # factual error, no edit -> no approval
        run(cli, "review", JOB, gid, "--doc", "cv", "--ratings", ratings, "--error", "wrong date", "--approve")
    edited = tmp / "cv_edit.md"
    edited.write_text(cv.replace("Candidate", "Test Person"))
    run(cli, "review", JOB, gid, "--doc", "cv", "--ratings", ratings, "--error", "wrong date",
        "--edited", str(edited), "--approve")
    fb = json.loads(next((data / "feedback" / JOB).glob("*.json")).read_text())
    assert fb["approved"] and fb["preference_pair"]["chosen"].endswith(".edited.md")
    assert fb["kb_version"] and fb["evidence_ids"]


def test_claim_check_flags_inflated_number(env):
    from src import generate, kb
    k = kb.load(FIX / "kb")
    doc = [{"text": "Raised benchmark agreement to 99%.", "evidence_ids": ["test-llm-eval"], "kind": "claim"},
           {"text": "Led a team of 12.", "evidence_ids": [], "kind": "claim"}]
    types = [f["type"] for f in generate.check_claims(doc, k)]
    assert "unsupported_number" in types and "no_evidence" in types


def test_missing_required_caps_apply_at_consider():
    from src import match
    ok = {"id": "r01", "type": "required", "status": "matched"}
    gap = {"id": "r02", "type": "required", "status": "unsupported"}
    nice = {"id": "r03", "type": "preferred", "status": "unsupported"}
    assert match.recommend(0.8, [ok, nice]) == ("apply", [])         # preferred gap: no cap
    assert match.recommend(0.8, [ok, gap]) == ("consider", ["r02"])  # required gap: capped
    assert match.recommend(0.3, [ok, gap]) == ("skip", ["r02"])      # cap never raises a score


def test_label_after_prediction_is_rejected(env):
    from src import evaluate
    evaluate.save_label(JOB, "apply", "tester", "test")
    pred = {"job_id": JOB, "prediction_id": "p1", "recommendation": "apply",
            "created_at": "2000-01-01T00:00:00Z"}
    with pytest.raises(ValueError):
        evaluate.compare(pred)


def test_judge_reads_p_yes_from_logprobs():
    from src import judge
    import math
    fake = {"choices": [{"message": {"content": "Yes"}, "logprobs": {"content": [{"top_logprobs": [
        {"token": "Yes", "logprob": math.log(0.6)}, {"token": " yes", "logprob": math.log(0.1)},
        {"token": "No", "logprob": math.log(0.3)}]}]}}]}
    sent = {}
    r = judge.judge("JD text", "CV text", "some/model", call=lambda body: sent.update(body) or fake)
    assert r["p_yes"] == 0.7
    assert "label" not in sent["messages"][0]["content"].lower()
    no_lp = {"choices": [{"message": {"content": "Yes"}}]}
    assert judge.judge("JD", "CV", "m", call=lambda b: no_lp)["p_yes"] is None


class FakeJev:
    """Stands in for OpenRouterJevClassifier: records the request, returns a fixed answer."""
    def __init__(self, *args, **kwargs):
        self.sent = None

    def invoke(self, request):
        from langchain_typesafe.types import ClassifierResponse, NoulAnswer
        self.sent = request
        return ClassifierResponse(
            model="~typesafe/jev-latest",
            answers={"shortlist": NoulAnswer(type="noul", noul=0.73)},
            usage={"input_tokens": 900, "output_tokens": 1, "total_tokens": 901},
        )


def test_jev_reads_probability():
    from src import jev_judge
    fake = FakeJev()
    r = jev_judge.judge("JD text", "CV text", classifier=fake)
    assert r["probability"] == 0.73 and r["input_tokens"] == 900
    assert fake.sent["state"] == {"job_description": "JD text", "cv": "CV text"}
    question = fake.sent["questions"]["shortlist"]
    assert "shortlisted" in question.instructions and question.criteria.true


def test_match_with_jev_compares_with_label(env, monkeypatch):
    from src import jev_judge
    monkeypatch.setattr(jev_judge, "OpenRouterJevClassifier", FakeJev)
    cli, tmp = env
    ingest(cli)
    run(cli, "label", JOB, "apply", "--by", "tester", "--source", "test")
    run(cli, "match", JOB, "--jev")
    data = tmp / "data"
    pred = json.loads(next((data / "predictions" / JOB).glob("*.json")).read_text())
    assert pred["jev"]["probability"] == 0.73
    cmp = json.loads(next((data / "evaluations" / JOB).glob("*.json")).read_text())
    assert cmp["jev_probability"] == 0.73 and cmp["jev_agree"] is True
