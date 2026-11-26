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
    from fmnj import store
    importlib.reload(store)
    from fmnj import cli
    return cli, tmp_path


def run(cli, *args):
    cli.main(list(args))


def ingest(cli):
    run(cli, "ingest", JOB, "--jd", str(FIX / "jd.md"), "--company", "Test Co", "--title", "Test Role")


def test_extract_sections_and_skills(env):
    from fmnj import jobs, kb
    reqs = jobs.extract((FIX / "jd.md").read_text(), kb.load(FIX / "kb"))
    types = {r["type"] for r in reqs}
    assert {"responsibility", "required", "preferred"} <= types
    assert not any("fictional" in r["text"] for r in reqs), "About Us must be ignored"
    r = next(r for r in reqs if "3+ years" in r["text"])
    assert r["years"] == 3 and {"python", "sql"} <= set(r["skills"])
    assert any(r["is_education"] for r in reqs)


def test_match_needs_label_first(env):
    cli, _ = env
    ingest(cli)
    with pytest.raises(SystemExit):
        run(cli, "match", JOB)


def test_model_code_never_reads_labels():
    """Extraction, matching and generation must not touch labels."""
    for name in ["jobs.py", "match.py", "generate.py"]:
        src = (Path(__file__).parent.parent / "fmnj" / name).read_text()
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
    from fmnj import generate, kb
    k = kb.load(FIX / "kb")
    doc = [{"text": "Raised benchmark agreement to 99%.", "evidence_ids": ["test-llm-eval"], "kind": "claim"},
           {"text": "Led a team of 12.", "evidence_ids": [], "kind": "claim"}]
    types = [f["type"] for f in generate.check_claims(doc, k)]
    assert "unsupported_number" in types and "no_evidence" in types


def test_label_after_prediction_is_rejected(env):
    from fmnj import evaluate
    evaluate.save_label(JOB, "apply", "tester", "test")
    pred = {"job_id": JOB, "prediction_id": "p1", "recommendation": "apply",
            "created_at": "2000-01-01T00:00:00Z"}
    with pytest.raises(ValueError):
        evaluate.compare(pred)
