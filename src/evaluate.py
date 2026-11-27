"""Labels and the prediction-vs-label comparison.

This is the only module that reads label values. Labels are saved before
any prediction, and the comparison checks that order.

How 'consider' is handled:
  strict:  agree only if prediction == label
  lenient: 'consider' counts as agreeing with 'apply' (not with 'skip')
We report both. One job is one case, not an accuracy number.
"""
from . import store


def save_label(job_id: str, value: str, decided_by: str, source: str) -> dict:
    assert value in ("apply", "skip"), "label must be apply or skip"
    p = store.path("labels", f"{job_id}.json")
    record = {
        "label_id": f"lbl-{job_id}-001",
        "job_id": job_id,
        "label_type": "apply_decision",
        "value": value,
        "recorded_at": store.now(),
        "provenance": {"source": source, "decided_by": decided_by,
                       "established_before_inference": True},
        "semantics": "Personal application preference. Not proof of fit or hiring success.",
        "model_input_policy": "excluded",
    }
    return store.write_json(p, record) and record


def label_exists(job_id: str) -> bool:
    """Used by the pipeline to enforce 'label first'. Does not read the value."""
    return store.path("labels", f"{job_id}.json").exists()


def compare(prediction: dict) -> dict:
    label = store.read_json(store.path("labels", f"{prediction['job_id']}.json"))
    if label["recorded_at"] > prediction["created_at"]:
        raise ValueError("Label was recorded after the prediction. Not a valid comparison.")
    pred, truth = prediction["recommendation"], label["value"]
    lenient = pred == truth or (pred == "consider" and truth == "apply")
    j = prediction.get("llm_judge") or {}
    judge_agree = None if j.get("p_yes") is None else (j["p_yes"] >= 0.5) == (truth == "apply")
    jev = prediction.get("jev") or {}
    jev_agree = None if jev.get("probability") is None else (jev["probability"] >= 0.5) == (truth == "apply")
    return {
        "job_id": prediction["job_id"],
        "judge_p_yes": j.get("p_yes"),
        "judge_agree": judge_agree,   # p_yes >= 0.5 counts as apply
        "jev_probability": jev.get("probability"),
        "jev_agree": jev_agree,       # probability >= 0.5 counts as apply
        "prediction_id": prediction["prediction_id"],
        "label_id": label["label_id"],
        "predicted": pred,
        "label": truth,
        "agree_strict": pred == truth,
        "agree_lenient": lenient,
        "label_recorded_before_prediction": True,
        "compared_at": store.now(),
    }
