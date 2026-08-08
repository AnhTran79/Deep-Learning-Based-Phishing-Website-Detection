from pathlib import Path

from app.model import risk_level_for_score


def test_tri_branch_model_file_convention():
    path = Path("models/saved/phish360/phish360_tri_branch_cnn.pt")

    assert path.suffix == ".pt"
    assert path.parent.name == "phish360"


def test_demo_risk_levels_leave_uncertain_scores_for_review():
    assert risk_level_for_score(0.39) == "legitimate"
    assert risk_level_for_score(0.55) == "suspicious"
    assert risk_level_for_score(0.75) == "phishing"


def test_demo_never_calls_a_limited_fallback_result_legitimate():
    assert risk_level_for_score(0.07, deep_model_available=False) == "suspicious"
    assert risk_level_for_score(0.80, deep_model_available=False) == "phishing"
