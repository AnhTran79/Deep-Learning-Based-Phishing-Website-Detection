from argparse import Namespace

from app.model import PhishingDetector
from src.evaluation.compare_models import _labels_from_confusion_matrix
from train_model import QUICK_EPOCHS, QUICK_MAX_ROWS, normalize_args


def test_quick_mode_sets_smoke_defaults():
    args = Namespace(quick=True, max_rows=0, epochs=None, cpu=False)

    normalized = normalize_args(args)

    assert normalized.max_rows == QUICK_MAX_ROWS
    assert normalized.epochs == QUICK_EPOCHS
    assert normalized.cpu is True


def test_quick_mode_preserves_explicit_values():
    args = Namespace(quick=True, max_rows=123, epochs=7, cpu=False)

    normalized = normalize_args(args)

    assert normalized.max_rows == 123
    assert normalized.epochs == 7
    assert normalized.cpu is True


def test_labels_from_confusion_matrix_reconstructs_support():
    labels, predictions = _labels_from_confusion_matrix("[[3, 1], [2, 4]]")

    assert labels == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert predictions == [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]


def test_model_score_loader_skips_invalid_f1(tmp_path):
    path = tmp_path / "model_comparison.csv"
    path.write_text(
        "model,f1\n"
        "good_model,0.91\n"
        "bad_model,not-a-number\n",
        encoding="utf-8",
    )

    scores = PhishingDetector.__new__(PhishingDetector)._load_model_scores(path)

    assert scores == {"good_model": 0.91}
