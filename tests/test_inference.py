from pathlib import Path


def test_dual_branch_model_file_convention():
    assert Path("models/saved/dual_branch_cnn.pt").suffix == ".pt"
