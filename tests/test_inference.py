from pathlib import Path


def test_tri_branch_model_file_convention():
    path = Path("models/saved/phish360/phish360_tri_branch_cnn.pt")

    assert path.suffix == ".pt"
    assert path.parent.name == "phish360"
