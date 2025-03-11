import shutil

import pytest

from anarcii import Anarcii


@pytest.fixture(scope="session")
def anarcii_model():
    model = Anarcii(
        seq_type="unknown",
        batch_size=1,
        cpu=False,
        ncpu=16,
        mode="accuracy",
        verbose=True,
    )
    model.number(str(tmp_path / "1kb5.pdb"))
    model.number(str(tmp_path / "8kdm.pdb"))
    
    return tmp_path


@pytest.mark.parametrize("filename", ["1kb5_anarcii.pdb", "8kdm_anarcii.pdb"])
def test_files_are_identical(anarcii_model, filename):
    """Generate and check both text and json files."""
    expected_file = f"data/expected_data/{filename}"
    test_file = anarcii_model / filename
    
    with open(expected_file) as f1, open(test_file) as f2:
        content1 = f1.read().strip()
        content2 = f2.read().strip()

    assert content1 == content2, (
        f"Files {expected_file} and {test_file} are different!"
    )
