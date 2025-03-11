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
    model.number("data/raw_data/1kb5.pdb")
    model.number("data/raw_data/8kdm.pdb")


def test_files_are_identical(anarcii_model, tmp_path):
    shutil.move("data/raw_data/1kb5_anarcii.pdb", tmp_path / "1kb5_anarcii.pdb")

    shutil.move("data/raw_data/8kdm_anarcii.pdb", tmp_path / "8kdm_anarcii.pdb")

    expected_files = ["1kb5_anarcii.pdb", "8kdm_anarcii.pdb"]

    # Generate and check both text and json files
    for file in expected_files:
        test_file = tmp_path / file
        expected_file = "data/expected_data/" + file

        with open(expected_file) as f1, open(test_file) as f2:
            content1 = f1.read().strip()
            content2 = f2.read().strip()

        assert content1 == content2, (
            f"Files {expected_file} and {test_file}" + " are different!"
        )
