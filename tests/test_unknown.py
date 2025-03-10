import pytest

from anarcii import Anarcii


@pytest.fixture(scope="session")
def anarcii_model():
    model = Anarcii(
        seq_type="unknown", batch_size=64, cpu=False, ncpu=8, mode="speed", verbose=True
    )
    model.number("data/raw_data/unknown.fa")

    return model


def test_files_are_identical(anarcii_model, tmp_path):
    expected_file_templates = {
        "txt": "data/expected_data/unknown_expected_1.txt",
        "json": "data/expected_data/unknown_expected_1.json",
    }

    # Generate and check both text and json files
    for filetype in ["txt", "json"]:
        test_file = tmp_path / f"unknown_test_1.{filetype}"
        expected_file = expected_file_templates[filetype]

        if filetype == "txt":
            anarcii_model.to_text(test_file)
        else:
            anarcii_model.to_json(test_file)

        with open(expected_file) as f1, open(test_file) as f2:
            content1 = f1.read().strip()
            content2 = f2.read().strip()

        assert content1 == content2, (
            f"Files {expected_file} and {test_file}" + " are different!"
        )
