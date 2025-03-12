import pytest

from anarcii import Anarcii


@pytest.fixture(scope="session")
def anarcii_model(pytestconfig):
    model = Anarcii(
        seq_type="unknown",
        batch_size=64,
        cpu=False,
        ncpu=8,
        mode="accuracy",
        verbose=True,
    )

    seqs = pytestconfig.rootdir / "tests" / "data" / "raw_data" / "unknown.fa"

    # Seqs must be converted to a str fro some reason...
    model.number(str(seqs))

    return model


def test_files_are_identical(anarcii_model, tmp_path, pytestconfig):
    expected_file_templates = {
        "txt": (
            pytestconfig.rootdir / "tests" / "data/expected_data/unknown_expected_1.txt"
        ),
        "json": (
            pytestconfig.rootdir
            / "tests"
            / "data/expected_data/unknown_expected_1.json"
        ),
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
