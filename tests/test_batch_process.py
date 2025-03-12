import pytest

from anarcii import Anarcii


@pytest.fixture(scope="session")
def anarcii_model(pytestconfig):
    model = Anarcii(
        seq_type="antibody",
        batch_size=1,
        cpu=False,
        ncpu=8,
        mode="speed",
        verbose=True,
        # Need to manually change the # of seqs exceeded then run on 101 test seqs in
        # batches of 20 (101 seqs should be 6 batches).
        max_seqs_len=20,
    )

    seqs = pytestconfig.rootdir / "tests" / "data" / "raw_data" / "100_seqs.fa"

    # Seqs must be converted to a str fro some reason...
    model.number(str(seqs))

    return model


def test_files_are_identical(anarcii_model, tmp_path, pytestconfig):
    expected_file_templates = {
        "txt": (
            pytestconfig.rootdir / "tests" / "data/expected_data/batch_expected_1.txt"
        ),
        "json": (
            pytestconfig.rootdir / "tests" / "data/expected_data/batch_expected_1.json"
        ),
    }

    # Generate and check both text and json files
    for filetype in ["txt", "json"]:
        test_file = tmp_path / f"batch_test_1.{filetype}"
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
