import pytest

from anarcii import Anarcii


@pytest.fixture(scope="session")
def anarcii_model(pytestconfig):
    model = Anarcii(
        seq_type="antibody",
        batch_size=64,
        cpu=False,
        ncpu=12,
        mode="speed",
        verbose=False,
    )
    seqs = pytestconfig.rootdir / "tests" / "data" / "raw_data" / "sabdab_filtered.fa"

    # Seqs must be converted to a str fro some reason...
    model.number(str(seqs))

    return model


@pytest.mark.parametrize(
    "scheme", ["default", "chothia", "imgt", "martin", "kabat", "aho"]
)
def test_files_are_identical(anarcii_model, tmp_path, scheme, pytestconfig):
    suffix = "" if scheme == "default" else f"_{scheme}"

    expected_file_templates = {
        "txt": (
            pytestconfig.rootdir
            / "tests"
            / "data"
            / "expected_data"
            / f"antibody{suffix}_expected_1.txt"
        ),
        "json": (
            pytestconfig.rootdir
            / "tests"
            / "data"
            / "expected_data"
            / f"antibody{suffix}_expected_1.json"
        ),
    }

    suffix = "" if scheme == "default" else f"_{scheme}"

    # Switch scheme if necessary (skip for "default")
    if scheme != "default":
        anarcii_model.to_scheme(scheme)

    # Generate and check both text and json files
    for filetype in ["txt", "json"]:
        test_file = tmp_path / f"antibody{suffix}_test_1.{filetype}"
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
