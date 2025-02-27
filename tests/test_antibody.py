import pytest

from anarcii import Anarcii


@pytest.fixture
def anarcii_model():
    model = Anarcii(
        seq_type="antibody",
        batch_size=64,
        cpu=False,
        ncpu=12,
        mode="speed",
        verbose=False,
    )
    model.number("data/sabdab_filtered.fa")

    return model


def test_files_are_identical(anarcii_model, tmp_path):
    txt_file = tmp_path / "antibody_test_1.txt"
    json_file = tmp_path / "antibody_test_1.json"

    anarcii_model.to_text(tmp_path / txt_file)
    anarcii_model.to_json(tmp_path / json_file)

    for expected, test in zip(
        ["data/antibody_expected_1.txt", "data/antibody_expected_1.json"],
        [txt_file, json_file],
    ):
        with open(expected) as f1, open(test) as f2:
            content1 = f1.read().strip()
            content2 = f2.read().strip()

        assert content1 == content2, f"Files {expected} and {test} are different!"
