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

    # Alternate number schemes
    json_chothia_file = tmp_path / "antibody_chothia_test_1.json"
    anarcii_model.to_scheme("chothia")
    anarcii_model.to_json(tmp_path / json_chothia_file)

    json_imgt_file = tmp_path / "antibody_imgt_test_1.json"
    anarcii_model.to_scheme("imgt")
    anarcii_model.to_json(tmp_path / json_imgt_file)

    json_martin_file = tmp_path / "antibody_martin_test_1.json"
    anarcii_model.to_scheme("martin")
    anarcii_model.to_json(tmp_path / json_martin_file)

    json_kabat_file = tmp_path / "antibody_kabat_test_1.json"
    anarcii_model.to_scheme("kabat")
    anarcii_model.to_json(tmp_path / json_kabat_file)

    expected_files = [
        "data/antibody_expected_1.txt",
        "data/antibody_expected_1.json",
        "data/antibody_chothia_expected_1.json",
        "data/antibody_imgt_expected_1.json",
        "data/antibody_kabat_expected_1.json",
        "data/antibody_martin_expected_1.json",
    ]

    test_files = [
        txt_file,
        json_file,
        json_chothia_file,
        json_imgt_file,
        json_kabat_file,
        json_martin_file,
    ]

    for expected, test in zip(expected_files, test_files):
        with open(expected) as f1, open(test) as f2:
            content1 = f1.read().strip()
            content2 = f2.read().strip()

        assert content1 == content2, f"Files {expected} and {test} are different!"
