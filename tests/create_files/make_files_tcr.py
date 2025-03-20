from anarcii import Anarcii

# FILE GENERATION
model = Anarcii(
    seq_type="tcr",
    batch_size=64,
    cpu=False,
    ncpu=12,
    mode="accuracy",
    verbose=False,
)
model.number("../data/raw_data/tcr_check.fa")

model.to_text("../data/expected_data/tcr_expected_1.txt")
model.to_json("../data/expected_data/tcr_expected_1.json")

for scheme in [
    "kabat",
    "chothia",
    "martin",
    "imgt",
    #    "aho" # need to fix Aho TCR
]:
    out = model.to_scheme(f"{scheme}")

    model.to_json(f"../data/expected_data/tcr_{scheme}_expected_1.json")
    model.to_text(f"../data/expected_data/tcr_{scheme}_expected_1.txt")
