from anarcii import Anarcii

model = Anarcii(
    seq_type="antibody",
    batch_size=128,
    cpu=False,
    ncpu=4,
    mode="speed",
    verbose=True,
)
model.number("../data/raw_data/sabdab_filtered.fa")

model.to_text("../data/expected_data/antibody_expected_1.txt")
model.to_json("../data/expected_data/antibody_expected_1.json")

for scheme in ["kabat", "chothia", "martin", "imgt", "aho"]:
    out = model.to_scheme(f"{scheme}")

    model.to_json(f"../data/expected_data/antibody_{scheme}_expected_1.json")
    model.to_text(f"../data/expected_data/antibody_{scheme}_expected_1.txt")

model = Anarcii(
    seq_type="antibody",
    batch_size=1,
    cpu=False,
    ncpu=4,
    mode="speed",
    verbose=True,
    max_seqs_len=50,
)

model.number("../data/raw_data/100_seqs.fa")

# For 100 seqs run in normal mode - the results should be no different to running in
# batch mode if the max number of seqs was 20 before entering batch process.
model.to_text("../data/expected_data/batch_expected_1.txt")
model.to_json("../data/expected_data/batch_expected_1.json")
