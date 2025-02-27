from anarcii import Anarcii

# FILE GENERATION
model = Anarcii(
    seq_type="antibody",
    batch_size=64,
    cpu=False,
    ncpu=12,
    mode="speed",
    verbose=False,
)
model.number("data/sabdab_filtered.fa")

model.to_text("data/antibody_expected_1.txt")
model.to_json("data/antibody_expected_1.json")
