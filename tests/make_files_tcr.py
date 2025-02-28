from anarcii import Anarcii

# FILE GENERATION
model = Anarcii(
    seq_type="tcr",
    batch_size=64,
    cpu=False,
    ncpu=12,
    mode="speed",
    verbose=False,
)
model.number("data/stcrdab_filtered.fa")

model.to_text("data/tcr_expected_1.txt")
model.to_json("data/tcr_expected_1.json")

for scheme in [
    "kabat",
    "chothia",
    "martin",
    "imgt",
    #    "aho" # Cannot number TCRs in AHo...
]:
    out = model.to_scheme(f"{scheme}")

    model.to_json(f"data/tcr_{scheme}_expected_1.json")
    model.to_text(f"data/tcr_{scheme}_expected_1.txt")
