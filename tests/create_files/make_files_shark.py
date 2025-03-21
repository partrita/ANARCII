from anarcii import Anarcii

# FILE GENERATION
model = Anarcii(
    seq_type="shark",
    batch_size=64,
    cpu=False,
    ncpu=12,
    mode="accuracy",
    verbose=False,
)
model.number("../data/raw_data/shark_check.fa")

model.to_text("../data/expected_data/shark_expected_1.txt")
model.to_json("../data/expected_data/shark_expected_1.json")
