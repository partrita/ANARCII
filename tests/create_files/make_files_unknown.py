from anarcii import Anarcii

model = Anarcii(
    seq_type="unknown",
    batch_size=128,
    cpu=False,
    ncpu=4,
    mode="accuracy",
    verbose=True,
)
model.number("../data/raw_data/unknown.fa")

model.to_text("../data/expected_data/unknown_expected_1.txt")
model.to_json("../data/expected_data/unknown_expected_1.json")
