from anarcii import Anarcii

model = Anarcii(
    seq_type="antibody",
    batch_size=1,
    cpu=False,
    ncpu=4,
    mode="accuracy",
    verbose=True,
)
model.number("../data/raw_data/window_cwc.fa")

model.to_text("../data/expected_data/window_expected_1.txt")
model.to_json("../data/expected_data/window_expected_1.json")
