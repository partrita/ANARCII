from anarcii import Anarcii

model = Anarcii(
    seq_type="antibody",
    batch_size=1,
    cpu=False,
    ncpu=4,
    mode="accuracy",
    verbose=True,
)
model.number("../data/window_cwc.fa")

model.to_text("../data/window_expected_1.txt")
model.to_json("../data/window_expected_1.json")
