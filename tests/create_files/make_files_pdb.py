import shutil

from anarcii import Anarcii

model = Anarcii(
    seq_type="unknown",
    batch_size=128,
    cpu=False,
    ncpu=4,
    mode="accuracy",
    verbose=False,
)

# This needs to work in unknown mode...
model.number("../data/raw_data/1kb5.pdb")
model.number("../data/raw_data/8kdm.pdb")

shutil.move(
    "../data/raw_data/1kb5_anarcii.pdb", "../data/expected_data/1kb5_anarcii.pdb"
)

shutil.move(
    "../data/raw_data/8kdm_anarcii.pdb", "../data/expected_data/8kdm_anarcii.pdb"
)
