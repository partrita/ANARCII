import os
import time

from anarcii import Anarcii


def test_files_are_identical(expected, test):
    with open(expected) as f1, open(test) as f2:
        content1 = f1.read()
        content2 = f2.read()

    assert content1 == content2, f"Files {expected} and {test} are different!"

    if os.path.exists(test):
        os.remove(test)


begin = time.time()
model = Anarcii(
    seq_type="antibody", batch_size=64, cpu=False, ncpu=12, mode="speed", verbose=False
)

results = model.number("data/sabdab_filtered.fa")

# Original test result.
# model.to_text("data/antibody_expected_1.txt")
# model.to_json("data/antibody_expected_1.json")

txt_file = "data/antibody_test_1.txt"
json_file = "data/antibody_test_1.json"

model.to_text(txt_file)
model.to_json(json_file)

end = time.time()
runtime = round((end - begin) / 60, 2)
print("Runtime: ", runtime)

# Run the tests
test_files_are_identical("data/antibody_expected_1.txt", txt_file)
test_files_are_identical("data/antibody_expected_1.json", json_file)

print("Antibody test has passed.")
