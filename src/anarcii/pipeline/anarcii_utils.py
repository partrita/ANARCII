import gzip
import re
import sys

from .anarcii_constants import conserved


def is_tuple_list(obj):
    if all(isinstance(item, str) for item in obj):  # list of strings
        return False
    elif all(isinstance(item, tuple) for item in obj):  # list of tuples
        nms = [x[0] for x in obj]
        if len(nms) != len(set(nms)):  # Detect duplicates
            print(
                "\nError: Duplicate names found. Please ensure all names are unique.\n"
            )
            sys.exit(0)

        return True
    else:
        print("Contents of list is neither list of strings, nor list of tuples")


def count_lines_with_greater_than(file_path):
    count = 0
    if file_path.endswith(".gz"):
        open_file = gzip.open(file_path, "rt")  # Open gzipped file in text mode
    else:
        open_file = open(file_path)

    with open_file as file:
        for line in file:
            if ">" in line:
                count += 1
    return count


def split_sequence(name, sequence, verbose):
    # Check for delimiters
    if "-" in sequence or "/" in sequence or "\\" in sequence:
        if verbose:
            print(
                f"- or / found in sequence {name}, "
                "assuming this is a paired sequence - splitting into parts."
            )
        # Split the sequence on any of these delimiters
        split_parts = re.split(r"[-/\\]", sequence)
        # Create named parts
        return {f"{name}_{i + 1}": part for i, part in enumerate(split_parts)}
    else:
        # If no delimiters, return the sequence as-is
        return {name: sequence}


def read_fasta(file_path, verbose):
    sequences = []

    if file_path.endswith(".gz"):
        open_file = gzip.open(file_path, "rt")  # Open gzipped file in text mode
    else:
        open_file = open(file_path)

    with open_file as file:
        name = None
        seq = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    if "-" in seq or "/" in seq or "\\" in seq:
                        if verbose:
                            print(
                                f"- or / found in sequence {name}, "
                                "assuming this is a paired sequence - "
                                "splitting into parts."
                            )
                        split_parts = re.split(r"[-/\\]", seq)
                        for i, part in enumerate(split_parts, start=1):
                            sequences.append((f"{name}_{i}", part))
                    else:
                        sequences.append((name, seq))
                name = line[1:]  # Strip ">" from name
                seq = ""
            else:
                seq += line
        # Handle the last sequence
        if name is not None:
            if "-" in seq or "/" in seq or "\\" in seq:
                if verbose:
                    print(
                        f"-/\\ found in sequence {name}, "
                        "assuming this is a paired sequence - splitting into parts."
                    )
                split_parts = re.split(r"[-/\\]", seq)
                for i, part in enumerate(split_parts, start=1):
                    sequences.append((f"{name}_{i}", part))
            else:
                sequences.append((name, seq))

    nms = [x[0] for x in sequences]
    if len(nms) != len(set(nms)):  # Detect duplicates
        print("\nError: Duplicate names found. Please ensure all names are unique.\n")
        sys.exit(0)

    return sequences


def check_conserved(nums):
    num = 0
    for tup in nums:
        if tup in conserved:
            num += 1
    return num
