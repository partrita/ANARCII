__all__ = ["ATOM_RECORDS", "THREE_TO_ONE", "count_repeated_sequences"]

# Constants
ATOM_RECORDS = "ATOM"
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def count_repeated_sequences(sequence, k=200):
    """
    Counts the number of repeated subsequences of length k in the given sequence.

    Parameters:
    - sequence: The sequence string to search in.
    - k: Length of the subsequence to check for repetition.

    Returns:
    - An integer representing the number of repeated subsequences.
    """
    seen = {}
    repeat_count = 0

    for i in range(len(sequence) - k + 1):
        subseq = sequence[i : i + k]
        if subseq in seen:
            if (
                seen[subseq] == 1
            ):  # Increment count only the first time we detect a repeat
                repeat_count += 1
            seen[subseq] += 1
        else:
            seen[subseq] = 1

    return repeat_count
