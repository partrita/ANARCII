import numpy as np

class Tokenizer:
    def __init__(self, vocab_type="protein"):
        self.vocab_type = vocab_type
        self.pad = '<PAD>'
        self.start = '<SOS>'
        self.end = '<EOS>'
        self.skip = '<SKIP>'

        # Antibodies ==================================================
        if self.vocab_type == "protein_antibody":
            self.vocab = [
                self.pad,
                self.start,
                self.end,
                self.skip,
                *([x.upper() for x in 'aBcdefghiJklmnopqrstUvwXyZ']),
                'H', 'L', 'K'
            ]
            self.tokens = np.array(self.vocab)

        elif self.vocab_type == "number_antibody":
            self.vocab = [
                self.pad,
                self.start,
                self.end,
                self.skip,
                *([str(x) for x in range(1, 161)]),
                'X', 'H', 'L', 'K'
            ]
            self.tokens = np.array(self.vocab)

        # TCRs ======================================================
        elif self.vocab_type == "protein_tcr":
            self.vocab = [
                self.pad,
                self.start,
                self.end,
                self.skip,
                *([x.upper() for x in 'aBcdefghiJklmnopqrstUvwXyZ']),
                'A', 'B', 'G', 'D'
            ]
            self.tokens = np.array(self.vocab)

        elif self.vocab_type == "number_tcr":
            self.vocab = [
                self.pad,
                self.start,
                self.end,
                self.skip,
                *([str(x) for x in range(1, 161)]),
                'X', 'A', 'B', 'G', 'D'
            ]
            self.tokens = np.array(self.vocab)

        else:
            raise ValueError(f"Vocab type {vocab_type} not supported")

        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}

    def encode(self, ls):
        integer_encoded = np.array([self.char_to_int[char]
                                   for char in ls], dtype=np.int32)
        return integer_encoded

    # def encode(self, sequences):
    #     if isinstance(sequences, str):  # Single sequence
    #         return np.array([self.char_to_int[char] for char in sequences], dtype=np.int32)
    #     elif isinstance(sequences, list):  # Batch of sequences
    #         return [np.array([self.char_to_int[char] for char in seq], dtype=np.int32) for seq in sequences]
    #     else:
    #         raise ValueError("Input must be a string or a list of strings")
