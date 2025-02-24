import re

import torch

from .sequences_utils import find_scfvs, pick_window, split_seq

# from anarcii.pipeline.anarcii_constants import n_jump

cwc = re.compile(r".{,40}(?=C.{5,20}W.{50,80}C).{,160}")


class SequenceProcessor:
    def __init__(self, seqs, model, window_model, verbose, scfv=False):
        self.seqs = seqs
        self.model = model
        self.window_model = window_model
        self.verbose = verbose
        self.scfv = scfv
        self.offsets = {}

    def process_sequences(self):
        # Step 1: Handle long sequences
        self._handle_long_sequences()

        # Step 2: Convert dictionary to a list of tuples
        self._convert_to_tuple_list()

        # Step 3: Sort sequences by length
        self._sort_sequences_by_length()

        # Step 4: Tokenize sequences
        return self._tokenize_sequences(), self.offsets

    def _handle_long_sequences(self):
        if self.scfv:
            long_seqs = self.seqs
            # Jump of 2 to provide a granulary probability view.
            n_jump = 2

            # Splits the seqeucne in 60 length chunks - for granularity
            split_dict = {
                key: split_seq(seq, n_jump, 90) for key, seq in long_seqs.items()
            }
            res_dict = {
                key: find_scfvs(ls, self.window_model) for key, ls in split_dict.items()
            }

            for key, values in res_dict.items():
                if len(values) > 1:
                    num_peaks = 1
                    for value in values:
                        # Extract the window but include residues before = 20
                        # Add 110 to capture the whole thing += 140
                        # This should give a total length of 160 - without skipping the
                        # beginning.
                        start_index = max(
                            (value * n_jump) - 20, 0
                        )  # Ensures start_index is at least 0
                        end_index = (value * n_jump) + 140

                        print(long_seqs[key][start_index:end_index])

                        # Slice the sequence and update the key reflect multiple chains
                        self.seqs[key + "_" + str(num_peaks)] = long_seqs[key][
                            start_index:end_index
                        ]
                        num_peaks += 1

                    # need to delete the original key
                    _ = self.seqs.pop(key)  # Deletes the key and returns its value
                    print(key, "split into multiple seqs")  # 2

                else:
                    value = values[0]
                    start_index = max(
                        (value * n_jump) - 20, 0
                    )  # Ensures start_index is at least 0
                    end_index = (value * n_jump) + 140
                    # Slice the sequence
                    self.seqs[key] = long_seqs[key][start_index:end_index]
        else:
            # larger n_jump to reduce time.
            n_jump = 3
            long_seqs = {key: seq for key, seq in self.seqs.items() if len(seq) > 200}

            if long_seqs:
                if self.verbose:
                    print(
                        f"\n {len(long_seqs)} Long sequences detected - "
                        "running in sliding window. This is slow."
                    )

                for long_key in list(long_seqs.keys()):  # Iterate over a copy of keys
                    # first try a simple regex to look for cwc
                    cwcs = list(cwc.finditer(long_seqs[long_key]))
                    cwc_matches = [m.group() for m in cwcs]

                    if len(cwc_matches) > 1:
                        cwc_winner = pick_window(cwc_matches, self.window_model)

                        # Append the offset
                        self.offsets[long_key] = cwcs[cwc_winner].start()
                        self.seqs[long_key] = cwc_matches[cwc_winner]

                        # remove from long seqs dict
                        del long_seqs[long_key]

                    elif cwc_matches:
                        self.offsets[long_key] = cwcs[0].start()
                        self.seqs[long_key] = cwc_matches[0]
                        # remove from long seqs dict
                        del long_seqs[long_key]

                # Splits the seqeucne in 90 length chunks
                split_dict = {
                    key: split_seq(seq, n_jump=n_jump) for key, seq in long_seqs.items()
                }

                res_dict = {
                    key: pick_window(ls, self.window_model)
                    for key, ls in split_dict.items()
                }

                # magic number alert...
                for key, value in res_dict.items():
                    start_index = max(
                        (value * n_jump) - 40, 0
                    )  # Ensures start_index is at least 0
                    end_index = (value * n_jump) + 160

                    self.offsets[key] = start_index

                    # Slice the sequence
                    self.seqs[key] = long_seqs[key][start_index:end_index]

                if self.verbose:
                    print("Max probability windows selected.\n")

    def _convert_to_tuple_list(self):
        """
        Enumerates the list to give each seq an index.
        This allows sequences to be sorted by length and then recombined by this index.
        """
        self.seqs = [(i, nm, seq) for i, (nm, seq) in enumerate(self.seqs.items())]

    def _sort_sequences_by_length(self):
        self.seqs = sorted(self.seqs, key=lambda x: len(x[2]))

    def _tokenize_sequences(self):
        aa = self.model.sequence_tokeniser
        tokenized_seqs = []

        for seq in self.seqs:
            bookend_seq = [aa.start] + list(seq[2]) + [aa.end]
            try:
                tokenized_seq = torch.from_numpy(aa.encode(bookend_seq))
                tokenized_seqs.append((seq[0], seq[1], tokenized_seq))
            except KeyError as e:
                print(
                    f"Sequence could not be numbered. Contains an invalid residue: {e}"
                )
                tokenized_seqs.append(
                    (seq[0], seq[1], torch.from_numpy(aa.encode(["F"])))
                )

        return tokenized_seqs
