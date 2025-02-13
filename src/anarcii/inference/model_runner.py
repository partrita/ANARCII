import torch

from ..input_data_processing.tokeniser import Tokenizer
from .inference_utils import build_inward_list, dataloader, format_output
from .model_loader import Loader

# NEED TO COME BACK TO THIS CODE AND LOOK AT THE TRY EXCEPT LOOPS....
# SOMETHING SHOULD BE MODIFIED TO REDUCE THEM....


class ModelRunner:
    def __init__(self, sequence_type, mode, batch_size, device, verbose):
        self.type = sequence_type.lower()
        self.mode = mode.lower()
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        if self.type in ["antibody", "shark"]:
            self.sequence_tokeniser = Tokenizer("protein_antibody")
            self.number_tokeniser = Tokenizer("number_antibody")

        elif self.type == "tcr":
            self.sequence_tokeniser = Tokenizer("protein_tcr")
            self.number_tokeniser = Tokenizer("number_tcr")
        else:
            raise ValueError(f"Invalid model type: {self.type}")

        # Initialise the tokens here and place on the device
        self.pad_token = (
            torch.tensor(self.number_tokeniser.encode(["<PAD>"]))
            .unsqueeze(0)
            .to(self.device)
        )
        self.sos_token = (
            torch.tensor(self.number_tokeniser.encode(["<SOS>"]))
            .unsqueeze(0)
            .to(self.device)
        )
        self.eos_token = (
            torch.tensor(self.number_tokeniser.encode(["<EOS>"]))
            .unsqueeze(0)
            .to(self.device)
        )
        self.skip_token = (
            torch.tensor(self.number_tokeniser.encode(["<SKIP>"]))
            .unsqueeze(0)
            .to(self.device)
        )
        self.x_token = (
            torch.tensor(self.number_tokeniser.encode(["X"]))
            .unsqueeze(0)
            .to(self.device)
        )

        self.model = self._load_model()

    def _load_model(self):
        model_loader = Loader(self.type, self.mode, self.device)
        return model_loader.model

    def __call__(self, list_of_tuples):
        # Put into dataloader, make predictions, format the output.
        # NB: Provide a list of recommended batch sizes based on RAM and architecture

        seqs_only = [t[2] for t in list_of_tuples]  # tokenised
        names_only = [t[1] for t in list_of_tuples]
        indices = [t[0] for t in list_of_tuples]

        dl = dataloader(self.batch_size, seqs_only)
        numbering, alignment = self._predict_numbering(dl)
        numbered_output = format_output(indices, names_only, numbering, alignment)
        return numbered_output

    def _predict_numbering(self, dl):
        if self.verbose:
            print(f"Making predictions on {len(dl)} batches.")

        numbering = []
        alignment = []

        num = 0
        batch = 1

        pad_token = self.pad_token
        sos_token = self.sos_token
        eos_token = self.eos_token
        skip_token = self.skip_token
        x_token = self.x_token

        with torch.no_grad():
            for X in dl:
                batch += 1
                src = X.to(self.device)
                batch_size = src.shape[0]
                trg_len = src.shape[1] + 1  # Need to add 1 to include chain ID

                src_mask = self.model.make_src_mask(src)
                enc_src = self.model.encoder(src, src_mask)
                input = src[:, 0].unsqueeze(1)

                max_input = torch.zeros(
                    batch_size, trg_len, device=self.device, dtype=torch.long
                )
                max_input[:, 0] = src[:, 0]

                for t in range(1, trg_len):
                    trg_pad_mask, trg_causal_mask = self.model.make_trg_mask(input)
                    output = self.model.decoder(
                        input, enc_src, trg_pad_mask, trg_causal_mask, src_mask
                    )

                    pred_token = output.argmax(2)[:, -1].unsqueeze(1)
                    max_input[:, t : t + 1] = pred_token
                    input = max_input[:, : t + 1]

                # tokenise and transfer the batch to cpu
                src_tokens = self.sequence_tokeniser.tokens[src.to("cpu")]
                pred_tokens = self.number_tokeniser.tokens[max_input.to("cpu")]

                scores = output.topk(1, dim=2).values[:, :trg_len]
                scores = scores.squeeze(
                    -1
                )  # Remove the last dimension; shape becomes [batch_size, trg_len]

                mask = (
                    (max_input != skip_token)
                    & (max_input != x_token)
                    & (max_input != pad_token)
                    & (max_input != sos_token)
                )

                # Find the first occurrence of `True` (eos_token) along the last
                # dimension (trg_len)
                eos_positions = max_input == eos_token
                # Get the indices of the first occurrence of True along dimension 1
                # (trg_len), for each batch
                first_eos_positions = torch.argmax(eos_positions.to(torch.int64), dim=1)
                # Check if no EOS token is found for each batch
                no_eos_found = ~(
                    eos_positions.any(dim=1)
                )  # True if no EOS token is found in the row
                # Set the position to trg_len if no EOS is found
                first_eos_positions[no_eos_found] = torch.tensor(
                    trg_len - 1, device=self.device
                )

                # New code plan: Preallocte to numpy array.
                # Place at designated positions in the numpy array >>> Process an entire
                # output string, read from a text file.
                for batch_no in range(batch_size):
                    num += 1
                    error_msg = None

                    # # Ensure the score is calculated for numbered positions only.
                    # eos_positions = (max_input[batch_no] == eos_token).nonzero()
                    # eos_position = (
                    #     eos_positions[0, 1]
                    #     if eos_positions.size(0) > 0
                    #     else trg_len - 1
                    # )

                    # mask = (max_input[batch_no, :eos_position] != skip_token) & \
                    #     (max_input[batch_no, :eos_position] != x_token) & \
                    #     (max_input[batch_no, :eos_position] != pad_token) & \
                    #     (max_input[batch_no, :eos_position] != sos_token)

                    # mask = mask.squeeze(0)  # Ensure mask has shape [eos_position]
                    # print(mask[batch_no, :eos_position].shape)

                    eos_position = first_eos_positions[batch_no]

                    valid_indices = torch.arange(eos_position, device=self.device)[
                        mask[batch_no, :eos_position]
                    ]
                    valid_scores = scores[batch_no, valid_indices]

                    if len(valid_indices) >= 50:
                        normalized_score = valid_scores.mean().item()
                    else:
                        normalized_score = 0.0
                        error_msg = "Less than 50 non insertion residues numbered."

                    # This is the antibody cutoff - need a new one for TCRS
                    if round(normalized_score, 3) < 12.0:
                        numbering.append(None)

                        alignment.append(
                            {
                                "chain_type": "F",
                                "score": round(normalized_score, 3),
                                "query_start": None,
                                "query_end": None,
                                "error": error_msg or "Score less than cut off.",
                            }
                        )

                    else:
                        seqs, nums = [], []
                        backfill_seqs = []
                        started = False
                        in_x_run, x_count = False, 0
                        start_index = None
                        end_index = None

                        eos_position = eos_position.item()

                        try:
                            for seq_position in range(2, eos_position):
                                if src_tokens[batch_no, seq_position - 1] == "<EOS>":
                                    end_index = seq_position - 3
                                    break

                                elif (
                                    pred_tokens[batch_no, seq_position] == "<SKIP>"
                                    and started
                                ):  # Break if hitting a skip post at the end.
                                    end_index = seq_position - 3
                                    break

                                elif (
                                    pred_tokens[batch_no, seq_position] == "<SKIP>"
                                    and not started
                                ):  # Append as backfill up to the start.
                                    backfill_seqs.append(
                                        str(src_tokens[batch_no, seq_position - 1])
                                    )
                                    continue

                                elif pred_tokens[batch_no, seq_position] == "X":
                                    x_count += 1
                                    in_x_run = True

                                elif (
                                    pred_tokens[batch_no, seq_position].isdigit()
                                    and in_x_run
                                ):
                                    construction = build_inward_list(
                                        length=x_count,
                                        # number before X began
                                        start_num=int(
                                            pred_tokens[
                                                batch_no, (seq_position - (x_count + 1))
                                            ]
                                        ),
                                        # current number
                                        end_num=int(
                                            pred_tokens[batch_no, seq_position]
                                        ),
                                    )

                                    # Add the construction over the previous sequence
                                    nums[(seq_position - x_count) : seq_position] = (
                                        construction
                                    )
                                    # add the end
                                    nums.append(
                                        (int(pred_tokens[batch_no, seq_position]), " ")
                                    )
                                    in_x_run = False
                                    x_count = 0
                                else:
                                    nums.append(
                                        (int(pred_tokens[batch_no, seq_position]), " ")
                                    )

                                seqs.append(str(src_tokens[batch_no, seq_position - 1]))
                                if not started:
                                    start_index = seq_position - 2
                                started = True

                            if not end_index:
                                end_index = eos_position - 3
                                # eos_position - 1: Moves to the token before <EOS>,
                                # excluding the <EOS> marker itself.
                                # Subtracting an additional 1 for SOS and 1 for CLS:
                                # Adjusts further to skip over these two tokens.

                            # Backfill >>>>
                            try:
                                first_num = int(nums[0][0])  # get first number
                            except (IndexError, ValueError):
                                # When numbering has failed, `nums` is an empty list.
                                # When the sequences are messed up, the first number can
                                # be a string, like an EOS or an X token.
                                first_num = 1

                            # Should not do this before 10 in case of failure to
                            # identify the gap.
                            if (
                                first_num > 1
                                and first_num < 9
                                and len(backfill_seqs) > 0
                            ):
                                # This creates a list from 1 to first_num - 1
                                vals = list(range(1, first_num))
                                # the problem here is if there is a lot of junk...
                                vals = vals[-len(backfill_seqs) :]
                                nums = [(i, " ") for i in vals] + nums
                                seqs = list(backfill_seqs[-len(vals) :]) + seqs

                                # Adjust the start index for the backfill
                                start_index = start_index - len(backfill_seqs)

                            # Fill in up to 1 with gaps >>>>>
                            try:
                                first_num = int(nums[0][0])  # get first number
                            except (IndexError, ValueError):
                                # When numbering has failed, `nums` is an empty list.
                                # When the sequences are messed up, the first number can
                                # be a string, like an EOS or an X token.
                                first_num = 1

                            for missing_num in range(
                                first_num - 1, 0, -1
                            ):  # Start from first_num - 1, stop at 1, step by -1
                                nums.insert(0, (missing_num, " "))
                                seqs.insert(0, "-")

                            # Add gaps to nums >>>>>
                            i = 1
                            while i < len(nums):
                                if (int(nums[i][0]) - 1) > int(nums[i - 1][0]):
                                    nums.insert(i, (int(nums[i - 1][0]) + 1, " "))
                                    seqs.insert(i, "-")
                                else:
                                    i += 1  # Only increment if no insertion is made

                            # Ensure the last number is 128 >>>>>
                            last_num = int(nums[-1][0])
                            for missing_num in range(last_num + 1, 129):
                                nums.append((missing_num, " "))
                                seqs.append("-")

                            # Successful - append.
                            numbering.append(list(zip(nums, seqs)))
                            alignment.append(
                                {
                                    "chain_type": str(pred_tokens[batch_no, 1]),
                                    "score": round(normalized_score, 3),
                                    "query_start": start_index,
                                    "query_end": end_index,
                                    "error": None,
                                    "scheme": "imgt",
                                }
                            )

                        except Exception as e:
                            # Capture the error message from the exception
                            captured_error = str(e)
                            numbering.append(None)
                            alignment.append(
                                {
                                    "chain_type": "F",
                                    "score": round(normalized_score, 3),
                                    "query_start": None,
                                    "query_end": None,
                                    "error": "Could not apply numbering: "
                                    f"{captured_error}",
                                    "scheme": "imgt",
                                }
                            )

            return numbering, alignment
