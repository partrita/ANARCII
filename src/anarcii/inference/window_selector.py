from .inference_utils import dataloader
from .model_loader import Loader
import matplotlib.pyplot as plt

from ..input_data_processing.tokeniser import Tokenizer

import torch

def detect_peaks(data, threshold=32.5):

    peaks = []
    peak_values = []

    in_peak = False
    current_peak_value = None
    current_peak_index = None

    for i in range(1, len(data)):
        # Handle the last point
        is_last_point = i == len(data) - 1

        # Check if current point is a peak
        if not is_last_point and data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] > threshold:
            if not in_peak:
                # Start of a new peak
                in_peak = True
                current_peak_value = data[i]
                current_peak_index = i
            else:
                # Update peak if current value is higher
                if data[i] > current_peak_value:
                    current_peak_value = data[i]
                    current_peak_index = i
        elif in_peak and (is_last_point or data[i] <= threshold or data[i] < data[i - 1]):
            # End of a peak or at the last point
            peaks.append(current_peak_index)
            peak_values.append(current_peak_value)
            in_peak = False
            current_peak_value = None

    print("Number of high scoring chains found: ", len(peaks), "\n",
          "Indices: ", peaks, "\n",
          "Values: ", peak_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='b', label='Data')
    plt.axhline(y=32.5, color='r', linestyle='--', label='Threshold (25)')
    plt.title("Data with Potential Peaks")
    plt.xlabel("Index")
    plt.ylabel("Score of window")
    plt.legend()
    plt.grid(True)
    plt.show()

    return peaks


class WindowFinder:
    def __init__(self,
                 sequence_type,
                 mode,
                 batch_size,
                 device,
                 scfv):

        self.type = sequence_type.lower()
        self.mode = mode.lower()
        self.batch_size = batch_size
        self.device = device
        self.scfv = scfv

        if self.type in ["antibody", "shark"]:
            self.sequence_tokeniser = Tokenizer("protein_antibody")
            self.number_tokeniser = Tokenizer("number_antibody")

        elif self.type == "tcr":
            self.sequence_tokeniser = Tokenizer("protein_tcr")
            self.number_tokeniser = Tokenizer("number_tcr")
        else:
            raise ValueError(f"Invalid model type: {self.type}")

        self.model = self._load_model()

    def _load_model(self):
        model_loader = Loader(self.type,
                              self.mode,
                              self.device)
        return model_loader.model

    def __call__(self, list_of_seqs):
        dl = dataloader(self.batch_size, list_of_seqs)
        predictions = self._predict_numbering(dl)
        return predictions

    def _predict_numbering(self, dl):
        preds = []
        with torch.no_grad():
            for X in dl:
                src = X.to(self.device)
                batch_size = src.shape[0]

                src_mask = self.model.make_src_mask(src)
                enc_src = self.model.encoder(src, src_mask)
                input = src[:, 0].unsqueeze(1)

                trg_pad_mask, trg_causal_mask = self.model.make_trg_mask(input)
                output = self.model.decoder(input, enc_src,
                                            trg_pad_mask, trg_causal_mask,
                                            src_mask)
                likelihoods = output.topk(1, dim=2).values[:, 0]

                for batch_no in range(batch_size):
                    normalized_likelihood = likelihoods[batch_no, 0].item()
                    preds.append(round(normalized_likelihood, 3))

            # return the index of the max
            if self.scfv:
                # print(preds)
                indices = detect_peaks(preds)
                return indices

            return preds.index(max(preds))
