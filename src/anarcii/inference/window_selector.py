from .inference_utils import dataloader
from .model_loader import Loader
# import matplotlib.pyplot as plt

from ..input_data_processing.tokeniser import Tokenizer

import torch

import matplotlib.pyplot as plt

def first_index_above_threshold(preds, threshold=35):
    for i, val in enumerate(preds):
        if val > threshold:
            return i
    return None 

def detect_peaks(data, threshold=35, min_distance=65):
    peaks = []
    peak_values = []
    
    for i in range(1, len(data) - 1):
        # Check if current point is a peak above the threshold
        if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] > threshold:
            # Ensure a minimum distance from the last detected peak
            if len(peaks) == 0 or (i - peaks[-1] >= min_distance):
                peaks.append(i)
                peak_values.append(data[i])
    
    print("Number of high scoring chains found: ", len(peaks), "\n",
          "Indices: ", peaks, "\n",
          "Values: ", peak_values)
    
    # #### Plot the data and peaks
    # plt.figure(figsize=(10, 6))
    # plt.plot(data, marker='o', linestyle='-', color='b', label='Data')
    # plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    # plt.scatter(peaks, [data[i] for i in peaks], color='orange', label='Detected Peaks', zorder=5)
    # plt.title("Data with Potential Peaks")
    # plt.xlabel("Index")
    # plt.ylabel("Score of window")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
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

            # find first index over 35
            over_thirty = first_index_above_threshold(preds, 35)

            # if nothing is over 35 then drop the threshold to 30
            if not over_thirty:
                over_thirty = first_index_above_threshold(preds, 30)

            if self.scfv:
                indices = detect_peaks(preds)

                if len(indices) > 0:
                    return indices
                elif over_thirty:
                    return [over_thirty]
                else:
                    return [preds.index(max(preds))]

            if over_thirty != None:
                return over_thirty
            else:
                return preds.index(max(preds))
