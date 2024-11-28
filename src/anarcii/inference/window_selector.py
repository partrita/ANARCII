from .inference_utils import dataloader
from .model_loader import Loader

from ..input_data_processing.tokeniser import Tokenizer

import torch


class WindowFinder:
    def __init__(self,
                 sequence_type,
                 mode,
                 batch_size,
                 device):

        self.type = sequence_type.lower()
        self.mode = mode.lower()
        self.batch_size = batch_size
        self.device = device

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
            # print(preds)
            # print(preds.index(max(preds)))
            return preds.index(max(preds))
