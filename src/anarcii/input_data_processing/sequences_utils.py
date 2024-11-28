from anarcii.pipeline.anarcii_constants import n_jump

import torch


def split_seq(seq):
    jump = n_jump
    num = (len(seq)-90) // jump
    ls = [seq[(jump*x):(jump*x + 90)] for x in range(num)]
    return ls

def pick_window(list_of_seqs, model):
    aa = model.sequence_tokeniser
    ls = []

    for seq in list_of_seqs:
        bookend_seq = [aa.start] + [s for s in seq] + [aa.end]
        try:
            tokenised_seq = torch.from_numpy(aa.encode(bookend_seq))
            ls.append(tokenised_seq)
        except KeyError as e:
            print(
                f"Sequence could not be numbered. Contains an invalid residue: {e}")
            ls.append([])

    max_index = model(ls)
    return max_index