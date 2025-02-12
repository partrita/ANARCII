import torch


def split_seq(seq, n_jump, window_size=90):
    jump = n_jump
    num = (len(seq) - window_size) // jump
    ls = [seq[(jump * x) : (jump * x + window_size)] for x in range(num)]
    return ls


def pick_window(list_of_seqs, model):
    # Find the index of the highest scoring window
    aa = model.sequence_tokeniser
    ls = []

    for seq in list_of_seqs:
        bookend_seq = [aa.start] + list(seq) + [aa.end]
        try:
            tokenised_seq = torch.from_numpy(aa.encode(bookend_seq))
            ls.append(tokenised_seq)
        except KeyError as e:
            print(f"Sequence could not be numbered. Contains an invalid residue: {e}")
            ls.append([])

    max_index = model(ls)
    return max_index


def find_scfvs(list_of_seqs, model):
    # Returns a list of indices of the peaks.
    aa = model.sequence_tokeniser
    ls = []

    for seq in list_of_seqs:
        bookend_seq = [aa.start] + list(seq) + [aa.end]
        try:
            tokenised_seq = torch.from_numpy(aa.encode(bookend_seq))
            ls.append(tokenised_seq)
        except KeyError as e:
            print(f"Sequence could not be numbered. Contains an invalid residue: {e}")
            ls.append([])

    list_of_indices = model(ls)
    return list_of_indices
