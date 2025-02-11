import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader



def join_mixed_types(antibodies, tcrs):
    print("joining types")
    joined = antibodies + tcrs
    return joined



def floating_pad(list_of_lists, batch_size):
    X = []
    for i in range(0, len(list_of_lists), batch_size):
        chunk = list_of_lists[i:i+batch_size]
        X += rnn_utils.pad_sequence(chunk, batch_first=True, padding_value=0)
    return X



def dataloader(batch_size, ls):
    padded = floating_pad(ls, batch_size)
    dldr = DataLoader(padded,
                    batch_size=batch_size,
                    shuffle=False)
    return dldr



def split_sequences(indices, names_only, seqs_only, classes):
    '''
    Takes a list of the classes and spits out tcrs and antibodies as separate lists with indexes maintained.
    '''
    antibodies, tcrs = [], []
    for x,y,z,c in zip(indices, names_only, seqs_only, classes):
        if c == "A":
            antibodies.append((x, y, z))
        elif c == "T":
            tcrs.append((x, y, z))
        else:
            print(f"Could not classify sequence: {y}")
    return antibodies, tcrs

