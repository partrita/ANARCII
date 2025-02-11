import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader



def join_mixed_types(antibodies, tcrs, names):
    print("joining types")

    dict1 = {item[1]['query_name']:item for item in antibodies}
    dict2 = {item[1]['query_name']:item for item in tcrs}

    joined = []
    for name in names:
        one = dict1.get(name, [])
        if one:
            joined.append(one)
        else:
            joined.append(dict2.get(name, []))
            
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



def split_types(seqs, classes):
    '''
    Takes a list of the classes and spits out tcrs and antibodies as separate lists with indexes maintained.
    '''
    antibodies, tcrs = [], []
    for z,c in zip(seqs, classes):
        if c == "A":
            antibodies.append(z)
        elif c == "T":
            tcrs.append(z)
        else:
            print(f"Could not classify sequence: {z}")
    return antibodies, tcrs

