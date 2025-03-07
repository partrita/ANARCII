# utils that are specific to the classifii class


def join_mixed_types(antibodies, tcrs, names):
    print("\nJoining types: TCR and Antibodies.")

    dict1 = {item[1]["query_name"]: item for item in antibodies}
    dict2 = {item[1]["query_name"]: item for item in tcrs}

    joined = []
    for name in names:
        one = dict1.get(name, [])
        if one:
            joined.append(one)
        else:
            joined.append(dict2.get(name, []))

    return joined


def split_types(seqs, classes):
    """
    Takes a list of the classes and spits out tcrs and antibodies as separate lists with
    indexes maintained.
    """
    antibodies, tcrs = [], []
    for z, c in zip(seqs, classes):
        if c == "A":
            antibodies.append(z)
        elif c == "T":
            tcrs.append(z)
        else:
            print(f"Could not classify sequence: {z}")
    return antibodies, tcrs
