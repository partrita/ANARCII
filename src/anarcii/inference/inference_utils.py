import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

alphabet = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "BB",
    "CC",
    "DD",
    "EE",
    "FF",
    "GG",
    "HH",
    "II",
    "JJ",
    "KK",
    "LL",
    "MM",
    "NN",
    "OO",
    "PP",
    "QQ",
    "RR",
    "SS",
    "TT",
    "UU",
    "VV",
    "WW",
    "XX",
    "YY",
    "ZZ",
    " ",
]


def floating_pad(list_of_lists, batch_size):
    """
    Need a template.
    One liner
    Paragraph - if complex
    Args... - descrption.
    Returns
    Raises
    """
    X = []
    for i in range(0, len(list_of_lists), batch_size):
        chunk = list_of_lists[i : i + batch_size]
        X += rnn_utils.pad_sequence(chunk, batch_first=True, padding_value=0)
    return X


def dataloader(batch_size, ls):
    padded = floating_pad(ls, batch_size)
    dldr = DataLoader(padded, batch_size=batch_size, shuffle=False)
    return dldr


def build_inward_list(length, start_num, end_num):
    cdrs = list(range(27, 38)) + list(range(56, 65)) + list(range(105, 117))

    result = []
    if int(start_num) in cdrs:
        # Calculate midpoint
        midpoint = length // 2  # Find the middle index by floor division
        # 5 becomes 2 ensures that we start at the higher number if uneven
        for i in range(midpoint):
            result.append((int(start_num), alphabet[i % len(alphabet)]))
        for i in range(midpoint, length):
            if length % 2 != 0:  # odd
                result.append(
                    (int(end_num), alphabet[(midpoint * 2 - i) % len(alphabet)])
                )
            elif length % 2 == 0:  # even
                result.append(
                    (int(end_num), alphabet[(midpoint * 2 - (i + 1)) % len(alphabet)])
                )
        return result

    else:
        for i in range(length):
            result.append((int(start_num), alphabet[i % len(alphabet)]))
        return result


def format_output(indices, names, numbering, alignment):
    assert len(indices) == len(names) == len(numbering) == len(alignment), (
        "Length of names does not equal predictions, an error has occurred."
    )

    # Update `align` with `query_name`
    for nm, align in zip(names, alignment):
        align["query_name"] = nm

    output = [
        (index, nm, number, align)
        for index, nm, number, align in zip(indices, names, numbering, alignment)
    ]

    # Sort by index
    output = sorted(output, key=lambda x: x[0])
    # Remove the original index to get back the original list format
    output = [(number, align) for _, _, number, align in output]
    return output
