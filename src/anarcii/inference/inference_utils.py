import string

import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

# All upper case letters, then all upper case letters doubled, then a space.
alphabet = (
    # All upper case letters.
    list(string.ascii_uppercase)
    # All upper case letters, doubled.
    + [2 * letter for letter in string.ascii_uppercase]
    # A space.
    + [" "]
)


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


def format_output(indices, names, numbering, alignment, offsets):
    if not len(indices) == len(names) == len(numbering) == len(alignment):
        exit("Length of names does not equal predictions, an error has occurred.")

    # Update `align` with `query_name`
    for name, align in zip(names, alignment):
        align["query_name"] = name
        if offset := offsets.get(name):
            try:
                align["query_start"] += offset
                align["query_end"] += offset
            except TypeError:
                # catch None type in query start and end
                continue

    output = sorted(zip(indices, numbering, alignment))

    # Remove the original index to get back the original list format
    output = [(number, align) for _, number, align in output]
    return output
