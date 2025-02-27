import string

# All upper case letters, then all upper case letters doubled, then a space.
alphabet = (
    # All upper case letters.
    list(string.ascii_uppercase)
    # All upper case letters, doubled.
    + [2 * letter for letter in string.ascii_uppercase]
)

# Allowed and forbidden IMGT instertion
full_cdrs = list(range(27, 39)) + list(range(56, 66)) + list(range(105, 118))
cdr_instertion_starts = [32, 60, 111]
forbidden_cdr_insertions = [x for x in full_cdrs if x not in cdr_instertion_starts]
allowed_non_cdr_instertions = [x for x in range(1, 129) if x not in full_cdrs]


def custom_build_inward_list(length: int, start_num: int, end_num: int):
    result = []
    if int(start_num) in cdr_instertion_starts:
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

    elif int(start_num) in forbidden_cdr_insertions:
        # raise ValueError("Forbidden cdr insertion predicted.")
        return []

    elif int(start_num) in allowed_non_cdr_instertions:
        for i in range(length):
            result.append((int(start_num), alphabet[i % len(alphabet)]))
        return result

    else:
        raise ValueError("Error in converting predicted insertions labels.")


def make_csv_col_names():
    final_vector = [(1, " ")]
    imgt_nums = list(range(1, 128))
    for x in imgt_nums:
        insertions = custom_build_inward_list(
            length=len(alphabet), start_num=x, end_num=x + 1
        )
        final_vector += insertions
        final_vector += [(x + 1, " ")]

    csv_colnames = [str(x[0]) + str(x[1]) for x in final_vector]

    return csv_colnames


print(make_csv_col_names())
print(len(make_csv_col_names()))
