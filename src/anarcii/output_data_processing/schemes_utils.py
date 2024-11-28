from .schemes_constants import *

def conversion_function(anarcii_numbered_seq, scheme_name):
    '''Takes one anarcii number sequence and applies the conversion scheme.
    Works on one sequence at a time.
    '''
    # Call the scheme
    scheme = schemes[scheme_name]

    nums = anarcii_numbered_seq[0]
    name = anarcii_numbered_seq[1]['query_name']
    chain = anarcii_numbered_seq[1]['chain_type']    
    score = anarcii_numbered_seq[1]['score']
    
    query_state = ["X" if x[0][1] == " " else "I" for x in nums]
    query_nums = [int(x[0][0]) for x in nums]
    letters = [x[1] for x in nums]

    ## Debug only
    # print(len(query_state), "".join(query_state))
    # print(len(query_nums), "-".join([str(x) for x in query_nums]))
    # print(len(letters), "".join(letters), "\n")

    n_regions = scheme['n_regions']
    region_string = scheme['region_string']
    region_index_dict = scheme['region_index_dict']

    state_string = scheme['state_string']
    rels = scheme['rels']

    _regions = [[] for _ in range(n_regions)]
    _letters = [[] for _ in range(n_regions)]
    insert_count = 0
    last_num = 0

    for num, state, letter in zip(query_nums, query_state, letters):
        num_region_index = region_index_dict[region_string[num-1]]
        num_conversion_state = state_string[num-1]

        if num_conversion_state == "X":
            if state == "X":
                offset = rels[num_region_index]
                NEW_NUM = num + offset

                # if NEW_NUM > last_num:
                _regions[num_region_index].append(((NEW_NUM, " "), letter))
                insert_count = 0

            elif state == "I":
                offset = rels[num_region_index]
                NEW_NUM = num + offset

                if NEW_NUM > last_num:
                    _regions[num_region_index].append(((NEW_NUM, " "), letter))
                    insert_count = 0
                else:
                    _regions[num_region_index].append(((str(NEW_NUM), alphabet[insert_count]), letter))
                    insert_count += 1

        elif num_conversion_state == "I" and state == "X":
            # Update the relative numbering from the imgt states - this is is crucial
            rels[num_region_index] -= 1
            offset = rels[num_region_index]
            NEW_NUM = num + offset

            if NEW_NUM > last_num:
                _regions[num_region_index].append(((NEW_NUM, " "), letter))
                insert_count = 0
            else:
                _regions[num_region_index].append(((str(NEW_NUM), alphabet[insert_count]), letter))
                insert_count += 1

        elif num_conversion_state == "I" and state == "I":
            # Do not update if both number converse and query_state are insertions
            offset = rels[num_region_index]

            NEW_NUM = num + offset
            _regions[num_region_index].append(((str(NEW_NUM), alphabet[insert_count]), letter))
            insert_count += 1

        # if NEW_NUM in [65, 66, 67, 68]:
        #     print(letter, num, offset, "=", NEW_NUM, ">>>",  num_conversion_state, state, insert_count)

        last_num = NEW_NUM
        _letters[num_region_index].append(letter)

    # DEBUG Purposes
    # for x, y in zip(_regions, _letters):
    #     result = " ".join(
    #         [(
    #             str(item[0][0]) + str(item[0][1]) + "-" + item[1]
    #             ).replace(" ", "") 
    #         for item in x]
    #         )
    #     print(result, "\n")

    return (name, [chain] + [score] + [x for y in _regions for x in y])
