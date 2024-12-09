from .schemes_constants import *

def conversion_function(anarcii_numbered_seq, scheme_name):
    '''Takes one anarcii number sequence and applies the conversion scheme.
    Works on one sequence at a time.
    '''
    # Call the scheme
    scheme = schemes[scheme_name]

    print(scheme_name)

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
    
    # must create a copy to avoid changing the imported object.
    rels = scheme['rels'].copy()

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
    for x, y in zip(_regions, _letters):
        result = " ".join(
            [(str(item[0][0]) + str(item[0][1]) + "-" + item[1]).replace(" ", "") for item in x])
        print(result, "\n")

    ##### Renumbering required for specific regions. #####
    

    length = len( _regions[6] )    
    if length > 36:
        # Too many insertions. Do not apply numbering. 
        print(f"Too many insertions, cannot apply {scheme} scheme to sequence.") 
        return [] 
    
    annotations = get_cdr3_annotations(length, scheme="kabat", chain_type="heavy")
    
    _regions[6]  = [ (annotations[i], _regions[6][i][1]) for i in range(length)  ]


    return ([x for y in _regions for x in y], 
            {
                "chain_type": chain,
                "score": score,
                "query_start": None,
                "query_end": None,
                "error": None,
                "query_name": name
                                })

    
def get_cdr3_annotations(length, scheme="imgt", chain_type=""):
    """
    Given a length of a cdr3 give back a list of the annotations that should be applied to the sequence.
    
    This function should be depreciated
    """
    az = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
    za = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    
    if scheme=="imgt":
        start, end = 105, 118 # start (inclusive) end (exclusive)
        annotations = [None for _ in range(max(length,13))]
        front = 0
        back  = -1
        assert (length-13) < 50, "Too many insertions for numbering scheme to handle" # We ran out of letters.
        for i in range(min(length,13)):
            if i%2:
                annotations[back] = (end+back, " ")
                back -= 1
            else:
                annotations[front] = (start+front, " ")
                front += 1
        for i in range(max(0,length-13)): # add insertions onto 111 and 112 in turn
            if i%2:
                annotations[back] = (112, za[back+6])
                back-=1
            else:
                annotations[front] = (111, az[front-7])
                front +=1        
        return annotations

    elif scheme in [ "chothia", "kabat"] and chain_type=="heavy": # For chothia and kabat
        # Number forwards from 93
        insertions = max(length - 10, 0)
        assert insertions < 27, "Too many insertions for numbering scheme to handle" # We ran out of letters.
        ordered_deletions = [ (100, ' '), (99,' '), (98,' '), (97,' '), (96,' '), (95,' '), (101,' '),(102,' '),(94,' '), (93,' ') ]
        annotations = sorted( ordered_deletions[ max(0, 10-length): ] + [ (100,a) for a in az[:insertions ] ] )
        return annotations

    elif scheme in [ "chothia", "kabat"] and chain_type=="light":
        # Number forwards from 89
        insertions = max(length - 9, 0)
        assert insertions < 27, "Too many insertions for numbering scheme to handle" # We ran out of letters.
        ordered_deletions = [ (95,' '),(94,' '),(93,' '),( 92,' '),(91,' '),(96,' '),(97,' '),(90,' '),(89,' ') ]
        annotations = sorted( ordered_deletions[ max(0, 9-length): ] + [ (95,a) for a in az[:insertions ] ] )
        return annotations

    else:
        raise AssertionError("Unimplemented scheme")

