# These are used to apply custom modifications for each scheme.
  
def get_cdr3_annotations(length, scheme="imgt", chain_type=""):
    """
    Given a length of a cdr3 give back a list of the annotations that should be applied to the sequence.
    
    This function should be depreciated - Why?
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



def scheme_specific_numbering(regions, scheme, chain):
    print("Fucks sake - focus on kabat heavy...\n")

    length = len( regions[6] )    
    if length > 36:
        # Too many insertions. Do not apply numbering. 
        print(f"Too many insertions, cannot apply {scheme} scheme to sequence.") 
        return [] 

    annotations = get_cdr3_annotations(length, scheme=scheme, chain_type=chain)
    
    regions[6]  = [ (annotations[i], regions[6][i][1]) for i in range(length)  ]

    return regions