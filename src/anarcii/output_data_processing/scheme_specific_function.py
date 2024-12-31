# These are used to apply custom modifications for each scheme.
from .schemes_constants import *


def scheme_specifics(regions, scheme, chain):
    construct_scheme = scheme + "_" + chain
    function = function_dict[construct_scheme]
    result = function(regions, scheme, chain)
    return result


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


def kabat_heavy(regions, scheme, chain):
    # Kabat H region 1 (index 0)
    # Insertions are placed at Kabat position 6.
    # Count how many we recognised as insertion by the hmm
    insertions = len( [ 1 for _ in regions[0] if _[0][1] != " " ] ) 
    # We will place all insertion in this region at Kabat position 6.
    if insertions:
        start = regions[0][0][0][0] # The starting Kabat number as found by the HMM (could easily start from 2 for example)
        # I have a feeling this may be a source of a bug in very unusual cases. Can't break for now. Will catch mistakes in a validate function. 
        length = len( regions[0] )
        annotations = [ (_, " ") for _ in range(start, 7) ] + [ (6, alphabet[_]) for _ in range(insertions) ] + [(7," "),(8," "),(9," ")]
        regions[0] =  [ (annotations[i], regions[0][i][1]) for i in range(length) ]
    else:
        regions[0] = regions[0]

    # CDR1 
    # Kabat H region 3 (index 2)
    # Put insertions onto 35. Delete from 35 backwards
    length = len( regions[2] )
    insertions = max(0,length - 13)
    annotations = [(_,' ') for _ in range(23, 36)][:length] 
    annotations += [(35, alphabet[i]) for i in range(insertions) ]
    regions[2] = [ (annotations[i], regions[2][i][1]) for i in range(length) ]

    # CDR2
    # Chothia H region 5 (index 4) 
    # put insertions onto 52
    length = len( regions[4] )
    # 50 to 57 inclusive
    insertions = max(length - 8, 0) # Eight positions can be accounted for, the remainder are insertions
    # Delete in the order, 52, 51, 50,53, 54 ,55, 56, 57
    annotations  =  [(50, " "),(51, " "), (52, " ")][:max(0,length-5)]
    annotations += [(52, alphabet[i]) for i in range(insertions) ]
    annotations += [(53, " "),(54, " "),(55, " "),(56, " "),(57, " ")][ abs( min(0,length-5) ):]
    regions[4] = [ (annotations[i], regions[4][i][1]) for i in range(length) ]

    # CDR3
    # Chothia H region 7 (index 6) 
    # put insertions onto 100
    length = len( regions[6] )    
    if length > 36:
        # Too many insertions. Do not apply numbering. 
        print(f"Too many insertions, cannot apply {scheme} scheme to sequence.") 
        return [] 

    annotations = get_cdr3_annotations(length, scheme=scheme, chain_type=chain)
    regions[6]  = [ (annotations[i], regions[6][i][1]) for i in range(length)  ]

    return regions


function_dict = {
    "kabat_heavy": kabat_heavy,
    "kabat_light": kabat_light,
    "martin_heavy": martin_heavy,
    "martin_light": martin_light,
    "chothia_heavy": chothia_heavy,
    "chothia_light": chothia_light,
}