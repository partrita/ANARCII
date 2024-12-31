# These are used to apply custom modifications for each scheme.
from .schemes_constants import *


def scheme_specifics(regions, scheme, chain):
    construct_scheme = scheme + "_" + chain
    function = function_dict[construct_scheme]
    result = function(regions)
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


### ### HEAVY FUNCTIONS ### ###
def chothia_heavy(regions):
    # Chothia H region 1 (index 0) >>> Insertions are placed at Chothia position 6.
    # Count how many we recognised as insertion by the hmm
    insertions = len( [ 1 for _ in regions[0] if _[0][1] != " " ] ) 
    # We will place all insertion in this region at Chothia position 6.
    if insertions:
        start = regions[0][0][0][0] # The starting Chothia number as found by the HMM (could easily start from 2 for example)
        # I have a feeling this may be a source of a bug in very unusual cases. Can't break for now. Will catch mistakes in a validate function. 
        length = len( regions[0] )
        annotations = [ (_, " ") for _ in range(start, 7) ] + [ (6, alphabet[_]) for _ in range(insertions) ] + [(7," "),(8," "),(9," ")]
        regions[0] =  [ (annotations[i], regions[0][i][1]) for i in range(length) ]
    else:
        regions[0] = regions[0]
    
    # CDR1 
    # Chothia H region 3 (index 2) >>> put insertions onto 31
    length = len( regions[2] )
    insertions = max(length - 11, 0) # Pulled back to the cysteine as heavily engineered cdr1's are not playing nicely
    if insertions:
        annotations = [(_, " ") for _ in range(23,32)] + [(31, alphabet[i]) for i in range(insertions) ] + [(32," "),(33," ")]
    else:
        annotations = [(_, " ") for _ in range(23,32)][:length-2] + [(32," "),(33," ")][:length]
    regions[2] = [ (annotations[i], regions[2][i][1]) for i in range(length) ]
 
    # CDR2
    # Chothia H region 5 (index 4) >>> put insertions onto 52
    length = len( regions[4] )
    # 50 to 57 inclusive
    insertions = max(length - 8, 0) # Eight positions can be accounted for, the remainder are insertions
    # Delete in the order, 52, 51, 50,53, 54 ,55, 56, 57
    annotations  =  [(50, " "),(51, " "), (52, " ")][:max(0,length-5)]
    annotations += [(52, alphabet[i]) for i in range(insertions) ]
    annotations += [(53, " "),(54, " "),(55, " "),(56, " "),(57, " ")][ abs( min(0,length-5) ):]
    regions[4] = [ (annotations[i], regions[4][i][1]) for i in range(length) ]
     
    # CDR3
    # Chothia H region 7 (index 6) >>> put insertions onto 100
    length = len( regions[6] )    
    if length > 36: return []
    annotations = get_cdr3_annotations(length, scheme="chothia", chain_type="heavy")
    regions[6]  = [ (annotations[i], regions[6][i][1]) for i in range(length)  ]

    return regions                                  


def kabat_heavy(regions):
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
    # Kabat H region 3 (index 2) >>> Put insertions onto 35. Delete from 35 backwards
    length = len( regions[2] )
    insertions = max(0,length - 13)
    annotations = [(_,' ') for _ in range(23, 36)][:length] 
    annotations += [(35, alphabet[i]) for i in range(insertions) ]
    regions[2] = [ (annotations[i], regions[2][i][1]) for i in range(length) ]

    # CDR2
    # Chothia H region 5 (index 4) >>> put insertions onto 52
    length = len( regions[4] )
    # 50 to 57 inclusive
    insertions = max(length - 8, 0) # Eight positions can be accounted for, the remainder are insertions
    # Delete in the order, 52, 51, 50,53, 54 ,55, 56, 57
    annotations  =  [(50, " "),(51, " "), (52, " ")][:max(0,length-5)]
    annotations += [(52, alphabet[i]) for i in range(insertions) ]
    annotations += [(53, " "),(54, " "),(55, " "),(56, " "),(57, " ")][ abs( min(0,length-5) ):]
    regions[4] = [ (annotations[i], regions[4][i][1]) for i in range(length) ]

    # CDR3
    # Chothia H region 7 (index 6) >>> put insertions onto 100
    length = len( regions[6] )  
    if length > 36: return [] # Too many insertions. Do not apply numbering. 

    annotations = get_cdr3_annotations(length, scheme="kabat", chain_type="heavy")
    regions[6]  = [ (annotations[i], regions[6][i][1]) for i in range(length)  ]

    return regions


def martin_heavy(regions):
    # Chothia H region 1 (index 0)
    # Insertions are placed at Chothia position 8.
    insertions = len( [ 1 for _ in regions[0] if _[0][1] != " " ] ) 
    # We will place all insertion in this region at Chothia position 8.
    if insertions:
        start = regions[0][0][0][0] # The starting Chothia number as found by the HMM (could easily start from 2 for example)
        # I have a feeling this may be a source of a bug in very unusual cases. Can't break for now. Will catch mistakes in a validate function. 
        length = len( regions[0] )
        annotations = [ (_, " ") for _ in range(start, 9) ] + [ (8, alphabet[_]) for _ in range(insertions) ] + [(9," ")]
        regions[0] =  [ (annotations[i], regions[0][i][1]) for i in range(length) ]
    else:
        regions[0] = regions[0]
    
    # CDR1 
    # Chothia H region 3 (index 2) >>> put insertions onto 31
    length = len( regions[2] )
    insertions = max(length - 11, 0) # Pulled back to the cysteine as heavily engineered cdr1's are not playing nicely
    if insertions:
        annotations = [(_, " ") for _ in range(23,32)] + [(31, alphabet[i]) for i in range(insertions) ] + [(32," "),(33," ")]
    else:
        annotations = [(_, " ") for _ in range(23,32)][:length-2] + [(32," "),(33," ")][:length]
    regions[2] = [ (annotations[i], regions[2][i][1]) for i in range(length) ]
 
    # CDR2
    # Chothia H region 5 (index 4) >>> put insertions onto 52
    length = len( regions[4] )
    # 50 to 57 inclusive
    insertions = max(length - 8, 0) # Eight positions can be accounted for, the remainder are insertions
    # Delete in the order, 52, 51, 50,53, 54 ,55, 56, 57
    annotations  =  [(50, " "),(51, " "), (52, " ")][:max(0,length-5)]
    annotations += [(52, alphabet[i]) for i in range(insertions) ]
    annotations += [(53, " "),(54, " "),(55, " "),(56, " "),(57, " ")][ abs( min(0,length-5) ):]
    regions[4] = [ (annotations[i], regions[4][i][1]) for i in range(length) ]

    # FW3 
    # Place all insertions on 72 explicitly.
    # This is in contrast to Chothia implementation where 3 insertions are on 82 and then further insertions are placed by the  alignment
    # Gaps are placed according to the alignment...
    length = len( regions[5] )
    insertions = max(length - 35, 0)
    if insertions > 0: # Insertions on 72
        annotations = [(i,' ') for i in range(58,73)]+[(72, alphabet[i]) for i in range(insertions) ]+[(i,' ') for i in range(73,93)]
        regions[5] = [ (annotations[i], regions[5][i][1]) for i in range(length) ]
    else: # Deletions - all alignment to place them. 
        regions[4] = regions[4]

    # CDR3
    # Chothia H region 7 (index 6) >>> put insertions onto 100
    length = len( regions[6] )    
    if length > 36: return [] # Too many insertions. Do not apply numbering. 

    annotations = get_cdr3_annotations(length, scheme="chothia", chain_type="heavy")
    regions[6]  = [ (annotations[i], regions[6][i][1]) for i in range(length)  ]

    return regions


### ### LIGHT FUNCTIONS ### ###
def chothia_light(regions):
    # CDR1 
    # Chothia L region 2 (index 1)
    # put insertions onto 30
    length = len( regions[1] )
    insertions = max(length - 11, 0) # Eleven positions can be accounted for, the remainder are insertions
    # Delete forward from 31 
    annotations  =  [(24, " "),(25, " "), (26, " "), (27, " "), (28, " "),(29, " "),(30, " ")][:max(0,length)] 
    annotations += [(30, alphabet[i]) for i in range(insertions) ]
    annotations += [(31, " "),(32, " "),(33, " "),(34, " ")][ abs( min(0,length-11) ):] 
    regions[1] = [ (annotations[i], regions[1][i][1]) for i in range(length) ]


    # CDR2
    # Chothia L region 4 (index 3) 
    # put insertions onto 52. 
    length = len( regions[3] )
    insertions = max( length - 4, 0 )
    if insertions > 0:
        annotations  = [(51, " "),(52, " ")] + [(52, alphabet[i]) for i in range(insertions) ] + [(53, " "),(54, " ")]
        regions[3] = [ (annotations[i], regions[3][i][1]) for i in range(length) ]
    else: # How to gap L2 in Chothia/Kabat/Martin is unclear so we let the alignment do it.
        regions[3] = regions[3]
    
    # FW3
    # Insertions on 68. First deletion 68. Otherwise default to alignment
    length = len( regions[4] )
    insertions = max(length - 34, 0)
    if insertions > 0: # Insertions on 68
        annotations = [(i," ") for i in range(55,69)]+[(68, alphabet[i]) for i in range(insertions) ]+[(i," ") for i in range(69,89)]
        regions[4] = [ (annotations[i], regions[4][i][1]) for i in range(length) ]
    elif length == 33: # First deletion on 68
        annotations = [(i," ") for i in range(55,68)]+[(i," ") for i in range(69,89)]            
        regions[4] = [ (annotations[i], regions[4][i][1]) for i in range(length) ]
    else: # More deletions - allow alignment to place them 
        regions[4] = regions[4]

    # CDR3
    # Chothia L region 6 (index 5) 
    # put insertions onto 95
    length = len( regions[5] )    
    if length > 36: return [] # Too many insertions. Do not apply numbering. 

    annotations = get_cdr3_annotations(length, scheme="chothia", chain_type="light")
    regions[5]  = [ (annotations[i], regions[5][i][1]) for i in range(length)  ]


def kabat_light(regions):
    # CDR1 
    # Kabat L region 2 (index 1) >>> put insertions onto 27
    length = len( regions[1] )
    insertions = max(length - 11, 0) # Eleven positions can be accounted for, the remainder are insertions
    # Delete forward from 28 
    annotations  =  [(24, " "),(25, " "), (26, " "), (27, " ")][:max(0,length)] 
    annotations += [(27, alphabet[i]) for i in range(insertions) ]
    annotations += [(28, " "),(29, " "),(30, " "),(31, " "),(32, " "),(33, " "),(34, " ")][ abs( min(0,length-11) ):] 
    regions[1] = [ (annotations[i], regions[1][i][1]) for i in range(length) ]
  
    # CDR2
    # Chothia L region 4 (index 3) >>> put insertions onto 52. 
    length = len( regions[3] )
    insertions = max( length - 4, 0 )
    if insertions > 0:
        annotations  = [(51, " "),(52, " ")] + [(52, alphabet[i]) for i in range(insertions) ] + [(53, " "),(54, " ")]
        regions[3] = [ (annotations[i], regions[3][i][1]) for i in range(length) ]
    else: # How to gap L2 in Chothia/Kabat/Martin is unclear so we let the alignment do it.
        regions[3] = regions[3]

    # CDR3
    # Chothia L region 6 (index 5) >>> put insertions onto 95
    length = len( regions[5] )    
    if length > 36: return [] # Too many insertions. Do not apply numbering. 

    annotations = get_cdr3_annotations(length, scheme="kabat", chain_type="light")
    regions[5]  = [ (annotations[i], regions[5][i][1]) for i in range(length)  ]
    
    return regions


def martin_light(regions):
    # The Martin and Chothia specification for light chains are very similar. Martin is more explicit in the location of indels 
    # but unlike the heavy chain these are additional instead of changes to the Chothia scheme. Thus, Chothia light is implemented
    # as martin light.
    return chothia_light(regions)


# Dict to call by string
function_dict = {
    "kabat_heavy": kabat_heavy,
    "kabat_light": kabat_light,
    "martin_heavy": martin_heavy,
    "martin_light": martin_light,
    "chothia_heavy": chothia_heavy,
    "chothia_light": chothia_light,
}