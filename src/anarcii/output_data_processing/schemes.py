from .schemes_utils import conversion_function


def convert_number_scheme(list_numbered_seqs, scheme):
    '''Renumber a list of IMGT seqs with new scheme.

    This takes a list of IMGT numbered sequences.
    It works out if each sequence is a heavy or light chain
    Defines the scheme to be applied
    Then calls the conversion function on that sequence
    '''

    converted_seqs = []
    for sublist in list_numbered_seqs:
        # print(sublist)
        chain_call = sublist[1]['chain_type']
        chain = "heavy" if chain_call == "H" else "light"
        
        if scheme.lower() == "imgt":
            converted_seqs.append(
            conversion_function(sublist, scheme.lower())
            )
        else:
            scheme_name = scheme.lower() + "_" + chain
            converted_seqs.append(
                conversion_function(sublist, scheme_name)
            )

    return converted_seqs
