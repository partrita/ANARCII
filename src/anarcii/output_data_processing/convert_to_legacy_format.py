def convert_output(ls, format, verbose):
    if format == "simple":
         return ls
    elif format == "legacy":
          if verbose:
               print("Converting to legacy format. Three separate lists. \n",
                  "A list of numberings, a list of all alignment details (contains, id, chain and score), and an empty list for hit tables. \n")
          
          numbering, alignment_details, hit_tables = [], [], []
          for x in ls:
            numbering.append(x[0])
            alignment_details.append(x[1])
            hit_tables.append([])

          return numbering, alignment_details, hit_tables