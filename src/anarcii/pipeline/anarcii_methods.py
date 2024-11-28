import torch, ast, shutil

from anarcii.output_data_processing.list_to import write_csv, write_text, write_json, return_dict, return_imgt_regions

from anarcii.output_data_processing.schemes import convert_number_scheme


def print_initial_configuration(self):
    """Print initial configuration details if verbose mode is enabled."""
    if self.verbose:
        print(f"Batch size: {self.batch_size}")
        print(
            "\tSpeed is a balance of batch size and length diversity. Adjust accordingly.\n",
            "\tSeqs all similar length (+/-5), increase batch size. Mixed lengths (+/-30), reduce.\n"
        )
        if not self.cpu:
            if self.batch_size < 512:
                print("Consider a batch size of at least 512 for optimal GPU performance.")
            elif self.batch_size > 512:
                print("For A100 GPUs, a batch size of 1024 is recommended.")
        else:
            print("Recommended batch size for CPU: 8.")


def to_text(self, file_path):
    # Check if there's output to save
    if self._last_numbered_output is None:
        raise ValueError("No output to save. Run the model first.")
    
    if self.max_len_exceed:
        shutil.copy2(self.text_, file_path)

    else:
        write_text(self._last_numbered_output, file_path)
        print(f"Last output saved to {file_path}")



def to_csv(self, file_path):
    # Check if there's output to save
    if self._last_numbered_output is None:
        raise ValueError("No output to save. Run the model first.")
    
    if self.max_len_exceed:
        with open(self.text_, "r") as file:
            loaded_data = [ast.literal_eval(line.strip()) for line in file]
            
        write_csv(loaded_data, file_path)
        print(f"Last output saved to {file_path}")
        
    else:
        write_csv(self._last_numbered_output, file_path)
        print(f"Last output saved to {file_path}")



def to_json(self, file_path):
    # Check if there's output to save
    if self._last_numbered_output is None:
        raise ValueError("No output to save. Run the model first.")
    
    if self.max_len_exceed:
        with open(self.text_, "r") as file:
            loaded_data = [ast.literal_eval(line.strip()) for line in file]
            
        write_json(loaded_data, file_path)
        print(f"Last output saved to {file_path}")
        
    else:
        write_json(self._last_numbered_output, file_path)
        print(f"Last output saved to {file_path}")



def to_dict(self):
    # Check if there's output to save
    if self._last_numbered_output is None:
        raise ValueError("No output. Run the model first.")
    
    if self.max_len_exceed:
        with open(self.text_, "r") as file:
            loaded_data = [ast.literal_eval(line.strip()) for line in file]
        dt = return_dict(loaded_data)
        return dt
        
    else:
        dt = return_dict(self._last_numbered_output)
        return dt
    


def to_imgt_regions(self):
    # Check if there's output to save
    if self._last_numbered_output is None:
        raise ValueError("No output. Run the model first.")
    
    if self.max_len_exceed:
        with open(self.text_, "r") as file:
            loaded_data = [ast.literal_eval(line.strip()) for line in file]
        ls = return_imgt_regions(loaded_data)
        return ls
        
    else:
        ls = return_imgt_regions(self._last_numbered_output)
        return ls



def to_scheme(self, scheme="imgt"):
    # Check if there's output to save
    if self._last_numbered_output is None:
        raise ValueError("No output to convert. Run the model first.")
    
    converted_seqs = convert_number_scheme(self._last_numbered_output, scheme)
    print(f"Last output converted to {scheme}")
    
    return converted_seqs
