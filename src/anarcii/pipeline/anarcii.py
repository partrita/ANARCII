import os
import time

from anarcii.pipeline.configuration_utils import configure_cpus, configure_device
from anarcii.pipeline.anarcii_methods import print_initial_configuration

# Functions for processing input
from anarcii.pipeline.anarcii_utils import read_fasta, count_lines_with_greater_than, is_tuple_list, split_sequence
from anarcii.pipeline.anarcii_constants import max_seqs_len
from anarcii.pipeline.anarcii_batch_process import batch_process
from anarcii.pdb_process.pdb import renumber_pdb_with_anarcii

from anarcii.classifier.classifii import Classifii
from anarcii.classifier.classifii_utils import join_mixed_types

# Classes
from anarcii.input_data_processing.sequences import SequenceProcessor
from anarcii.inference.model_runner import ModelRunner
from anarcii.inference.window_selector import WindowFinder

# Processing output
from anarcii.pipeline.anarcii_methods import to_text, to_csv, to_json, to_dict, to_imgt_regions
from anarcii.output_data_processing.schemes import convert_number_scheme
from anarcii.output_data_processing.convert_to_legacy_format import convert_output


# This is the orchestrator of the whole pipeline.
class Anarcii:
    def __init__(self,
                 seq_type: str = "antibody",
                 mode: str = "accuracy",
                 scfv_or_concatenated_chains: bool = False, # For use with SCFVs or artificial constructs
                 batch_size: int = 8,
                 cpu: bool = False,
                 ncpu: int = -1,
                 output_format: str = "simple", # legacy for old ANARCI
                 verbose: bool = False,
                 debug_input = False):
        
        # need checks that all adhere before running code.
        self.seq_type = seq_type.lower()
        self.mode = mode.lower()
        self.batch_size = batch_size
        self.verbose = verbose
        self.cpu=cpu
        self.text_ = "output.txt"
        self.max_len_exceed = False
        self._last_numbered_output = None
        self.output_format = output_format.lower()
        self.scfv = scfv_or_concatenated_chains
        self.debug_input = debug_input

        # Attach methods
        self.print_initial_configuration = print_initial_configuration.__get__(self)
        self.to_text = to_text.__get__(self)
        self.to_csv = to_csv.__get__(self)
        self.to_json = to_json.__get__(self)
        self.to_dict = to_dict.__get__(self)
        self.to_imgt_regions = to_imgt_regions.__get__(self)

        # Get device and ncpu config
        self.ncpu = configure_cpus(ncpu)
        self.device = configure_device(self.cpu, self.ncpu)
        self.print_initial_configuration()

        # # shark model
        # self.shark_model = ModelRunner("shark", 
        #                             self.mode, self.batch_size, self.device, self.verbose)
        # self.shark_window = WindowFinder("shark",
        #                               self.mode, self.batch_size, self.device, self.scfv)
        # Antibody model
        self.ig_model = ModelRunner("antibody", 
                                    self.mode, self.batch_size, self.device, self.verbose)
        self.ig_window = WindowFinder("antibody",
                                      self.mode, self.batch_size, self.device, self.scfv)
        # TCR model
        # self.tcr_model = ModelRunner("tcr",
        #                              self.mode, self.batch_size, self.device, self.verbose)
        # self.tcr_window = WindowFinder("tcr",
        #                                self.mode, self.batch_size, self.device, self.scfv)
        

    def number(self, seqs):
        if self.seq_type.lower() == "unknown" and not (".pdb" in seqs or ".mmcif" in seqs):
            
            # classify the  sequences into tcrs or antibodies
            classifii_seqs = Classifii(batch_size=self.batch_size, device=self.device)

            antibodies, tcrs = classifii_seqs(seqs)
            if self.verbose:
                print("### Ran antibody/TCR classifier. ###\n")

            # Run both sets of numbering.
            if self.verbose:
                print("### Running Antibody model ###")
            antis_out = self.number_with_type(antibodies, "antibody")


            if self.verbose:
                print("## Running TCR model. ###")
            tcrs_out = self.number_with_type(tcrs, "tcr")


            # Choose the best option based on the conserved residues and likelihoods.
            self._last_numbered_output = join_mixed_types(antis_out, tcrs_out)

            return convert_output(ls=self._last_numbered_output, 
                                  format=self.output_format, 
                                  verbose=self.verbose)
        
        elif self.seq_type.lower() == "unknown" and (".pdb" in seqs or ".mmcif" in seqs):
            # This does not matter as the function is run within so set to antibody
            self._last_numbered_output = self.number_with_type(seqs, "antibody")
            return convert_output(ls=self._last_numbered_output, 
                                  format=self.output_format, 
                                  verbose=self.verbose)
            
        else:
            self._last_numbered_output = self.number_with_type(seqs, self.seq_type)
            return convert_output(ls=self._last_numbered_output, 
                                  format=self.output_format, 
                                  verbose=self.verbose)
        

    def to_scheme(self, scheme="imgt"):
        # Check if there's output to save
        if self._last_numbered_output is None:
            raise ValueError("No output to convert. Run the model first.")
        
        converted_seqs = convert_number_scheme(self._last_numbered_output, scheme)
        print(f"Last output converted to {scheme}")
        return converted_seqs


    def number_with_type(self, seqs, inner_type):
        if inner_type == "shark":
            model = self.shark_model
            window_model = self.shark_window
        elif inner_type == "antibody":
            model = self.ig_model
            window_model = self.ig_window
        elif inner_type == "tcr":
            model = self.tcr_model
            window_model = self.ig_window
        else:
            print("Error in defining sequence type to number.")

        # Reset this
        self.max_len_exceed = False

        # clear the output file
        if hasattr(self, "text_") and os.path.exists(self.text_):
            os.remove(self.text_)

        chunk_list = []
        begin = time.time()
        
        if isinstance(seqs, list):
            if is_tuple_list(seqs):
                if self.verbose:
                    print("Running on a list of tuples of format: [(name,sequence), ...].")
                    
                list_of_seqs = {}
                for t in seqs:
                    # Split each sequence as needed and update the dictionary
                    split_seqs = split_sequence(t[0], t[1], self.verbose)
                    list_of_seqs.update(split_seqs)

            elif not is_tuple_list(seqs):
                if self.verbose:
                    print("Running on a list of strings: [sequence, ...].")
                
                list_of_seqs = {}
                for n, t in enumerate(seqs):
                    name = f"seq{n}"
                    # Split each sequence as needed and update the dictionary
                    split_seqs = split_sequence(name, t, self.verbose)
                    list_of_seqs.update(split_seqs)

            if self.verbose:
                print("Length of sequence list: ", len(list_of_seqs))

            # If the list is huge - breakup into chunks of 1M.
            if len(list_of_seqs) > max_seqs_len:
                print(f"Max # of seqs exceeded. Running chunks of {max_seqs_len}.\n")
                num_chunks = (len(list_of_seqs)//max_seqs_len)+1
                chunk_list = {}
                for i in range(num_chunks):
                    chunk_list[i] = list_of_seqs[i*max_seqs_len: (i+1)*max_seqs_len]
                self.max_len_exceed = True

        # Fasta file
        elif isinstance(seqs, str) and os.path.exists(seqs) and ".fa" in seqs:
            if self.verbose:
                print("Running on fasta file.")
            
            num_seqs = count_lines_with_greater_than(seqs)
            if self.verbose:
                print("Length of sequence list: ", num_seqs)

            if num_seqs > max_seqs_len:
                print(f"Max # of seqs exceeded. Running chunks of {max_seqs_len}.\n")
                num_chunks = (num_seqs//max_seqs_len)+1
                for i in range(num_chunks):
                    fastas = read_fasta(seqs, self.verbose)[i*max_seqs_len: (i+1)*max_seqs_len]
                    chunk_list.append({t[0]: t[1] for t in fastas})
                self.max_len_exceed = True
            
            else:
                fastas = read_fasta(seqs, self.verbose)
                list_of_seqs = {t[0]: t[1] for t in fastas}

        elif isinstance(seqs, str) and os.path.exists(seqs) and (".pdb" in seqs or ".mmcif" in seqs):
            # This makes an infinite loop if unknown >>> Need to separate it out.
            numbered_chains = renumber_pdb_with_anarcii(seqs,
                                                        inner_seq_type=self.seq_type,
                                                        inner_mode=self.mode,
                                                        inner_batch_size=self.batch_size,
                                                        inner_cpu=self.cpu)
            return numbered_chains


        else:
            print("Input is not a list of sequences, nor a valid path to a fasta file (must end in .fa or .fasta), nor a pdb file (.pdb).")
            return []
        

        if len(chunk_list) > 1:
            numbered_seqs = batch_process(chunk_list, model, window_model, self.verbose, self.text_)
            
            end = time.time()            
            runtime = round((end - begin)/60, 2)

            if self.verbose:
                print(f"Numbered {num_seqs} seqs in {runtime} mins. \n")

            print(f"Output written to {self.text_}. Convert to csv or text with: model.to_csv(filepath) or model.to_text(filepath) ")

            return numbered_seqs

        else:
            # instantiate the Sequences class and process
            sequences = SequenceProcessor(list_of_seqs, model, window_model, self.verbose, self.scfv)
            processed_seqs = sequences.process_sequences()

            # ==============================================================================
            numbered_seqs = model(processed_seqs)
            # ==============================================================================

            end = time.time()            
            runtime = round((end - begin)/60, 2)

            if self.verbose:
                print(f"Numbered {len(numbered_seqs)} seqs in {runtime} mins. \n")
            
            if self.debug_input:
                print("Printing list of input seqs.")
                print(list_of_seqs)

            return numbered_seqs
