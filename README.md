# ANARCII

ANARCII is a generalised language model for antigen receptor numbering.

For the user guide please read the [wiki page](https://github.com/oxpig/ANARCII/wiki).

## Web tool

We have a live [web tool](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarcii/):

<https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarcii/>

If you use this code or the web tool in your work, please [cite the paper](#the-paper).

## How to use

### Installation

To install ANARCII, you'll first need to ensure **uv** is installed on your system. If you don't have it yet, you can find installation instructions on the [uv GitHub page](https://www.google.com/search?q=https://github.com/astral-sh/uv%23installation).

Once uv is set up, you can install ANARCII using the following commands:

```bash
gh repo clone oxpig/ANARCII
cd ANARCII
uv sync
```

### Command line interface

ANARCII runs the ANARCII model on sequences or a FASTA file to perform protein sequence numbering. You can execute it using the `anarcii` command in your terminal.

#### Basic Usage

The most basic way to use ANARCII is by providing an input sequence string or the path to a FASTA file:

```bash
uv run anarcii "input_sequence_or_fasta_file"
```

#### Key Options

The `anarcii` command offers several options to control how the model runs and how the output is formatted.

* `-t <sequence_type>`, `--seq_type <sequence_type>`: Specifies the type of sequence to process.
    * Choices: `antibody` (default), `tcr`, `vnar`, `vhh`, `shark`, `unknown`
    * Example: `anarcii my_tcr_seq.fasta -t tcr`

* `-o <output_file>`, `--output <output_file>`: Specifies the path to an output file to save the results. The filename must end with `.csv` or `.msgpack`.
    * Note: If no output file is specified, the results will be printed to the console.
    * Example: `anarcii input.fasta -o output.csv`

* `--scheme <numbering_scheme>`: Specifies the numbering scheme to use.
    * Choices: `martin`, `kabat`, `chothia`, `imgt` (default), `aho`
    * Example: `anarcii my_seq.fasta --scheme kabat`

#### Advanced Options

* `-b <N>`, `--batch_size <N>`: Sets the batch size for processing (default: `512`).
    * Example: `anarcii input.fasta -b 1024`

* `-c`, `--cpu`: Forces the model to run on the CPU only, even if a GPU is available.
    * Example: `anarcii input.fasta -c`

* `-n <N>`, `--ncpu <N>`: Specifies the number of CPU threads to use. If set to `-1` (the default), ANARCII will use one thread per available CPU core.
    * Example: `anarcii input.fasta -n 4`

* `--max_seqs_len <N>`: Sets the maximum number of sequences to process before moving to batch mode and saving the numbered sequences in a MessagePack file (default: `102400`).
    * Example: `anarcii input.fasta --max_seqs_len 50000`

* `-m <mode>`, `--mode <mode>`: Specifies the model running mode.
    * Choices: `accuracy` (default), `speed`
    * `accuracy` mode is more precise but slower, while `speed` mode is faster but may have slightly lower accuracy.
    * Example: `anarcii input.fasta -m speed`

* `-v`, `--verbose`: Enables verbose output.
    * Example: `anarcii input.fasta -v`

* `-V`, `--version`: Prints the current version of ANARCII.
    * Example: `anarcii -V`


#### Example: Processing a FASTA file and Saving to CSV

The following command will process the `my_sequences.fasta` file as an `antibody` type and save the results to `output.csv` using the `kabat` numbering scheme:

```bash
uv run anarcii notebook/example_data/monoclonals_clean.fasta -t antibody -o output.csv --scheme kabat
```

### Python API

You can use ANARCII in Python via the `anarcii` package.

```python
from anarcii import Anarcii

model = Anarcii(
    seq_type="antibody",
    batch_size=128,
    cpu=False,
    ncpu=1,
    mode="accuracy",
    verbose=False,
)

seq = "SYVLTQPPSVSVAPGKTARITCGGNNIGSKSVHWYQQKPGQAPVLVVYDDSDRPSGIPERFSGSNSGNTATLTISRVEAGDEADYFCQVWDGSGDHPGYVFGTGTKVTVL"

results = model.number(seq)
```

## Citing this work

### The paper

[![DOI](https://zenodo.org/badge/DOI/10.1101/2025.04.16.648720.svg)](https://doi.org/10.1101/2025.04.16.648720)

All code is based on the following paper:

*ANARCII: A Generalised Language Model for Antigen Receptor Numbering*

<https://www.biorxiv.org/content/10.1101/2025.04.16.648720v1>

Please cite:

```
ANARCII: A Generalised Language Model for Antigen Receptor Numbering
Alexander Greenshields-Watson, Parth Agarwal, Sarah A Robinson, Benjamin Heathcote Williams, Gemma L Gordon, Henriette L Capel, Yushi Li, Fabian C Spoendlin, Fergus Boyles, Charlotte M Deane
bioRxiv 2025.04.16.648720; doi: https://doi.org/10.1101/2025.04.16.648720
```

### The software

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15274840.svg)](https://doi.org/10.5281/zenodo.15274840)

Releases of this software are [archived on Zenodo](https://doi.org/10.5281/zenodo.15274840).
If you use ANARCII in your work, please [cite the paper](#the-paper).  If you also want to cite the software specifically, this DOI always resolves to the Zenodo record for the latest version:

<https://doi.org/10.5281/zenodo.15274840>

If you want to cite a specific version, Zenodo provides individual DOIs for each, which you can find via the above link to the main record.

