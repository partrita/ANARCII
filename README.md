# ANARCII-INTERNAL
Internal version of ANARCII package for antibody group.


# Installation: CPU only

```
conda install numpy
conda install pytorch cpuonly -c pytorch

cd ANARCII-DEV
pip install .

```

# Installation: GPU only

```
conda install numpy pandas
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

cd ANARCII-DEV
pip install .

```

# CMD Line usage
Pass a sequence, or a fasta file and this will be run with default params. Output will print to the screen.
```
anarcii QQVRQSPQSLTVWEGETAILNCSYEDSTFNYFPWYQQFPGEGPALLISIRSVSDKKEDGRFTIFFNKREKKLSLHITDSQPGDSATYFCAARYQGGRALIFGTGTTVSVSPGSADAAAVTLLEQNPRWRLVPRGQ

# point to a fasta
anarcii ./notebook/example_data/monoclonals_clean.fasta.gz
```

Specify parameters to capture output to a text, csv, or json file.
```
anarcii ./notebook/example_data/monoclonals_clean.fasta.gz -o my_numbered_seqs.csv
```