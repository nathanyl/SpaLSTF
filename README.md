# SpaLSTF
SpaLSTF

if gene num < 512, batchsize = 2048, hiddensize = 512; \
if gene num > 512 and num < 1024, batchsize = 512, hiddensize = 1024.

# Requirements
- python = 3.9.19
- pytorch = 2.4.1
- pytorch-geometric = 2.5.3
- sklearn = 1.3.0
# Installation
Download DeepSGE by
```
git clone https://github.com/nathanyl/SpaLSTF
```
## Run SpaLSTF model
```
python test-SpaLSTF.py 
```
```
 parameters:  
 - `gene_num`: int.  
  Amount of genes.
 - `head`: int, default `16`.  
  The number of heads of the xca_Transformer module.
 - `depth`: int, default `6`.  
  Number of Transformer blocks.
 arguments:
 - `document`: the name of current using dataset
 - `save_path_prefix`: the storage address of the training model
```
## Datasets
The datasets 1-12 in the experiment were all from the paper [Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type](https://www.nature.com/articles/s41592-022-01480-9). The raw data were initially processed in 'process/data_process.py' to convert the txt file to h5ad, and only the genes shared by both ST and scRNA-seq were retained. \

The all datasets are available for download at:
[https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/](https://github.com/nathanyl/SpaLSTF/tree/main/datasets)
 
