# SpaLSTF
SpaLSTF
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
The all datasets are available for download at:
https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/
 
