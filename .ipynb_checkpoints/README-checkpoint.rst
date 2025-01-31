

GASTON-Mix - A unified model of spatial gradients and domains with spatial mixture-of-experts
===========================================================================================

Overview
--------

GASTON-Mix is a spatial mixture-of-experts (MoE) model for learning domain-specific topographic maps of a tissue slice from spatially resolved transcriptomics (SRT) data.

Installation
------------

We will make GASTON-Mix `pip`-installable soon. In the meanwhile, you can directly install the conda environment from the `environment.yml` file:

First install conda environment from `environment.yml` file:


    conda env create -f environment.yml


Then install GASTON using pip (will add to pypi soon!)

    conda activate gaston-mix
    pip install -e .

Installation should take <10 minutes. 

Software dependencies
---------------------
- torch
- matplotlib
- pandas
- scikit-learn
- numpy
- jupyterlab
- seaborn
- tqdm
- scipy
- scanpy

See the `environment.yml` file for full list.

## Getting started
Try out the Jupyter notebook tutorial: `tutorial.ipynb`. 
