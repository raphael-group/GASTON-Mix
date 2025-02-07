# GASTON-Mix

GASTON-Mix is a spatial mixture-of-experts (MoE) model for learning domain-specific _topographic maps_ of a tissue slice from spatially resolved transcriptomics (SRT) data.

## Installation
We will make GASTON-Mix `pip`-installable soon. In the meanwhile, you can directly install the conda environment from the `environment.yml` file:

```
conda env create -f environment.yml
```

Then install GASTON-Mix using pip.

```
conda activate gaston-mix
pip install -e .
```

Installation should take <10 minutes. 

## Getting started

See our tutorial `tutorial.ipynb` and check out our [readthedocs](https://gaston-mix.readthedocs.io/en/latest/).

## Software dependencies
* torch
* matplotlib 
* pandas
* scikit-learn
* numpy
* jupyterlab
* seaborn
* tqdm
* scipy
* scanpy

See full list in `environment.yml` file. GASTON-Mix can be run on CPU or GPU.

We note that GASTON-Mix sometimes uses clusters from CellCharter to initialize its gating network. We suggest either making a separate environment to run CellCharter and follow their tutorial, or using a different initialization (see tutorial). 

## Citations

The GASTON-Mix pre-print is available on [biorXiv](https://www.biorxiv.org/content/10.1101/2025.01.31.635955v1). If you use GASTON-Mix for your work, please cite our paper.

```
@article {Chitra2025,
	author = {Chitra, Uthsav and Dan, Shu and Krienen, Fenna and Raphael, Benjamin J.},
	title = {GASTON-Mix: a unified model of spatial gradients and domains using spatial mixture-of-experts},
	elocation-id = {2025.01.31.635955},
	year = {2025},
	doi = {10.1101/2025.01.31.635955},
	publisher = {Cold Spring Harbor Laboratory},
	eprint = {https://www.biorxiv.org/content/early/2025/02/04/2025.01.31.635955.full.pdf},
	journal = {bioRxiv}
}
```

## Support
For questions or comments, please file a Github issue and/or email Uthsav Chitra (uchitra@broadinstitute.org)
