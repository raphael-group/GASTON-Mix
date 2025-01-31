# GASTON-Mix

GASTON-Mix is a spatial mixture-of-experts (MoE) model for learning domain-specific _topographic maps_ of a tissue slice from spatially resolved transcriptomics (SRT) data.

## Installation
You can directly install the conda environment from the `environment.yml` file:

```
conda env create -f environment.yml
```

Then install GASTON-Mix using pip.

```
conda activate gaston-mix
pip install -e .
```

Installation should take <10 minutes. We will add GASTON-Mix to `pip` soon!

## Getting started

See our tutorial `tutorial.ipynb` (TODO: make a readthedocs).

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

The GASTON-Mix pre-print is available at [add link] If you use GASTON-Mix for your work, please cite our paper.

```

@article{Chitra2025,
	...
}

```

## Support
For questions or comments, please file a Github issue and/or email Uthsav Chitra (uchitra@broadinstitute.org)
