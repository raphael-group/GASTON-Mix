
GASTON - Mapping the topography of spatial gene expression with interpretable deep learning
===========================================================================================

GASTON-Mix is a spatial mixture-of-experts (MoE) model for learning domain-specific _topographic maps_ of a tissue slice from spatially resolved transcriptomics (SRT) data.

.. image:: https://raw.githubusercontent.com/raphael-group/GASTON/main/docs/_static/img/method_figure_v1.png
    :alt: GASTON model architecture
    :width: 400px
    :align: center

- Learn spatial domains in tissue slice, i.e. tissue geometry
- Learn 1-d coordinate that varies smoothly across each domain, providing *local topographic map* of gene expression in the domain.
- Modeling *continuous gradients* of gene expression for individual genes, e.g. gradients of metabolism in cancer


Manuscript
----------
Please see our manuscript for more details.

Getting started with GASTON-Mix
---------------------------
- Browse :doc:`notebooks/tutorials/index` for a quick start guide to GASTON.
- Discuss usage and issues on `github`_.


.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation

.. toctree::
    :caption: Tutorial
    :maxdepth: 2
    :hidden:

    notebooks/tutorials/index

.. _github: https://github.com/raphael-group/GASTON-Mix