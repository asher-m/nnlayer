# nnlayer

Layers for various deep learning tasks in PyTorch.


## Overview

The nnlayer package provides several useful layers for various deep learning
tasks with PyTorch.  Inspired by their analytic counterparts, these layers
allow networks to feasibly encode problem structure in the network architecture
serving to enable algorithms to better learn target problems.


## Getting Started

### Installing nnlayer

#### From Source

After cloning this repo, install nnlayer:
```sh
python -m pip install -e ./
```

Use these layers as you would any other `torch.nn.Module` subclass.

#### From PyPI
*Coming soon!*

### Tutorial Notebooks

The tutorial notebooks in `tutorial` folder show example implementations of
many of the layers and tools in this package, as well as loosely cover some of
the analysis and numerical detail behind the construction of these layers.

To execute the notebooks, be sure to install `jupyter`, `jupyterlab`, and
`matplotlib`:
```sh
python -m pip install jupyter jupyterlab matplotlib
```

## Documentation

Read the docs at [asher-m.github.io/nnlayer](https://asher-m.github.io/nnlayer),
covering the analytic basis as well as detailed numerical overview of the
construction of these layers.

To build the docs yourself, navigate to the `doc` folder in the top level of
this repo and open its readme.
