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
python -m pip install -e .
```

Use these layers as you would any other `torch.nn.Module` subclass.

#### From PyPI
*Coming soon!*

### Tutorial Notebooks

The tutorial notebooks in `tutorial` folder show example implementations of
many of the layers and tools in this package, as well as loosely cover some of
the analysis and numerical detail behind the construction of these layers.

To execute the notebooks, install the nnlayer tutorial target:
```sh
python -m pip install -e ".[tutorial]"
```

## Documentation

Read the docs at [asher-m.github.io/nnlayer](https://asher-m.github.io/nnlayer),
covering the analytic basis as well as detailed numerical overview of the
construction of these layers.

To build the docs, install the nnlayer doc target:
```sh
python -m pip install -e ".[doc]"
```

Then, from the `doc` directory, run `make.bat html` or `make html` depending on
your system.  The documentation root can be found at `build/html/index.html`.

## Development

At this time, development is primarily motivated by my own interest and need of
the tools, layers, and features here.

If you'd like to get involved for any reason, start by installing the nnlayer
dev target:
```sh
python -m pip install -e ".[dev]"
```
