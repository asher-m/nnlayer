"""
The nnlayer package contains layers for various deep learning tasks in PyTorch.
"""

from . import conv
DiffConvCubicBSpline = conv.DiffConvCubicBSpline

from . import linear
LocallyConnected2d = linear.LocallyConnected2d

from . import weight
GaussianDistanceWeight = weight.GaussianDistanceWeight


__version__ = '0.0.0'
# Defining `all` without submodules prevents `from nnlayer import *`
# from grabbing submodules without nnlayer prefix and keeps importer
# namespace clean.
__all__ = [
    # conv
    "DiffConvCubicBSpline",

    # linear
    "LocallyConnected2d",

    # weight
    "GaussianDistanceWeight",
]
