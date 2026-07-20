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
__all__ = [
    # conv
    "DiffConvCubicBSpline",

    # linear
    "LocallyConnected2d",

    # weight
    "GaussianDistanceWeight",
]
