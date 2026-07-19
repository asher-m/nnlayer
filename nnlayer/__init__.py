"""
The nnlayer package contains layers for various deep learning tasks in PyTorch.
"""

from .conv import (
    DiffConvCubicBSpline
)
from .linear import (
    LocallyConnected2d
)
from .weight import (
    GaussianDistanceWeight
)

__version__ = '0.0.0'
__all__ = [
    "DiffConvCubicBSpline",
    "GaussianDistanceWeight",
    "LocallyConnected2d"
]
