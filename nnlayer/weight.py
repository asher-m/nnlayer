"""
Weight layers and helpers.
"""

import torch
import torch.nn as nn


__all__ = [
    "GaussianDistanceWeight"
]


class GaussianDistanceWeight(nn.Module):
    def __init__(
            self,
            *coordinates: tuple[torch.Tensor]
    ):
        """
        Compute the Gaussian weight across an array of coordinates to a point x.

        Args:
            coordinates:        Tuple of 1-d tensors of coordinate ranges.  Coordinate ranges are usually
                                monotonic, but I don't check and I don't think it will break anything.

        Every tuple in ``*coordinates`` should have the same number of coordinate ranges (i.e., the same spatial dimension,)
        and corresponding coordinate ranges should have the same length.  For example, if coordinates = ((x1, y1), (x2, y2)),
        then len(x1) = len(x2) and len(y1) = len(y2).
        """
        super().__init__()

        if len(coordinates) == 1:
            self.coordinates = nn.Buffer(
                torch.stack(torch.meshgrid(*coordinates[0], indexing='ij'))
            )
            self.batched = False
        elif len(coordinates) > 1:
            self.coordinates = nn.Buffer(
                torch.stack([torch.stack(torch.meshgrid(*c, indexing='ij')) for c in coordinates])
            )
            self.batched = True
        else:
            raise ValueError('Received no coordinate ranges!')

        self.dim = len(coordinates[0])
        """ Number of spatial dimensions. """

        self.width = nn.Parameter(torch.ones(1))
        """ Width of the Gaussian. """

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Args:
            x:      A tensor where the last dimension corresponds to the number of dimensions
                    (i.e. coordinate ranges) used at instance initialization.

        If batched (i.e., multiple coordinate ranges were used at instance initialization,) the first axis of the input
        tensor must correspond to the number of batches (coordinate ranges or number of tuples) passed at initialization.
        """
        assert x.shape[-1] == self.dim, f'Received x with incompatible number of dimensions (x.shape[-1] = {x.shape[-1]}, expected {self.dim})!'
        if self.batched:
            assert len(x) == len(self.coordinates), f'Received x with incompatible batch number (len(x) = {len(x)}, expected {len(self.coordinates)})!'
            dsq = torch.sum((self.coordinates[:, *(None,) * (x.ndim - 2), ...] - x[:, ..., *(None,) * self.dim])**2, -3)
        else:
            dsq = torch.sum((self.coordinates[*(None,) * (x.ndim - 1), ...] - x[..., *(None,) * self.dim])**2, -3)
        return torch.exp(-1/2 * dsq / self.width**2)
