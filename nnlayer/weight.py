"""
Weight layers and helpers.
"""

import torch
import torch.nn as nn


class GaussianDistanceWeight(nn.Module):
    """
    Gaussian distance weighting layer over one or more coordinate grids.
    """

    def __init__(
            self,
            *coordinates: tuple[torch.Tensor]
    ):
        """
        Computes Gaussian weights from evaluation points to a coordinate grid.

        The layer evaluates :math:`exp(-0.5 * ||c - x||^2 / w^2)` for coordinate
        locations :math:`c`, input points :math:`x`, and width :math:`w`. Passing
        a single coordinate tuple creates one coordinate grid; passing multiple
        coordinate tuples creates a batched collection of coordinate grids.

        Args:
            coordinates (tuple of Tensors):
                One or more tuples of 1-d coordinate tensors. Each tuple defines a
                tensor-product coordinate grid.

        Note:
            Each coordinate tuple should have the same number of coordinate ranges.
            When multiple coordinate tuples are supplied, corresponding coordinate
            ranges should have matching lengths.

        Attributes:
            batched (bool):
                Indicates whether multiple coordinate grids were supplied.
            dim (int):
                Number of spatial dimensions.
            width (torch.nn.Parameter):
                Learnable Gaussian width.
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

        self.dim: int = len(coordinates[0])
        self.width: nn.Parameter = nn.Parameter(torch.ones(1))

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Computes Gaussian weights for the input points.

        Args:
            x (torch.Tensor):
                Input tensor whose last dimension is equal to ``dim``.

        Note:
            If the layer is batched, the leading dimension of ``x`` must match the
            number of coordinate grids supplied at initialization.

        Returns:
            torch.Tensor:
                Gaussian weights evaluated between ``x`` and the stored coordinate
                grid or grids.
        """
        assert x.shape[-1] == self.dim, f'Received x with incompatible number of dimensions (x.shape[-1] = {x.shape[-1]}, expected {self.dim})!'
        if self.batched:
            assert len(x) == len(self.coordinates), f'Received x with incompatible batch number (len(x) = {len(x)}, expected {len(self.coordinates)})!'
            dsq = torch.sum((self.coordinates[:, *(None,) * (x.ndim - 2), ...] - x[:, ..., *(None,) * self.dim])**2, -3)
        else:
            dsq = torch.sum((self.coordinates[*(None,) * (x.ndim - 1), ...] - x[..., *(None,) * self.dim])**2, -3)
        return torch.exp(-1/2 * dsq / self.width**2)
