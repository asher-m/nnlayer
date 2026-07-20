"""
Linear (fully-connected) layers and helpers.

Formally speaking, these transformations are *affine* linear,
but "linear" is the prevailing terminology, so we use that here, too.
"""

import functools
import torch
import torch.nn as nn


class LocallyConnected2d(nn.Module):
    """
    Locally connected two-dimensional affine layer.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_shape_spatial: tuple[int, ...],
            kernel_size: int,
            dilation: int = 1,
            padding: int = 0,
            stride: int = 1,
    ):
        """
        Applies a spatially-local affine transformation over image-like input.

        Unlike a convolution, this module learns a separate weight and bias for each
        output spatial location. It is suitable for dense local transformations when a
        fully connected layer would be prohibitively large.

        See Also:
            See :class:`torch.nn.Unfold` for an overview on how local blocks are
            extracted. Parameter names in this class follow the notation used by
            :class:`torch.nn.Unfold` where possible.

        Warning:
            Currently, only 4-d input tensors (batched image-like tensors) are supported,
            as in :class:`torch.nn.Unfold` and :func:`torch.nn.functional.unfold`.

        Args:
            in_channels (int):
                Number of channels in the input tensor.
            out_channels (int):
                Number of output channels produced at each spatial location.
            in_shape_spatial (tuple of int):
                Spatial shape of the input tensor.
            kernel_size (int):
                Size of the sliding local block.
            dilation (int):
                Stride of elements within a kernel neighborhood.
            stride (int):
                Stride for the sliding window.
            padding (int):
                Zero-padding added to both sides of the input.

        Attributes:
            out_shape_spatial (tuple of int):
                Spatial shape of the output tensor inferred from the constructor
                arguments.
            total_blocks (int):
                Number of local blocks extracted from each input sample.
            weight (torch.nn.Parameter):
                Learnable weights with shape
                ``(out_channels, *out_shape_spatial, block_size)``.
            bias (torch.nn.Parameter):
                Learnable bias with shape ``(out_channels, *out_shape_spatial)``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape_spatial = in_shape_spatial
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.out_shape_spatial: tuple[int, ...] = tuple(
            (
                (self.in_shape_spatial[d] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
                for d in range(len(self.in_shape_spatial))
            )
        )
        self.total_blocks: int = functools.reduce(lambda x, y: x * y, self.out_shape_spatial)

        block_size = self.in_channels * self.kernel_size**len(self.in_shape_spatial)
        self.weight = nn.Parameter(torch.ones(out_channels, *self.out_shape_spatial, block_size) / block_size)
        self.bias = nn.Parameter(torch.randn(out_channels, *self.out_shape_spatial))

    def forward(self, x: torch.Tensor):
        """
        Applies the locally connected transformation to the input tensor.

        Warning:
            Currently, only 4-d input tensors (batched image-like tensors) are supported,
            as in :class:`torch.nn.Unfold` and :func:`torch.nn.functional.unfold`.

            If ``x`` is 3-d, the method adds a leading batch dimension internally
            and removes it before returning. Other input ranks are not supported.

        Args:
            x (torch.Tensor):
                Input tensor with shape ``(batch_size, in_channels, *in_shape_spatial)``
                or ``(in_channels, *in_shape_spatial)``.

        Returns:
            torch.Tensor:
                Output tensor with shape
                ``(batch_size, out_channels, *out_shape_spatial)`` or
                ``(out_channels, *out_shape_spatial)`` for unbatched input.
        """
        if x.ndim == 4:
            batch_size = x.size(0)

            blocks = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            out = torch.einsum('bil,oli->bol', blocks, self.weight.flatten(1, -2)) + self.bias.flatten(1, -1).unsqueeze(0)
            out = out.reshape((batch_size, self.out_channels, *self.out_shape_spatial))
            return out

        elif x.ndim == 3:
            x = x.unsqueeze(0)
            batch_size = 1

            blocks = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            out = torch.einsum('bil,oli->bol', blocks, self.weight.flatten(1, -2)) + self.bias.flatten(1, -1).unsqueeze(0)
            out = out.reshape((batch_size, self.out_channels, *self.out_shape_spatial))
            return out.squeeze(0)

        else:
            raise NotImplementedError(
                'Only 4-d (batched image-like tensors) or 3-d tensors (unbatched image-like tensors) are supported!\n'
                'nn.Unfold and nn.functional.unfold only support 4-d tensors.'
            )
