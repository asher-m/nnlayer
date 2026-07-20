"""
Convolutional layers and helpers.
"""

import torch
import torch.nn as nn


def seperable_contraction(data, *vecs):
    # d.shape == coordinate_shape
    # vecs[k].shape == (coordinate_shape[k],)
    for v in reversed(vecs):
        data = data @ v
    return data


def seperable_contraction_batched(data, vectors):
    return torch.vmap(
        seperable_contraction,
        in_dims=(0, *([0] * len(vectors))),
    )(data, *vectors)


class DiffConvCubicBSpline(nn.Module):
    def __init__(
            self,
            n_coordinates: int,
    ):
        r"""
        Applies a cubic B-spline convolution to data that is differentiable with
        respect to the evaluation coordinates.

        Qualitatively speaking, this produces a differentiable quasi-interpolant of
        the data evaluated at provided collocation points.

        Args:
            n_coordinates (int): Number of coordinates.

        In general, this method computes the convolution of a smooth cubic B-spline
        kernel against some data.  Consider some (not necessarily smooth) function
        representing data

        .. math::
            f : \Omega \rightarrow \mathbb{R}

        on a domain :math:`\Omega` such that
        :math:`\operatorname{dim}(\Omega) = \texttt{n\_coordinates}`.

        For notational convenience, let
        :math:`N = \operatorname{dim}(\Omega) = \texttt{n\_coordinates}`.

        This method computes the convolution

        .. math::
            [k \star f](\mathbf{x}) =
                \int_\Omega k(\mathbf{x}
                - \mathbf{x}') f(\mathbf{x}') d\mathbf{x}'

        where

        .. math::
            k(\mathbf{x}) = \prod_{\mu=1}^{N} b(x_\mu; \Delta x_\mu)

        for the cubic B-spline kernel :math:`b` given by

        .. math::
            b(z; r) = \begin{cases}
                \dfrac{4}{3 r} - \dfrac{8 z^2}{r^3} + \dfrac{8 |z|^3}{r^4} & |z|<\dfrac{r}{2} \\
                \dfrac{8 (r - |z|)^3}{3 r^4} & \dfrac{r}{2} \le |z| < r \\
                0 & |z| \ge r.
            \end{cases}

        Because :math:`k` is compactly supported, this is equivalent to

        .. math::
            [k \star f](\mathbf{x}) =
                \int_{\Omega \cap (\mathbf{x} - \operatorname{supp}(k))} k(\mathbf{x}
                - \mathbf{x}') f(\mathbf{x}') d\mathbf{x}'.

        In the discrete case, we consider data :math:`f` defined on a uniform Cartesian grid
        :math:`\Gamma`,

        .. math::
            \Gamma = \prod_{\mu=1}^N \Gamma_\mu \subset \Omega
                \qquad \text{where} \qquad \Gamma_\mu = \{ x_\mu^i \}_{i=1}^{N_\mu}
        
        such that each coordinate has regular spacing, 
        
        .. math::
            \Delta x_\mu =  x_\mu^{i+1} -  x_\mu^i
        
        independent of :math:`i`. The grid spacing need not be isotropic;
        that is :math:`\Delta x_\mu` need not equal :math:`\Delta x_\nu` for
        :math:`\mu \neq \nu`.

        At an evaluation point :math:`\mathbf{x}`, define the one-dimensional weighting vector
        for coordinate :math:`\mu` as

        .. math::
            w_\mu(x_\mu) = \begin{bmatrix}
                \displaystyle \int_{x_\mu^1 - \Delta x_\mu / 2}^{x_\mu^1 + \Delta x_\mu / 2} 
                    b(s - x_\mu; \Delta x_\mu) ds \\
                \displaystyle \int_{x_\mu^2 - \Delta x_\mu / 2}^{x_\mu^2 + \Delta x_\mu / 2} 
                    b(s - x_\mu; \Delta x_\mu) ds \\
                \vdots \\
                \displaystyle \int_{x_\mu^{N_\mu} - \Delta x_\mu / 2}^{x_\mu^{N_\mu} + \Delta x_\mu / 2} 
                    b(s - x_\mu; \Delta x_\mu) ds
            \end{bmatrix}.

        We then define the weighting tensor associated with the cubic B-spline kernel as

        .. math::
            K(\mathbf{x}) = \bigotimes_{\mu=1}^N w_\mu(x_\mu).

        Then, the discrete convolution is the full contraction of the data :math:`f` with the
        weighting tensor :math:`K`, as

        .. math::
            [k \star f](\mathbf{x}) = \langle f, K(\mathbf{x}) \rangle
                = \sum_{\mathbf{y} \in \Gamma} f(\mathbf{y})
                    \prod_{\mu=1}^N \left[ w_{\mu}(x_\mu) \right]_{y_\mu}

        This is a weighted sum over the grid, or equivalently the full tensor
        contraction of :math:`f` with :math:`K(\mathbf{x})`.

        As a pedagogical note, we remark that there exist analytically straightforward
        though otherwise technically challenging extensions of this notion of discrete
        convolution to more general definitions of data and kernels, e.g., data on non-uniform
        tensor product grids, positive-dimension manifolds, or non-stationary kernels,
        but these are outside the scope of this class.
        """
        super().__init__()
        self.n_coordinates = n_coordinates

    @staticmethod
    def K(
        data: torch.Tensor,
        xi: torch.Tensor,
        yi: torch.Tensor,
        ti: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute convolution of data with tensor product cubic B-spline kernel of radius equal to cell diameter.

        Arguments:
            data:       Tensor of shape (B, C, N, M) of data values about (t, x, y).
            xi:         Tensor of shape (B,) of x index translation of kernel relative to data center.
            yi:         Tensor of shape (B,) of y index translation of kernel relative to data center.
            ti:         Tensor of shape (B,) of t index translation of kernel relative to data center.
        """
        b, c, n, m = data.shape
        if c != 3 or n != 3 or m != 3:
            raise ValueError(f'Expected (C, N, M) = (3, 3, 3), got {c}!')
        
        bx, = xi.shape
        by, = yi.shape
        bt, = ti.shape
        if bx != b or by != b or bt != b:
            raise ValueError

        if torch.any(torch.abs(xi) > 0.5) or torch.any(torch.abs(yi) > 0.5) or torch.any(torch.abs(ti) > 0.5):
            raise ValueError(f'Expected all(abs(xi, yi, ti) < (0.5, 0.5, 0.5))!')

        return DiffConvCubicBSpline._K_impl(data, xi, yi, ti)

    @staticmethod
    def _K_impl(
        data: torch.Tensor,
        xi: torch.Tensor,
        yi: torch.Tensor,
        ti: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the spline convolution without input validation so it can be used under torch.func transforms.
        """

        dtype = data.dtype
        device = data.device
        coord = torch.arange(3, dtype=dtype, device=device)[None, :]

        ta = coord - 1 - 1/2 - ti[:, None]
        tb = coord - 1 + 1/2 - ti[:, None]
        xa = coord - 1 - 1/2 - xi[:, None]
        xb = coord - 1 + 1/2 - xi[:, None]
        ya = coord - 1 - 1/2 - yi[:, None]
        yb = coord - 1 + 1/2 - yi[:, None]

        wt = DiffConvCubicBSpline.b(ta, tb)
        wx = DiffConvCubicBSpline.b(xa, xb)
        wy = DiffConvCubicBSpline.b(ya, yb)

        w = wt[:, :, None, None] * wx[:, None, :, None] * wy[:, None, None, :]

        return torch.einsum('bcnm,bcnm->b', data, w)

    @staticmethod
    def b(
        x_a: torch.Tensor,
        x_b: torch.Tensor
    ):
        """
        Compute the intergral of the cubic B-spline from x_a to x_b.
        """
        return DiffConvCubicBSpline._b_half(x_b) - DiffConvCubicBSpline._b_half(x_a)

    @staticmethod
    def _b_half(
        x: torch.Tensor
    ):
        """
        Compute the integral of the cubic B-spline from -inf to x.
        """
        if x.ndim > 2:
            raise ValueError(f'Expected b.ndim < 3, got {x.ndim}!')

        idx_0 = x < -1
        idx_1 = torch.logical_and(-1 <= x, x < -0.5)
        idx_2 = torch.logical_and(-0.5 <= x, x < 0.5)
        idx_3 = torch.logical_and(0.5 <= x, x < 1.)

        out_1 = -2/3 * x**4 + (8/3) * x**3 - 4 * x**2 + (8/3) * x + (4/3) * torch.clamp(x, max=0)**4 + 8 * torch.clamp(x, max=0)**2 + 2/3
        out_2 = 1 / 24 + 2 * x**4 - 8/3 * x**3 + (4/3) * x - 4 * torch.clamp(x, max=0)**4 + 11/24
        out_3 = 23 / 24 - 2/3 * x**4 + (8/3) * x**3 - 4 * x**2 + (8/3) * x - 5/8

        return torch.where(
            idx_0,
            torch.zeros_like(x),
            torch.where(
                idx_1,
                out_1,
                torch.where(
                    idx_2,
                    out_2,
                    torch.where(idx_3, out_3, torch.ones_like(x))
                )
            )
        )
