"""
Convolutional layers and helpers.
"""

import torch
import torch.nn as nn


def seperable_contraction(data: torch.Tensor, *vectors: torch.Tensor):
    r"""
    Contracts ``data`` against a weighting tensor defined by a sequence of
    one-dimensional vectors.

    Let a weighting tensor :math:`W` given by the tensor product of
    :math:`\texttt{vectors} = \{ u, v, w, \dots \}`,

    .. math::
        W = u \otimes v \otimes w \otimes \dots

    This computes the contraction of :math:`W` with ``data``:

    .. math::
        \texttt{result} = \langle W,\, \texttt{data} \rangle.

    Attention:
        This method is **unsuitable** for contractions by batched weighting tensors.

        This method is intended to be used for batched or unbatched data and
        **unbatched** weighting tensors.

        To perform contractions of **batched** data and **batched** weighting tensors,
        see :func:`seperable_contraction_batched`.

    Args:
        data (torch.Tensor):
            Tensor with shape matching the coordinate dimensions represented by
            ``vectors``.
        vectors (torch.Tensor):
            One-dimensional tensors used for successive contractions.

    Shape:
        Let ``data`` have shape ``(*out_shape, e_1, e_2, ..., e_k)``, then
        ``vectors[i]`` must be of shape ``(e_i,)``. The result will be
        of shape ``out_shape``.

    Returns:
        torch.Tensor:
            Tensor resulting from contracting ``data`` over each supplied vector.
    """
    for i, v in enumerate(vectors):
        if v.shape != (data.shape[i - len(vectors)],):
            raise ValueError(
                f'Got vectors of incompatible shape: '
                f'vector {i} of shape {(*v.shape,)}, '
                f'expected {(data.shape[i - len(vectors)],)}!'
            )

    for v in reversed(vectors):
        data = data @ v
    return data


def seperable_contraction_batched(data: torch.Tensor, *vectors: torch.Tensor):
    r"""
    Applies :func:`seperable_contraction` over a batch of tensors and vectors.

    Let a **batched** weighting tensor :math:`W` given by the tensor product of
    **batched** :math:`\texttt{vectors}_b = \{ u_b, v_b, w_b, \dots \}`,

    .. math::
        W_b = u_b \otimes v_b \otimes w_b \otimes \dots

    This computes the contraction of :math:`W` with ``data`` in **non-batch**
    dimensions:

    .. math::
        \texttt{result}_b = \langle W_b,\, \texttt{data}_b \rangle.

    Args:
        data (torch.Tensor):
            Batched tensor whose leading dimension indexes the batch.
        vectors (torch.Tensor):
            Batched one-dimensional tensors used for contraction.

    Shape:
        Let ``data`` have shape ``(*batch_shape, e_1, e_2, ..., e_k)``, then
        ``vectors[i]`` must be of shape ``(*batch_shape, e_i)``. The result will be
        of shape ``batch_shape``.

    Returns:
        torch.Tensor:
            Batched contraction result.

    See Also:
        See :func:`seperable_contraction` for a contraction of **one** (i.e.,
        unbatched) weighting tensor against ``data``.
    """
    batch_shape = data.shape[:-len(vectors)]

    for i, v in enumerate(vectors):
        if v.shape != (*batch_shape, data.shape[i - len(vectors)]):
            raise ValueError(
                f'Got vectors of shape incompatible with data: '
                f'vector {i} of shape {(*v.shape,)}, '
                f'expected {(*batch_shape, data.shape[i - len(vectors)])}!'
            )

    data = torch.vmap(
        seperable_contraction,
        in_dims=(0, *([0] * len(vectors))),
    )(
        data.reshape((-1, *data.shape[-len(vectors):])),
        *(v.reshape((-1, v.shape[-1])) for v in vectors)
    )
    data = data.reshape(batch_shape)

    return data


def get_nearest_index(x: torch.Tensor, y: torch.Tensor):
    r"""
    Get index in ``x`` nearest to ``y``.

    Specifically,

    .. math::
        \texttt{index}_i = \operatorname*{arg\,min}_j | x_j - y_i |

    Args:
        x (torch.Tensor):
            Array in which to find indices.
        y (torch.Tensor):
            Array from which to compare.
    """
    index_right = torch.searchsorted(x, y)
    index_right = index_right.clamp(max=x.numel() - 1)
    index_left = (index_right - 1).clamp(min=0)

    dist_right = torch.abs(y - x[index_right])
    dist_left = torch.abs(y - x[index_left])

    return torch.where(dist_left <= dist_right, index_left, index_right)


class DiffConvCubicBSpline(nn.Module):
    """
    Differentiable cubic B-spline convolution module.
    """

    def __init__(
        self,
        dx: torch.Tensor,
    ):
        r"""
        Applies a cubic B-spline convolution that is differentiable with respect
        to the evaluation coordinates.

        This module computes a differentiable quasi-interpolant of data evaluated
        at supplied collocation points.

        Args:
            dx (torch.Tensor):
                Tensor of grid spacings in each dimension.

        Shape:
            ``dx`` must have shape ``(n_coordinates,)`` where ``n_coordinates`` is the
            number of coordinates.

        Attributes:
            dx (torch.Tensor):
                Tensor of grid spacings in each dimension from instantiation.
            n_coordinates (int):
                Number of coordinate dimensions, inferred from :attr:`dx`.

        In general, this module computes the convolution of a smooth cubic B-spline
        kernel against data. Consider a function representing data

        .. math::
            f : \Omega \rightarrow \mathbb{R}

        on a domain :math:`\Omega` such that
        :math:`\operatorname{dim}(\Omega) = \texttt{n\_coordinates}`.

        For notational convenience, let
        :math:`N = \operatorname{dim}(\Omega) = \texttt{n\_coordinates}`.

        The module computes the convolution

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

        The discrete convolution is the full contraction of the data :math:`f` with the
        weighting tensor :math:`K`, as

        .. math::
            [k \star f](\mathbf{x}) = \langle f, K(\mathbf{x}) \rangle
                = \sum_{\mathbf{y} \in \Gamma} f(\mathbf{y})
                    \prod_{\mu=1}^N \left[ w_{\mu}(x_\mu) \right]_{y_\mu}

        This is a weighted sum over the grid, or equivalently the full tensor
        contraction of :math:`f` with :math:`K(\mathbf{x})`.

        The cubic B-spline kernel is scaled to have support radius :attr:`dx` in each
        dimension, thus a one-dimensional evaluation offset in 
        ``[-0.5 * dx, 0.5 * dx]`` can overlap only the three cells centered at indices
        ``-1``, ``0``, and ``1`` relative to the middle cell of the one-dimensional
        stencil. Therefore, data must provide the ``3 ** N`` lattice of data about
        the cell at ``x``.

        As a pedagogical note, we remark that there exist analytically straightforward
        though otherwise technically challenging extensions of this notion of discrete
        convolution to more general definitions of data and kernels, e.g., data on non-uniform
        tensor product grids, positive-dimension manifolds, or non-stationary kernels,
        but these are outside the scope of this class.
        """
        if dx.ndim != 1:
            raise ValueError  # TODO: fill in this ValueError.
        if dx.shape[0] < 1:
            # TODO: fill in this ValueError; need to explain that n_coordinates,
            # inferred from dx, must be greater than or equal to 1.
            raise ValueError

        super().__init__()
        self.dx = dx
        self.n_coordinates = dx.shape[0]

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_eval: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the tensor-product cubic B-spline convolution.

        The cubic B-spline kernel has radius equal to one grid spacing in each
        coordinate direction.

        Args:
            x (torch.Tensor):
                Tensor of coordinates of center cell in data.
            y (torch.Tensor):
                Tensor of data in ``3 ** N`` lattice about each coordinate in ``x``.
            x_eval (torch.Tensor):
                Tensor of evaluation coordinates.

        Shape:
            ``x`` must have shape ``(*batch_shape, n_coordinates)``.

            ``y`` must have shape ``(*batch_shape, 3, ..., 3)``, with one trailing
            extent-3 dimension per coordinate dimension.

            ``x_eval`` must have shape ``(*batch_shape, n_coordinates)``.

        The cubic B-spline kernel is scaled to have support radius :attr:`dx` in each
        dimension, thus a one-dimensional evaluation offset in 
        ``[-0.5 * dx, 0.5 * dx]`` can overlap only the three cells centered at indices
        ``-1``, ``0``, and ``1`` relative to the middle cell of the one-dimensional
        stencil. Therefore, ``y`` must provide the ``3 ** N`` lattice of data about
        the cell at ``x``.

        Returns:
            torch.Tensor:
                Tensor with shape ``batch_shape`` containing one convolution value per
                evaluation point.
        """
        batch_shape = y.shape[:-self.n_coordinates]
        coord_shape = y.shape[-self.n_coordinates:]

        if coord_shape != (3,) * self.n_coordinates:
            # TODO: fill in this ValueError: data do not have the correct extent for
            # the support of the kernel; need to document/explain the support of the
            # kernel and scaling of tensor lattice in all dimensions by \Delta x_i.
            raise ValueError

        if x.shape[:-1] != batch_shape or x_eval.shape[:-1] != batch_shape:
            # TODO: fill in this ValueError: coordinates do not have right batch shape.
            raise ValueError

        if x.shape[-1] != self.n_coordinates or x_eval.shape[-1] != self.n_coordinates:
            # TODO: fill in this ValueError.
            raise ValueError

        x_rel = (x_eval - x) / self.dx

        x_a = x_rel.unsqueeze(-1) + torch.arange(-1, 1 + 1) - 0.5
        x_b = x_rel.unsqueeze(-1) + torch.arange(-1, 1 + 1) + 0.5

        vectors = DiffConvCubicBSpline.integrate_b(x_a, x_b)
        vectors = vectors.permute(-2, *range(len(batch_shape)), -1)

        y = torch.flip(y, tuple(range(-self.n_coordinates, 0)))
        y = seperable_contraction_batched(y, *vectors)

        return y

    @staticmethod
    def integrate_b(
        x_a: torch.Tensor,
        x_b: torch.Tensor
    ):
        """
        Computes the integral of the cubic B-spline from ``x_a`` to ``x_b``.

        Args:
            x_a (torch.Tensor):
                Lower integration bound.
            x_b (torch.Tensor):
                Upper integration bound.

        Returns:
            torch.Tensor:
                Integral of the cubic B-spline over ``[x_a, x_b]``.
        """
        return (
            DiffConvCubicBSpline._integrate_b_half(x_b)
            - DiffConvCubicBSpline._integrate_b_half(x_a)
        )

    @staticmethod
    def _integrate_b_half(
        x: torch.Tensor
    ):
        """
        Computes the integral of the cubic B-spline from ``-inf`` to ``x``.

        Args:
            x (torch.Tensor):
                Upper integration bound.

        Returns:
            torch.Tensor:
                Integral of the cubic B-spline from ``-inf`` to ``x``.
        """
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
                    torch.where(
                        idx_3,
                        out_3,
                        torch.ones_like(x)
                    )
                )
            )
        )

    @staticmethod
    def get_nearest_index(x, y):
        """
        Alias of :func:`nnlayer.conv.get_nearest_index`.
        """
        return get_nearest_index(x, y)
