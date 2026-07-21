Tutorials
=========

:doc:`collections/tutorial/DiffConvCubicBSpline`
------------------------------------------------

:class:`nnlayer.DiffConvCubicBSpline` applies a cubic B-spline convolution that is
differentiable with respect to the evaluation coordinates.  This notebook covers the
necessary mathematical detail to understand the support of the kernel and how to use
the class, demonstrating construction and differentiation of the interpolant.

The notebook covers both 1- and 2-d cases, with the latter providing an obvious
extension to higher dimensions.  Finally, the notebook summarizes common pitfalls,
in particular computing un-optimized gradients of the interpolant and how to
do so reasonably quickly.

.. toctree::
   :hidden:
   :maxdepth: 1

   collections/tutorial/DiffConvCubicBSpline
