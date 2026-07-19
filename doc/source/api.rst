nnlayer API Reference
=====================

.. seealso::

   See :doc:`Aliases in nnlayer <aliases>` for the location of these classes in
   nested nnlayer namespaces.


Convolutional Layers
--------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   DiffConvCubicBSpline <api/conv.DiffConvCubicBSpline>

.. list-table::
   :header-rows: 1
   :widths: 45 55
   :width: 100%

   * - Member
     - Description
   * - :class:`nnlayer.DiffConvCubicBSpline`
     - Applies a cubic B-spline convolution to data that is differentiable with
       respect to the evaluation coordinates.


Linear Layers
-------------

.. toctree::
   :hidden:
   :maxdepth: 1

   LocallyConnected2d <api/linear.LocallyConnected2d>

.. list-table::
   :header-rows: 1
   :widths: 45 55
   :width: 100%

   * - Member
     - Description
   * - :class:`nnlayer.LocallyConnected2d`
     - Applies a local compactly-supported affine transformation to the
       incoming data, suitable for emulating a global affine transformation when
       the data are too large for a dense transformation.


Weight Layers
-------------

.. toctree::
   :hidden:
   :maxdepth: 1

   GaussianDistanceWeight <api/weight.GaussianDistanceWeight>

.. list-table::
   :header-rows: 1
   :widths: 45 55
   :width: 100%

   * - Member
     - Description
   * - :class:`nnlayer.GaussianDistanceWeight`
     - Compute the Gaussian weight across an array of coordinates to a point x.
