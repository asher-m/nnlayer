
Getting Started
===============

Here, we cover installation and tutorial notebook instructions.


Installing nnlayer
^^^^^^^^^^^^^^^^^^

From Source
~~~~~~~~~~~

After cloning the repo from
`github.com/asher-m/nnlayer <https://github.com/asher-m/nnlayer>`_,
install nnlayer:

.. code-block:: sh

   python -m pip install -e .

Use these layers as you would any other ``torch.nn.Module`` subclass.

From PyPI
~~~~~~~~~

*Coming soon!*


Tutorial Notebooks
^^^^^^^^^^^^^^^^^^

The tutorial notebooks in ``tutorial`` folder show example implementations of
many of the layers and tools in this package, as well as loosely cover some of
the analysis and numerical detail behind the construction of these layers.

To execute the notebooks, change to the project root and install the nnlayer
tutorial target:

.. code-block:: sh

   python -m pip install -e ".[tutorial]"

Then start Jupyter and enjoy!
