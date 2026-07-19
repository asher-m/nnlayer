Documentation
=============

The documentation on this site covers the analytic basis as well as detailed
numerical overview of the construction of these layers.

To build the docs, change to the project root install the nnlayer doc target:

.. code-block:: sh

   python -m pip install -e ".[doc]"

Then, from the ``doc`` directory, run ``make.bat html`` or ``make html`` depending
on your system.  The documentation root can be found at ``build/html/index.html``.
