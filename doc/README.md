# nnlayer docs

To build the docs, in addition to installing nnlayer and its dependencies, your
environment needs `sphinx`, `furo`, `myst-nb`, `jupyter`, `jupyterlab`,
and `matplotlib`:
```sh
python -m pip install sphinx furo myst-nb jupyter jupyterlab matplotlib
```

Then, from this directory, run `make.bat html` or `make html` depending on your
system.  The documentation root can be found at `build/html/index.html`.

More advanced users are, of course, welcome to play around with doc targets.
