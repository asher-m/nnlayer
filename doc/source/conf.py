# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nnlayer'
copyright = '2026, A.S. Merrill'
author = 'A.S. Merrill'
release = '0.0.0'


# -- Project root ------------------------------------------------------------

import sys
import re
import inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
    'sphinx_collections'
]

templates_path = ['templates']
exclude_patterns = []


# -- extension configuration -------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True

autodoc_typehints = 'description'
autoclass_content = 'both'
autodoc_mock_imports = ['torch']

# myst_nb
nb_execution_mode = 'auto'
myst_enable_extensions = [
    'amsmath',
    'dollarmath',
]

# sphinx_collections
collections_target = 'collections'
collections_final_clean = False
collections = {
   'tutorial': {
      'driver': 'copy_folder',
      'source': '../../tutorial/',
   }
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['static'] if (Path(__file__).parent / 'static').is_dir() else []
html_css_files = []


# -- Signature cleanup -------------------------------------------------------

_ARG_LINE = re.compile(r'^\s*([*]{0,2}[A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:')

def _signature_from_init_args(obj):
    """Build a simple class signature from the Args section of __init__."""
    init = getattr(obj, '__init__', None)
    docstring = inspect.getdoc(init)
    if not docstring:
        return None

    args = []
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()

        if stripped in {'Args:', 'Arguments:', 'Parameters:'}:
            in_args = True
            continue

        if not in_args:
            continue

        if stripped and not line.startswith((' ', '\t')):
            break

        match = _ARG_LINE.match(line)
        if match:
            args.append(match.group(1).lstrip('*'))

    return f"({', '.join(args)})" if args else None

def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    if what != 'class':
        return None

    init_args_signature = _signature_from_init_args(obj)
    if init_args_signature is None:
        return None

    return init_args_signature, return_annotation

def setup(app):
    app.connect('autodoc-process-signature', autodoc_process_signature)
