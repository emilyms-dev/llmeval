"""Sphinx configuration for llmeval documentation."""

import os
import sys

# Make the package importable without installing it.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "llmeval"
author = "Emily Suh"
copyright = "2024, Emily Suh"  # noqa: A001

from llmeval import __version__  # noqa: E402

release = __version__
version = __version__

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Napoleon — Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
always_document_param_types = False

# Intersphinx — link to Python stdlib and Pydantic docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_title = f"llmeval {release}"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#4f46e5",
        "color-brand-content": "#4f46e5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#818cf8",
        "color-brand-content": "#818cf8",
    },
}

html_static_path = ["_static"]

# Don't show the full module path in class/function headings.
add_module_names = False

# Keep the source reST files out of the built HTML.
html_copy_source = False
html_show_sourcelink = False
