# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'paces'
copyright = '2026, R. Kevin Kessing'
author = 'R. Kevin Kessing'
with open("../paces/__init__.py") as f:
    for line in f.readlines():
        if line[:11] == "__version__":
            release = line.split("=")[-1].strip()[1:-1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "autoapi.extension"]

autoapi_dirs = ["../paces"]
autoapi_add_toctree_entry = False

autoapi_options = [ 'members', 'show-inheritance', 'show-module-summary',
                        'special-members', 'imported-members', ]
autoapi_python_class_content = "both"
autoapi_own_page_level = "class"

#autodoc_mock_imports = ["cupy", "cupyx", "numpy"]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
