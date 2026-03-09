# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import inspect
import ast

project = 'paces'
copyright = '2026, R. Kevin Kessing'
author = 'R. Kevin Kessing'
with open("../paces/__init__.py") as f:
    for line in f.readlines():
        if line[:11] == "__version__":
            release = line.split("=")[-1].strip()[1:-1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "autoapi.extension", "sphinx.ext.linkcode"]

autoapi_dirs = ["../paces"]
autoapi_add_toctree_entry = False

autoapi_options = [ 'members', 'show-inheritance', 'show-module-summary',
                        'special-members', 'imported-members', ]
autoapi_python_class_content = "both"
autoapi_own_page_level = "class"

#autodoc_mock_imports = ["cupy", "cupyx", "numpy"]

templates_path = ['_templates']
exclude_patterns = []

# -- Function that links to source code --------------------------------------

def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if info["module"]:
        module = info["module"]
        mname = info["fullname"]
        class_name = None
    else:
        module, class_name, mname = find_module_and_name(info["fullname"])

    filename = "../" + module.replace(".", "/") + ".py"
    try:
        tree = ast.parse(Path(filename).read_text())
    except FileNotFoundError: # if the path is a module
        filename = filename[:-3] + "/__init__.py"
        tree = ast.parse(Path(filename).read_text())

    if class_name is None:
        res = get_function_lines(tree, mname)
    else:
        if mname is None:
            res = get_class_lines(tree, class_name)
        else:
            res = get_method_lines(tree, class_name, mname)

    url_str = filename[3:]
    if res is not None:
        start, end = res
        url_str += f"#L{start}-L{end}"
    return "https://github.com/rkevk/paces/tree/main/" + url_str


def get_function_lines(tree, function_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node.lineno, node.end_lineno
    return None # if we didn't find anything


def get_method_lines(tree, class_name, function_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child.lineno, child.end_lineno
    return None # if we didn't find anything


def get_class_lines(tree, class_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node.lineno, node.end_lineno
    return None # if we didn't find anything


def find_module_and_name(full_name):
    """Hacky way to determine the module a correctly camel-cased class resides in."""
    l = full_name.split(".")
    module = l[0]
    mname = None
    for i, x in enumerate(l[1:]):
        if x[0].isupper():
            break
        else:
            module += "." + x
    if l[-1][0].isupper():  # if the object in question is a class itself
        class_name = ".".join(l[i+1:])
    else:   # if the object in question is a method of a class
        class_name = ".".join(l[i+1:-1])
        mname = l[-1]
    return module, class_name, mname


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
