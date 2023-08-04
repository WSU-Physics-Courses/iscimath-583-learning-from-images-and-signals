# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os.path
import subprocess
from sphinx.util.fileutil import copy_asset

import mmf_setup

mmf_setup.set_path()

# import sys
# sys.path.insert(0, os.path.abspath('.'))


# This is True if we are building on Read the Docs in case we need to customize.
on_rtd = os.environ.get("READTHEDOCS") == "True"
on_cocalc = "ANACONDA2020" in os.environ

# -- Project information -----------------------------------------------------

project = "Math 583 - Learning from Signals"
copyright = "2023, Michael McNeil Forbes"
author = "Michael McNeil Forbes"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.zopeext.autointerface",
    "matplotlib.sphinxext.plot_directive",
    # From jupyterbook
    # "jupyter_book",
    # "sphinx_thebe",
    # "sphinx_external_toc",
    #
    "sphinx_jupyterbook_latex",
    #
    "sphinx_comments",  # Hypothes.is comments and annotations
    "sphinx_design",
    "sphinx_togglebutton",
    # "recommonmark",
]

# Make sure that .rst comes first or autosummary will fail.  See
# https://github.com/sphinx-doc/sphinx/issues/9891
source_suffix = {  # As of 3.7, dicts are ordered.
    ".rst": "restructuredtext",  # Make sure this is first!
    ".myst": "myst-nb",
    ".md": "myst-nb",
    # '.ipynb': 'myst-nb',  # Ignore notebooks.  Does not work.  See below.
}

# https://myst-parser.readthedocs.io/en/latest/using/syntax-optional.html
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#substitutions-with-jinja2
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    # "tasklist",
]

# https://github.com/mcmtroffaes/sphinxcontrib-bibtex
# BibTeX files
bibtex_bibfiles = [
    "macros.bib",
    "references.bib",
]

bibtex_reference_style = "author_year"

# autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = False
autosummary_imported_members = False
add_module_names = False

autodoc_default_options = {
    "member-order": "bysource",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Cache notebook output to speed generation.
# https://myst-nb.readthedocs.io/en/latest/computation/execute.html
nb_execution_mode = "cache"
nb_execution_allow_errors = True
nb_execution_timeout = 300
nbsphinx_timeout = 300  # Time in seconds; use -1 for no timeout

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"  # Default Sphinx theme
html_theme = "sphinx_book_theme"  # Theme for JupyterBook
html_logo = "_static/wsu-logo.svg"

html_theme_options = {
    #
}

# Override version number in title... not relevant for docs.
html_title = project

# html_sidebars = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "Python 3": ("https://docs.python.org/3", None),
    "matplotlib [stable]": ("https://matplotlib.org/stable/", None),
    "numpy [stable]": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sphinx": ("https://www.sphinx-doc.org/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "igraph": ("https://igraph.readthedocs.io/en/stable/api/", None),
}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
######################################################################
# Variables with course information
course_package = "math_583"


######################################################################

######################################################################
# Custom Admonitions
# https://docutils.sourceforge.io/docs/howto/rst-directives.html

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.admonitions import BaseAdmonition, Admonition
from docutils.parsers.rst.roles import set_classes
from sphinx_togglebutton import Toggle


class SolutionAdmonition(BaseAdmonition):
    required_arguments = 0
    optional_arguments = 1
    # final_argument_whitespace = True
    # has_content = True
    node_class = nodes.admonition
    # name = "Solution"
    option_spec = dict(BaseAdmonition.option_spec, show=directives.flag)
    show_all = False  #  If True, always show.  Useful for writing

    def run(self):
        set_classes(self.options)
        self.assert_has_content()
        text = "\n".join(self.content)
        admonition_node = self.node_class(text, **self.options)
        self.add_name(admonition_node)
        title_text = "Solution"
        if self.arguments:
            title_text = self.arguments[0]
        textnodes, messages = self.state.inline_text(title_text, self.lineno)
        title = nodes.title(title_text, "", *textnodes)
        title.source, title.line = self.state_machine.get_source_and_line(self.lineno)
        admonition_node += title
        admonition_node += messages
        if not "classes" in self.options:
            admonition_node["classes"].extend(
                ["admonition-" + nodes.make_id(title_text), "dropdown"]
            )
        if "show" in self.options or self.show_all:
            admonition_node["classes"].append("toggle-shown")

        self.state.nested_parse(self.content, self.content_offset, admonition_node)
        return [admonition_node]


math_defs_filename = "_static/math_defs.tex"

html_context = {
    "mathjax_defines": "",
}

mathjax3_config = {
    "loader": {"load": ["[tex]/mathtools"]},
    "tex": {"packages": {"[+]": ["mathtools"]}},
}

# Hypothes.is comments and annotations
comments_config = {"hypothesis": True}


def config_inited_handler(app, config):
    """Insert contents of `math_defs_filename` into html_context['mathjax_defines'].

    Note: this requires a customized `_template/layout.html` which inserts this content
    onto each page.
    """
    global math_defs_filename
    filename = os.path.join(
        "" if os.path.isabs(math_defs_filename) else app.confdir, math_defs_filename
    )

    defines = config.html_context.get("mathjax_defines", "").splitlines()
    try:
        with open(filename, "r") as _f:
            defines.extend(_f.readlines())
    except IOError:
        pass

    config.html_context["mathjax_defines"] = "\n".join(defines)


# Allows us to perform initialization before building the docs.  We use this to install
# the named kernel so we can keep the name in the notebooks.
def my_init(app):
    """Run `anaconda-project run init`, or the equivalent if on RtD.

    We must customize this for RtD because we trick RTD into installing everything from
    `anaconda-project.yaml` as a conda environment.  If we then run `anaconda-project
    run init` as normal, this will create a **whole new conda environment** and install
    the kernel from there.
    """
    if on_rtd:
        pass
        """
        subprocess.check_call(
            [
                "python3",
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                "math-583",
                "--display-name",
                "Python 3 (math-583)",
            ]
        )
        """
    else:
        print("Not On RTD!")
        subprocess.check_call(["make", "-C", "..", "init"])

    mathjax_offline = False
    # mathjax_offline = True
    if mathjax_offline:
        # For this to work, you need to put mathjax js files in Docs/_static/mathjax
        # Docs/_static/
        # |-- math_defs.tex
        # |-- mathjax
        #     |-- a11y
        #     |-- adaptors
        #     |-- core.js
        #     |-- input
        #      ...
        #     |-- sre
        #      ...
        #     |-- startup.js
        #     |-- tex-chtml-full.js
        #     |-- tex-chtml.js
        #     |-- tex-mml-chtml.js
        #     |-- tex-mml-svg.js
        #     |-- tex-svg-full.js
        #     |-- tex-svg.js
        #     `-- ui
        #
        # Copied from the following to put static mathjax files in place if offline:
        # https://gitlab.com/thomaswucher/sphinx-mathjax-offline/-/blob/master/sphinx-mathjax-offline/__init__.py

        ext_dir = os.path.dirname(os.path.abspath(__file__))
        mathjax_dir = ext_dir + "_static/mathjax"
        copy_asset(mathjax_dir, os.path.join(app.outdir, "_static", "mathjax"))
        app.config.mathjax_path = "mathjax/tex-chtml.js"
        app.config.mathjax_path = "mathjax/tex-svg.js"
        app.config.mathjax_path = "mathjax/tex-chtml.js?config=TeX-AMS-MML_HTMLorMML"

        # I don't know why this is needed, but if it is not turned off, then
        # "mathjax_ignore" is added to the top-level class, preventing local rendering.
        app.config.myst_update_mathjax = False


def setup(app):
    app.connect("config-inited", config_inited_handler)
    # Ignore .ipynb files
    app.registry.source_suffix.pop(".ipynb", None)
    app.add_config_value("on_rtd", on_rtd, "env")
    app.add_config_value("on_cocalc", on_cocalc, "env")
    my_init(app)
    # app.add_directive("solution", SolutionAdmonition)
    app.add_directive("solution", Toggle)
