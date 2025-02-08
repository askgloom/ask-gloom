"""
Sphinx configuration file for Ask-Gloom documentation.
"""

import os
import sys
from datetime import datetime

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'Ask-Gloom'
copyright = f'{datetime.now().year}, Your Name'
author = 'Your Name'

# The full version, including alpha/beta/rc tags
release = '0.1.0'
version = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx_rtd_theme',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# Custom sidebar templates
html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
        'navigation.html',
    ]
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/transformers/master/', None),
}

# autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# todo configuration
todo_include_todos = True
todo_emit_warnings = True

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'amsmath',
    'html_image',
]

# -- Custom setup function -------------------------------------------------

def setup(app):
    """Custom setup function for Sphinx documentation."""
    # Add custom CSS
    app.add_css_file('custom.css')
    
    # Add custom JavaScript
    app.add_js_file('custom.js')
    
    # Add custom roles and directives if needed
    # app.add_role('custom-role', custom_role_function)