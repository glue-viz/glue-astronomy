# -*- coding: utf-8 -*-
#
# Glue documentation build configuration file

import os

from glue_astronomy import __version__

# -- General configuration ----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.6'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'numpydoc',
              'sphinx_automodapi.automodapi',
              'sphinx_automodapi.smart_resolver',
              'sphinxcontrib.spelling']

# Workaround for RTD where the default encoding is ASCII
if os.environ.get('READTHEDOCS') == 'True':
    import locale
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')

intersphinx_cache_limit = 10     # days to keep the cached inventories
intersphinx_mapping = {
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'glue-astronomy': ('http://docs.glueviz.org/en/latest/', None),
    'specutils': ('https://specutils.readthedocs.io/en/latest/', None),
    'regions': ('https://astropy-regions.readthedocs.io/en/latest/', None),
    'spectral-cube': ('https://spectral-cube.readthedocs.io/en/latest/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'glue-astronomy'
copyright = u'2019, Thomas Robitaille'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# The full version, including alpha/beta/rc tags.
release = __version__
# The short X.Y version.
version = '.'.join(release.split('.')[:2])

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_templates', '.eggs']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output --------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
try:  # use ReadTheDocs theme, if installed
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path(), ]
except ImportError:
    pass

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'Gluedoc'

# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'Glue.tex', u'Glue Documentation',
     u'Thomas Robitaille', 'manual'),
]

# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'glue-astronomy', u'Glue Documentation',
     [u'Thomas Robitaille'], 1)
]

# -- Options for Texinfo output -----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'glue-astronomy', u'Glue Documentation',
     u'Thomas Robitaille',
     'glue-astronomy', 'One line description of project.', 'Miscellaneous'),
]

# -- Additional options------- ------------------------------------------------

todo_include_todos = True
autoclass_content = 'both'

nitpicky = True
nitpick_ignore = [('py:class', 'object'), ('py:class', 'str'),
                  ('py:class', 'list'), ('py:obj', 'numpy array'),
                  ('py:obj', 'integer'), ('py:obj', 'Callable'),
                  ('py:obj', 'list'),
                  ('py:class', 'PyQt5.QtWidgets.QMainWindow'),
                  ('py:class', 'PyQt5.QtWidgets.QWidget'),
                  ('py:class', 'PyQt5.QtWidgets.QTextEdit'),
                  ('py:class', 'PyQt5.QtWidgets.QTabBar'),
                  ('py:class', 'PyQt5.QtWidgets.QLabel'),
                  ('py:class', 'PyQt5.QtWidgets.QComboBox'),
                  ('py:class', 'PyQt5.QtWidgets.QMessageBox'),
                  ('py:class', 'PyQt5.QtWidgets.QDialog'),
                  ('py:class', 'PyQt5.QtWidgets.QToolBar'),
                  ('py:class', 'PyQt5.QtWidgets.QStyledItemDelegate'),
                  ('py:class', 'PyQt5.QtCore.QMimeData'),
                  ('py:class', 'PyQt5.QtCore.QAbstractListModel'),
                  ('py:class', 'PyQt5.QtCore.QThread'),
                  ('py:obj', "str ('file' | 'directory' | 'label')"),
                  ('py:obj', 'function(application)'),
                  ('py:class', 'builtins.object'),
                  ('py:class', 'builtins.list'),
                  ('py:class', 'builtins.type'),
                  ('py:class', 'glue.viewers.histogram.layer_artist.HistogramLayerBase'),
                  ('py:class', 'glue.viewers.scatter.layer_artist.ScatterLayerBase'),
                  ('py:class', 'glue.viewers.image.layer_artist.ImageLayerBase'),
                  ('py:class', 'glue.viewers.image.layer_artist.RGBImageLayerBase'),
                  ('py:mod', 'glue.core'),
                  ('py:mod', 'glue.viewers'),
                  ('py:mod', 'glue.viewers.scatter'),
                  ('py:mod', 'glue.viewers.common'),
                  ('py:mod', 'glue.viewers.common.qt.mouse_mode'),
                  ('py:mod', 'glue.viewers.common.qt.toolbar_mode'),
                  ('py:mod', 'glue.dialogs.custom_component'),
                  ('py:class', 'glue.external.echo.core.HasCallbackProperties'),
                  ('py:class', 'glue.external.echo.core.CallbackProperty'),
                  ('py:class', 'glue.external.echo.selection.SelectionCallbackProperty'),
                  ('py:class', 'glue.viewers.image.state.BaseImageLayerState'),
                  ('py:class', 'glue.viewers.common.qt.data_viewer_with_state.DataViewerWithState')
              ]

viewcode_follow_imported_members = False

numpydoc_show_class_members = False
autosummary_generate = True
automodapi_toctreedirnm = 'api'

linkcheck_ignore = [r'https://www.glueviz.org.s3']
linkcheck_retries = 5
linkcheck_timeout = 10
