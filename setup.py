#!/usr/bin/env python

from __future__ import print_function

import sys
from distutils.version import LooseVersion

try:
    from setuptools import setup, __version__
    assert LooseVersion(__version__) >= LooseVersion('30.3')
except (ImportError, AssertionError):
    sys.stderr.write("ERROR: setuptools 30.3 or later is required by glue-plotly\n")
    sys.exit(1)

setup(use_scm_version=True, setup_requires=['setuptools_scm'])
