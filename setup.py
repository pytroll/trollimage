#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, 2016 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam.Dybbroe <adam.dybbroe@smhi.se>


# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Setup for trollimage."""
import sys

from setuptools import setup
import versioneer
import numpy as np
from Cython.Build import build_ext
from Cython.Distutils import Extension

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3"]

EXTENSIONS = [
    Extension(
        'trollimage._colorspaces',
        sources=['trollimage/_colorspaces.pyx'],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
    ),
]

try:
    sys.argv.remove("--cython-coverage")
    cython_coverage = True
except ValueError:
    cython_coverage = False


cython_directives = {
    "language_level": "3",
}
define_macros = []
if cython_coverage:
    print("Enabling directives/macros for Cython coverage support")
    cython_directives.update({
        "linetrace": True,
        "profile": True,
    })
    define_macros.extend([
        ("CYTHON_TRACE", "1"),
        ("CYTHON_TRACE_NOGIL", "1"),
    ])
    for ext in EXTENSIONS:
        ext.define_macros = define_macros
        ext.cython_directives.update(cython_directives)

cmdclass = versioneer.get_cmdclass(cmdclass={"build_ext": build_ext})

with open('README.rst', 'r') as readme_file:
    long_description = readme_file.read()

setup(name="trollimage",
      version=versioneer.get_version(),
      cmdclass=cmdclass,
      description='Pytroll imaging library',
      long_description=long_description,
      author='Martin Raspaud',
      author_email='martin.raspaud@smhi.se',
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering"],
      url="https://github.com/pytroll/trollimage",
      packages=['trollimage'],
      zip_safe=False,
      install_requires=['numpy>=1.20', 'pillow'],
      python_requires='>=3.6',
      extras_require={
          'geotiff': ['rasterio>=1.0'],
          'xarray': ['xarray', 'dask[array]'],
      },
      tests_require=['xarray', 'dask[array]', 'pyproj', 'pyresample'],
      ext_modules=EXTENSIONS,
      )
