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
from typing import Any

from setuptools import setup, find_packages
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

cython_directives: dict[str, Any] = {
    "language_level": "3",
}


class CythonCoverageBuildExtCommand(build_ext):
    """Simple command extension to add Cython coverage flag.

    With this class included in the build we are able to pass
    ``--cython-coverage`` to compile Cython modules with flags necessary to
    report test coverage.

    """

    user_options = build_ext.user_options + [
        ('cython-coverage', None, None),
    ]

    def initialize_options(self):
        """Initialize command line flag options to default values."""
        super().initialize_options()
        self.cython_coverage = False  # noqa

    def run(self):
        """Build extensions and handle cython coverage flags."""
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        if self.cython_coverage:
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
        super().run()


cmdclass = versioneer.get_cmdclass(cmdclass={"build_ext": CythonCoverageBuildExtCommand})

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
      packages=find_packages(),
      zip_safe=False,
      install_requires=['numpy>=1.20', 'pillow'],
      python_requires='>=3.9',
      extras_require={
          'geotiff': ['rasterio>=1.0'],
          'xarray': ['xarray', 'dask[array]'],
      },
      tests_require=['xarray', 'dask[array]', 'pyproj', 'pyresample'],
      ext_modules=EXTENSIONS,
      )
