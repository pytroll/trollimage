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

from setuptools import setup
import versioneer

with open('README.rst', 'r') as readme_file:
    long_description = readme_file.read()

setup(name="trollimage",
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Pytroll imaging library',
      long_description=long_description,
      author='Martin Raspaud',
      author_email='martin.raspaud@smhi.se',
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License v3 " +
                   "or later (GPLv3+)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering"],
      url="https://github.com/pytroll/trollimage",
      packages=['trollimage'],
      zip_safe=False,
      install_requires=['numpy >=1.13', 'pillow'],
      python_requires='>=3.6',
      extras_require={
          'geotiff': ['rasterio'],
          'xarray': ['xarray', 'dask[array]'],
      },
      tests_require=['xarray', 'dask[array]'],
      )
