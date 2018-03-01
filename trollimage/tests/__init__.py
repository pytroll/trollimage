#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""The tests package.
"""

from trollimage import image, colormap
from trollimage.tests import test_image, test_colormap
import unittest
import doctest


def suite():
    """The global test suite.
    """
    mysuite = unittest.TestSuite()
    # Test the documentation strings
    mysuite.addTests(doctest.DocTestSuite(image))
    # Use the unittests also
    mysuite.addTests(test_image.suite())

    mysuite.addTests(doctest.DocTestSuite(colormap))
    mysuite.addTests(test_colormap.suite())
    
    return mysuite


def load_tests(loader, tests, pattern):
    return suite()

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
