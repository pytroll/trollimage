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

"""Test colormap.py
"""


import unittest
from trollimage import colormap
import numpy as np

class TestColormapClass(unittest.TestCase):
    """Test case for the colormap object.
    """
    def test_colorize(self):
        """Test colorize
        """
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        data = np.array([1, 2, 3, 4])

        channels = cm_.colorize(data)
        for i in range(3):
            self.assertTrue(np.allclose(channels[i],
                                        cm_.colors[:, i],
                                        atol=0.001))

    def test_palettize(self):
        """Test palettize
        """
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        data = np.array([1, 2, 3, 4])

        channels, colors = cm_.palettize(data)
        self.assertTrue(np.allclose(colors, cm_.colors))
        self.assertTrue(all(channels == [0, 1, 2, 3]))

    def test_set_range(self):
        """Test set_range
        """
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        cm_.set_range(0, 8)
        self.assertTrue(cm_.values[0] == 0)
        self.assertTrue(cm_.values[-1] == 8)

    def test_invert_set_range(self):
        """Test inverted set_range
        """
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        cm_.set_range(8, 0)
        self.assertTrue(cm_.values[0] == 0)
        self.assertTrue(cm_.values[-1] == 8)

    def test_reverse(self):
        """Test reverse
        """
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))
        colors = cm_.colors
        cm_.reverse()
        self.assertTrue(np.allclose(np.flipud(colors), cm_.colors)) 

    def test_add(self):
        """Test adding colormaps
        """
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        cm1 = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)))
        cm2 = colormap.Colormap((3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        cm3 = cm1 + cm2

        self.assertTrue(np.allclose(cm3.colors, cm_.colors))
        self.assertTrue(np.allclose(cm3.values, cm_.values))
        
    def test_colorbar(self):
        """Test colorbar
        """
        cm_ = colormap.Colormap((1, (1, 1, 0)),
                                (2, (0, 1, 1)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        channels = colormap.colorbar(1, 4, cm_)
        for i in range(3):
            self.assertTrue(np.allclose(channels[i],
                                        cm_.colors[:, i],
                                        atol=0.001))

    def test_palettebar(self):
        """Test colorbar
        """
        cm_ = colormap.Colormap((1, (1, 1, 0)),
                                (2, (0, 1, 1)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        channel, palette = colormap.palettebar(1, 4, cm_)

        self.assertTrue(np.allclose(channel, np.arange(4)))
        self.assertTrue(np.allclose(palette, cm_.colors))
        

def suite():
    """The suite for test_colormap.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestColormapClass))
    
    return mysuite
