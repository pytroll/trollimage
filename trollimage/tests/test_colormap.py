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
"""Test colormap.py."""

import unittest
from trollimage import colormap
import numpy as np


class TestColormapClass(unittest.TestCase):
    """Test case for the colormap object."""

    def setUp(self):
        """Set up the test case."""
        self.colormap = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                          (2, (0.0, 1.0, 1.0)),
                                          (3, (1, 1, 1)),
                                          (4, (0, 0, 0)))

    def test_colorize_no_interpolation(self):
        """Test colorize."""
        data = np.array([1, 2, 3, 4])

        channels = self.colormap.colorize(data)
        for i in range(3):
            self.assertTrue(np.allclose(channels[i],
                                        self.colormap.colors[:, i],
                                        atol=0.001))

    def test_colorize_with_interpolation(self):
        """Test colorize."""
        data = np.array([1.5, 2.5, 3.5, 4])

        expected_channels = [np.array([0.22178232, 0.61069262, 0.50011605, 0.]),
                             np.array([1.08365532, 0.94644083, 0.50000605, 0.]),
                             np.array([0.49104964, 1.20509947, 0.49989589, 0.])]

        channels = self.colormap.colorize(data)
        for i in range(3):
            self.assertTrue(np.allclose(channels[i],
                                        expected_channels[i],
                                        atol=0.001))

    def test_colorize_dask_with_interpolation(self):
        """Test colorize dask arrays."""
        import dask.array as da
        data = da.from_array(np.array([[1.5, 2.5, 3.5, 4],
                                       [1.5, 2.5, 3.5, 4],
                                       [1.5, 2.5, 3.5, 4]]), chunks=2)

        expected_channels = [np.array([[0.22178232, 0.61069262, 0.50011605, 0.],
                                       [0.22178232, 0.61069262, 0.50011605, 0.],
                                       [0.22178232, 0.61069262, 0.50011605, 0.]]),
                             np.array([[1.08365532, 0.94644083, 0.50000605, 0.],
                                       [1.08365532, 0.94644083, 0.50000605, 0.],
                                       [1.08365532, 0.94644083, 0.50000605, 0.]]),
                             np.array([[0.49104964, 1.20509947, 0.49989589, 0.],
                                       [0.49104964, 1.20509947, 0.49989589, 0.],
                                       [0.49104964, 1.20509947, 0.49989589, 0.]])]

        channels = self.colormap.colorize(data)
        for i, expected_channel in enumerate(expected_channels):
            current_channel = channels[i, :, :]
            assert isinstance(current_channel, da.Array)
            self.assertTrue(np.allclose(current_channel.compute(),
                                        expected_channel,
                                        atol=0.001))

    def test_palettize(self):
        """Test palettize."""
        data = np.array([1, 2, 3, 4])

        channels, colors = self.colormap.palettize(data)
        self.assertTrue(np.allclose(colors, self.colormap.colors))
        self.assertTrue(all(channels == [0, 1, 2, 3]))

        cm_ = colormap.Colormap((0, (0.0, 0.0, 0.0)),
                                (1, (1.0, 1.0, 1.0)),
                                (2, (2, 2, 2)),
                                (3, (3, 3, 3)))

        data = np.arange(-1, 5)

        channels, colors = cm_.palettize(data)
        self.assertTrue(np.allclose(colors, cm_.colors))
        self.assertTrue(all(channels == [0, 0, 1, 2, 3, 3]))

        data = np.arange(-1.0, 5.0)
        data[-1] = np.nan

        channels, colors = cm_.palettize(data)
        self.assertTrue(np.allclose(colors, cm_.colors))
        self.assertTrue(all(channels == [0, 0, 1, 2, 3, 3]))

    def test_palettize_dask(self):
        """Test palettize on a dask array."""
        import dask.array as da
        data = da.from_array(np.array([[1, 2, 3, 4],
                                       [1, 2, 3, 4],
                                       [1, 2, 3, 4]]), chunks=2)
        channels, colors = self.colormap.palettize(data)
        assert isinstance(channels, da.Array)
        self.assertTrue(np.allclose(colors, self.colormap.colors))
        self.assertTrue(np.allclose(channels.compute(), [[0, 1, 2, 3],
                                                         [0, 1, 2, 3],
                                                         [0, 1, 2, 3]]))

    def test_set_range(self):
        """Test set_range."""
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        cm_.set_range(0, 8)
        self.assertTrue(cm_.values[0] == 0)
        self.assertTrue(cm_.values[-1] == 8)

    def test_invert_set_range(self):
        """Test inverted set_range."""
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        cm_.set_range(8, 0)
        self.assertTrue(cm_.values[0] == 0)
        self.assertTrue(cm_.values[-1] == 8)

    def test_reverse(self):
        """Test reverse."""
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))
        colors = cm_.colors
        cm_.reverse()
        self.assertTrue(np.allclose(np.flipud(colors), cm_.colors))

    def test_add(self):
        """Test adding colormaps."""
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
        """Test colorbar."""
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
        """Test colorbar."""
        cm_ = colormap.Colormap((1, (1, 1, 0)),
                                (2, (0, 1, 1)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        channel, palette = colormap.palettebar(1, 4, cm_)

        self.assertTrue(np.allclose(channel, np.arange(4)))
        self.assertTrue(np.allclose(palette, cm_.colors))

    def test_to_rio(self):
        """Test conversion to rasterio colormap."""
        cm_ = colormap.Colormap((1, (1, 1, 0)),
                                (2, (0, 1, 1)),
                                (3, (1, 1, 1)),
                                (4, (0, 0, 0)))

        d = cm_.to_rio()
        exp = {1: (255, 255, 0), 2: (0, 255, 255),
               3: (255, 255, 255), 4: (0, 0, 0)}

        self.assertEqual(d, exp)
