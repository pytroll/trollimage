#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2021 trollimage developers
#
# This file is part of trollimage.
#
# trollimage is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# trollimage is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with trollimage.  If not, see <http://www.gnu.org/licenses/>.
"""Module for testing the image and xrimage modules."""
import os
import random
import sys
import tempfile
import unittest
from unittest import mock
from collections import OrderedDict
from tempfile import NamedTemporaryFile

import dask.array as da
import numpy as np
import xarray as xr
import rasterio as rio
import pytest

from trollimage import image, xrimage
from trollimage.colormap import Colormap, brbg
from trollimage._xrimage_rasterio import RIODataset
from .utils import assert_maximum_dask_computes


EPSILON = 0.0001


class TestEmptyImage(unittest.TestCase):
    """Class for testing the mpop.imageo.image module."""

    def setUp(self):
        """Set up the test case."""
        self.img = image.Image()
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_shape(self):
        """Shape of an empty image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertEqual(self.img.shape, (0, 0))
        self.img.convert(oldmode)

    def test_is_empty(self):
        """Test if an image is empty."""
        self.assertEqual(self.img.is_empty(), True)

    def test_clip(self):
        """Clip an empty image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertEqual(self.img.channels, [])
        self.img.convert(oldmode)

    def test_convert(self):
        """Convert an empty image."""
        for mode1 in self.modes:
            for mode2 in self.modes:
                self.img.convert(mode1)
                self.assertEqual(self.img.mode, mode1)
                self.assertEqual(self.img.channels, [])
                self.img.convert(mode2)
                self.assertEqual(self.img.mode, mode2)
                self.assertEqual(self.img.channels, [])
        while True:
            randstr = random_string(random.choice(range(1, 7)))
            if randstr not in self.modes:
                break
        self.assertRaises(ValueError, self.img.convert, randstr)

    def test_stretch(self):
        """Stretch an empty image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.stretch()
            self.assertEqual(self.img.channels, [])
            self.img.stretch("linear")
            self.assertEqual(self.img.channels, [])
            self.img.stretch("histogram")
            self.assertEqual(self.img.channels, [])
            self.img.stretch("crude")
            self.assertEqual(self.img.channels, [])
            self.img.stretch((0.05, 0.05))
            self.assertEqual(self.img.channels, [])
            self.assertRaises(ValueError, self.img.stretch, (0.05, 0.05, 0.05))

            # Generate a random string
            while True:
                testmode = random_string(random.choice(range(1, 7)))
                if testmode not in self.modes:
                    break

            self.assertRaises(ValueError, self.img.stretch, testmode)
            self.assertRaises(TypeError, self.img.stretch, 1)
        self.img.convert(oldmode)

    def test_gamma(self):
        """Gamma correction on an empty image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            # input a single value
            self.img.gamma()
            self.assertEqual(self.img.channels, [])
            self.img.gamma(0.5)
            self.assertEqual(self.img.channels, [])
            self.img.gamma(1)
            self.assertEqual(self.img.channels, [])
            self.img.gamma(1.5)
            self.assertEqual(self.img.channels, [])

            # input a tuple
            self.assertRaises(ValueError, self.img.gamma, list(range(10)))
            self.assertRaises(ValueError, self.img.gamma, (0.2, 3.5))

        self.img.convert(oldmode)

    def test_invert(self):
        """Invert an empty image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.invert()
            self.assertEqual(self.img.channels, [])
            self.img.invert(True)
            self.assertEqual(self.img.channels, [])
            self.assertRaises(ValueError, self.img.invert, [True, False])
            self.assertRaises(ValueError, self.img.invert,
                              [True, False, True, False,
                               True, False, True, False])
        self.img.convert(oldmode)

    def test_pil_image(self):
        """Return an empty PIL image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            if mode == "YCbCrA":
                self.assertRaises(ValueError, self.img.pil_image)
            elif mode == "YCbCr":
                continue
            else:
                pilimg = self.img.pil_image()
                self.assertEqual(pilimg.size, (0, 0))
        self.img.convert(oldmode)

    def test_putalpha(self):
        """Add an alpha channel to en empty image."""
        # Putting alpha channel to an empty image should not do anything except
        # change the mode if necessary.
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.putalpha(np.array([]))
            self.assertEqual(self.img.channels, [])
            if mode.endswith("A"):
                self.assertEqual(self.img.mode, mode)
            else:
                self.assertEqual(self.img.mode, mode + "A")

            self.img.convert(oldmode)

            self.img.convert(mode)
            self.assertRaises(ValueError, self.img.putalpha,
                              np.random.rand(3, 2))

        self.img.convert(oldmode)

    def test_save(self):
        """Save an empty image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertRaises(IOError, self.img.save, "test.png")

        self.img.convert(oldmode)

    def test_replace_luminance(self):
        """Replace luminance in an empty image."""
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.replace_luminance([])
            self.assertEqual(self.img.mode, mode)
            self.assertEqual(self.img.channels, [])
            self.assertEqual(self.img.shape, (0, 0))
        self.img.convert(oldmode)

    def test_resize(self):
        """Resize an empty image."""
        self.assertRaises(ValueError, self.img.resize, (10, 10))

    def test_merge(self):
        """Merging of an empty image with another."""
        newimg = image.Image()
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2, 3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)


class TestImageCreation(unittest.TestCase):
    """Class for testing the mpop.imageo.image module."""

    def setUp(self):
        """Set up the test case."""
        self.img = {}
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]
        self.modes_len = [1, 2, 3, 4, 3, 4, 1, 2]

    def test_creation(self):
        """Test creation of an image."""
        self.assertRaises(TypeError, image.Image,
                          channels=random.randint(1, 1000))
        self.assertRaises(TypeError, image.Image,
                          channels=random.random())
        self.assertRaises(TypeError, image.Image,
                          channels=random_string(random.randint(1, 10)))

        chs = [np.random.rand(random.randint(1, 10), random.randint(1, 10)),
               np.random.rand(random.randint(1, 10), random.randint(1, 10)),
               np.random.rand(random.randint(1, 10), random.randint(1, 10)),
               np.random.rand(random.randint(1, 10), random.randint(1, 10))]

        self.assertRaises(ValueError, image.Image, channels=chs)

        one_channel = np.random.rand(random.randint(1, 10),
                                     random.randint(1, 10))

        i = 0

        for mode in self.modes:
            # Empty image, no channels
            self.img[mode] = image.Image(mode=mode)
            self.assertEqual(self.img[mode].channels, [])

            # Empty image, no channels, fill value

            self.img[mode] = image.Image(mode=mode, fill_value=0)
            self.assertEqual(self.img[mode].channels, [])

            # Empty image, no channels, fill value, wrong color_range

            self.assertRaises(ValueError,
                              image.Image,
                              mode=mode,
                              fill_value=0,
                              color_range=((0, (1, 2))))

            self.assertRaises(ValueError,
                              image.Image,
                              mode=mode,
                              fill_value=0,
                              color_range=((0, 0), (1, 2), (0, 0),
                                           (1, 2), (0, 0), (1, 2)))

            # Regular image, too many channels

            self.assertRaises(ValueError, image.Image,
                              channels=([one_channel] *
                                        (self.modes_len[i] + 1)),
                              mode=mode)

            # Regular image, not enough channels

            self.assertRaises(ValueError, image.Image,
                              channels=([one_channel] *
                                        (self.modes_len[i] - 1)),
                              mode=mode)

            # Regular image, channels

            self.img[mode] = image.Image(channels=([one_channel] *
                                                   (self.modes_len[i])),
                                         mode=mode)

            for nb_chan in range(self.modes_len[i]):
                self.assertTrue(np.all(self.img[mode].channels[nb_chan] ==
                                       one_channel))
                self.assertTrue(isinstance(self.img[mode].channels[nb_chan],
                                           np.ma.core.MaskedArray))

            i = i + 1


class TestRegularImage(unittest.TestCase):
    """Class for testing the image module."""

    def setUp(self):
        """Set up the test case."""
        one_channel = np.random.rand(random.randint(1, 10),
                                     random.randint(1, 10))
        self.rand_img = image.Image(channels=[one_channel] * 3,
                                    mode="RGB")
        self.rand_img2 = image.Image(channels=[one_channel] * 3,
                                     mode="RGB",
                                     fill_value=(0, 0, 0))

        two_channel = np.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]])
        self.img = image.Image(channels=[two_channel] * 3,
                               mode="RGB")

        self.flat_channel = [[1, 1, 1], [1, 1, 1]]
        self.flat_img = image.Image(channels=[self.flat_channel],
                                    mode="L",
                                    fill_value=0)

        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]
        self.modes_len = [1, 2, 3, 4, 3, 4, 1, 2]

        # create an unusable directory for permission error checking

        self.tempdir = tempfile.mkdtemp()
        os.chmod(self.tempdir, 0o444)

    def test_shape(self):
        """Shape of an image."""
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            self.assertEqual(self.img.shape, (2, 3))
        self.img.convert(oldmode)

    def test_is_empty(self):
        """Test if an image is empty."""
        self.assertEqual(self.img.is_empty(), False)

    def test_clip(self):
        """Clip an image."""
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            for chn in self.img.channels:
                self.assertTrue(chn.max() <= 1.0)
                self.assertTrue(chn.max() >= 0.0)
        self.img.convert(oldmode)

    def test_convert(self):
        """Convert an image."""
        i = 0
        for mode1 in self.modes:
            j = 0
            for mode2 in self.modes:
                self.img.convert(mode1)
                self.assertEqual(self.img.mode, mode1)
                self.assertEqual(len(self.img.channels),
                                 self.modes_len[i])
                self.img.convert(mode2)
                self.assertEqual(self.img.mode, mode2)
                self.assertEqual(len(self.img.channels),
                                 self.modes_len[j])

                self.rand_img2.convert(mode1)
                self.assertEqual(self.rand_img2.mode, mode1)
                self.assertEqual(len(self.rand_img2.channels),
                                 self.modes_len[i])
                if mode1 not in ["P", "PA"]:
                    self.assertEqual(len(self.rand_img2.fill_value),
                                     self.modes_len[i])
                self.rand_img2.convert(mode2)
                self.assertEqual(self.rand_img2.mode, mode2)
                self.assertEqual(len(self.rand_img2.channels),
                                 self.modes_len[j])
                if mode2 not in ["P", "PA"]:
                    self.assertEqual(len(self.rand_img2.fill_value),
                                     self.modes_len[j])
                j = j + 1
            i = i + 1
        while True:
            randstr = random_string(random.choice(range(1, 7)))
            if randstr not in self.modes:
                break
        self.assertRaises(ValueError, self.img.convert, randstr)

    def test_stretch(self):
        """Stretch an image."""
        oldmode = self.img.mode

        for mode in "L":
            self.img.convert(mode)
            old_channels = []
            for chn in self.img.channels:
                old_channels.append(chn)

            linear = np.array([[0., 1.00048852, 1.00048852],
                               [1.00048852, 0.50024426, 0.50024426]])
            crude = np.array([[0, 1, 1], [1, 0.5, 0.5]])
            histo = np.array([[0.0, 0.99951171875, 0.99951171875],
                              [0.99951171875, 0.39990234375, 0.39990234375]])
            self.img.stretch()
            self.assertTrue(np.all((self.img.channels[0] - crude) < EPSILON))
            self.img.stretch("linear")
            self.assertTrue(np.all((self.img.channels[0] - linear) < EPSILON))
            self.img.stretch("crude")
            self.assertTrue(np.all((self.img.channels[0] - crude) < EPSILON))
            self.img.stretch("histogram")
            self.assertTrue(
                np.all(np.abs(self.img.channels[0] - histo) < EPSILON))
            self.img.stretch((0.05, 0.05))
            self.assertTrue(np.all((self.img.channels[0] - linear) < EPSILON))
            self.assertRaises(ValueError, self.img.stretch, (0.05, 0.05, 0.05))

            # Generate a random string
            while True:
                testmode = random_string(random.choice(range(1, 7)))
                if testmode not in self.modes:
                    break

            self.assertRaises(ValueError, self.img.stretch, testmode)
            self.assertRaises(TypeError, self.img.stretch, 1)

            self.img.channels = old_channels

        self.img.convert(oldmode)

    def test_gamma(self):
        """Gamma correction on an image."""
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)

            old_channels = []
            for chn in self.img.channels:
                old_channels.append(chn)

            # input a single value
            self.img.gamma()
            for i in range(len(self.img.channels)):
                self.assertTrue(np.all(self.img.channels[i] == old_channels[i]))
            self.img.gamma(0.5)
            for i in range(len(self.img.channels)):
                self.assertTrue(np.all(self.img.channels[i] -
                                       old_channels[i] ** 2 < EPSILON))
            self.img.gamma(1)
            for i in range(len(self.img.channels)):
                self.assertTrue(np.all(self.img.channels[i] -
                                       old_channels[i] ** 2 < EPSILON))

            # self.img.gamma(2)
            # for i in range(len(self.img.channels)):
            #     print self.img.channels[i]
            #     print old_channels[i]
            #     self.assertTrue(np.all(np.abs(self.img.channels[i] -
            #                                old_channels[i]) < EPSILON))

            # input a tuple
            self.assertRaises(ValueError, self.img.gamma, list(range(10)))
            self.assertRaises(
                ValueError, self.img.gamma, (0.2, 3., 8., 1., 9.))

        self.img.convert(oldmode)

    def test_invert(self):
        """Invert an image."""
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            old_channels = []
            for chn in self.img.channels:
                old_channels.append(chn)
            self.img.invert()
            for i in range(len(self.img.channels)):
                self.assertTrue(np.all(self.img.channels[i] ==
                                       1 - old_channels[i]))
            self.img.invert(True)
            for i in range(len(self.img.channels)):
                self.assertTrue(np.all(self.img.channels[i] -
                                       old_channels[i] < EPSILON))
            self.assertRaises(ValueError, self.img.invert,
                              [True, False, True, False,
                               True, False, True, False])
        self.img.convert(oldmode)

    def test_pil_image(self):
        """Return an PIL image."""
        # FIXME: Should test on palette images
        oldmode = self.img.mode
        for mode in self.modes:
            if (mode == "YCbCr" or
                    mode == "YCbCrA" or
                    mode == "P" or
                    mode == "PA"):
                continue
            self.img.convert(mode)
            pilimg = self.img.pil_image()
            self.assertEqual(pilimg.size, (3, 2))
        self.img.convert(oldmode)

    def test_putalpha(self):
        """Add an alpha channel."""
        # Putting alpha channel to an image should not do anything except
        # change the mode if necessary.
        oldmode = self.img.mode
        alpha = np.array(np.random.rand(2, 3))
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            self.img.putalpha(alpha)
            self.assertTrue(np.all(self.img.channels[-1] == alpha))
            if mode.endswith("A"):
                self.assertEqual(self.img.mode, mode)
            else:
                self.assertEqual(self.img.mode, mode + "A")

            self.img.convert(oldmode)

            self.img.convert(mode)
            self.assertRaises(ValueError,
                              self.img.putalpha,
                              np.random.rand(4, 5))

        self.img.convert(oldmode)

    @unittest.skipIf(sys.platform.startswith('win'),
                     "Read-only tmp dir not working under Windows")
    def test_save(self):
        """Save an image."""
        oldmode = self.img.mode
        for mode in self.modes:
            if (mode == "YCbCr" or
                    mode == "YCbCrA" or
                    mode == "P" or
                    mode == "PA"):
                continue
            self.img.convert(mode)
            self.img.save("test.png")
            self.assertTrue(os.path.exists("test.png"))
            os.remove("test.png")

            # permissions
            self.assertRaises(IOError,
                              self.img.save,
                              os.path.join(self.tempdir, "test.png"))

        self.img.convert(oldmode)

    @unittest.skipIf(sys.platform.startswith('win'),
                     "Read-only tmp dir not working under Windows")
    def test_save_jpeg(self):
        """Save a jpeg image."""
        oldmode = self.img.mode
        self.img.convert('L')
        self.img.save("test.jpg")
        self.assertTrue(os.path.exists("test.jpg"))
        os.remove("test.jpg")

        # permissions
        self.assertRaises(IOError,
                          self.img.save,
                          os.path.join(self.tempdir, "test.jpg"))

        self.img.convert(oldmode)

    def test_replace_luminance(self):
        """Replace luminance in an image."""
        oldmode = self.img.mode
        for mode in self.modes:
            if (mode == "P" or
                    mode == "PA"):
                continue
            self.img.convert(mode)
            luma = np.ma.array([[0, 0.5, 0.5],
                                [0.5, 0.25, 0.25]])
            self.img.replace_luminance(luma)
            self.assertEqual(self.img.mode, mode)
            if (self.img.mode.endswith("A")):
                chans = self.img.channels[:-1]
            else:
                chans = self.img.channels
            for chn in chans:
                self.assertTrue(np.all(chn - luma < EPSILON))
        self.img.convert(oldmode)

    def test_resize(self):
        """Resize an image."""
        self.img.resize((6, 6))
        res = np.array([[0, 0, 0.5, 0.5, 0.5, 0.5],
                        [0, 0, 0.5, 0.5, 0.5, 0.5],
                        [0, 0, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
                        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
                        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]])
        self.assertTrue(np.all(res == self.img.channels[0]))
        self.img.resize((2, 3))
        res = np.array([[0, 0.5, 0.5],
                        [0.5, 0.25, 0.25]])
        self.assertTrue(np.all(res == self.img.channels[0]))

    def test_merge(self):
        """Merging of an image with another."""
        newimg = image.Image()
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2, 3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.ma.array([[1, 2, 3], [4, 5, 6]],
                                         mask=[[1, 0, 0], [1, 1, 0]]),
                             mode="L")

        self.img.convert("L")
        newimg.merge(self.img)
        self.assertTrue(np.all(np.abs(newimg.channels[0] -
                                      np.array([[0, 2, 3], [0.5, 0.25, 6]])) <
                               EPSILON))

    def tearDown(self):
        """Clean up the mess."""
        os.chmod(self.tempdir, 0o777)
        os.rmdir(self.tempdir)


class TestFlatImage(unittest.TestCase):
    """Test a flat image, ie an image where min == max."""

    def setUp(self):
        """Set up the test case."""
        channel = np.ma.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]],
                              mask=[[1, 1, 1], [1, 1, 0]])
        self.img = image.Image(channels=[channel] * 3,
                               mode="RGB")
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_stretch(self):
        """Stretch a flat image."""
        self.img.stretch()
        self.assertTrue(self.img.channels[0].shape == (2, 3) and
                        np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.stretch("crude")
        self.assertTrue(self.img.channels[0].shape == (2, 3) and
                        np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.crude_stretch(1, 2)
        self.assertTrue(self.img.channels[0].shape == (2, 3) and
                        np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.stretch("linear")
        self.assertTrue(self.img.channels[0].shape == (2, 3) and
                        np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.stretch("histogram")
        self.assertTrue(self.img.channels[0].shape == (2, 3) and
                        np.ma.count_masked(self.img.channels[0]) == 5)


class TestNoDataImage(unittest.TestCase):
    """Test an image filled with no data."""

    def setUp(self):
        """Set up the test case."""
        channel = np.ma.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]],
                              mask=[[1, 1, 1], [1, 1, 1]])
        self.img = image.Image(channels=[channel] * 3,
                               mode="RGB")
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_stretch(self):
        """Stretch a no data image."""
        self.img.stretch()
        self.assertTrue(self.img.channels[0].shape == (2, 3))
        self.img.stretch("crude")
        self.assertTrue(self.img.channels[0].shape == (2, 3))
        self.img.crude_stretch(1, 2)
        self.assertTrue(self.img.channels[0].shape == (2, 3))
        self.img.stretch("linear")
        self.assertTrue(self.img.channels[0].shape == (2, 3))
        self.img.stretch("histogram")
        self.assertTrue(self.img.channels[0].shape == (2, 3))


def random_string(length,
                  choices="abcdefghijklmnopqrstuvwxyz"
                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Generate a random string with elements from *set* of the specified *length*."""
    return "".join([random.choice(choices)
                    for dummy in range(length)])


class TestXRImage:
    """Test XRImage objects."""

    def test_init(self):
        """Test object initialization."""
        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=['y', 'x'])
        img = xrimage.XRImage(data)
        assert img.mode == 'L'

        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]])
        img = xrimage.XRImage(data)
        assert img.mode == 'L'
        assert img.data.dims == ('bands', 'y', 'x')

        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=['x', 'y_2'])
        img = xrimage.XRImage(data)
        assert img.mode == 'L'
        assert img.data.dims == ('bands', 'x', 'y')

        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=['x_2', 'y'])
        img = xrimage.XRImage(data)
        assert img.mode == 'L'
        assert img.data.dims == ('bands', 'x', 'y')

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert img.mode == 'RGB'

        data = xr.DataArray(np.arange(100).reshape(5, 5, 4), dims=[
            'y', 'x', 'bands'], coords={'bands': ['Y', 'Cb', 'Cr', 'A']})
        img = xrimage.XRImage(data)
        assert img.mode == 'YCbCrA'

    def test_init_writability(self):
        """Test data is writable after init.

        Xarray >0.15 makes data read-only after expand_dims.

        """
        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=['y', 'x'])
        img = xrimage.XRImage(data)
        assert img.mode == 'L'
        n_arr = np.asarray(img.data)
        # if this succeeds then its writable
        n_arr[n_arr == 0.5] = 1

    def test_regression_double_format_save(self):
        """Test that double format information isn't passed to save."""
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3) / 74., dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        with mock.patch.object(xrimage.XRImage, 'pil_save') as pil_save:
            img = xrimage.XRImage(data)

            img.save(filename='bla.png', fformat='png', format='png')
            assert 'format' not in pil_save.call_args_list[0][1]

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_rgb_save(self):
        """Test saving RGB/A data to simple image formats."""
        from dask.delayed import Delayed

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3) / 74., dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = (np.arange(75.).reshape(5, 5, 3) / 74. * 255).round()
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)  # completely opaque

        data = data.where(data > (10 / 74.0))
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)

        # dask delayed save
        with NamedTemporaryFile(suffix='.png') as tmp:
            delay = img.save(tmp.name, compute=False)
            assert isinstance(delay, Delayed)
            delay.compute()

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_single_band_jpeg(self):
        """Test saving single band to jpeg formats."""
        # Single band image
        data = np.arange(75).reshape(15, 5, 1) / 74.
        data[-1, -1, 0] = np.nan
        data = xr.DataArray(data, dims=[
            'y', 'x', 'bands'], coords={'bands': ['L']})
        # Single band image to JPEG
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.jpg') as tmp:
            img.save(tmp.name, fill_value=0)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (1, 15, 5)
            # can't check data accuracy because jpeg compression will
            # change the values

        # Jpeg fails without fill value (no alpha handling)
        with NamedTemporaryFile(suffix='.jpg') as tmp:
            # make sure fill_value is mentioned in the error message
            with pytest.raises(OSError, match=r".*fill_value.*"):
                img.save(tmp.name)

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_single_band_png(self):
        """Test saving single band images to simple image formats."""
        # Single band image
        data = np.arange(75).reshape(15, 5, 1) / 74.
        data[-1, -1, 0] = np.nan
        data = xr.DataArray(data, dims=[
            'y', 'x', 'bands'], coords={'bands': ['L']})
        # Single band image to JPEG
        img = xrimage.XRImage(data)

        # Single band image to PNG - min fill (check fill value scaling)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name, fill_value=0)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (1, 15, 5)
            exp = (np.arange(75.).reshape(1, 15, 5) / 74. * 254 + 1).round()
            exp[0, -1, -1] = 0
            np.testing.assert_allclose(file_data, exp)

        # Single band image to PNG - max fill (check fill value scaling)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name, fill_value=255)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (1, 15, 5)
            exp = (np.arange(75.).reshape(1, 15, 5) / 74. * 254).round()
            exp[0, -1, -1] = 255
            np.testing.assert_allclose(file_data, exp)

        # As PNG that support alpha channel
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (2, 15, 5)
            # bad value should be transparent in alpha channel
            assert file_data[1, -1, -1] == 0
            # all other pixels should be opaque
            assert file_data[1, 0, 0] == 255

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_palettes(self):
        """Test saving paletted images to simple image formats."""
        # Single band image palettized
        from trollimage.colormap import brbg, Colormap
        data = xr.DataArray(np.arange(75).reshape(15, 5, 1) / 74., dims=[
            'y', 'x', 'bands'], coords={'bands': ['L']})
        img = xrimage.XRImage(data)
        img.palettize(brbg)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)
        img = xrimage.XRImage(data)
        # RGBA colormap
        bw = Colormap(
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 0.5)),
        )

        img.palettize(bw)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_float(self):
        """Test saving geotiffs when input data is float."""
        # numpy array image - scale to 0 to 1 first
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3) / 75.,
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = (np.arange(75.).reshape(5, 5, 3) / 75. * 255).round()
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)  # completely opaque

        data = xr.DataArray(da.from_array(np.arange(75.).reshape(5, 5, 3) / 75., chunks=5),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        # Regular default save
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = (np.arange(75.).reshape(5, 5, 3) / 75. * 255).round()
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)  # completely opaque

        # with NaNs
        data = data.where(data > 10. / 75.)
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75.).reshape(5, 5, 3) / 75.
            exp[exp <= 10. / 75.] = 0  # numpy converts NaNs to 0s
            exp = (exp * 255).round()
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            is_null = (exp == 0).all(axis=2)
            np.testing.assert_allclose(file_data[3][~is_null], 255)  # completely opaque
            np.testing.assert_allclose(file_data[3][is_null], 0)  # completely transparent

        # with fill value
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, fill_value=128)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (3, 5, 5)  # no alpha band
            exp = np.arange(75.).reshape(5, 5, 3) / 75.
            exp2 = (exp * 255).round()
            exp2[exp <= 10. / 75.] = 128
            np.testing.assert_allclose(file_data[0], exp2[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp2[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp2[:, :, 2])

        # float type - floats can't have alpha channel
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, dtype=np.float32)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (3, 5, 5)  # no alpha band
            exp = np.arange(75.).reshape(5, 5, 3) / 75.
            # fill value is forced to 0
            exp[exp <= 10. / 75.] = 0
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])

        # float type with NaN fill value
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, dtype=np.float32, fill_value=np.nan)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (3, 5, 5)  # no alpha band
            exp = np.arange(75.).reshape(5, 5, 3) / 75.
            exp[exp <= 10. / 75.] = np.nan
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])

        # float type with non-NaN fill value
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, dtype=np.float32, fill_value=128)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (3, 5, 5)  # no alpha band
            exp = np.arange(75.).reshape(5, 5, 3) / 75.
            exp[exp <= 10. / 75.] = 128
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])

        # float input with fill value saved to int16 (signed!)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, dtype=np.int16, fill_value=-128)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (3, 5, 5)  # no alpha band
            exp = np.arange(75.).reshape(5, 5, 3) / 75.
            exp2 = (exp * (2 ** 16 - 1) - (2 ** 15)).round()
            exp2[exp <= 10. / 75.] = -128.
            np.testing.assert_allclose(file_data[0], exp2[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp2[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp2[:, :, 2])

        # dask delayed save
        with NamedTemporaryFile(suffix='.tif') as tmp:
            delay = img.save(tmp.name, compute=False)
            assert isinstance(delay, tuple)
            assert isinstance(delay[0], da.Array)
            assert isinstance(delay[1], RIODataset)
            da.store(*delay)
            delay[1].close()

        # float RGBA input to uint8
        alpha = xr.ones_like(data[:, :, 0])
        alpha = alpha.where(data.notnull().all(dim='bands'), 0)
        alpha['bands'] = 'A'
        # make a float version of a uint8 RGBA
        rgb_data = xr.concat((data, alpha), dim='bands')
        img = xrimage.XRImage(rgb_data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band already existed
            exp = np.arange(75.).reshape(5, 5, 3) / 75.
            exp[exp <= 10. / 75.] = 0  # numpy converts NaNs to 0s
            exp = (exp * 255.).round()
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            not_null = (alpha != 0).values
            np.testing.assert_allclose(file_data[3][not_null], 255)  # completely opaque
            np.testing.assert_allclose(file_data[3][~not_null], 0)  # completely transparent

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_datetime(self):
        """Test saving geotiffs when start_time is in the attributes."""
        import datetime as dt

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})

        # "None" as start_time in the attributes
        data.attrs['start_time'] = None
        tags = _get_tags_after_writing_to_geotiff(data)
        assert "TIFFTAG_DATETIME" not in tags

        # Valid datetime
        data.attrs['start_time'] = dt.datetime.utcnow()
        tags = _get_tags_after_writing_to_geotiff(data)
        assert "TIFFTAG_DATETIME" in tags

    @pytest.mark.parametrize("output_ext", [".tif", ".tiff"])
    @pytest.mark.parametrize("use_dask", [False, True])
    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int(self, output_ext, use_dask):
        """Test saving geotiffs when input data is int."""
        arr = np.arange(75).reshape(5, 5, 3)
        if use_dask:
            arr = da.from_array(arr, chunks=5)

        data = xr.DataArray(arr,
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=output_ext) as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_delayed(self):
        """Test saving a geotiff but not computing the result immediately."""
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            delay = img.save(tmp.name, compute=False)
            assert isinstance(delay, tuple)
            assert isinstance(delay[0], da.Array)
            assert isinstance(delay[1], RIODataset)
            da.store(*delay)
            delay[1].close()

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_gcps(self):
        """Test saving geotiffs when input data is int and has GCPs."""
        from rasterio.control import GroundControlPoint
        from pyresample import SwathDefinition

        gcps = [GroundControlPoint(1, 1, 100.0, 1000.0, z=0.0),
                GroundControlPoint(2, 3, 400.0, 2000.0, z=0.0)]
        crs = 'epsg:4326'

        lons = xr.DataArray(da.from_array(np.arange(25).reshape(5, 5), chunks=5),
                            dims=['y', 'x'],
                            attrs={'gcps': gcps,
                                   'crs': crs})

        lats = xr.DataArray(da.from_array(np.arange(25).reshape(5, 5), chunks=5),
                            dims=['y', 'x'],
                            attrs={'gcps': gcps,
                                   'crs': crs})
        swath_def = SwathDefinition(lons, lats)

        data = xr.DataArray(da.from_array(np.arange(75).reshape(5, 5, 3), chunks=5),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']},
                            attrs={'area': swath_def})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                fgcps, fcrs = f.gcps
            for ref, val in zip(gcps, fgcps):
                assert ref.col == val.col
                assert ref.row == val.row
                assert ref.x == val.x
                assert ref.y == val.y
                assert ref.z == val.z
            assert crs == fcrs

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_no_gcp_swath(self):
        """Test saving geotiffs when input data whose SwathDefinition has no GCPs.

        If shouldn't fail, but it also shouldn't have a non-default transform.

        """
        from pyresample import SwathDefinition

        lons = xr.DataArray(da.from_array(np.arange(25).reshape(5, 5), chunks=5),
                            dims=['y', 'x'],
                            attrs={})

        lats = xr.DataArray(da.from_array(np.arange(25).reshape(5, 5), chunks=5),
                            dims=['y', 'x'],
                            attrs={})
        swath_def = SwathDefinition(lons, lats)

        data = xr.DataArray(da.from_array(np.arange(75).reshape(5, 5, 3), chunks=5),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']},
                            attrs={'area': swath_def})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                assert f.transform.a == 1.0
                assert f.transform.b == 0.0
                assert f.transform.c == 0.0

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_rio_colormap(self):
        """Test saving geotiffs when input data is int and a rasterio colormap is provided."""
        exp_cmap = {i: (i, 255 - i, i, 255) for i in range(256)}
        data = xr.DataArray(da.from_array(np.arange(81).reshape(9, 9, 1), chunks=9),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['P']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, keep_palette=True, cmap=exp_cmap)
            with rio.open(tmp.name) as f:
                file_data = f.read()
                cmap = f.colormap(1)
            assert file_data.shape == (1, 9, 9)  # no alpha band
            exp = np.arange(81).reshape(9, 9, 1)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            assert cmap == exp_cmap

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_fill(self):
        """Test saving geotiffs when input data is int and a fill value is specified."""
        data = np.arange(75).reshape(5, 5, 3)
        # second pixel is all bad
        # pixel [0, 1, 1] is also naturally 5 by arange above
        data[0, 1, :] = 5
        data = xr.DataArray(da.from_array(data, chunks=5),
                            dims=['y', 'x', 'bands'],
                            attrs={'_FillValue': 5},
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, fill_value=128)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (3, 5, 5)  # no alpha band
            exp = np.arange(75).reshape(5, 5, 3)
            exp[0, 1, :] = 128
            exp[0, 1, 1] = 128
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_fill_and_alpha(self):
        """Test saving int geotiffs with a fill value and input alpha band."""
        data = np.arange(75).reshape(5, 5, 3)
        # second pixel is all bad
        # pixel [0, 1, 1] is also naturally 5 by arange above
        data[0, 1, :] = 5
        data = xr.DataArray(da.from_array(data, chunks=5),
                            dims=['y', 'x', 'bands'],
                            attrs={'_FillValue': 5},
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # no alpha band
            exp = np.arange(75).reshape(5, 5, 3)
            exp[0, 1, :] = 5
            exp[0, 1, 1] = 5
            exp_alpha = np.ones((5, 5)) * 255
            exp_alpha[0, 1] = 0
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], exp_alpha)

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_area_def(self):
        """Test saving a integer image with an AreaDefinition."""
        from pyproj import CRS
        from pyresample import AreaDefinition
        crs = CRS.from_user_input(4326)
        area_def = AreaDefinition(
            "test", "test", "",
            crs, 5, 5, [-300, -250, 200, 250],
        )

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']},
                            attrs={"area": area_def})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
                assert f.crs.to_epsg() == 4326
                geotransform = f.transform
                assert geotransform.a == 100
                assert geotransform.c == -300
                assert geotransform.e == -100
                assert geotransform.f == 250
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    @pytest.mark.parametrize(
        "cmap",
        [
            Colormap(*tuple((i, (i / 20, i / 20, i / 20)) for i in range(20))),
            Colormap(*tuple((i + 0.00001, (i / 20, i / 20, i / 20)) for i in range(20))),
            Colormap(*tuple((i if i != 2 else 2.00000001, (i / 20, i / 20, i / 20)) for i in range(20))),
        ]
    )
    def test_save_geotiff_int_with_cmap(self, cmap):
        """Test saving integer data to geotiff with a colormap.

        Rasterio specifically can't handle colormaps that are not round
        integers. Unfortunately it only warns when it finds a value in the
        color table that it doesn't expect. For example if an unsigned 8-bit
        color table is being filled with a trollimage Colormap where due to
        floating point one of the values is 15.0000001 instead of 15.0,
        rasterio will issue a warning and then not add a color for that value.
        This test makes sure the colormap written is the colormap read back.

        """
        exp_cmap = {i: (int(i * 255 / 19), int(i * 255 / 19), int(i * 255 / 19), 255) for i in range(20)}
        exp_cmap.update({i: (0, 0, 0, 255) for i in range(20, 256)})
        data = xr.DataArray(da.from_array(np.arange(81).reshape(9, 9, 1), chunks=9),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['P']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, keep_palette=True, cmap=cmap)
            with rio.open(tmp.name) as f:
                file_data = f.read()
                cmap = f.colormap(1)
            assert file_data.shape == (1, 9, 9)  # no alpha band
            exp = np.arange(81).reshape(9, 9, 1)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            assert cmap == exp_cmap

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_bad_cmap(self):
        """Test saving integer data to geotiff with a bad colormap."""
        t_cmap = Colormap(*tuple((i, (i / 20, i / 20, i / 20)) for i in range(20)))
        bad_cmap = [[i, [i, i, i]] for i in range(256)]
        data = xr.DataArray(da.from_array(np.arange(81).reshape(9, 9, 1), chunks=9),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['P']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            with pytest.raises(ValueError):
                img.save(tmp.name, keep_palette=True, cmap=bad_cmap)
            with pytest.raises(ValueError):
                img.save(tmp.name, keep_palette=True, cmap=t_cmap,
                         dtype='uint16')

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_closed_file(self):
        """Test saving geotiffs when the geotiff file has been closed.

        This is to mimic a situation where garbage collection would cause the
        file handler to close the underlying geotiff file that will be written
        to.

        """
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            results = img.save(tmp.name, compute=False)
            results[1].close()  # mimic garbage collection
            da.store(results[0], results[1])
            results[1].close()  # required to flush writes to disk
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_jp2_int(self):
        """Test saving jp2000 when input data is int."""
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.jp2') as tmp:
            img.save(tmp.name, quality=100, reversible=True)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_cloud_optimized_geotiff(self):
        """Test saving cloud optimized geotiffs."""
        # trigger COG driver to create 2 overview levels
        # COG driver is only available in GDAL 3.1 or later
        if rio.__gdal_version__ >= '3.1':
            data = xr.DataArray(np.arange(1200*1200*3).reshape(1200, 1200, 3), dims=[
                'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
            img = xrimage.XRImage(data)
            assert np.issubdtype(img.data.dtype, np.integer)
            with NamedTemporaryFile(suffix='.tif') as tmp:
                img.save(tmp.name, tiled=True, overviews=[], driver='COG')
                with rio.open(tmp.name) as f:
                    # The COG driver should add a tag indicating layout
                    assert (f.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG')
                    assert len(f.overviews(1)) == 2

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_overviews(self):
        """Test saving geotiffs with overviews."""
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, overviews=[2, 4])
            with rio.open(tmp.name) as f:
                assert len(f.overviews(1)) == 2

        # auto-levels
        data = np.zeros(25*25*3, dtype=np.uint8).reshape(25, 25, 3)
        data = xr.DataArray(data, dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, overviews=[], overviews_minsize=2)
            with rio.open(tmp.name) as f:
                assert len(f.overviews(1)) == 4

        # auto-levels and resampling
        data = np.zeros(25*25*3, dtype=np.uint8).reshape(25, 25, 3)
        data = xr.DataArray(data, dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, overviews=[], overviews_minsize=2,
                     overviews_resampling='average')
            with rio.open(tmp.name) as f:
                # no way to check resampling method from the file
                assert len(f.overviews(1)) == 4

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_tags(self):
        """Test saving geotiffs with tags."""
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        tags = {'avg': img.data.mean(), 'current_song': 'disco inferno'}
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, tags=tags)
            tags['avg'] = '37.0'
            with rio.open(tmp.name) as f:
                assert f.tags() == tags

    def test_gamma(self):
        """Test gamma correction."""
        arr = np.arange(75).reshape(5, 5, 3) / 75.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.gamma(.5)
        assert np.allclose(img.data.values, arr ** 2)
        assert img.data.attrs['enhancement_history'][0] == {'gamma': 0.5}

        img.gamma([2., 2., 2.])
        assert len(img.data.attrs['enhancement_history']) == 2
        assert np.allclose(img.data.values, arr)

    def test_crude_stretch(self):
        """Check crude stretching."""
        arr = np.arange(75).reshape(5, 5, 3)
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.crude_stretch()
        red = img.data.sel(bands='R')
        green = img.data.sel(bands='G')
        blue = img.data.sel(bands='B')
        enhs = img.data.attrs['enhancement_history'][0]
        scale_expected = np.array([0.01388889, 0.01388889, 0.01388889])
        offset_expected = np.array([0., -0.01388889, -0.02777778])
        np.testing.assert_allclose(enhs['scale'].values, scale_expected)
        np.testing.assert_allclose(enhs['offset'].values, offset_expected)
        np.testing.assert_allclose(red, arr[:, :, 0] / 72.)
        np.testing.assert_allclose(green, (arr[:, :, 1] - 1.) / (73. - 1.))
        np.testing.assert_allclose(blue, (arr[:, :, 2] - 2.) / (74. - 2.))

        arr = np.arange(75).reshape(5, 5, 3).astype(float)
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.crude_stretch(0, 74)
        np.testing.assert_allclose(img.data.values, arr / 74.)

    def test_invert(self):
        """Check inversion of the image."""
        arr = np.arange(75).reshape(5, 5, 3) / 75.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)

        img.invert(True)
        enhs = img.data.attrs['enhancement_history'][0]
        assert enhs == {'scale': -1, 'offset': 1}
        assert np.allclose(img.data.values, 1 - arr)

        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)

        img.invert([True, False, True])
        offset = xr.DataArray(np.array([1, 0, 1]), dims=['bands'],
                              coords={'bands': ['R', 'G', 'B']})
        scale = xr.DataArray(np.array([-1, 1, -1]), dims=['bands'],
                             coords={'bands': ['R', 'G', 'B']})
        np.testing.assert_allclose(img.data.values, (data * scale + offset).values)

    def test_linear_stretch(self):
        """Test linear stretching with cutoffs."""
        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch_linear()
        enhs = img.data.attrs['enhancement_history'][0]
        np.testing.assert_allclose(enhs['scale'].values, np.array([1.03815937, 1.03815937, 1.03815937]))
        np.testing.assert_allclose(enhs['offset'].values, np.array([-0.00505051, -0.01907969, -0.03310887]), atol=1e-8)
        res = np.array([[[-0.005051, -0.005051, -0.005051],
                         [0.037037, 0.037037, 0.037037],
                         [0.079125, 0.079125, 0.079125],
                         [0.121212, 0.121212, 0.121212],
                         [0.1633, 0.1633, 0.1633]],
                        [[0.205387, 0.205387, 0.205387],
                         [0.247475, 0.247475, 0.247475],
                         [0.289562, 0.289562, 0.289562],
                         [0.33165, 0.33165, 0.33165],
                         [0.373737, 0.373737, 0.373737]],
                        [[0.415825, 0.415825, 0.415825],
                         [0.457912, 0.457912, 0.457912],
                         [0.5, 0.5, 0.5],
                         [0.542088, 0.542088, 0.542088],
                         [0.584175, 0.584175, 0.584175]],
                        [[0.626263, 0.626263, 0.626263],
                         [0.66835, 0.66835, 0.66835],
                         [0.710438, 0.710438, 0.710438],
                         [0.752525, 0.752525, 0.752525],
                         [0.794613, 0.794613, 0.794613]],
                        [[0.8367, 0.8367, 0.8367],
                         [0.878788, 0.878788, 0.878788],
                         [0.920875, 0.920875, 0.920875],
                         [0.962963, 0.962963, 0.962963],
                         [1.005051, 1.005051, 1.005051]]])

        np.testing.assert_allclose(img.data.values, res, atol=1.e-6)

    def test_histogram_stretch(self):
        """Test histogram stretching."""
        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch('histogram')
        enhs = img.data.attrs['enhancement_history'][0]
        assert enhs == {'hist_equalize': True}
        res = np.array([[[0., 0., 0.],
                         [0.04166667, 0.04166667, 0.04166667],
                         [0.08333333, 0.08333333, 0.08333333],
                         [0.125, 0.125, 0.125],
                         [0.16666667, 0.16666667, 0.16666667]],

                        [[0.20833333, 0.20833333, 0.20833333],
                         [0.25, 0.25, 0.25],
                         [0.29166667, 0.29166667, 0.29166667],
                         [0.33333333, 0.33333333, 0.33333333],
                         [0.375, 0.375, 0.375]],

                        [[0.41666667, 0.41666667, 0.41666667],
                         [0.45833333, 0.45833333, 0.45833333],
                         [0.5, 0.5, 0.5],
                         [0.54166667, 0.54166667, 0.54166667],
                         [0.58333333, 0.58333333, 0.58333333]],

                        [[0.625, 0.625, 0.625],
                         [0.66666667, 0.66666667, 0.66666667],
                         [0.70833333, 0.70833333, 0.70833333],
                         [0.75, 0.75, 0.75],
                         [0.79166667, 0.79166667, 0.79166667]],

                        [[0.83333333, 0.83333333, 0.83333333],
                         [0.875, 0.875, 0.875],
                         [0.91666667, 0.91666667, 0.91666667],
                         [0.95833333, 0.95833333, 0.95833333],
                         [0.99951172, 0.99951172, 0.99951172]]])

        np.testing.assert_allclose(img.data.values, res, atol=1.e-6)

    @pytest.mark.parametrize(
        ("min_stretch", "max_stretch"),
        [
            (None, None),
            ([0.0, 1.0 / 74.0, 2.0 / 74.0], [72.0 / 74.0, 73.0 / 74.0, 1.0]),
        ]
    )
    @pytest.mark.parametrize("base", ["e", "10", "2"])
    def test_logarithmic_stretch(self, min_stretch, max_stretch, base):
        """Test logarithmic strecthing."""
        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(data)
            img.stretch(stretch='logarithmic',
                        min_stretch=min_stretch,
                        max_stretch=max_stretch,
                        base=base)
        enhs = img.data.attrs['enhancement_history'][0]
        assert enhs == {'log_factor': 100.0}
        res = np.array([[[0., 0., 0.],
                         [0.35484693, 0.35484693, 0.35484693],
                         [0.48307087, 0.48307087, 0.48307087],
                         [0.5631469, 0.5631469, 0.5631469],
                         [0.62151902, 0.62151902, 0.62151902]],

                        [[0.66747806, 0.66747806, 0.66747806],
                         [0.70538862, 0.70538862, 0.70538862],
                         [0.73765396, 0.73765396, 0.73765396],
                         [0.76573946, 0.76573946, 0.76573946],
                         [0.79060493, 0.79060493, 0.79060493]],

                        [[0.81291336, 0.81291336, 0.81291336],
                         [0.83314196, 0.83314196, 0.83314196],
                         [0.85164569, 0.85164569, 0.85164569],
                         [0.86869572, 0.86869572, 0.86869572],
                         [0.88450394, 0.88450394, 0.88450394]],

                        [[0.899239, 0.899239, 0.899239],
                         [0.9130374, 0.9130374, 0.9130374],
                         [0.92601114, 0.92601114, 0.92601114],
                         [0.93825325, 0.93825325, 0.93825325],
                         [0.94984187, 0.94984187, 0.94984187]],

                        [[0.96084324, 0.96084324, 0.96084324],
                         [0.97131402, 0.97131402, 0.97131402],
                         [0.98130304, 0.98130304, 0.98130304],
                         [0.99085269, 0.99085269, 0.99085269],
                         [1., 1., 1.]]])

        np.testing.assert_allclose(img.data.values, res, atol=1.e-6)

    def test_weber_fechner_stretch(self):
        """Test applying S=2.3klog10I+C to the data."""
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch_weber_fechner(2.5, 0.2)
        enhs = img.data.attrs['enhancement_history'][0]
        assert enhs == {'weber_fechner': (2.5, 0.2)}
        res = np.array([[[-np.inf, -6.73656795, -5.0037],
                         [-3.99003723, -3.27083205, -2.71297317],
                         [-2.25716928, -1.87179258, -1.5379641],
                         [-1.24350651, -0.98010522, -0.74182977],
                         [-0.52430133, -0.32419456, -0.13892463]],

                        [[0.03355755, 0.19490385, 0.34646541],
                         [0.48936144, 0.6245295, 0.75276273],
                         [0.87473814, 0.99103818, 1.10216759],
                         [1.20856662, 1.31062161, 1.40867339],
                         [1.50302421, 1.59394332, 1.68167162]],

                        [[1.7664255, 1.84840006, 1.92777181],
                         [2.00470095, 2.07933336, 2.1518022],
                         [2.22222939, 2.29072683, 2.35739745],
                         [2.42233616, 2.48563068, 2.54736221],
                         [2.60760609, 2.66643234, 2.72390613]],

                        [[2.78008827, 2.83503554, 2.88880105],
                         [2.94143458, 2.99298279, 3.04348956],
                         [3.09299613, 3.14154134, 3.18916183],
                         [3.23589216, 3.28176501, 3.32681127],
                         [3.37106022, 3.41453957, 3.45727566]],

                        [[3.49929345, 3.54061671, 3.58126801],
                         [3.62126886, 3.66063976, 3.69940022],
                         [3.7375689, 3.7751636, 3.81220131],
                         [3.84869831, 3.88467015, 3.92013174],
                         [3.95509735, 3.98958065, 4.02359478]]])

        np.testing.assert_allclose(img.data.values, res, atol=1.e-6)

    def test_jpeg_save(self):
        """Test saving to jpeg."""
        pass

    def test_gtiff_save(self):
        """Test saving to geotiff."""
        pass

    def test_save_masked(self):
        """Test saving masked data."""
        pass

    def test_LA_save(self):
        """Test LA saving."""
        pass

    def test_L_save(self):
        """Test L saving."""
        pass

    def test_P_save(self):
        """Test P saving."""
        pass

    def test_PA_save(self):
        """Test PA saving."""
        pass

    def test_convert_modes(self):
        """Test modes convertions."""
        from trollimage.colormap import brbg, Colormap

        # RGBA colormap
        bw = Colormap(
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 0.5)),
        )

        arr1 = np.arange(150).reshape(1, 15, 10) / 150.
        arr2 = np.append(arr1, np.ones(150).reshape(arr1.shape)).reshape(2, 15, 10)
        arr3 = (np.arange(150).reshape(2, 15, 5) / 15).astype('int64')
        dataset1 = xr.DataArray(arr1.copy(), dims=['bands', 'y', 'x'],
                                coords={'bands': ['L']})
        dataset2 = xr.DataArray(arr2.copy(), dims=['bands', 'x', 'y'],
                                coords={'bands': ['L', 'A']})
        dataset3 = xr.DataArray(arr3.copy(), dims=['bands', 'x', 'y'],
                                coords={'bands': ['P', 'A']})

        img = xrimage.XRImage(dataset1)
        new_img = img.convert(img.mode)
        assert new_img is not None
        # make sure it is a copy
        assert new_img is not img
        assert new_img.data is not img.data

        # L -> LA (int)
        with assert_maximum_dask_computes(1):
            img = xrimage.XRImage((dataset1 * 150).astype(np.uint8))
            img.data.attrs['_FillValue'] = 0  # set fill value
            img = img.convert('LA')
            assert np.issubdtype(img.data.dtype, np.integer)
            assert img.mode == 'LA'
            assert len(img.data.coords['bands']) == 2
            # make sure the alpha band is all opaque except the first pixel
            alpha = img.data.sel(bands='A').values.ravel()
            np.testing.assert_allclose(alpha[0], 0)
            np.testing.assert_allclose(alpha[1:], 255)

        # L -> LA (float)
        with assert_maximum_dask_computes(1):
            img = xrimage.XRImage(dataset1)
            img = img.convert('LA')
            assert img.mode == 'LA'
            assert len(img.data.coords['bands']) == 2
            # make sure the alpha band is all opaque
            np.testing.assert_allclose(img.data.sel(bands='A'), 1.)

        # LA -> L (float)
        with assert_maximum_dask_computes(0):
            img = img.convert('L')
            assert img.mode == 'L'
            assert len(img.data.coords['bands']) == 1

        # L -> RGB (float)
        with assert_maximum_dask_computes(1):
            img = img.convert('RGB')
            assert img.mode == 'RGB'
            assert len(img.data.coords['bands']) == 3
            data = img.data.compute()
            np.testing.assert_allclose(data.sel(bands=['R']), arr1)
            np.testing.assert_allclose(data.sel(bands=['G']), arr1)
            np.testing.assert_allclose(data.sel(bands=['B']), arr1)

        # RGB -> RGBA (float)
        with assert_maximum_dask_computes(1):
            img = img.convert('RGBA')
            assert img.mode == 'RGBA'
            assert len(img.data.coords['bands']) == 4
            assert np.issubdtype(img.data.dtype, np.floating)
            data = img.data.compute()
            np.testing.assert_allclose(data.sel(bands=['R']), arr1)
            np.testing.assert_allclose(data.sel(bands=['G']), arr1)
            np.testing.assert_allclose(data.sel(bands=['B']), arr1)
            # make sure the alpha band is all opaque
            np.testing.assert_allclose(data.sel(bands='A'), 1.)

        # RGB -> RGBA (int)
        with assert_maximum_dask_computes(1):
            img = xrimage.XRImage((dataset1 * 150).astype(np.uint8))
            img = img.convert('RGB')  # L -> RGB
            assert np.issubdtype(img.data.dtype, np.integer)
            img = img.convert('RGBA')
            assert img.mode == 'RGBA'
            assert len(img.data.coords['bands']) == 4
            assert np.issubdtype(img.data.dtype, np.integer)
            data = img.data.compute()
            np.testing.assert_allclose(data.sel(bands=['R']), (arr1 * 150).astype(np.uint8))
            np.testing.assert_allclose(data.sel(bands=['G']), (arr1 * 150).astype(np.uint8))
            np.testing.assert_allclose(data.sel(bands=['B']), (arr1 * 150).astype(np.uint8))
            # make sure the alpha band is all opaque
            np.testing.assert_allclose(data.sel(bands='A'), 255)

        # LA -> RGBA (float)
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(dataset2)
            img = img.convert('RGBA')
            assert img.mode == 'RGBA'
            assert len(img.data.coords['bands']) == 4

        # L -> palettize -> RGBA (float)
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(dataset1)
            img.palettize(brbg)
            pal = img.palette

            img2 = img.convert('RGBA')
            assert np.issubdtype(img2.data.dtype, np.floating)
            assert img2.mode == 'RGBA'
            assert len(img2.data.coords['bands']) == 4

        # PA -> RGB (float)
        img = xrimage.XRImage(dataset3)
        img.palette = pal
        with assert_maximum_dask_computes(0):
            img = img.convert('RGB')
            assert np.issubdtype(img.data.dtype, np.floating)
            assert img.mode == 'RGB'
            assert len(img.data.coords['bands']) == 3

        with pytest.raises(ValueError):
            img.convert('A')

        # L -> palettize -> RGBA (float) with RGBA colormap
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(dataset1)
            img.palettize(bw)

            img2 = img.convert('RGBA')
            assert np.issubdtype(img2.data.dtype, np.floating)
            assert img2.mode == 'RGBA'
            assert len(img2.data.coords['bands']) == 4
            # convert to RGB, use RGBA from colormap regardless
            img2 = img.convert('RGB')
            assert np.issubdtype(img2.data.dtype, np.floating)
            assert img2.mode == 'RGBA'
            assert len(img2.data.coords['bands']) == 4

    def test_final_mode(self):
        """Test final_mode."""
        from trollimage import xrimage

        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert img.final_mode(None) == 'RGBA'
        assert img.final_mode(0) == 'RGB'

    def test_stack(self):
        """Test stack."""
        from trollimage import xrimage

        # background image
        arr1 = np.zeros((2, 2))
        data1 = xr.DataArray(arr1, dims=['y', 'x'])
        bkg = xrimage.XRImage(data1)

        # image to be stacked
        arr2 = np.full((2, 2), np.nan)
        arr2[0] = 1
        data2 = xr.DataArray(arr2, dims=['y', 'x'])
        img = xrimage.XRImage(data2)

        # expected result
        arr3 = arr1.copy()
        arr3[0] = 1
        data3 = xr.DataArray(arr3, dims=['y', 'x'])
        res = xrimage.XRImage(data3)

        # stack image over the background
        bkg.stack(img)

        # check result
        np.testing.assert_allclose(bkg.data, res.data, rtol=1e-05)

    def test_merge(self):
        """Test merge."""
        pass

    def test_blend(self):
        """Test blend."""
        from trollimage import xrimage

        core1 = np.arange(75).reshape(5, 5, 3) / 75.0
        alpha1 = np.linspace(0, 1, 25).reshape(5, 5, 1)
        arr1 = np.concatenate([core1, alpha1], 2)
        data1 = xr.DataArray(arr1, dims=['y', 'x', 'bands'],
                             coords={'bands': ['R', 'G', 'B', 'A']})
        img1 = xrimage.XRImage(data1)

        core2 = np.arange(75, 0, -1).reshape(5, 5, 3) / 75.0
        alpha2 = np.linspace(1, 0, 25).reshape(5, 5, 1)
        arr2 = np.concatenate([core2, alpha2], 2)
        data2 = xr.DataArray(arr2, dims=['y', 'x', 'bands'],
                             coords={'bands': ['R', 'G', 'B', 'A']})
        img2 = xrimage.XRImage(data2)

        img3 = img1.blend(img2)

        np.testing.assert_allclose(
                (alpha1 + alpha2 * (1 - alpha1)).squeeze(),
                img3.data.sel(bands="A"))

        np.testing.assert_allclose(
            img3.data.sel(bands="R").values,
            np.array(
                [[1.,           0.95833635, 0.9136842,  0.8666667,  0.8180645],
                 [0.768815,     0.72,       0.6728228,  0.62857145, 0.5885714],
                 [0.55412847,   0.5264665,  0.50666666, 0.495612,   0.49394494],
                 [0.5020408,    0.52,       0.5476586,  0.5846154,  0.63027024],
                 [0.683871,     0.7445614,  0.81142855, 0.8835443,  0.96]]))

        with pytest.raises(TypeError):
            img1.blend("Salekhard")

        wrongimg = xrimage.XRImage(
                xr.DataArray(np.zeros((0, 0)), dims=("y", "x")))
        with pytest.raises(ValueError):
            img1.blend(wrongimg)

    def test_replace_luminance(self):
        """Test luminance replacement."""
        pass

    def test_putalpha(self):
        """Test putalpha."""
        pass

    def test_show(self):
        """Test that the show commands calls PIL.show."""
        from trollimage import xrimage

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3) / 75., dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage.PILImage.Image, 'show', return_value=None) as s:
            img.show()
            s.assert_called_once()

    def test_apply_pil(self):
        """Test the apply_pil method."""
        from trollimage import xrimage

        np_data = np.arange(75).reshape(5, 5, 3) / 75.
        data = xr.DataArray(np_data, dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})

        dummy_args = [(OrderedDict(), ), {}]

        def dummy_fun(pil_obj, *args, **kwargs):
            dummy_args[0] = args
            dummy_args[1] = kwargs
            return pil_obj

        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage, "PILImage") as pi:
            pil_img = mock.MagicMock()
            pi.fromarray = mock.Mock(wraps=lambda *args, **kwargs: pil_img)
            res = img.apply_pil(dummy_fun, 'RGB')
            # check that the pil image generation is delayed
            pi.fromarray.assert_not_called()
            # make it happen
            res.data.data.compute()
            pil_img.convert.assert_called_with('RGB')

        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage, "PILImage") as pi:
            pil_img = mock.MagicMock()
            pi.fromarray = mock.Mock(wraps=lambda *args, **kwargs: pil_img)
            res = img.apply_pil(dummy_fun, 'RGB',
                                fun_args=('Hey', 'Jude'),
                                fun_kwargs={'chorus': "La lala lalalala"})
            assert dummy_args == [({}, ), {}]
            res.data.data.compute()
            assert dummy_args == [(OrderedDict(), 'Hey', 'Jude'), {'chorus': "La lala lalalala"}]

        # Test HACK for _burn_overlay
        dummy_args = [(OrderedDict(), ), {}]

        def _burn_overlay(pil_obj, *args, **kwargs):
            dummy_args[0] = args
            dummy_args[1] = kwargs
            return pil_obj

        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage, "PILImage") as pi:
            pil_img = mock.MagicMock()
            pi.fromarray = mock.Mock(wraps=lambda *args, **kwargs: pil_img)
            res = img.apply_pil(_burn_overlay, 'RGB')
            # check that the pil image generation is delayed
            pi.fromarray.assert_not_called()
            # make it happen
            res.data.data.compute()
            pil_img.convert.assert_called_with('RGB')


class TestImageColorize:
    """Test the colorize method of the Image class."""

    def test_colorize_la_rgb(self):
        """Test colorizing an LA image with an RGB colormap."""
        arr = np.arange(75).reshape(5, 15) / 74.
        alpha = arr > (8.0 / 74.0)
        img = image.Image(channels=[arr.copy(), alpha], mode="LA")
        img.colorize(brbg)

        expected = list(TestXRImageColorize._expected)
        np.testing.assert_allclose(img.channels[0], expected[0])
        np.testing.assert_allclose(img.channels[1], expected[1])
        np.testing.assert_allclose(img.channels[2], expected[2])
        np.testing.assert_allclose(img.channels[3], alpha)


class TestXRImageColorize:
    """Test the colorize method of the XRImage class."""

    _expected = np.array([[
        [3.29411737e-01, 3.57655096e-01, 3.86434124e-01, 4.15693619e-01,
         4.45354613e-01, 4.75400874e-01, 5.05821379e-01, 5.36605942e-01,
         5.65154991e-01, 5.92088509e-01, 6.19067983e-01, 6.46087257e-01,
         6.73140335e-01, 7.00221370e-01, 7.27324655e-01],
        [7.52329780e-01, 7.68885192e-01, 7.85480725e-01, 8.02165039e-01,
         8.18991658e-01, 8.36019210e-01, 8.53311582e-01, 8.70937944e-01,
         8.84215466e-01, 8.96340861e-01, 9.08470028e-01, 9.20615989e-01,
         9.32792726e-01, 9.45015152e-01, 9.57299069e-01],
        [9.56159426e-01, 9.37986875e-01, 9.25192897e-01, 9.18645198e-01,
         9.18939761e-01, 9.26295798e-01, 9.40479522e-01, 9.60784274e-01,
         9.40020132e-01, 9.17935908e-01, 8.94639401e-01, 8.70237106e-01,
         8.44833768e-01, 8.18532129e-01, 7.91432917e-01],
        [7.58448199e-01, 7.21741672e-01, 6.84822740e-01, 6.47626523e-01,
         6.10070658e-01, 5.72048971e-01, 5.33422004e-01, 4.94570868e-01,
         4.57464108e-01, 4.20002646e-01, 3.82018470e-01, 3.43266534e-01,
         3.03372589e-01, 2.61727477e-01, 2.17242874e-01],
        [1.89905775e-01, 1.67063045e-01, 1.43524430e-01, 1.18889134e-01,
         9.24115382e-02, 6.24349277e-02, 2.53761544e-02, 4.08184216e-03,
         4.27989281e-03, 4.17932136e-03, 3.78664262e-03, 3.12694131e-03,
         2.24025474e-03, 1.17808547e-03, 4.27413533e-08]],
       [[1.88235338e-01, 2.05148716e-01, 2.22246545e-01, 2.39526080e-01,
         2.56989499e-01, 2.74629834e-01, 2.92440006e-01, 3.10413434e-01,
         3.32343826e-01, 3.57065431e-01, 3.82068290e-01, 4.07348972e-01,
         4.32903771e-01, 4.58728828e-01, 4.84820214e-01],
        [5.12920816e-01, 5.47946941e-01, 5.82732555e-01, 6.17314767e-01,
         6.51719374e-01, 6.85963755e-01, 7.20058907e-01, 7.54010908e-01,
         7.76938582e-01, 7.97119672e-01, 8.17286151e-01, 8.37436055e-01,
         8.57567250e-01, 8.77677448e-01, 8.97764224e-01],
        [9.15264356e-01, 9.27481082e-01, 9.37652197e-01, 9.45817488e-01,
         9.52058640e-01, 9.56503095e-01, 9.59331143e-01, 9.60784378e-01,
         9.54729702e-01, 9.48843339e-01, 9.43068589e-01, 9.37350480e-01,
         9.31636367e-01, 9.25876402e-01, 9.20023868e-01],
        [9.08501739e-01, 8.93232140e-01, 8.77927050e-01, 8.62584953e-01,
         8.47204361e-01, 8.31783816e-01, 8.16321883e-01, 7.98071160e-01,
         7.68921244e-01, 7.39943772e-01, 7.11141605e-01, 6.82517736e-01,
         6.54075295e-01, 6.25817550e-01, 5.97747915e-01],
        [5.70776460e-01, 5.44247790e-01, 5.17943022e-01, 4.91868010e-01,
         4.66028951e-01, 4.40432416e-01, 4.15085387e-01, 3.90762614e-01,
         3.67819668e-01, 3.45100725e-01, 3.22617410e-01, 3.00381898e-01,
         2.78407005e-01, 2.56706279e-01, 2.35294121e-01]],
       [[1.96078107e-02, 2.42548730e-02, 2.74972914e-02, 2.96227826e-02,
         3.17156346e-02, 3.38568632e-02, 3.60498856e-02, 3.82990518e-02,
         5.17340258e-02, 7.13424642e-02, 9.00791521e-02, 1.08349534e-01,
         1.26372972e-01, 1.44280400e-01, 1.62155446e-01],
        [1.84723738e-01, 2.25766596e-01, 2.66872663e-01, 3.08395895e-01,
         3.50522797e-01, 3.93349769e-01, 4.36919875e-01, 4.81242213e-01,
         5.19495736e-01, 5.56210031e-01, 5.93054327e-01, 6.30051826e-01,
         6.67218415e-01, 7.04564497e-01, 7.42096291e-01],
        [7.75783700e-01, 8.05228737e-01, 8.35977793e-01, 8.66550574e-01,
         8.95532082e-01, 9.21622726e-01, 9.43684792e-01, 9.60784304e-01,
         9.46951904e-01, 9.34756333e-01, 9.24234153e-01, 9.15407055e-01,
         9.08281595e-01, 9.02849071e-01, 8.99085540e-01],
        [8.86846761e-01, 8.68087760e-01, 8.49200399e-01, 8.30188003e-01,
         8.11054012e-01, 7.91801986e-01, 7.72435611e-01, 7.51368879e-01,
         7.24059059e-01, 6.97016441e-01, 6.70243019e-01, 6.43740907e-01,
         6.17512340e-01, 5.91559696e-01, 5.65885501e-01],
        [5.39262097e-01, 5.12603472e-01, 4.86221761e-01, 4.60123407e-01,
         4.34315308e-01, 4.08804870e-01, 3.83600068e-01, 3.58016760e-01,
         3.31909014e-01, 3.06406099e-01, 2.81515767e-01, 2.57245707e-01,
         2.33603643e-01, 2.10597450e-01, 1.88235292e-01]]])

    @pytest.mark.parametrize("colormap_tag", [None, "colormap"])
    def test_colorize_geotiff_tag(self, tmp_path, colormap_tag):
        """Test that a colorized colormap can be saved to a geotiff tag."""
        new_range = (0.0, 0.5)
        arr = np.arange(75).reshape(5, 15) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        new_brbg = brbg.set_range(*new_range, inplace=False)
        img = xrimage.XRImage(data)
        img.colorize(new_brbg)

        dst = str(tmp_path / "test.tif")
        img.save(dst, colormap_tag=colormap_tag)
        with rio.open(dst, "r") as gtiff_file:
            metadata = gtiff_file.tags()
            if colormap_tag is None:
                assert "colormap" not in metadata
            else:
                assert "colormap" in metadata
                loaded_brbg = Colormap.from_file(metadata["colormap"])
                np.testing.assert_allclose(new_brbg.values, loaded_brbg.values)
                np.testing.assert_allclose(new_brbg.colors, loaded_brbg.colors)

    @pytest.mark.parametrize(
        ("new_range", "input_scale", "input_offset", "expected_scale", "expected_offset"),
        [
            ((0.0, 1.0), 1.0, 0.0, 1.0, 0.0),
            ((0.0, 0.5), 1.0, 0.0, 2.0, 0.0),
            ((2.0, 4.0), 2.0, 2.0, 0.5, -1.0),
        ],
    )
    def test_colorize_l_rgb(self, new_range, input_scale, input_offset, expected_scale, expected_offset):
        """Test colorize with an RGB colormap."""
        arr = np.arange(75).reshape(5, 15) / 74. * input_scale + input_offset
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        new_brbg = brbg.set_range(*new_range, inplace=False)
        img = xrimage.XRImage(data)
        img.colorize(new_brbg)
        values = img.data.compute()

        if new_range[1] == 0.5:
            expected2 = self._expected.copy().reshape((3, 75))
            flat_expected = self._expected.reshape((3, 75))
            expected2[:, :38] = flat_expected[:, ::2]
            expected2[:, 38:] = flat_expected[:, -1:]
            expected = expected2.reshape((3, 5, 15))
        else:
            expected = self._expected
        np.testing.assert_allclose(values, expected)
        assert "enhancement_history" in img.data.attrs
        assert img.data.attrs["enhancement_history"][-1]["scale"] == expected_scale
        assert img.data.attrs["enhancement_history"][-1]["offset"] == expected_offset
        assert isinstance(img.data.attrs["enhancement_history"][-1]["colormap"], Colormap)

    def test_colorize_int_l_rgb_with_fills(self):
        """Test integer data with _FillValue is masked (NaN) when colorized."""
        arr = np.arange(75, dtype=np.uint8).reshape(5, 15)
        arr[1, :] = 255
        data = xr.DataArray(arr.copy(), dims=['y', 'x'],
                            attrs={"_FillValue": 255})
        new_brbg = brbg.set_range(5, 20, inplace=False)
        img = xrimage.XRImage(data)
        img.colorize(new_brbg)
        values = img.data.compute()
        assert values.shape == (3,) + arr.shape  # RGB
        np.testing.assert_allclose(values[:, 1, :], np.nan)
        assert np.count_nonzero(np.isnan(values)) == arr.shape[1] * 3

        assert "enhancement_history" in img.data.attrs
        assert img.data.attrs["enhancement_history"][-1]["scale"] == 1 / (20 - 5)
        assert img.data.attrs["enhancement_history"][-1]["offset"] == -5 / (20 - 5)
        assert isinstance(img.data.attrs["enhancement_history"][-1]["colormap"], Colormap)

    def test_colorize_la_rgb(self):
        """Test colorizing an LA image with an RGB colormap."""
        arr = np.arange(75).reshape((5, 15)) / 74.
        alpha = arr > 40.
        data = xr.DataArray([arr.copy(), alpha],
                            dims=['bands', 'y', 'x'],
                            coords={'bands': ['L', 'A']})
        img = xrimage.XRImage(data)
        img.colorize(brbg)

        values = img.data.values
        expected = np.concatenate((self._expected,
                                   alpha.reshape((1,) + alpha.shape)))
        np.testing.assert_allclose(values, expected)
        assert "enhancement_history" in img.data.attrs
        assert img.data.attrs["enhancement_history"][-1]["scale"] == 1.0
        assert img.data.attrs["enhancement_history"][-1]["offset"] == 0.0
        assert isinstance(img.data.attrs["enhancement_history"][-1]["colormap"], Colormap)

    def test_colorize_rgba(self):
        """Test colorize with an RGBA colormap."""
        from trollimage import xrimage
        from trollimage.colormap import Colormap

        # RGBA colormap
        bw = Colormap(
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 0.5)),
        )

        arr = np.arange(75).reshape(5, 15) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        img = xrimage.XRImage(data)
        img.colorize(bw)
        values = img.data.compute()
        assert (4, 5, 15) == values.shape
        np.testing.assert_allclose(values[:, 0, 0], [1.0, 1.0, 1.0, 1.0], rtol=1e-03)
        np.testing.assert_allclose(values[:, -1, -1], [0.0, 0.0, 0.0, 0.5])
        assert "enhancement_history" in img.data.attrs
        assert img.data.attrs["enhancement_history"][-1]["scale"] == 1.0
        assert img.data.attrs["enhancement_history"][-1]["offset"] == 0.0
        assert isinstance(img.data.attrs["enhancement_history"][-1]["colormap"], Colormap)


class TestXRImagePalettize:
    """Test the XRImage palettize method."""

    @pytest.mark.parametrize(
        ("new_range", "input_scale", "input_offset", "expected_scale", "expected_offset"),
        [
            ((0.0, 1.0), 1.0, 0.0, 1.0, 0.0),
            ((0.0, 0.5), 1.0, 0.0, 2.0, 0.0),
            ((2.0, 4.0), 2.0, 2.0, 0.5, -1.0),
        ],
    )
    def test_palettize(self, new_range, input_scale, input_offset, expected_scale, expected_offset):
        """Test palettize with an RGB colormap."""
        arr = np.arange(75).reshape(5, 15) / 74. * input_scale + input_offset
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        img = xrimage.XRImage(data)
        new_brbg = brbg.set_range(*new_range, inplace=False)
        img.palettize(new_brbg)

        values = img.data.values
        expected = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10]]])
        if new_range[1] == 0.5:
            flat_expected = expected.reshape((1, 75))
            expected2 = flat_expected.copy()
            expected2[:, :38] = flat_expected[:, ::2]
            expected2[:, 38:] = flat_expected[:, -1:]
            expected = expected2.reshape((1, 5, 15))
        np.testing.assert_allclose(values, expected)
        assert "enhancement_history" in img.data.attrs
        assert img.data.attrs["enhancement_history"][-1]["scale"] == expected_scale
        assert img.data.attrs["enhancement_history"][-1]["offset"] == expected_offset

    def test_palettize_rgba(self):
        """Test palettize with an RGBA colormap."""
        from trollimage import xrimage
        from trollimage.colormap import Colormap

        # RGBA colormap
        bw = Colormap(
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 0.5)),
        )

        arr = np.arange(75).reshape(5, 15) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        img = xrimage.XRImage(data)
        img.palettize(bw)

        values = img.data.values
        assert (1, 5, 15) == values.shape
        assert (2, 4) == bw.colors.shape

    @pytest.mark.parametrize("colormap_tag", [None, "colormap"])
    @pytest.mark.parametrize("keep_palette", [False, True])
    def test_palettize_geotiff_tag(self, tmp_path, colormap_tag, keep_palette):
        """Test that a palettized image can be saved to a geotiff tag."""
        new_range = (0.0, 0.5)
        arr = np.arange(75).reshape(5, 15) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        new_brbg = brbg.set_range(*new_range, inplace=False)
        img = xrimage.XRImage(data)
        img.palettize(new_brbg)

        dst = str(tmp_path / "test.tif")
        img.save(dst, colormap_tag=colormap_tag, keep_palette=keep_palette)
        with rio.open(dst, "r") as gtiff_file:
            metadata = gtiff_file.tags()
            if colormap_tag is None:
                assert "colormap" not in metadata
            else:
                assert "colormap" in metadata
                loaded_brbg = Colormap.from_file(metadata["colormap"])
                np.testing.assert_allclose(new_brbg.values, loaded_brbg.values)
                np.testing.assert_allclose(new_brbg.colors, loaded_brbg.colors)


class TestXRImageSaveScaleOffset:
    """Test case for saving an image with scale and offset tags."""

    def setup_method(self) -> None:
        """Set up the test case."""
        from trollimage import xrimage
        data = xr.DataArray(np.arange(25).reshape(5, 5, 1), dims=[
            'y', 'x', 'bands'], coords={'bands': ['L']})
        self.img = xrimage.XRImage(data)
        rgb_data = xr.DataArray(np.arange(3 * 25).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        self.rgb_img = xrimage.XRImage(rgb_data)

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset(self):
        """Test saving geotiffs with tags."""
        expected_tags = {'scale': 24.0 / 255, 'offset': 0}

        self.img.stretch()
        with pytest.warns(DeprecationWarning):
            self._save_and_check_tags(
                expected_tags,
                include_scale_offset_tags=True)

    def test_gamma_geotiff_scale_offset(self, tmp_path):
        """Test that saving gamma-enhanced data to a geotiff with scale/offset tags doesn't fail."""
        self.img.gamma(.5)
        out_fn = str(tmp_path / "test.tif")
        self.img.save(out_fn, scale_offset_tags=("scale", "offset"))
        with rio.open(out_fn, "r") as ds:
            assert np.isnan(float(ds.tags()["scale"]))
            assert np.isnan(float(ds.tags()["offset"]))

    def test_rgb_geotiff_scale_offset(self, tmp_path):
        """Test that saving RGB data to a geotiff with scale/offset tags doesn't fail."""
        self.rgb_img.stretch(
            stretch="crude",
            min_stretch=[-25, -40, 243],
            max_stretch=[0, 5, 208]
        )
        out_fn = str(tmp_path / "test.tif")
        self.rgb_img.save(out_fn, scale_offset_tags=("scale", "offset"))
        with rio.open(out_fn, "r") as ds:
            assert np.isnan(float(ds.tags()["scale"]))
            assert np.isnan(float(ds.tags()["offset"]))

    def _save_and_check_tags(self, expected_tags, **kwargs):
        with NamedTemporaryFile(suffix='.tif') as tmp:
            self.img.save(tmp.name, **kwargs)

            import rasterio as rio
            with rio.open(tmp.name) as f:
                ftags = f.tags()
                for key, val in expected_tags.items():
                    np.testing.assert_almost_equal(float(ftags[key]), val)

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset_from_lists(self):
        """Test saving geotiffs with tags that come from lists."""
        expected_tags = {'scale': 23.0 / 255, 'offset': 1}

        self.img.crude_stretch([1], [24])
        self._save_and_check_tags(
                expected_tags,
                scale_offset_tags=("scale", "offset"))

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset_custom_labels(self):
        """Test saving GeoTIFF with different scale/offset tag labels."""
        expected_tags = {"gradient": 24.0 / 255, "axis_intercept": 0}
        self.img.stretch()
        self._save_and_check_tags(
                expected_tags,
                scale_offset_tags=("gradient", "axis_intercept"))


def _get_tags_after_writing_to_geotiff(data):
    import rasterio as rio

    img = xrimage.XRImage(data)
    with NamedTemporaryFile(suffix='.tif') as tmp:
        img.save(tmp.name)
        with rio.open(tmp.name) as f:
            return f.tags()
