#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2015, 2017.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.
"""Module for testing the image and xrimage modules."""
import os
import random
import sys
import tempfile
import unittest
from unittest import mock
from collections import OrderedDict
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from trollimage import image

EPSILON = 0.0001


class CustomScheduler(object):
    """Custom dask scheduler that raises an exception if dask is computed too many times."""

    def __init__(self, max_computes=1):
        """Set starting and maximum compute counts."""
        self.max_computes = max_computes
        self.total_computes = 0

    def __call__(self, dsk, keys, **kwargs):
        """Compute dask task and keep track of number of times we do so."""
        import dask
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError("Too many dask computations were scheduled: {}".format(self.total_computes))
        return dask.get(dsk, keys, **kwargs)


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
    """Class for testing the mpop.imageo.image module."""

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
        import xarray as xr
        from trollimage import xrimage
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
        import xarray as xr
        import numpy as np
        from trollimage import xrimage
        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=['y', 'x'])
        img = xrimage.XRImage(data)
        assert img.mode == 'L'
        n_arr = np.asarray(img.data)
        # if this succeeds then its writable
        n_arr[n_arr == 0.5] = 1

    def test_regression_double_format_save(self):
        """Test that double format information isn't passed to save."""
        import xarray as xr
        from trollimage import xrimage

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
        import xarray as xr
        import dask.array as da
        from dask.delayed import Delayed
        from trollimage import xrimage
        from trollimage.colormap import brbg, Colormap
        import rasterio as rio

        # RGBA colormap
        bw = Colormap(
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 0.5)),
        )

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
        import xarray as xr
        from trollimage import xrimage
        import rasterio as rio

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
        import xarray as xr
        from trollimage import xrimage
        import rasterio as rio

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
        import xarray as xr
        from trollimage import xrimage

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
        import xarray as xr
        import dask.array as da
        from trollimage import xrimage
        import rasterio as rio

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
            assert isinstance(delay[1], xrimage.RIODataset)
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
        import xarray as xr
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

    @pytest.mark.skipif(sys.platform.startswith('win'),
                        reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int(self):
        """Test saving geotiffs when input data is int."""
        import xarray as xr
        import dask.array as da
        from trollimage import xrimage
        import rasterio as rio
        from rasterio.control import GroundControlPoint

        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)
        # test .tiff too
        with NamedTemporaryFile(suffix='.tiff') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

        data = xr.DataArray(da.from_array(np.arange(75).reshape(5, 5, 3), chunks=5),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        # Regular default save
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

        # dask delayed save
        with NamedTemporaryFile(suffix='.tif') as tmp:
            delay = img.save(tmp.name, compute=False)
            assert isinstance(delay, tuple)
            assert isinstance(delay[0], da.Array)
            assert isinstance(delay[1], xrimage.RIODataset)
            da.store(*delay)
            delay[1].close()

        # GCPs
        class FakeArea():
            def __init__(self, lons, lats):
                self.lons = lons
                self.lats = lats

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

        data = xr.DataArray(da.from_array(np.arange(75).reshape(5, 5, 3), chunks=5),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']},
                            attrs={'area': FakeArea(lons, lats)})
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

        # with rasterio colormap provided
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

        # with trollimage colormap provided
        from trollimage.colormap import Colormap
        t_cmap = Colormap(*tuple((i, (i, i, i)) for i in range(20)))
        exp_cmap = {i: (int(i * 255 / 19), int(i * 255 / 19), int(i * 255 / 19), 255) for i in range(20)}
        exp_cmap.update({i: (0, 0, 0, 255) for i in range(20, 256)})
        data = xr.DataArray(da.from_array(np.arange(81).reshape(9, 9, 1), chunks=9),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['P']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, keep_palette=True, cmap=t_cmap)
            with rio.open(tmp.name) as f:
                file_data = f.read()
                cmap = f.colormap(1)
            assert file_data.shape == (1, 9, 9)  # no alpha band
            exp = np.arange(81).reshape(9, 9, 1)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            assert cmap == exp_cmap

        # with bad colormap provided
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

        # with input fill value
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

        # input fill value but alpha on output
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

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_jp2_int(self):
        """Test saving jp2000 when input data is int."""
        import xarray as xr
        from trollimage import xrimage
        import rasterio as rio

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
    def test_save_overviews(self):
        """Test saving geotiffs with overviews."""
        import xarray as xr
        from trollimage import xrimage
        import rasterio as rio

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
        import xarray as xr
        from trollimage import xrimage
        import rasterio as rio

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

    @pytest.mark.skipif(sys.platform.startswith('win'), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset(self):
        """Test saving geotiffs with tags."""
        import xarray as xr
        from trollimage import xrimage
        import rasterio as rio

        data = xr.DataArray(np.arange(25).reshape(5, 5, 1), dims=[
            'y', 'x', 'bands'], coords={'bands': ['L']})
        img = xrimage.XRImage(data)
        img.stretch()
        with NamedTemporaryFile(suffix='.tif') as tmp:
            img.save(tmp.name, include_scale_offset_tags=True)
            tags = {'scale': 24.0 / 255, 'offset': 0}
            with rio.open(tmp.name) as f:
                ftags = f.tags()
                for key, val in tags.items():
                    np.testing.assert_almost_equal(float(ftags[key]), val)

    def test_gamma(self):
        """Test gamma correction."""
        import xarray as xr
        from trollimage import xrimage

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
        import xarray as xr
        from trollimage import xrimage

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
        import xarray as xr
        from trollimage import xrimage

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
        import xarray as xr
        from trollimage import xrimage

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
        import xarray as xr
        from trollimage import xrimage

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

    def test_logarithmic_stretch(self):
        """Test logarithmic strecthing."""
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch(stretch='logarithmic')
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
        import xarray as xr
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
        import dask
        import xarray as xr
        from trollimage import xrimage
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
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            img = xrimage.XRImage((dataset1 * 150).astype(np.uint8))
            img.data.attrs['_FillValue'] = 0  # set fill value
            img = img.convert('LA')
            assert np.issubdtype(img.data.dtype, np.integer)
            assert img.mode == 'LA'
            assert len(img.data.coords['bands'] == 2)
            # make sure the alpha band is all opaque except the first pixel
            alpha = img.data.sel(bands='A').values.ravel()
            np.testing.assert_allclose(alpha[0], 0)
            np.testing.assert_allclose(alpha[1:], 255)

        # L -> LA (float)
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            img = xrimage.XRImage(dataset1)
            img = img.convert('LA')
            assert img.mode == 'LA'
            assert len(img.data.coords['bands']) == 2
            # make sure the alpha band is all opaque
            np.testing.assert_allclose(img.data.sel(bands='A'), 1.)

        # LA -> L (float)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            img = img.convert('L')
            assert img.mode == 'L'
            assert len(img.data.coords['bands']) == 1

        # L -> RGB (float)
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            img = img.convert('RGB')
            assert img.mode == 'RGB'
            assert len(img.data.coords['bands']) == 3
            data = img.data.compute()
            np.testing.assert_allclose(data.sel(bands=['R']), arr1)
            np.testing.assert_allclose(data.sel(bands=['G']), arr1)
            np.testing.assert_allclose(data.sel(bands=['B']), arr1)

        # RGB -> RGBA (float)
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
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
        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
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
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            img = xrimage.XRImage(dataset2)
            img = img.convert('RGBA')
            assert img.mode == 'RGBA'
            assert len(img.data.coords['bands']) == 4

        # L -> palettize -> RGBA (float)
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
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
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            img = img.convert('RGB')
            assert np.issubdtype(img.data.dtype, np.floating)
            assert img.mode == 'RGB'
            assert len(img.data.coords['bands']) == 3

        with pytest.raises(ValueError):
            img.convert('A')

        # L -> palettize -> RGBA (float) with RGBA colormap
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
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
        import xarray as xr
        from trollimage import xrimage

        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        assert img.final_mode(None) == 'RGBA'
        assert img.final_mode(0) == 'RGB'

    def test_colorize(self):
        """Test colorize with an RGB colormap."""
        import xarray as xr
        from trollimage import xrimage
        from trollimage.colormap import brbg

        arr = np.arange(75).reshape(5, 15) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        img = xrimage.XRImage(data)
        img.colorize(brbg)
        values = img.data.compute()

        expected = np.array([[
            [3.29409498e-01, 3.59108764e-01, 3.88800969e-01,
             4.18486092e-01, 4.48164112e-01, 4.77835010e-01,
             5.07498765e-01, 5.37155355e-01, 5.65419479e-01,
             5.92686124e-01, 6.19861622e-01, 6.46945403e-01,
             6.73936907e-01, 7.00835579e-01, 7.27640871e-01],
            [7.58680358e-01, 8.01695237e-01, 8.35686284e-01,
             8.60598212e-01, 8.76625002e-01, 8.84194741e-01,
             8.83948647e-01, 8.76714923e-01, 8.95016030e-01,
             9.14039881e-01, 9.27287161e-01, 9.36546985e-01,
             9.43656076e-01, 9.50421050e-01, 9.58544227e-01],
            [9.86916929e-01, 1.02423117e+00, 1.03591220e+00,
             1.02666645e+00, 1.00491333e+00, 9.80759775e-01,
             9.63746819e-01, 9.60798629e-01, 9.47739946e-01,
             9.27428067e-01, 9.01184523e-01, 8.71168132e-01,
             8.40161241e-01, 8.11290344e-01, 7.87705814e-01],
            [7.57749840e-01, 7.20020026e-01, 6.82329616e-01,
             6.44678929e-01, 6.07068282e-01, 5.69497990e-01,
             5.31968369e-01, 4.94025422e-01, 4.54275131e-01,
             4.14517560e-01, 3.74757709e-01, 3.35000583e-01,
             2.95251189e-01, 2.55514533e-01, 2.15795621e-01],
            [1.85805611e-01, 1.58245609e-01, 1.30686714e-01,
             1.03128926e-01, 7.55722460e-02, 4.80166757e-02,
             2.04622160e-02, 3.79809920e-03, 3.46310306e-03,
             3.10070529e-03, 2.68579661e-03, 2.19341216e-03,
             1.59875239e-03, 8.77203803e-04, 4.35952940e-06]],

            [[1.88249866e-01, 2.05728128e-01, 2.23209861e-01,
              2.40695072e-01, 2.58183766e-01, 2.75675949e-01,
              2.93171625e-01, 3.10670801e-01, 3.32877903e-01,
              3.58244116e-01, 3.83638063e-01, 4.09059827e-01,
              4.34509485e-01, 4.59987117e-01, 4.85492795e-01],
             [5.04317660e-01, 4.97523483e-01, 4.92879482e-01,
              4.90522941e-01, 4.90521579e-01, 4.92874471e-01,
              4.97514769e-01, 5.04314130e-01, 5.48356836e-01,
              6.02679755e-01, 6.57930117e-01, 7.13582394e-01,
              7.69129132e-01, 8.24101035e-01, 8.78084923e-01],
             [9.05957986e-01, 9.00459829e-01, 9.01710827e-01,
              9.09304816e-01, 9.21567297e-01, 9.36002510e-01,
              9.49878533e-01, 9.60836244e-01, 9.50521017e-01,
              9.42321192e-01, 9.36098294e-01, 9.31447978e-01,
              9.27737112e-01, 9.24164130e-01, 9.19837458e-01],
             [9.08479555e-01, 8.93119640e-01, 8.77756168e-01,
              8.62389039e-01, 8.47018155e-01, 8.31643415e-01,
              8.16264720e-01, 7.98248733e-01, 7.69688456e-01,
              7.41111049e-01, 7.12515170e-01, 6.83899486e-01,
              6.55262669e-01, 6.26603399e-01, 5.97920364e-01],
             [5.71406981e-01, 5.45439361e-01, 5.19471340e-01,
              4.93502919e-01, 4.67534097e-01, 4.41564875e-01,
              4.15595252e-01, 3.91172349e-01, 3.69029170e-01,
              3.46833147e-01, 3.24591169e-01, 3.02310146e-01,
              2.79997004e-01, 2.57658679e-01, 2.35302110e-01]],

            [[1.96102817e-02, 2.23037080e-02, 2.49835320e-02,
              2.76497605e-02, 3.03024001e-02, 3.29414575e-02,
              3.55669395e-02, 3.81788529e-02, 5.03598778e-02,
              6.89209657e-02, 8.74757090e-02, 1.06024973e-01,
              1.24569626e-01, 1.43110536e-01, 1.61648577e-01],
             [1.82340027e-01, 2.15315774e-01, 2.53562955e-01,
              2.95884521e-01, 3.41038527e-01, 3.87773687e-01,
              4.34864157e-01, 4.81142673e-01, 5.00410360e-01,
              5.19991397e-01, 5.47394263e-01, 5.82556639e-01,
              6.25097005e-01, 6.74344521e-01, 7.29379582e-01],
             [7.75227971e-01, 8.13001048e-01, 8.59395545e-01,
              9.04577146e-01, 9.40342288e-01, 9.61653621e-01,
              9.67479211e-01, 9.60799542e-01, 9.63421077e-01,
              9.66445062e-01, 9.67352042e-01, 9.63790783e-01,
              9.53840372e-01, 9.36234978e-01, 9.10530024e-01],
             [8.86771441e-01, 8.67903107e-01, 8.48953980e-01,
              8.29924111e-01, 8.10813555e-01, 7.91622365e-01,
              7.72350598e-01, 7.51439565e-01, 7.24376642e-01,
              6.97504841e-01, 6.70822717e-01, 6.44328750e-01,
              6.18021348e-01, 5.91898843e-01, 5.65959492e-01],
             [5.40017537e-01, 5.14048293e-01, 4.88079755e-01,
              4.62111921e-01, 4.36144791e-01, 4.10178361e-01,
              3.84212632e-01, 3.58028450e-01, 3.31935148e-01,
              3.06445966e-01, 2.81566598e-01, 2.57302099e-01,
              2.33656886e-01, 2.10634733e-01, 1.88238767e-01]]])

        np.testing.assert_allclose(values, expected)

        # try it with an RGB
        arr = np.arange(75).reshape(5, 15) / 74.
        alpha = arr > 40.
        data = xr.DataArray([arr.copy(), alpha],
                            dims=['bands', 'y', 'x'],
                            coords={'bands': ['L', 'A']})
        img = xrimage.XRImage(data)
        img.colorize(brbg)

        values = img.data.values
        expected = np.concatenate((expected,
                                   alpha.reshape((1,) + alpha.shape)))
        np.testing.assert_allclose(values, expected)

    def test_colorize_rgba(self):
        """Test colorize with an RGBA colormap."""
        import xarray as xr
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

    def test_palettize(self):
        """Test palettize with an RGB colormap."""
        import xarray as xr
        from trollimage import xrimage
        from trollimage.colormap import brbg

        arr = np.arange(75).reshape(5, 15) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x'])
        img = xrimage.XRImage(data)
        img.palettize(brbg)

        values = img.data.values
        expected = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10]]])
        np.testing.assert_allclose(values, expected)

    def test_palettize_rgba(self):
        """Test palettize with an RGBA colormap."""
        import xarray as xr
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

    def test_stack(self):
        """Test stack."""
        import xarray as xr
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
        import xarray as xr
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
        import xarray as xr
        from trollimage import xrimage

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3) / 75., dims=[
            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage.PILImage.Image, 'show', return_value=None) as s:
            img.show()
            s.assert_called_once()

    def test_apply_pil(self):
        """Test the apply_pil method."""
        import xarray as xr
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


def _get_tags_after_writing_to_geotiff(data):
    from trollimage import xrimage
    import rasterio as rio

    img = xrimage.XRImage(data)
    with NamedTemporaryFile(suffix='.tif') as tmp:
        img.save(tmp.name)
        with rio.open(tmp.name) as f:
            return f.tags()
