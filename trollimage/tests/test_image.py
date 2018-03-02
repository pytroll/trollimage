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
"""Module for testing the imageo.image module.
"""
import random
import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from trollimage import image

EPSILON = 0.0001


class TestEmptyImage(unittest.TestCase):

    """Class for testing the mpop.imageo.image module
    """

    def setUp(self):
        """Setup the test.
        """
        self.img = image.Image()
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_shape(self):
        """Shape of an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertEqual(self.img.shape, (0, 0))
        self.img.convert(oldmode)

    def test_is_empty(self):
        """Test if an image is empty.
        """
        self.assertEqual(self.img.is_empty(), True)

    def test_clip(self):
        """Clip an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertEqual(self.img.channels, [])
        self.img.convert(oldmode)

    def test_convert(self):
        """Convert an empty image.
        """
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
        """Stretch an empty image
        """
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
        """Gamma correction on an empty image.
        """
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
        """Invert an empty image.
        """
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
        """Return an empty PIL image.
        """
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
        """Add an alpha channel to en empty image
        """
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
        """Save an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertRaises(IOError, self.img.save, "test.png")

        self.img.convert(oldmode)

    def test_replace_luminance(self):
        """Replace luminance in an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.replace_luminance([])
            self.assertEqual(self.img.mode, mode)
            self.assertEqual(self.img.channels, [])
            self.assertEqual(self.img.shape, (0, 0))
        self.img.convert(oldmode)

    def test_resize(self):
        """Resize an empty image.
        """
        self.assertRaises(ValueError, self.img.resize, (10, 10))

    def test_merge(self):
        """Merging of an empty image with another.
        """
        newimg = image.Image()
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2, 3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)


class TestImageCreation(unittest.TestCase):

    """Class for testing the mpop.imageo.image module
    """

    def setUp(self):
        """Setup the test.
        """
        self.img = {}
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]
        self.modes_len = [1, 2, 3, 4, 3, 4, 1, 2]

    def test_creation(self):
        """Creation of an image.
        """

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

    """Class for testing the mpop.imageo.image module
    """

    def setUp(self):
        """Setup the test.
        """
        import os
        import tempfile
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
        os.chmod(self.tempdir, 0000)

    def test_shape(self):
        """Shape of an image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            self.assertEqual(self.img.shape, (2, 3))
        self.img.convert(oldmode)

    def test_is_empty(self):
        """Test if an image is empty.
        """
        self.assertEqual(self.img.is_empty(), False)

    def test_clip(self):
        """Clip an image.
        """
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
        """Convert an image.
        """
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
        """Stretch an image.
        """
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
        """Gamma correction on an image.
        """
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
        """Invert an image.
        """
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
        """Return an PIL image.
        """

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
        """Add an alpha channel.
        """
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

    def test_save(self):
        """Save an image.
        """
        import os
        import os.path

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

    def test_save_jpeg(self):
        """Save a jpeg image.
        """
        import os
        import os.path

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
        """Replace luminance in an image.
        """
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
            if(self.img.mode.endswith("A")):
                chans = self.img.channels[:-1]
            else:
                chans = self.img.channels
            for chn in chans:
                self.assertTrue(np.all(chn - luma < EPSILON))
        self.img.convert(oldmode)

    def test_resize(self):
        """Resize an image.
        """
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
        """Merging of an image with another.
        """
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
        """Clean up the mess.
        """
        import os
        os.rmdir(self.tempdir)


class TestFlatImage(unittest.TestCase):

    """Test a flat image, ie an image where min == max.
    """

    def setUp(self):
        channel = np.ma.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]],
                              mask=[[1, 1, 1], [1, 1, 0]])
        self.img = image.Image(channels=[channel] * 3,
                               mode="RGB")
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_stretch(self):
        """Stretch a flat image.
        """
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

    """Test an image filled with no data.
    """

    def setUp(self):
        channel = np.ma.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]],
                              mask=[[1, 1, 1], [1, 1, 1]])
        self.img = image.Image(channels=[channel] * 3,
                               mode="RGB")
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_stretch(self):
        """Stretch a no data image.
        """
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
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    return "".join([random.choice(choices)
                    for dummy in range(length)])


class TestXRImage(unittest.TestCase):

    def test_init(self):
        import xarray as xr
        from trollimage import xrimage
        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=['y', 'x'])
        img = xrimage.XRImage(data)
        self.assertEqual(img.mode, 'L')

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=[
                            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        self.assertEqual(img.mode, 'RGB')

        data = xr.DataArray(np.arange(100).reshape(5, 5, 4), dims=[
                            'y', 'x', 'bands'], coords={'bands': ['Y', 'Cb', 'Cr', 'A']})
        img = xrimage.XRImage(data)
        self.assertEqual(img.mode, 'YCbCrA')

    def test_save(self):
        import xarray as xr
        import dask.array as da
        from trollimage import xrimage

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3) / 75., dims=[
                            'y', 'x', 'bands'], coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)

        data = xr.DataArray(da.from_array(np.arange(75).reshape(5, 5, 3) / 75.,
                                          chunks=5),
                            dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)
        data = data.where(data > (10 / 75.0))
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix='.png') as tmp:
            img.save(tmp.name)

    def test_gamma(self):
        """Test gamma correction."""
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 75.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.gamma(.5)
        self.assertTrue(np.allclose(img.data.values, arr ** 2))

        img.gamma([2., 2., 2.])
        self.assertTrue(np.allclose(img.data.values, arr))

    def test_crude_stretch(self):
        """Check crude stretching."""
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.crude_stretch()
        self.assertTrue(np.allclose(img.data.values, arr))

        arr = np.arange(75).reshape(5, 5, 3).astype(float)
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.crude_stretch(0, 74)
        self.assertTrue(np.allclose(img.data.values, arr / 74.))

    def test_invert(self):
        """Check inversion of the image."""
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 75.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)

        img.invert(True)

        self.assertTrue(np.allclose(img.data.values, 1 - arr))

        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)

        img.invert([True, False, True])
        offset = xr.DataArray(np.array([1, 0, 1]), dims=['bands'],
                              coords={'bands': ['R', 'G', 'B']})
        scale = xr.DataArray(np.array([-1, 1, -1]), dims=['bands'],
                             coords={'bands': ['R', 'G', 'B']})
        self.assertTrue(np.allclose(img.data.values, (data * scale + offset).values))

    def test_linear_stretch(self):
        """Test linear stretching with cutoffs."""
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch_linear()
        res = np.array([[[-0.005051, -0.005051, -0.005051],
                         [ 0.037037,  0.037037,  0.037037],
                         [ 0.079125,  0.079125,  0.079125],
                         [ 0.121212,  0.121212,  0.121212],
                         [ 0.1633  ,  0.1633  ,  0.1633  ]],
                        [[ 0.205387,  0.205387,  0.205387],
                         [ 0.247475,  0.247475,  0.247475],
                         [ 0.289562,  0.289562,  0.289562],
                         [ 0.33165 ,  0.33165 ,  0.33165 ],
                         [ 0.373737,  0.373737,  0.373737]],
                        [[ 0.415825,  0.415825,  0.415825],
                         [ 0.457912,  0.457912,  0.457912],
                         [ 0.5     ,  0.5     ,  0.5     ],
                         [ 0.542088,  0.542088,  0.542088],
                         [ 0.584175,  0.584175,  0.584175]],
                        [[ 0.626263,  0.626263,  0.626263],
                         [ 0.66835 ,  0.66835 ,  0.66835 ],
                         [ 0.710438,  0.710438,  0.710438],
                         [ 0.752525,  0.752525,  0.752525],
                         [ 0.794613,  0.794613,  0.794613]],
                        [[ 0.8367  ,  0.8367  ,  0.8367  ],
                         [ 0.878788,  0.878788,  0.878788],
                         [ 0.920875,  0.920875,  0.920875],
                         [ 0.962963,  0.962963,  0.962963],
                         [ 1.005051,  1.005051,  1.005051]]])

        self.assertTrue(np.allclose(img.data.values, res, atol=1.e-6))

    def test_histogram_stretch(self):
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch('histogram')
        res = np.array([[[ 0.        ,  0.        ,  0.        ],
                         [ 0.04166667,  0.04166667,  0.04166667],
                         [ 0.08333333,  0.08333333,  0.08333333],
                         [ 0.125     ,  0.125     ,  0.125     ],
                         [ 0.16666667,  0.16666667,  0.16666667]],

                        [[ 0.20833333,  0.20833333,  0.20833333],
                         [ 0.25      ,  0.25      ,  0.25      ],
                         [ 0.29166667,  0.29166667,  0.29166667],
                         [ 0.33333333,  0.33333333,  0.33333333],
                         [ 0.375     ,  0.375     ,  0.375     ]],

                        [[ 0.41666667,  0.41666667,  0.41666667],
                         [ 0.45833333,  0.45833333,  0.45833333],
                         [ 0.5       ,  0.5       ,  0.5       ],
                         [ 0.54166667,  0.54166667,  0.54166667],
                         [ 0.58333333,  0.58333333,  0.58333333]],

                        [[ 0.625     ,  0.625     ,  0.625     ],
                         [ 0.66666667,  0.66666667,  0.66666667],
                         [ 0.70833333,  0.70833333,  0.70833333],
                         [ 0.75      ,  0.75      ,  0.75      ],
                         [ 0.79166667,  0.79166667,  0.79166667]],

                        [[ 0.83333333,  0.83333333,  0.83333333],
                         [ 0.875     ,  0.875     ,  0.875     ],
                         [ 0.91666667,  0.91666667,  0.91666667],
                         [ 0.95833333,  0.95833333,  0.95833333],
                         [ 0.99951172,  0.99951172,  0.99951172]]])

        self.assertTrue(np.allclose(img.data.values, res, atol=1.e-6))

    def test_logarithmic_stretch(self):
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch(stretch='logarithmic')
        res = np.array([[[ 0.        ,  0.        ,  0.        ],
                         [ 0.35484693,  0.35484693,  0.35484693],
                         [ 0.48307087,  0.48307087,  0.48307087],
                         [ 0.5631469 ,  0.5631469 ,  0.5631469 ],
                         [ 0.62151902,  0.62151902,  0.62151902]],

                        [[ 0.66747806,  0.66747806,  0.66747806],
                         [ 0.70538862,  0.70538862,  0.70538862],
                         [ 0.73765396,  0.73765396,  0.73765396],
                         [ 0.76573946,  0.76573946,  0.76573946],
                         [ 0.79060493,  0.79060493,  0.79060493]],

                        [[ 0.81291336,  0.81291336,  0.81291336],
                         [ 0.83314196,  0.83314196,  0.83314196],
                         [ 0.85164569,  0.85164569,  0.85164569],
                         [ 0.86869572,  0.86869572,  0.86869572],
                         [ 0.88450394,  0.88450394,  0.88450394]],

                        [[ 0.899239  ,  0.899239  ,  0.899239  ],
                         [ 0.9130374 ,  0.9130374 ,  0.9130374 ],
                         [ 0.92601114,  0.92601114,  0.92601114],
                         [ 0.93825325,  0.93825325,  0.93825325],
                         [ 0.94984187,  0.94984187,  0.94984187]],

                        [[ 0.96084324,  0.96084324,  0.96084324],
                         [ 0.97131402,  0.97131402,  0.97131402],
                         [ 0.98130304,  0.98130304,  0.98130304],
                         [ 0.99085269,  0.99085269,  0.99085269],
                         [ 1.        ,  1.        ,  1.        ]]])

        self.assertTrue(np.allclose(img.data.values, res, atol=1.e-6))

    def test_weber_fechner_stretch(self):
        """S=2.3klog10I+C """
        import xarray as xr
        from trollimage import xrimage

        arr = np.arange(75).reshape(5, 5, 3) / 74.
        data = xr.DataArray(arr.copy(), dims=['y', 'x', 'bands'],
                            coords={'bands': ['R', 'G', 'B']})
        img = xrimage.XRImage(data)
        img.stretch_weber_fechner(2.5, 0.2)
        res = np.array([[[    -np.inf, -6.73656795, -5.0037    ],
                         [-3.99003723, -3.27083205, -2.71297317],
                         [-2.25716928, -1.87179258, -1.5379641 ],
                         [-1.24350651, -0.98010522, -0.74182977],
                         [-0.52430133, -0.32419456, -0.13892463]],

                        [[ 0.03355755,  0.19490385,  0.34646541],
                         [ 0.48936144,  0.6245295 ,  0.75276273],
                         [ 0.87473814,  0.99103818,  1.10216759],
                         [ 1.20856662,  1.31062161,  1.40867339],
                         [ 1.50302421,  1.59394332,  1.68167162]],

                        [[ 1.7664255 ,  1.84840006,  1.92777181],
                         [ 2.00470095,  2.07933336,  2.1518022 ],
                         [ 2.22222939,  2.29072683,  2.35739745],
                         [ 2.42233616,  2.48563068,  2.54736221],
                         [ 2.60760609,  2.66643234,  2.72390613]],

                        [[ 2.78008827,  2.83503554,  2.88880105],
                         [ 2.94143458,  2.99298279,  3.04348956],
                         [ 3.09299613,  3.14154134,  3.18916183],
                         [ 3.23589216,  3.28176501,  3.32681127],
                         [ 3.37106022,  3.41453957,  3.45727566]],

                        [[ 3.49929345,  3.54061671,  3.58126801],
                         [ 3.62126886,  3.66063976,  3.69940022],
                         [ 3.7375689 ,  3.7751636 ,  3.81220131],
                         [ 3.84869831,  3.88467015,  3.92013174],
                         [ 3.95509735,  3.98958065,  4.02359478]]])

        self.assertTrue(np.allclose(img.data.values, res, atol=1.e-6))

    def test_jpeg_save(self):
        pass

    def test_gtiff_save(self):
        pass

    def test_save_masked(self):
        pass

    def test_LA_save(self):
        pass

    def test_L_save(self):
        pass

    def test_P_save(self):
        pass

    def test_PA_save(self):
        pass

    def test_convert_modes(self):
        pass

    def test_colorize(self):
        pass

    def test_palettize(self):
        pass

    def test_merge(self):
        pass

    def test_blend(self):
        pass

    def test_replace_luminance(self):
        pass

    def test_putalpha(self):
        pass

    def test_show(self):
        pass


def suite():
    """The suite for test_image."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestEmptyImage))
    mysuite.addTest(loader.loadTestsFromTestCase(TestImageCreation))
    mysuite.addTest(loader.loadTestsFromTestCase(TestRegularImage))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFlatImage))
    mysuite.addTest(loader.loadTestsFromTestCase(TestNoDataImage))
    mysuite.addTest(loader.loadTestsFromTestCase(TestXRImage))

    return mysuite


if __name__ == '__main__':
    unittest.main()
