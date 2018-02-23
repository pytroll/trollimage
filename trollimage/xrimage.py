#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2009-2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>

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

"""This module defines the image class. It overlaps largely the PIL library,
but has the advandage of using masked arrays as pixel arrays, so that data
arrays containing invalid values may be properly handled.
"""

import logging
import os

import numpy as np
import six
from PIL import Image as Pil

import rasterio
from rasterio.windows import Window
import xarray as xr
import xarray.ufuncs as xu
import dask.array as da


logger = logging.getLogger(__name__)



class RIOFile(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.rfile = None


    def __setitem__(self, key, item):
        chy_off = key[1].start
        chy = key[1].stop - key[1].start
        chx_off = key[2].start
        chx = key[2].stop - key[2].start

        self.rfile.write(item, window=Window(chx_off, chy_off, chx, chy))

    def __enter__(self):
        self.rfile = rasterio.open(*self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.rfile.close()

def color_interp(data):
    """Get the color interpretation for this image."""
    from rasterio.enums import ColorInterp as ci
    modes = {'L': [ci.gray],
             'LA': [ci.gray, ci.alpha],
             'YCbCr': [ci.Y, ci.Cb, ci.Cr],
             'YCbCrA': [ci.Y, ci.Cb, ci.Cr, ci.alpha]}

    try:
        mode = ''.join(data['bands'].values)
        return modes[mode]
    except KeyError:
        colors = {'R': ci.red,
                  'G': ci.green,
                  'B': ci.blue,
                  'A': ci.alpha,
                  'C': ci.cyan,
                  'M': ci.magenta,
                  'Y': ci.yellow,
                  'H': ci.hue,
                  'S': ci.saturation,
                  'L': ci.lightness,
                  'K': ci.black,
                  }
        return [colors[band] for band in data['bands'].values]

class XRImage(object):

    modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def __init__(self, data, mode="RGB"):

        try:
            dims = data.dims
        except AttributeError:
            raise TypeError("Data must be a dims attribute.")

        if "bands" not in dims:
            if data.ndim <= 2:
                self.data = data.expand_dims('bands')
                self.data['bands'] = ['L']
            else:
                raise ValueError("No 'bands' dimension provided.")
        else:
            self.data = data

    @property
    def mode(self):
        return ''.join(self.data['bands'].values)


    def save(self, filename, compression=6, fformat=None, fill_value=None):
        """Save the image to the given *filename*.

        For some formats like jpg and png, the work is delegated to
        :meth:`pil_save`, which doesn't support the *compression* option.
        """
        #self.pil_save(filename, compression, fformat, fill_value)
        self.rio_save(filename, fformat, fill_value)

    def rio_save(self, filename, fformat=None, fill_value=None):
        """Save the image using rasterio."""
        fformat = fformat or os.path.splitext(filename)[1][1:4]
        drivers = {'jpg': 'JPEG',
                   'png': 'PNG',
                   'tif': 'GTiff'}
        driver = drivers.get(fformat, fformat)

        data, mode = self._finalize()
        data = data.transpose('bands', 'y', 'x')
        data.attrs = self.data.attrs

        new_tags = {}
        crs = None
        transform = None
        if driver == 'GTiff':
            try:
                crs = rasterio.crs.CRS(data.attrs['area'].proj_dict)
                west, south, east, north = data.attrs['area'].area_extent
                height, width = data.sizes['y'], data.sizes['x']
                transform = rasterio.transform.from_bounds(west, south,
                                                           east, north,
                                                           width, height)
                if "start_time" in data.attrs:
                    new_tags = {'TIFFTAG_DATETIME': data.attrs["start_time"].strftime("%Y:%m:%d %H:%M:%S")}

            except KeyError:
                pass
        elif driver == 'JPEG' and 'A' in mode:
            raise ValueError('JPEG does not support alpha')

        # FIXME photometric works only for GTiff
        # FIXME add png metadata

        with RIOFile(filename, 'w', driver=driver,
                     width=data.sizes['x'], height=data.sizes['y'],
                     count=data.sizes['bands'],
                     dtype=data.dtype.type,
                     crs=crs, transform=transform) as r_file:

            r_file.rfile.colorinterp = color_interp(data)

            r_file.rfile.update_tags(**new_tags)

            try:
                data = data.data.rechunk(chunks=(data.sizes['bands'],
                                                 4096, 4096))
            except AttributeError:
                data = da.from_array(data, chunks=(data.sizes['bands'],
                                                 4096, 4096))
            da.store(data, r_file, lock=False)

    def pil_save(self, filename, compression=6, fformat=None, fill_value=None):
        """Save the image to the given *filename* using PIL.

        For now, the compression level [0-9] is ignored, due to PIL's lack of
        support. See also :meth:`save`.
        """
        # PIL does not support compression option.
        del compression

        if isinstance(filename, (str, six.text_type)):
            ensure_dir(filename)

        fformat = fformat or os.path.splitext(filename)[1][1:4]
        fformat = check_image_format(fformat)

        params = {}

        if fformat == 'png':
            # Take care of GeoImage.tags (if any).
            params['pnginfo'] = self._pngmeta()

        img = self.pil_image(fill_value)
        img.save(filename, fformat, **params)

    def _pngmeta(self):
        """It will return GeoImage.tags as a PNG metadata object.

        Inspired by:
        public domain, Nick Galbreath
        http://blog.modp.com/2007/08/python-pil-and-png-metadata-take-2.html
        """
        reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')

        try:
            tags = self.tags
        except AttributeError:
            tags = {}

        # Undocumented class
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()

        # Copy from tags to new dict
        for k__, v__ in tags.items():
            if k__ not in reserved:
                meta.add_text(k__, v__, 0)

        return meta

    def fill_or_alpha(self, data, fill_value=None):
        import xarray.ufuncs as xu
        import xarray as xr
        nan_mask = xu.isnan(data).sum('bands').expand_dims('bands').astype(bool)
        nan_mask['bands'] = ['A']

        # if not any(nan_mask):
        #     return data
        if fill_value is None:
            if self.mode.endswith("A"):
                data.sel(bands='A')[nan_mask] = 0
            else:
                data = xr.concat([data, (1 - nan_mask).astype(data.dtype)],
                                 dim="bands")
        else:
            # FIXME: is this inplace ???
            data.fillna(fill_value)
        return data

    def _finalize(self, fill_value=None, dtype=np.uint8):
        """Finalize the image.

        This sets the channels in unsigned 8bit format ([0,255] range)
        (if the *dtype* doesn't say otherwise).
        """

        # if self.mode == "P":
        #     self.convert("RGB")
        # if self.mode == "PA":
        #     self.convert("RGBA")
        #
        final_data = self.fill_or_alpha(self.data, fill_value)
        #final_data = self.data
        if np.issubdtype(dtype, np.integer):
            final_data = final_data.clip(0, 1) * np.iinfo(dtype).max
            final_data = final_data.round().astype(dtype)

        final_data.attrs = self.data.attrs

        return final_data, ''.join(final_data['bands'].values)

    def pil_image(self, fill_value=None):
        """Return a PIL image from the current image."""
        channels, mode = self._finalize(fill_value)
        return Pil.fromarray(np.asanyarray(channels.transpose('y', 'x', 'bands')), mode)

    def xrify_tuples(self, tup):
        return xr.DataArray(tup,
                            dims=['bands'],
                            coords={'bands': self.data['bands']})

    def gamma(self, gamma=1.0):
        """Apply gamma correction to the channels of the image.

        If *gamma* is a
        tuple, then it should have as many elements as the channels of the
        image, and the gamma correction is applied elementwise. If *gamma* is a
        number, the same gamma correction is applied on every channel, if there
        are several channels in the image. The behaviour of :func:`gamma` is
        undefined outside the normal [0,1] range of the channels.
        """

        if isinstance(gamma, (list, tuple)):
           gamma = self.xrify_tuples(gamma)
        elif gamma == 1.0:
            return

        logger.debug("Applying gamma %s", str(gamma))
        attrs = self.data.attrs
        self.data = self.data.clip(min=0)
        self.data **= 1.0 / gamma
        self.data.attrs = attrs

    def stretch(self, stretch="crude", **kwargs):
        """Apply stretching to the current image. The value of *stretch* sets
        the type of stretching applied. The values "histogram", "linear",
        "crude" (or "crude-stretch") perform respectively histogram
        equalization, contrast stretching (with 5% cutoff on both sides), and
        contrast stretching without cutoff. The value "logarithmic" or "log"
        will do a logarithmic enhancement towards white. If a tuple or a list
        of two values is given as input, then a contrast stretching is performed
        with the values as cutoff. These values should be normalized in the
        range [0.0,1.0].
        """

        logger.debug("Applying stretch %s with parameters %s",
                     stretch, str(kwargs))

        # FIXME: do not apply stretch to alpha channel

        if isinstance(stretch, (tuple, list)):
            if len(stretch) == 2:
                self.stretch_linear(cutoffs=stretch)
            else:
                raise ValueError(
                    "Stretch tuple must have exactly two elements")
        elif stretch == "linear":
            self.stretch_linear(**kwargs)
        elif stretch == "histogram":
            self.stretch_hist_equalize(**kwargs)
        elif stretch in ["crude", "crude-stretch"]:
            self.crude_stretch(**kwargs)
        elif stretch in ["log", "logarithmic"]:
            self.stretch_logarithmic(**kwargs)
        elif stretch == "no":
            return
        elif isinstance(stretch, str):
            raise ValueError("Stretching method %s not recognized." % stretch)
        else:
            raise TypeError("Stretch parameter must be a string or a tuple.")

    def stretch_linear(self, cutoffs=(0.005, 0.005)):
        """Stretch linearly the contrast of the current image.

        Use *cutoffs* for left and right trimming.
        """
        logger.debug("Perform a linear contrast stretch.")

        logger.debug("Calculate the histogram quantiles: ")
        logger.debug("Left and right quantiles: " +
                     str(cutoffs[0]) + " " + str(cutoffs[1]))
        # Quantile requires the data to be loaded, not supported on dask arrays
        self.data.load()
        left, right = self.data.quantile([cutoffs[0], 1. - cutoffs[1]],
                                         dim=['x', 'y'])
        logger.debug("Interval: left=%s, right=%s", str(left), str(right))
        self.crude_stretch(left, right)

    def crude_stretch(self, min_stretch=None, max_stretch=None):
        """Perform simple linear stretching.

        This is done without any cutoff on the current image and normalizes to
        the [0,1] range.
        """
        if min_stretch is None:
            min_stretch = self.data.min()
        if max_stretch is None:
            max_stretch = self.data.max()

        if isinstance(min_stretch, (list, tuple)):
            min_stretch = self.xrify_tuples(min_stretch)
        if isinstance(max_stretch, (list, tuple)):
            max_stretch = self.xrify_tuples(max_stretch)

        # FIXME this doesn't work on single values !
        delta = (max_stretch - min_stretch)
        scale_factor = (1 / delta).fillna(0)
        attrs = self.data.attrs
        self.data -= min_stretch
        self.data *= scale_factor
        self.data.attrs = attrs

    # def stretch_hist_equalize(self):
    #     """Stretch the current image's colors through histogram equalization."""
    #     logger.info("Perform a histogram equalized contrast stretch.")
    #
    #     if(self.channels[ch_nb].size ==
    #        np.ma.count_masked(self.channels[ch_nb])):
    #         logger.warning("Nothing to stretch !")
    #         return
    #
    #     arr = self.data
    #
    #     nwidth = 2048.0
    #
    #     carr = arr.compressed()
    #
    #     cdf = np.arange(0.0, 1.0, 1 / nwidth)
    #     logger.debug("Make histogram bins having equal amount of data, " +
    #                  "using numpy percentile function:")
    #     bins = np.percentile(carr, list(cdf * 100))
    #
    #     res = np.ma.empty_like(arr)
    #     res.mask = np.ma.getmaskarray(arr)
    #     res[~res.mask] = np.interp(carr, bins, cdf)
    #
    #     self.channels[ch_nb] = res

    def stretch_logarithmic(self, ch_nb, factor=100.):
        """Move data into range [1:factor] and do a normalized logarithmic
        enhancement.
        """
        logger.debug("Perform a logarithmic contrast stretch.")
        if ((self.channels[ch_nb].size ==
             np.ma.count_masked(self.channels[ch_nb])) or
                (self.channels[ch_nb].min() == self.channels[ch_nb].max())):
            logger.warning("Nothing to stretch !")
            return

        crange = (0., 1.0)

        arr = self.channels[ch_nb]
        b__ = float(crange[1] - crange[0]) / np.log(factor)
        c__ = float(crange[0])
        slope = (factor - 1.) / float(arr.max() - arr.min())
        arr = 1. + (arr - arr.min()) * slope
        arr = c__ + b__ * np.log(arr)
        self.channels[ch_nb] = arr

    def stretch_weber_fechner(self, k, s0):
        """Stretch according to the Weber-Fechner law.

        p = k.ln(S/S0)
        p is perception, S is the stimulus, S0 is the stimulus threshold (the
        highest unpercieved stimulus), and k is the factor.
        """
        attrs = self.data.attrs
        self.data = k*xu.log(self.data / s0)
        self.data.attrs = attrs

    def invert(self, invert=True):
        """Inverts all the channels of a image according to *invert*.

        If invert is a tuple or a list, elementwise invertion is performed,
        otherwise all channels are inverted if *invert* is true (default).

        Note: 'Inverting' means that black becomes white, and vice-versa, not
        that the values are negated !
        """
        logger.debug("Applying invert with parameters %s", str(invert))
        if isinstance(invert, (tuple, list)):
            invert = self.xrify_tuples(invert)
            offset = invert.astype(int)
            scale = (-1) ** offset
        elif invert:
            offset = 1
            scale = -1
        attrs = self.data.attrs
        self.data = self.data * scale + offset
        self.data.attrs = attrs

    def show(self):
        """Display the image on screen."""
        self.pil_image().show()