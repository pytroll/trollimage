#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2018
#
# Author(s):
#
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""This module defines the XRImage class. It overlaps largely with the PIL
library, but has the advantage of using :class:`~xarray.DataArray` objects
backed by :class:`dask arrays <dask.array.Array>` as pixel arrays. This
allows for invalid values to be tracked, metadata to be assigned, and
stretching to be lazy evaluated. With the optional ``rasterio`` library
installed dask array chunks can be saved in parallel.

"""

import logging
import os

import numpy as np
from PIL import Image as PILImage
import xarray as xr
import xarray.ufuncs as xu
import dask
import dask.array as da

from trollimage.image import check_image_format

try:
    import rasterio
except ImportError:
    rasterio = None


try:
    # rasterio 1.0+
    from rasterio.windows import Window
except ImportError:
    # raster 0.36.0
    # remove this once rasterio 1.0+ is officially available
    def Window(x_off, y_off, x_size, y_size):
        """Replace the missing Window object in rasterio < 1.0."""
        return (y_off, y_off + y_size), (x_off, x_off + x_size)


logger = logging.getLogger(__name__)


class RIOFile(object):
    """Rasterio wrapper to allow da.store to do window saving."""

    def __init__(self, path, mode='w', **kwargs):
        """Initialize the object."""
        self.path = path
        self.mode = mode
        self.kwargs = kwargs
        self.rfile = None
        self._closed = True

    def __setitem__(self, key, item):
        """Put the data chunk in the image."""
        if len(key) == 3:
            indexes = list(range(
                key[0].start + 1,
                key[0].stop + 1,
                key[0].step or 1
            ))
            y = key[1]
            x = key[2]
        else:
            indexes = 1
            y = key[0]
            x = key[1]
        chy_off = y.start
        chy = y.stop - y.start
        chx_off = x.start
        chx = x.stop - x.start

        # band indexes
        self.rfile.write(item, window=Window(chx_off, chy_off, chx, chy),
                         indexes=indexes)

    def open(self, mode=None):
        mode = mode or self.mode
        if self._closed:
            self.rfile = rasterio.open(self.path, mode, **self.kwargs)
            self._closed = False

    def close(self):
        if not self._closed:
            self.rfile.close()
            self._closed = True

    def __enter__(self):
        """Enter method."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        self.close()

    def __del__(self):
        try:
            self.close()
        except (IOError, OSError):
            pass

    @property
    def colorinterp(self):
        """Return the color interpretation of the image."""
        return self.rfile.colorinterp

    @colorinterp.setter
    def colorinterp(self, val):
        if rasterio.__version__.startswith("0."):
            # not supported in older versions, set by PHOTOMETRIC tag
            logger.warning("Rasterio 1.0+ required for setting colorinterp")
        else:
            self.rfile.colorinterp = val


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
    """Image class using an :class:`xarray.DataArray` as internal storage."""

    def __init__(self, data):
        """Initialize the image with a :class:`~xarray.DataArray`."""
        data = self._correct_dims(data)

        # 'data' is an XArray, get the data from it as a dask array
        if not isinstance(data.data, da.Array):
            logger.debug("Convert image data to dask array")
            data.data = da.from_array(data.data, chunks=(data.sizes['bands'], 4096, 4096))

        self.data = data
        self.height, self.width = self.data.sizes['y'], self.data.sizes['x']
        self.palette = None

    @staticmethod
    def _correct_dims(data):
        """Standardize dimensions to bands, y, and x."""
        if not hasattr(data, 'dims'):
            raise TypeError("Data must have a 'dims' attribute.")

        # doesn't actually copy the data underneath
        # we don't want our operations to change the user's data
        data = data.copy()

        if 'y' not in data.dims or 'x' not in data.dims:
            if data.ndim != 2:
                raise ValueError("Data must have a 'y' and 'x' dimension")

            # rename dimensions so we can use them
            # don't rename 'x' or 'y' if they already exist
            if 'y' not in data.dims:
                # find a dimension that isn't 'x'
                old_dim = [d for d in data.dims if d != 'x'][0]
                data = data.rename({old_dim: 'y'})
            if 'x' not in data.dims:
                # find a dimension that isn't 'y'
                old_dim = [d for d in data.dims if d != 'y'][0]
                data = data.rename({old_dim: 'x'})

        if "bands" not in data.dims:
            if data.ndim <= 2:
                data = data.expand_dims('bands')
                data['bands'] = ['L']
            else:
                raise ValueError("No 'bands' dimension provided.")

        return data

    @property
    def mode(self):
        """Mode of the image."""
        return ''.join(self.data['bands'].values)

    def save(self, filename, fformat=None, fill_value=None, compute=True,
             **format_kwargs):
        """Save the image to the given *filename*.

        Args:
            filename (str): Output filename
            fformat (str): File format of output file (optional). Can be
                           one of many image formats supported by the
                           `rasterio` or `PIL` libraries ('jpg', 'png',
                           'tif'). By default this is determined by the
                           extension of the provided filename.
            fill_value (float): Replace invalid data values with this value
                                and do not produce an Alpha band. Default
                                behavior is to create an alpha band.
            compute (bool): If True (default) write the data to the file
                            immediately. If False the return value is either
                            a `dask.Delayed` object or a tuple of
                            ``(source, target)`` to be passed to
                            `dask.array.store`.
            format_kwargs: Additional format options to pass to `rasterio`
                           or `PIL` saving methods.

        Returns:
            Either `None` if `compute` is True or a `dask.Delayed` object or
            ``(source, target)`` pair to be passed to `dask.array.store`.
            If compute is False the return value depends on format and how
            the image backend is used. If ``(source, target)`` is provided
            then target is an open file-like object that must be closed by
            the caller.

        """
        fformat = fformat or os.path.splitext(filename)[1][1:4]
        if fformat == 'tif' and rasterio:
            return self.rio_save(filename, fformat=fformat,
                                 fill_value=fill_value, compute=compute,
                                 **format_kwargs)
        else:
            return self.pil_save(filename, fformat, fill_value,
                                 compute=compute, **format_kwargs)

    def rio_save(self, filename, fformat=None, fill_value=None,
                 dtype=np.uint8, compute=True, tags=None, **format_kwargs):
        """Save the image using rasterio."""
        fformat = fformat or os.path.splitext(filename)[1][1:4]
        drivers = {'jpg': 'JPEG',
                   'png': 'PNG',
                   'tif': 'GTiff'}
        driver = drivers.get(fformat, fformat)

        if tags is None:
            tags = {}

        data, mode = self.finalize(fill_value, dtype=dtype)
        data = data.transpose('bands', 'y', 'x')
        data.attrs = self.data.attrs

        crs = None
        transform = None
        if driver == 'GTiff':
            if not np.issubdtype(data.dtype, np.floating):
                format_kwargs.setdefault('compress', 'DEFLATE')
            photometric_map = {
                'RGB': 'RGB',
                'RGBA': 'RGB',
                'CMYK': 'CMYK',
                'CMYKA': 'CMYK',
                'YCBCR': 'YCBCR',
                'YCBCRA': 'YCBCR',
            }
            if mode.upper() in photometric_map:
                format_kwargs.setdefault('photometric',
                                         photometric_map[mode.upper()])

            try:
                crs = rasterio.crs.CRS(data.attrs['area'].proj_dict)
                west, south, east, north = data.attrs['area'].area_extent
                height, width = data.sizes['y'], data.sizes['x']
                transform = rasterio.transform.from_bounds(west, south,
                                                           east, north,
                                                           width, height)
                if "start_time" in data.attrs:
                    stime = data.attrs['start_time']
                    stime_str = stime.strftime("%Y:%m:%d %H:%M:%S")
                    tags.setdefault('TIFFTAG_DATETIME', stime_str)

            except (KeyError, AttributeError):
                logger.info("Couldn't create geotransform")
        elif driver == 'JPEG' and 'A' in mode:
            raise ValueError('JPEG does not support alpha')

        # FIXME add metadata
        r_file = RIOFile(filename, 'w', driver=driver,
                         width=data.sizes['x'], height=data.sizes['y'],
                         count=data.sizes['bands'],
                         dtype=dtype,
                         nodata=fill_value,
                         crs=crs, transform=transform, **format_kwargs)
        r_file.open()
        r_file.colorinterp = color_interp(data)
        r_file.rfile.update_tags(**tags)

        if compute:
            # write data to the file now
            res = da.store(data.data, r_file)
            r_file.close()
            return res
        # provide the data object and the opened file so the caller can
        # store them when they would like. Caller is responsible for
        # closing the file
        return data.data, r_file

    def pil_save(self, filename, fformat=None, fill_value=None,
                 compute=True, **format_kwargs):
        """Save the image to the given *filename* using PIL.

        For now, the compression level [0-9] is ignored, due to PIL's lack of
        support. See also :meth:`save`.
        """
        fformat = fformat or os.path.splitext(filename)[1][1:4]
        fformat = check_image_format(fformat)

        if fformat == 'png':
            # Take care of GeoImage.tags (if any).
            format_kwargs['pnginfo'] = self._pngmeta()

        def _create_save_image(fill_value, filename, fformat, format_kwargs):
            img = self.pil_image(fill_value)
            img.save(filename, fformat, **format_kwargs)
        delay = dask.delayed(_create_save_image)(
            fill_value, filename, fformat, format_kwargs)
        if compute:
            return delay.compute()
        return delay

    def _pngmeta(self):
        """Return GeoImage.tags as a PNG metadata object.

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
        """Fill the data with fill_value or create an alpha channel."""
        if fill_value is None and not self.mode.endswith("A"):
            not_alpha = [b for b in data.coords['bands'].values if b != 'A']
            # if any of the bands are valid, we don't want transparency
            null_mask = data.sel(bands=not_alpha).notnull().any(dim='bands')
            null_mask = null_mask.expand_dims('bands')
            null_mask['bands'] = ['A']
            data = xr.concat([data, null_mask.astype(data.dtype)], dim="bands")
        elif fill_value is not None:
            data = data.fillna(fill_value)
        return data

    def _check_modes(self, modes):
        """Check that the image is in one of the given *modes*, raise an exception otherwise."""
        if not isinstance(modes, (tuple, list, set)):
            modes = [modes]
        if self.mode not in modes:
            raise ValueError("Image not in suitable mode, expected: %s, got: %s" % (modes, self.mode))

    def _from_p(self, mode):
        """Convert the image from P or PA to RGB or RGBA."""
        self._check_modes(("P", "PA"))

        if self.mode.endswith("A"):
            alpha = self.data.sel(bands=["A"]).data
            mode = mode + "A" if not mode.endswith("A") else mode
        else:
            alpha = None

        if not self.palette:
            raise RuntimeError("Can't convert palettized image, missing palette.")

        pal = np.array(self.palette)
        pal = da.from_array(pal, chunks=pal.shape)
        flat_indexes = self.data.data[0].ravel().astype('int64')
        new_shape = (3,) + self.data.shape[1:3]
        new_data = pal[flat_indexes].reshape(new_shape)
        coords = dict(self.data.coords)
        coords["bands"] = list(mode)

        if alpha is not None:
            new_arr = da.concatenate((new_data, alpha), axis=0)
            data = xr.DataArray(new_arr, coords=coords, attrs=self.data.attrs, dims=self.data.dims)
        else:
            data = xr.DataArray(new_data, coords=coords, attrs=self.data.attrs, dims=self.data.dims)

        return data

    def _l2rgb(self, mode):
        """Convert from L (black and white) to RGB.
        """
        self._check_modes(("L", "LA"))

        bands = ["L"] * 3
        if mode[-1] == "A":
            bands.append("A")
        data = self.data.sel(bands=bands)
        data["bands"] = list(mode)
        return data

    def convert(self, mode):
        if mode == self.mode:
            return

        if mode not in ["P", "PA", "L", "LA", "RGB", "RGBA"]:
            raise ValueError("Mode %s not recognized." % (mode))

        if mode == self.mode + "A":
            data = self.fill_or_alpha(self.data).data
            coords = dict(self.data.coords)
            coords["bands"] = list(mode)
            data = xr.DataArray(data, coords=coords, attrs=self.data.attrs, dims=self.data.dims)
            new_img = XRImage(data)
        elif mode + "A" == self.mode:
            # Remove the alpha band from our current image
            no_alpha = self.data.sel(bands=[b for b in self.data.coords["bands"].data if b != "A"]).data
            coords = dict(self.data.coords)
            coords["bands"] = list(mode)
            data = xr.DataArray(no_alpha, coords=coords, attrs=self.data.attrs, dims=self.data.dims)
            new_img = XRImage(data)
        elif mode.endswith("A") and not self.mode.endswith("A"):
            img = self.convert(self.mode + "A")
            new_img = img.convert(mode)
        elif self.mode.endswith("A") and not mode.endswith("A"):
            img = self.convert(self.mode[:-1])
            new_img = img.convert(mode)
        else:
            cases = {
                "P": {"RGB": self._from_p},
                "PA": {"RGBA": self._from_p},
                "L": {"RGB": self._l2rgb},
                "LA": {"RGBA": self._l2rgb}
            }
            try:
                data = cases[self.mode][mode](mode)
                new_img = XRImage(data)
            except KeyError:
                raise ValueError("Conversion from %s to %s not implemented !"
                                 % (self.mode, mode))

        if self.mode.startswith('P') and new_img.mode.startswith('P'):
            # need to copy the palette
            new_img.palette = self.palette
        return new_img

    def _finalize(self, fill_value=None, dtype=np.uint8):
        """Wrapper around 'finalize' method for backwards compatibility."""
        import warnings
        warnings.warn("'_finalize' is deprecated, use 'finalize' instead.",
                      DeprecationWarning)
        return self.finalize(fill_value, dtype)

    def finalize(self, fill_value=None, dtype=np.uint8):
        """Finalize the image.

        This sets the channels in unsigned 8bit format ([0,255] range)
        (if the *dtype* doesn't say otherwise).
        """
        if self.mode == "P":
            return self.convert("RGB").finalize(fill_value=fill_value, dtype=dtype)
        if self.mode == "PA":
            return self.convert("RGBA").finalize(fill_value=fill_value, dtype=dtype)

        if np.issubdtype(dtype, np.floating) and fill_value is None:
            logger.warning("Image with floats cannot be transparent, so "
                           "setting fill_value to 0")
            fill_value = 0

        final_data = self.fill_or_alpha(self.data, fill_value)

        if np.issubdtype(dtype, np.integer):
            final_data = final_data.clip(0, 1) * np.iinfo(dtype).max
            final_data = final_data.round().astype(dtype)
        else:
            final_data = final_data.astype(dtype)

        final_data.attrs = self.data.attrs

        return final_data, ''.join(final_data['bands'].values)

    def pil_image(self, fill_value=None):
        """Return a PIL image from the current image."""
        channels, mode = self.finalize(fill_value)
        res = np.asanyarray(channels.transpose('y', 'x', 'bands').values)
        return PILImage.fromarray(np.squeeze(res), mode)

    def xrify_tuples(self, tup):
        """Make xarray.DataArray from tuple."""
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
        """Apply stretching to the current image.

        The value of *stretch* sets the type of stretching applied. The values
        "histogram", "linear", "crude" (or "crude-stretch") perform respectively
        histogram equalization, contrast stretching (with 5% cutoff on both
        sides), and contrast stretching without cutoff. The value "logarithmic"
        or "log" will do a logarithmic enhancement towards white. If a tuple or
        a list of two values is given as input, then a contrast stretching is
        performed with the values as cutoff. These values should be normalized
        in the range [0.0,1.0].
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
        def _compute_quantile(data, cutoffs):
            # delayed will provide us the fully computed xarray with ndarray
            left, right = data.quantile([cutoffs[0], 1. - cutoffs[1]],
                                        dim=['x', 'y'])
            logger.debug("Interval: left=%s, right=%s", str(left), str(right))
            return left.data, right.data

        cutoff_type = np.float64
        # numpy percentile (which quantile calls) returns 64-bit floats
        # unless the value is a higher order float
        if np.issubdtype(self.data.dtype, np.floating) and \
                np.dtype(self.data.dtype).itemsize > 8:
            cutoff_type = self.data.dtype
        left, right = dask.delayed(_compute_quantile, nout=2)(self.data, cutoffs)
        left_data = da.from_delayed(left,
                                    shape=(self.data.sizes['bands'],),
                                    dtype=cutoff_type)
        left = xr.DataArray(left_data, dims=('bands',),
                            coords={'bands': self.data['bands']})
        right_data = da.from_delayed(right,
                                     shape=(self.data.sizes['bands'],),
                                     dtype=cutoff_type)
        right = xr.DataArray(right_data, dims=('bands',),
                             coords={'bands': self.data['bands']})

        self.crude_stretch(left, right)

    def crude_stretch(self, min_stretch=None, max_stretch=None):
        """Perform simple linear stretching.

        This is done without any cutoff on the current image and normalizes to
        the [0,1] range.
        """
        if min_stretch is None:
            non_band_dims = tuple(x for x in self.data.dims if x != 'bands')
            min_stretch = self.data.min(dim=non_band_dims)
        if max_stretch is None:
            non_band_dims = tuple(x for x in self.data.dims if x != 'bands')
            max_stretch = self.data.max(dim=non_band_dims)

        if isinstance(min_stretch, (list, tuple)):
            min_stretch = self.xrify_tuples(min_stretch)
        if isinstance(max_stretch, (list, tuple)):
            max_stretch = self.xrify_tuples(max_stretch)

        delta = (max_stretch - min_stretch)
        if isinstance(delta, xr.DataArray):
            # fillna if delta is NaN
            scale_factor = (1.0 / delta).fillna(0)
        else:
            scale_factor = 1.0 / delta
        attrs = self.data.attrs
        self.data -= min_stretch
        self.data *= scale_factor
        self.data.attrs = attrs

    def stretch_hist_equalize(self, approximate=False):
        """Stretch the current image's colors through histogram equalization.

        Args:
            approximate (bool): Use a faster less-accurate percentile
                                calculation. At the time of writing the dask
                                version of `percentile` is not as accurate as
                                the numpy version. This will likely change in
                                the future. Current dask version 0.17.

        """
        logger.info("Perform a histogram equalized contrast stretch.")

        nwidth = 2048.
        logger.debug("Make histogram bins having equal amount of data, " +
                     "using numpy percentile function:")

        def _band_hist(band_data):
            cdf = da.arange(0., 1., 1. / nwidth, chunks=nwidth)
            if approximate:
                # need a 1D array
                flat_data = band_data.ravel()
                # replace with nanpercentile in the future, if available
                # dask < 0.17 returns all NaNs for this
                bins = da.percentile(flat_data[da.notnull(flat_data)],
                                     cdf * 100.)
            else:
                bins = dask.delayed(np.nanpercentile)(band_data, cdf * 100.)
                bins = da.from_delayed(bins, shape=(nwidth,), dtype=cdf.dtype)
            res = dask.delayed(np.interp)(band_data, bins, cdf)
            res = da.from_delayed(res, shape=band_data.shape,
                                  dtype=band_data.dtype)
            return res

        band_results = []
        for band in self.data['bands'].values:
            if band == 'A':
                continue
            band_data = self.data.sel(bands=band)
            res = _band_hist(band_data.data)
            band_results.append(res)

        if 'A' in self.data.coords['bands'].values:
            band_results.append(self.data.sel(bands='A'))
        self.data.data = da.stack(band_results,
                                  axis=self.data.dims.index('bands'))

    def stretch_logarithmic(self, factor=100.):
        """Move data into range [1:factor] through normalized logarithm."""
        logger.debug("Perform a logarithmic contrast stretch.")
        crange = (0., 1.0)

        b__ = float(crange[1] - crange[0]) / np.log(factor)
        c__ = float(crange[0])

        def _band_log(arr):
            slope = (factor - 1.) / float(arr.max() - arr.min())
            arr = 1. + (arr - arr.min()) * slope
            arr = c__ + b__ * da.log(arr)
            return arr

        band_results = []
        for band in self.data['bands'].values:
            if band == 'A':
                continue
            band_data = self.data.sel(bands=band)
            res = _band_log(band_data.data)
            band_results.append(res)

        if 'A' in self.data.coords['bands'].values:
            band_results.append(self.data.sel(bands='A'))
        self.data.data = da.stack(band_results,
                                  axis=self.data.dims.index('bands'))

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

    def merge(self, img):
        """Use the provided image as background for the current *img* image,
        that is if the current image has missing data.
        """
        raise NotImplementedError("This method has not be implemented for "
                                  "xarray support.")
        if self.is_empty():
            raise ValueError("Cannot merge an empty image.")

        if self.mode != img.mode:
            raise ValueError("Cannot merge image of different modes.")

        selfmask = self.channels[0].mask
        for chn in self.channels[1:]:
            selfmask = np.ma.mask_or(selfmask, chn.mask)

        for i in range(len(self.channels)):
            self.channels[i] = np.ma.where(selfmask,
                                           img.channels[i],
                                           self.channels[i])
            self.channels[i].mask = np.logical_and(selfmask,
                                                   img.channels[i].mask)

    def colorize(self, colormap):
        """Colorize the current image using `colormap`.

        .. note::

            Works only on "L" or "LA" images.

        """

        if self.mode not in ("L", "LA"):
            raise ValueError("Image should be grayscale to colorize")

        if self.mode == "LA":
            alpha = self.data.sel(bands=['A'])
        else:
            alpha = None

        l_data = self.data.sel(bands=['L'])

        def _colorize(l_data, colormap):
            # 'l_data' is (1, rows, cols)
            # 'channels' will be a list of 3 (RGB) or 4 (RGBA) arrays
            channels = colormap.colorize(l_data)
            return np.concatenate(channels, axis=0)

        new_data = l_data.data.map_blocks(_colorize, colormap,
                                          chunks=(3,) + l_data.data.chunks[1:], dtype=np.float64)

        if alpha is not None:
            new_data = da.concatenate([new_data, alpha.data], axis=0)
            mode = "RGBA"
        else:
            mode = "RGB"

        # copy the coordinates so we don't affect the original
        coords = dict(self.data.coords)
        coords['bands'] = list(mode)
        attrs = self.data.attrs
        dims = self.data.dims
        self.data = xr.DataArray(new_data, coords=coords, attrs=attrs, dims=dims)

    def palettize(self, colormap):
        """Palettize the current image using `colormap`.

        .. note::

            Works only on "L" or "LA" images.

        """

        if self.mode not in ("L", "LA"):
            raise ValueError("Image should be grayscale to colorize")

        l_data = self.data.sel(bands=['L'])

        def _palettize(data):
            # returns data and palette, only need data
            return colormap.palettize(data)[0]

        new_data = l_data.data.map_blocks(_palettize, dtype=l_data.dtype)
        self.palette = tuple(colormap.colors)

        if self.mode == "L":
            mode = "P"
        else:
            mode = "PA"
            new_data = da.concatenate([new_data, self.data.sel(bands=['A'])], axis=0)

        self.data.data = new_data
        self.data.coords['bands'] = list(mode)

    def blend(self, other):
        """Alpha blend *other* on top of the current image."""
        raise NotImplementedError("This method has not be implemented for "
                                  "xarray support.")

        if self.mode != "RGBA" or other.mode != "RGBA":
            raise ValueError("Images must be in RGBA")
        src = other
        dst = self
        outa = src.channels[3] + dst.channels[3] * (1 - src.channels[3])
        for i in range(3):
            dst.channels[i] = (src.channels[i] * src.channels[3] +
                               dst.channels[i] * dst.channels[3] *
                               (1 - src.channels[3])) / outa
            dst.channels[i][outa == 0] = 0
        dst.channels[3] = outa

    def show(self):
        """Display the image on screen."""
        self.pil_image().show()

    def _repr_png_(self):
        import io
        b = io.BytesIO()
        self.pil_image().save(b, format='png')
        return b.getvalue()
