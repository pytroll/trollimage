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
"""This module defines the XRImage class.

It overlaps largely with the PIL library, but has the advantage of using
:class:`~xarray.DataArray` objects backed by :class:`dask arrays
<dask.array.Array>` as pixel arrays. This allows for invalid values to
be tracked, metadata to be assigned, and stretching to be lazy
evaluated. With the optional ``rasterio`` library installed dask array
chunks can be saved in parallel.

"""

import logging
import numbers
import os
import warnings

import dask
import dask.array as da
import numpy as np
import xarray as xr
from PIL import Image as PILImage
from dask.delayed import delayed
from trollimage.image import check_image_format

logger = logging.getLogger(__name__)


def combine_scales_offsets(*args):
    """Combine ``(scale, offset)`` tuples in one, considering they are applied from left to right.

    For example, if we have our base data called ```orig_data`` and apply to it
    ``(scale_1, offset_1)``, then ``(scale_2, offset_2)`` such that::

      data_1 = orig_data * scale_1 + offset_1
      data_2 = data_1 * scale_2 + offset_2

    this function will return the tuple ``(scale, offset)`` such that::

      data_2 = orig_data * scale + offset

    given the arguments ``(scale_1, offset_1), (scale_2, offset_2)``.

    """
    cscale = 1
    coffset = 0
    for scale, offset in args:
        cscale *= scale
        coffset = coffset * scale + offset
    return cscale, coffset


def invert_scale_offset(scale, offset):
    """Invert scale and offset to allow reverse transformation.

    Ie, it will return ``rscale, roffset`` such that::

      orig_data = rscale * data + roffset

    if::

      data = scale * orig_data + offset

    """
    return 1 / scale, -offset / scale


@delayed(nout=1, pure=True)
def delayed_pil_save(img, *args, **kwargs):
    """Dask delayed saving of PIL Image object.

    Special wrapper to handle `fill_value` try/except catch and provide a
    more useful error message.

    """
    try:
        img.save(*args, **kwargs)
    except OSError as e:
        # ex: cannot write mode LA as JPEG
        if "A as JPEG" in str(e):
            new_msg = ("Image mode not supported for this format. Specify "
                       "`fill_value=0` to set invalid values to black.")
            raise OSError(new_msg) from e
        raise


class XRImage:
    """Image class using an :class:`xarray.DataArray` as internal storage.

    It can be saved to a variety of image formats, but if Rasterio is
    installed, it can save to geotiff and jpeg2000 with geographical
    information.

    The enhancements functions are recording some parameters in the image's
    data attribute called `enhancement_history`.

    """

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

        # doesn't actually copy the data underneath
        # we don't want our operations to change the user's data
        # we do this last in case `expand_dims` made the data read only
        data = data.copy()

        return data

    @property
    def mode(self):
        """Mode of the image."""
        return ''.join(self.data['bands'].values)

    @staticmethod
    def _gtiff_to_cog_kwargs(format_kwargs):
        """Convert GDAL Geotiff driver options to COG driver options.

        The COG driver automatically sets some format options
        but zlevel is called level and blockxsize is called blocksize.
        Convert kwargs to save() from GTiff driver to COG driver.

        """
        format_kwargs.pop('photometric', None)
        if 'zlevel' in format_kwargs:
            format_kwargs['level'] = format_kwargs.pop('zlevel')
        if 'jpeg_quality' in format_kwargs:
            format_kwargs['quality'] = format_kwargs.pop('jpeg_quality')
        format_kwargs.pop('tiled', None)
        if 'blockxsize' in format_kwargs:
            format_kwargs['blocksize'] = format_kwargs.pop('blockxsize')
        format_kwargs.pop('blockysize', None)
        return format_kwargs

    def save(self, filename, fformat=None, fill_value=None, compute=True,
             keep_palette=False, cmap=None, driver=None, **format_kwargs):
        """Save the image to the given *filename*.

        Args:
            filename (str): Output filename
            fformat (str): File format of output file (optional). Can be
                           one of many image formats supported by the
                           `rasterio` or `PIL` libraries ('jpg', 'png',
                           'tif'). By default this is determined by the
                           extension of the provided filename.
                           If the format allows, geographical information will
                           be saved to the ouput file, in the form of grid
                           mapping or ground control points.
            driver (str): can override the choice of rasterio/gdal driver
                        which is normally selected from the filename or fformat.
                        This is an implementation detail normally avoided but
                        can be necessary if you wish to distinguish between
                        GeoTIFF drivers ("GTiff" is the default, but you can
                        specify "COG" to write a Cloud-Optimized GeoTIFF).
            fill_value (float): Replace invalid data values with this value
                                and do not produce an Alpha band. Default
                                behavior is to create an alpha band.
            compute (bool): If True (default) write the data to the file
                            immediately. If False the return value is either
                            a `dask.Delayed` object or a tuple of
                            ``(source, target)`` to be passed to
                            `dask.array.store`.
            keep_palette (bool): Saves the palettized version of the image if
                                 set to True. False by default.  Warning: this
                                 does not automatically write the colormap
                                 (palette) to the file.  To write the colormap
                                 to the file, one should additionally pass the
                                 colormap with the ``cmap`` keyword argument.
            cmap (Colormap or dict): Colormap to be applied to the image when
                                     saving with rasterio, used with
                                     keep_palette=True. Should be uint8.
            format_kwargs: Additional format options to pass to `rasterio`
                           or `PIL` saving methods. Any format argument passed
                           at this stage would be superseeded by `fformat`.

        Returns:
            Either `None` if `compute` is True or a `dask.Delayed` object or
            ``(source, target)`` pair to be passed to `dask.array.store`.
            If compute is False the return value depends on format and how
            the image backend is used. If ``(source, target)`` is provided
            then target is an open file-like object that must be closed by
            the caller.

        """
        kwformat = format_kwargs.pop('format', None)
        fformat = fformat or kwformat or os.path.splitext(filename)[1][1:]
        if fformat in ('tif', 'tiff', 'jp2'):
            try:
                return self.rio_save(filename, fformat=fformat, driver=driver,
                                     fill_value=fill_value, compute=compute,
                                     keep_palette=keep_palette, cmap=cmap,
                                     **format_kwargs)
            except ImportError:
                logger.warning("Missing 'rasterio' dependency to save GeoTIFF "
                               "image. Will try using PIL...")
        return self.pil_save(filename, fformat, fill_value,
                             compute=compute, **format_kwargs)

    def rio_save(self, filename, fformat=None, fill_value=None,
                 dtype=np.uint8, compute=True, tags=None,
                 keep_palette=False, cmap=None, overviews=None,
                 overviews_minsize=256, overviews_resampling=None,
                 include_scale_offset_tags=False,
                 scale_offset_tags=None,
                 colormap_tag=None,
                 driver=None,
                 **format_kwargs):
        """Save the image using rasterio.

        Args:
            filename (string): The filename to save to.
            fformat (string): The format to save to. If not specified (default),
                it will be infered from the file extension.
            driver (string): The gdal driver to use. If not specified (default),
                it will be inferred from the fformat, but you can override the
                default GeoTIFF driver ("GTiff") with "COG" if you want to create
                a Cloud_Optimized GeoTIFF (and set `tiled=True,overviews=[]`).
            fill_value (number): The value to fill the missing data with.
                Default is ``None``, translating to trying to keep the data
                transparent.
            dtype (np.dtype): The type to save the data to. Defaults to
                np.uint8.
            compute (bool): Whether (default) or not to compute the lazy data.
            tags (dict): Tags to include in the file.
            keep_palette (bool): Whether or not (default) to keep the image in
                P mode.
            cmap (colormap): The colormap to use for the data.
            overviews (list): The reduction factors of the overviews to include
                in the image, eg::

                    img.rio_save('myfile.tif', overviews=[2, 4, 8, 16])

                If provided as an empty list, then levels will be
                computed as powers of two until the last level has less
                pixels than `overviews_minsize`.  If driver='COG' then use
                `overviews=[]` to get a Cloud-Optimized GeoTIFF with a correct
                set of overviews created automatically.
                Default is to not add overviews.
            overviews_minsize (int): Minimum number of pixels for the smallest
                overview size generated when `overviews` is auto-generated.
                Defaults to 256.
            overviews_resampling (str): Resampling method
                to use when generating overviews. This must be the name of an
                enum value from :class:`rasterio.enums.Resampling` and
                only takes effect if the `overviews` keyword argument is
                provided. Common values include `nearest` (default),
                `bilinear`, `average`, and many others. See the rasterio
                documentation for more information.
            scale_offset_tags (Tuple[str, str] or None):
                If set to a ``(str, str)`` tuple, scale and offset will be
                stored in GDALMetaData tags.  Those can then be used to
                retrieve the original data values from pixel values.
                Scale and offset will be set to (NaN, NaN) for images that had
                non-linear enhancements applied (ex. gamma) as they can't be
                represented by a simple scale and offset. Scale and offset
                are also saved as (NaN, NaN) for multi-band images (ex. RGB)
                as storing multiple values in a single GDALMetaData tag is not
                currently supported.
            colormap_tag (str or None):
                If set and the image was colorized or palettized, a tag will
                be added with this name with the value of a comma-separated
                version of the Colormap that was used. See
                :meth:`trollimage.colormap.Colormap.to_csv` for more
                information.

        Returns:
            The delayed or computed result of the saving.

        """
        from ._xrimage_rasterio import RIOFile, RIODataset, split_regular_vs_lazy_tags
        fformat = fformat or os.path.splitext(filename)[1][1:]
        drivers = {'jpg': 'JPEG',
                   'png': 'PNG',
                   'tif': 'GTiff',
                   'tiff': 'GTiff',
                   'jp2': 'JP2OpenJPEG'}
        # If fformat is specified but not driver then convert it into a driver
        driver = driver or drivers.get(fformat, fformat)
        # The COG driver adds overviews so we don't need to create them ourself.
        # One thing we can't do is prevent any overviews, if we use None then
        # the COG driver will create automatically, we can't pass OVERVIEWS=NONE.
        if driver == 'COG' and overviews == []:
            overviews = None
        if include_scale_offset_tags:
            warnings.warn(
                "include_scale_offset_tags is deprecated, please use "
                "scale_offset_tags to indicate tag labels",
                DeprecationWarning, stacklevel=2)
            scale_offset_tags = scale_offset_tags or ("scale", "offset")

        if tags is None:
            tags = {}

        data, mode = self.finalize(fill_value, dtype=dtype,
                                   keep_palette=keep_palette)
        data = data.transpose('bands', 'y', 'x')

        crs = None
        gcps = None
        transform = None
        if driver in ['COG', 'GTiff', 'JP2OpenJPEG']:
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

            from ._xrimage_rasterio import get_data_arr_crs_transform_gcps
            crs, transform, gcps = get_data_arr_crs_transform_gcps(data)

            stime = data.attrs.get("start_time")
            if stime:
                stime_str = stime.strftime("%Y:%m:%d %H:%M:%S")
                tags.setdefault('TIFFTAG_DATETIME', stime_str)
        if driver == 'JPEG' and 'A' in mode:
            raise ValueError('JPEG does not support alpha')

        enhancement_colormap = self._get_colormap_from_enhancement_history(data)
        if colormap_tag and enhancement_colormap is not None:
            tags[colormap_tag] = enhancement_colormap.to_csv()
        if scale_offset_tags:
            self._add_scale_offset_to_tags(scale_offset_tags, data, tags)

        # If we are changing the driver then use appropriate kwargs
        if driver == 'COG':
            format_kwargs = self._gtiff_to_cog_kwargs(format_kwargs)

        # FIXME add metadata
        r_file = RIOFile(filename, 'w', driver=driver,
                         width=data.sizes['x'], height=data.sizes['y'],
                         count=data.sizes['bands'],
                         dtype=dtype,
                         nodata=fill_value,
                         crs=crs,
                         transform=transform,
                         gcps=gcps,
                         **format_kwargs)
        r_file.open()
        if not keep_palette:
            from ._xrimage_rasterio import color_interp
            r_file.colorinterp = color_interp(data)

        if keep_palette and cmap is not None:
            if data.dtype != 'uint8':
                raise ValueError('Rasterio only supports 8-bit colormaps')
            try:
                from trollimage.colormap import Colormap
                cmap = cmap.to_rio() if isinstance(cmap, Colormap) else cmap
                r_file.rfile.write_colormap(1, cmap)
            except AttributeError:
                raise ValueError("Colormap is not formatted correctly")

        tags, da_tags = split_regular_vs_lazy_tags(tags, r_file)
        r_file.rfile.update_tags(**tags)
        r_dataset = RIODataset(r_file, overviews,
                               overviews_resampling=overviews_resampling,
                               overviews_minsize=overviews_minsize)

        to_store = (data.data, r_dataset)
        if da_tags:
            to_store = list(zip(*([to_store] + da_tags)))

        if compute:
            # write data to the file now
            res = da.store(*to_store)
            to_close = to_store[1]
            if not isinstance(to_close, tuple):
                to_close = [to_close]
            for item in to_close:
                item.close()
            return res
        # provide the data object and the opened file so the caller can
        # store them when they would like. Caller is responsible for
        # closing the file
        return to_store

    @staticmethod
    def _get_colormap_from_enhancement_history(data_arr):
        for enhance_dict in reversed(data_arr.attrs.get('enhancement_history', [])):
            if "colormap" in enhance_dict:
                return enhance_dict["colormap"]
        return None

    def pil_save(self, filename, fformat=None, fill_value=None,
                 compute=True, **format_kwargs):
        """Save the image to the given *filename* using PIL.

        For now, the compression level [0-9] is ignored, due to PIL's
        lack of support. See also :meth:`save`.

        """
        fformat = fformat or os.path.splitext(filename)[1][1:]
        fformat = check_image_format(fformat)

        if fformat == 'png':
            # Take care of GeoImage.tags (if any).
            format_kwargs['pnginfo'] = self._pngmeta()

        img = self.pil_image(fill_value, compute=False)
        delay = delayed_pil_save(img, filename, fformat, **format_kwargs)
        if compute:
            return delay.compute()
        return delay

    def _add_scale_offset_to_tags(self, scale_offset_tags, data_arr, tags):
        scale_label, offset_label = scale_offset_tags
        scale, offset = self.get_scaling_from_history(data_arr.attrs.get('enhancement_history', []))
        tags[scale_label], tags[offset_label] = invert_scale_offset(scale, offset)

    def get_scaling_from_history(self, history=None):
        """Merge the scales and offsets from the history.

        If ``history`` isn't provided, the history of the current image will be
        used.
        """
        if history is None:
            history = self.data.attrs.get('enhancement_history', [])
        try:
            scaling = [(item['scale'], item['offset']) for item in history]
        except KeyError as err:
            logger.debug("Can only get combine scaling from a list of linear "
                         f"scaling operations: {err}. Setting scale and offset "
                         "to (NaN, NaN).")
            return np.nan, np.nan
        scale, offset = combine_scales_offsets(*scaling)
        scale_is_not_scalar = not isinstance(scale, numbers.Number) and len(scale) != 1
        offset_is_not_scalar = not isinstance(offset, numbers.Number) and len(offset) != 1
        if scale_is_not_scalar or offset_is_not_scalar:
            logger.debug("Multi-band scale/offset tags can't be saved to "
                         "geotiff. Setting scale and offset to (NaN, NaN).")
            return np.nan, np.nan
        return scale, offset

    @delayed(nout=1, pure=True)
    def _delayed_apply_pil(self, fun, pil_image, fun_args, fun_kwargs,
                           image_metadata=None, output_mode=None):
        if fun_args is None:
            fun_args = tuple()
        if fun_kwargs is None:
            fun_kwargs = dict()
        if image_metadata is None:
            image_metadata = dict()
        new_img = fun(pil_image, image_metadata, *fun_args, **fun_kwargs)
        if output_mode is not None:
            new_img = new_img.convert(output_mode)
        return np.array(new_img) / self.data.dtype.type(255.0)

    def apply_pil(self, fun, output_mode, pil_args=None, pil_kwargs=None, fun_args=None, fun_kwargs=None):
        """Apply a function `fun` on the pillow image corresponding to the instance of the XRImage.

        The function shall take a pil image as first argument, and is then passed fun_args and fun_kwargs.
        In addition, the current images's metadata is passed as a keyword argument called `image_mda`.
        It is expected to return the modified pil image.
        This function returns a new XRImage instance with the modified image data.

        The pil_args and pil_kwargs are passed to the `pil_image` method of the XRImage instance.

        """
        if pil_args is None:
            pil_args = tuple()
        if pil_kwargs is None:
            pil_kwargs = dict()
        pil_image = self.pil_image(*pil_args, compute=False, **pil_kwargs)

        # HACK: aggdraw.Font objects cause segmentation fault in dask tokenize
        # Remove this when aggdraw is either updated to allow type(font_obj)
        # or pycoast is updated to not accept Font objects
        # See https://github.com/pytroll/pycoast/issues/43
        # The last positional argument to the _burn_overlay function in Satpy
        # is the 'overlay' dict. This could include aggdraw.Font objects so we
        # completely remove it.
        delayed_kwargs = {}
        if fun.__name__ == "_burn_overlay":
            from dask.base import tokenize
            from dask.utils import funcname
            func = self._delayed_apply_pil
            if fun_args is None:
                fun_args = tuple()
            if fun_kwargs is None:
                fun_kwargs = dict()
            tokenize_args = (fun, pil_image, fun_args[:-1], fun_kwargs,
                             self.data.attrs, output_mode)
            dask_key_name = "%s-%s" % (
                funcname(func),
                tokenize(func.key, *tokenize_args, pure=True),
            )
            delayed_kwargs['dask_key_name'] = dask_key_name

        new_array = self._delayed_apply_pil(fun, pil_image, fun_args, fun_kwargs,
                                            self.data.attrs, output_mode,
                                            **delayed_kwargs)
        bands = len(output_mode)
        arr = da.from_delayed(new_array, dtype=self.data.dtype,
                              shape=(self.data.sizes['y'], self.data.sizes['x'], bands))

        new_data = xr.DataArray(arr, dims=['y', 'x', 'bands'],
                                coords={'y': self.data.coords['y'],
                                        'x': self.data.coords['x'],
                                        'bands': list(output_mode)},
                                attrs=self.data.attrs)
        return XRImage(new_data)

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

    def _create_alpha(self, data, fill_value=None):
        """Create an alpha band DataArray object.

        If `fill_value` is provided and input data is an integer type
        then it is used to determine invalid "null" pixels instead of
        xarray's `isnull` and `notnull` methods.

        The returned array is 1 where data is valid, 0 where invalid.

        """
        not_alpha = [b for b in data.coords['bands'].values if b != 'A']
        null_mask = data.sel(bands=not_alpha)
        if np.issubdtype(data.dtype, np.integer) and fill_value is not None:
            null_mask = null_mask != fill_value
        else:
            null_mask = null_mask.notnull()
        # if any of the bands are valid, we don't want transparency
        null_mask = null_mask.any(dim='bands')
        null_mask = null_mask.expand_dims('bands')
        null_mask['bands'] = ['A']
        # changes to null_mask attrs should not effect the original attrs
        # XRImage never uses them either
        null_mask.attrs = {}
        return null_mask

    def _add_alpha(self, data, alpha=None):
        """Create an alpha channel and concatenate it to the provided data.

        If ``data`` is an integer type then the alpha band will be scaled
        to use the smallest (min) value as fully transparent and the largest
        (max) value as fully opaque. If a `_FillValue` attribute is found for
        integer type data then it is used to identify null values in the data.
        Otherwise xarray's `isnull` is used.

        For float types the alpha band spans 0 to 1.

        """
        fill_value = data.attrs.get('_FillValue', None)  # integer fill value
        null_mask = alpha if alpha is not None else self._create_alpha(data, fill_value)
        # if we are using integer data, then alpha needs to be min-int to max-int
        # otherwise for floats we want 0 to 1
        if np.issubdtype(data.dtype, np.integer):
            # xarray sometimes upcasts this calculation, so cast again
            null_mask = self._scale_to_dtype(null_mask, data.dtype).astype(data.dtype)
        attrs = data.attrs.copy()
        data = xr.concat([data, null_mask], dim="bands")
        data.attrs = attrs
        return data

    def _get_dtype_scale_offset(self, dtype, fill_value):
        dinfo = np.iinfo(dtype)
        scale = dinfo.max - dinfo.min
        offset = dinfo.min
        if fill_value is not None:
            if fill_value == dinfo.min:
                # leave the lowest value for fill value only
                offset = offset + 1
                scale = scale - 1
            elif fill_value == dinfo.max:
                # leave the top value for fill value only
                scale = scale - 1
            else:
                warnings.warn(
                    "Specified fill value will overlap with valid "
                    "data. To avoid this warning specify a fill_value "
                    "that is the minimum or maximum for the data type "
                    "being saved to.", stacklevel=3)
        return scale, offset

    def _scale_to_dtype(self, data, dtype, fill_value=None):
        """Scale provided data to dtype range assuming a 0-1 range.

        Float input data is assumed to be normalized to a 0 to 1 range.
        Integer input data is not scaled, only clipped. A float output
        type is not scaled since both outputs and inputs are assumed to
        be in the 0-1 range already.

        """
        attrs = data.attrs.copy()
        if np.issubdtype(dtype, np.integer):
            if np.issubdtype(data, np.integer):
                # preserve integer data type
                data = data.clip(np.iinfo(dtype).min, np.iinfo(dtype).max)
            else:
                # scale float data (assumed to be 0 to 1) to full integer space
                # leave room for fill value if needed
                scale, offset = self._get_dtype_scale_offset(dtype, fill_value)
                data = data.clip(0, 1) * scale + offset
                attrs.setdefault('enhancement_history', list()).append({'scale': scale, 'offset': offset})
            data = data.round()
        data.attrs = attrs
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

        if not self.palette:
            raise RuntimeError("Can't convert palettized image, missing palette.")
        pal = np.array(self.palette)
        pal = da.from_array(pal, chunks=pal.shape)

        if pal.shape[1] == 4:
            # colormap's alpha overrides data alpha
            mode = "RGBA"
            alpha = None
        elif self.mode.endswith("A"):
            # add a new/fake 'bands' dimension to the end
            alpha = self.data.sel(bands="A").data[..., None]
            mode = mode + "A" if not mode.endswith("A") else mode
        else:
            alpha = None

        flat_indexes = self.data.sel(bands='P').data.ravel().astype('int64')
        dim_sizes = ((key, val) for key, val in self.data.sizes.items() if key != 'bands')
        dims, new_shape = zip(*dim_sizes)
        dims = dims + ('bands',)
        new_shape = new_shape + (pal.shape[1],)
        new_data = pal[flat_indexes].reshape(new_shape)
        coords = dict(self.data.coords)
        coords["bands"] = list(mode)

        if alpha is not None:
            new_arr = da.concatenate((new_data, alpha), axis=-1)
            data = xr.DataArray(new_arr, coords=coords, attrs=self.data.attrs, dims=dims)
        else:
            data = xr.DataArray(new_data, coords=coords, attrs=self.data.attrs, dims=dims)

        return data

    def _l2rgb(self, mode):
        """Convert from L (black and white) to RGB."""
        self._check_modes(("L", "LA"))

        bands = ["L"] * 3
        if mode[-1] == "A":
            bands.append("A")
        data = self.data.sel(bands=bands)
        data["bands"] = list(mode)
        return data

    def convert(self, mode):
        """Convert image to *mode*."""
        if mode == self.mode:
            return self.__class__(self.data)

        if mode not in ["P", "PA", "L", "LA", "RGB", "RGBA"]:
            raise ValueError("Mode %s not recognized." % (mode))

        if mode == self.mode + "A":
            data = self._add_alpha(self.data).data
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

    def final_mode(self, fill_value=None):
        """Get the mode of the finalized image when provided this fill_value."""
        if fill_value is None and not self.mode.endswith('A'):
            return self.mode + 'A'
        return self.mode

    def _add_alpha_and_scale(self, data, ifill, dtype):
        alpha = self._create_alpha(data, fill_value=ifill)
        data = self._scale_to_dtype(data, dtype)
        data = data.astype(dtype)
        data = self._add_alpha(data, alpha=alpha)
        return data

    def _replace_fill_value(self, data, ifill, fill_value, dtype):
        # Add fill_value after all other calculations have been done to
        # make sure it is not scaled for the data type
        if ifill is not None and fill_value is not None:
            # cast fill value to output type so we don't change data type
            fill_value = dtype(fill_value)
            # integer fields have special fill values
            data = data.where(data != ifill, dtype(fill_value))
        elif fill_value is not None:
            data = data.fillna(dtype(fill_value))

        return data

    def _get_input_fill_value(self, data):
        # if the data are integers then this fill value will be used to check for invalid values
        if np.issubdtype(data, np.integer):
            return data.attrs.get('_FillValue')
        return None

    def _scale_and_replace_fill_value(self, data, input_fill_value, fill_value, dtype):
        # scale float data to the proper dtype
        # this method doesn't cast yet so that we can keep track of NULL values
        data = self._scale_to_dtype(data, dtype, fill_value)
        data = self._replace_fill_value(data, input_fill_value, fill_value, dtype)
        return data

    def _scale_alpha_or_fill_data(self, data, fill_value, dtype):
        input_fill_value = self._get_input_fill_value(data)
        needs_alpha = fill_value is None and not self.mode.endswith('A')
        if needs_alpha:
            # We don't have a fill value or an alpha, let's add an alpha
            return self._add_alpha_and_scale(data, input_fill_value, dtype)
        return self._scale_and_replace_fill_value(data, input_fill_value, fill_value, dtype)

    def finalize(self, fill_value=None, dtype=np.uint8, keep_palette=False):
        """Finalize the image to be written to an output file.

        This adds an alpha band or fills data with a fill_value (if
        specified). It also scales float data to the output range of the
        data type (0-255 for uint8, default). For integer input data
        this method assumes the data is already scaled to the proper
        desired range. It will still fill in invalid values and add an
        alpha band if needed. Integer input data's fill value is
        determined by a special ``_FillValue`` attribute in the
        ``DataArray`` ``.attrs`` dictionary.

        Args:
            fill_value (int or float or None): Output value to use to
                represent invalid or missing pixels. By default this is
                `None` meaning an Alpha channel will be used to represent
                the invalid values; transparent for invalid, opaque
                otherwise. Some output formats do not support alpha channels
                so a ``fill_value`` must be provided. This is determined by
                the underlying library doing the writing (pillow or rasterio).
                If specified, it should be the minimum or maximum of the
                ``dtype`` (ex. 0 or 255 for uint8). Floating point image data
                is then scaled to fit the remainder of the data type space.
                Integer image data will **not** be scaled. For example, a
                ``dtype`` of ``numpy.uint8`` and a ``fill_value`` of 0 will
                result in floating-point data being scaled linearly from 1 to 255.
            dtype (numpy.dtype): Output data type to convert the current image
                data to. Default is unsigned 8-bit integer
                (:class:`numpy.uint8`).
            keep_palette (bool): Whether to convert a paletted image to RGB/A
                or not. If ``False`` (default) then ``P`` mode images will be
                converted to ``RGB`` and ``PA`` will be converted to ``RGBA``.
                If ``True``, images with mode ``P`` or ``PA`` are kept as is
                and will not be scaled in order for their index values into a
                palette to be maintained. This flag should always be ``False``
                for non-paletted images.

        """
        if keep_palette and not self.mode.startswith('P'):
            keep_palette = False

        if not keep_palette:
            finalize_kwargs = dict(
                fill_value=fill_value, dtype=dtype,
                keep_palette=keep_palette,
            )
            if self.mode == "P":
                return self.convert("RGB").finalize(**finalize_kwargs)
            if self.mode == "PA":
                return self.convert("RGBA").finalize(**finalize_kwargs)

        if np.issubdtype(dtype, np.floating) and fill_value is None:
            logger.warning("Image with floats cannot be transparent, so "
                           "setting fill_value to 0")
            fill_value = 0

        final_data = self.data.copy()
        try:
            final_data.attrs['enhancement_history'] = list(self.data.attrs['enhancement_history'])
        except KeyError:
            pass
        with xr.set_options(keep_attrs=True):
            attrs = final_data.attrs
            if not keep_palette:
                final_data = self._scale_alpha_or_fill_data(final_data, fill_value, dtype)
            final_data = final_data.astype(dtype)
            final_data.attrs = attrs

        return final_data, ''.join(final_data['bands'].values)

    def pil_image(self, fill_value=None, compute=True):
        """Return a PIL image from the current image.

        Args:
            fill_value (int or float): Value to use for NaN null values.
                See :meth:`~trollimage.xrimage.XRImage.finalize` for more
                info.
            compute (bool): Whether to return a fully computed PIL.Image
                object (True) or return a dask Delayed object representing
                the Image (False). This is True by default.

        """
        channels, mode = self.finalize(fill_value)
        res = channels.transpose('y', 'x', 'bands')
        img = dask.delayed(PILImage.fromarray)(np.squeeze(res.data), mode)
        if compute:
            img = img.compute()
        return img

    def xrify_tuples(self, tup):
        """Make xarray.DataArray from tuple."""
        return xr.DataArray(tup,
                            dims=['bands'],
                            coords={'bands': self.data['bands']})

    def gamma(self, gamma=None):
        """Apply gamma correction to the channels of the image.

        If *gamma* is a tuple, then it should have as many elements as
        the channels of the image, and the gamma correction is applied
        elementwise. If *gamma* is a number, the same gamma correction
        is applied on every channel, if there are several channels in
        the image. The behaviour of :func:`gamma` is undefined outside
        the normal [0,1] range of the channels.

        """
        if _is_unity_or_none(gamma):
            return

        inverse_gamma = self._get_inverse_gamma(gamma)
        logger.debug("Applying gamma %s", str(gamma))
        attrs = self.data.attrs
        self.data = self.data.clip(min=0)
        self.data **= inverse_gamma
        self.data.attrs = attrs
        self.data.attrs.setdefault('enhancement_history', []).append({'gamma': gamma})

    def _get_inverse_gamma(self, gamma):
        if np.issubdtype(self.data.dtype, np.floating):
            dtype = self.data.dtype
        else:
            dtype = np.float32
        if isinstance(gamma, (list, tuple)):
            gamma = self.xrify_tuples(gamma).astype(dtype)
        else:
            gamma = np.array(gamma, dtype=dtype)
        return 1.0 / gamma

    def stretch(self, stretch="crude", **kwargs):
        """Apply stretching to the current image.

        The value of *stretch* sets the type of stretching applied. The
        values "histogram", "linear", "crude" (or "crude-stretch")
        perform respectively histogram equalization, contrast stretching
        (with 5% cutoff on both sides), and contrast stretching without
        cutoff. The value "logarithmic" or "log" will do a logarithmic
        enhancement towards white. If a tuple or a list of two values is
        given as input, then a contrast stretching is performed with the
        values as cutoff. These values should be normalized in the range
        [0.0,1.0].

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

    @staticmethod
    def _compute_quantile(data, dims, cutoffs):
        """Compute quantile for stretch_linear.

        Dask delayed functions need to be non-internal functions (created
        inside a function) to be serializable on a multi-process scheduler.

        Quantile requires the data to be loaded since it not supported on
        dask arrays yet.

        """
        # numpy doesn't get a 'quantile' function until 1.15
        # for better backwards compatibility we use xarray's version
        data_arr = xr.DataArray(data, dims=dims)
        # delayed will provide us the fully computed xarray with ndarray
        left, right = data_arr.quantile([cutoffs[0], 1. - cutoffs[1]], dim=['x', 'y'])
        logger.debug("Interval: left=%s, right=%s", str(left), str(right))
        return left.data, right.data

    def stretch_linear(self, cutoffs=(0.005, 0.005)):
        """Stretch linearly the contrast of the current image.

        Use *cutoffs* for left and right trimming.

        """
        logger.debug("Perform a linear contrast stretch.")

        left, right = self._get_left_and_right_quantiles_for_linear_stretch(cutoffs)

        self.crude_stretch(left, right)

    def _get_left_and_right_quantiles_for_linear_stretch(self, cutoffs):
        logger.debug("Calculate the histogram quantiles: ")
        logger.debug("Left and right quantiles: " +
                     str(cutoffs[0]) + " " + str(cutoffs[1]))
        cutoff_type = np.float64
        # numpy percentile (which quantile calls) returns 64-bit floats
        # unless the value is a higher order float
        if np.issubdtype(self.data.dtype, np.floating) and \
                np.dtype(self.data.dtype).itemsize > 8:
            cutoff_type = self.data.dtype

        data = self.data
        if 'A' in self.data.coords['bands'].values:
            data = self.data.sel(bands=self.data.coords['bands'].values[:-1])

        left_data, right_data = self._get_left_and_right_quantiles_without_alpha(data, cutoffs, cutoff_type)

        if 'A' in self.data.coords['bands'].values:
            left_data = np.hstack([left_data, np.array([0])])
            right_data = np.hstack([right_data, np.array([1])])
        left = xr.DataArray(left_data, dims=('bands',),
                            coords={'bands': self.data['bands']})
        right = xr.DataArray(right_data, dims=('bands',),
                             coords={'bands': self.data['bands']})
        return left, right

    def _get_left_and_right_quantiles_without_alpha(self, data, cutoffs, cutoff_type):
        left, right = dask.delayed(self._compute_quantile, nout=2)(data.data, data.dims, cutoffs)
        left_data = da.from_delayed(left,
                                    shape=(data.sizes['bands'],),
                                    dtype=cutoff_type)
        right_data = da.from_delayed(right,
                                     shape=(data.sizes['bands'],),
                                     dtype=cutoff_type)
        return left_data, right_data

    def crude_stretch(self, min_stretch=None, max_stretch=None):
        """Perform simple linear stretching.

        This is done without any cutoff on the current image and
        normalizes to the [0,1] range.

        """
        min_stretch = self._check_stretch_value(min_stretch, kind='min')
        max_stretch = self._check_stretch_value(max_stretch, kind='max')
        scale_factor = self._get_scale_factor(min_stretch, max_stretch)

        attrs = self.data.attrs
        offset = -min_stretch * scale_factor
        self.data = np.multiply(self.data, scale_factor, dtype=scale_factor.dtype) + offset
        self.data.attrs = attrs
        self.data.attrs.setdefault('enhancement_history', []).append({'scale': scale_factor,
                                                                      'offset': offset})

    def _check_stretch_value(self, val, kind='min'):
        if val is None:
            non_band_dims = tuple(x for x in self.data.dims if x != 'bands')
            val = getattr(self.data, kind)(dim=non_band_dims)

        if isinstance(val, (list, tuple)):
            val = self.xrify_tuples(val)

        try:
            val = val.astype(self.data.dtype)
        except AttributeError:
            val = self.data.dtype.type(val)

        return val

    def _get_scale_factor(self, min_stretch, max_stretch):
        delta = (max_stretch - min_stretch)
        dtype = self._infer_scale_factor_dtype()
        if isinstance(delta, xr.DataArray):
            # fillna if delta is NaN
            scale_factor = (1.0 / delta).fillna(0).astype(dtype)
        else:
            scale_factor = np.array(1.0 / delta, dtype=dtype)

        return scale_factor

    def _infer_scale_factor_dtype(self):
        if np.issubdtype(self.data.dtype, np.integer):
            return np.float32
        return self.data.dtype

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
        self.data.attrs.setdefault('enhancement_history', []).append({'hist_equalize': True})

    def stretch_logarithmic(self, factor=100., base="e", min_stretch=None, max_stretch=None):
        """Move data into range [1:factor] through normalized logarithm.

        Args:
            factor (float): Maximum of the range data will be scaled to
                before applying the log function. Image data will be scaled
                to a 1 to ``factor`` range.
            base (str): Type of log to use. Defaults to natural log ("e"),
                but can also be "10" for base 10 log or "2" for base 2 log.
            min_stretch (float or list): Minimum input value to scale from.
                Data will be clipped to this value before being scaled to
                the 1:factor range. By default (None), the limits are computed
                on the fly but with a performance penalty. May also be a list
                for multi-band images.
            max_stretch (float or list): Maximum input value to scale from.
                Data will be clipped to this value before being scaled to
                the 1:factor range. By default (None), the limits are computed
                on the fly but with a performance penalty. May also be a list
                for multi-band images.

        """
        logger.debug("Perform a logarithmic contrast stretch.")
        crange = (0., 1.0)
        log_func = np.log if base == "e" else getattr(np, "log" + base)
        min_stretch, max_stretch = self._convert_log_minmax_stretch(min_stretch, max_stretch)

        b__ = float(crange[1] - crange[0]) / log_func(factor)
        c__ = float(crange[0])

        def _band_log(arr, min_input, max_input):
            slope = (factor - 1.) / (max_input - min_input)
            arr = np.clip(arr, min_input, max_input)
            arr = 1. + (arr - min_input) * slope
            arr = c__ + b__ * log_func(arr)
            return arr

        band_results = []
        for band_idx, band in enumerate(self.data['bands'].values):
            if band == 'A':
                continue
            band_data = self.data.sel(bands=band)
            res = _band_log(band_data.data,
                            min_stretch[band_idx],
                            max_stretch[band_idx])
            band_results.append(res)

        if 'A' in self.data.coords['bands'].values:
            band_results.append(self.data.sel(bands='A'))
        self.data.data = da.stack(band_results, axis=self.data.dims.index('bands'))
        self.data.attrs.setdefault('enhancement_history', []).append({'log_factor': factor})

    def _convert_log_minmax_stretch(self, min_stretch, max_stretch):
        non_band_dims = tuple(x for x in self.data.dims if x != 'bands')
        if min_stretch is None:
            min_stretch = [m.data for m in self.data.min(dim=non_band_dims)]
        if max_stretch is None:
            max_stretch = [m.data for m in self.data.max(dim=non_band_dims)]
        if not isinstance(min_stretch, (list, tuple)):
            min_stretch = [min_stretch] * self.data.sizes.get("bands", 1)
        if not isinstance(max_stretch, (list, tuple)):
            max_stretch = [max_stretch] * self.data.sizes.get("bands", 1)
        return min_stretch, max_stretch

    def stretch_weber_fechner(self, k, s0):
        """Stretch according to the Weber-Fechner law.

        p = k.ln(S/S0)
        p is perception, S is the stimulus, S0 is the stimulus threshold (the
        highest unpercieved stimulus), and k is the factor.

        """
        attrs = self.data.attrs
        self.data = k * np.log(self.data / s0)
        self.data.attrs = attrs
        self.data.attrs.setdefault('enhancement_history', []).append({'weber_fechner': (k, s0)})

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
        self.data.attrs.setdefault('enhancement_history', []).append({'scale': scale,
                                                                      'offset': offset})

    def stack(self, img):
        """Stack the provided image on top of the current image."""
        # TODO: Conversions between different modes with notification
        # to the user, i.e. proper logging
        if self.mode != img.mode:
            raise NotImplementedError("Cannot stack images of different modes.")

        self.data = self.data.where(img.data.isnull(), img.data)

    def merge(self, img):
        """Use the provided image as background for the current *img* image.

        That is if the current image has missing data.

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
        """Colorize the current image using ``colormap``.

        Convert a greyscale image (mode "L" or "LA") to a color image (mode
        "RGB" or "RGBA") by applying a colormap. If floating point data being
        colorized contains NaNs then the result will also contain NaNs instead
        of a color from the colormap. Integer data that includes
        a ``.attrs['_FillValue']`` will be converted to a floating point array
        and values equal to ``_FillValue`` replaced with NaN before being
        colorized.

        To create a color image in mode "P" or "PA", use
        :meth:`~XRImage.palettize`.

        Args:
            colormap (:class:`~trollimage.colormap.Colormap`):
                Colormap to be applied to the image.

        .. note::

            Works only on "L" or "LA" images.

        """
        if self.mode not in ("L", "LA"):
            raise ValueError("Image should be grayscale to colorize")

        colormap = self._adjust_colormap_dtype(colormap)
        l_data = self._get_masked_floating_luminance_data()
        alpha = self.data.sel(bands=['A']) if self.mode == "LA" else None
        new_data = colormap.colorize(l_data.data)

        if colormap.colors.shape[1] == 4:
            mode = "RGBA"
        elif alpha is not None:
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
        scale_factor, offset = self._get_colormap_scale_offset(colormap)
        self.data.attrs.setdefault('enhancement_history', []).append({
            'scale': scale_factor,
            'offset': offset,
            'colormap': colormap,
        })

    def _adjust_colormap_dtype(self, colormap):
        if np.issubdtype(self.data.dtype, np.floating) and colormap.colors.dtype != self.data.dtype:
            colormap.colors = colormap.colors.astype(self.data.dtype)
            colormap.values = colormap.values.astype(self.data.dtype)
        return colormap

    def _get_masked_floating_luminance_data(self):
        l_data = self.data.sel(bands='L')
        # mask any integer fields with _FillValue
        # assume NaN is used otherwise
        if self.mode == "L" and np.issubdtype(self.data.dtype, np.integer):
            fill_value = self._get_input_fill_value(self.data)
            if fill_value is not None:
                l_data = l_data.where(l_data != fill_value)
        return l_data

    def palettize(self, colormap):
        """Palettize the current image using ``colormap``.

        Convert a mode "L" (or "LA") grayscale image to a mode "P" (or "PA")
        palette image and store the palette in the ``palette`` attribute.
        To store this image in mode "P", call :meth:`~XRImage.save` with
        ``keep_palette=True``.  To include color information in the output
        format (if supported), call :meth:`~XRImage.save` with
        ``keep_palette=True`` *and* ``cmap=colormap``.

        To (directly) get an image in mode "RGB" or "RGBA", use
        :meth:`~XRImage.colorize`.

        Args:
            colormap (:class:`~trollimage.colormap.Colormap`):
                Colormap to be applied to the image.

        Notes:
            Works only on "L" or "LA" images.

            Similar to other enhancement methods (colorize, stretch, etc)
            this method adds an ``enhancement_history`` list to the metadata
            stored in the image ``DataArray``'s metadata (``.attrs``).
            In other methods, however, the metadata directly translates to
            the linear operations performed in that enhancement. The palettize
            operation converts data values to indices into a colormap.
            This result is based on the range of values defined in the Colormap
            (``cmap.values``). To be most useful, the enhancement history scale
            and offset values represent the range of the colormap as if scaling
            the data to a 0-1 range. This means that once the data is saved to
            a format as an RGB (the palette colors are applied) the scale and
            offset can be used to determine the original range of the data
            based on the min/max of the data type of the format (ex. uint8).
            For example:

            .. code-block:: python

                dtype_min = 0
                dtype_max = 255
                scale = ...  # scale from geotiff
                offset = ...  # offset from geotiff
                data_min = offset
                data_max = (dtype_max - dtype_min) * scale + offset

            If a geotiff is saved with ``keep_palette=True`` then the data
            saved to the geotiff are the palette indices and will not be
            scaled to the data type of the format. There will also be a
            standard geotiff color table in the geotiff to identify that
            these are indices rather than some other type of image data. This
            means in this case the scale and offset can be used to determine
            the original range of the data starting from a 0-1 range
            (``dtype_min`` is 0 and ``dtype_max`` is 1 in the code above).

        """
        if self.mode not in ("L", "LA"):
            raise ValueError("Image should be grayscale to colorize")

        l_data = self.data.sel(bands=['L'])
        new_data, self.palette = colormap.palettize(l_data.data)

        if self.mode == "L":
            mode = "P"
        else:
            mode = "PA"
            new_data = da.concatenate([new_data, self.data.sel(bands=['A'])], axis=0)

        self.data.data = new_data
        self.data.coords['bands'] = list(mode)
        # See docstring notes above for how scale/offset should be used
        scale_factor, offset = self._get_colormap_scale_offset(colormap)
        self.data.attrs.setdefault('enhancement_history', []).append({
            'scale': scale_factor,
            'offset': offset,
            'colormap': colormap,
        })

    @staticmethod
    def _get_colormap_scale_offset(colormap):
        cmap_min = colormap.values[0]
        cmap_max = colormap.values[-1]
        scale_factor = 1.0 / (cmap_max - cmap_min)
        offset = -cmap_min * scale_factor
        return scale_factor, offset

    def blend(self, src):
        r"""Alpha blend *src* on top of the current image.

        Perform `alpha blending`_ of *src* on top of the current image.
        Alpha blending is defined as:

        .. math::

           \begin{cases}
            \mathrm{out}_A =
             \mathrm{src}_A + \mathrm{dst}_A (1 - \mathrm{src}_A) \\
            \mathrm{out}_{RGB} =
             \bigl(\mathrm{src}_{RGB}\mathrm{src}_A
                 + \mathrm{dst}_{RGB} \mathrm{dst}_A
                   \left(1 - \mathrm{src}_A \right) \bigr)
             \div \mathrm{out}_A \\
            \mathrm{out}_A = 0 \Rightarrow \mathrm{out}_{RGB} = 0
           \end{cases}

        Both images must have mode ``"RGBA"``.

        Args:
            src (:class:`XRImage` with mode ``"RGBA"``)
                Image to be blended on top of current image.

        .. _alpha blending: https://en.wikipedia.org/w/index.php?title=Alpha_compositing&oldid=891033105#Alpha_blending

        Returns
            XRImage with mode "RGBA", blended as described above

        """
        # NB: docstring maths copy-pasta from enwiki

        if self.mode != "RGBA":
            raise ValueError(
                    "Expected self.mode='RGBA', got {md!s}".format(
                        md=self.mode))
        if not isinstance(src, XRImage):
            raise TypeError("Expected XRImage, got {tp!s}".format(
                tp=type(src)))
        if src.mode != "RGBA":
            raise ValueError("Expected src.mode='RGBA', got {sm!s}".format(
                sm=src.mode))

        srca = src.data.sel(bands="A")
        dsta = self.data.sel(bands="A")
        outa = srca + dsta * (1-srca)
        bi = {"bands": ["R", "G", "B"]}
        rgb = ((src.data.loc[bi] * srca
               + self.data.loc[bi] * dsta * (1-srca))
               / outa).where(outa != 0, 0)
        return self.__class__(
                xr.concat(
                    [rgb, outa.expand_dims("bands")],
                    dim="bands"))

    def show(self):
        """Display the image on screen."""
        self.pil_image().show()

    def _repr_png_(self):
        import io
        b = io.BytesIO()
        self.pil_image().save(b, format='png')
        return b.getvalue()


def _is_unity_or_none(gamma):
    if gamma is None or gamma == 1.0:
        return True
    if not hasattr(gamma, "__iter__"):
        return False
    return all(g == 1.0 for g in gamma) or all(g is None for g in gamma)
