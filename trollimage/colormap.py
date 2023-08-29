#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012, 2013, 2014 Martin Raspaud

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
"""A simple colormap module."""

import contextlib
import copy
import os
from io import StringIO
from typing import Optional
import warnings
from numbers import Number
import pathlib
import sys

import numpy as np
from trollimage.colorspaces import rgb2lch, lch2rgb


@contextlib.contextmanager
def _file_or_stringio(filename_or_none):
    if filename_or_none is None:
        yield StringIO()
    else:
        with open(filename_or_none, "w") as file_obj:
            yield file_obj


def colorize(arr, colors, values):
    """Colorize a monochromatic array *arr*, based *colors* given for *values*.

    Args:
        arr (numpy array, numpy masked array, dask array):
            Data to be mapped to the colors in the colors array using values
            as control points. Data can be any shape, but must represent a
            single (luminance) band of data (not RGB or any other colorspace).
        colors (numpy array):
            Colors to map the data to. Colors can be RGB or RGBA in the
            0 to 1 range. The array should be in the shape (N, 3 or 4) where
            N is the size of the colormap (the number of colors) and the last
            dimension is the band dimension where each element represents
            Red (R), Green (G), Blue (B), and optionally Alpha (A).
        values (numpy array):
            Control points mapping input data values to the colors to be
            mapped to. Should be one dimension and the same number of elements
            as the ``colors`` array has colors (N).

    Returns: Resulting RGB/A array with the shape (3 or 4, ...) where the
        first dimension is 3 if the colors array was RGB and 4 if the colors
        array was RGBA. The remaining shape of the result matches the provided
        ``data`` input shape. Like ``colors`` the color values will be between
        0 and 1.

    """
    if can_be_block_mapped(arr):
        return _colorize_dask(arr, colors, values)
    else:
        return _colorize(arr, colors, values)


def can_be_block_mapped(data):
    """Check if the array can be processed in chunks."""
    return hasattr(data, 'map_blocks')


def _colorize_dask(dask_array, colors, values):
    """Colorize a dask array.

    The channels are stacked on the first dimension.
    """
    return dask_array.map_blocks(_colorize, colors, values, dtype=colors.dtype, new_axis=0,
                                 chunks=[colors.shape[1]] + list(dask_array.chunks))


def _colorize(arr, colors, values):
    """Colorize the array."""
    channels = _interpolate_rgb_colors(arr, colors, values)
    alpha = _interpolate_alpha(arr, colors, values)
    channels.extend(alpha)
    channels = _mask_channels(channels, arr)
    return np.stack(channels, axis=0)


def _interpolate_rgb_colors(arr, colors, values):
    interp_xp_coords = np.array(values)
    interp_y_coords = rgb2lch(colors)
    if values[0] > values[-1]:
        # monotonically decreasing
        interp_xp_coords = interp_xp_coords[::-1]
        interp_y_coords = interp_y_coords[::-1]
    # Make sure hue (radians) are consistently increasing or decreasing
    interp_lch = np.zeros(arr.shape + (3,), dtype=interp_y_coords.dtype)
    interp_lch[..., 0] = np.interp(arr, interp_xp_coords, interp_y_coords[..., 0])
    interp_lch[..., 1] = np.interp(arr, interp_xp_coords, interp_y_coords[..., 1])
    interp_y_coords[..., 2] = np.unwrap(interp_y_coords[..., 2])
    interp_lch[..., 2] = np.interp(arr, interp_xp_coords, interp_y_coords[..., 2])
    interp_lch[..., 2] = _ununwrap(interp_lch[..., 2])
    new_rgb = lch2rgb(interp_lch)
    return [new_rgb[..., 0], new_rgb[..., 1], new_rgb[..., 2]]


def _ununwrap(input_radians):
    """Undo the operations performed by numpy unwrap.

    Taken from https://stackoverflow.com/a/15927914/433202

    """
    return (input_radians + np.pi) % (2 * np.pi) - np.pi


def _interpolate_alpha(arr, colors, values):
    alpha = [np.interp(arr,
                       np.array(values),
                       np.array(colors)[:, i + 3])
             for i in range(np.array(colors).shape[1] - 3)]
    return alpha


def _mask_channels(channels, arr):
    """Mask the channels if arr is a masked array."""
    return [_mask_array(channel, arr) for channel in channels]


def _mask_array(new_array, arr):
    """Mask new_array with the mask from array."""
    try:
        return np.ma.array(new_array, mask=arr.mask)
    except AttributeError:
        return new_array


def palettize(arr, colors, values):
    """Apply *colors* to *data* from start *values*.

    Args:
        arr (numpy array, numpy masked array, dask array):
            data to be palettized.
        colors (numpy array):
            the colors to use (R, G, B)
        values (numpy array):
            the values corresponding to the colors in the array
    """
    if can_be_block_mapped(arr):
        return _palettize_dask(arr, colors, values), tuple(colors)
    else:
        return _palettize(arr, values), tuple(colors)


def _palettize_dask(darr, colors, values):
    """Apply a palette to a dask array."""
    return darr.map_blocks(_palettize, values, dtype=int)


def _palettize(arr, values):
    """Apply palette to array."""
    new_arr = _digitize_array(arr, values)
    reshaped_array = new_arr.reshape(arr.shape)
    return _mask_array(reshaped_array, arr)


def _digitize_array(arr, values):
    if values[0] <= values[-1]:
        # monotonic increasing values
        outside_range_bin = max(np.nanmax(arr), values.max()) + 1
        right = False
    else:
        # monotonic decreasing values
        outside_range_bin = min(np.nanmin(arr), values.min()) - 1
        right = True
    bins = np.concatenate((values, [outside_range_bin]))

    new_arr = np.digitize(arr.ravel(), bins, right=right)
    new_arr -= 1
    new_arr = new_arr.clip(min=0, max=len(values) - 1)
    return new_arr


class Colormap(object):
    """The colormap object.

    Args:
        *args: Series of (value, color) tuples. These positional arguments
            are only used if the ``values`` and ``colors`` keyword arguments
            aren't provided.
        values: One dimensional array-like of control points where
            each corresponding color is applied. Must be the same number of
            elements as colors and must be monotonic.
        colors: Two dimensional array-like of RGB or RGBA colors where each
            color is applied to a specific control point. Must be the same
            number of colors as control points (values). Colors should be
            floating point numbers between 0 and 1.

    Initialize with tuples of (value, (colors)), like this::

      Colormap((-75.0, (1.0, 1.0, 0.0)),
               (-40.0001, (0.0, 1.0, 1.0)),
               (-40.0, (1, 1, 1)),
               (30.0, (0, 0, 0)))

    You can also concatenate colormaps together, try::

      cm = cm1 + cm2

    """

    def __init__(self, *tuples, **kwargs):
        """Set up the instance."""
        if 'colors' in kwargs and 'values' in kwargs:
            values = kwargs['values']
            colors = kwargs['colors']
        elif 'colors' in kwargs or 'values' in kwargs:
            raise ValueError("Both 'colors' and 'values' must be provided.")
        else:
            values = [a for (a, b) in tuples]
            colors = [b for (a, b) in tuples]
        self.values = np.array(values)
        self.colors = self._validate_colors(colors)
        if self.values.shape[0] != self.colors.shape[0]:
            raise ValueError("'values' and 'colors' should have the same "
                             "number of elements. Got "
                             f"{self.values.shape[0]} and {self.colors.shape[0]}.")

    def _validate_colors(self, colors):
        colors = np.array(colors)
        if colors.ndim != 2 or colors.shape[-1] not in (3, 4):
            raise ValueError("Colormap 'colors' must be RGB or RGBA. Got unexpected shape: {}".format(colors.shape))
        if not np.issubdtype(colors.dtype, np.floating):
            warnings.warn("Colormap 'colors' should be flotaing point numbers between 0 and 1.", stacklevel=3)
            colors = colors.astype(np.float64)
        return colors

    def colorize(self, data):
        """Colorize a monochromatic array *data*, based on the current colormap."""
        return colorize(data, self.colors, self.values)

    def palettize(self, data):
        """Palettize a monochromatic array *data* based on the current colormap."""
        return palettize(data, self.colors, self.values)

    def to_rgb(self):
        """Return colormap with RGB colors.

        If already RGB then the same instance is returned.
        If an Alpha channel exists in the colormap, it is dropped.

        """
        if self.colors.shape[-1] == 3:
            return self

        values = self.values.copy()
        colors = self.colors.copy()
        return Colormap(
            values=values,
            colors=colors[:, :3]
        )

    def to_rgba(self):
        """Return colormap with RGBA colors.

        If already RGBA then the same instance is returned.
        If not already RGBA, a completely opaque (1.0) color

        """
        if self.colors.shape[-1] == 4:
            return self

        values = self.values.copy()
        colors = np.empty((self.colors.shape[0], 4), dtype=self.colors.dtype)
        colors[:, :3] = self.colors
        colors[:, 3] = 1.0
        return Colormap(
            values=values,
            colors=colors
        )

    def __add__(self, other):
        """Append colormap together."""
        old, other = self._normalize_color_arrays(self, other)
        values = np.concatenate((old.values, other.values))
        if not self._monotonic_one_direction(values):
            raise ValueError("Merged colormap 'values' are not monotonically "
                             "increasing, monotonically decreasing, or equal.")
        colors = np.concatenate((old.colors, other.colors))
        return Colormap(
            values=values,
            colors=colors,
        )

    @staticmethod
    def _monotonic_one_direction(values):
        delta = np.diff(values)
        all_increasing = (delta >= 0).all()
        all_decreasing = (delta <= 0).all()
        return all_increasing or all_decreasing

    @staticmethod
    def _normalize_color_arrays(cmap1, cmap2):
        num_bands1 = cmap1.colors.shape[-1]
        num_bands2 = cmap2.colors.shape[-1]
        if num_bands1 == num_bands2:
            return cmap1, cmap2
        return cmap1.to_rgba(), cmap2.to_rgba()

    def reverse(self, inplace=True):
        """Reverse the current colormap in place.

        Args:
            inplace (bool): If True (default), modify the colors of this
                Colormap inplace. If False, return a new instance.

        """
        colors = np.flipud(self.colors)
        if not inplace:
            return Colormap(
                values=self.values.copy(),
                colors=colors
            )
        self.colors = colors
        return self

    def set_range(self, min_val, max_val, inplace=True):
        """Set the range of the colormap to [*min_val*, *max_val*].

        The Colormap's values will match the range specified even if "min_val"
        is greater than "max_val". To flip the order of the colors, use
        :meth:`reversed`.

        Args:
            min_val (float): New minimum value for the control points in
                this colormap.
            max_val (float): New maximum value for the control points in
                this colormap.
            inplace (bool): If True (default), modify the values inplace.
                If False, return a new Colormap instance.

        """
        cmap = self
        values = (((cmap.values * 1.0 - cmap.values[0]) /
                   (cmap.values[-1] - cmap.values[0]))
                  * (max_val - min_val) + min_val)
        if not inplace:
            return Colormap(
                values=values,
                colors=cmap.colors.copy()
            )

        cmap.values = values
        return cmap

    def to_rio(self):
        """Convert the colormap to a rasterio colormap.

        Note that rasterio requires color tables to have round integer value
        control points. This method assumes that the range of this Colormap
        is already in the desired output range and to avoid issues with
        rasterio will round the values and convert them to unsigned integers.
        """
        colors = (((self.colors * 1.0 - self.colors.min()) /
                   (self.colors.max() - self.colors.min())) * 255)
        # rasterio doesn't allow non-integer colormap values
        values = np.round(self.values).astype(np.uint)
        return dict(zip(values, tuple(map(tuple, colors))))

    def to_csv(
            self,
            filename: Optional[str] = None,
            color_scale: Number = 255,
    ) -> Optional[str]:
        """Save Colormap to a comma-separated text file or string.

        The CSV data will have 4 to 5 columns for each row where each
        each row will contain the value (V), red (R), green (B), blue (B),
        and if configured alpha (A).

        The values will remain in whatever range is currently set on the
        colormap. The colors of the colormap (assumed to be between 0 and 1)
        will be multiplied by 255 to produce a traditional unsigned 8-bit
        integer value.

        Args:
            filename: The filename of the CSV file to save to.
                If not provided or None a string is returned with the contents.
            color_scale: Scale colors by this factor before converting to a
                CSV. Colors are stored in the Colormap in a 0 to 1 range.
                Defaults to 255. If not equal to 1 values are converted to
                integers too.

        """
        with _file_or_stringio(filename) as csv_file:
            for value, color in zip(self.values, self.colors):
                scaled_color = [x * color_scale for x in color]
                if color_scale != 1.0:
                    scaled_color = [int(x) for x in scaled_color]
                csv_file.write(",".join(["{:0.6f}".format(value)] + [str(x) for x in scaled_color]) + "\n")
        if isinstance(csv_file, StringIO):
            return csv_file.getvalue()

    @classmethod
    def from_file(
            cls,
            filename: str,
            colormap_mode: Optional[str] = None,
            color_scale: Number = 255,
    ):
        """Create Colormap from a comma-separated or binary file of colormap data.

        Args:
            filename: Filename of a binary or CSV file
            colormap_mode: Force the scheme of the colormap data (ex. RGBA).
                See information below on other possible values and how they
                are interpreted. By default this is determined based on the
                number of columns in the data.
            color_scale: The maximum possible color value in the colormap data
                provided. For example, if the colors in the provided data were
                8-bit unsigned integers this should be 255 (the default). This
                value will be used to normalize the colors from 0 to 1.

        Colormaps can be loaded from ``.npy``, ``.npz``, or comma-separated text
        files. Numpy (npy/npz) files should be 2D arrays with rows for each color.
        Comma-separated files should have a row for each color with each column
        representing a single value/channel. A filename
        ending with ``.npy`` or ``.npz`` is read as a numpy file with
        :func:`numpy.load`. All other extensions are
        read as a comma-separated file. For ``.npz`` files the data must be stored
        as a positional list where the first element represents the colormap to
        use. See :func:`numpy.savez` for more information.

        The colormap is interpreted as 1 of 4 different "colormap modes":
        ``RGB``, ``RGBA``, ``VRGB``, or ``VRGBA``. The
        colormap mode can be forced with the ``colormap_mode`` keyword
        argument. If it is not provided then a default will be chosen
        based on the number of columns in the array (3: RGB, 4: VRGB, 5: VRGBA).

        The "V" in the possible colormap modes represents the control value of
        where that color should be applied. If "V" is not provided in the colormap
        data it defaults to the row index in the colormap array (0, 1, 2, ...)
        divided by the total number of colors to produce a number between 0 and 1.
        See the "Set Range" section below for more information.
        The remaining elements in the colormap array represent the Red (R),
        Green (G), and Blue (B) color to be mapped to.

        See the "Color Scale" section below for more information on the value
        range of provided numbers.

        To read from a string containing CSV, use :meth:`~Colormap.from_csv`.

        To get a named colormap, use :meth:`~Colormap.from_name` or load the
        colormap directly as a module attribute.

        To get a colormap from an ndarray, use :meth:`~Colormap.from_ndarray`.

        **Color Scale**

        By default colors are expected to be in a 0-255 range. This
        can be overridden by specifying ``color_scale`` keyword argument.
        A common alternative to 255 is ``1`` to specify floating
        point numbers between 0 and 1. The resulting Colormap uses the normalized
        color values (0-1).
        """
        if _is_actually_a_csv_string(filename):
            warnings.warn(
                "Passing a data string to Colormap.from_file is deprecated. "
                "Please use Colormap.from_string.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return cls.from_string(filename, colormap_mode, color_scale)
        values, colors = _get_values_colors_from_file(filename, colormap_mode, color_scale)
        return cls(values=values, colors=colors)

    @classmethod
    def from_string(cls, string, *args, **kwargs):
        """Create colormap from string.

        Create a colormap from a string that contains comma seperated values
        (CSV).

        To read from an external file that contains CSV, use :meth:`from_csv`.

        Args:
            string (str): String containing CSV.  Must have no less than three
                and no more than five columns and describe entirely numeric
                data.
            colormap_mode (str or None): Optional. Can be None, "RGB", "RGBA", "VRGB", or
                "VRGBA".  If None (default), this is inferred from the dimensions of
                the data contained in the CSV.  Modes starting with V have in
                the first column the values to which the color relates.
            color_scale (number): The value that represents white in the
                numbers describing the colors. Defaults to 255, could also be 1
                or something else.
        """
        openfile = StringIO(string)
        cmap_data = _read_colormap_data_from_file(openfile)
        return cls.from_ndarray(cmap_data, *args, **kwargs)

    @classmethod
    def from_np(cls, path, *args, **kwargs):
        """Create Colormap from a numpy-file.

        Create a colormap from a numpy data file ``.npy`` or ``.npz``.

        The data should contain at least three and at most five columns.

        Args:
            path (str or Pathlib.Path): Path to file containing numpy data.
            colormap_mode (str or None): Optional. Can be None, "RGB", "RGBA", "VRGB", or
                "VRGBA".  If None (default), this is inferred from the dimensions of
                the data contained in the CSV.  Modes starting with V have in
                the first column the values to which the color relates.
            color_scale (number): The value that represents white in the
                numbers describing the colors. Defaults to 255, could also be 1
                or something else.
        """
        cmap_data = _read_colormap_data_from_np(path)
        return cls.from_ndarray(cmap_data, *args, **kwargs)

    @classmethod
    def from_csv(cls, path, colormap_mode=None, color_scale=255):
        """Create Colormap from CSV file.

        Create a Colormap from a file that contains comma seperated values
        (CSV).

        To read from a string that contains CSV, use :meth:`from_string`.

        Args:
            string (str or pathlib.Path): Path to file containing CSV.
                The CSV must have at least three and at most five columns and
                describe entirely numeric data.
            colormap_mode (str or None): Optional. Can be None, "RGB", "RGBA", "VRGB", or
                "VRGBA".  If None (default), this is inferred from the dimensions of
                the data contained in the CSV.  Modes starting with V have in
                the first column the values to which the color relates.
            color_scale (number): The value that represents white in the
                numbers describing the colors. Defaults to 255, could also be 1
                or something else.
        """
        cmap_data = np.loadtxt(path, delimiter=",")
        return cls.from_ndarray(cmap_data, colormap_mode, color_scale)

    @classmethod
    def from_ndarray(cls, cmap_data, colormap_mode=None, color_scale=255):
        """Create Colormap from ndarray.

        Create a colormap from a numpy data array.

        The data should contain at least three and at most five columns.

        For historical reasons, this method exists alongside
        :meth:`from_sequence_of_colors` and :meth:`from_array_with_metadata` despite similar
        functionality.

        Args:
            cmap_data (ndarray): Array describing the colours.
                Must have at least three and at most five columns and
                have a numeric dtype.
            colormap_mode (str or None): Optional. Can be None, "RGB", "RGBA", "VRGB", or
                "VRGBA".  If None (default), this is inferred from the dimensions of
                the data contained in the CSV.  Modes starting with V have in
                the first column the values to which the color relates.
            color_scale (number): The value that represents white in the
                numbers describing the colors. Defaults to 255, could also be 1
                or something else.
        """
        values, colors = _get_values_colors_from_ndarray(cmap_data, colormap_mode, color_scale)
        return cls(values=values, colors=colors)

    @classmethod
    def from_name(cls, name):
        """Return named colormap.

        Return a colormap by name.  Supported colormaps are the ones defined in
        the module namespace.

        Args:
            name (str): Name of colormap.
        """
        cmap = getattr(sys.modules[__name__], name)
        return copy.copy(cmap)

    @classmethod
    def from_sequence_of_colors(cls, colors, values=None, color_scale=255):
        """Create Colormap from sequence of colors.

        Create a colormap from a sequence of colors, such as a list of colors.
        If values is not given, assume values between 0 and 1, linearly spaced
        according to the total number of colors.

        For historical reasons, this method exists alongside
        :meth:`from_ndarray` and :meth:`from_array_with_metadata` despite similar
        functionality.

        Args:
            colors (Sequence): List of colors, where each element must itself
                be a sequence of 3 or 4 numbers (RGB or RGBA).
            values (array, optional): Values associated with the colors.  If
                not given, assume linear between 0 and 1.
            color_scale (number): The value that represents white in the
                numbers describing the colors. Defaults to 255, could also be 1
                or something else.
        """
        # this method was moved from satpy. where it was in
        # satpy.enhancements.create_colormap
        # then it was refactored/rewritten
        color_array = np.array(colors)
        if values is None:
            values = np.linspace(0, 1, len(colors))
        else:
            values = np.asarray(values)
        color_array = np.concatenate((values[:, np.newaxis], color_array), axis=1)
        return cls.from_ndarray(
            color_array,
            "VRGB" if color_array.shape[1] == 4 else "VRGBA",
            color_scale=color_scale)

    @classmethod
    def from_array_with_metadata(
            cls, palette, dtype, color_scale=255,
            valid_range=None, scale_factor=1, add_offset=0,
            remove_last=True):
        """Create Colormap from an array with metadata.

        Create a colormap from an array with associated metadata, either in
        attributes or passed on to the function.

        For historical reasons, this method exists alongside
        :meth:`from_ndarray` and :meth:`from_sequence_of_colors` despite similar
        functionality.

        If ``palette`` is an xarray dataarray with the attribute
        ``palette_meanings``, those meanings are interpreted as values
        associated with the colormap.

        If no values can be interpreted from the metadata, values will be
        linearly interpolated between 0 and 255 (if ``dtype`` is ``np.uint8``)
        or according to ``valid_range``.

        Args:
            palette (ndarray or xarray.DataArray)
                Array describing colors, possibly with metadata.  If it has a
                ``palette_meanings`` attribute, this will be used for color
                interpretation.
            dtype
                dtype for the colormap
            color_scale (number): The value that represents white in the
                numbers describing the colors. Defaults to 255, could also be 1
                or something else.
            valid_range
                valid range for colors, if colormap is not of dtype uint8
            scale_factor
                scale factor to apply to the colormap
            add_offset
                add offset to apply to the colormap
            remove_last
                Remove the last value if the array has no metadata associated.
                Defaults to true for historical reasons.

        """
        # this method was moved from satpy, where it was in
        # satpy.composites.ColormapCompositor.build_colormap
        #
        # then it was refactored/rewritten for trollimage
        squeezed_palette = np.asanyarray(palette).squeeze()
        set_range = True
        if hasattr(palette, 'attrs') and 'palette_meanings' in palette.attrs:
            set_range = False
            values = np.asarray(palette.attrs['palette_meanings'])
        else:
            # remove the last value because monkeys don't like water sprays
            # on a more serious note, I don't know why we are removing the last
            # value here, but this behaviour was copied from ancient satpy code
            values = np.arange(squeezed_palette.shape[0]-remove_last)
            if remove_last:
                squeezed_palette = squeezed_palette[:-remove_last, :]

        color_array = np.concatenate((values[:, np.newaxis], squeezed_palette), axis=1)
        colormap = cls.from_ndarray(
            color_array,
            "VRGB" if color_array.shape[1] == 4 else "VRGBA",
            color_scale=color_scale)
        if dtype == np.dtype("uint8"):
            return colormap

        if valid_range is not None:
            if set_range:
                colormap.set_range(
                    *(np.array(valid_range) * scale_factor
                      + add_offset))
            return colormap

        raise AttributeError("Data need to have either a valid_range or be of type uint8" +
                             " in order to be displayable with an attached color-palette!")


def _is_actually_a_csv_string(string):
    """Try to guess whether this string contains CSV."""
    return string.count("\n") > 0 and string.count(",") > 0


def _get_values_colors_from_file(filename, colormap_mode, color_scale):
    data = _read_colormap_data_from_file(filename)
    return _get_values_colors_from_ndarray(data, colormap_mode, color_scale)


def _get_values_colors_from_ndarray(data, colormap_mode, color_scale):
    cols = data.shape[1]
    default_modes = {
        3: 'RGB',
        4: 'VRGB',
        5: 'VRGBA'
    }
    default_mode = default_modes.get(cols)
    if colormap_mode is None:
        colormap_mode = default_mode
    if colormap_mode is None or len(colormap_mode) != cols:
        raise ValueError(
            "Unexpected colormap shape for mode '{}'".format(colormap_mode))
    rows = data.shape[0]
    if colormap_mode[0] == 'V':
        colors = data[:, 1:]
        if color_scale != 1:
            colors = data[:, 1:] / float(color_scale)
        values = data[:, 0]
    else:
        colors = data
        if color_scale != 1:
            colors = colors / float(color_scale)
        values = np.arange(rows) / float(rows - 1)
    return values, colors


def _read_colormap_data_from_file(filename_or_file_obj):
    if isinstance(filename_or_file_obj, str):
        ext = os.path.splitext(filename_or_file_obj)[1]
        if ext in (".npy", ".npz"):
            return _read_colormap_data_from_np(filename_or_file_obj)
    # CSV file or file-like object of CSV string data
    return np.loadtxt(filename_or_file_obj, delimiter=",")


def _read_colormap_data_from_np(path):
    path = pathlib.Path(path)
    file_content = np.load(path)
    if path.suffix == ".npz":
        # .npz is a collection
        # assume position list-like and get the first element
        file_content = file_content["arr_0"]
    return file_content


# matlab jet "#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow",
# "#FF7F00", "red", "#7F0000"

rainbow = Colormap((0.000, (0.0, 0.0, 0.5)),
                   (0.125, (0.0, 0.0, 1.0)),
                   (0.250, (0.0, 0.5, 1.0)),
                   (0.375, (0.0, 1.0, 1.0)),
                   (0.500, (0.5, 1.0, 0.5)),
                   (0.625, (1.0, 1.0, 0.0)),
                   (0.750, (1.0, 0.5, 0.0)),
                   (0.875, (1.0, 0.0, 0.0)),
                   (1.000, (0.5, 0.0, 0.0)))

# * Colors from www.ColorBrewer.org by Cynthia A. Brewer, Geography,
# * Pennsylvania State University.

# * Single hue *

blues = Colormap((0.000, (247 / 255.0, 251 / 255.0, 1.0)),
                 (1.000, (8 / 255.0, 48 / 255.0, 107 / 255.0)))

greens = Colormap((0.000, (247 / 255.0, 252 / 255.0, 245 / 255.0)),
                  (1.000, (0.0, 68 / 255.0, 27 / 255.0)))

greys = Colormap((0.0, (1.0, 1.0, 1.0)),
                 (1.0, (0.0, 0.0, 0.0)))

oranges = Colormap((0.0, (1.0, 245 / 255.0, 235 / 255.0)),
                   (1.0, (127 / 255.0, 39 / 255.0, 4 / 255.0)))

purples = Colormap((0.0, (252 / 255.0, 251 / 255.0, 253 / 255.0)),
                   (1.0, (63 / 255.0, 0.0, 125 / 255.0)))

reds = Colormap((0.0, (1.0, 245 / 255.0, 240 / 255.0)),
                (1.0, (103 / 255.0, 0.0, 13 / 255.0)))

# * Multihue *

# BuGn

bugn = Colormap((0.000, (247 / 255.0, 252 / 255.0, 253 / 255.0)),
                (1.000, (0.0, 68 / 255.0, 27 / 255.0)))

# BuPu

bupu = Colormap((0.000, (247 / 255.0, 252 / 255.0, 253 / 255.0)),
                (1.000, (77 / 255.0, 0.0, 75 / 255.0)))

# GnBu

gnbu = Colormap((0.000, (247 / 255.0, 252 / 255.0, 240 / 255.0)),
                (1.000, (8 / 255.0, 64 / 255.0, 129 / 255.0)))

# OrRd

orrd = Colormap((0.000, (255 / 255.0, 247 / 255.0, 236 / 255.0)),
                (1.000, (127 / 255.0, 0.0, 0.0)))

# PuBu

pubu = Colormap((0.000, (1.0, 247 / 255.0, 251 / 255.0)),
                (0.500, (116 / 255.0, 169 / 255.0, 207 / 255.0)),
                (1.000, (2 / 255.0, 56 / 255.0, 88 / 255.0)))

# PuBuGn

pubugn = Colormap((0.000, (1.0, 247 / 255.0, 251 / 255.0)),
                  (0.500, (103 / 255.0, 169 / 255.0, 207 / 255.0)),
                  (1.000, (1 / 255.0, 70 / 255.0, 54 / 255.0)))

# PuRd

purd = Colormap((0.000, (247 / 255.0, 244 / 255.0, 249 / 255.0)),
                (1.000, (103 / 255.0, 0.0, 31 / 255.0)))

# RdPu

rdpu = Colormap((0.000, (1.0, 247 / 255.0, 243 / 255.0)),
                (1.000, (73 / 255.0, 0.0, 106 / 255.0)))

# YlGn

ylgn = Colormap((0.000, (1.0, 1.0, 229 / 255.0)),
                (1.000, (0.0, 69 / 255.0, 41 / 255.0)))

# YlGnBu

ylgnbu = Colormap((0.000, (1.0, 1.0, 217 / 255.0)),
                  (0.500, (65 / 255.0, 182 / 255.0, 196 / 255.0)),
                  (1.000, (8 / 255.0, 29 / 255.0, 88 / 255.0)))

# YlOrBr

ylorbr = Colormap((0.000, (1.0, 1.0, 229 / 255.0)),
                  (0.500, (254 / 255.0, 153 / 255.0, 41 / 255.0)),
                  (1.000, (102 / 255.0, 37 / 255.0, 6 / 255.0)))

# YlOrRd

ylorrd = Colormap((0.000, (1.0, 1.0, 204 / 255.0)),
                  (0.500, (254 / 255.0, 141 / 255.0, 60 / 255.0)),
                  (1.000, (128 / 255.0, 0.0, 38 / 255.0)))

sequential_colormaps = [blues, greens, greys, oranges, purples, reds,
                        bugn, bupu, gnbu, orrd, pubu, pubugn, purd, rdpu,
                        ylgn, ylgnbu, ylorbr, ylorrd]

# * Diverging *

brbg = Colormap((0.0, (84 / 255.0, 48 / 255.0, 5 / 255.0)),
                (0.1, (140 / 255.0, 81 / 255.0, 10 / 255.0)),
                (0.2, (191 / 255.0, 129 / 255.0, 45 / 255.0)),
                (0.3, (223 / 255.0, 194 / 255.0, 125 / 255.0)),
                (0.4, (246 / 255.0, 232 / 255.0, 195 / 255.0)),
                (0.5, (245 / 255.0, 245 / 255.0, 245 / 255.0)),
                (0.6, (199 / 255.0, 234 / 255.0, 229 / 255.0)),
                (0.7, (128 / 255.0, 205 / 255.0, 193 / 255.0)),
                (0.8, (53 / 255.0, 151 / 255.0, 143 / 255.0)),
                (0.9, (1 / 255.0, 102 / 255.0, 94 / 255.0)),
                (1.0, (0 / 255.0, 60 / 255.0, 48 / 255.0)))

piyg = Colormap((0.0, (142 / 255.0, 1 / 255.0, 82 / 255.0)),
                (0.1, (197 / 255.0, 27 / 255.0, 125 / 255.0)),
                (0.2, (222 / 255.0, 119 / 255.0, 174 / 255.0)),
                (0.3, (241 / 255.0, 182 / 255.0, 218 / 255.0)),
                (0.4, (253 / 255.0, 224 / 255.0, 239 / 255.0)),
                (0.5, (247 / 255.0, 247 / 255.0, 247 / 255.0)),
                (0.6, (230 / 255.0, 245 / 255.0, 208 / 255.0)),
                (0.7, (184 / 255.0, 225 / 255.0, 134 / 255.0)),
                (0.8, (127 / 255.0, 188 / 255.0, 65 / 255.0)),
                (0.9, (77 / 255.0, 146 / 255.0, 33 / 255.0)),
                (1.0, (39 / 255.0, 100 / 255.0, 25 / 255.0)))

prgn = Colormap((0.0, (64 / 255.0, 0 / 255.0, 75 / 255.0)),
                (0.1, (118 / 255.0, 42 / 255.0, 131 / 255.0)),
                (0.2, (153 / 255.0, 112 / 255.0, 171 / 255.0)),
                (0.3, (194 / 255.0, 165 / 255.0, 207 / 255.0)),
                (0.4, (231 / 255.0, 212 / 255.0, 232 / 255.0)),
                (0.5, (247 / 255.0, 247 / 255.0, 247 / 255.0)),
                (0.6, (217 / 255.0, 240 / 255.0, 211 / 255.0)),
                (0.7, (166 / 255.0, 219 / 255.0, 160 / 255.0)),
                (0.8, (90 / 255.0, 174 / 255.0, 97 / 255.0)),
                (0.9, (27 / 255.0, 120 / 255.0, 55 / 255.0)),
                (1.0, (0 / 255.0, 68 / 255.0, 27 / 255.0)))

puor = Colormap((0.0, (127 / 255.0, 59 / 255.0, 8 / 255.0)),
                (0.1, (179 / 255.0, 88 / 255.0, 6 / 255.0)),
                (0.2, (224 / 255.0, 130 / 255.0, 20 / 255.0)),
                (0.3, (253 / 255.0, 184 / 255.0, 99 / 255.0)),
                (0.4, (254 / 255.0, 224 / 255.0, 182 / 255.0)),
                (0.5, (247 / 255.0, 247 / 255.0, 247 / 255.0)),
                (0.6, (216 / 255.0, 218 / 255.0, 235 / 255.0)),
                (0.7, (178 / 255.0, 171 / 255.0, 210 / 255.0)),
                (0.8, (128 / 255.0, 115 / 255.0, 172 / 255.0)),
                (0.9, (84 / 255.0, 39 / 255.0, 136 / 255.0)),
                (1.0, (45 / 255.0, 0 / 255.0, 75 / 255.0)))

rdbu = Colormap((0.0, (103 / 255.0, 0 / 255.0, 31 / 255.0)),
                (0.1, (178 / 255.0, 24 / 255.0, 43 / 255.0)),
                (0.2, (214 / 255.0, 96 / 255.0, 77 / 255.0)),
                (0.3, (244 / 255.0, 165 / 255.0, 130 / 255.0)),
                (0.4, (253 / 255.0, 219 / 255.0, 199 / 255.0)),
                (0.5, (247 / 255.0, 247 / 255.0, 247 / 255.0)),
                (0.6, (209 / 255.0, 229 / 255.0, 240 / 255.0)),
                (0.7, (146 / 255.0, 197 / 255.0, 222 / 255.0)),
                (0.8, (67 / 255.0, 147 / 255.0, 195 / 255.0)),
                (0.9, (33 / 255.0, 102 / 255.0, 172 / 255.0)),
                (1.0, (5 / 255.0, 48 / 255.0, 97 / 255.0)))

rdgy = Colormap((0.0, (103 / 255.0, 0 / 255.0, 31 / 255.0)),
                (0.1, (178 / 255.0, 24 / 255.0, 43 / 255.0)),
                (0.2, (214 / 255.0, 96 / 255.0, 77 / 255.0)),
                (0.3, (244 / 255.0, 165 / 255.0, 130 / 255.0)),
                (0.4, (253 / 255.0, 219 / 255.0, 199 / 255.0)),
                (0.5, (255 / 255.0, 255 / 255.0, 255 / 255.0)),
                (0.6, (224 / 255.0, 224 / 255.0, 224 / 255.0)),
                (0.7, (186 / 255.0, 186 / 255.0, 186 / 255.0)),
                (0.8, (135 / 255.0, 135 / 255.0, 135 / 255.0)),
                (0.9, (77 / 255.0, 77 / 255.0, 77 / 255.0)),
                (1.0, (26 / 255.0, 26 / 255.0, 26 / 255.0)))

rdylbu = Colormap((0.0, (165 / 255.0, 0 / 255.0, 38 / 255.0)),
                  (0.1, (215 / 255.0, 48 / 255.0, 39 / 255.0)),
                  (0.2, (244 / 255.0, 109 / 255.0, 67 / 255.0)),
                  (0.3, (253 / 255.0, 174 / 255.0, 97 / 255.0)),
                  (0.4, (254 / 255.0, 224 / 255.0, 144 / 255.0)),
                  (0.5, (255 / 255.0, 255 / 255.0, 191 / 255.0)),
                  (0.6, (224 / 255.0, 243 / 255.0, 248 / 255.0)),
                  (0.7, (171 / 255.0, 217 / 255.0, 233 / 255.0)),
                  (0.8, (116 / 255.0, 173 / 255.0, 209 / 255.0)),
                  (0.9, (69 / 255.0, 117 / 255.0, 180 / 255.0)),
                  (1.0, (49 / 255.0, 54 / 255.0, 149 / 255.0)))

rdylgn = Colormap((0.0, (165 / 255.0, 0 / 255.0, 38 / 255.0)),
                  (0.1, (215 / 255.0, 48 / 255.0, 39 / 255.0)),
                  (0.2, (244 / 255.0, 109 / 255.0, 67 / 255.0)),
                  (0.3, (253 / 255.0, 174 / 255.0, 97 / 255.0)),
                  (0.4, (254 / 255.0, 224 / 255.0, 139 / 255.0)),
                  (0.5, (255 / 255.0, 255 / 255.0, 191 / 255.0)),
                  (0.6, (217 / 255.0, 239 / 255.0, 139 / 255.0)),
                  (0.7, (166 / 255.0, 217 / 255.0, 106 / 255.0)),
                  (0.8, (102 / 255.0, 189 / 255.0, 99 / 255.0)),
                  (0.9, (26 / 255.0, 152 / 255.0, 80 / 255.0)),
                  (1.0, (0 / 255.0, 104 / 255.0, 55 / 255.0)))

spectral = Colormap((0.0, (158 / 255.0, 1 / 255.0, 66 / 255.0)),
                    (0.1, (213 / 255.0, 62 / 255.0, 79 / 255.0)),
                    (0.2, (244 / 255.0, 109 / 255.0, 67 / 255.0)),
                    (0.3, (253 / 255.0, 174 / 255.0, 97 / 255.0)),
                    (0.4, (254 / 255.0, 224 / 255.0, 139 / 255.0)),
                    (0.5, (255 / 255.0, 255 / 255.0, 191 / 255.0)),
                    (0.6, (230 / 255.0, 245 / 255.0, 152 / 255.0)),
                    (0.7, (171 / 255.0, 221 / 255.0, 164 / 255.0)),
                    (0.8, (102 / 255.0, 194 / 255.0, 165 / 255.0)),
                    (0.9, (50 / 255.0, 136 / 255.0, 189 / 255.0)),
                    (1.0, (94 / 255.0, 79 / 255.0, 162 / 255.0)))

diverging_colormaps = [brbg, piyg, prgn, puor, rdbu, rdgy, rdylbu, rdylgn,
                       spectral]

# * qualitative colormaps *

set1 = Colormap((0, (228 / 255.0, 26 / 255.0, 28 / 255.0)),
                (1, (55 / 255.0, 126 / 255.0, 184 / 255.0)),
                (2, (77 / 255.0, 175 / 255.0, 74 / 255.0)),
                (3, (152 / 255.0, 78 / 255.0, 163 / 255.0)),
                (4, (255 / 255.0, 127 / 255.0, 0 / 255.0)),
                (5, (255 / 255.0, 255 / 255.0, 51 / 255.0)),
                (6, (166 / 255.0, 86 / 255.0, 40 / 255.0)),
                (7, (247 / 255.0, 129 / 255.0, 191 / 255.0)),
                (8, (153 / 255.0, 153 / 255.0, 153 / 255.0)))

set2 = Colormap((0, (102 / 255.0, 194 / 255.0, 165 / 255.0)),
                (1, (252 / 255.0, 141 / 255.0, 98 / 255.0)),
                (2, (141 / 255.0, 160 / 255.0, 203 / 255.0)),
                (3, (231 / 255.0, 138 / 255.0, 195 / 255.0)),
                (4, (166 / 255.0, 216 / 255.0, 84 / 255.0)),
                (5, (255 / 255.0, 217 / 255.0, 47 / 255.0)),
                (6, (229 / 255.0, 196 / 255.0, 148 / 255.0)),
                (7, (179 / 255.0, 179 / 255.0, 179 / 255.0)))

set3 = Colormap((0, (141 / 255.0, 211 / 255.0, 199 / 255.0)),
                (1, (255 / 255.0, 255 / 255.0, 179 / 255.0)),
                (2, (190 / 255.0, 186 / 255.0, 218 / 255.0)),
                (3, (251 / 255.0, 128 / 255.0, 114 / 255.0)),
                (4, (128 / 255.0, 177 / 255.0, 211 / 255.0)),
                (5, (253 / 255.0, 180 / 255.0, 98 / 255.0)),
                (6, (179 / 255.0, 222 / 255.0, 105 / 255.0)),
                (7, (252 / 255.0, 205 / 255.0, 229 / 255.0)),
                (8, (217 / 255.0, 217 / 255.0, 217 / 255.0)),
                (9, (188 / 255.0, 128 / 255.0, 189 / 255.0)),
                (10, (204 / 255.0, 235 / 255.0, 197 / 255.0)),
                (11, (255 / 255.0, 237 / 255.0, 111 / 255.0)))

paired = Colormap((0, (166 / 255.0, 206 / 255.0, 227 / 255.0)),
                  (1, (31 / 255.0, 120 / 255.0, 180 / 255.0)),
                  (2, (178 / 255.0, 223 / 255.0, 138 / 255.0)),
                  (3, (51 / 255.0, 160 / 255.0, 44 / 255.0)),
                  (4, (251 / 255.0, 154 / 255.0, 153 / 255.0)),
                  (5, (227 / 255.0, 26 / 255.0, 28 / 255.0)),
                  (6, (253 / 255.0, 191 / 255.0, 111 / 255.0)),
                  (7, (255 / 255.0, 127 / 255.0, 0 / 255.0)),
                  (8, (202 / 255.0, 178 / 255.0, 214 / 255.0)),
                  (9, (106 / 255.0, 61 / 255.0, 154 / 255.0)),
                  (10, (255 / 255.0, 255 / 255.0, 153 / 255.0)),
                  (11, (177 / 255.0, 89 / 255.0, 40 / 255.0)))

accent = Colormap((0, (127 / 255.0, 201 / 255.0, 127 / 255.0)),
                  (1, (190 / 255.0, 174 / 255.0, 212 / 255.0)),
                  (2, (253 / 255.0, 192 / 255.0, 134 / 255.0)),
                  (3, (255 / 255.0, 255 / 255.0, 153 / 255.0)),
                  (4, (56 / 255.0, 108 / 255.0, 176 / 255.0)),
                  (5, (240 / 255.0, 2 / 255.0, 127 / 255.0)),
                  (6, (191 / 255.0, 91 / 255.0, 23 / 255.0)),
                  (7, (102 / 255.0, 102 / 255.0, 102 / 255.0)))

dark2 = Colormap((0, (27 / 255.0, 158 / 255.0, 119 / 255.0)),
                 (1, (217 / 255.0, 95 / 255.0, 2 / 255.0)),
                 (2, (117 / 255.0, 112 / 255.0, 179 / 255.0)),
                 (3, (231 / 255.0, 41 / 255.0, 138 / 255.0)),
                 (4, (102 / 255.0, 166 / 255.0, 30 / 255.0)),
                 (5, (230 / 255.0, 171 / 255.0, 2 / 255.0)),
                 (6, (166 / 255.0, 118 / 255.0, 29 / 255.0)),
                 (7, (102 / 255.0, 102 / 255.0, 102 / 255.0)))

pastel1 = Colormap((0, (251 / 255.0, 180 / 255.0, 174 / 255.0)),
                   (1, (179 / 255.0, 205 / 255.0, 227 / 255.0)),
                   (2, (204 / 255.0, 235 / 255.0, 197 / 255.0)),
                   (3, (222 / 255.0, 203 / 255.0, 228 / 255.0)),
                   (4, (254 / 255.0, 217 / 255.0, 166 / 255.0)),
                   (5, (255 / 255.0, 255 / 255.0, 204 / 255.0)),
                   (6, (229 / 255.0, 216 / 255.0, 189 / 255.0)),
                   (7, (253 / 255.0, 218 / 255.0, 236 / 255.0)),
                   (8, (242 / 255.0, 242 / 255.0, 242 / 255.0)))

pastel2 = Colormap((0, (179 / 255.0, 226 / 255.0, 205 / 255.0)),
                   (1, (253 / 255.0, 205 / 255.0, 172 / 255.0)),
                   (2, (203 / 255.0, 213 / 255.0, 232 / 255.0)),
                   (3, (244 / 255.0, 202 / 255.0, 228 / 255.0)),
                   (4, (230 / 255.0, 245 / 255.0, 201 / 255.0)),
                   (5, (255 / 255.0, 242 / 255.0, 174 / 255.0)),
                   (6, (241 / 255.0, 226 / 255.0, 204 / 255.0)),
                   (7, (204 / 255.0, 204 / 255.0, 204 / 255.0)))

qualitative_colormaps = [set1, set2, set3,
                         paired, accent, dark2,
                         pastel1, pastel2]


def colorbar(height, length, colormap, category=False):
    """Return the channels of a colorbar."""
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cmin = colormap.values.min()
    cmax = colormap.values.max()
    crange = (cmax - cmin)
    if category:
        # add an extra buffer around colormap limits to show full category
        cbar = cbar * (crange + 1) + (cmin - 0.5)
        cbar = np.round(cbar)
    else:
        cbar = (cbar * crange) + colormap.values.min()

    return colormap.colorize(cbar)


def palettebar(height, length, colormap):
    """Return the channels of a palettebar."""
    cbar = np.tile(np.arange(length) * 1.0 / (length - 1), (height, 1))
    cbar = (cbar * (colormap.values.max() + 1 - colormap.values.min())
            + colormap.values.min())

    return colormap.palettize(cbar)
