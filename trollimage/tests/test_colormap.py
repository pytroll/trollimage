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

import os
import contextlib
import unittest
from trollimage import colormap
import numpy as np
from tempfile import NamedTemporaryFile
import xarray

import pytest


class TestColormapClass(unittest.TestCase):
    """Test case for the colormap object."""

    def setUp(self):
        """Set up the test case."""
        self.colormap = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                          (2, (0.0, 1.0, 1.0)),
                                          (3, (1, 1, 1)),
                                          (4, (0, 0, 0)))

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
        assert cm_.values[0] == 8
        assert cm_.values[-1] == 0
        _assert_monotonic_values(cm_, increasing=False)
        np.testing.assert_allclose(cm_.colors[0], (1.0, 1.0, 0.0))
        np.testing.assert_allclose(cm_.colors[-1], (0.0, 0.0, 0.0))

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
        cm2 = colormap.Colormap((3, (1.0, 1.0, 1.0)),
                                (4, (0.0, 0.0, 0.0)))

        cm3 = cm1 + cm2

        self.assertTrue(np.allclose(cm3.colors, cm_.colors))
        self.assertTrue(np.allclose(cm3.values, cm_.values))

    def test_colorbar(self):
        """Test colorbar."""
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1.0, 1.0, 1.0)),
                                (4, (0.0, 0.0, 0.0)))

        channels = colormap.colorbar(1, 4, cm_)
        for i in range(3):
            self.assertTrue(np.allclose(channels[i],
                                        cm_.colors[:, i],
                                        atol=0.001))

    def test_palettebar(self):
        """Test colorbar."""
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1.0, 1.0, 1.0)),
                                (4, (0.0, 0.0, 0.0)))

        channel, palette = colormap.palettebar(1, 4, cm_)

        self.assertTrue(np.allclose(channel, np.arange(4)))
        self.assertTrue(np.allclose(palette, cm_.colors))

    def test_to_rio(self):
        """Test conversion to rasterio colormap."""
        cm_ = colormap.Colormap((1, (1.0, 1.0, 0.0)),
                                (2, (0.0, 1.0, 1.0)),
                                (3, (1.0, 1.0, 1.0)),
                                (4, (0.0, 0.0, 0.0)))
        orig_colors = cm_.colors.copy()

        d = cm_.to_rio()
        exp = {1: (255, 255, 0), 2: (0, 255, 255),
               3: (255, 255, 255), 4: (0, 0, 0)}

        self.assertEqual(d, exp)
        # assert original colormap information hasn't changed
        np.testing.assert_allclose(orig_colors, cm_.colors)


COLORS_RGB1 = np.array([
    [0.0, 0.0, 0.0],
    [0.2, 0.2, 0.0],
    [0.0, 0.2, 0.2],
    [0.0, 0.2, 0.0],
])

COLORS_RGBA1 = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [0.2, 0.2, 0.0, 0.5],
    [0.0, 0.2, 0.2, 0.0],
    [0.0, 0.2, 0.0, 1.0],
])


def _mono_inc_colormap():
    """Create fake monotonically increasing colormap."""
    values = [1, 2, 3, 4]
    cmap = colormap.Colormap(values=values, colors=_four_rgb_colors())
    return cmap


def _mono_dec_colormap():
    values = [4, 3, 2, 1]
    cmap = colormap.Colormap(values=values, colors=_four_rgb_colors())
    return cmap


def _four_rgb_colors():
    return [
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1, 1, 1),
        (0, 0, 0),
    ]


class TestColormap:
    """Pytest tests for colormap objects."""

    def test_bad_color_dims(self):
        """Test passing colors that aren't RGB or RGBA."""
        # Nonsense
        with pytest.raises(ValueError, match=r".*colors.*shape.*"):
            colormap.Colormap(
                colors=np.arange(5 * 5, dtype=np.float64).reshape((5, 5)),
                values=np.linspace(0, 1, 5 * 5),
            )
        # LA
        with pytest.raises(ValueError, match=r".*colors.*shape.*"):
            colormap.Colormap(
                colors=np.arange(5 * 2, dtype=np.float64).reshape((5, 2)),
                values=np.linspace(0, 1, 5 * 2),
            )

    def test_only_colors_only_values(self):
        """Test passing only colors or only values keyword arguments."""
        with pytest.raises(ValueError, match=r"Both 'colors' and 'values'.*"):
            colormap.Colormap(
                colors=np.arange(5 * 3, dtype=np.float64).reshape((5, 3)),
            )
        with pytest.raises(ValueError, match=r"Both 'colors' and 'values'.*"):
            colormap.Colormap(
                values=np.linspace(0, 1, 5 * 3),
            )

    def test_diff_colors_values(self):
        """Test failure when colors and values have different number of elements."""
        with pytest.raises(ValueError, match=r".*same number.*"):
            colormap.Colormap(
                colors=np.arange(5 * 3, dtype=np.float64).reshape((5, 3)),
                values=np.linspace(0, 1, 6),
            )

    def test_nonfloat_colors(self):
        """Pass integer colors to colormap."""
        colormap.Colormap(
            colors=np.arange(5 * 3, dtype=np.uint8).reshape((5, 3)),
            values=np.linspace(0, 1, 5),
        )

    def test_merge_nonmonotonic(self):
        """Test that merged colormaps must have monotonic values."""
        cmap1 = colormap.Colormap(
            colors=np.arange(5 * 3).reshape((5, 3)),
            values=np.linspace(2, 3, 5),
        )
        cmap2 = colormap.Colormap(
            colors=np.arange(5 * 3).reshape((5, 3)),
            values=np.linspace(0, 1, 5),
        )
        with pytest.raises(ValueError, match=r".*monotonic.*"):
            cmap1 + cmap2
        # this should succeed
        cmap2 + cmap1

    @pytest.mark.parametrize(
        'colors',
        [
            COLORS_RGB1,
            COLORS_RGBA1
        ]
    )
    def test_to_rgb(self, colors):
        """Test 'to_rgb' method."""
        cmap = colormap.Colormap(
            values=np.linspace(0.2, 0.5, colors.shape[0]),
            colors=colors.copy(),
        )
        rgb_cmap = cmap.to_rgb()
        assert rgb_cmap.colors.shape[-1] == 3
        if colors.shape[-1] == 3:
            assert rgb_cmap is cmap
        else:
            assert rgb_cmap is not cmap

    @pytest.mark.parametrize(
        'colors',
        [
            COLORS_RGB1,
            COLORS_RGBA1
        ]
    )
    def test_to_rgba(self, colors):
        """Test 'to_rgba' method."""
        cmap = colormap.Colormap(
            values=np.linspace(0.2, 0.5, colors.shape[0]),
            colors=colors.copy(),
        )
        rgb_cmap = cmap.to_rgba()
        assert rgb_cmap.colors.shape[-1] == 4
        if colors.shape[-1] == 4:
            assert rgb_cmap is cmap
        else:
            assert rgb_cmap is not cmap

    @pytest.mark.parametrize(
        'colors1',
        [
            COLORS_RGB1,
            COLORS_RGBA1
        ]
    )
    @pytest.mark.parametrize(
        'colors2',
        [
            COLORS_RGB1,
            COLORS_RGBA1
        ]
    )
    def test_merge_rgb_rgba(self, colors1, colors2):
        """Test that two colormaps with RGB or RGBA colors can be merged."""
        cmap1 = colormap.Colormap(
            values=np.linspace(0.2, 0.5, colors1.shape[0]),
            colors=colors1,
        )
        cmap2 = colormap.Colormap(
            values=np.linspace(0.51, 0.8, colors2.shape[0]),
            colors=colors2,
        )
        new_cmap = cmap1 + cmap2
        assert new_cmap.values.shape[0] == colors1.shape[0] + colors2.shape[0]

    def test_merge_equal_values(self):
        """Test that merged colormaps can have equal values at the merge point."""
        cmap1 = colormap.Colormap(
            colors=np.arange(5 * 3).reshape((5, 3)),
            values=np.linspace(0, 1, 5),
        )
        cmap2 = colormap.Colormap(
            colors=np.arange(5 * 3).reshape((5, 3)),
            values=np.linspace(1, 2, 5),
        )
        assert cmap1.values[-1] == cmap2.values[0]
        # this should succeed
        _ = cmap1 + cmap2

    def test_merge_monotonic_decreasing(self):
        """Test that merged colormaps can be monotonically decreasing."""
        cmap1 = colormap.Colormap(
            colors=np.arange(5 * 3).reshape((5, 3)),
            values=np.linspace(2, 1, 5),
        )
        cmap2 = colormap.Colormap(
            colors=np.arange(5 * 3).reshape((5, 3)),
            values=np.linspace(1, 0, 5),
        )
        _assert_monotonic_values(cmap1, increasing=False)
        _assert_monotonic_values(cmap2, increasing=False)
        # this should succeed
        _ = cmap1 + cmap2

    @pytest.mark.parametrize('inplace', [False, True])
    def test_reverse(self, inplace):
        """Test colormap reverse."""
        values = np.linspace(0.2, 0.5, 10)
        colors = np.repeat(np.linspace(0.2, 0.8, 10)[:, np.newaxis], 3, 1)
        orig_values = values.copy()
        orig_colors = colors.copy()

        cmap = colormap.Colormap(values=values, colors=colors)
        new_cmap = cmap.reverse(inplace)
        _assert_inplace_worked(cmap, new_cmap, inplace)
        _assert_reversed_colors(cmap, new_cmap, inplace, orig_colors)
        _assert_unchanged_values(cmap, new_cmap, inplace, orig_values)

    @pytest.mark.parametrize(
        'new_range',
        [
            (0.0, 1.0),
            (1.0, 0.0),
            (210.0, 300.0),
            (300.0, 210.0),
        ])
    @pytest.mark.parametrize('inplace', [False, True])
    def test_set_range(self, new_range, inplace):
        """Test 'set_range' method."""
        values = np.linspace(0.2, 0.5, 10)
        colors = np.repeat(np.linspace(0.2, 0.8, 10)[:, np.newaxis], 3, 1)
        orig_values = values.copy()
        orig_colors = colors.copy()

        cmap = colormap.Colormap(values=values, colors=colors)
        new_cmap = cmap.set_range(*new_range, inplace)
        flipped_range = new_range[0] > new_range[1]
        _assert_inplace_worked(cmap, new_cmap, inplace)
        _assert_monotonic_values(cmap, increasing=not inplace or not flipped_range)
        _assert_monotonic_values(new_cmap, increasing=not flipped_range)
        _assert_values_changed(cmap, new_cmap, inplace, orig_values)
        _assert_unchanged_colors(cmap, new_cmap, orig_colors)

    @pytest.mark.parametrize(
        ("input_cmap_func", "expected_result"),
        [
            (_mono_inc_colormap, (0, 1, 2, 3)),
            (_mono_dec_colormap, (3, 2, 1, 0)),
        ]
    )
    def test_palettize_in_range(self, input_cmap_func, expected_result):
        """Test palettize with values inside the set range."""
        data = np.array([1, 2, 3, 4])
        cm = input_cmap_func()
        channels, colors = cm.palettize(data)
        np.testing.assert_allclose(colors, cm.colors)
        assert all(channels == expected_result)

    def test_palettize_mono_inc_out_range(self):
        """Test palettize with a value outside the colormap values."""
        cm = colormap.Colormap(values=[0, 1, 2, 3],
                               colors=_four_rgb_colors())
        data = np.arange(-1, 5)
        channels, colors = cm.palettize(data)
        np.testing.assert_allclose(colors, cm.colors)
        assert all(channels == [0, 0, 1, 2, 3, 3])

    def test_palettize_mono_inc_nan(self):
        """Test palettize with monotonic increasing values with a NaN."""
        cm = colormap.Colormap(values=[0, 1, 2, 3],
                               colors=_four_rgb_colors())
        data = np.arange(-1.0, 5.0)
        data[-1] = np.nan
        channels, colors = cm.palettize(data)
        np.testing.assert_allclose(colors, cm.colors)
        assert all(channels == [0, 0, 1, 2, 3, 3])

    def test_palettize_mono_inc_in_range_dask(self):
        """Test palettize on a dask array."""
        import dask.array as da
        data = da.from_array(np.array([[1, 2, 3, 4],
                                       [1, 2, 3, 4],
                                       [1, 2, 3, 4]]), chunks=2)
        cm = _mono_inc_colormap()
        channels, colors = cm.palettize(data)
        assert isinstance(channels, da.Array)
        np.testing.assert_allclose(colors, cm.colors)
        np.testing.assert_allclose(channels.compute(), [[0, 1, 2, 3],
                                                        [0, 1, 2, 3],
                                                        [0, 1, 2, 3]])
        assert channels.dtype == int

    @pytest.mark.parametrize(
        ("input_cmap_func", "expected_result"),
        [
            (_mono_inc_colormap, _four_rgb_colors()),
            (_mono_dec_colormap, _four_rgb_colors()[::-1]),
        ]
    )
    def test_colorize_no_interpolation(self, input_cmap_func, expected_result):
        """Test colorize."""
        data = np.array([1, 2, 3, 4])
        cm = input_cmap_func()
        channels = cm.colorize(data)
        output_colors = [channels[:, i] for i in range(data.size)]
        for output_color, expected_color in zip(output_colors, expected_result):
            np.testing.assert_allclose(output_color, expected_color, atol=0.001)

    @pytest.mark.parametrize(
        ("input_cmap_func", "expected_result"),
        [
            (_mono_inc_colormap,
             np.array([
                 [0.22178232, 1.08365532, 0.49104964],
                 [0.61069262, 0.94644083, 1.20509947],
                 [0.50011605, 0.50000605, 0.49989589],
                 [0.0, 0.0, 0.0]])),
            (_mono_dec_colormap,
             np.array([
                 [0.50011605, 0.50000605, 0.49989589],
                 [0.61069262, 0.94644083, 1.20509947],
                 [0.22178232, 1.08365532, 0.49104964],
                 [1.0, 1.0, 0.0]])),
        ]
    )
    def test_colorize_with_interpolation(self, input_cmap_func, expected_result):
        """Test colorize where data values require interpolation between colors."""
        data = np.array([1.5, 2.5, 3.5, 4])
        cm = input_cmap_func()
        channels = cm.colorize(data)
        output_colors = [channels[:, i] for i in range(data.size)]
        for output_color, expected_color in zip(output_colors, expected_result):
            np.testing.assert_allclose(output_color, expected_color, atol=0.001)

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

        cm = _mono_inc_colormap()
        channels = cm.colorize(data)
        for i, expected_channel in enumerate(expected_channels):
            current_channel = channels[i, :, :]
            assert isinstance(current_channel, da.Array)
            np.testing.assert_allclose(current_channel.compute(),
                                       expected_channel,
                                       atol=0.001)


@contextlib.contextmanager
def closed_named_temp_file(**kwargs):
    """Named temporary file context manager that closes the file after creation.

    This helps with Windows systems which can get upset with opening or
    deleting a file that is already open.

    """
    try:
        with NamedTemporaryFile(delete=False, **kwargs) as tmp_cmap:
            yield tmp_cmap.name
    finally:
        os.remove(tmp_cmap.name)


def _write_cmap_to_file(cmap_filename, cmap_data):
    ext = os.path.splitext(cmap_filename)[1]
    if ext in (".npy",):
        np.save(cmap_filename, cmap_data)
    elif ext in (".npz",):
        np.savez(cmap_filename, cmap_data)
    else:
        np.savetxt(cmap_filename, cmap_data, delimiter=",")


def _generate_cmap_test_data(color_scale, colormap_mode):
    cmap_data = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
    ], dtype=np.float64)
    if len(colormap_mode) != 3:
        _cmap_data = cmap_data
        cmap_data = np.empty((cmap_data.shape[0], len(colormap_mode)),
                             dtype=np.float64)
        if colormap_mode.startswith("V") or colormap_mode.endswith("A"):
            cmap_data[:, 0] = np.array([128, 130, 132, 134]) / 255.0
            cmap_data[:, -3:] = _cmap_data
        if colormap_mode.startswith("V") and colormap_mode.endswith("A"):
            cmap_data[:, 1] = np.array([128, 130, 132, 134]) / 255.0
    if color_scale is None or color_scale == 255:
        cmap_data = (cmap_data * 255).astype(np.uint8)
    return cmap_data


class TestFromFileCreation:
    """Tests for loading Colormaps from files."""

    @pytest.mark.parametrize("csv_filename", [None, "test.cmap"])
    @pytest.mark.parametrize("new_range", [None, (25.0, 75.0)])
    @pytest.mark.parametrize("color_scale", [1.0, 255, 65535])
    def test_csv_roundtrip(self, tmp_path, csv_filename, new_range, color_scale):
        """Test saving and loading a Colormap from a CSV file."""
        orig_cmap = colormap.brbg
        if new_range is not None:
            orig_cmap = orig_cmap.set_range(*new_range, inplace=False)
        if isinstance(csv_filename, str):
            csv_filename = str(tmp_path / csv_filename)
            res = orig_cmap.to_csv(csv_filename, color_scale=color_scale)
            assert res is None
            new_cmap = colormap.Colormap.from_file(csv_filename, color_scale=color_scale)
        else:
            res = orig_cmap.to_csv(None, color_scale=color_scale)
            assert isinstance(res, str)
            new_cmap = colormap.Colormap.from_file(res, color_scale=color_scale)
        np.testing.assert_allclose(orig_cmap.values, new_cmap.values)
        np.testing.assert_allclose(orig_cmap.colors, new_cmap.colors)

    @pytest.mark.parametrize("color_scale", [None, 1.0])
    @pytest.mark.parametrize("colormap_mode", ["RGB", "VRGB", "VRGBA"])
    @pytest.mark.parametrize("filename_suffix", [".npy", ".npz", ".csv"])
    def test_cmap_from_file(self, color_scale, colormap_mode, filename_suffix):
        """Test that colormaps can be loaded from a binary or CSV file."""
        # create the colormap file on disk
        with closed_named_temp_file(suffix=filename_suffix) as cmap_filename:
            cmap_data = _generate_cmap_test_data(color_scale, colormap_mode)
            _write_cmap_to_file(cmap_filename, cmap_data)

            unset_first_value = 128.0 / 255.0 if colormap_mode.startswith("V") else 0.0
            unset_last_value = 134.0 / 255.0 if colormap_mode.startswith("V") else 1.0
            if (color_scale is None or color_scale == 255) and colormap_mode.startswith("V"):
                unset_first_value *= 255
                unset_last_value *= 255

            first_color = [1.0, 0.0, 0.0]
            if colormap_mode == "VRGBA":
                first_color = [128.0 / 255.0] + first_color

            kwargs1 = {}
            if color_scale is not None:
                kwargs1["color_scale"] = color_scale

            cmap = colormap.Colormap.from_file(cmap_filename, **kwargs1)
            assert cmap.colors.shape[0] == 4
            np.testing.assert_equal(cmap.colors[0], first_color)
            assert cmap.values.shape[0] == 4
            assert cmap.values[0] == unset_first_value
            assert cmap.values[-1] == unset_last_value

    def test_cmap_vrgb_as_rgba(self):
        """Test that data created as VRGB still reads as RGBA."""
        with closed_named_temp_file(suffix=".npy") as cmap_filename:
            cmap_data = _generate_cmap_test_data(None, "VRGB")
            np.save(cmap_filename, cmap_data)
            cmap = colormap.Colormap.from_file(cmap_filename, colormap_mode="RGBA")
            assert cmap.colors.shape[0] == 4
            assert cmap.colors.shape[1] == 4  # RGBA
            np.testing.assert_equal(cmap.colors[0], [128 / 255., 1.0, 0, 0])
            assert cmap.values.shape[0] == 4
            assert cmap.values[0] == 0
            assert cmap.values[-1] == 1.0

    @pytest.mark.parametrize(
        ("real_mode", "forced_mode"),
        [
            ("VRGBA", "RGBA"),
            ("VRGBA", "VRGB"),
            ("RGBA", "RGB"),
        ]
    )
    @pytest.mark.parametrize("filename_suffix", [".npy", ".csv"])
    def test_cmap_bad_mode(self, real_mode, forced_mode, filename_suffix):
        """Test that reading colormaps with the wrong mode fails."""
        with closed_named_temp_file(suffix=filename_suffix) as cmap_filename:
            cmap_data = _generate_cmap_test_data(None, real_mode)
            _write_cmap_to_file(cmap_filename, cmap_data)
            # Force colormap_mode VRGBA to RGBA and we should see an exception
            with pytest.raises(ValueError):
                colormap.Colormap.from_file(cmap_filename, colormap_mode=forced_mode)

    def test_cmap_from_file_bad_shape(self):
        """Test that unknown array shape causes an error."""
        with closed_named_temp_file(suffix='.npy') as cmap_filename:
            np.save(cmap_filename, np.array([
                [0],
                [64],
                [128],
                [255],
            ]))

            with pytest.raises(ValueError):
                colormap.Colormap.from_file(cmap_filename)

    def test_cmap_from_string(self):
        """Test creating a colormap from a string."""
        s = "0,0,0,0\n1,1,1,1\n2,2,2,2"
        cmap = colormap.Colormap.from_string(s, color_scale=1)
        np.testing.assert_array_equal(cmap.values, [0, 1, 2])
        np.testing.assert_array_equal(cmap.colors, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    def test_cmap_from_np(self, tmp_path):
        """Test creating a colormap from a numpy file."""
        cmap_data = _generate_cmap_test_data(None, "RGB")
        fnp = tmp_path / "test.npy"
        np.save(fnp, cmap_data)
        cmap = colormap.Colormap.from_np(fnp, color_scale=1)
        np.testing.assert_allclose(cmap.values, [0, 0.33333333, 0.6666667, 1])
        np.testing.assert_array_equal(cmap.colors, cmap_data)

    def test_cmap_from_csv(self, tmp_path, color_scale=1):
        """Test creating a colormap from a CSV file."""
        cmap_data = _generate_cmap_test_data(None, "RGB")
        fnp = tmp_path / "test.csv"
        np.savetxt(fnp, cmap_data, delimiter=",")
        cmap = colormap.Colormap.from_csv(fnp, color_scale=1)
        np.testing.assert_allclose(cmap.values, [0, 0.33333333, 0.66666667, 1])
        np.testing.assert_array_equal(cmap.colors, cmap_data)


def test_cmap_from_ndarray():
    """Test creating a colormap from a numpy array."""
    cmap_data = _generate_cmap_test_data(None, "RGB")
    cmap = colormap.Colormap.from_ndarray(cmap_data, color_scale=1)
    np.testing.assert_allclose(cmap.values, [0, 0.33333333, 0.66666667, 1])
    np.testing.assert_array_equal(cmap.colors, cmap_data)


def test_cmap_from_name():
    """Test creating a colormap from a string representing a name."""
    cmap = colormap.Colormap.from_name("puor")
    np.testing.assert_array_equal(cmap.values, colormap.puor.values)
    np.testing.assert_array_equal(cmap.colors, colormap.puor.colors)


def test_cmap_from_sequence_of_colors():
    """Test creating a colormap from a sequence of colors."""
    colors = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    cmap = colormap.Colormap.from_sequence_of_colors(colors, color_scale=2)
    np.testing.assert_allclose(cmap.values, [0, 0.33333333, 0.66666667, 1])
    np.testing.assert_array_equal(cmap.colors*2, colors)

    vals = [0, 5, 10, 15]
    cmap = colormap.Colormap.from_sequence_of_colors(colors, values=vals, color_scale=2)
    np.testing.assert_allclose(cmap.values, [0, 5, 10, 15])


def test_build_colormap_with_int_data_and_without_meanings():
    """Test colormap building."""
    palette = np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]])
    cmap = colormap.Colormap.from_array_with_metadata(palette, np.uint8)
    np.testing.assert_array_equal(cmap.values, [0, 1])

    with pytest.raises(AttributeError):
        colormap.Colormap.from_array_with_metadata(palette/100, np.float32)

    cmap = colormap.Colormap.from_array_with_metadata(
            palette,
            np.float32,
            valid_range=[0, 100],
            scale_factor=2,
            remove_last=True)

    np.testing.assert_array_equal(cmap.values, [0, 200])

    cmap = colormap.Colormap.from_array_with_metadata(
            palette,
            np.float32,
            valid_range=[0, 100],
            scale_factor=2,
            remove_last=False)

    np.testing.assert_array_equal(cmap.values, [0, 100, 200])


def test_build_colormap_with_int_data_and_with_meanings():
    """Test colormap building."""
    palette = xarray.DataArray(np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]]),
                               dims=['value', 'band'])
    palette.attrs['palette_meanings'] = [2, 3, 4]
    cmap = colormap.Colormap.from_array_with_metadata(palette, np.uint8)
    np.testing.assert_array_equal(cmap.values, [2, 3, 4])


def _assert_monotonic_values(cmap, increasing=True):
    delta = np.diff(cmap.values)
    np.testing.assert_allclose(delta > 0, increasing)


def _assert_unchanged_values(cmap, new_cmap, inplace, orig_values):
    if inplace:
        assert cmap is new_cmap
        np.testing.assert_allclose(cmap.values, orig_values)
    else:
        assert cmap is not new_cmap
        np.testing.assert_allclose(cmap.values, orig_values)
        np.testing.assert_allclose(new_cmap.values, orig_values)


def _assert_unchanged_colors(cmap, new_cmap, orig_colors):
    np.testing.assert_allclose(cmap.colors, orig_colors)
    np.testing.assert_allclose(new_cmap.colors, orig_colors)


def _assert_reversed_colors(cmap, new_cmap, inplace, orig_colors):
    if inplace:
        assert cmap is new_cmap
        np.testing.assert_allclose(cmap.colors, orig_colors[::-1])
    else:
        assert cmap is not new_cmap
        np.testing.assert_allclose(cmap.colors, orig_colors)
        np.testing.assert_allclose(new_cmap.colors, orig_colors[::-1])


def _assert_values_changed(cmap, new_cmap, inplace, orig_values):
    assert not np.allclose(new_cmap.values, orig_values)
    if not inplace:
        np.testing.assert_allclose(cmap.values, orig_values)
    else:
        assert not np.allclose(cmap.values, orig_values)


def _assert_inplace_worked(cmap, new_cmap, inplace):
    if not inplace:
        assert new_cmap is not cmap
    else:
        assert new_cmap is cmap
