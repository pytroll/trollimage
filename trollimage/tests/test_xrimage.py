"""Module for testing the xrimage module."""

import sys
import warnings
from collections import OrderedDict
from datetime import UTC, datetime
from tempfile import NamedTemporaryFile
from typing import ClassVar
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import rasterio as rio
import xarray as xr

from trollimage import xrimage
from trollimage._xrimage_rasterio import RIODataset
from trollimage.colormap import Colormap, brbg

from .utils import assert_maximum_dask_computes


class TestXRImage:
    """Test XRImage objects."""

    def test_init(self):
        """Test object initialization."""
        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=["y", "x"])
        img = xrimage.XRImage(data)
        assert img.mode == "L"

        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]])
        img = xrimage.XRImage(data)
        assert img.mode == "L"
        assert img.data.dims == ("bands", "y", "x")

        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=["x", "y_2"])
        img = xrimage.XRImage(data)
        assert img.mode == "L"
        assert img.data.dims == ("bands", "x", "y")

        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=["x_2", "y"])
        img = xrimage.XRImage(data)
        assert img.mode == "L"
        assert img.data.dims == ("bands", "x", "y")

        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert img.mode == "RGB"

        data = xr.DataArray(
            np.arange(100).reshape(5, 5, 4), dims=["y", "x", "bands"], coords={"bands": ["Y", "Cb", "Cr", "A"]}
        )
        img = xrimage.XRImage(data)
        assert img.mode == "YCbCrA"

    def test_init_writability(self):
        """Test data is writable after init.

        Xarray >0.15 makes data read-only after expand_dims.

        """
        data = xr.DataArray([[0, 0.5, 0.5], [0.5, 0.25, 0.25]], dims=["y", "x"])
        img = xrimage.XRImage(data)
        assert img.mode == "L"
        n_arr = np.asarray(img.data)
        # if this succeeds then its writable
        n_arr[n_arr == 0.5] = 1

    def test_regression_double_format_save(self):
        """Test that double format information isn't passed to save."""
        data = xr.DataArray(
            np.arange(75).reshape(5, 5, 3) / 74.0, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]}
        )
        with mock.patch.object(xrimage.XRImage, "pil_save") as pil_save:
            img = xrimage.XRImage(data)

            img.save(filename="bla.png", fformat="png", format="png")
            assert "format" not in pil_save.call_args_list[0][1]

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_rgb_save(self):
        """Test saving RGB/A data to simple image formats."""
        data = xr.DataArray(
            np.arange(75).reshape(5, 5, 3) / 74.0, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]}
        )
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".png") as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = (np.arange(75.0).reshape(5, 5, 3) / 74.0 * 255).round()
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)  # completely opaque

        data = data.where(data > (10 / 74.0))
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".png") as tmp:
            img.save(tmp.name)

        # dask delayed save
        with NamedTemporaryFile(suffix=".png") as tmp:
            delay = img.save(tmp.name, compute=False)
            assert isinstance(delay, da.Array)
            delay.compute()

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_single_band_jpeg(self):
        """Test saving single band to jpeg formats."""
        # Single band image
        data = np.arange(75).reshape(15, 5, 1) / 74.0
        data[-1, -1, 0] = np.nan
        data = xr.DataArray(data, dims=["y", "x", "bands"], coords={"bands": ["L"]})
        # Single band image to JPEG
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".jpg") as tmp:
            img.save(tmp.name, fill_value=0)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (1, 15, 5)
            # can't check data accuracy because jpeg compression will
            # change the values

        # Jpeg fails without fill value (no alpha handling)
        with (
            NamedTemporaryFile(suffix=".jpg") as tmp,
            # make sure fill_value is mentioned in the error message
            pytest.raises(OSError, match=r".*fill_value.*"),
        ):
            img.save(tmp.name)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_single_band_png(self):
        """Test saving single band images to simple image formats."""
        # Single band image
        data = np.arange(75).reshape(15, 5, 1) / 74.0
        data[-1, -1, 0] = np.nan
        data = xr.DataArray(data, dims=["y", "x", "bands"], coords={"bands": ["L"]})
        # Single band image to JPEG
        img = xrimage.XRImage(data)

        # Single band image to PNG - min fill (check fill value scaling)
        with NamedTemporaryFile(suffix=".png") as tmp:
            img.save(tmp.name, fill_value=0)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (1, 15, 5)
            exp = (np.arange(75.0).reshape(1, 15, 5) / 74.0 * 254 + 1).round()
            exp[0, -1, -1] = 0
            np.testing.assert_allclose(file_data, exp)

        # Single band image to PNG - max fill (check fill value scaling)
        with NamedTemporaryFile(suffix=".png") as tmp:
            img.save(tmp.name, fill_value=255)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (1, 15, 5)
            exp = (np.arange(75.0).reshape(1, 15, 5) / 74.0 * 254).round()
            exp[0, -1, -1] = 255
            np.testing.assert_allclose(file_data, exp)

        # As PNG that support alpha channel
        with NamedTemporaryFile(suffix=".png") as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (2, 15, 5)
            # bad value should be transparent in alpha channel
            assert file_data[1, -1, -1] == 0
            # all other pixels should be opaque
            assert file_data[1, 0, 0] == 255

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_palettes(self):
        """Test saving paletted images to simple image formats."""
        # Single band image palettized
        from trollimage.colormap import Colormap, brbg

        data = xr.DataArray(np.arange(75).reshape(15, 5, 1) / 74.0, dims=["y", "x", "bands"], coords={"bands": ["L"]})
        img = xrimage.XRImage(data)
        img.palettize(brbg)
        with NamedTemporaryFile(suffix=".png") as tmp:
            img.save(tmp.name)
        img = xrimage.XRImage(data)
        # RGBA colormap
        bw = Colormap(
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 0.5)),
        )

        img.palettize(bw)
        with NamedTemporaryFile(suffix=".png") as tmp:
            img.save(tmp.name)

    def test_save_geotiff_float_numpy_array(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        # numpy array image - scale to 0 to 1 first
        data = xr.DataArray(
            np.arange(75).reshape((5, 5, 3)) / 75.0, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]}
        )
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        img.save(filename)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (4, 5, 5)  # alpha band added
        exp = (np.arange(75.0).reshape(5, 5, 3) / 75.0 * 255).round()
        np.testing.assert_allclose(file_data[0], exp[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp[:, :, 2])
        np.testing.assert_allclose(file_data[3], 255)  # completely opaque

    def test_save_geotiff_float_dask_array(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        img.save(filename)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (4, 5, 5)  # alpha band added
        exp = (np.arange(75.0).reshape(5, 5, 3) / 75.0 * 255).round()
        np.testing.assert_allclose(file_data[0], exp[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp[:, :, 2])
        np.testing.assert_allclose(file_data[3], 255)  # completely opaque

    def test_save_geotiff_float_dask_array_with_nans(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            img.save(filename)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (4, 5, 5)  # alpha band added
        exp = np.arange(75.0).reshape(5, 5, 3) / 75.0
        exp[exp <= 10.0 / 75.0] = 0  # numpy converts NaNs to 0s
        exp = (exp * 255).round()
        np.testing.assert_allclose(file_data[0], exp[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp[:, :, 2])
        is_null = (exp == 0).all(axis=2)
        np.testing.assert_allclose(file_data[3][~is_null], 255)  # completely opaque
        np.testing.assert_allclose(file_data[3][is_null], 0)  # completely transparent

    def test_save_geotiff_float_dask_array_with_nans_and_fill_value(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        with pytest.warns(UserWarning, match="fill value will overlap with valid data"):
            img.save(filename, fill_value=128)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (3, 5, 5)  # no alpha band
        exp = np.arange(75.0).reshape(5, 5, 3) / 75.0
        exp2 = (exp * 255).round()
        exp2[exp <= 10.0 / 75.0] = 128
        np.testing.assert_allclose(file_data[0], exp2[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp2[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp2[:, :, 2])

    def test_save_geotiff_float_dask_array_to_float(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        img.save(filename, dtype=np.float32)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (3, 5, 5)  # no alpha band
        exp = np.arange(75.0).reshape(5, 5, 3) / 75.0
        # fill value is forced to 0
        exp[exp <= 10.0 / 75.0] = 0
        np.testing.assert_allclose(file_data[0], exp[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp[:, :, 2])

    def test_save_geotiff_float_dask_array_to_float_with_nans_fill_value(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        img.save(filename, dtype=np.float32, fill_value=np.nan)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (3, 5, 5)  # no alpha band
        exp = np.arange(75.0).reshape(5, 5, 3) / 75.0
        exp[exp <= 10.0 / 75.0] = np.nan
        np.testing.assert_allclose(file_data[0], exp[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp[:, :, 2])

    def test_save_geotiff_float_dask_array_to_float_with_numeric_fill_value(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        img.save(filename, dtype=np.float32, fill_value=128)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (3, 5, 5)  # no alpha band
        exp = np.arange(75.0).reshape(5, 5, 3) / 75.0
        exp[exp <= 10.0 / 75.0] = 128
        np.testing.assert_allclose(file_data[0], exp[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp[:, :, 2])

    def test_save_geotiff_float_dask_array_to_signed_int(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        with pytest.warns(UserWarning, match="fill value will overlap with valid data"):
            img.save(filename, dtype=np.int16, fill_value=-128)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (3, 5, 5)  # no alpha band
        exp = np.arange(75.0).reshape(5, 5, 3) / 75.0
        exp2 = (exp * (2**16 - 1) - (2**15)).round()
        exp2[exp <= 10.0 / 75.0] = -128.0
        np.testing.assert_allclose(file_data[0], exp2[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp2[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp2[:, :, 2])

    def test_delayed_save_geotiff_float_dask_array(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        img = xrimage.XRImage(data)
        filename = tmp_path / "image.tif"

        delay = img.save(filename, compute=False)
        assert isinstance(delay, tuple)
        assert isinstance(delay[0], list)
        assert isinstance(delay[1], list)
        assert isinstance(delay[0][0], da.Array)
        assert isinstance(delay[1][0], RIODataset)
        da.store(*delay)
        delay[1][0].close()

    def test_save_geotiff_float_dask_array_with_alpha(self, tmp_path):
        """Test saving geotiffs when input data is float."""
        data = xr.DataArray(
            da.from_array(np.arange(75.0).reshape((5, 5, 3)) / 75.0, chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        data = data.where(data > 10.0 / 75.0)
        alpha = xr.ones_like(data[:, :, 0])
        alpha = alpha.where(data.notnull().all(dim="bands"), 0)
        alpha["bands"] = "A"
        # make a float version of a uint8 RGBA
        rgb_data = xr.concat((data, alpha), dim="bands")
        img = xrimage.XRImage(rgb_data)
        filename = tmp_path / "image.tif"

        img.save(filename)
        with rio.open(filename) as f:
            file_data = f.read()
        assert file_data.shape == (4, 5, 5)  # alpha band already existed
        exp = np.arange(75.0).reshape(5, 5, 3) / 75.0
        exp[exp <= 10.0 / 75.0] = 0  # numpy converts NaNs to 0s
        exp = (exp * 255.0).round()
        np.testing.assert_allclose(file_data[0], exp[:, :, 0])
        np.testing.assert_allclose(file_data[1], exp[:, :, 1])
        np.testing.assert_allclose(file_data[2], exp[:, :, 2])
        not_null = (alpha != 0).values
        np.testing.assert_allclose(file_data[3][not_null], 255)  # completely opaque
        np.testing.assert_allclose(file_data[3][~not_null], 0)  # completely transparent

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_datetime(self):
        """Test saving geotiffs when start_time is in the attributes."""
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})

        # "None" as start_time in the attributes
        data.attrs["start_time"] = None
        tags = _get_tags_after_writing_to_geotiff(data)
        assert "TIFFTAG_DATETIME" not in tags

        # Valid datetime
        data.attrs["start_time"] = datetime.now(UTC)
        tags = _get_tags_after_writing_to_geotiff(data)
        assert "TIFFTAG_DATETIME" in tags

    @pytest.mark.parametrize("output_ext", [".tif", ".tiff"])
    @pytest.mark.parametrize("use_dask", [False, True])
    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int(self, output_ext, use_dask):
        """Test saving geotiffs when input data is int."""
        arr = np.arange(75).reshape(5, 5, 3)
        if use_dask:
            arr = da.from_array(arr, chunks=5)

        data = xr.DataArray(arr, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
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

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_delayed(self):
        """Test saving a geotiff but not computing the result immediately."""
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            delay = img.save(tmp.name, compute=False)
            assert isinstance(delay, tuple)
            assert isinstance(delay[0], list)
            assert isinstance(delay[1], list)
            assert isinstance(delay[0][0], da.Array)
            assert isinstance(delay[1][0], RIODataset)
            da.store(*delay)
            delay[1][0].close()

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_gcps(self):
        """Test saving geotiffs when input data is int and has GCPs."""
        from pyresample import SwathDefinition
        from rasterio.control import GroundControlPoint

        gcps = [GroundControlPoint(1, 1, 100.0, 1000.0, z=0.0), GroundControlPoint(2, 3, 400.0, 2000.0, z=0.0)]
        crs = "epsg:4326"

        lons = xr.DataArray(
            da.from_array(np.arange(25).reshape(5, 5), chunks=5), dims=["y", "x"], attrs={"gcps": gcps, "crs": crs}
        )

        lats = xr.DataArray(
            da.from_array(np.arange(25).reshape(5, 5), chunks=5), dims=["y", "x"], attrs={"gcps": gcps, "crs": crs}
        )
        swath_def = SwathDefinition(lons, lats)

        data = xr.DataArray(
            da.from_array(np.arange(75).reshape(5, 5, 3), chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
            attrs={"area": swath_def},
        )
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                fgcps, fcrs = f.gcps
            for ref, val in zip(gcps, fgcps, strict=True):
                assert ref.col == val.col
                assert ref.row == val.row
                assert ref.x == val.x
                assert ref.y == val.y
                assert ref.z == val.z
            assert crs == fcrs

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_no_gcp_swath(self):
        """Test saving geotiffs when input data whose SwathDefinition has no GCPs.

        If shouldn't fail, but it also shouldn't have a non-default transform.

        """
        from pyresample import SwathDefinition

        lons = xr.DataArray(da.from_array(np.arange(25).reshape(5, 5), chunks=5), dims=["y", "x"], attrs={})

        lats = xr.DataArray(da.from_array(np.arange(25).reshape(5, 5), chunks=5), dims=["y", "x"], attrs={})
        swath_def = SwathDefinition(lons, lats)

        data = xr.DataArray(
            da.from_array(np.arange(75).reshape(5, 5, 3), chunks=5),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
            attrs={"area": swath_def},
        )
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name)
            with rio.open(tmp.name) as f:
                assert f.transform.a == 1.0
                assert f.transform.b == 0.0
                assert f.transform.c == 0.0

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_rio_colormap(self):
        """Test saving geotiffs when input data is int and a rasterio colormap is provided."""
        exp_cmap = {i: (i, 255 - i, i, 255) for i in range(256)}
        data = xr.DataArray(
            da.from_array(np.arange(81).reshape(9, 9, 1), chunks=9), dims=["y", "x", "bands"], coords={"bands": ["P"]}
        )
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name, keep_palette=True, cmap=exp_cmap)
            with rio.open(tmp.name) as f:
                file_data = f.read()
                cmap = f.colormap(1)
            assert file_data.shape == (1, 9, 9)  # no alpha band
            exp = np.arange(81).reshape(9, 9, 1)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            assert cmap == exp_cmap

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_fill(self):
        """Test saving geotiffs when input data is int and a fill value is specified."""
        data = np.arange(75).reshape(5, 5, 3)
        # second pixel is all bad
        # pixel [0, 1, 1] is also naturally 5 by arange above
        data[0, 1, :] = 5
        data = xr.DataArray(
            da.from_array(data, chunks=5),
            dims=["y", "x", "bands"],
            attrs={"_FillValue": 5},
            coords={"bands": ["R", "G", "B"]},
        )
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
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

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_fill_and_alpha(self):
        """Test saving int geotiffs with a fill value and input alpha band."""
        data = np.arange(75).reshape(5, 5, 3)
        # second pixel is all bad
        # pixel [0, 1, 1] is also naturally 5 by arange above
        data[0, 1, :] = 5
        data = xr.DataArray(
            da.from_array(data, chunks=5),
            dims=["y", "x", "bands"],
            attrs={"_FillValue": 5},
            coords={"bands": ["R", "G", "B"]},
        )
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
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

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_area_def(self):
        """Test saving a integer image with an AreaDefinition."""
        from pyproj import CRS
        from pyresample import AreaDefinition

        crs = CRS.from_user_input(4326)
        area_def = AreaDefinition("test", "test", "", crs, 5, 5, [-300, -250, 200, 250])

        data = xr.DataArray(
            np.arange(75).reshape(5, 5, 3),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
            attrs={"area": area_def},
        )
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
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

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    @pytest.mark.parametrize(
        "cmap",
        [
            Colormap(*tuple((i, (i / 20, i / 20, i / 20)) for i in range(20))),
            Colormap(*tuple((i + 0.00001, (i / 20, i / 20, i / 20)) for i in range(20))),
            Colormap(*tuple((i if i != 2 else 2.00000001, (i / 20, i / 20, i / 20)) for i in range(20))),
        ],
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
        exp_cmap.update(dict.fromkeys(range(20, 256), (0, 0, 0, 255)))
        data = xr.DataArray(
            da.from_array(np.arange(81).reshape(9, 9, 1), chunks=9), dims=["y", "x", "bands"], coords={"bands": ["P"]}
        )
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name, keep_palette=True, cmap=cmap)
            with rio.open(tmp.name) as f:
                file_data = f.read()
                cmap = f.colormap(1)
            assert file_data.shape == (1, 9, 9)  # no alpha band
            exp = np.arange(81).reshape(9, 9, 1)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            assert cmap == exp_cmap

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_int_with_bad_cmap(self):
        """Test saving integer data to geotiff with a bad colormap."""
        t_cmap = Colormap(*tuple((i, (i / 20, i / 20, i / 20)) for i in range(20)))
        bad_cmap = [[i, [i, i, i]] for i in range(256)]
        data = xr.DataArray(
            da.from_array(np.arange(81).reshape(9, 9, 1), chunks=9), dims=["y", "x", "bands"], coords={"bands": ["P"]}
        )
        img = xrimage.XRImage(data)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            with pytest.raises(ValueError, match="Colormap is not formatted correctly"):
                img.save(tmp.name, keep_palette=True, cmap=bad_cmap)
            with pytest.raises(ValueError, match="Rasterio only supports 8-bit colormaps"):
                img.save(tmp.name, keep_palette=True, cmap=t_cmap, dtype="uint16")

    def test_save_geotiff_with_cmap_and_fill_value(self, tmp_path):
        """Test saving GeoTIFF with colormap and fill value."""
        import rasterio

        test_file = tmp_path / "test.tif"
        fv = np.uint8(42)
        arr = np.ones((1, 5, 5), dtype="uint8")
        arr[0, 2, 2] = 255
        data = xr.DataArray(arr, dims=["bands", "y", "x"], attrs={"_FillValue": 255}, coords={"bands": ["P"]})
        img = xrimage.XRImage(data)
        img.save(test_file, keep_palette=True, cmap=brbg, fill_value=fv)
        with rasterio.open(test_file) as f:
            cont = f.read()
            assert cont[0, 2, 2] == fv

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_geotiff_closed_file(self):
        """Test saving geotiffs when the geotiff file has been closed.

        This is to mimic a situation where garbage collection would cause the
        file handler to close the underlying geotiff file that will be written
        to.

        """
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            results = img.save(tmp.name, compute=False)
            results[1][0].close()  # mimic garbage collection
            da.store(results[0], results[1])
            results[1][0].close()  # required to flush writes to disk
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_jp2_int(self):
        """Test saving jp2000 when input data is int."""
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".jp2") as tmp:
            img.save(tmp.name, quality=100, reversible=True)
            with rio.open(tmp.name) as f:
                file_data = f.read()
            assert file_data.shape == (4, 5, 5)  # alpha band added
            exp = np.arange(75).reshape(5, 5, 3)
            np.testing.assert_allclose(file_data[0], exp[:, :, 0])
            np.testing.assert_allclose(file_data[1], exp[:, :, 1])
            np.testing.assert_allclose(file_data[2], exp[:, :, 2])
            np.testing.assert_allclose(file_data[3], 255)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_cloud_optimized_geotiff(self):
        """Test saving cloud optimized geotiffs."""
        # trigger COG driver to create 2 overview levels
        # COG driver is only available in GDAL 3.1 or later
        if rio.__gdal_version__ >= "3.1":
            data = xr.DataArray(
                np.arange(1200 * 1200 * 3).reshape(1200, 1200, 3),
                dims=["y", "x", "bands"],
                coords={"bands": ["R", "G", "B"]},
            )
            img = xrimage.XRImage(data)
            assert np.issubdtype(img.data.dtype, np.integer)
            with NamedTemporaryFile(suffix=".tif") as tmp:
                img.save(tmp.name, tiled=True, overviews=[], driver="COG")
                with rio.open(tmp.name) as f:
                    # The COG driver should add a tag indicating layout
                    assert f.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
                    assert len(f.overviews(1)) == 2

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_overviews(self):
        """Test saving geotiffs with overviews."""
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name, overviews=[2, 4])
            with rio.open(tmp.name) as f:
                assert len(f.overviews(1)) == 2

        # auto-levels
        data = np.zeros(25 * 25 * 3, dtype=np.uint8).reshape(25, 25, 3)
        data = xr.DataArray(data, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name, overviews=[], overviews_minsize=2)
            with rio.open(tmp.name) as f:
                assert len(f.overviews(1)) == 4

        # auto-levels and resampling
        data = np.zeros(25 * 25 * 3, dtype=np.uint8).reshape(25, 25, 3)
        data = xr.DataArray(data, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name, overviews=[], overviews_minsize=2, overviews_resampling="average")
            with rio.open(tmp.name) as f:
                # no way to check resampling method from the file
                assert len(f.overviews(1)) == 4

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_tags(self):
        """Test saving geotiffs with tags."""
        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        tags = {"avg": img.data.mean(), "current_song": "disco inferno"}
        assert np.issubdtype(img.data.dtype, np.integer)
        with NamedTemporaryFile(suffix=".tif") as tmp:
            img.save(tmp.name, tags=tags)
            tags["avg"] = "37.0"
            with rio.open(tmp.name) as f:
                assert f.tags() == tags

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_gamma_single_value(self, dtype):
        """Test gamma correction for one value for all channels."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 75.0
        data = xr.DataArray(arr, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        img.gamma(0.5)
        assert img.data.dtype == dtype
        np.testing.assert_allclose(img.data.values, arr**2)
        assert img.data.attrs["enhancement_history"][0] == {"gamma": 0.5}

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    @pytest.mark.parametrize(
        ("gamma_val"),
        [
            (None),
            (1.0),
            ([1.0, 1.0, 1.0]),
            ([None, None, None]),
        ],
    )
    def test_gamma_noop(self, gamma_val, dtype):
        """Test variety of unity gamma corrections."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 75.0
        data = xr.DataArray(arr, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        img.gamma(gamma_val)
        assert img.data.dtype == dtype
        np.testing.assert_equal(img.data.values, arr)
        assert "enhancement_history" not in img.data.attrs

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_gamma_per_channel(self, dtype):
        """Test gamma correction with a value for each channel."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 75.0
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        img.gamma([2.0, 2.0, 2.0])
        assert img.data.dtype == dtype
        assert img.data.attrs["enhancement_history"][0] == {"gamma": [2.0, 2.0, 2.0]}
        np.testing.assert_allclose(img.data.values, arr**0.5, atol=1e-7)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_crude_stretch(self, dtype):
        """Check crude stretching."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3)
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        img.crude_stretch()
        red = img.data.sel(bands="R")
        green = img.data.sel(bands="G")
        blue = img.data.sel(bands="B")
        enhs = img.data.attrs["enhancement_history"][0]
        scale_expected = np.array([0.01388889, 0.01388889, 0.01388889])
        offset_expected = np.array([0.0, -0.01388889, -0.02777778])
        assert img.data.dtype == dtype
        np.testing.assert_allclose(enhs["scale"].values, scale_expected)
        np.testing.assert_allclose(enhs["offset"].values, offset_expected)
        expected_red = arr[:, :, 0] / 72.0
        np.testing.assert_allclose(red, expected_red.astype(dtype), rtol=1e-6)
        expected_green = (arr[:, :, 1] - 1.0) / (73.0 - 1.0)
        np.testing.assert_allclose(green, expected_green.astype(dtype), rtol=1e-6)
        expected_blue = (arr[:, :, 2] - 2.0) / (74.0 - 2.0)
        np.testing.assert_allclose(blue, expected_blue.astype(dtype), rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_crude_stretch_with_limits(self, dtype):
        """Test crude stretch with different input dtypes."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3)
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        img.crude_stretch(0, 74)
        assert img.data.dtype == dtype
        np.testing.assert_allclose(img.data.values, arr / 74.0, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.uint8, int])
    # include a stretch within 8-bit uint and outside
    @pytest.mark.parametrize("max_stretch", [74, 74 * 4])
    def test_crude_stretch_integer_data(self, dtype, max_stretch):
        """Test crude stretch with different input integer dtypes."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3)
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        img.crude_stretch(0, max_stretch)
        assert img.data.dtype == np.float32
        np.testing.assert_allclose(img.data.values, arr.astype(np.float32) / max_stretch, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_invert_single_parameter(self, dtype):
        """Check inversion of the image for single inversion parameter."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 75.0
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)

        img.invert(True)
        enhs = img.data.attrs["enhancement_history"][0]
        assert enhs == {"scale": -1, "offset": 1}
        assert img.data.dtype == dtype
        assert np.allclose(img.data.values, 1 - arr)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_invert_parameter_for_each_channel(self, dtype):
        """Check inversion of the image for single inversion parameter."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 75.0
        data = xr.DataArray(arr, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)

        img.invert([True, False, True])
        offset = xr.DataArray(np.array([1, 0, 1]), dims=["bands"], coords={"bands": ["R", "G", "B"]})
        scale = xr.DataArray(np.array([-1, 1, -1]), dims=["bands"], coords={"bands": ["R", "G", "B"]})
        np.testing.assert_allclose(img.data.values, (data * scale + offset).values)
        assert img.data.dtype == dtype

    @pytest.mark.parametrize("with_bands", [False, True])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_linear_stretch_single_band(self, with_bands, dtype):
        """Test linear stretching with cutoffs for single band data."""
        new_shape = (5, 5)
        kwargs = {"dims": ("y", "x")}
        if with_bands:
            kwargs["dims"] += ("bands",)
            kwargs["coords"] = {"bands": ["L"]}
            new_shape += (1,)
        arr = np.arange(25, dtype=dtype).reshape(*new_shape) / 74.0
        data = xr.DataArray(arr.copy(), **kwargs)
        img = xrimage.XRImage(data)
        with assert_maximum_dask_computes(0):
            img.stretch_linear()
        assert img.data.dtype == dtype
        enhs = img.data.attrs["enhancement_history"][0]
        np.testing.assert_allclose(enhs["scale"].values, np.array([3.114479], dtype=dtype), atol=1e-6)
        np.testing.assert_allclose(enhs["offset"].values, np.array([-0.00505051], dtype=dtype), atol=1e-8)
        res = np.array(
            [
                [
                    [-0.005051, 0.037037, 0.079125, 0.121212, 0.1633],
                    [0.205387, 0.247475, 0.289562, 0.33165, 0.373737],
                    [0.415825, 0.457913, 0.5, 0.542088, 0.584175],
                    [0.6262627, 0.66835034, 0.71043783, 0.7525254, 0.79461294],
                    [0.83670044, 0.87878805, 0.9208756, 0.9629631, 1.0050505],
                ]
            ],
            dtype=dtype,
        )
        if with_bands:
            # switch from (1, 5, 5) to (5, 5, 1)
            res = res.reshape(new_shape)

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_linear_stretch(self, dtype):
        """Test linear stretching with cutoffs."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 74.0
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        with assert_maximum_dask_computes(0):
            img.stretch_linear()
        assert img.data.dtype == dtype
        enhs = img.data.attrs["enhancement_history"][0]
        np.testing.assert_allclose(enhs["scale"].values, np.array([1.03815937, 1.03815937, 1.03815937]))
        np.testing.assert_allclose(enhs["offset"].values, np.array([-0.00505051, -0.01907969, -0.03310887]), atol=1e-8)
        res = np.array(
            [
                [
                    [-0.005051, -0.005051, -0.005051],
                    [0.037037, 0.037037, 0.037037],
                    [0.079125, 0.079125, 0.079125],
                    [0.121212, 0.121212, 0.121212],
                    [0.1633, 0.1633, 0.1633],
                ],
                [
                    [0.205387, 0.205387, 0.205387],
                    [0.247475, 0.247475, 0.247475],
                    [0.289562, 0.289562, 0.289562],
                    [0.33165, 0.33165, 0.33165],
                    [0.373737, 0.373737, 0.373737],
                ],
                [
                    [0.415825, 0.415825, 0.415825],
                    [0.457912, 0.457912, 0.457912],
                    [0.5, 0.5, 0.5],
                    [0.542088, 0.542088, 0.542088],
                    [0.584175, 0.584175, 0.584175],
                ],
                [
                    [0.626263, 0.626263, 0.626263],
                    [0.66835, 0.66835, 0.66835],
                    [0.710438, 0.710438, 0.710438],
                    [0.752525, 0.752525, 0.752525],
                    [0.794613, 0.794613, 0.794613],
                ],
                [
                    [0.8367, 0.8367, 0.8367],
                    [0.878788, 0.878788, 0.878788],
                    [0.920875, 0.920875, 0.920875],
                    [0.962963, 0.962963, 0.962963],
                    [1.005051, 1.005051, 1.005051],
                ],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_linear_stretch_does_not_affect_alpha(self, dtype):
        """Test linear stretching with cutoffs."""
        arr = np.arange(100, dtype=dtype).reshape(5, 5, 4) / 74.0
        arr[:, :, -1] = 1  # alpha channel, fully opaque
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B", "A"]})
        img = xrimage.XRImage(data)
        with assert_maximum_dask_computes(0):
            img.stretch_linear((0.005, 0.005))
        assert img.data.dtype == dtype
        res = np.array(
            [
                [
                    [-0.005051, -0.005051, -0.005051, 1.0],
                    [0.037037, 0.037037, 0.037037, 1.0],
                    [0.079125, 0.079125, 0.079125, 1.0],
                    [0.121212, 0.121212, 0.121212, 1.0],
                    [0.1633, 0.1633, 0.1633, 1.0],
                ],
                [
                    [0.205387, 0.205387, 0.205387, 1.0],
                    [0.247475, 0.247475, 0.247475, 1.0],
                    [0.289562, 0.289562, 0.289562, 1.0],
                    [0.33165, 0.33165, 0.33165, 1.0],
                    [0.373737, 0.373737, 0.373737, 1.0],
                ],
                [
                    [0.415825, 0.415825, 0.415825, 1.0],
                    [0.457912, 0.457912, 0.457912, 1.0],
                    [0.5, 0.5, 0.5, 1.0],
                    [0.542088, 0.542088, 0.542088, 1.0],
                    [0.584175, 0.584175, 0.584175, 1.0],
                ],
                [
                    [0.626263, 0.626263, 0.626263, 1.0],
                    [0.66835, 0.66835, 0.66835, 1.0],
                    [0.710438, 0.710438, 0.710438, 1.0],
                    [0.752525, 0.752525, 0.752525, 1.0],
                    [0.794613, 0.794613, 0.794613, 1.0],
                ],
                [
                    [0.8367, 0.8367, 0.8367, 1.0],
                    [0.878788, 0.878788, 0.878788, 1.0],
                    [0.920875, 0.920875, 0.920875, 1.0],
                    [0.962963, 0.962963, 0.962963, 1.0],
                    [1.005051, 1.005051, 1.005051, 1.0],
                ],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_linear_stretch_does_not_affect_alpha_with_partial_cutoffs(self, dtype):
        """Test linear stretching with cutoffs."""
        arr = np.arange(100, dtype=dtype).reshape(5, 5, 4) / 74.0
        arr[:, :, -1] = 1  # alpha channel, fully opaque
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B", "A"]})
        img = xrimage.XRImage(data)
        with assert_maximum_dask_computes(0):
            img.stretch_linear([(0.005, 0.005), (0.005, 0.005), (0.005, 0.005)])
        assert img.data.dtype == dtype
        res = np.array(
            [
                [
                    [-0.005051, -0.005051, -0.005051, 1.0],
                    [0.037037, 0.037037, 0.037037, 1.0],
                    [0.079125, 0.079125, 0.079125, 1.0],
                    [0.121212, 0.121212, 0.121212, 1.0],
                    [0.1633, 0.1633, 0.1633, 1.0],
                ],
                [
                    [0.205387, 0.205387, 0.205387, 1.0],
                    [0.247475, 0.247475, 0.247475, 1.0],
                    [0.289562, 0.289562, 0.289562, 1.0],
                    [0.33165, 0.33165, 0.33165, 1.0],
                    [0.373737, 0.373737, 0.373737, 1.0],
                ],
                [
                    [0.415825, 0.415825, 0.415825, 1.0],
                    [0.457912, 0.457912, 0.457912, 1.0],
                    [0.5, 0.5, 0.5, 1.0],
                    [0.542088, 0.542088, 0.542088, 1.0],
                    [0.584175, 0.584175, 0.584175, 1.0],
                ],
                [
                    [0.626263, 0.626263, 0.626263, 1.0],
                    [0.66835, 0.66835, 0.66835, 1.0],
                    [0.710438, 0.710438, 0.710438, 1.0],
                    [0.752525, 0.752525, 0.752525, 1.0],
                    [0.794613, 0.794613, 0.794613, 1.0],
                ],
                [
                    [0.8367, 0.8367, 0.8367, 1.0],
                    [0.878788, 0.878788, 0.878788, 1.0],
                    [0.920875, 0.920875, 0.920875, 1.0],
                    [0.962963, 0.962963, 0.962963, 1.0],
                    [1.005051, 1.005051, 1.005051, 1.0],
                ],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_linear_stretch_does_affect_alpha_with_explicit_cutoffs(self, dtype):
        """Test linear stretching with full explicit cutoffs."""
        arr = np.arange(100, dtype=dtype).reshape(5, 5, 4) / 74.0
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B", "A"]})
        img = xrimage.XRImage(data)
        with assert_maximum_dask_computes(0):
            img.stretch_linear([(0.005, 0.005), (0.005, 0.005), (0.005, 0.005), (0.005, 0.005)])
        assert img.data.dtype == dtype
        res = np.array(
            [
                [
                    [-0.005051, -0.005051, -0.005051, -0.005051],
                    [0.037037, 0.037037, 0.037037, 0.037037],
                    [0.079125, 0.079125, 0.079125, 0.079125],
                    [0.121212, 0.121212, 0.121212, 0.121212],
                    [0.1633, 0.1633, 0.1633, 0.1633],
                ],
                [
                    [0.205387, 0.205387, 0.205387, 0.205387],
                    [0.247475, 0.247475, 0.247475, 0.247475],
                    [0.289562, 0.289562, 0.289562, 0.289562],
                    [0.33165, 0.33165, 0.33165, 0.33165],
                    [0.373737, 0.373737, 0.373737, 0.373737],
                ],
                [
                    [0.415825, 0.415825, 0.415825, 0.415825],
                    [0.457912, 0.457912, 0.457912, 0.457912],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.542088, 0.542088, 0.542088, 0.542088],
                    [0.584175, 0.584175, 0.584175, 0.584175],
                ],
                [
                    [0.626263, 0.626263, 0.626263, 0.626263],
                    [0.66835, 0.66835, 0.66835, 0.66835],
                    [0.710438, 0.710438, 0.710438, 0.710438],
                    [0.752525, 0.752525, 0.752525, 0.752525],
                    [0.794613, 0.794613, 0.794613, 0.794613],
                ],
                [
                    [0.8367, 0.8367, 0.8367, 0.8367],
                    [0.878788, 0.878788, 0.878788, 0.878788],
                    [0.920875, 0.920875, 0.920875, 0.920875],
                    [0.962963, 0.962963, 0.962963, 0.962963],
                    [1.005051, 1.005051, 1.005051, 1.005051],
                ],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    @pytest.mark.parametrize(
        ("dtype", "max_val", "exp_min", "exp_max"),
        [
            (np.uint8, 255, -0.005358012691140175, 1.0053772069513798),
            (np.int8, 127, -0.004926108196377754, 1.0058689523488282),
            (np.uint16, 65535, -0.005050825305515899, 1.005050893505104),
            (np.int16, 32767, -0.005052744992717635, 1.0050527782880818),
            (np.uint32, 4294967295, -0.005050505077517274, 1.0050505395923495),
            (np.int32, 2147483647, -0.00505050499355784, 1.0050505395923495),
            (int, 2147483647, -0.00505050499355784, 1.0050505395923495),
        ],
    )
    def test_linear_stretch_integers(self, dtype, max_val, exp_min, exp_max):
        """Test linear stretch with low-bit unsigned integer data."""
        arr = np.linspace(0, max_val, num=75, dtype=dtype).reshape(5, 5, 3)
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        with assert_maximum_dask_computes(0):
            img.stretch_linear()
        assert img.data.values.min() == pytest.approx(exp_min)
        assert img.data.values.max() == pytest.approx(exp_max)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_histogram_stretch(self, dtype):
        """Test histogram stretching."""
        arr = da.arange(75, dtype=dtype).reshape(5, 5, 3) / 74.0
        arr = arr.rechunk((2, 2, 1))
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        with assert_maximum_dask_computes(0):
            img.stretch("histogram")
        enhs = img.data.attrs["enhancement_history"][0]
        assert enhs == {"hist_equalize": True}
        assert img.data.dtype == dtype
        assert img.data.chunks == ((2, 2, 1), (2, 2, 1), (1, 1, 1))
        res = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.04166667, 0.04166667, 0.04166667],
                    [0.08333333, 0.08333333, 0.08333333],
                    [0.125, 0.125, 0.125],
                    [0.16666667, 0.16666667, 0.16666667],
                ],
                [
                    [0.20833333, 0.20833333, 0.20833333],
                    [0.25, 0.25, 0.25],
                    [0.29166667, 0.29166667, 0.29166667],
                    [0.33333333, 0.33333333, 0.33333333],
                    [0.375, 0.375, 0.375],
                ],
                [
                    [0.41666667, 0.41666667, 0.41666667],
                    [0.45833333, 0.45833333, 0.45833333],
                    [0.5, 0.5, 0.5],
                    [0.54166667, 0.54166667, 0.54166667],
                    [0.58333333, 0.58333333, 0.58333333],
                ],
                [
                    [0.625, 0.625, 0.625],
                    [0.66666667, 0.66666667, 0.66666667],
                    [0.70833333, 0.70833333, 0.70833333],
                    [0.75, 0.75, 0.75],
                    [0.79166667, 0.79166667, 0.79166667],
                ],
                [
                    [0.83333333, 0.83333333, 0.83333333],
                    [0.875, 0.875, 0.875],
                    [0.91666667, 0.91666667, 0.91666667],
                    [0.95833333, 0.95833333, 0.95833333],
                    [0.99951172, 0.99951172, 0.99951172],
                ],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    @pytest.mark.parametrize(
        ("min_stretch", "max_stretch"),
        [
            (None, None),
            ([0.0, 1.0 / 74.0, 2.0 / 74.0], [72.0 / 74.0, 73.0 / 74.0, 1.0]),
        ],
    )
    @pytest.mark.parametrize("base", ["e", "10", "2"])
    def test_logarithmic_stretch(self, min_stretch, max_stretch, base, dtype):
        """Test logarithmic strecthing."""
        arr = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 74.0
        data = xr.DataArray(arr.copy(), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(data)
            img.stretch(stretch="logarithmic", min_stretch=min_stretch, max_stretch=max_stretch, base=base)
        enhs = img.data.attrs["enhancement_history"][0]
        assert enhs == {"log_factor": 100.0}
        assert img.data.dtype == dtype
        res = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.35484693, 0.35484693, 0.35484693],
                    [0.48307087, 0.48307087, 0.48307087],
                    [0.5631469, 0.5631469, 0.5631469],
                    [0.62151902, 0.62151902, 0.62151902],
                ],
                [
                    [0.66747806, 0.66747806, 0.66747806],
                    [0.70538862, 0.70538862, 0.70538862],
                    [0.73765396, 0.73765396, 0.73765396],
                    [0.76573946, 0.76573946, 0.76573946],
                    [0.79060493, 0.79060493, 0.79060493],
                ],
                [
                    [0.81291336, 0.81291336, 0.81291336],
                    [0.83314196, 0.83314196, 0.83314196],
                    [0.85164569, 0.85164569, 0.85164569],
                    [0.86869572, 0.86869572, 0.86869572],
                    [0.88450394, 0.88450394, 0.88450394],
                ],
                [
                    [0.899239, 0.899239, 0.899239],
                    [0.9130374, 0.9130374, 0.9130374],
                    [0.92601114, 0.92601114, 0.92601114],
                    [0.93825325, 0.93825325, 0.93825325],
                    [0.94984187, 0.94984187, 0.94984187],
                ],
                [
                    [0.96084324, 0.96084324, 0.96084324],
                    [0.97131402, 0.97131402, 0.97131402],
                    [0.98130304, 0.98130304, 0.98130304],
                    [0.99085269, 0.99085269, 0.99085269],
                    [1.0, 1.0, 1.0],
                ],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_weber_fechner_stretch(self, dtype):
        """Test applying S=2.3klog10I+C to the data."""
        from trollimage import xrimage

        arr = np.arange(75.0, dtype=dtype).reshape(5, 5, 3) / 74.0
        data = xr.DataArray(arr.copy() + 0.1, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        img.stretch_weber_fechner(2.5, 0.2)
        enhs = img.data.attrs["enhancement_history"][0]
        assert enhs == {"weber_fechner": (2.5, 0.2)}
        assert img.data.dtype == dtype
        res = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0993509],
                    [0.25663552, 0.40460747, 0.54430866],
                    [0.6766145, 0.8022693, 0.92190945],
                ],
                [
                    [1.0360844, 1.1452721, 1.2498899],
                    [1.350305, 1.446842, 1.5397894],
                    [1.629405, 1.7159187, 1.7995386],
                    [1.8804514, 1.9588281, 2.0348217],
                    [2.1085734, 2.1802118, 2.2498538],
                ],
                [
                    [2.3176088, 2.3835757, 2.4478464],
                    [2.5105066, 2.571634, 2.631303],
                    [2.689581, 2.7465308, 2.8022122],
                    [2.8566809, 2.9099874, 2.9621818],
                    [3.013308, 3.0634098, 3.1125278],
                ],
                [
                    [3.1606987, 3.207959, 3.2543423],
                    [3.299881, 3.344605, 3.3885431],
                    [3.431722, 3.4741676, 3.515905],
                    [3.5569568, 3.597345, 3.6370919],
                    [3.6762161, 3.714738, 3.7526748],
                ],
                [
                    [3.7900448, 3.826864, 3.863149],
                    [3.8989153, 3.934177, 3.968948],
                    [4.0032415, 4.037072, 4.07045],
                    [4.103389, 4.135899, 4.1679916],
                    [4.199678, 4.2309675, 4.2618704],
                ],
            ],
            dtype=dtype,
        )

        np.testing.assert_allclose(img.data.values, res, atol=1.0e-6)

    def test_jpeg_save(self):
        """Test saving to jpeg."""

    def test_gtiff_save(self):
        """Test saving to geotiff."""

    def test_save_masked(self):
        """Test saving masked data."""

    def test_LA_save(self):
        """Test LA saving."""

    def test_L_save(self):
        """Test L saving."""

    def test_P_save(self):
        """Test P saving."""

    def test_PA_save(self):
        """Test PA saving."""

    def test_convert_modes(self):
        """Test modes convertions."""
        from trollimage.colormap import Colormap, brbg

        # RGBA colormap
        bw = Colormap(
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 0.5)),
        )

        arr1 = np.arange(150).reshape(1, 15, 10) / 150.0
        arr2 = np.append(arr1, np.ones(150).reshape(arr1.shape)).reshape(2, 15, 10)
        arr3 = (np.arange(150).reshape(2, 15, 5) / 15).astype("int64")
        dataset1 = xr.DataArray(arr1.copy(), dims=["bands", "y", "x"], coords={"bands": ["L"]})
        dataset2 = xr.DataArray(arr2.copy(), dims=["bands", "x", "y"], coords={"bands": ["L", "A"]})
        dataset3 = xr.DataArray(arr3.copy(), dims=["bands", "x", "y"], coords={"bands": ["P", "A"]})

        img = xrimage.XRImage(dataset1)
        new_img = img.convert(img.mode)
        assert new_img is not None
        # make sure it is a copy
        assert new_img is not img
        assert new_img.data is not img.data

        # L -> LA (int)
        with assert_maximum_dask_computes(1):
            img = xrimage.XRImage((dataset1 * 150).astype(np.uint8))
            img.data.attrs["_FillValue"] = 0  # set fill value
            img = img.convert("LA")
            assert np.issubdtype(img.data.dtype, np.integer)
            assert img.mode == "LA"
            assert len(img.data.coords["bands"]) == 2
            # make sure the alpha band is all opaque except the first pixel
            alpha = img.data.sel(bands="A").values.ravel()
            np.testing.assert_allclose(alpha[0], 0)
            np.testing.assert_allclose(alpha[1:], 255)

        # L -> LA (float)
        with assert_maximum_dask_computes(1):
            img = xrimage.XRImage(dataset1)
            img = img.convert("LA")
            assert img.mode == "LA"
            assert len(img.data.coords["bands"]) == 2
            # make sure the alpha band is all opaque
            np.testing.assert_allclose(img.data.sel(bands="A"), 1.0)

        # LA -> L (float)
        with assert_maximum_dask_computes(0):
            img = img.convert("L")
            assert img.mode == "L"
            assert len(img.data.coords["bands"]) == 1

        # L -> RGB (float)
        with assert_maximum_dask_computes(1):
            img = img.convert("RGB")
            assert img.mode == "RGB"
            assert len(img.data.coords["bands"]) == 3
            data = img.data.compute()
            np.testing.assert_allclose(data.sel(bands=["R"]), arr1)
            np.testing.assert_allclose(data.sel(bands=["G"]), arr1)
            np.testing.assert_allclose(data.sel(bands=["B"]), arr1)

        # RGB -> RGBA (float)
        with assert_maximum_dask_computes(1):
            img = img.convert("RGBA")
            assert img.mode == "RGBA"
            assert len(img.data.coords["bands"]) == 4
            assert np.issubdtype(img.data.dtype, np.floating)
            data = img.data.compute()
            np.testing.assert_allclose(data.sel(bands=["R"]), arr1)
            np.testing.assert_allclose(data.sel(bands=["G"]), arr1)
            np.testing.assert_allclose(data.sel(bands=["B"]), arr1)
            # make sure the alpha band is all opaque
            np.testing.assert_allclose(data.sel(bands="A"), 1.0)

        # RGB -> RGBA (int)
        with assert_maximum_dask_computes(1):
            img = xrimage.XRImage((dataset1 * 150).astype(np.uint8))
            img = img.convert("RGB")  # L -> RGB
            assert np.issubdtype(img.data.dtype, np.integer)
            img = img.convert("RGBA")
            assert img.mode == "RGBA"
            assert len(img.data.coords["bands"]) == 4
            assert np.issubdtype(img.data.dtype, np.integer)
            data = img.data.compute()
            np.testing.assert_allclose(data.sel(bands=["R"]), (arr1 * 150).astype(np.uint8))
            np.testing.assert_allclose(data.sel(bands=["G"]), (arr1 * 150).astype(np.uint8))
            np.testing.assert_allclose(data.sel(bands=["B"]), (arr1 * 150).astype(np.uint8))
            # make sure the alpha band is all opaque
            np.testing.assert_allclose(data.sel(bands="A"), 255)

        # LA -> RGBA (float)
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(dataset2)
            img = img.convert("RGBA")
            assert img.mode == "RGBA"
            assert len(img.data.coords["bands"]) == 4

        # L -> palettize -> RGBA (float)
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(dataset1)
            img.palettize(brbg)
            pal = img.palette

            img2 = img.convert("RGBA")
            assert np.issubdtype(img2.data.dtype, np.floating)
            assert img2.mode == "RGBA"
            assert len(img2.data.coords["bands"]) == 4

        # PA -> RGB (float)
        img = xrimage.XRImage(dataset3)
        img.palette = pal
        with assert_maximum_dask_computes(0):
            img = img.convert("RGB")
            assert np.issubdtype(img.data.dtype, np.floating)
            assert img.mode == "RGB"
            assert len(img.data.coords["bands"]) == 3

        with pytest.raises(ValueError, match="Mode A not recognized"):
            img.convert("A")

        # L -> palettize -> RGBA (float) with RGBA colormap
        with assert_maximum_dask_computes(0):
            img = xrimage.XRImage(dataset1)
            img.palettize(bw)

            img2 = img.convert("RGBA")
            assert np.issubdtype(img2.data.dtype, np.floating)
            assert img2.mode == "RGBA"
            assert len(img2.data.coords["bands"]) == 4
            # convert to RGB, use RGBA from colormap regardless
            img2 = img.convert("RGB")
            assert np.issubdtype(img2.data.dtype, np.floating)
            assert img2.mode == "RGBA"
            assert len(img2.data.coords["bands"]) == 4

    def test_final_mode(self):
        """Test final_mode."""
        from trollimage import xrimage

        # numpy array image
        data = xr.DataArray(np.arange(75).reshape(5, 5, 3), dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})
        img = xrimage.XRImage(data)
        assert img.final_mode(None) == "RGBA"
        assert img.final_mode(0) == "RGB"

    def test_stack(self):
        """Test stack."""
        from trollimage import xrimage

        # background image
        arr1 = np.zeros((2, 2), dtype=np.float32)
        data1 = xr.DataArray(arr1, dims=["y", "x"])
        bkg = xrimage.XRImage(data1)

        # image to be stacked
        arr2 = np.full((2, 2), np.nan, dtype=np.float32)
        arr2[0] = 1
        data2 = xr.DataArray(arr2, dims=["y", "x"])
        img = xrimage.XRImage(data2)

        # expected result
        arr3 = arr1.copy()
        arr3[0] = 1.0
        data3 = xr.DataArray(arr3, dims=["y", "x"])
        res = xrimage.XRImage(data3)

        # stack image over the background
        bkg.stack(img)

        # check result
        np.testing.assert_allclose(bkg.data, res.data, rtol=1e-05)

    def test_merge(self):
        """Test merge."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, float])
    def test_blend(self, dtype):
        """Test blend."""
        from trollimage import xrimage

        core1 = np.arange(75, dtype=dtype).reshape(5, 5, 3) / 75.0
        alpha1 = np.linspace(0, 1, 25, dtype=dtype).reshape(5, 5, 1)
        arr1 = np.concatenate([core1, alpha1], 2)
        data1 = xr.DataArray(arr1, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B", "A"]})
        img1 = xrimage.XRImage(data1)

        core2 = np.arange(75, 0, -1, dtype=dtype).reshape(5, 5, 3) / 75.0
        alpha2 = np.linspace(1, 0, 25, dtype=dtype).reshape(5, 5, 1)
        arr2 = np.concatenate([core2, alpha2], 2)
        data2 = xr.DataArray(arr2, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B", "A"]})
        img2 = xrimage.XRImage(data2)
        img3 = img1.blend(img2)

        assert img3.data.dtype == dtype
        np.testing.assert_allclose((alpha1 + alpha2 * (1 - alpha1)).squeeze(), img3.data.sel(bands="A"))

        np.testing.assert_allclose(
            img3.data.sel(bands="R").values,
            np.array(
                [
                    [1.0, 0.95833635, 0.9136842, 0.8666667, 0.8180645],
                    [0.768815, 0.72, 0.6728228, 0.62857145, 0.5885714],
                    [0.55412847, 0.5264665, 0.50666666, 0.495612, 0.49394494],
                    [0.5020408, 0.52, 0.5476586, 0.5846154, 0.63027024],
                    [0.683871, 0.7445614, 0.81142855, 0.8835443, 0.96],
                ],
                dtype=dtype,
            ),
            rtol=2e-6,
        )

        with pytest.raises(TypeError):
            img1.blend("Salekhard")

        wrongimg = xrimage.XRImage(xr.DataArray(np.zeros((0, 0)), dims=("y", "x")))
        with pytest.raises(ValueError, match=r"Expected src\.mode='RGBA'"):
            img1.blend(wrongimg)

    def test_replace_luminance(self):
        """Test luminance replacement."""

    def test_putalpha(self):
        """Test putalpha."""

    def test_show(self):
        """Test that the show commands calls PIL.show."""
        from trollimage import xrimage

        data = xr.DataArray(
            np.arange(75).reshape(5, 5, 3) / 75.0, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]}
        )
        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage.PILImage.Image, "show", return_value=None) as s:
            img.show()
            s.assert_called_once()

    def test_apply_pil(self):
        """Test the apply_pil method."""
        from trollimage import xrimage

        np_data = np.arange(75).reshape(5, 5, 3) / 75.0
        data = xr.DataArray(np_data, dims=["y", "x", "bands"], coords={"bands": ["R", "G", "B"]})

        dummy_args = [(OrderedDict(),), {}]

        def dummy_fun(pil_obj, *args, **kwargs):
            dummy_args[0] = args
            dummy_args[1] = kwargs
            return pil_obj

        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage, "PILImage") as pi:
            pil_img = mock.MagicMock()
            pi.fromarray = mock.Mock(wraps=lambda *args, **kwargs: pil_img)
            res = img.apply_pil(dummy_fun, "RGB")
            # check that the pil image generation is delayed
            pi.fromarray.assert_not_called()
            # make it happen
            res.data.data.compute()
            pil_img.convert.assert_called_with("RGB")

        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage, "PILImage") as pi:
            pil_img = mock.MagicMock()
            pi.fromarray = mock.Mock(wraps=lambda *args, **kwargs: pil_img)
            res = img.apply_pil(dummy_fun, "RGB", fun_args=("Hey", "Jude"), fun_kwargs={"chorus": "La lala lalalala"})
            assert dummy_args == [({},), {}]
            res.data.data.compute()
            assert dummy_args == [(OrderedDict(), "Hey", "Jude"), {"chorus": "La lala lalalala"}]

        # Test HACK for _burn_overlay
        dummy_args = [(OrderedDict(),), {}]

        def _burn_overlay(pil_obj, *args, **kwargs):
            dummy_args[0] = args
            dummy_args[1] = kwargs
            return pil_obj

        img = xrimage.XRImage(data)
        with mock.patch.object(xrimage, "PILImage") as pi:
            pil_img = mock.MagicMock()
            pi.fromarray = mock.Mock(wraps=lambda *args, **kwargs: pil_img)
            res = img.apply_pil(_burn_overlay, "RGB")
            # check that the pil image generation is delayed
            pi.fromarray.assert_not_called()
            # make it happen
            res.data.data.compute()
            pil_img.convert.assert_called_with("RGB")


class TestXRImageColorize:
    """Test the colorize method of the XRImage class."""

    _expected: ClassVar[dict] = {
        np.float64: np.array(
            [
                [
                    [
                        3.29411737e-01,
                        3.57655096e-01,
                        3.86434124e-01,
                        4.15693619e-01,
                        4.45354613e-01,
                        4.75400874e-01,
                        5.05821379e-01,
                        5.36605942e-01,
                        5.65154991e-01,
                        5.92088509e-01,
                        6.19067983e-01,
                        6.46087257e-01,
                        6.73140335e-01,
                        7.00221370e-01,
                        7.27324655e-01,
                    ],
                    [
                        7.52329780e-01,
                        7.68885192e-01,
                        7.85480725e-01,
                        8.02165039e-01,
                        8.18991658e-01,
                        8.36019210e-01,
                        8.53311582e-01,
                        8.70937944e-01,
                        8.84215466e-01,
                        8.96340861e-01,
                        9.08470028e-01,
                        9.20615989e-01,
                        9.32792726e-01,
                        9.45015152e-01,
                        9.57299069e-01,
                    ],
                    [
                        9.64944680e-01,
                        9.65328486e-01,
                        9.65401069e-01,
                        9.65153988e-01,
                        9.64578376e-01,
                        9.63664900e-01,
                        9.62403720e-01,
                        9.60784235e-01,
                        9.36896731e-01,
                        9.12885905e-01,
                        8.88737447e-01,
                        8.64435259e-01,
                        8.39961154e-01,
                        8.15294485e-01,
                        7.90411692e-01,
                    ],
                    [
                        7.58448199e-01,
                        7.21741672e-01,
                        6.84822740e-01,
                        6.47626523e-01,
                        6.10070658e-01,
                        5.72048971e-01,
                        5.33422004e-01,
                        4.94570868e-01,
                        4.57464108e-01,
                        4.20002646e-01,
                        3.82018470e-01,
                        3.43266534e-01,
                        3.03372589e-01,
                        2.61727477e-01,
                        2.17242874e-01,
                    ],
                    [
                        1.89905775e-01,
                        1.67063045e-01,
                        1.43524430e-01,
                        1.18889134e-01,
                        9.24115382e-02,
                        6.24349277e-02,
                        2.53761544e-02,
                        4.08184216e-03,
                        4.27989281e-03,
                        4.17932136e-03,
                        3.78664262e-03,
                        3.12694131e-03,
                        2.24025474e-03,
                        1.17808547e-03,
                        4.27413532e-08,
                    ],
                ],
                [
                    [
                        1.88235338e-01,
                        2.05148716e-01,
                        2.22246545e-01,
                        2.39526080e-01,
                        2.56989499e-01,
                        2.74629834e-01,
                        2.92440006e-01,
                        3.10413434e-01,
                        3.32343826e-01,
                        3.57065431e-01,
                        3.82068290e-01,
                        4.07348972e-01,
                        4.32903771e-01,
                        4.58728828e-01,
                        4.84820214e-01,
                    ],
                    [
                        5.12920816e-01,
                        5.47946941e-01,
                        5.82732555e-01,
                        6.17314767e-01,
                        6.51719374e-01,
                        6.85963755e-01,
                        7.20058907e-01,
                        7.54010908e-01,
                        7.76938582e-01,
                        7.97119672e-01,
                        8.17286151e-01,
                        8.37436055e-01,
                        8.57567250e-01,
                        8.77677448e-01,
                        8.97764224e-01,
                    ],
                    [
                        9.12516448e-01,
                        9.19319188e-01,
                        9.26152806e-01,
                        9.33017318e-01,
                        9.39912732e-01,
                        9.46839047e-01,
                        9.53796254e-01,
                        9.60784383e-01,
                        9.55106487e-01,
                        9.49381255e-01,
                        9.43608692e-01,
                        9.37788802e-01,
                        9.31921582e-01,
                        9.26007026e-01,
                        9.20045124e-01,
                    ],
                    [
                        9.08501739e-01,
                        8.93232140e-01,
                        8.77927050e-01,
                        8.62584953e-01,
                        8.47204361e-01,
                        8.31783816e-01,
                        8.16321883e-01,
                        7.98071160e-01,
                        7.68921244e-01,
                        7.39943772e-01,
                        7.11141605e-01,
                        6.82517736e-01,
                        6.54075295e-01,
                        6.25817550e-01,
                        5.97747915e-01,
                    ],
                    [
                        5.70776460e-01,
                        5.44247790e-01,
                        5.17943022e-01,
                        4.91868010e-01,
                        4.66028951e-01,
                        4.40432416e-01,
                        4.15085387e-01,
                        3.90762614e-01,
                        3.67819668e-01,
                        3.45100725e-01,
                        3.22617410e-01,
                        3.00381898e-01,
                        2.78407005e-01,
                        2.56706279e-01,
                        2.35294121e-01,
                    ],
                ],
                [
                    [
                        1.96078107e-02,
                        2.42548730e-02,
                        2.74972914e-02,
                        2.96227826e-02,
                        3.17156346e-02,
                        3.38568632e-02,
                        3.60498856e-02,
                        3.82990518e-02,
                        5.17340258e-02,
                        7.13424642e-02,
                        9.00791521e-02,
                        1.08349534e-01,
                        1.26372972e-01,
                        1.44280400e-01,
                        1.62155446e-01,
                    ],
                    [
                        1.84723738e-01,
                        2.25766596e-01,
                        2.66872663e-01,
                        3.08395895e-01,
                        3.50522797e-01,
                        3.93349769e-01,
                        4.36919875e-01,
                        4.81242213e-01,
                        5.19495736e-01,
                        5.56210031e-01,
                        5.93054327e-01,
                        6.30051826e-01,
                        6.67218415e-01,
                        7.04564497e-01,
                        7.42096291e-01,
                    ],
                    [
                        7.75261226e-01,
                        8.01661519e-01,
                        8.28085451e-01,
                        8.54540356e-01,
                        8.81032565e-01,
                        9.07567554e-01,
                        9.34150068e-01,
                        9.60784367e-01,
                        9.52251703e-01,
                        9.43735611e-01,
                        9.35236188e-01,
                        9.26753529e-01,
                        9.18287731e-01,
                        9.09838895e-01,
                        9.01407118e-01,
                    ],
                    [
                        8.86846761e-01,
                        8.68087760e-01,
                        8.49200399e-01,
                        8.30188003e-01,
                        8.11054012e-01,
                        7.91801986e-01,
                        7.72435611e-01,
                        7.51368879e-01,
                        7.24059059e-01,
                        6.97016441e-01,
                        6.70243019e-01,
                        6.43740907e-01,
                        6.17512340e-01,
                        5.91559696e-01,
                        5.65885501e-01,
                    ],
                    [
                        5.39262097e-01,
                        5.12603472e-01,
                        4.86221761e-01,
                        4.60123407e-01,
                        4.34315308e-01,
                        4.08804870e-01,
                        3.83600068e-01,
                        3.58016760e-01,
                        3.31909014e-01,
                        3.06406099e-01,
                        2.81515767e-01,
                        2.57245707e-01,
                        2.33603643e-01,
                        2.10597450e-01,
                        1.88235292e-01,
                    ],
                ],
            ],
            dtype=np.float64,
        ),
        np.float32: np.array(
            [
                [
                    [
                        3.29411685e-01,
                        3.57655078e-01,
                        3.86434019e-01,
                        4.15693581e-01,
                        4.45354581e-01,
                        4.75400865e-01,
                        5.05821288e-01,
                        5.36605954e-01,
                        5.65154970e-01,
                        5.92088461e-01,
                        6.19067848e-01,
                        6.46087348e-01,
                        6.73140287e-01,
                        7.00221300e-01,
                        7.27324724e-01,
                    ],
                    [
                        7.52329946e-01,
                        7.68885195e-01,
                        7.85480618e-01,
                        8.02165091e-01,
                        8.18991542e-01,
                        8.36019218e-01,
                        8.53311539e-01,
                        8.70938063e-01,
                        8.84215295e-01,
                        8.96340668e-01,
                        9.08469796e-01,
                        9.20615852e-01,
                        9.32792485e-01,
                        9.45014775e-01,
                        9.57298815e-01,
                    ],
                    [
                        9.64944422e-01,
                        9.65328395e-01,
                        9.65400875e-01,
                        9.65153754e-01,
                        9.64578092e-01,
                        9.63664591e-01,
                        9.62403357e-01,
                        9.60784256e-01,
                        9.36896801e-01,
                        9.12885845e-01,
                        8.88737440e-01,
                        8.64435196e-01,
                        8.39960992e-01,
                        8.15294504e-01,
                        7.90411770e-01,
                    ],
                    [
                        7.58448243e-01,
                        7.21741796e-01,
                        6.84822679e-01,
                        6.47626460e-01,
                        6.10070348e-01,
                        5.72048545e-01,
                        5.33421874e-01,
                        4.94570613e-01,
                        4.57464218e-01,
                        4.20002341e-01,
                        3.82018358e-01,
                        3.43266338e-01,
                        3.03372920e-01,
                        2.61727750e-01,
                        2.17242956e-01,
                    ],
                    [
                        1.89906210e-01,
                        1.67063355e-01,
                        1.43524617e-01,
                        1.18889250e-01,
                        9.24117342e-02,
                        6.24350235e-02,
                        2.53762640e-02,
                        4.08192072e-03,
                        4.27983468e-03,
                        4.17933753e-03,
                        3.78649426e-03,
                        3.12698260e-03,
                        2.24010134e-03,
                        1.17787975e-03,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        1.88235313e-01,
                        2.05148667e-01,
                        2.22246468e-01,
                        2.39526033e-01,
                        2.56989419e-01,
                        2.74629742e-01,
                        2.92439938e-01,
                        3.10413390e-01,
                        3.32343757e-01,
                        3.57065320e-01,
                        3.82068306e-01,
                        4.07348871e-01,
                        4.32903767e-01,
                        4.58728850e-01,
                        4.84820247e-01,
                    ],
                    [
                        5.12920737e-01,
                        5.47946930e-01,
                        5.82732499e-01,
                        6.17314816e-01,
                        6.51719451e-01,
                        6.85963690e-01,
                        7.20058918e-01,
                        7.54010856e-01,
                        7.76938558e-01,
                        7.97119737e-01,
                        8.17286134e-01,
                        8.37435901e-01,
                        8.57567251e-01,
                        8.77677441e-01,
                        8.97764146e-01,
                    ],
                    [
                        9.12516296e-01,
                        9.19319153e-01,
                        9.26152706e-01,
                        9.33017135e-01,
                        9.39912558e-01,
                        9.46838915e-01,
                        9.53796089e-01,
                        9.60784137e-01,
                        9.55106318e-01,
                        9.49381173e-01,
                        9.43608582e-01,
                        9.37788725e-01,
                        9.31921482e-01,
                        9.26006973e-01,
                        9.20045018e-01,
                    ],
                    [
                        9.08501685e-01,
                        8.93232048e-01,
                        8.77927125e-01,
                        8.62584949e-01,
                        8.47204328e-01,
                        8.31783831e-01,
                        8.16321850e-01,
                        7.98071146e-01,
                        7.68921137e-01,
                        7.39943743e-01,
                        7.11141646e-01,
                        6.82517648e-01,
                        6.54075205e-01,
                        6.25817478e-01,
                        5.97747862e-01,
                    ],
                    [
                        5.70776403e-01,
                        5.44247806e-01,
                        5.17943025e-01,
                        4.91867840e-01,
                        4.66028810e-01,
                        4.40432310e-01,
                        4.15085286e-01,
                        3.90762508e-01,
                        3.67819637e-01,
                        3.45100671e-01,
                        3.22617382e-01,
                        3.00381780e-01,
                        2.78406948e-01,
                        2.56706238e-01,
                        2.35294104e-01,
                    ],
                ],
                [
                    [
                        1.96078010e-02,
                        2.42548790e-02,
                        2.74972897e-02,
                        2.96228528e-02,
                        3.17156874e-02,
                        3.38568650e-02,
                        3.60499322e-02,
                        3.82991321e-02,
                        5.17340526e-02,
                        7.13424012e-02,
                        9.00791660e-02,
                        1.08349539e-01,
                        1.26372933e-01,
                        1.44280463e-01,
                        1.62155449e-01,
                    ],
                    [
                        1.84723705e-01,
                        2.25766510e-01,
                        2.66872436e-01,
                        3.08395833e-01,
                        3.50522637e-01,
                        3.93349618e-01,
                        4.36919838e-01,
                        4.81242061e-01,
                        5.19495547e-01,
                        5.56209862e-01,
                        5.93054295e-01,
                        6.30051672e-01,
                        6.67218328e-01,
                        7.04564393e-01,
                        7.42096305e-01,
                    ],
                    [
                        7.75261164e-01,
                        8.01661491e-01,
                        8.28085482e-01,
                        8.54540348e-01,
                        8.81032467e-01,
                        9.07567501e-01,
                        9.34150100e-01,
                        9.60784256e-01,
                        9.52251494e-01,
                        9.43735421e-01,
                        9.35236037e-01,
                        9.26753461e-01,
                        9.18287575e-01,
                        9.09838736e-01,
                        9.01407063e-01,
                    ],
                    [
                        8.86846781e-01,
                        8.68087709e-01,
                        8.49200428e-01,
                        8.30188036e-01,
                        8.11053872e-01,
                        7.91801989e-01,
                        7.72435486e-01,
                        7.51368821e-01,
                        7.24058926e-01,
                        6.97016478e-01,
                        6.70242965e-01,
                        6.43740833e-01,
                        6.17512286e-01,
                        5.91559589e-01,
                        5.65885365e-01,
                    ],
                    [
                        5.39262116e-01,
                        5.12603402e-01,
                        4.86221790e-01,
                        4.60123241e-01,
                        4.34315115e-01,
                        4.08804744e-01,
                        3.83599907e-01,
                        3.58016640e-01,
                        3.31908882e-01,
                        3.06405991e-01,
                        2.81515718e-01,
                        2.57245511e-01,
                        2.33603537e-01,
                        2.10597396e-01,
                        1.88235223e-01,
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    }

    @pytest.mark.parametrize("colormap_tag", [None, "colormap"])
    def test_colorize_geotiff_tag(self, tmp_path, colormap_tag):
        """Test that a colorized colormap can be saved to a geotiff tag."""
        new_range = (0.0, 0.5)
        arr = np.arange(75).reshape(5, 15) / 74.0
        data = xr.DataArray(arr.copy(), dims=["y", "x"])
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
                loaded_brbg = Colormap.from_string(metadata["colormap"])
                np.testing.assert_allclose(new_brbg.values, loaded_brbg.values)
                np.testing.assert_allclose(new_brbg.colors, loaded_brbg.colors)

    @pytest.mark.parametrize(
        ("new_range", "input_scale", "input_offset", "expected_scale", "expected_offset", "dtype"),
        [
            ((0.0, 1.0), 1.0, 0.0, 1.0, 0.0, np.float32),
            ((0.0, 0.5), 1.0, 0.0, 2.0, 0.0, np.float32),
            ((2.0, 4.0), 2.0, 2.0, 0.5, -1.0, np.float32),
            ((0.0, 1.0), 1.0, 0.0, 1.0, 0.0, np.float64),
            ((0.0, 0.5), 1.0, 0.0, 2.0, 0.0, np.float64),
            ((2.0, 4.0), 2.0, 2.0, 0.5, -1.0, np.float64),
        ],
    )
    def test_colorize_l_rgb(self, new_range, input_scale, input_offset, expected_scale, expected_offset, dtype):
        """Test colorize with a RGB colormap."""
        img = self._get_input_image(dtype, input_scale, input_offset)
        new_brbg = brbg.set_range(*new_range, inplace=False)
        img.colorize(new_brbg)
        values = img.data.compute()
        assert values.dtype == dtype

        expected = self._get_expected_colorize_l_rgb(new_range, dtype)
        np.testing.assert_allclose(values, expected, atol=1e-6)
        assert "enhancement_history" in img.data.attrs
        assert img.data.attrs["enhancement_history"][-1]["scale"] == expected_scale
        assert img.data.attrs["enhancement_history"][-1]["offset"] == expected_offset
        assert isinstance(img.data.attrs["enhancement_history"][-1]["colormap"], Colormap)

    @staticmethod
    def _get_input_image(dtype, input_scale, input_offset):
        arr = np.arange(75, dtype=dtype).reshape(5, 15) / 74.0 * input_scale + input_offset
        data = xr.DataArray(arr, dims=["y", "x"])
        return xrimage.XRImage(data)

    def _get_expected_colorize_l_rgb(self, new_range, dtype):
        if new_range[1] == 0.5:
            expected2 = self._expected[dtype].copy().reshape((3, 75))
            flat_expected = self._expected[dtype].reshape((3, 75))
            expected2[:, :38] = flat_expected[:, ::2]
            expected2[:, 38:] = flat_expected[:, -1:]
            expected = expected2.reshape((3, 5, 15))
        else:
            expected = self._expected[dtype]
        return expected

    def test_colorize_int_l_rgb_with_fills(self):
        """Test integer data with _FillValue is masked (NaN) when colorized."""
        arr = np.arange(75, dtype=np.uint8).reshape(5, 15)
        arr[1, :] = 255
        data = xr.DataArray(arr.copy(), dims=["y", "x"], attrs={"_FillValue": 255})
        new_brbg = brbg.set_range(5, 20, inplace=False)
        img = xrimage.XRImage(data)
        img.colorize(new_brbg)
        values = img.data.compute()
        # Integer data inherits dtype from the colormap when colorized
        assert values.dtype == new_brbg.colors.dtype
        assert values.shape == (3,) + arr.shape  # RGB
        np.testing.assert_allclose(values[:, 1, :], np.nan)
        assert np.count_nonzero(np.isnan(values)) == arr.shape[1] * 3

        assert "enhancement_history" in img.data.attrs
        assert img.data.attrs["enhancement_history"][-1]["scale"] == 1 / (20 - 5)
        assert img.data.attrs["enhancement_history"][-1]["offset"] == -5 / (20 - 5)
        assert isinstance(img.data.attrs["enhancement_history"][-1]["colormap"], Colormap)

    def test_colorize_la_rgb(self):
        """Test colorizing an LA image with an RGB colormap."""
        arr = np.arange(75).reshape((5, 15)) / 74.0
        alpha = arr > 40.0
        data = xr.DataArray([arr.copy(), alpha], dims=["bands", "y", "x"], coords={"bands": ["L", "A"]})
        img = xrimage.XRImage(data)
        img.colorize(brbg)

        values = img.data.values
        expected = np.concatenate((self._expected[np.float64], alpha.reshape((1,) + alpha.shape)))
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

        arr = np.arange(75).reshape(5, 15) / 74.0
        data = xr.DataArray(arr.copy(), dims=["y", "x"])
        img = xrimage.XRImage(data)
        img.colorize(bw)
        values = img.data.compute()
        assert values.shape == (4, 5, 15)
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
        arr = np.arange(75).reshape(5, 15) / 74.0 * input_scale + input_offset
        data = xr.DataArray(arr.copy(), dims=["y", "x"])
        img = xrimage.XRImage(data)
        new_brbg = brbg.set_range(*new_range, inplace=False)
        img.palettize(new_brbg)

        values = img.data.values
        expected = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
                    [6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7],
                    [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10],
                ]
            ]
        )
        if new_range[1] == 0.5:
            flat_expected = expected.reshape((1, 75))
            expected2 = flat_expected.copy()
            expected2[:, :38] = flat_expected[:, ::2]
            expected2[:, 38:] = flat_expected[:, -1:]
            expected = expected2.reshape((1, 5, 15))
        assert np.issubdtype(values.dtype, np.integer)
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

        arr = np.arange(75).reshape(5, 15) / 74.0
        data = xr.DataArray(arr.copy(), dims=["y", "x"])
        img = xrimage.XRImage(data)
        img.palettize(bw)

        values = img.data.values
        assert values.shape == (1, 5, 15)
        assert bw.colors.shape == (2, 4)

    @pytest.mark.parametrize("colormap_tag", [None, "colormap"])
    @pytest.mark.parametrize("keep_palette", [False, True])
    def test_palettize_geotiff_tag(self, tmp_path, colormap_tag, keep_palette):
        """Test that a palettized image can be saved to a geotiff tag."""
        new_range = (0.0, 0.5)
        arr = np.arange(75).reshape(5, 15) / 74.0
        data = xr.DataArray(arr.copy(), dims=["y", "x"])
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
                loaded_brbg = Colormap.from_string(metadata["colormap"])
                np.testing.assert_allclose(new_brbg.values, loaded_brbg.values)
                np.testing.assert_allclose(new_brbg.colors, loaded_brbg.colors)

    def test_palettize_fill_value(self):
        """Test that fill values are adapted."""
        arr = np.arange(25, dtype="float32").reshape(5, 5) / 25
        arr[2, 2] = np.nan
        data = xr.DataArray(arr.copy(), dims=["y", "x"], attrs={"_FillValue": np.nan})
        img = xrimage.XRImage(data)
        img.palettize(brbg)
        assert img.data[0, 2, 2] == img.data.attrs["_FillValue"]

    def test_palettize_bad_fill_value(self):
        """Test that palettize warns with a strange fill value."""
        arr = np.arange(25, dtype="uint8").reshape(5, 5)
        data = xr.DataArray(arr.copy(), dims=["y", "x"], attrs={"_FillValue": 10})
        img = xrimage.XRImage(data)
        with pytest.warns(
            UserWarning,
            match="Palettizing uint8 data with the _FillValue attribute set to 10, "
            "but palettize is not generally fill value aware",
        ):
            img.palettize(brbg)


class TestXRImageSaveScaleOffset:
    """Test case for saving an image with scale and offset tags."""

    def setup_method(self) -> None:
        """Set up the test case."""
        from trollimage import xrimage

        data = xr.DataArray(
            np.arange(25, dtype=np.float32).reshape(5, 5, 1), dims=["y", "x", "bands"], coords={"bands": ["L"]}
        )
        self.img = xrimage.XRImage(data)
        rgb_data = xr.DataArray(
            np.arange(3 * 25, dtype=np.float32).reshape(5, 5, 3),
            dims=["y", "x", "bands"],
            coords={"bands": ["R", "G", "B"]},
        )
        self.rgb_img = xrimage.XRImage(rgb_data)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset(self):
        """Test saving geotiffs with tags."""
        expected_tags = {"scale": 24.0 / 255, "offset": 0}

        self.img.stretch()
        with pytest.warns(DeprecationWarning, match="include_scale_offset_tags is deprecated"):
            self._save_and_check_tags(expected_tags, include_scale_offset_tags=True)

    def test_gamma_geotiff_scale_offset(self, tmp_path):
        """Test that saving gamma-enhanced data to a geotiff with scale/offset tags doesn't fail."""
        self.img.gamma(0.5)
        out_fn = str(tmp_path / "test.tif")
        self.img.save(out_fn, scale_offset_tags=("scale", "offset"))
        with rio.open(out_fn, "r") as ds:
            assert np.isnan(float(ds.tags()["scale"]))
            assert np.isnan(float(ds.tags()["offset"]))

    def test_rgb_geotiff_scale_offset(self, tmp_path):
        """Test that saving RGB data to a geotiff with scale/offset tags doesn't fail."""
        self.rgb_img.stretch(stretch="crude", min_stretch=[-25, -40, 243], max_stretch=[0, 5, 208])
        out_fn = str(tmp_path / "test.tif")
        self.rgb_img.save(out_fn, scale_offset_tags=("scale", "offset"))
        with rio.open(out_fn, "r") as ds:
            assert np.isnan(float(ds.tags()["scale"]))
            assert np.isnan(float(ds.tags()["offset"]))

    def _save_and_check_tags(self, expected_tags, **kwargs):
        with NamedTemporaryFile(suffix=".tif") as tmp:
            self.img.save(tmp.name, **kwargs)

            import rasterio as rio

            with rio.open(tmp.name) as f:
                ftags = f.tags()
                for key, val in expected_tags.items():
                    np.testing.assert_almost_equal(float(ftags[key]), val)

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset_from_lists(self):
        """Test saving geotiffs with tags that come from lists."""
        expected_tags = {"scale": 23.0 / 255, "offset": 1}

        self.img.crude_stretch([1], [24])
        self._save_and_check_tags(expected_tags, scale_offset_tags=("scale", "offset"))

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset_custom_labels(self):
        """Test saving GeoTIFF with different scale/offset tag labels."""
        expected_tags = {"gradient": 24.0 / 255, "axis_intercept": 0}
        self.img.stretch()
        self._save_and_check_tags(expected_tags, scale_offset_tags=("gradient", "axis_intercept"))

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="'NamedTemporaryFile' not supported on Windows")
    def test_save_scale_offset_custom_values(self):
        """Test saving GeoTIFF overriding the scale/offset values."""
        expected_tags = {"gradient": 1, "axis_intercept": 0}
        self.img.stretch()
        self._save_and_check_tags(expected_tags, scale_offset_tags={"gradient": 1, "axis_intercept": 0})


def _get_tags_after_writing_to_geotiff(data):
    import rasterio as rio

    img = xrimage.XRImage(data)
    with NamedTemporaryFile(suffix=".tif") as tmp:
        img.save(tmp.name)
        with rio.open(tmp.name) as f:
            return f.tags()


@pytest.mark.parametrize("attrs", [{}, {"mode": "XYZ"}])
def test_missing_bands_coord(attrs):
    """Test that 'bands' dimenisons need a corresponding coordinate."""
    data = xr.DataArray(
        da.zeros((3, 10, 5), dtype=np.float32),
        dims=("bands", "y", "x"),
        attrs=attrs,
    )
    with pytest.warns(UserWarning, match="Missing 'bands' coordinate.*"):
        img = xrimage.XRImage(data)
    exp_bands = ["R", "G", "B"] if not attrs else ["X", "Y", "Z"]
    np.testing.assert_array_equal(img.data.coords["bands"], exp_bands)


@pytest.mark.parametrize("fill_value", [None, 255])
def test_pil_array(fill_value):
    """Test 'pil_array' method."""
    data = xr.DataArray(
        da.zeros((10, 5), dtype=np.float32, chunks=2),
        dims=("y", "x"),
    )
    img = xrimage.XRImage(data)
    pil_arr, mode = img.pil_array(fill_value)
    assert isinstance(pil_arr, da.Array)
    assert mode == ("L" if fill_value is not None else "LA")
    np_arr = pil_arr.compute()
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.dtype == pil_arr.dtype
