# Copyright (c) 2022 trollimage developers
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
"""RasterIO-specific utilities needed by the XRImage class."""

import logging
import threading
from contextlib import suppress

import dask.array as da

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def get_data_arr_crs_transform_gcps(data_arr):
    """Convert DataArray's AreaDefinition or SwathDefinition to rasterio geolocation information.

    If possible, a rasterio geotransform object will be created. If it can't be made
    then it is assumed the provided geometry object is a SwathDefinition and will be
    checked for GCP coordinates (``swath_def.lons.attrs['gcps']``).

    Args:
        data_arr: Xarray DataArray.

    Returns:
        Tuple of (crs, transform, gcps). Each element defaults to ``None`` if
        it couldn't be calculated.

    """
    crs = None
    transform = None
    gcps = None

    try:
        area = data_arr.attrs["area"]
        if rasterio.__gdal_version__ >= '3':
            wkt_version = 'WKT2_2018'
        else:
            wkt_version = 'WKT1_GDAL'
        if hasattr(area, 'crs'):
            crs = rasterio.crs.CRS.from_wkt(area.crs.to_wkt(version=wkt_version))
        else:
            crs = rasterio.crs.CRS(area.proj_dict)
        west, south, east, north = area.area_extent
        height, width = area.shape
        transform = rasterio.transform.from_bounds(west, south,
                                                   east, north,
                                                   width, height)

    except KeyError:  # No area
        logger.info("Couldn't create geotransform")
    except AttributeError:
        try:
            gcps = data_arr.attrs["area"].lons.attrs['gcps']
            crs = data_arr.attrs["area"].lons.attrs['crs']
        except KeyError:
            logger.info("Couldn't create geotransform")
    return crs, transform, gcps


def split_regular_vs_lazy_tags(tags, r_file):
    """Split tags into regular vs lazy (dask) tags."""
    da_tags = []
    for key, val in list(tags.items()):
        try:
            if isinstance(val.data, da.Array):
                da_tags.append((val.data, RIOTag(r_file, key)))
                tags.pop(key)
            else:
                tags[key] = val.item()
        except AttributeError:
            continue
    return tags, da_tags


class RIOFile(object):
    """Rasterio wrapper to allow da.store to do window saving."""

    def __init__(self, path, mode='w', **kwargs):
        """Initialize the object."""
        self.path = path
        self.mode = mode
        self.kwargs = kwargs
        self.rfile = None
        self.lock = threading.Lock()

    @property
    def width(self):
        """Width of the band images."""
        return self.kwargs['width']

    @property
    def height(self):
        """Height of the band images."""
        return self.kwargs['height']

    @property
    def closed(self):
        """Check if the file is closed."""
        return self.rfile is None or self.rfile.closed

    def open(self, mode=None):
        """Open the file."""
        mode = mode or self.mode
        if self.closed:
            self.rfile = rasterio.open(self.path, mode, **self.kwargs)

    def close(self):
        """Close the file."""
        with self.lock:
            if not self.closed:
                self.rfile.close()

    def __enter__(self):
        """Enter method."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        self.close()

    def __del__(self):
        """Delete the instance."""
        with suppress(IOError, OSError):
            self.close()

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

    def write(self, *args, **kwargs):
        """Write to the file."""
        with self.lock:
            self.open('r+')
            return self.rfile.write(*args, **kwargs)

    def build_overviews(self, *args, **kwargs):
        """Write overviews."""
        with self.lock:
            self.open('r+')
            return self.rfile.build_overviews(*args, **kwargs)

    def update_tags(self, *args, **kwargs):
        """Update tags."""
        with self.lock:
            self.open('a')
            return self.rfile.update_tags(*args, **kwargs)


class RIOTag:
    """Rasterio wrapper to allow da.store on tag."""

    def __init__(self, rfile, name):
        """Init the rasterio tag."""
        self.rfile = rfile
        self.name = name

    def __setitem__(self, key, item):
        """Put the data in the tag."""
        kwargs = {self.name: item.item()}
        self.rfile.update_tags(**kwargs)

    def close(self):
        """Close the file."""
        return self.rfile.close()


class RIODataset:
    """A wrapper for a rasterio dataset."""

    def __init__(self, rfile, overviews=None, overviews_resampling=None,
                 overviews_minsize=256):
        """Init the rasterio dataset."""
        self.rfile = rfile
        self.overviews = overviews
        if overviews_resampling is None:
            overviews_resampling = 'nearest'
        self.overviews_resampling = Resampling[overviews_resampling]
        self.overviews_minsize = overviews_minsize

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

    def close(self):
        """Close the file."""
        if self.overviews is not None:
            overviews = self.overviews
            # it's an empty list
            if len(overviews) == 0:
                from rasterio.rio.overview import get_maximum_overview_level
                width = self.rfile.width
                height = self.rfile.height
                max_level = get_maximum_overview_level(
                    width, height, self.overviews_minsize)
                overviews = [2 ** j for j in range(1, max_level + 1)]
            logger.debug('Building overviews %s with %s resampling',
                         str(overviews), self.overviews_resampling.name)
            self.rfile.build_overviews(overviews, resampling=self.overviews_resampling)

        return self.rfile.close()


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
