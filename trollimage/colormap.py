#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012, 2013 Martin Raspaud

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

"""Reimplement colorsys with XYZ as base, and using numpy to run on arrays.

A colorize function is also provided.
"""

import numpy as np
from trollimage.colorspaces import rgb2hcl, hcl2rgb

def colorize(arr, colors, values):
    """Colorize a monochromatic array *arr*, based *colors* given for
    *values*. Interpolation is used. *values* must be in ascending order.
    """
    hcolors = np.array([rgb2hcl(*i) for i in colors])
    # unwrap colormap in hcl space
    hcolors[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(np.array(hcolors)[:, 0])))
    channels = [np.interp(arr,
                          np.array(values),
                          np.array(hcolors)[:, i])
                for i in range(3)]

    channels = hcl2rgb(*channels)

    try:
        return [np.ma.array(channel, mask=arr.mask) for channel in channels]
    except:
        return channels


class Colormap(object):
    """The colormap object.

    Initialize with tuples of (value, (colors)), like this::
    
      Colormap((-75.0, (1.0, 1.0, 0.0)),
               (-40.0001, (0.0, 1.0, 1.0)),
               (-40.0, (1, 1, 1)),
               (30.0, (0, 0, 0)))


    You can also concatenate colormaps together, try::

      cm = cm1 + cm2

    """

    def __init__(self, *tuples, **kwargs):
        values = [a for (a, b) in tuples]
        colors = [b for (a, b) in tuples]
        self.values = np.array(values)
        self.colors = np.array(colors)

        self.interpolation = kwargs.get("interpolation", None)
        
    def colorize(self, data):
        """Colorize a monochromatic array *data*, based on the current colormap.
        """
        return colorize(data,
                        self.colors,
                        self.values)

    def __add__(self, other):
        new = Colormap()
        new.values = np.concatenate((self.values, other.values))
        new.colors = np.concatenate((self.colors, other.colors))
        return new

    def reverse(self):
        """Reverse the current colormap in place.
        """
        self.values = np.flipud(self.values)

if __name__ == '__main__':

    # unit tests...
    # print lab2xyz(50, 50, 50)
    # print xyz2lab(*lab2xyz(50, 50, 50))

    # print lab2hcl(50, 50, 50)
    # print hcl2lab(*lab2hcl(50, 50, 50))
    
    # print rgb2xyz(50, 50, 50)
    # print xyz2rgb(*rgb2xyz(50, 50, 50))

    from mpop.satellites import GeostationaryFactory
    from mpop.imageo.geo_image import GeoImage
    from datetime import datetime
    t = datetime(2009, 10, 8, 14, 30)
    g = GeostationaryFactory.create_scene("meteosat", "09", "seviri", t)
    g.load([10.8])
    #l = g.project("SouthAmerica_flat")
    g.area = g[10.8].area
    l = g
    # ch = colorize(l[10.8].data,
    #               np.array(((0.0, 0.0, 0.0),
    #                         (255.0, 0.0, 0.0),
    #                         (255, 255, 0),
    #                         (0, 0, 255.0),
    #                         (255, 255, 255.0),
    #                         (0.0, 0.0, 0.0))),
    #               np.array((-75.0, -70.0, -60.0, -40.001, -40.0, 30.0)) + 273.15)
    # ch = colorize(l[10.8].data,
    #               np.array(((255.0, 255.0, 0.0),
    #                         (0.0, 255.0, 255.0),
    #                         (255, 255, 255.0),
    #                         (0.0, 0.0, 0.0))),
    #               np.array((-75.0, -40.001, -40.0, 30.0)) + 273.15)


    colormap = Colormap((-75.0 + 273.15, (255.0, 255.0, 0.0)),
                        (-40.0001 + 273.15, (0.0, 255.0, 255.0)),
                        (-40.0 + 273.15, (255, 255, 255)),
                        (30.0 + 273.15, (0, 0, 0)))

    ch = colormap.colorize(l[10.8].data)

    ch[0][l[10.8].data.mask] = 0.0
    img = GeoImage(ch, area=l.area, time_slot=l.time_slot, mode="RGB", crange=[(0, 255), (0, 255), (0,255)], fill_value=None)
    img.add_overlay(color=(240, 185, 19), width=1.0)
    img.show()
