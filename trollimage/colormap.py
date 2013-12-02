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

# matlab jet    "#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"

rainbow = Colormap((0.000, (0.0, 0.0, 0.5)),
                   (0.125, (0.0, 0.0, 1.0)),
                   (0.250, (0.0, 0.5, 1.0)),
                   (0.375, (0.0, 1.0, 1.0)),
                   (0.500, (0.5, 1.0, 0.5)),
                   (0.625, (1.0, 1.0, 0.0)),
                   (0.750, (1.0, 0.5, 0.0)),
                   (0.875, (1.0, 0.0, 0.0)),
                   (1.000, (0.5, 0.0, 0.0)))

### Colors from www.ColorBrewer.org by Cynthia A. Brewer, Geography, Pennsylvania State University.

## Single hue

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


## Multihue

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


## Diverging

brbg = Colormap((0.0, (84 / 255.0, 48 / 255.0, 5 / 255.0)),
                (0.1, (140 / 255.0, 81 / 255.0, 10 / 255.0)),
                (0.2, (191 / 255.0, 129 / 255.0, 45 / 255.0)),
                (0.3, (223 / 255.0, 129 / 255.0, 125 / 255.0)),
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


diverging_colormaps = [brbg, piyg, prgn, puor, rdbu, rdgy, rdylbu, rdylgn, spectral]

if __name__ == '__main__':


    from trollimage.image import Image

    cm = Colormap((0, (1.0, 1.0, 0.0)),
                  (0.3, (0.0, 1.0, 1.0)),
                  (0.6, (1, 1, 1)),
                  (1, (0, 0, 0)))

    cm = rainbow + rainbow
    length = len(cm.values)
    cm.values = np.arange(length) * 1.0 / length

    img = Image(colorbar(25, 500, rainbow), mode="RGB")
    img.show()

    # # unit tests...
    # # print lab2xyz(50, 50, 50)
    # # print xyz2lab(*lab2xyz(50, 50, 50))

    # # print lab2hcl(50, 50, 50)
    # # print hcl2lab(*lab2hcl(50, 50, 50))
    
    # # print rgb2xyz(50, 50, 50)
    # # print xyz2rgb(*rgb2xyz(50, 50, 50))

    # from mpop.satellites import GeostationaryFactory
    # from mpop.imageo.geo_image import GeoImage
    # from datetime import datetime
    # t = datetime(2009, 10, 8, 14, 30)
    # g = GeostationaryFactory.create_scene("meteosat", "09", "seviri", t)
    # g.load([10.8])
    # #l = g.project("SouthAmerica_flat")
    # g.area = g[10.8].area
    # l = g
    # # ch = colorize(l[10.8].data,
    # #               np.array(((0.0, 0.0, 0.0),
    # #                         (255.0, 0.0, 0.0),
    # #                         (255, 255, 0),
    # #                         (0, 0, 255.0),
    # #                         (255, 255, 255.0),
    # #                         (0.0, 0.0, 0.0))),
    # #               np.array((-75.0, -70.0, -60.0, -40.001, -40.0, 30.0)) + 273.15)
    # # ch = colorize(l[10.8].data,
    # #               np.array(((255.0, 255.0, 0.0),
    # #                         (0.0, 255.0, 255.0),
    # #                         (255, 255, 255.0),
    # #                         (0.0, 0.0, 0.0))),
    # #               np.array((-75.0, -40.001, -40.0, 30.0)) + 273.15)


    # colormap = Colormap((-75.0 + 273.15, (255.0, 255.0, 0.0)),
    #                     (-40.0001 + 273.15, (0.0, 255.0, 255.0)),
    #                     (-40.0 + 273.15, (255, 255, 255)),
    #                     (30.0 + 273.15, (0, 0, 0)))

    # ch = colormap.colorize(l[10.8].data)

    # ch[0][l[10.8].data.mask] = 0.0
    # img = GeoImage(ch, area=l.area, time_slot=l.time_slot, mode="RGB", crange=[(0, 255), (0, 255), (0,255)], fill_value=None)
    # img.add_overlay(color=(240, 185, 19), width=1.0)
    # img.show()
