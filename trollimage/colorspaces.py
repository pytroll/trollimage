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

"""Color spaces using numpy to run on arrays.
"""

import numpy as np

def lab2xyz(l, a, b):
    """Convert L*a*b* to XYZ, L*a*b* expressed within [0, 1].
    """


    def f_inv(arr):
        """Inverse of the f function.
        """
        return np.where(arr > 6.0/29.0,
                        arr ** 3,
                        3 * (6.0 / 29.0) * (6.0 / 29.0) * (arr - 4.0 / 29.0))

    new_l = (l + 16.0) / 116.0
    x = 95.047 * f_inv(new_l + a / 500.0)
    y = 100.0 * f_inv(new_l)
    z = 108.883 * f_inv(new_l - b / 200.0)

    return x, y, z

def xyz2lab(x, y, z):
    """Convert XYZ to L*a*b*.
    """

    def f(arr):
        """f function
        """

        return np.where(arr > 216.0 / 24389.0,
                        arr ** (1.0/3.0),
                        (1.0 / 3.0) * (29.0 / 6.0) * (29.0 / 6.0) * arr
                        + 4.0 / 29.0)
    fy = f(y / 100.0)
    l = 116 * fy - 16
    a = 500.0 * (f(x / 95.047) - fy)
    b = 200.0 * (fy - f(z / 108.883))
    return l, a, b


def hcl2lab(h, c, l):
    h_rad = np.deg2rad(h)
    l_ = l * 61 + 9
    angle = np.pi / 3.0 - h_rad
    r = (l * 311 + 125) * c
    a = np.sin(angle) * r
    b = np.cos(angle) * r
    return l_, a, b

def lab2hcl(l, a, b):

    l_ = (l - 9) / 61.0
    r = np.sqrt(a*a + b*b)
    s = r / (l_ * 311 + 125)
    angle = np.arctan2(a, b)
    c = np.rad2deg(np.pi / 3 - angle)%360
    return c, s, l_

def rgb2xyz(r, g, b):

    r__ = ( r / 255.0 )        
    g__ = ( g / 255.0 )        
    b__ = ( b / 255.0 )        

    def f(arr):
        return np.where(arr > 0.04045,
                        ((arr + 0.055) / 1.055) ** 2.4,
                        arr / 12.92)
     

    r__ = f(r__) * 100
    g__ = f(g__) * 100
    b__ = f(b__) * 100

    x__ = r__ * 0.4124 + g__ * 0.3576 + b__ * 0.1805
    y__ = r__ * 0.2126 + g__ * 0.7152 + b__ * 0.0722
    z__ = r__ * 0.0193 + g__ * 0.1192 + b__ * 0.9505

    return x__, y__, z__

def xyz2rgb(x, y, z):
    x__ = x / 100.0
    y__ = y / 100.0
    z__ = z / 100.0

    r__ = x__ *  3.2406 + y__ * -1.5372 + z__ * -0.4986
    g__ = x__ * -0.9689 + y__ *  1.8758 + z__ *  0.0415
    b__ = x__ *  0.0557 + y__ * -0.2040 + z__ *  1.0570

    def finv(arr):
        return np.where(arr > 0.0031308,
                        1.055 * (arr ** (1.0 / 2.4)) - 0.055,
                        12.92 * arr)

    R = finv(r__) * 255
    G = finv(g__) * 255
    B = finv(b__) * 255

    return R, G, B

def rgb2hcl(r, g, b):
    return lab2hcl(*xyz2lab(*rgb2xyz(r, g, b)))

def hcl2rgb(h, c, l):
    return xyz2rgb(*lab2xyz(*hcl2lab(h, c, l)))

def rgb2lab(r, g, b):
    return xyz2lab(*rgb2xyz(r, g, b))

def lab2rgb(h, c, l):
    return xyz2rgb(*lab2xyz(h, c, l))
