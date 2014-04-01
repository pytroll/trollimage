#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, 2014 Martin Raspaud

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

def lab2xyz(l__, a__, b__):
    """Convert L*a*b* to XYZ, L*a*b* expressed within [0, 1].
    """


    def f_inv(arr):
        """Inverse of the f function.
        """
        return np.where(arr > 6.0/29.0,
                        arr ** 3,
                        3 * (6.0 / 29.0) * (6.0 / 29.0) * (arr - 4.0 / 29.0))

    new_l = (l__ + 16.0) / 116.0
    x__ = 95.047 * f_inv(new_l + a__ / 500.0)
    y__ = 100.0 * f_inv(new_l)
    z__ = 108.883 * f_inv(new_l - b__ / 200.0)

    return x__, y__, z__

def xyz2lab(x__, y__, z__):
    """Convert XYZ to L*a*b*.
    """

    def f__(arr):
        """f__ function
        """

        return np.where(arr > 216.0 / 24389.0,
                        arr ** (1.0/3.0),
                        (1.0 / 3.0) * (29.0 / 6.0) * (29.0 / 6.0) * arr
                        + 4.0 / 29.0)
    fy_ = f__(y__ / 100.0)
    l__ = 116 * fy_ - 16
    a__ = 500.0 * (f__(x__ / 95.047) - fy_)
    b__ = 200.0 * (fy_ - f__(z__ / 108.883))
    return l__, a__, b__


def hcl2lab(h__, c__, l__):
    """HCL to L*ab
    """
    h_rad = np.deg2rad(h__)
    l2_ = l__ * 61 + 9
    angle = np.pi / 3.0 - h_rad
    r__ = (l__ * 311 + 125) * c__
    a__ = np.sin(angle) * r__
    b__ = np.cos(angle) * r__
    return l2_, a__, b__

def lab2hcl(l__, a__, b__):
    """L*a*b* to HCL
    """
    l2_ = (l__ - 9) / 61.0
    r__ = np.sqrt(a__*a__ + b__*b__)
    s__ = r__ / (l2_ * 311 + 125)
    angle = np.arctan2(a__, b__)
    c__ = np.rad2deg(np.pi / 3 - angle)%360
    return c__, s__, l2_

def rgb2xyz(r__, g__, b__):
    """RGB to XYZ
    """

    r2_ = r__ / 255.0
    g2_ = g__ / 255.0
    b2_ = b__ / 255.0

    def f__(arr):
        """Forward
        """
        return np.where(arr > 0.04045,
                        ((arr + 0.055) / 1.055) ** 2.4,
                        arr / 12.92)


    r2_ = f__(r2_) * 100
    g2_ = f__(g2_) * 100
    b2_ = f__(b2_) * 100

    x__ = r2_ * 0.4124 + g2_ * 0.3576 + b2_ * 0.1805
    y__ = r2_ * 0.2126 + g2_ * 0.7152 + b2_ * 0.0722
    z__ = r2_ * 0.0193 + g2_ * 0.1192 + b2_ * 0.9505

    return x__, y__, z__

def xyz2rgb(x__, y__, z__):
    """XYZ colorspace to RGB
    """
    x2_ = x__ / 100.0
    y2_ = y__ / 100.0
    z2_ = z__ / 100.0

    r__ = x2_ *  3.2406 + y2_ * -1.5372 + z2_ * -0.4986
    g__ = x2_ * -0.9689 + y2_ *  1.8758 + z2_ *  0.0415
    b__ = x2_ *  0.0557 + y2_ * -0.2040 + z2_ *  1.0570

    def finv(arr):
        """Inverse
        """
        return np.where(arr > 0.0031308,
                        1.055 * (arr ** (1.0 / 2.4)) - 0.055,
                        12.92 * arr)

    return finv(r__) * 255, finv(g__) * 255, finv(b__) * 255

def rgb2hcl(r__, g__, b__):
    """RGB to HCL
    """
    return lab2hcl(*rgb2lab(r__, g__, b__))

def hcl2rgb(h__, c__, l__):
    """HCL to RGB
    """
    return lab2rgb(*hcl2lab(h__, c__, l__))

def rgb2lab(r__, g__, b__):
    """RGB to L*ab
    """
    return xyz2lab(*rgb2xyz(r__, g__, b__))

def lab2rgb(h__, c__, l__):
    """L*ab to RGB
    """
    return xyz2rgb(*lab2xyz(h__, c__, l__))
