# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False, cpow=True
cimport cython

from libc.math cimport cos, sin, atan2
import numpy as np
cimport numpy as np

ctypedef fused floating:
    np.float32_t
    np.float64_t


np.import_array()

# cdef extern from "numpy/npy_math.h":
#     np.float32_t NPY_NANF
#     bint npy_isnan(np.float64_t x) nogil
#     bint npy_isnan(np.float32_t x) nogil


def rgb2lch(
        object rgba_arr,
        # bint allow_nans = True,
):
    """Convert numpy RGB[A] arrays to CIE LCh_ab (lch).

    Args:
        rgba_arr: Numpy array of RGB or RGBA colors. The array can be any
            shape as long as the channel (band) dimension is the last (-1)
            dimension. If an Alpha (A) channel is provided it is ignored.
            Values should be between 0 and 1.
        gamma: Correction factor (see related paper). In normal use this does
            not need to be changed from the default of 3.0.
        y_0: White reference luminance. In normal use this does not need to
            be changed from the default of 100.0.
        allow_nans: Boolean flag to specify that NaNs can be returned in
            operations that would produce it. This is most useful when
            grayscale colors (Chroma=0) are converted where Hue (H) has no
            effect. This allows for more control when interpolating in the HCL
            color space. Defaults to True.

    Returns: LCH_ab (l, c, h) numpy array where the last dimension represents Hue, Chroma,
        and Luminance. Hue is in radians from -pi to pi. Chroma is from 0 to
        1. Luminance is also from 0 and 1 (usually a maximum of ~0.5).

    """
    cdef object rgb_arr = rgba_arr[..., :3]
    cdef tuple shape = rgb_arr.shape
    cdef np.ndarray rgb_2d = rgb_arr.reshape((-1, 3))
    cdef np.ndarray lch
    if rgb_arr.dtype == np.float32:
        lch = _call_rgb_to_lch(<np.ndarray[np.float32_t, ndim=2]> rgb_2d)
    else:
        lch = _call_rgb_to_lch(<np.ndarray[np.float64_t, ndim=2]> rgb_2d)
    return lch.reshape(shape)


cdef np.ndarray[floating, ndim=2] _call_rgb_to_lch(
        np.ndarray[floating, ndim=2] rgb,
        # bint allow_nans = True,
):
    cdef np.ndarray[floating, ndim=1] red = rgb[:, 0]
    cdef np.ndarray[floating, ndim=1] green = rgb[:, 1]
    cdef np.ndarray[floating, ndim=1] blue = rgb[:, 2]
    cdef floating[:] red_view, green_view, blue_view
    red_view = red
    green_view = green
    blue_view = blue
    cdef np.ndarray[floating, ndim=2] lch = np.empty((red_view.shape[0], 3), dtype=rgb.dtype)
    cdef floating[:, ::1] lch_view = lch
    with nogil:
        _rgb_to_lch(red_view, green_view, blue_view, lch_view)
    return lch


def lch2rgb(
        object lch_arr,
        # float gamma = 3.0,
        # float y_0 = 100.0
):
    """Convert an HCL (hue, chroma, luminance) array to RGB.

    Args:
        lch_arr: Numpy array of HCL values. The array can be any
            shape as long as the channel (band) dimension is the last (-1)
            dimension. Hue must be between -pi to pi. Chroma and Luminance
            should be between 0 and 1.
        gamma: Correction factor (see related paper). In normal use this does
            not need to be changed from the default of 3.0.
        y_0: White reference luminance. In normal use this does not need to
            be changed from the default of 100.0.

    Returns: RGB array where each Red, Green, and Blue channel is between 0 and 1.

    """
    cdef tuple shape = lch_arr.shape
    cdef np.ndarray lch_2d = lch_arr.reshape((-1, 3))
    cdef np.ndarray rgb
    if lch_arr.dtype == np.float32:
        rgb = _call_lch_to_rgb(<np.ndarray[np.float32_t, ndim=2]> lch_2d)
    else:
        rgb = _call_lch_to_rgb(<np.ndarray[np.float64_t, ndim=2]> lch_2d)
    return rgb.reshape(shape)


cdef np.ndarray[floating, ndim=2] _call_lch_to_rgb(
        np.ndarray[floating, ndim=2] lch,
):
    cdef np.ndarray[floating, ndim=1] luminance = lch[:, 0]
    cdef np.ndarray[floating, ndim=1] chroma = lch[:, 1]
    cdef np.ndarray[floating, ndim=1] hue = lch[:, 2]  # in radians
    cdef floating[:] hue_view, chroma_view, luminance_view
    hue_view = hue
    chroma_view = chroma
    luminance_view = luminance
    cdef np.ndarray[floating, ndim=2] rgb = np.empty((hue_view.shape[0], 3), dtype=lch.dtype)
    cdef floating[:, ::1] rgb_view = rgb
    with nogil:
        _lch_to_rgb(luminance_view, chroma_view, hue_view, rgb_view)
    return rgb


# cdef void _rgb_to_lab(floating[:] r_arr, floating[:] g_arr, floating[:] b_arr, floating[:, ::1] lab_arr) nogil:
#     cdef floating[:] l, a, b
#     _rgb_to_xyz(r_arr, g_arr, b_arr, lab_arr)
#     l = lab_arr[:, 0]
#     a = lab_arr[:, 1]
#     b = lab_arr[:, 2]
#     _xyz_to_lab(l, a, b, lab_arr)


cdef void _rgb_to_lch(floating[:] r_arr, floating[:] g_arr, floating[:] b_arr, floating[:, ::1] lch_arr) nogil:
    cdef floating[:] l, c, h
    _rgb_to_xyz(r_arr, g_arr, b_arr, lch_arr)
    l, c, h = lch_arr[:, 0], lch_arr[:, 1], lch_arr[:, 2]
    _xyz_to_lab(l, c, h, lch_arr)
    _lab_to_lch(l, c, h, lch_arr)
    # _xyz_to_lab(lch_arr[:, 0], lch_arr[:, 1], lch_arr[:, 2], lch_arr)
    # _lab_to_lch(lch_arr[:, 0], lch_arr[:, 1], lch_arr[:, 2], lch_arr)


# cdef void _rgb_to_luv(floating[:] r_arr, floating[:] g_arr, floating[:] b_arr, floating[:, ::1] luv_arr) nogil:
#     _rgb_to_xyz(r_arr, g_arr, b_arr, luv_arr)
#     _xyz_to_luv(luv_arr[:, 0], luv_arr[:, 1], luv_arr[:, 2], luv_arr)
#
#
# cdef void _xyz_to_lch(floating[:] x_arr, floating[:] y_arr, floating[:] z_arr, floating[:, ::1] lch_arr) nogil:
#     _xyz_to_lab(x_arr, y_arr, z_arr, lch_arr)
#     _lab_to_lch(lch_arr[:, 0], lch_arr[:, 1], lch_arr[:, 2], lch_arr)
#
#
# cdef void _lab_to_rgb(floating[:] l_arr, floating[:] a_arr, floating[:] b_arr, floating[:, ::1] rgb_arr) nogil:
#     _lab_to_xyz(l_arr, a_arr, b_arr, rgb_arr)
#     _xyz_to_rgb(rgb_arr[:, 0], rgb_arr[:, 1], rgb_arr[:, 2], rgb_arr)
#
#
# cdef void _lab_to_luv(floating[:] l_arr, floating[:] a_arr, floating[:] b_arr, floating[:, ::1] luv_arr) nogil:
#     _lab_to_xyz(l_arr, a_arr, b_arr, luv_arr)
#     _xyz_to_luv(luv_arr[:, 0], luv_arr[:, 1], luv_arr[:, 2], luv_arr)
#
#
# cdef void _lch_to_xyz(floating[:] l_arr, floating[:] c_arr, floating[:] h_arr, floating[:, ::1] xyz_arr) nogil:
#     _lch_to_lab(l_arr, c_arr, h_arr, xyz_arr)
#     _lab_to_xyz(xyz_arr[:, 0], xyz_arr[:, 1], xyz_arr[:, 2], xyz_arr)
#
#
cdef void _lch_to_rgb(floating[:] l_arr, floating[:] c_arr, floating[:] h_arr, floating[:, ::1] rgb_arr) nogil:
    cdef floating[:] r, g, b
    _lch_to_lab(l_arr, c_arr, h_arr, rgb_arr)
    r = rgb_arr[:, 0]
    g = rgb_arr[:, 1]
    b = rgb_arr[:, 2]
    _lab_to_xyz(r, g, b, rgb_arr)
    _xyz_to_rgb(r, g, b, rgb_arr)


# cdef void _lch_to_luv(floating[:] l_arr, floating[:] c_arr, floating[:] h_arr, floating[:, ::1] luv_arr) nogil:
#     _lch_to_lab(l_arr, c_arr, h_arr, luv_arr)
#     _lab_to_xyz(luv_arr[:, 0], luv_arr[:, 1], luv_arr[:, 2], luv_arr)
#     _xyz_to_rgb(luv_arr[:, 0], luv_arr[:, 1], luv_arr[:, 2], luv_arr)
#
#
# cdef void _luv_to_lab(floating[:] l_arr, floating[:] u_arr, floating[:] v_arr, floating[:, ::1] lab_arr) nogil:
#     _luv_to_xyz(l_arr, u_arr, v_arr, lab_arr)
#     _xyz_to_lab(lab_arr[:, 0], lab_arr[:, 1], lab_arr[:, 2], lab_arr)
#
#
# cdef void _luv_to_rgb(floating[:] l_arr, floating[:] u_arr, floating[:] v_arr, floating[:, ::1] rgb_arr) nogil:
#     _luv_to_xyz(l_arr, u_arr, v_arr, rgb_arr)
#     _xyz_to_rgb(rgb_arr[:, 0], rgb_arr[:, 1], rgb_arr[:, 2], rgb_arr)
#
#
# cdef void _luv_to_lch(floating[:] l_arr, floating[:] u_arr, floating[:] v_arr, floating[:, ::1] lch_arr) nogil:
#     _luv_to_xyz(l_arr, u_arr, v_arr, lch_arr)
#     _xyz_to_lab(lch_arr[:, 0], lch_arr[:, 1], lch_arr[:, 2], lch_arr)
#     _lab_to_lch(lch_arr[:, 0], lch_arr[:, 1], lch_arr[:, 2], lch_arr)


# Constants
cdef:
    np.float32_t bintercept = 4.0 / 29  # 0.137931
    np.float32_t delta = 6.0 / 29  # 0.206896
    np.float32_t t0 = delta ** 3  # 0.008856
    np.float32_t alpha = (delta ** -2) / 3  # 7.787037
    np.float32_t third = 1.0 / 3
    np.float32_t kappa = (29.0 / 3) ** 3  # 903.3
    np.float32_t gamma = 2.2
    np.float32_t xn = 0.95047
    np.float32_t yn = 1.0
    np.float32_t zn = 1.08883
    np.float32_t denom_n = xn + (15 * yn) + (3 * zn)
    np.float32_t uprime_n = (4 * xn) / denom_n
    np.float32_t vprime_n = (9 * yn) / denom_n


    # Compile time option to use
    # sRGB companding (default, True) or simplified gamma (False)
    # sRGB companding is slightly slower but is more accurate at
    # the extreme ends of scale
    # Unit tests tuned to sRGB companding, change with caution
    bint SRGB_COMPAND = True


# Direct colorspace conversions

cdef void _rgb_to_xyz(floating[:] red_arr, floating[:] green_arr, floating[:] blue_arr, floating[:, ::1] xyz_arr) nogil:
    cdef floating r, g, b, rl, gl, bl, x, y, z
    cdef Py_ssize_t idx

    for idx in range(red_arr.shape[0]):
        r = red_arr[idx]
        g = green_arr[idx]
        b = blue_arr[idx]

        # convert RGB to linear scale
        if SRGB_COMPAND:
            if r <= 0.04045:
                rl = r / 12.92
            else:
                rl = ((r + 0.055) / 1.055) ** 2.4
            if g <= 0.04045:
                gl = g / 12.92
            else:
                gl = ((g + 0.055) / 1.055) ** 2.4
            if b <= 0.04045:
                bl = b / 12.92
            else:
                bl = ((b + 0.055) / 1.055) ** 2.4
        else:
            # Use "simplified sRGB"
            rl = r ** gamma
            gl = g ** gamma
            bl = b ** gamma

        # matrix mult for srgb->xyz,
        # includes adjustment for reference white
        x = ((rl * 0.4124564) + (gl * 0.3575761) + (bl * 0.1804375)) / xn
        y = ((rl * 0.2126729) + (gl * 0.7151522) + (bl * 0.0721750))
        z = ((rl * 0.0193339) + (gl * 0.1191920) + (bl * 0.9503041)) / zn

        xyz_arr[idx, 0] = x
        xyz_arr[idx, 1] = y
        xyz_arr[idx, 2] = z


cdef void _xyz_to_lab(floating[:] x_arr, floating[:] y_arr, floating[:] z_arr, floating[:, ::1] lab) nogil:
    cdef floating x, y, z, fx, fy, fz
    cdef floating L, a, b
    cdef Py_ssize_t idx

    for idx in range(x_arr.shape[0]):
        x = x_arr[idx]
        y = y_arr[idx]
        z = z_arr[idx]

        # convert XYZ to LAB colorspace
        if x > t0:
            fx = x ** third
        else:
            fx = (alpha * x) + bintercept

        if y > t0:
            fy = y ** third
        else:
            fy = (alpha * y) + bintercept

        if z > t0:
            fz = z ** third
        else:
            fz = (alpha * z) + bintercept

        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        lab[idx, 0] = L
        lab[idx, 1] = a
        lab[idx, 2] = b


cdef void _lab_to_lch(floating[:] L_arr, floating[:] a_arr, floating[:] b_arr, floating[:, ::1] lch) nogil:
    cdef Py_ssize_t idx
    cdef floating c, h

    for idx in range(L_arr.shape[0]):
        lch[idx, 0] = L_arr[idx]
        # store temporary results then write output to avoid corruption
        # if the output array is the same as the input arrays
        c = ((a_arr[idx] * a_arr[idx]) + (b_arr[idx] * b_arr[idx])) ** 0.5
        h = atan2(b_arr[idx], a_arr[idx])
        lch[idx, 1] = c
        lch[idx, 2] = h


cdef void _lch_to_lab(floating[:] l_arr, floating[:] c_arr, floating[:] h_arr, floating[:, ::1] lab_arr) nogil:
    cdef floating a, b
    cdef Py_ssize_t idx

    for idx in range(l_arr.shape[0]):
        a = c_arr[idx] * cos(h_arr[idx])
        b = c_arr[idx] * sin(h_arr[idx])
        lab_arr[idx, 0] = l_arr[idx]
        lab_arr[idx, 1] = a
        lab_arr[idx, 2] = b


cdef void _lab_to_xyz(floating[:] l_arr, floating[:] a_arr, floating[:] b_arr, floating[:, ::1] xyz_arr) nogil:
    cdef floating x, y, z, L, a, b, tx, ty, tz
    cdef Py_ssize_t idx

    for idx in range(l_arr.shape[0]):
        L = l_arr[idx]
        a = a_arr[idx]
        b = b_arr[idx]

        tx = ((L + 16) / 116.0) + (a / 500.0)
        if tx > delta:
            x = tx ** 3
        else:
            x = 3 * delta * delta * (tx - bintercept)

        ty = (L + 16) / 116.0
        if ty > delta:
            y = ty ** 3
        else:
            y = 3 * delta * delta * (ty - bintercept)

        tz = ((L + 16) / 116.0) - (b / 200.0)
        if tz > delta:
            z = tz ** 3
        else:
            z = 3 * delta * delta * (tz - bintercept)

        xyz_arr[idx, 0] = x
        xyz_arr[idx, 1] = y
        xyz_arr[idx, 2] = z


cdef void _xyz_to_rgb(floating[:] x_arr, floating[:] y_arr, floating[:] z_arr, floating[:, ::1] rgb_arr) nogil:
    cdef floating rlin, glin, blin, r, g, b, x, y, z
    cdef Py_ssize_t idx

    for idx in range(x_arr.shape[0]):
        x = x_arr[idx]
        y = y_arr[idx]
        z = z_arr[idx]

        # uses reference white d65
        x = x * xn
        z = z * zn

        # XYZ to sRGB
        # expanded matrix multiplication
        rlin = (x * 3.2404542) + (y * -1.5371385) + (z * -0.4985314)
        glin = (x * -0.9692660) + (y * 1.8760108) + (z * 0.0415560)
        blin = (x * 0.0556434) + (y * -0.2040259) + (z * 1.0572252)

        if SRGB_COMPAND:
            if rlin <= 0.0031308:
                r = 12.92 * rlin
            else:
                r = (1.055 * (rlin ** (1 / 2.4))) - 0.055
            if glin <= 0.0031308:
                g = 12.92 * glin
            else:
                g = (1.055 * (glin ** (1 / 2.4))) - 0.055
            if blin <= 0.0031308:
                b = 12.92 * blin
            else:
                b = (1.055 * (blin ** (1 / 2.4))) - 0.055
        else:
            # Use simplified sRGB
            r = rlin ** (1 / gamma)
            g = glin ** (1 / gamma)
            b = blin ** (1 / gamma)

        # constrain to 0..1 to deal with any float drift
        if r > 1.0:
            r = 1.0
        elif r < 0.0:
            r = 0.0
        if g > 1.0:
            g = 1.0
        elif g < 0.0:
            g = 0.0
        if b > 1.0:
            b = 1.0
        elif b < 0.0:
            b = 0.0

        rgb_arr[idx, 0] = r
        rgb_arr[idx, 1] = g
        rgb_arr[idx, 2] = b


cdef void _xyz_to_luv(floating[:] x_arr, floating[:] y_arr, floating[:] z_arr, floating[:, ::1] luv_arr) nogil:
    cdef floating L, u, v, uprime, vprime, denom, x, y, z
    cdef Py_ssize_t idx

    for idx in range(x_arr.shape[0]):
        x = x_arr[idx]
        y = y_arr[idx]
        z = z_arr[idx]

        denom = x + (15 * y) + (3 * z)
        uprime = (4 * x) / denom
        vprime = (9 * y) / denom

        y = y / yn

        if y <= t0:
            L = kappa * y
        else:
            L = (116 * (y ** third)) - 16

        u = 13 * L * (uprime - uprime_n)
        v = 13 * L * (vprime - vprime_n)

        luv_arr[idx, 0] = L
        luv_arr[idx, 1] = u
        luv_arr[idx, 2] = v


cdef void _luv_to_xyz(floating[:] l_arr, floating[:] u_arr, floating[:] v_arr, floating[:, ::1] xyz_arr) nogil:
    cdef floating x, y, z, uprime, vprime, L, u, v
    cdef Py_ssize_t idx

    for idx in range(l_arr.shape[0]):
        L = l_arr[idx]
        u = u_arr[idx]
        v = v_arr[idx]

        if L == 0.0:
            xyz_arr[idx, 0] = 0.0
            xyz_arr[idx, 1] = 0.0
            xyz_arr[idx, 2] = 0.0
            continue

        uprime = (u / (13 * L)) + uprime_n
        vprime = (v / (13 * L)) + vprime_n

        if L <= 8.0:
            y = L / kappa
        else:
            y = ((L + 16) / 116.0) ** 3

        x = y * ((9 * uprime) / (4 * vprime))
        z = y * ((12 - (3 * uprime) - (20 * vprime)) / (4 * vprime))

        xyz_arr[idx, 0] = x
        xyz_arr[idx, 1] = y
        xyz_arr[idx, 2] = z
