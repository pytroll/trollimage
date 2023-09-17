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

# Function pointer type to allow for generic high-level functions
ctypedef void (*CONVERT_FUNC)(floating[:, ::1] in_out_arr) nogil


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


def rgb2lch(
        object rgba_arr,
):
    """Convert numpy RGB[A] arrays to CIE LCh_ab (Luminance, Chroma, Hue).

    See :func:`convert_colors` for more information on color spaces.

    Args:
        rgba_arr: Numpy array of RGB or RGBA colors. The array can be any
            shape as long as the channel (band) dimension is the last (-1)
            dimension. If an Alpha (A) channel is provided it is ignored.
            Values should be between 0 and 1.

    Returns: LCH_ab (l, c, h) numpy array where the last dimension represents Hue, Chroma,
        and Luminance. Hue is in radians from -pi to pi. Chroma is from 0 to
        1. Luminance is also from 0 and 1 (usually a maximum of ~0.5).

    """
    return convert_colors(rgba_arr, "rgb", "lch")


def lch2rgb(object lch_arr):
    """Convert an LCH (luminance, chroma, hue) array to RGB.

    See :func:`convert_colors` for more information on color spaces.

    Args:
        lch_arr: Numpy array of HCL values. The array can be any
            shape as long as the channel (band) dimension is the last (-1)
            dimension. Hue must be between -pi to pi. Chroma and Luminance
            should be between 0 and 1.

    Returns: RGB array where each Red, Green, and Blue channel is between 0 and 1.

    """
    return convert_colors(lch_arr, "lch", "rgb")


def convert_colors(object input_colors, str in_space, str out_space):
    """Convert from one color space to another.

    Color Spaces
    ^^^^^^^^^^^^

    * **rgb**: Red, Green, and Blue. Each channel should be in a 0 to 1
        normalized range.
    * **lch**: LCh_ab (LCH). The CIELAB Cylindrical Luminance, Chroma, and
        Hue color space. Luminance values range from about 0 to about 100.
        Chroma values range from about 0 to 120. Hue is in radians and is
        from -pi to pi.
        See the
        `wikipedia article <https://en.wikipedia.org/wiki/CIELAB_color_space#CIEHLC_cylindrical_model>`_
        for more information.
    * **lab**: CIELAB. The cartesian version of "lch". Luminance is the same
        value as LCh. The ``a*`` and ``b*`` values range from about -120 to 120.
        See the
        `wikipedia article <https://en.wikipedia.org/wiki/CIELAB_color_space>`_
        for more info.
    * **luv**: CIELUV. Luminance and a ``u*`` and ``v*`` start component. The
        luminance values range from 0 to 100. The u and v values range from
        about -200 to 200. See the
        `wikipedia article <https://en.wikipedia.org/wiki/CIELUV>`_ for more
        info.
    * **xyz**: CIE XYZ. Values range from about 0 to 1. See the
        `wikipedia article <https://en.wikipedia.org/wiki/CIE_1931_color_space>`_
        for more info.

    Args:
        input_colors: Numpy array of input colors in ``in_space`` color space.
            The array can be of any shape, but the color dimension must be the
            last dimension. Only the first three elements in the color
            dimension will be used. So if an Alpha (A) channel is provided it
            is ignored.
        in_space: String name of the color space of the input data. Can be one
            of "rgb", "lch", "lab", "luv", or "xyz".
        out_space: String name of the color space to convert to. Available
            options are the same as for ``in_space``.

    Returns:
        Numpy array with equal shape to the input, but the last dimension is
        always length 3 to match the ``out_space`` color space.

    Notes:
        This function is called by all the individual ``<space>2<space>``
        functions. This function and all color conversion functions are
        heavily based on or taken from the
        `rio-color <https://github.com/mapbox/rio-color>`_ project which is
        under an MIT license. A copy of this license is available in the
        ``trollimage`` package and root of the git repository. The majority
        of changes made to the ``rio-color`` code were to support memory views
        in a "no GIL" way and allow for 32-bit and 64-bit floating point data.

    """
    cdef object in123_arr = input_colors[..., :3]
    cdef tuple shape = in123_arr.shape
    cdef np.ndarray in123_2d = in123_arr.reshape((-1, 3))
    cdef np.ndarray out123
    if in123_arr.dtype == np.float32:
        out123 = _call_convert_func[np.float32_t](in123_2d, in_space, out_space)
    else:
        out123 = _call_convert_func[np.float64_t](in123_2d, in_space, out_space)
    return out123.reshape(shape)


cdef np.ndarray[floating, ndim=2] _call_convert_func(
        floating[:, :] in_colors, str in_space, str out_space,
):
    cdef CONVERT_FUNC conv_func = NULL
    if in_space == "rgb":
        if out_space == "lch":
            conv_func = _rgb_to_lch[floating]
        elif out_space == "lab":
            conv_func = _rgb_to_lab[floating]
        elif out_space == "luv":
            conv_func = _rgb_to_luv[floating]
        elif out_space == "xyz":
            conv_func = _rgb_to_xyz[floating]
    elif in_space == "lch":
        if out_space == "rgb":
            conv_func = _lch_to_rgb[floating]
        elif out_space == "lab":
            conv_func = _lch_to_lab[floating]
        elif out_space == "luv":
            conv_func = _lch_to_luv[floating]
        elif out_space == "xyz":
            conv_func = _lch_to_xyz[floating]
    elif in_space == "lab":
        if out_space == "rgb":
            conv_func = _lab_to_rgb[floating]
        elif out_space == "lch":
            conv_func = _lab_to_lch[floating]
        elif out_space == "luv":
            conv_func = _lab_to_luv[floating]
        elif out_space == "xyz":
            conv_func = _lab_to_xyz[floating]
    elif in_space == "luv":
        if out_space == "rgb":
            conv_func = _luv_to_rgb[floating]
        elif out_space == "lch":
            conv_func = _luv_to_lch[floating]
        elif out_space == "lab":
            conv_func = _luv_to_lab[floating]
        elif out_space == "xyz":
            conv_func = _luv_to_xyz[floating]
    elif in_space == "xyz":
        if out_space == "rgb":
            conv_func = _xyz_to_rgb[floating]
        elif out_space == "lch":
            conv_func = _xyz_to_lch[floating]
        elif out_space == "lab":
            conv_func = _xyz_to_lab[floating]
        elif out_space == "luv":
            conv_func = _xyz_to_luv[floating]

    if conv_func is NULL:
        raise ValueError("Unknown colorspace combination")

    cdef object dtype
    if floating is np.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef floating[:, ::1] in_out_view = in_colors.copy()
    with nogil:
        conv_func(in_out_view)
    return np.asarray(in_out_view)



cdef void _rgb_to_lab(floating[:, ::1] rgb_to_lab_arr) noexcept nogil:
    _rgb_to_xyz[floating](rgb_to_lab_arr)
    _xyz_to_lab[floating](rgb_to_lab_arr)


cdef void _rgb_to_lch(floating[:, ::1] rgb_to_lch_arr) noexcept nogil:
    _rgb_to_xyz(rgb_to_lch_arr)
    _xyz_to_lab[floating](rgb_to_lch_arr)
    _lab_to_lch[floating](rgb_to_lch_arr)


cdef void _rgb_to_luv(floating[:, ::1] rgb_to_luv_arr) noexcept nogil:
    _rgb_to_xyz(rgb_to_luv_arr)
    _xyz_to_luv[floating](rgb_to_luv_arr)


cdef void _xyz_to_lch(floating[:, ::1] xyz_to_lch_arr) noexcept nogil:
    _xyz_to_lab(xyz_to_lch_arr)
    _lab_to_lch[floating](xyz_to_lch_arr)


cdef void _lab_to_rgb(floating[:, ::1] lab_to_rgb_arr) noexcept nogil:
    _lab_to_xyz(lab_to_rgb_arr)
    _xyz_to_rgb[floating](lab_to_rgb_arr)


cdef void _lab_to_luv(floating[:, ::1] lab_to_luv_arr) noexcept nogil:
    _lab_to_xyz(lab_to_luv_arr)
    _xyz_to_luv[floating](lab_to_luv_arr)


cdef void _lch_to_xyz(floating[:, ::1] lch_to_xyz_arr) noexcept nogil:
    _lch_to_lab(lch_to_xyz_arr)
    _lab_to_xyz[floating](lch_to_xyz_arr)


cdef void _lch_to_rgb(floating[:, ::1] lch_to_rgb_arr) noexcept nogil:
    _lch_to_lab(lch_to_rgb_arr)
    _lab_to_xyz[floating](lch_to_rgb_arr)
    _xyz_to_rgb[floating](lch_to_rgb_arr)


cdef void _lch_to_luv(floating[:, ::1] lch_to_luv_arr) noexcept nogil:
    _lch_to_lab(lch_to_luv_arr)
    _lab_to_xyz[floating](lch_to_luv_arr)
    _xyz_to_rgb[floating](lch_to_luv_arr)


cdef void _luv_to_lab(floating[:, ::1] luv_to_lab_arr) noexcept nogil:
    _luv_to_xyz(luv_to_lab_arr)
    _xyz_to_lab[floating](luv_to_lab_arr)


cdef void _luv_to_rgb(floating[:, ::1] luv_to_rgb_arr) noexcept nogil:
    _luv_to_xyz(luv_to_rgb_arr)
    _xyz_to_rgb[floating](luv_to_rgb_arr)


cdef void _luv_to_lch(floating[:, ::1] luv_to_lch_arr) noexcept nogil:
    _luv_to_xyz(luv_to_lch_arr)
    _xyz_to_lab[floating](luv_to_lch_arr)
    _lab_to_lch[floating](luv_to_lch_arr)


# Direct colorspace conversions

cdef void _rgb_to_xyz(floating[:, ::1] rgb_to_xyz_arr) noexcept nogil:
    cdef floating r, g, b, rl, gl, bl, x, y, z
    cdef Py_ssize_t idx

    for idx in range(rgb_to_xyz_arr.shape[0]):
        r = rgb_to_xyz_arr[idx, 0]
        g = rgb_to_xyz_arr[idx, 1]
        b = rgb_to_xyz_arr[idx, 2]

        # convert RGB to linear scale
        rl = _to_linear_rgb(r)
        gl = _to_linear_rgb(g)
        bl = _to_linear_rgb(b)

        # matrix mult for srgb->xyz,
        # includes adjustment for reference white
        x = ((rl * 0.4124564) + (gl * 0.3575761) + (bl * 0.1804375)) / xn
        y = ((rl * 0.2126729) + (gl * 0.7151522) + (bl * 0.0721750))
        z = ((rl * 0.0193339) + (gl * 0.1191920) + (bl * 0.9503041)) / zn

        rgb_to_xyz_arr[idx, 0] = x
        rgb_to_xyz_arr[idx, 1] = y
        rgb_to_xyz_arr[idx, 2] = z


cdef inline floating _to_linear_rgb(floating rgb_component) noexcept nogil:
    if SRGB_COMPAND:
        return _to_linear_srgb_expand(rgb_component)
    # Use "simplified sRGB"
    return rgb_component ** gamma


cdef inline floating _to_linear_srgb_expand(floating rgb_component) noexcept nogil:
    if rgb_component <= 0.04045:
        return rgb_component / 12.92
    return ((rgb_component + 0.055) / 1.055) ** 2.4


cdef void _xyz_to_lab(floating[:, ::1] xyz_to_lab_arr) noexcept nogil:
    cdef floating x, y, z, fx, fy, fz
    cdef floating L, a, b
    cdef Py_ssize_t idx

    for idx in range(xyz_to_lab_arr.shape[0]):
        x = xyz_to_lab_arr[idx, 0]
        y = xyz_to_lab_arr[idx, 1]
        z = xyz_to_lab_arr[idx, 2]

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

        xyz_to_lab_arr[idx, 0] = L
        xyz_to_lab_arr[idx, 1] = a
        xyz_to_lab_arr[idx, 2] = b


cdef void _lab_to_lch(floating[:, ::1] lab_to_lch_arr) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c, h

    for idx in range(lab_to_lch_arr.shape[0]):
        # lab_to_lch_arr[idx, 0] = lab_to_lch_arr[idx, 0]
        # store temporary results then write output to avoid corruption
        # if the output array is the same as the input arrays
        c = ((lab_to_lch_arr[idx, 1] * lab_to_lch_arr[idx, 1]) + (lab_to_lch_arr[idx, 2] * lab_to_lch_arr[idx, 2])) ** 0.5
        h = atan2(lab_to_lch_arr[idx, 2], lab_to_lch_arr[idx, 1])
        lab_to_lch_arr[idx, 1] = c
        lab_to_lch_arr[idx, 2] = h


cdef void _lch_to_lab(floating[:, ::1] lch_to_lab_arr) noexcept nogil:
    cdef floating a, b
    cdef Py_ssize_t idx

    for idx in range(lch_to_lab_arr.shape[0]):
        a = lch_to_lab_arr[idx, 1] * cos(lch_to_lab_arr[idx, 2])
        b = lch_to_lab_arr[idx, 1] * sin(lch_to_lab_arr[idx, 2])
        lch_to_lab_arr[idx, 0] = lch_to_lab_arr[idx, 0]
        lch_to_lab_arr[idx, 1] = a
        lch_to_lab_arr[idx, 2] = b


cdef void _lab_to_xyz(floating[:, ::1] lab_to_xyz_arr) noexcept nogil:
    cdef floating x, y, z, L, a, b, tx, ty, tz
    cdef Py_ssize_t idx

    for idx in range(lab_to_xyz_arr.shape[0]):
        L = lab_to_xyz_arr[idx, 0]
        a = lab_to_xyz_arr[idx, 1]
        b = lab_to_xyz_arr[idx, 2]

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

        lab_to_xyz_arr[idx, 0] = x
        lab_to_xyz_arr[idx, 1] = y
        lab_to_xyz_arr[idx, 2] = z


cdef void _xyz_to_rgb(floating[:, ::1] xyz_to_rgb_arr) noexcept nogil:
    cdef floating rlin, glin, blin, r, g, b, x, y, z
    cdef Py_ssize_t idx

    for idx in range(xyz_to_rgb_arr.shape[0]):
        x = xyz_to_rgb_arr[idx, 0]
        y = xyz_to_rgb_arr[idx, 1]
        z = xyz_to_rgb_arr[idx, 2]

        # uses reference white d65
        x = x * xn
        z = z * zn

        # XYZ to sRGB
        # expanded matrix multiplication
        rlin = (x * 3.2404542) + (y * -1.5371385) + (z * -0.4985314)
        glin = (x * -0.9692660) + (y * 1.8760108) + (z * 0.0415560)
        blin = (x * 0.0556434) + (y * -0.2040259) + (z * 1.0572252)

        r  = _to_nonlinear_rgb(rlin)
        g  = _to_nonlinear_rgb(glin)
        b  = _to_nonlinear_rgb(blin)

        # constrain to 0..1 to deal with any float drift
        r = _clamp_0_1(r)
        g = _clamp_0_1(g)
        b = _clamp_0_1(b)

        xyz_to_rgb_arr[idx, 0] = r
        xyz_to_rgb_arr[idx, 1] = g
        xyz_to_rgb_arr[idx, 2] = b


cdef inline floating _clamp_0_1(floating val) noexcept nogil:
    val = <floating>0.0 if val < <floating>0.0 else val
    return <floating>1.0 if val > <floating>1.0 else val


cdef inline floating _to_nonlinear_rgb(floating rgb_component) noexcept nogil:
    if SRGB_COMPAND:
        return _to_nonlinear_srgb_compand(rgb_component)
    # Use "simplified sRGB"
    return rgb_component ** (1 / gamma)


cdef inline floating _to_nonlinear_srgb_compand(floating rgb_component) noexcept nogil:
    if rgb_component <= 0.0031308:
        return 12.92 * rgb_component
    return (1.055 * (rgb_component ** (1 / 2.4))) - 0.055


cdef void _xyz_to_luv(floating[:, ::1] xyz_to_luv_arr) noexcept nogil:
    cdef floating L, u, v, uprime, vprime, denom, x, y, z
    cdef Py_ssize_t idx

    for idx in range(xyz_to_luv_arr.shape[0]):
        x = xyz_to_luv_arr[idx, 0]
        y = xyz_to_luv_arr[idx, 1]
        z = xyz_to_luv_arr[idx, 2]

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

        xyz_to_luv_arr[idx, 0] = L
        xyz_to_luv_arr[idx, 1] = u
        xyz_to_luv_arr[idx, 2] = v


cdef void _luv_to_xyz(floating[:, ::1] luv_to_xyz_arr) noexcept nogil:
    cdef floating x, y, z, uprime, vprime, L, u, v
    cdef Py_ssize_t idx

    for idx in range(luv_to_xyz_arr.shape[0]):
        L = luv_to_xyz_arr[idx, 0]
        u = luv_to_xyz_arr[idx, 1]
        v = luv_to_xyz_arr[idx, 2]

        if L == 0.0:
            luv_to_xyz_arr[idx, 0] = 0.0
            luv_to_xyz_arr[idx, 1] = 0.0
            luv_to_xyz_arr[idx, 2] = 0.0
            continue

        uprime = (u / (13 * L)) + uprime_n
        vprime = (v / (13 * L)) + vprime_n

        if L <= 8.0:
            y = L / kappa
        else:
            y = ((L + 16) / 116.0) ** 3

        x = y * ((9 * uprime) / (4 * vprime))
        z = y * ((12 - (3 * uprime) - (20 * vprime)) / (4 * vprime))

        luv_to_xyz_arr[idx, 0] = x
        luv_to_xyz_arr[idx, 1] = y
        luv_to_xyz_arr[idx, 2] = z
