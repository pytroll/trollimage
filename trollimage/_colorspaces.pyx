# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
# cython: initializedcheck=False, nonecheck=False, cpow=True
"""Cython colorspace conversion kernels.

Every conversion is a single fused loop over pixels: the per-pixel scalar
steps (``_*_px``) are inlined by the C compiler, so a three-step conversion
like RGB -> XYZ -> Lab -> LCh reads and writes each pixel exactly once. The
loops are bound by the libm calls (``pow``, ``atan2``, ``sincos``), not by
memory traffic, so the float32 specialization deliberately uses the float
libm entry points (``powf`` etc.) and float constants to avoid promoting
everything to double.

"""

from libc.math cimport atan2, atan2f, cos, cosf, pow, powf, sin, sinf, sqrt, sqrtf
import numpy as np
cimport numpy as np

ctypedef fused floating:
    np.float32_t
    np.float64_t


np.import_array()

# Function pointer type to allow for generic high-level functions
ctypedef void (*CONVERT_FUNC)(const floating[:, :] in_arr, floating[:, :] out_arr) noexcept nogil

# Compile-time double constants. Using C macros (instead of module-level
# ``cdef`` variables) lets the compiler fold them, and the ``<floating>``
# casts at the use sites keep the float32 specialization in single precision.
cdef extern from *:
    """
    #define TI_BINTERCEPT (4.0 / 29.0)              /* 0.137931 */
    #define TI_DELTA (6.0 / 29.0)                   /* 0.206896 */
    #define TI_T0 (TI_DELTA * TI_DELTA * TI_DELTA)  /* 0.008856 */
    #define TI_ALPHA ((1.0 / (TI_DELTA * TI_DELTA)) / 3.0)  /* 7.787037 */
    #define TI_THIRD (1.0 / 3.0)
    #define TI_KAPPA ((29.0 / 3.0) * (29.0 / 3.0) * (29.0 / 3.0))  /* 903.3 */
    #define TI_GAMMA 2.2
    #define TI_XN 0.95047
    #define TI_YN 1.0
    #define TI_ZN 1.08883
    #define TI_DENOM_N (TI_XN + (15 * TI_YN) + (3 * TI_ZN))
    #define TI_UPRIME_N ((4 * TI_XN) / TI_DENOM_N)
    #define TI_VPRIME_N ((9 * TI_YN) / TI_DENOM_N)

    /* Compile time option to use sRGB companding (default, 1 - True) or
     * simplified gamma (0 - False). sRGB companding is slightly slower but is
     * more accurate at the extreme ends of scale.
     * Unit tests tuned to sRGB companding, change with caution. */
    #define TI_SRGB_COMPAND 1
    """
    double BINTERCEPT "TI_BINTERCEPT"
    double DELTA "TI_DELTA"
    double T0 "TI_T0"
    double ALPHA "TI_ALPHA"
    double THIRD "TI_THIRD"
    double KAPPA "TI_KAPPA"
    double GAMMA "TI_GAMMA"
    double XN "TI_XN"
    double YN "TI_YN"
    double ZN "TI_ZN"
    double UPRIME_N "TI_UPRIME_N"
    double VPRIME_N "TI_VPRIME_N"
    bint SRGB_COMPAND "TI_SRGB_COMPAND"


def rgb2lch(object rgba_arr, object out=None):
    """Convert numpy RGB[A] arrays to CIE LCh_ab (Luminance, Chroma, Hue).

    See :func:`convert_colors` for more information on color spaces.

    Args:
        rgba_arr: Numpy array of RGB or RGBA colors. The array can be any
            shape as long as the channel (band) dimension is the last (-1)
            dimension. If an Alpha (A) channel is provided it is ignored.
            Values should be between 0 and 1.
        out: Optional output array. See :func:`convert_colors`.

    Returns: LCH_ab (l, c, h) numpy array where the last dimension represents Hue, Chroma,
        and Luminance. Hue is in radians from -pi to pi. Chroma is from 0 to
        1. Luminance is also from 0 and 1 (usually a maximum of ~0.5).

    """
    return convert_colors(rgba_arr, "rgb", "lch", out=out)


def lch2rgb(object lch_arr, object out=None):
    """Convert an LCH (luminance, chroma, hue) array to RGB.

    See :func:`convert_colors` for more information on color spaces.

    Args:
        lch_arr: Numpy array of HCL values. The array can be any
            shape as long as the channel (band) dimension is the last (-1)
            dimension. Hue must be between -pi to pi. Chroma and Luminance
            should be between 0 and 1.
        out: Optional output array. See :func:`convert_colors`.

    Returns: RGB array where each Red, Green, and Blue channel is between 0 and 1.

    """
    return convert_colors(lch_arr, "lch", "rgb", out=out)


def convert_colors(object input_colors, str in_space, str out_space, object out=None):
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
            is ignored. Float32 and float64 arrays are converted in their own
            precision; any other dtype is converted as float64. The array does
            not need to be contiguous.
        in_space: String name of the color space of the input data. Can be one
            of "rgb", "lch", "lab", "luv", or "xyz".
        out_space: String name of the color space to convert to. Available
            options are the same as for ``in_space``.
        out: Optional array to write the result into. Must have the same
            shape as ``input_colors[..., :3]`` and the dtype the conversion is
            performed in. It may be a non-contiguous view (for example a
            transposed "planar" ``(3, ...)`` array), which lets callers get
            channel-first output without an extra copy.

    Returns:
        Numpy array with equal shape to the input, but the last dimension is
        always length 3 to match the ``out_space`` color space. If ``out`` was
        provided it is returned.

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
    cdef object in123_arr = np.asarray(input_colors)[..., :3]
    cdef tuple shape = in123_arr.shape
    cdef object dtype = in123_arr.dtype
    if dtype != np.float32 and dtype != np.float64:
        dtype = np.dtype(np.float64)
        in123_arr = in123_arr.astype(dtype)
    if out is None:
        out = np.empty(shape, dtype=dtype)
    elif out.shape != shape or out.dtype != dtype:
        raise ValueError(f"'out' must have shape {shape} and dtype {dtype}, got {out.shape} and {out.dtype}")

    # reshape to (N, 3); this is a view unless the color axis was sliced off
    # a wider (e.g. RGBA) array, in which case numpy copies
    cdef object in123_2d = in123_arr.reshape((-1, 3))
    cdef object out123_2d = out.reshape((-1, 3))
    if dtype == np.float32:
        _call_convert_func[np.float32_t](in123_2d, out123_2d, in_space, out_space)
    else:
        _call_convert_func[np.float64_t](in123_2d, out123_2d, in_space, out_space)
    return out


cdef void _call_convert_func(
        const floating[:, :] in_colors, floating[:, :] out_colors, str in_space, str out_space,
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

    with nogil:
        conv_func(in_colors, out_colors)


# Typed libm wrappers: pick the float or double entry point at compile time

cdef inline floating _pow(floating x, floating y) noexcept nogil:
    if floating is np.float32_t:
        return powf(x, y)
    else:
        return pow(x, y)


cdef inline floating _sqrt(floating x) noexcept nogil:
    if floating is np.float32_t:
        return sqrtf(x)
    else:
        return sqrt(x)


cdef inline floating _sin(floating x) noexcept nogil:
    if floating is np.float32_t:
        return sinf(x)
    else:
        return sin(x)


cdef inline floating _cos(floating x) noexcept nogil:
    if floating is np.float32_t:
        return cosf(x)
    else:
        return cos(x)


cdef inline floating _atan2(floating y, floating x) noexcept nogil:
    if floating is np.float32_t:
        return atan2f(y, x)
    else:
        return atan2(y, x)


# Per-pixel scalar steps. These are all inlined into the fused loops below.

cdef inline floating _to_linear_rgb(floating rgb_component) noexcept nogil:
    if SRGB_COMPAND:
        return _to_linear_srgb_expand(rgb_component)
    # Use "simplified sRGB"
    return _pow(rgb_component, <floating>GAMMA)


cdef inline floating _to_linear_srgb_expand(floating rgb_component) noexcept nogil:
    if rgb_component <= <floating>0.04045:
        return rgb_component * <floating>(1.0 / 12.92)
    return _pow((rgb_component + <floating>0.055) * <floating>(1.0 / 1.055), <floating>2.4)


cdef inline floating _to_nonlinear_rgb(floating rgb_component) noexcept nogil:
    if SRGB_COMPAND:
        return _to_nonlinear_srgb_compand(rgb_component)
    # Use "simplified sRGB"
    return _pow(rgb_component, <floating>(1.0 / GAMMA))


cdef inline floating _to_nonlinear_srgb_compand(floating rgb_component) noexcept nogil:
    if rgb_component <= <floating>0.0031308:
        return <floating>12.92 * rgb_component
    return (<floating>1.055 * _pow(rgb_component, <floating>(1.0 / 2.4))) - <floating>0.055


cdef inline floating _clamp_0_1(floating val) noexcept nogil:
    # written so that NaN falls through both comparisons and stays NaN
    val = <floating>0.0 if val < <floating>0.0 else val
    return <floating>1.0 if val > <floating>1.0 else val


cdef inline floating _lab_f(floating t) noexcept nogil:
    # glibc's cbrt is slower than its pow, so keep pow here
    if t > <floating>T0:
        return _pow(t, <floating>THIRD)
    return (<floating>ALPHA * t) + <floating>BINTERCEPT


cdef inline floating _lab_finv(floating t) noexcept nogil:
    if t > <floating>DELTA:
        return t * t * t
    return <floating>(3 * DELTA * DELTA) * (t - <floating>BINTERCEPT)


cdef inline void _rgb_to_xyz_px(floating r, floating g, floating b,
                                floating* x, floating* y, floating* z) noexcept nogil:
    # convert RGB to linear scale
    cdef floating rl = _to_linear_rgb(r)
    cdef floating gl = _to_linear_rgb(g)
    cdef floating bl = _to_linear_rgb(b)

    # matrix mult for srgb->xyz,
    # includes adjustment for reference white
    x[0] = ((rl * <floating>0.4124564) + (gl * <floating>0.3575761) + (bl * <floating>0.1804375)) * <floating>(1.0 / XN)
    y[0] = ((rl * <floating>0.2126729) + (gl * <floating>0.7151522) + (bl * <floating>0.0721750))
    z[0] = ((rl * <floating>0.0193339) + (gl * <floating>0.1191920) + (bl * <floating>0.9503041)) * <floating>(1.0 / ZN)


cdef inline void _xyz_to_rgb_px(floating x, floating y, floating z,
                                floating* r, floating* g, floating* b) noexcept nogil:
    cdef floating rlin, glin, blin
    # uses reference white d65
    x = x * <floating>XN
    z = z * <floating>ZN

    # XYZ to sRGB
    # expanded matrix multiplication
    rlin = (x * <floating>3.2404542) + (y * <floating>-1.5371385) + (z * <floating>-0.4985314)
    glin = (x * <floating>-0.9692660) + (y * <floating>1.8760108) + (z * <floating>0.0415560)
    blin = (x * <floating>0.0556434) + (y * <floating>-0.2040259) + (z * <floating>1.0572252)

    # constrain to 0..1 to deal with any float drift
    r[0] = _clamp_0_1(_to_nonlinear_rgb(rlin))
    g[0] = _clamp_0_1(_to_nonlinear_rgb(glin))
    b[0] = _clamp_0_1(_to_nonlinear_rgb(blin))


cdef inline void _xyz_to_lab_px(floating x, floating y, floating z,
                                floating* L, floating* a, floating* b) noexcept nogil:
    cdef floating fx = _lab_f(x)
    cdef floating fy = _lab_f(y)
    cdef floating fz = _lab_f(z)
    L[0] = (<floating>116 * fy) - <floating>16
    a[0] = <floating>500 * (fx - fy)
    b[0] = <floating>200 * (fy - fz)


cdef inline void _lab_to_xyz_px(floating L, floating a, floating b,
                                floating* x, floating* y, floating* z) noexcept nogil:
    cdef floating ty = (L + <floating>16) * <floating>(1.0 / 116.0)
    cdef floating tx = ty + (a * <floating>(1.0 / 500.0))
    cdef floating tz = ty - (b * <floating>(1.0 / 200.0))
    x[0] = _lab_finv(tx)
    y[0] = _lab_finv(ty)
    z[0] = _lab_finv(tz)


cdef inline void _lab_to_lch_px(floating L, floating a, floating b,
                                floating* L_out, floating* c, floating* h) noexcept nogil:
    L_out[0] = L
    c[0] = _sqrt((a * a) + (b * b))
    h[0] = _atan2(b, a)


cdef inline void _lch_to_lab_px(floating L, floating c, floating h,
                                floating* L_out, floating* a, floating* b) noexcept nogil:
    L_out[0] = L
    a[0] = c * _cos(h)
    b[0] = c * _sin(h)


cdef inline void _xyz_to_luv_px(floating x, floating y, floating z,
                                floating* L, floating* u, floating* v) noexcept nogil:
    cdef floating denom, uprime, vprime, L_val
    # x and z arrive normalized by the reference white (see _rgb_to_xyz_px),
    # but u' and v' are defined on absolute XYZ
    x = x * <floating>XN
    z = z * <floating>ZN
    denom = x + (<floating>15 * y) + (<floating>3 * z)
    if denom == <floating>0.0:
        # black: avoid 0 / 0
        L[0] = <floating>0.0
        u[0] = <floating>0.0
        v[0] = <floating>0.0
        return
    uprime = (<floating>4 * x) / denom
    vprime = (<floating>9 * y) / denom

    y = y * <floating>(1.0 / YN)
    if y <= <floating>T0:
        L_val = <floating>KAPPA * y
    else:
        L_val = (<floating>116 * _pow(y, <floating>THIRD)) - <floating>16

    L[0] = L_val
    u[0] = <floating>13 * L_val * (uprime - <floating>UPRIME_N)
    v[0] = <floating>13 * L_val * (vprime - <floating>VPRIME_N)


cdef inline void _luv_to_xyz_px(floating L, floating u, floating v,
                                floating* x, floating* y, floating* z) noexcept nogil:
    cdef floating uprime, vprime, y_val, t
    if L == <floating>0.0:
        x[0] = <floating>0.0
        y[0] = <floating>0.0
        z[0] = <floating>0.0
        return

    uprime = (u / (<floating>13 * L)) + <floating>UPRIME_N
    vprime = (v / (<floating>13 * L)) + <floating>VPRIME_N

    if L <= <floating>8.0:
        y_val = L * <floating>(1.0 / KAPPA)
    else:
        t = (L + <floating>16) * <floating>(1.0 / 116.0)
        y_val = t * t * t

    # produce XYZ normalized by the reference white like _rgb_to_xyz_px does
    x[0] = y_val * ((<floating>9 * uprime) / (<floating>4 * vprime)) * <floating>(1.0 / XN)
    y[0] = y_val
    z[0] = y_val * ((<floating>12 - (<floating>3 * uprime) - (<floating>20 * vprime)) / (<floating>4 * vprime))
    z[0] = z[0] * <floating>(1.0 / ZN)


# Fused array loops: one read and one write per pixel for every conversion

cdef void _rgb_to_xyz(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _rgb_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _rgb_to_lab(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _rgb_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_lab_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _rgb_to_lch(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _rgb_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_lab_px(c1, c2, c3, &c1, &c2, &c3)
        _lab_to_lch_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _rgb_to_luv(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _rgb_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_luv_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _xyz_to_rgb(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _xyz_to_rgb_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _xyz_to_lab(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _xyz_to_lab_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _xyz_to_lch(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _xyz_to_lab_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _lab_to_lch_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _xyz_to_luv(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _xyz_to_luv_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lab_to_xyz(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _lab_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lab_to_rgb(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _lab_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_rgb_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lab_to_lch(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _lab_to_lch_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lab_to_luv(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _lab_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_luv_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lch_to_lab(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _lch_to_lab_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lch_to_xyz(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _lch_to_lab_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _lab_to_xyz_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lch_to_rgb(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _lch_to_lab_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _lab_to_xyz_px(c1, c2, c3, &c1, &c2, &c3)
        _xyz_to_rgb_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _lch_to_luv(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _lch_to_lab_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _lab_to_xyz_px(c1, c2, c3, &c1, &c2, &c3)
        _xyz_to_luv_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _luv_to_xyz(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    for idx in range(inp.shape[0]):
        _luv_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _luv_to_lab(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _luv_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_lab_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _luv_to_rgb(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _luv_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_rgb_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])


cdef void _luv_to_lch(const floating[:, :] inp, floating[:, :] out) noexcept nogil:
    cdef Py_ssize_t idx
    cdef floating c1, c2, c3
    for idx in range(inp.shape[0]):
        _luv_to_xyz_px(inp[idx, 0], inp[idx, 1], inp[idx, 2], &c1, &c2, &c3)
        _xyz_to_lab_px(c1, c2, c3, &c1, &c2, &c3)
        _lab_to_lch_px(c1, c2, c3, &out[idx, 0], &out[idx, 1], &out[idx, 2])
