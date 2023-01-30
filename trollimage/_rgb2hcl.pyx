cimport cython

from libc.math cimport exp, tan, M_PI
cimport numpy as np
import numpy as np

ctypedef fused floating:
    np.float32_t
    np.float64_t


np.import_array()


def hcl2rgb_numpy(
        object hcl_arr,
        float gamma = 3.0,
        float y_0 = 100.0):
    """Convert an HCL (hue, chroma, luminance) array to RGB.

    This algorithm is based on the work described in:

    Madenda, Sarifuddin. (2005). A new perceptually uniform color space with associated
    color similarity measure for content-based image and video retrieval. Multimedia
    Information Retrieval Workshop, 28th Annual ACM SIGIR Conference.

    Additionally, the code is a numpy-friendly port of the matlab code in:

    Sarifuddin Madenda (2023). RGB to HCL and HCL to RGB color conversion
    (https://www.mathworks.com/matlabcentral/fileexchange/100878-rgb-to-hcl-and-hcl-to-rgb-color-conversion),
    MATLAB Central File Exchange. Retrieved January 20, 2023.

    Lastly, the python code here is inspired by a similar implementation of the
    same algorithm in the `colour-science` python package:

    https://github.com/colour-science/colour

    Args:
        hcl_arr: Numpy array of HCL values. The array can be any
            shape as long as the channel (band) dimension is the last (-1)
            dimension. Hue must be between -pi to pi. Chroma and Luminance
            should be between 0 and 1.
        gamma: Correction factor (see related paper). In normal use this does
            not need to be changed from the default of 3.0.
        y_0: White reference luminance. In normal use this does not need to
            be changed from the default of 100.0.

    Returns: RGB array where each Red, Green, and Blue channel is between 0 and 1.

    """
    cdef tuple shape = hcl_arr.shape
    cdef np.ndarray hcl_2d = hcl_arr.reshape((-1, 3))
    cdef np.ndarray rgb
    if hcl_arr.dtype == np.float32:
        rgb = _call_hcl_to_rgb(<np.ndarray[np.float32_t, ndim=2]> hcl_2d, gamma, y_0)
    else:
        rgb = _call_hcl_to_rgb(<np.ndarray[np.float64_t, ndim=2]> hcl_2d, gamma, y_0)
    return rgb.reshape(shape)


cdef np.ndarray[floating, ndim=2] _call_hcl_to_rgb(
        np.ndarray[floating, ndim=2] hcl,
        float gamma,
        float y_0,
):
    cdef np.ndarray[floating, ndim=1] hue = hcl[:, 0]  # in radians
    cdef np.ndarray[floating, ndim=1] chroma = hcl[:, 1]
    cdef np.ndarray[floating, ndim=1] luminance = hcl[:, 2]
    cdef floating[:] hue_view, chroma_view, luminance_view
    hue_view = hue
    chroma_view = chroma
    luminance_view = luminance
    cdef np.ndarray[floating, ndim=2] rgb = np.empty((hue_view.shape[0], 3), dtype=hcl.dtype)
    cdef floating[:, :] rgb_view = rgb
    with nogil:
        _hcl_to_rgb(hue_view, chroma_view, luminance_view, rgb_view, gamma, y_0)
    return rgb


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _hcl_to_rgb(
        floating[:] hue_arr,
        floating[:] chroma_arr,
        floating[:] luminance_arr,
        floating[:, :] rgb_arr,
        floating gamma,
        floating y_0,
) nogil:
    cdef Py_ssize_t idx
    cdef floating hue, chroma, luminance
    cdef floating q, rgb_min, rgb_max, tmp
    for idx in range(hue_arr.shape[0]):
        hue = hue_arr[idx]
        luminance = luminance_arr[idx]
        chroma = chroma_arr[idx]
        if luminance == 0:
            q = 1.0
        else:
            q = exp((1 - ((chroma * 3.0) / (luminance * 4.0))) * gamma / y_0)

        rgb_min = ((luminance * 4.0) - (chroma * 3.0)) / (4.0 * q - 2.0)
        rgb_max = rgb_min + (chroma * 3.0) / (2.0 * q)

        if 0 <= hue <= M_PI / 3.0:
            rgb_arr[idx, 0] = rgb_max
            tmp = tan(1.5 * hue)
            rgb_arr[idx, 1] = (rgb_max * tmp + rgb_min) / (1 + tmp)
            rgb_arr[idx, 2] = rgb_min
        elif M_PI / 3.0 < hue <= (2.0 / 3.0) * M_PI:
            tmp = tan(0.75 * (hue - M_PI))
            rgb_arr[idx, 0] = (rgb_max * (1 + tmp) - rgb_min) / tmp
            rgb_arr[idx, 1] = rgb_max
            rgb_arr[idx, 2] = rgb_min
        elif (2.0 / 3.0) * M_PI < hue <= M_PI:
            rgb_arr[idx, 0] = rgb_min
            rgb_arr[idx, 1] = rgb_max
            tmp = tan(0.75 * (hue - M_PI))
            rgb_arr[idx, 2] = rgb_max * (1 + tmp) - (rgb_min * tmp)
        elif -(1.0 / 3.0) * M_PI <= hue < 0:
            rgb_arr[idx, 0] = rgb_max
            rgb_arr[idx, 1] = rgb_min
            tmp = tan(0.75 * hue)
            rgb_arr[idx, 2] = rgb_min * (1 + tmp) - (rgb_max * tmp)
        elif -(2.0 / 3.0) * M_PI <= hue < -(1.0 / 3.0) * M_PI:
            tmp = tan(0.75 * hue)
            rgb_arr[idx, 0] = (rgb_min * (1 + tmp) - rgb_max) / tmp
            rgb_arr[idx, 1] = rgb_min
            rgb_arr[idx, 2] = rgb_max
        else:
            rgb_arr[idx, 0] = rgb_min
            tmp = tan(1.5 * (hue + M_PI))
            rgb_arr[idx, 1] = (rgb_min * tmp + rgb_max) / (1 + tmp)
            rgb_arr[idx, 2] = rgb_max

