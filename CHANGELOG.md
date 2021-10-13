## Version 1.16.0 (2021/10/12)

### Issues Closed

* [Issue 87](https://github.com/pytroll/trollimage/issues/87) - ReportError "ValueError: Merged colormap 'values' are not monotonically increasing." ([PR 91](https://github.com/pytroll/trollimage/pull/91) by [@djhoese](https://github.com/djhoese))
* [Issue 84](https://github.com/pytroll/trollimage/issues/84) - allow customization of GDALMetaData tags for scale and offset ([PR 85](https://github.com/pytroll/trollimage/pull/85) by [@gerritholl](https://github.com/gerritholl))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 91](https://github.com/pytroll/trollimage/pull/91) - Fix colormaps not allowing merge points of the same value ([87](https://github.com/pytroll/trollimage/issues/87))
* [PR 83](https://github.com/pytroll/trollimage/pull/83) - Fix palettize dtype for dask arrays

#### Features added

* [PR 88](https://github.com/pytroll/trollimage/pull/88) - List supported "simple image" formats in docstrings and error message ([86](https://github.com/pytroll/trollimage/issues/86), [1345](https://github.com/pytroll/satpy/issues/1345))
* [PR 85](https://github.com/pytroll/trollimage/pull/85) - Allow customising scale and offset labels ([84](https://github.com/pytroll/trollimage/issues/84))
* [PR 82](https://github.com/pytroll/trollimage/pull/82) - Add 'inplace' keyword argument to Colormap.reverse and Colormap.set_range

In this release 5 pull requests were closed.


## Version 1.15.1 (2021/07/20)

### Pull Requests Merged

#### Bugs fixed

* [PR 81](https://github.com/pytroll/trollimage/pull/81) - Fix Colormap not being able to merge RGB and RGBA colormaps ([1658](https://github.com/pytroll/satpy/issues/1658))

In this release 1 pull request was closed.


## Version 1.15.0 (2021/03/12)

### Issues Closed

* [Issue 74](https://github.com/pytroll/trollimage/issues/74) - MNT: Stop using ci-helpers in appveyor.yml

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 78](https://github.com/pytroll/trollimage/pull/78) - Fix list stretch tags
* [PR 77](https://github.com/pytroll/trollimage/pull/77) - Remove defaults channel from ci conda environment and strict priority

#### Features added

* [PR 76](https://github.com/pytroll/trollimage/pull/76) - Add GitHub Actions for CI tests and sdist deployment
* [PR 75](https://github.com/pytroll/trollimage/pull/75) - Change XRImage.save to keep fill_value separate from valid data
* [PR 73](https://github.com/pytroll/trollimage/pull/73) - Refactor finalize method in XRImage class

In this release 5 pull requests were closed.


## Version 1.14.0 (2020/09/18)


### Pull Requests Merged

#### Bugs fixed

* [PR 71](https://github.com/pytroll/trollimage/pull/71) - Fix tiff tag writing if start_time is None

#### Features added

* [PR 70](https://github.com/pytroll/trollimage/pull/70) - Implement colorize for dask arrays

In this release 2 pull requests were closed.


## Version 1.13.0 (2020/06/08)

### Issues Closed

* [Issue 65](https://github.com/pytroll/trollimage/issues/65) - Writing to file.tiff raises KeyError ([PR 69](https://github.com/pytroll/trollimage/pull/69))
* [Issue 61](https://github.com/pytroll/trollimage/issues/61) - Add rasterio overview resampling argument ([PR 67](https://github.com/pytroll/trollimage/pull/67))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 68](https://github.com/pytroll/trollimage/pull/68) - Add workaround for broken aggdraw.Font usage in satpy/pycoast

#### Features added

* [PR 69](https://github.com/pytroll/trollimage/pull/69) - Add .tiff as recognized geotiff extension ([65](https://github.com/pytroll/trollimage/issues/65))
* [PR 67](https://github.com/pytroll/trollimage/pull/67) - Add option for geotiff overview resampling and auto-levels ([61](https://github.com/pytroll/trollimage/issues/61))
* [PR 66](https://github.com/pytroll/trollimage/pull/66) - Add more helpful error message when saving JPEG with alpha band

In this release 4 pull requests were closed.


## Version 1.12.0 (2020/03/02)

### Pull Requests Merged

#### Bugs fixed

* [PR 64](https://github.com/pytroll/trollimage/pull/64) - Add long description for display by PyPI
* [PR 63](https://github.com/pytroll/trollimage/pull/63) - Fix XRImage producing read-only data arrays and switch to pytest

In this release 2 pull requests were closed.

## Version 1.11.0 (2019/10/24)

### Pull Requests Merged

#### Bugs fixed

* [PR 58](https://github.com/pytroll/trollimage/pull/58) - Make tags containing values to compute use store for saving

#### Features added

* [PR 60](https://github.com/pytroll/trollimage/pull/60) - Add tests on py 3.7
* [PR 59](https://github.com/pytroll/trollimage/pull/59) - Add scale and offset inclusion utility when rio saving
* [PR 57](https://github.com/pytroll/trollimage/pull/57) - Add the `apply_pil` method

In this release 4 pull requests were closed.


## Version 1.10.1 (2019/09/26)

### Pull Requests Merged

#### Bugs fixed

* [PR 56](https://github.com/pytroll/trollimage/pull/56) - Fix WKT version used to convert CRS to GeoTIFF CRS

In this release 1 pull request was closed.


## Version 1.10.0 (2019/09/20)

### Pull Requests Merged

#### Bugs fixed

* [PR 53](https://github.com/pytroll/trollimage/pull/53) - Fix double format passing in saving functions

#### Features added

* [PR 55](https://github.com/pytroll/trollimage/pull/55) - Add enhancement-history to the image
* [PR 54](https://github.com/pytroll/trollimage/pull/54) - Add ability to use AreaDefinitions new "crs" property
* [PR 52](https://github.com/pytroll/trollimage/pull/52) - Add 'colors' and 'values' keyword arguments to Colormap

In this release 4 pull requests were closed.


## Version 1.9.0 (2019/06/18)

### Pull Requests Merged

#### Bugs fixed

* [PR 51](https://github.com/pytroll/trollimage/pull/51) - Fix _FillValue not being respected when converting to alpha image

#### Features added

* [PR 49](https://github.com/pytroll/trollimage/pull/49) - Add a new method for image stacking.

In this release 2 pull requests were closed.


## Version 1.8.0 (2019/05/10)

### Issues Closed

* [Issue 45](https://github.com/pytroll/trollimage/issues/45) - img.stretch gives TypeError where img.data is xarray.DataArray and img.data.data is a dask.array

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 47](https://github.com/pytroll/trollimage/pull/47) - Fix xrimage palettize and colorize delaying internal functions

#### Features added

* [PR 46](https://github.com/pytroll/trollimage/pull/46) - Implement blend method for XRImage class

In this release 2 pull requests were closed.


## Version 1.7.0 (2019/02/28)

### Issues Closed

* [Issue 27](https://github.com/pytroll/trollimage/issues/27) - Add "overviews" to save options
* [Issue 5](https://github.com/pytroll/trollimage/issues/5) - Add alpha channel to Colormaps

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 42](https://github.com/pytroll/trollimage/pull/42) - Fix stretch_linear to be dask serializable
* [PR 41](https://github.com/pytroll/trollimage/pull/41) - Refactor XRImage pil_save to be serializable

#### Features added

* [PR 44](https://github.com/pytroll/trollimage/pull/44) - Add support for adding overviews to rasterio-managed files
* [PR 43](https://github.com/pytroll/trollimage/pull/43) - Add support for jpeg2000 writing
* [PR 40](https://github.com/pytroll/trollimage/pull/40) - Modify colorize routine to allow colorizing using colormaps with alpha channel
* [PR 39](https://github.com/pytroll/trollimage/pull/39) - Add 'keep_palette' keyword argument 'XRImage.save' to prevent P -> RGB conversion on save
* [PR 36](https://github.com/pytroll/trollimage/pull/36) - Add support for saving gcps

In this release 7 pull requests were closed.


## Version 1.6.3 (2018/12/20)

### Pull Requests Merged

#### Bugs fixed

* [PR 38](https://github.com/pytroll/trollimage/pull/38) - Fix casting and scaling float arrays in finalize

In this release 1 pull request was closed.


## Version 1.6.2 (2018/12/20)

### Pull Requests Merged

#### Bugs fixed

* [PR 37](https://github.com/pytroll/trollimage/pull/37) - Fix and cleanup alpha and fill value handling in XRImage finalize
* [PR 35](https://github.com/pytroll/trollimage/pull/35) - Fix xrimage alpha creation in finalize

In this release 2 pull requests were closed.


## Version 1.6.1 (2018/12/19)

### Pull Requests Merged

#### Bugs fixed

* [PR 34](https://github.com/pytroll/trollimage/pull/34) - Fix XRImage's finalize method when input data are integers

In this release 1 pull request was closed.


## Version 1.6.0 (2018/12/18)

### Issues Closed

* [Issue 30](https://github.com/pytroll/trollimage/issues/30) - ReadTheDoc page builds not up to date ([PR 32](https://github.com/pytroll/trollimage/pull/32))
* [Issue 21](https://github.com/pytroll/trollimage/issues/21) - Add 'convert' method to XRImage

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 33](https://github.com/pytroll/trollimage/pull/33) - Fix crude stretch not calculating min/max per-band

#### Features added

* [PR 28](https://github.com/pytroll/trollimage/pull/28) - Add convert method of XRImage

In this release 2 pull requests were closed.

## Previous Versions

See changelog.rst for previous release notes.
