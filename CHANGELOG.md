## Version 1.22.0 (2023/11/23)


### Pull Requests Merged

#### Bugs fixed

* [PR 145](https://github.com/pytroll/trollimage/pull/145) - Do not apply linear stretch to alpha band

#### Features added

* [PR 151](https://github.com/pytroll/trollimage/pull/151) - Preserve dtypes in XRImage "enhancements"
* [PR 150](https://github.com/pytroll/trollimage/pull/150) - Keep the original dtype of the data when stretching
* [PR 141](https://github.com/pytroll/trollimage/pull/141) - Update colorbrew colormaps to be more accurate

In this release 4 pull requests were closed.


## Version 1.21.0 (2023/09/04)

### Issues Closed

* [Issue 106](https://github.com/pytroll/trollimage/issues/106) - Colorspace conversions are invalid or out of date
* [Issue 79](https://github.com/pytroll/trollimage/issues/79) - hcl2rgb bottleneck  in colorize methods ([PR 122](https://github.com/pytroll/trollimage/pull/122) by [@djhoese](https://github.com/djhoese))

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 124](https://github.com/pytroll/trollimage/pull/124) - Fix the brbg colormap

#### Features added

* [PR 122](https://github.com/pytroll/trollimage/pull/122) - Switch to new Cython-based RGB to CIE LCH_ab conversion for colorizing ([79](https://github.com/pytroll/trollimage/issues/79), [121](https://github.com/pytroll/trollimage/issues/121))

#### Documentation changes

* [PR 137](https://github.com/pytroll/trollimage/pull/137) - Add dynamic colormap generation to docs

In this release 3 pull requests were closed.


## Version 1.20.1 (2023/02/03)

### Pull Requests Merged

#### Bugs fixed

* [PR 123](https://github.com/pytroll/trollimage/pull/123) - Don't scale colormap values if they're ints ([2376](https://github.com/pytroll/satpy/issues/2376))

#### Features added

* [PR 120](https://github.com/pytroll/trollimage/pull/120) - [Snyk] Security upgrade setuptools from 39.0.1 to 65.5.1

In this release 2 pull requests were closed.


## Version 1.20.0 (2023/01/11)


### Pull Requests Merged

#### Bugs fixed

* [PR 118](https://github.com/pytroll/trollimage/pull/118) - Fix image colorization ([26](https://github.com/pytroll/pydecorate/issues/26))

#### Features added

* [PR 117](https://github.com/pytroll/trollimage/pull/117) - Refactor colormap creation ([2308](https://github.com/pytroll/satpy/issues/2308))

#### Documentation changes

* [PR 110](https://github.com/pytroll/trollimage/pull/110) - Add readthedocs config file to force newer dependencies

In this release 3 pull requests were closed.


## Version 1.19.0 (2022/10/21)

### Issues Closed

* [Issue 72](https://github.com/pytroll/trollimage/issues/72) - Add install instructions ([PR 105](https://github.com/pytroll/trollimage/pull/105) by [@djhoese](https://github.com/djhoese))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 109](https://github.com/pytroll/trollimage/pull/109) - Fix XRImage.colorize to mask integer data with _FillValue ([545](https://github.com/ssec/polar2grid/issues/545))
* [PR 108](https://github.com/pytroll/trollimage/pull/108) - Fix typo in rasterio AreaDefinition handling

#### Features added

* [PR 107](https://github.com/pytroll/trollimage/pull/107) - Refactor rasterio usage to improve import time

#### Documentation changes

* [PR 105](https://github.com/pytroll/trollimage/pull/105) - Add installation instructions ([72](https://github.com/pytroll/trollimage/issues/72))

In this release 4 pull requests were closed.


## Version 1.18.3 (2022/03/07)

### Pull Requests Merged

#### Bugs fixed

* [PR 104](https://github.com/pytroll/trollimage/pull/104) - Set scale/offset tags to (NaN, NaN) for incompatible enhancements

In this release 1 pull request was closed.


## Version 1.18.2 (2022/03/04)

### Pull Requests Merged

#### Bugs fixed

* [PR 103](https://github.com/pytroll/trollimage/pull/103) - Fix geotiff scale/offset failing for non-linear enhancements

In this release 1 pull request was closed.


## Version 1.18.1 (2022/02/28)

### Pull Requests Merged

#### Bugs fixed

* [PR 102](https://github.com/pytroll/trollimage/pull/102) - Fix enhancement_history not working with keep_palette=True

In this release 1 pull request was closed.


## Version 1.18.0 (2022/02/24)

### Pull Requests Merged

#### Features added

* [PR 101](https://github.com/pytroll/trollimage/pull/101) - Add to_csv/from_csv to Colormap
* [PR 100](https://github.com/pytroll/trollimage/pull/100) - Update Colormap.set_range to support flipped values
* [PR 99](https://github.com/pytroll/trollimage/pull/99) - Add colorize and palettize to XRImage enhancement_history
* [PR 98](https://github.com/pytroll/trollimage/pull/98) - Change tested Python versions to 3.8, 3.9 and 3.10

In this release 4 pull requests were closed.


## Version 1.17.0 (2021/12/07)

### Issues Closed

* [Issue 93](https://github.com/pytroll/trollimage/issues/93) - Add support for Cloud Optimized GeoTIFF ([PR 94](https://github.com/pytroll/trollimage/pull/94) by [@howff](https://github.com/howff))

In this release 1 issue was closed.

### Pull Requests Merged

#### Features added

* [PR 97](https://github.com/pytroll/trollimage/pull/97) - Improve 'log' stretching with static limits and choosing log base
* [PR 94](https://github.com/pytroll/trollimage/pull/94) - Use COG driver to write cloud optimized GeoTIFF ([93](https://github.com/pytroll/trollimage/issues/93))

In this release 2 pull requests were closed.


## Version 1.16.1 (2021/11/17)

### Pull Requests Merged

#### Bugs fixed

* [PR 96](https://github.com/pytroll/trollimage/pull/96) - Fix XRImage reopening geotiff with the incorrect mode
* [PR 95](https://github.com/pytroll/trollimage/pull/95) - Fix image format checking in special cases

In this release 2 pull requests were closed.


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
