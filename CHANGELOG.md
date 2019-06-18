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
