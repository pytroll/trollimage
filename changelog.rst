Changelog
=========


v1.3.0 (2018-03-05)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 1.2.1 → 1.3.0. [davidh-ssec]
- Merge pull request #8 from pytroll/feature-float-geotiffs. [David
  Hoese]

  Feature float geotiffs
- Fix xarray warnings about using contains with coords. [davidh-ssec]
- Change xrimage to not modify user provided alpha band. [davidh-ssec]
- Fix line too long in xrimage. [davidh-ssec]
- Add float geotiff writing to rio_save. [davidh-ssec]
- Fix left over hack from tests. [davidh-ssec]
- Add colorize and palettize to xrimage. [davidh-ssec]
- Add dimension checks to XRImage. [davidh-ssec]


v1.2.1 (2018-03-02)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 1.2.0 → 1.2.1. [Martin Raspaud]
- Add test for bugfix on crude stretch using ints. [Martin Raspaud]
- Style cleanup and docstrings for XRImage. [Martin Raspaud]
- Bugfix crude stretch when kwargs are ints. [Martin Raspaud]


v1.2.0 (2018-03-01)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 1.1.0 → 1.2.0. [davidh-ssec]
- Merge pull request #4 from pytroll/feature-xarray-support. [David
  Hoese]

  Add XArray DataArray support via XRImage
- Update logarithmic stretch to work with xarray. [davidh-ssec]
- Fix histogram stretch in XRImage. [davidh-ssec]
- Clean up XRImage tests. [davidh-ssec]
- Do not dump data after linear stretch computation. [Martin Raspaud]
- Pass extra format keywords to the underlying writing lib. [Martin
  Raspaud]
- Add compression and nodata to geotiff. [Martin Raspaud]
- Clean up. [Martin Raspaud]
- Do not keep data in memory after computing a linear stretch. [Martin
  Raspaud]
- Use pillow for saving images other than tif. [Martin Raspaud]
- Force copying of xarray structure so original data shouldn't change.
  [davidh-ssec]

  Not sure if this applies to numpy arrays but it seems to work for dask.

- Add better handling of failing to generate a geotiff geotransform.
  [davidh-ssec]
- Add workaround for rasterio 0.36.0. [davidh-ssec]

  Color interpretation set is not supported. We will have to depend on the
  defaults.

- Use dimension names to get the shape of the image. [Martin Raspaud]
- Fix XRImage to write to the proper band/channel index. [davidh-ssec]
- Add toolz to installation in travis. [Martin Raspaud]
- Fix rasterio version for travis. [Martin Raspaud]
- Add gdal-dev for rasterio installation on travis. [Martin Raspaud]
- Add a few dependencies to travis for testing. [Martin Raspaud]
- Remove duplicated code. [Martin Raspaud]
- Merge branch 'develop' into feature-xarray-support. [Martin Raspaud]
- Merge pull request #7 from pytroll/jpeg_does_not_support_transparency.
  [David Hoese]

  Check for format=jpeg and set fill_value to zero if not set and print…
- Less verbose on debug message when saving to jpeg. [Adam.Dybbroe]
- Pep8: Update keyword arguments using "{}.update()" instead of
  iterating over members. [Adam.Dybbroe]
- Combine if statement and only make a debug info when trying to save an
  LA mode image as jpeg. [Adam.Dybbroe]
- Set fill_value to a list of four zeros, so it also works for RGBs!
  [Adam.Dybbroe]
- Make pep8/pylint/flake happy. [Adam.Dybbroe]
- Check for format=jpeg and set fill_value to zero if not set and print
  warning. [Adam.Dybbroe]
- Move XRImage to it's own module. [Martin Raspaud]
- More work on xarray support. [Martin Raspaud]
- Start working on trollimage for xarrays. [Martin Raspaud]


v1.1.0 (2017-12-11)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 1.0.2 → 1.1.0. [Martin Raspaud]
- Add github templates. [Martin Raspaud]
- Merge pull request #3 from pytroll/feature-python3. [Martin Raspaud]

  Add support for python 3
- Add support for python 3. [Martin Raspaud]
- Do not change channels if linear stretch is not possible. [Martin
  Raspaud]


v1.0.2 (2016-10-27)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 1.0.1 → 1.0.2. [Martin Raspaud]
- Merge branch 'release-v1.0.1' [Martin Raspaud]
- Fix Numpy requirement inconsistency. [Adam.Dybbroe]

  trollimage now requires Numpy 1.6 or newer. The percentile function which
  is used was introduced in 1.5.x and not available in 1.4



v1.0.1 (2016-10-27)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 1.0.0 → 1.0.1. [Martin Raspaud]
- Add bump and changelog config files. [Martin Raspaud]
- Round data instead of truncation when saving to ints. [Martin Raspaud]


v1.0.0 (2015-12-14)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.4.0 → 1.0.0. [Martin Raspaud]
- Change development status to stable. [Martin Raspaud]
- Fix version file to just provide one string. [Martin Raspaud]
- Adapt to python3. [Martin Raspaud]
- Pep8 cleanup. [Martin Raspaud]
- Fix image inversion. (don't just negate the values !) [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Ipython wants a string... [Martin Raspaud]
- Avoid directory creation for image saving unless the filename is a
  path. [Martin Raspaud]
- Bugfix ipython inline display. [Martin Raspaud]
- Add support for ipython inline images. [Martin Raspaud]
- Add notifications to slack from travis. [Martin Raspaud]
- Fix gamma and invert tests. [Martin Raspaud]
- Small fixes. [Martin Raspaud]
- Allow stretch parameters in the enhance function. [Martin Raspaud]
- Fix travis for new repo place and containers. [Martin Raspaud]
- Fix unittests hopefully. [Martin Raspaud]
- Support alpha in colorize. [Martin Raspaud]
- Accept and ignore other kwargs in enhance. [Martin Raspaud]
- Add an explicit copy kwarg. [Martin Raspaud]
- Fix broken link in documentation. [Martin Raspaud]
- Adding setup.cfg for easy rpm generation. [Martin Raspaud]
- Add thumbnail capability to saving. [Martin Raspaud]
- For PNG files, geo_image.tags will be saved a PNG metadata. [Lars Orum
  Rasmussen]


v0.4.0 (2014-09-30)
-------------------
- Bump up version number. [Martin Raspaud]
- Ignore sphinx builds. [Martin Raspaud]
- Correct unittests for new stretch behaviour. [Martin Raspaud]
- More cleanup. [Martin Raspaud]
- Cleanup image.py. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Fix stretch, so that alpha channel doesn't get stretched... [Martin
  Raspaud]
- Change the title in README.rst. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Reshape the README. [Martin Raspaud]
- Support 16 bits images. [Martin Raspaud]
- Use global version number in documentation. [Martin Raspaud]
- Cleanup. [Martin Raspaud]


v0.3.0 (2013-12-13)
-------------------
- Bump up version number. [Martin Raspaud]
- Paletize is now spelled palettize. [Martin Raspaud]
- Fixed gitignore for emacs backups. [Martin Raspaud]
- Added qualitative palettes and a palettebar generator. [Martin
  Raspaud]
- Adding a qualitative colormap and a palette example. [Martin Raspaud]
- New badges. [Martin Raspaud]


v0.2.0 (2013-12-04)
-------------------
- Add travis-ci deploy. [Martin Raspaud]
- Bump up version number. [Martin Raspaud]
- Added test for inverted set_range (colormap) [Martin Raspaud]
- Testing colormap. [Martin Raspaud]
- Bugfixes in colormap. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Test for colormap. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Adding badges. [Martin Raspaud]
- Add test coverage computation. [Martin Raspaud]
- Reorganize tests in a tests directory. [Martin Raspaud]
- Do not test build for python 2.4 and 2.5. [Martin Raspaud]
- Pillow importing bugfix. [Martin Raspaud]
- Add pillow as a dependency. [Martin Raspaud]
- Unit tests for image. [Martin Raspaud]
- Support for travis-ci. [Martin Raspaud]
- Bugfix paletize. [Martin Raspaud]
- Added the paletize functionnality. [Martin Raspaud]
- More documentation. [Martin Raspaud]
- Add an image on the home page. [Martin Raspaud]
- Fixed documentation. [Martin Raspaud]
- Documentation enhancement. [Martin Raspaud]
- Added the set_range method to colormaps and fixed the colorbar
  function. [Martin Raspaud]
- Improved documentation. [Martin Raspaud]
- Added the colorbar function. [Martin Raspaud]
- Added default colormaps. [Martin Raspaud]
- Enhancements to colormap class. [Martin Raspaud]

   * __add__
   * reverse

- Added documentation to colormap. [Martin Raspaud]
- Unwrap hue when interpolating. [Martin Raspaud]
- Change development status to beta. [Martin Raspaud]
- Add documentation. [Martin Raspaud]
- Add alpha blending to image. [Martin Raspaud]
- Add colorization to image. [Martin Raspaud]
- Copied over image.py from mpop. [Martin Raspaud]
- Fix gitignore. [Martin Raspaud]
- Administrative stuff: added setup, __init__ and version. [Martin
  Raspaud]
- Don't show ~ files. [Martin Raspaud]
- Split between colorspaces and colormap stuff. [Martin Raspaud]
- Initial commit. [Martin Raspaud]
- Initial commit. [Martin Raspaud]


