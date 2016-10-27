Changelog
=========

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


