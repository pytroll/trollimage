"""Setup for trollimage."""
import sys
from typing import Any

from setuptools import setup, find_packages
import versioneer
import numpy as np
from Cython.Build import build_ext
from Cython.Distutils import Extension

if sys.platform.startswith("win"):
    extra_compile_args = []
else:
    extra_compile_args = ["-O3"]

EXTENSIONS = [
    Extension(
        'trollimage._colorspaces',
        sources=['trollimage/_colorspaces.pyx'],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
    ),
]

cython_directives: dict[str, Any] = {
    "language_level": "3",
    "freethreading_compatible": True,
}


class CythonCoverageBuildExtCommand(build_ext):
    """Simple command extension to add Cython coverage flag.

    With this class included in the build we are able to pass
    ``--cython-coverage`` to compile Cython modules with flags necessary to
    report test coverage.

    """

    user_options = build_ext.user_options + [
        ('cython-coverage', None, None),
    ]

    def initialize_options(self):
        """Initialize command line flag options to default values."""
        super().initialize_options()
        self.cython_coverage = False  # noqa

    def run(self):
        """Build extensions and handle cython coverage flags."""
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_25_API_VERSION")]
        if self.cython_coverage:
            print("Enabling directives/macros for Cython coverage support")
            cython_directives.update({
                "linetrace": True,
                "profile": True,
            })
            define_macros.extend([
                ("CYTHON_TRACE", "1"),
                ("CYTHON_TRACE_NOGIL", "1"),
            ])
        for ext in EXTENSIONS:
            ext.define_macros = define_macros
            ext.cython_directives.update(cython_directives)
        super().run()


cmdclass = versioneer.get_cmdclass(cmdclass={"build_ext": CythonCoverageBuildExtCommand})

with open('README.rst', 'r') as readme_file:
    long_description = readme_file.read()

setup(name="trollimage",
      version=versioneer.get_version(),
      cmdclass=cmdclass,
      description='Pytroll imaging library',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      author='Martin Raspaud',
      author_email='martin.raspaud@smhi.se',
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering",
          "Programming Language :: Python :: Free Threading :: 1 - Unstable",
      ],
      license="Apache-2.0",
      license_files=["LICENSE.txt", "LICENSE_RIO_COLOR.txt"],
      url="https://github.com/pytroll/trollimage",
      packages=find_packages(),
      zip_safe=False,
      install_requires=['numpy>=1.25', 'pillow'],
      python_requires='>=3.11',
      extras_require={
          'geotiff': ['rasterio>=1.0'],
          'xarray': ['xarray', 'dask[array]'],
          'tests': ['xarray', 'dask[array]', 'pyproj', 'pyresample', 'pytest'],
      },
      ext_modules=EXTENSIONS,
      )
