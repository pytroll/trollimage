============
Installation
============

Trollimage is available from conda-forge (via conda), PyPI (via pip), or from
source (via pip+git). The below instructions show how to install stable
versions of trollimage or from the source code.

Conda-based Installation
========================

Trollimage can be installed into a conda environment by installing the package
from the conda-forge channel. If you do not already have access to a conda
installation, we recommend installing
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ for the smallest
and easiest installation.

The commands below will use ``-c conda-forge`` to make sure packages are
downloaded from the conda-forge channel. Alternatively, you can tell conda
to always use conda-forge by running:

.. code-block:: bash

    $ conda config --add channels conda-forge

In a new conda environment
--------------------------

We recommend creating a separate environment for your work with trollimage. To
create a new environment and install trollimage all in one command you can
run:

.. code-block:: bash

    $ conda create -c conda-forge -n my_trollimage_env python trollimage

You must then activate the environment so any future python or
conda commands will use this environment.

.. code-block::

    $ conda activate my_trollimage_env

This method of creating an environment with trollimage (and optionally other
packages) installed can generally be created faster than creating an
environment and then later installing trollimage and other packages (see the
section below).

In an existing environment
--------------------------

.. note::

    It is recommended that when first exploring trollimage, you create a new
    environment specifically for this rather than modifying one used for
    other work.

If you already have a conda environment, it is activated, and would like to
install trollimage into it, run the following:

.. code-block:: bash

    $ conda install -c conda-forge trollimage

Pip-based Installation
======================

Trollimage is available from the Python Packaging Index (PyPI). A sandbox
environment for ``trollimage`` can be created using
`Virtualenv <http://pypi.python.org/pypi/virtualenv>`_.

To install the `trollimage` package and the minimum amount of python dependencies:

.. code-block:: bash

    $ pip install trollimage

Install from source
===================

To install directly from github into an existing environment (pip or conda-based):

.. code-block:: bash

    $ pip install git+https://github.com/pytroll/trollimage.git

If you have the ``git`` command installed this will automatically download the
source code and install it into the current environment. If you would like to
modify a local copy of trollimage and see the effects immediately in your
environment use the below command instead. This command should be run from the
root your cloned version of the git repository (where the ``setup.py`` is located):

.. code-block::

    $ pip install -e .
