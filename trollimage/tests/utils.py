#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 trollimage developers
#
# This file is part of trollimage.
#
# trollimage is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# trollimage is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with trollimage.  If not, see <http://www.gnu.org/licenses/>.
"""Helper classes and functions for running and writing tests."""

from contextlib import contextmanager


class CustomScheduler:
    """Scheduler raising an exception if data are computed too many times."""

    def __init__(self, max_computes=1):
        """Set starting and maximum compute counts."""
        self.max_computes = max_computes
        self.total_computes = 0

    def __call__(self, dsk, keys, **kwargs):
        """Compute dask task and keep track of number of times we do so."""
        import dask
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError("Too many dask computations were scheduled: "
                               "{}".format(self.total_computes))
        return dask.get(dsk, keys, **kwargs)


@contextmanager
def assert_maximum_dask_computes(max_computes=1):
    """Context manager to make sure dask computations are not executed more than ``max_computes`` times."""
    import dask
    with dask.config.set(scheduler=CustomScheduler(max_computes=max_computes)) as new_config:
        yield new_config
