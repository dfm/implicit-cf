#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from setuptools import setup, Extension

ext = Extension("icf._icf", ["icf/_icf.c"],
                include_dirs=[numpy.get_include(), "icf"],
                libraries=["blas", "lapack"])

setup(
    name="icf",
    ext_modules=[ext],
)
