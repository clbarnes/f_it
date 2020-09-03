# -*- coding: utf-8 -*-

"""Top-level package for f_it."""

from .fit import FIt
from .version import __version__

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"
__version_info__ = tuple(int(n) for n in __version__.split("."))

__all__ = ["FIt"]
