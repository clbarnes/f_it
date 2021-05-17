"""
# f_it package
"""
from .version import version as __version__  # noqa: F401
from .version import version_tuple as __version_info__  # noqa: F401

from .fit import FIt

__all__ = ["FIt"]
