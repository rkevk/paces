"""
aux: Auxiliary functions that may be required across different situations.

The _scipy submodule is adapted from SciPy. It contains matrix-exponential functions and is subject
to the BSD-3-clause license as mentioned in the root directory of this module.
"""

__all__ = ["cupy_search", "helpers", "expm_multiply", "expm_multiply_simple"]

from ._scipy._expm_multiply import expm_multiply, expm_multiply_simple
from . import cupy_search
