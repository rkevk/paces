"""Helper functions that may be of use across different situations."""

import numpy
import cupy     # pylint: disable=import-error

from ..config import SEARCHSORTED_TIMING_LEVEL

####################################################################################################

def cartesian_product(*arrays):
    """Return the Cartesian product of a list of arrays."""
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def print_searchsorted_timing(verb, delta_t, size1, size2):
    """Helper function to print the time it took to apply searchsorted."""
    if verb > SEARCHSORTED_TIMING_LEVEL:
        print("This application of searchsorted took"
                f" {delta_t*1000} ms (arg sizes {size1}, {size2}).")

def obs_attrs(**kwargs):
    """
    Decorator to add header and fname to observable functions.

    This should be called with at least the following kwargs:
        fname (str): Base name of the file where the corresponding observable expectation values
            should be saved. The entire path will be derived from this fname as
            [main_calc_dir]/observables/[fname].real and [main_calc_dir]/observables/[fname].imag.
        header (str): The header line to be printed at the top of the file, but without the
            preceding time and norm (i.e., just a description of the observables themselves).
    """
    def wrapper(f):
        for attr, val in kwargs.items():
            setattr(f, attr, val)
        return f
    return wrapper


def debug_lister(dbg_list):
    """
    Decorator to add debug description to generate_mel_xxx

    Args:
        dbg_list (list): list of strings providing names for the debug output.
    """
    def wrapper(f):
        f.dbg_list = dbg_list
        return f
    return wrapper

def flatten_dbg_dict(d):
    """Flatten the dbg dicts (header or vals) into a list"""
    return [x for l in [d[k] for k in d] for x in l]


def cupy_unique(array):
    """
    Find and return the unique, lexicographically sorted rows in an array.

    Replacement for numpy.unique with option axis=0.
    """
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr     = array[cupy.lexsort(array.T[::-1])]
    mask        = cupy.empty(array.shape[0], dtype=cupy.bool_)
    mask[0]     = True
    mask[1:]    = cupy.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]
