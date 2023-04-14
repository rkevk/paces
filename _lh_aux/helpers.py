#!/usr/bin/env python3
# coding: utf-8
 
import time
import inspect
import sys
import os
import os.path
import pprint

import numpy
import cupy
import cupyx

from device_config import *

#import logging

#logging.basicConfig(filename="last_instance.log", filemode='w')

#if os.path.dirname(sys.argv[0]) != '':
#    os.chdir(os.path.dirname(sys.argv[0]))

#for mod in (cupy, numpy):
#    mod.set_printoptions(linewidth=200, edgeitems=10)

if numpy.uintc != numpy.uint32:
    raise TypeError("Well well well, who's working on a non-64 bit system? This code will explode if run on a system whose integer size is not 32 bits.")


###################################################################################################################################################################################

###################################
# global (mostly helper) functions:
###################################

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def format_function_args(frame, start_time=None):
    if start_time is None:
        start_time  = time.time()
    localtime   = time.asctime(time.localtime(start_time))
    args, _, _, values  = inspect.getargvalues(frame)
    arg_list            = [(str(i) + "=" + str(values[i])) for i in args if str(i) != "self"]
    fname               = frame.f_code.co_name
    return ("\nFunction call at %f (%s local):\n   %s(" % (start_time, localtime, fname)) + ', '.join(arg_list) + ")\n"


def print_searchsorted_timing(verb, delta_t, size1, size2):
    if verb > searchsorted_timing_level:
        print("This application of searchsorted took %f ms (arg sizes %i, %i)." % (delta_t*1000, size1, size2))

def write_params(fname, obj, itemstr):
    try:
        value   = getattr(getattr(obj, itemstr), "__name__")
    except AttributeError:
        value   = getattr(obj, itemstr)
    fname.write(itemstr + " = " + str(value) + '\n')


###################################
# replacement for numpy.unique with option axis=0
###################################
def cupy_unique(array):
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr     = array[cupy.lexsort(array.T[::-1])]
    mask        = cupy.empty(array.shape[0], dtype=cupy.bool_)
    mask[0]     = True
    mask[1:]    = cupy.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]

###################################################################################################################################################################################


