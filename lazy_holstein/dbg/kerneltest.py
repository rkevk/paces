#!/usr/bin/env python3
# coding: utf-8

from tqdm import tqdm 
import timeit
import time
import numpy

import cupy

import sys
import os

from lazy_holstein_gendatseg import *

import cupy_gendatseg_kernels as cgds

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))

###################################################################################################################################################################################

nchain  = 75
HO_dim  = 12
g       = 2.0

HamObj  = lazy_holstein_model(nchain=nchain, max_HO_dims=[HO_dim,]*nchain, eps_sys=1.0, t_sys=-1.0, eps_bath=1.0, delta_eps=0.0, coupling_g=float(g), maxstates=int(12e6), periodic=False, search_mindiff=32, wordsize=32)

te	= time_evolution(HamObj, dirname="sandbox", verbose=True, m_star=100, debug_verb=6, shuffle_seed=0)

initpos     = nchain//2
centervals  = [10,]*7
te.create_nonuniform_basis([1,] * ((nchain-len(centervals))//2) + centervals + [1,] * ((nchain-len(centervals))//2), minpos=max(initpos, 0), maxpos=min(initpos+1, nchain))
#te.create_nonuniform_basis([1,]*nchain, minpos=initpos, maxpos=initpos+1)
#te.grow_optimal_basis(fillfac=1.0)

##########################################################################################

print(te.numstates)

#newres  = cupy.zeros(nchain)
def newfunc(vec):
    weights = cupy.abs(vec)**2
    return cgds.calculate_bath_n_b(weights, te.whoami, HamObj.posbitwidth, HamObj.QHObitwidth_v, wordsize=32)

for i in range(20):
    vector              = cupy.random.random(te.numstates)
    vector[int(1e4):]   = 0
    vector              /= cupy.linalg.norm(vector)
    print(cupy.linalg.norm(vector))

    oldres  = te.calculate_bath_n_b(vector)

    newres  = newfunc(vector)
#    print(oldres)
#    print(newres)
    diff    = cupy.abs(oldres - newres).sum()
    print(diff)
    assert(diff < 1e-13)
    print()

#print(cupyx.profiler.benchmark(te.calculate_bath_n_b, (vector,), n_repeat=10))
#print(cupyx.profiler.benchmark(newfunc, (vector,), n_repeat=10))

#assert(cupy.all(fm_new_lc == fm_old8))
#assert(cupy.all(fm_new == fm_old32))

##########################################################################################

#print(cupyx.profiler.benchmark(func_old8, (phonebook, findme), {"allow_escapes": ae}, n_repeat=10))
#print(cupyx.profiler.benchmark(func_old32, (phonebook, findme), {"allow_escapes": ae}, n_repeat=10))
#print(cupyx.profiler.benchmark(func_new, (phonebook, findme), {"allow_escapes": ae, "left_check": False}, n_repeat=10))
#print(cupyx.profiler.benchmark(func_new, (phonebook, findme), {"allow_escapes": ae, "left_check": True}, n_repeat=10))
#print(cupyx.profiler.benchmark(func_new_fb, (phonebook, findme), {"allow_escapes": ae}, n_repeat=10))
