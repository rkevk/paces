#!/usr/bin/env python3
# coding: utf-8
 
import argparse
import sys
import os

from _core import *

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))

###################################################################################################################################################################################

parser  = argparse.ArgumentParser(description="Parallelized multi-step dynamically adaptive (ParMuDA) basis set calculations of quantum dynamics")

if False:
    parser.add_argument("delta_eps", action="store",  help="delta_eps, specified as a string of three integers that are to be divided by 100 (e.g., input 075 is mapped to 0.75)")
    parser.add_argument("--save_every", action="store", default=200, nargs='?', type=int, help="save every n-th wavefunction to storage, default 200")
    parser.add_argument("--numstates", action="store", default=13e6, nargs='?', help="number of states to truncate to, default is 13e6")
    args    = parser.parse_args()
    if len(args.delta_eps) != deltadigits:
        raise ValueError("delta_eps incomprehensible: " + args.delta_eps)

# Physical system parameters:
nchain              = 25
eps_sys             = -1.0
t_sys               = 1.0
omega               = 1.0
coupling_g          = 4.0
delta_eps           = 0.0
max_HO_dim          = 128

# Initial state position:
initpos = nchain//2

# Simulation parameters:
maxstates           = int(13e6)
shuffle_seed        = 0
U_weighting_method  = "coherence"
use_U_weight_delta_t= 0.5
save_every          = 200
fillfac             = 0.3

t_array             = numpy.arange(0.00, 50.05, 0.05)
observables         = ["n_b", "H", "diagnostics"]

dirname             = "../adaptive_results/vib_gendatseg/negt_n%i_d%i_g%02i_coherence" % (nchain, max_HO_dim, int(coupling_g*10))

with vector_device:

    HamObj  = HamiltonianObject(nchain=nchain, max_HO_dims=[max_HO_dim,]*nchain, eps_sys=eps_sys, t_sys=t_sys, eps_bath=omega, delta_eps=delta_eps, coupling_g=coupling_g, maxstates=maxstates, periodic=False, search_mindiff=32, wordsize=32)

    te	= time_evolution(HamObj, dirname=dirname, verbose=True, m_star=100, debug_verb=6, shuffle_seed=shuffle_seed, U_weighting_method=U_weighting_method)

#    te.create_nonuniform_basis([1,] * ((nchain-9)//2) + [1,2,4,15,50,15,4,2,1] + [1,] * ((nchain-9)//2), minpos=max(initpos-25, 0), maxpos=min(initpos+25, nchain))
    te.create_nonuniform_basis([1,]*nchain, minpos=initpos, maxpos=initpos+1)
    te.grow_optimal_basis(fillfac=fillfac)
    te.create_initial_vector(vector_coeffs=[1], vector_coo=[[initpos,] + nchain*[0,]], auto_normalize=True)
    te.generate_timeline(t_array=t_array, observables=observables, garbage_tol=-1, dm_mindiff=32, use_U_weight_delta_t=use_U_weight_delta_t, save_first=True, save_last=True, save_every=save_every, enlarge_steps=1)




