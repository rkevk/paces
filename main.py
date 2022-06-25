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


debug_verb      = 6                 # verbosity, useful for debugging/diagnostics
verbose         = True              # general verbosity, useful for keeping track of the progress

###############################
# Hilbert space parameters:
nchain          = 9
max_HO_dim      = 128               # the actual input value is nchain times this, computed below
periodic        = False

###############################
# Hamiltonian parameters:
param_dict = {
    "diag":     dict(
        eps_sys     = -4.0,
        hbar_omega  = 1.0,
        delta_eps   = 1.0,
        ),
    "hopping":  dict(
        J   = -1.0,
        ),
    "coupling": dict(
        g   = 4.0,
        ),
    }

###############################
# Initial state position:
initpos     = nchain//2
# Percentage of total available states to
# occupy with the initial basis set:
fillfac     = 0.3

###############################
# general time_evolution parameters:
te_dict = dict(
    maxstates           = int(7.5e6), #int(13e6),
    shuffle_seed        = None,
    U_weighting_method  = "coherence",
    m_star              = 100,
    )

prefix = "pos" if param_dict["hopping"]["J"] > 0 else "neg"
dirname = "../adaptive_results/vib_gendatseg/%st_n%i_d%i_g%02i_coherence" % (prefix, nchain, max_HO_dim, int(param_dict["coupling"]['g']*10))

###############################
# generate_timeline parameters:
timeline_params = dict(
    t_array             = numpy.arange(0.00, 50.05, 0.05),
    observables         = ["n_b", "H", "diagnostics"],
    use_U_weight_delta_t= 0.5,
    save_every          = 200,
    )

###############################
# Remaining technical parameters,
# these should most likely be left unchanged:

# technical parameters of HilbertSkeleton
tech_dict   = dict(
    use_complex_type    = numpy.complex128,
    search_mindiff      = 32,
    wordsize            = 32
    )

# technical parameters of generate_timeline
timeline_tech_params =  dict(
    enlarge_steps   = 1,
    save_first      = True,
    save_last       = True,
    garbage_tol     = -1,
    dm_mindiff      = 32,
    )

# End of parameters, start of calculations
###################################################################################################################################################################################

with vector_device:
    ##############################################################
    # Generate the fundamental Hilbert space:
    HamObj  = HamiltonianObject(
                nchain=nchain, max_HO_dims=[max_HO_dim,]*nchain, periodic=periodic,
                **tech_dict,
                use_terms=param_dict)

    ##############################################################
    # Instantiate the general time_evolution object:
    te	    = time_evolution(HamObj, 
                dirname=dirname, verbose=verbose, debug_verb=debug_verb, 
                **te_dict)

    ##############################################################
    # Create an initial basis set and initial vector:

    te.create_nonuniform_basis([1,] * ((nchain-9)//2) + [1,2,4,15,50,15,4,2,1] + [1,] * ((nchain-9)//2), minpos=max(initpos-25, 0), maxpos=min(initpos+25, nchain))
#    te.create_nonuniform_basis([1,]*nchain, minpos=initpos, maxpos=initpos+1)
#    te.grow_optimal_basis(fillfac=fillfac)
    te.create_initial_vector(vector_coeffs=[1], vector_coo=[[initpos,] + nchain*[0,]], auto_normalize=True)

    ##############################################################
    # Calculate the actual timeline:
    te.generate_timeline(**timeline_params, **timeline_tech_params)




