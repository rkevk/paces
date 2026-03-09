"""Example calculation of a single-exciton 1D Holstein chain system."""

import sys
import os

import numpy

from paces.config import devices
from paces.models import holstein
from paces.core.time_evolution import CoeffSaveParams, ExpmParams, TimeEvoParams

# This makes relative paths refer to the location of this script:
if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))


####################################################################################################
# "Master" settings
####################################################################################################

# configure device and memory settings; the defaults will select a single GPU for everything:
devices.configure()

###############################
# debugging verbosity (for printing to stdout):
debug_verb = 6

###############################
# directory in which the data will be saved:
# (this directory has to be created before running this code!)
dirname = "results/example"

# name of the parameter file to save to:
params_file = os.path.join(dirname, "paces_holstein_test_params.log")


####################################################################################################
# Model-specific Hilbert-space and Hamiltonian settings
####################################################################################################

# Hilbert space parameters:
nchain          = 25        # number of sites on the Holstein chain
max_ho_dim      = 128       # we use this max. phonon mode dimension for each site

###############################
# Hamiltonian parameters:
term_param_dict = {
    "diag": dict(
        eps_sys     = -1.0, # on-site energy of an exciton (this is just a constant shift)
        hbar_omega  = 1.0,  # energy of a single phonon
        delta_eps   = 0.0,  # excitonic energy bias between neighboring sites on the chain
        ),
    "hopping": dict(
        J   = 1.0,          # excitonic hopping parameter
        ),
    "vib_coupling": dict(
        g   = 4.0,          # vibronic coupling parameter
        ),
    }

###############################
# Initial state position:
initpos = nchain//2
# Proportion of maxstates to occupy with the initial basis set:
fillfac = 0.3

###############################
# list of observables to compute:
obs_list = ["n_pho", "n_exc", "total_energy"]


####################################################################################################
# General time-evolution settings
####################################################################################################

###############################
# general parameters for the time evolution:
te_params = TimeEvoParams(
        maxstates       = int(8e6), # nominal truncation number at each timestep
        enlarge_steps   = 1,        # max. neighbor degree is enlarge_steps + 1
        diagnostics     = True,     # whether to save extra diagnostic data to file
        shuffle_seed    = 0,        # seed used for pseudo-randomized shuffling to avoid bias
    )

###############################
# how often to save the coefficients:
coeff_save_obj = CoeffSaveParams(save_every=200, save_first=True, save_last=True)

###############################
# timeline parameters:
timeline_params = {
    "t_array": numpy.arange(0.00, 5.05, 0.05),  # timesteps which will be computed
    "coeff_save_obj": coeff_save_obj,
    }

###############################
# parameters to use for the matrix exponentiation, we leave them at their defaults:
expm_params     = ExpmParams()

####################################################################################################
# End of parameter input, start of calculations
####################################################################################################

with devices.vector_dev:
    ##############################################################
    # Generate the fundamental Hilbert space:
    hamobj = holstein.Hamiltonian(
                nchain = nchain,
                max_ho_dims = [max_ho_dim,]*nchain,
                use_terms = term_param_dict,
                debug_verb = debug_verb,
                )

    ##############################################################
    # Initialize the general time_evolution object:
    te = holstein.TimeEvolution(
                hamobj,
                holstein.Observables,
                obs_list,
                dirname,
                te_params,
                expm_params,
                params_file,
                )

    ##############################################################
    # Create an initial basis set consisting of 0 phonons everywhere and a localized particle:
    te.create_nonuniform_basis([1,]*nchain, minpos=initpos, maxpos=initpos+1)

    # Enlarge the initial basis set:
    te.grow_optimal_basis(fillfac=fillfac)

    # Create the initial vector in this basis:
    te.create_initial_vector(vector_coeffs=[1], vector_coo=[[initpos,] + nchain*[0,]],
                                auto_normalize=True)

    ##############################################################
    # Calculate the actual timeline:
    te.generate_timeline(**timeline_params)
