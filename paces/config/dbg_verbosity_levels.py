"""
Debug verbosity levels used throughout the program.

These values may be modified to customize the default debugging verbosity levels,
but the main parameter that governs the verbosity of a single calculation is the
debug_verb argument passed to a subclass of HamiltonianFramework upon instantiation.

During a calculation, the value of debug_verb is compared to the levels given in this file:
Therefore, setting a level in this file to a larger number will cause it to *stop* being printed.
"""

###############################################################
# General logging on the progress of the calculation:
###############################################################

# Print information on initialization, etc. This will only print at the very beginning of a calc:
INIT_VERBOSITY_LEVEL            = 0

# Print the basic checkpoints at each timestep
# (e.g., determining next Hilbert space, applying U(δt), computing observables etc.):
BASE_STEP_LEVEL                 = 1

# Print detailed data on each evolution of the Hilbert space:
HILBERT_SPACE_LEVEL             = 2

# Print even more detailed data on each evolution of the Hilbert space:
HILBERT_SPACE_DETAILED_LEVEL    = 4

# Print the status of individual observables calculations:
OBSERVABLES_LEVEL               = 3

###############################################################
# "True" technical debugging info:
###############################################################

# Print the behavior of the norm at each iteration of the series expansion of U(δt):
EXPM_EVO_LEVEL                  = 6

# Print information on memory use:
MEM_INFO_LEVEL                  = 5

# Print information on inter-device memory transfer times:
MOVE_DATA_TIMING_LEVEL          = 5

# Print information on the duration of the applications of searchsorted:
SEARCHSORTED_TIMING_LEVEL       = 7
