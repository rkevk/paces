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
import _lh_aux.cupy_expm_multiply as cupy_expm_multiply
import _lh_aux.cupy_search as cupy_search
import _lh_aux.cupy_gendatseg_kernels as cupy_gendatseg_kernels

from ..aux.helpers import *
from ..device_config import *

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
# global variables:
###################################
# the following are debug verbosity levels, nothing critical:
searchsorted_timing_level       = 999
simple_taylor_level             = 6
move_data_timing_level          = 4
mem_info_level                  = 4


###################################
# helper functions:
###################################

def write_params(fname, obj, itemstr):
    try:
        value   = getattr(getattr(obj, itemstr), "__name__")
    except AttributeError:
        value   = getattr(obj, itemstr)
    fname.write(itemstr + " = " + str(value) + '\n')


###################################################################################################################################################################################

class TimeEvolutionFramework:
    def __init__(self, hamobj, maxstates, dirname=None, verbose=True, m_star=100, debug_verb=0, shuffle_seed=0, U_weighting_method="coherence", params_file=None):
        if verbose:
            print("Initializing time_evolution object...")
        self.hamobj     = hamobj
        self.maxstates  = maxstates
        self.dirname    = dirname
        self.verbose    = verbose
        self.m_star     = m_star                    # this is the max iteration of the Taylor approximation
        self.use_module         = self.hamobj.use_module
        self.debug_verb         = debug_verb
        self.hamobj.debug_verb  = self.debug_verb   # time_evolution debug_verb overrides hamobj debug_verb
        self.shuffle_seed       = shuffle_seed      # when determining the new Hilbert space, shuffle equal values (to avoid bias) using this value as the initial seed
        if shuffle_seed is not None:
            self.use_module.random.seed(shuffle_seed)
        self.first_order_U_importance   = getattr(self, "first_order_U_importance_" + U_weighting_method)

        self.real_valued                = all([all([value.imag == 0 for value in paramdict.values()]) for paramdict in self.hamobj.use_terms.values()])
        if not self.real_valued:
            raise NotImplementedError("Not all functions have been adapted to complex-valued off-diagonal Hamiltonian matrix elements.") # specifically, the first_order_U_importance methods

        if params_file is None:
            self.params_file    = os.path.join(self.dirname, "HamObj_params_run" + str(time.time()) + ".log")
        else:
            self.params_file    = params_file

        with open(self.params_file, "w") as HamObj_params_file:
            HamObj_params_file.write("### HilbertSkeleton parameters:\n")
            for itemstr in [i for i in vars(self.hamobj) if i != "use_terms"]:
                write_params(HamObj_params_file, self.hamobj, itemstr)

            HamObj_params_file.write("\n### HamiltonianObject parameters:\n")
            HamObj_params_file.write(pprint.pformat(self.hamobj.use_terms, width=1) + '\n')

            HamObj_params_file.write("\n### TimeEvolution parameters:\n")
            for itemstr in ("maxstates", "shuffle_seed", "first_order_U_importance", "m_star"):
                write_params(HamObj_params_file, self, itemstr)

            HamObj_params_file.write("\n### End of parameter list\n#################################################################\n")

        if verbose:
            print("Finished initialization of time_evolution object!\n")

    def write_current_params_to_file(self, frame, start_time=None):
        if start_time is None:
            start_time  = time.time()
        localtime   = time.asctime(time.localtime(start_time))
        args, _, _, values  = inspect.getargvalues(frame)
        arg_list            = [(str(i) + "=" + str(values[i])) for i in args if str(i) != "self"]
        fname               = frame.f_code.co_name
        header = ("\nFunction call at %f (%s local):\n   %s(" % (start_time, localtime, fname)) + ', '.join(arg_list) + ")\n"
        with open(self.params_file, 'a') as params_file:
            params_file.write(header)

    ###################################################################################################################################################################################

    def load_basis_from_file(self, loadfile):
        function_start_time = time.time()
        if self.verbose:
            print("Loading basis set from file...", end=' ')
            sys.stdout.flush()
        self.write_current_params_to_file(inspect.currentframe(), function_start_time)

        self.whoami     = self.use_module.load(loadfile)
        self.numstates  = self.whoami.shape[0]
        if self.whoami.shape[1] != self.hamobj.totwordsize:
            raise ValueError("Loaded basis set does not match total number of bits as given by hamobj construction.")
        if self.whoami.dtype != self.hamobj.dtype:
            raise ValueError("Loaded basis set does not match the wordsize as given by hamobj construction.")

        if self.verbose:
            print("Done!")

    def grow_optimal_basis(self, n_max=numpy.inf, fillfac=1.0):
        """
        Take a small initial basis set and grow an optimal larger one around it via Hamiltonian enlargement:

        n_max:      The maximum number of enlargement iterations to perform
        fillfac:    The proportion of maxstates to use up

        The enlargement is stopped once either of these is reached.
        """
        if self.verbose:
            print("Growing initial basis set...", end=' ')
            sys.stdout.flush()
        self.write_current_params_to_file(inspect.currentframe())

        initsize            = len(self.whoami)
        whoami              = self.whoami
        n                   = 0
        while len(whoami) <= self.maxstates * fillfac:
            previous_whoami = whoami
            n               += 1
            if n > n_max:
                break
            whoami          = self.hamobj.enlarge_basis_set(previous_whoami)

        # use previous_whoami, since the last was the one that triggered the break condition
        self.whoami         = previous_whoami
        self.numstates      = self.whoami.shape[0]
        if self.verbose:
            print("Done! Performed %i enlargements (grew from %i to %i states)." % (n, initsize, len(self.whoami)))

    ###################################
    # generate initial vector (requires an existing basis set)
    ###################################
    def create_initial_vector(self, vector_coeffs, vector_coo, auto_normalize=True):
        """
        Takes vector parameters in the format coeffs, (sys, bath) and, by looking up the positions in whoami, creates the state vector.
        """
        if self.verbose:
            print("Creating the initial state vector...", end=' ')
            sys.stdout.flush()

        try:
            vector_coo  = cupy.asnumpy(vector_coo)
        except NameError:
            vector_coo  = numpy.asarray(vector_coo)
        if type(vector_coo) is not numpy.ndarray:
            raise TypeError("Conversion of initial vector_coo to numpy.ndarray failed.")

        if vector_coo.ndim != 2:
            raise ValueError("The coo list must be 2D, even if it contains only the entry for one basis state.")
        if numpy.unique(vector_coo, axis=0).shape != vector_coo.shape:
            raise ValueError("There are duplicates in the specified vector coordinates.")
        if vector_coo.shape[0] != len(vector_coeffs):
            raise ValueError("The number of coefficients does not match the number of coordinates.")
        if auto_normalize:
            vector_coeffs /= numpy.linalg.norm(vector_coeffs)

        vector_coo      = self.hamobj.compress_states(cupy.asarray(vector_coo))

        self.vector     = self.use_module.zeros(len(self.whoami), dtype=self.hamobj.complex_type)
        for i, coo in enumerate(vector_coo):
            index               = self.use_module.all(self.whoami == self.use_module.asarray(coo), axis=1).nonzero()[0][0]
            self.vector[index]  = vector_coeffs[i]

        if self.verbose:
            print("Done!")

    ###################################
    # generate sparse matrices
    ###################################
    def create_matrices(self, diag_params, COO_dict):
        """
        Turn the COO and value arrays into actual sparse matrices (in the CSR format).
        
        Why is this a part of TimeEvolution and not the Hamiltonian objects?
        In part because it the latter never contain actual linear algebra objects,
        only said COO and value arrays.
        But, apart from that, the Hamiltonian "stuff" resides on whoami_device, whereas
        the time evolution along with all of the other linear algebra reside on vector_device.
        """
        self.sparse_mats_dict   = {term: self.use_module.sparse.coo_matrix(COO, shape=(self.numstates, self.numstates)).tocsr() for term, COO in COO_dict.items()}
        self.diag_vals          = diag_params


    def total_H(self, vector):
        """
        Apply total Hamiltonian H to a vector v and return the new vector Hv.
        Setting vector="ones" generates an everywhere-one input vector.
        """
        # diagonal terms are contained within the case distinction:
        if type(vector) is str and vector == "ones":
            B       = self.diag_vals
            vector  = self.use_module.ones(self.numstates)
        else:
            B       = self.diag_vals * vector

        # off-diagonal terms:
        for mat in self.sparse_mats_dict.values():
            B           += mat.dot(vector)
        return B


    ###################################
    # perform time evolution step
    ###################################
    def simple_taylor(self, delta_t, diagnostics=False, explosion_cutoff=1e5):
        n = self.numstates
        u_d = 2**-53
        tol = u_d
        B = self.vector
        if self.debug_verb > simple_taylor_level:
            print("Check 0.")
#        print(self.whoami[self.diag_vals.argmax()])
        if self.debug_verb > 3:
            print("  Norm prior to evolution: %f." % self.use_module.linalg.norm(B))

        converged   = False             # added
        final_m     = 0                 # added
        F = B.copy()
        if self.debug_verb > simple_taylor_level:
            print("Check 1.")
        c1 = cupy_expm_multiply.cupy_exact_inf_norm(B)
        if self.debug_verb > simple_taylor_level:
            print("Check 2.")
        for j in range(self.m_star):
            if self.debug_verb > 4:
                print("Norm of F: %f." % self.use_module.linalg.norm(F))
            final_m     += 1            # added
            coeff       = -1j * delta_t / float(j+1)
            if self.debug_verb > simple_taylor_level:
                print("Check 3.")
            B           = self.total_H(B)
            B           *= coeff
            if self.debug_verb > simple_taylor_level:
                print("Check 4.")
            c2          = cupy_expm_multiply.cupy_exact_inf_norm(B)
            if self.debug_verb > simple_taylor_level:
                print("Check 5.")
            F           += B
            c1_plus_c2  = c1 + c2
            if c1_plus_c2 <= tol * cupy_expm_multiply.cupy_exact_inf_norm(F):
                converged = True    # added
                break
            if self.debug_verb > simple_taylor_level:
                print("Check 6.")
            c1 = c2
        if self.debug_verb > simple_taylor_level:
            print("Check 7.")
        if self.debug_verb > 3:
            print("  Norm after evolution: %f." % self.use_module.linalg.norm(F))
#        B = F
        del B
        if self.use_module.linalg.norm(F) > explosion_cutoff:
            raise RuntimeError("Norm has exceeded preset explosion cutoff value (%f)!" % explosion_cutoff)
        if diagnostics:                 # added
            return F, converged, final_m, c1_plus_c2, c1_plus_c2/cupy_expm_multiply.cupy_exact_inf_norm(F)
        else:
            return F


    ###################################
    # perform entire time evolution
    ###################################
    def generate_timeline(self, t_array=numpy.arange(0, 50.25, 0.25), observables=["n_b", "H", "hopping"], save_every=None, save_first=False, save_last=False, garbage_tol=-1, dm_mindiff=32, use_U_weight_delta_t=0.0, enlarge_steps=0, use_two_streams=False):
        """
        This is the master time evolution function that evolves the system along the time-points given in t_array and calculates and saves to file the requested variables along the way.
        In order for this to work, the HamObj object must be initialized (which is a required argument for the initialization of the time_evolution object),
        and there must be an initial basis (see the initial basis set functions in the time_evolution object) and an initial vector (via te.create_initial_vector(args) or by loading from file).

        Arguments of this function:
        t_array:                float array, the initial state is assumed to correspond to the first time value in t_array. Then evolve step-by-step until the last value in t_array is reached.
        observables:            list of str, observables to compute and save. Must be a subset of the following: ["H", "hopping", "coupling", "n_b", "reduced_dm", "diagnostics"]
            Note: "diagnostics" saves non-observable but useful information about the health of the simulation, etc.
        save_every:             int, will save the wavefunction and basis set at every n-th timestep. Default is None, which disables saving (but is overridden by save_first and save_last).
        save_first:             bool, will save the initial wavefunction and basis set if True.
        save_last:              bool, will save the final wavefunction and basis set at the end of the time evolution if True.
        garbage_tol:            float, determines what value to use as a garbage cutoff when determining the Hilbert subspace evolution. Negative values disable the garbage function and keep all basis states.
            Note that using this function is generally not advisable if the initial basis set is well chosen (i.e. you should probably use a negative value)!
        dm_mindiff:             int, tells the cupy_search algorithm when to switch from a binary to a linear search (set to something between 10 and 100 for typical use).
        use_U_weight_delta_t:   float, the delta_t to use for the forward-looking part of the Hilbert subspace determination. 0 disables forward-looking.
        enlarge_steps:          int, the number of additional matrix elements to incorporate when determining the next Hilbert subspace. 0 takes only directly interacting basis states, 1 adds indirect interactions via 1 intermediate, 2 via 2 etc.


        THIS STILL NEEDS TO BE REWRITTEN.
        """
        timeline_start_time = time.time()
        if self.verbose:
            print("Setting up generate_timeline function...")
        self.write_current_params_to_file(inspect.currentframe(), timeline_start_time)

        if save_every is not None or save_first or save_last:     # boring string formatting, no physics here
            smallest_val    = numpy.min(numpy.abs(t_array))
            smallest_diff   = numpy.diff(numpy.abs(t_array)).min()
            if smallest_val == 0:
                decnum          = numpy.abs(numpy.floor(numpy.log10(smallest_diff))) + 1
            else:
                decnum          = numpy.abs(numpy.floor(numpy.log10(min(smallest_val, smallest_diff)))) + 1
            largest_val     = numpy.max(numpy.abs(t_array))
            intnum          = numpy.abs(numpy.ceil(numpy.log10(largest_val)))

            format_string   = "wf_file_{0:0%i.%if}" % (intnum + decnum + 1, decnum)
            wf_name_list    = [self.dirname + "/wf_coeffs/" + format_string.format(i) for i in t_array]

        obs_obj = Observables(observables, self.dirname)

        if vector_device != whoami_device and use_two_streams:
            with whoami_device:
                whoami_stream = cupy.cuda.Stream()

        if self.verbose:
            print("Finished generate_timeline setup! Beginning time evolution...")

        vector = self.vector
        diag_coo_debug  = None
        impstates       = None
        for i in range(len(t_array)):
            t       = t_array[i]
            delta_t = t - t_array[i-1]
            if i == 0:
                delta_t = 0        # turn off delta_t at the first timestep
            if self.debug_verb > 0:
                if t == 0:
                    print("\nCheck 1: Initiated initial timestep (no evolution in this step).")
                else:
                    print("\nCheck 1: Initiated timestep from %f to %f." % (t-delta_t, t))

            if self.debug_verb > 0:
                print("Check 2.1: Determining next Hilbert subspace.")
            if vector_device != whoami_device and use_two_streams:
                whoami_stream.synchronize()
#                prepare_results = cupy.asarray(prepare_results)
            ceiling_hits, post_adapt_norm, post_adapt_H     = self.generate_new_Hilbert_space_cupy( vector,
                                                                                                    enlarge_steps=enlarge_steps,
                                                                                                    garbage_tol=garbage_tol,
                                                                                                    do_fancy_stuff="diagnostics" in observables, 
                                                                                                    use_U_weight_function=(use_U_weight_delta_t != 0 and i != 0),
                                                                                                    delta_t=use_U_weight_delta_t,
                                                                                                    diag_coo_debug=diag_coo_debug,
                                                                                                    impstates=impstates)
#                ceiling_hits, post_adapt_norm, post_adapt_H     = self.generate_new_Hilbert_space_cupy(vector, garbage_tol=garbage_tol, do_fancy_stuff="diagnostics" in observables, use_U_weight_function=(use_U_weight_function and delta_t != 0), delta_t=0.2)
            if self.debug_verb > 0:
                print("Check 2.2: Finished determining next Hilbert subspace.")
            if t == 0:
                vector = self.vector    # this is the vector with the new whoami
            else:
                vector, expm_converged, final_m, final_expm_term, rel_error_expm = self.simple_taylor(delta_t, diagnostics=True)
            if self.debug_verb > 0:
                print("Check 3: Calculated v(t).")
            if vector_device != whoami_device and use_two_streams and i < len(t_array) - 1:
                select_whoami, impstates    = self.determine_select_whoami(vector, use_U_weight_function=(use_U_weight_delta_t != 0 and i != 0), delta_t=use_U_weight_delta_t, garbage_tol=garbage_tol)
                if self.debug_verb > 0:
                    print("Beginning concurrent calculation of next Hilbert space.")
                    with whoami_device:
                        with whoami_stream:
                            new_select      = cupy.asarray(select_whoami)
                            diag_coo_debug = self.HamObj.generate_mel(new_select, enlarge_steps=enlarge_steps)
                    del select_whoami

            norm        = self.use_module.linalg.norm(vector).item()
            if self.debug_verb > 0:
                print("Check 4: Calculated norm.")
            H_real_file_savedata = [t, norm]
            H_imag_file_savedata = [t, norm]

            hopping_complex = None
            n_b_sys         = None

            obs_obj.calculate_and_write_observables(INSERT_CORRECT_ARGS_HERE)

    def generate_new_Hilbert_space_cupy(self, vector, use_U_weight_function, enlarge_steps=0, delta_t=0, garbage_tol=0, do_fancy_stuff=False, diag_coo_debug=None, impstates=None):
        """
        Generate new Hilbert subspace from evolved vector using a cupy alternative to numpy.isin
        (the built-in cupy.isin method is very memory-inefficient).
        Completely agnostic of the Hilbert space.

        THIS STILL NEEDS TO BE REWRITTEN.
        """
        # create the matrix elements if they don't exist:
        if diag_coo_debug is None:
            select_whoami, impstates    = self.determine_select_whoami(vector, use_U_weight_function=use_U_weight_function, delta_t=delta_t, garbage_tol=garbage_tol)
            try:
                del self.sparse_mats_dict
            except NameError:   # if the dict doesn't exist yet
                pass
            # create the new matrices:
            # switch to whoami_device to use that device's RAM instead:
            if whoami_device != vector_device:
                with whoami_device:
                    diag_coo_debug = self.HamObj.generate_mel(cupy.asarray(select_whoami), enlarge_steps=enlarge_steps)
            else:
                diag_coo_debug = self.HamObj.generate_mel(select_whoami, enlarge_steps=enlarge_steps)

            del select_whoami

        # Now transfer to current_device, if necessary:
        if whoami_device != vector_device:
            t0 = time.time()
            diag_vals, new_inds = [cupy.asarray(i) for i in diag_coo_debug[0]]
            COO_dict            = {term: (cupy.asarray(vals), (cupy.asarray(inds_to), cupy.asarray(inds_from))) for term, (vals, (inds_to, inds_from)) in diag_coo_debug[1].items()}
            debug_dict          = {term: cupy.asarray(vals) for term, vals in diag_coo_debug[2].items()}
            if self.debug_verb > move_data_timing_level:
                t1 = time.time()
                print("Moving data from device %i to device %i took %f ms." % (whoami_device, vector_device, (t1-t0)*1000))
        else:
            (diag_vals, new_inds), COO_dict, debug_dict = diag_coo_debug

        if self.debug_verb > 1:
            print("  a: Determined important states: (%i in total)" % impstates)

        if self.debug_verb > 1:
            print("  b: Created matrix elements.")

        # self.numstates now represents the new numstates:
        self.numstates      = new_inds.shape[0]
        self.create_matrices(diag_vals, COO_dict) # this needs the new numstates to work
        del COO_dict

        # insert old vector coefficients; this assumes that all whoami's are sorted
        if self.debug_verb > 2:
            print("    Norm prior to reassignment: %f." % cupy.linalg.norm(vector))
        new_vector          = self.use_module.zeros(self.numstates, dtype=self.HamObj.complex_type)
        t0 = time.time()
        ind_array           = self.HamObj.searchsorted(self.whoami, new_inds, allow_escapes=True)
        mask_array          = cupy.all(self.whoami[ind_array] == new_inds, axis=1)        # True only where an old value can be copied, i.e. new is in old
        if self.debug_verb > searchsorted_timing_level:
            t1 = time.time()
            print("This application of searchsorted (and masking) took %f ms (arg sizes %i, %i)." % ((t1-t0)*1000, self.whoami.shape[0], new_inds.shape[0]))
            t0 = time.time()
#        new_vector[mask_array]  = vector[cupy.all(new_inds[cupy_search.searchsorted_multidim_list(new_inds, self.whoami, allow_escapes=True, mindiff=mindiff)] == self.whoami, axis=1)]       # assign common values to positions

        new_vector[mask_array]  = vector[ind_array][mask_array]     # assign common values to positions
        del mask_array, ind_array

        self.whoami         = new_inds
        self.partition_lens = cupy_search.calc_partition_lens(self.HamObj.get_pos(self.whoami), self.HamObj.nchain)
        self.vector         = new_vector

        pre_evolve_norm     = -1
        pre_evolve_H        = numpy.nan
        if do_fancy_stuff:
            pre_evolve_norm =   self.use_module.linalg.norm(self.vector).item()
            pre_evolve_H    =   self.calculate_total_energy(self.vector)
            if pre_evolve_H.imag > 1e-12:
                raise RuntimeError("Energy has non-vanishing imaginary part: %1.15f." % pre_evolve_H.imag)
            else:
                pre_evolve_H    = pre_evolve_H.real

        if self.debug_verb > 2:
            if pre_evolve_norm == -1:
                print("    Norm after reassignment: %f." % cupy.linalg.norm(self.vector))
            else:
                print("    Norm after reassignment: %f." % pre_evolve_norm)

        if self.debug_verb > 1:
            print("  c: Inserted previous vector coefficients.")

        # The return values are only for diagnostic purposes:
        # The meat of this method (calculating the new matrices etc.) is done via attributes of the object
        return debug_dict["coupling"], pre_evolve_norm, pre_evolve_H



    ###################################
    # if we have two GPUs at our disposal, we can perform
    # the first half of the adaptation step concurrently to the observable calculations:
    ###################################
    def determine_select_whoami(self, vector, use_U_weight_function, delta_t=0, garbage_tol=0):
        """
        Wrapper function to determine the most relevant states.
        Completely agnostic of the Hilbert space.
        """
        if use_U_weight_function:
            if delta_t == 0:
                raise ValueError("Please specify a non-zero delta_t for this function to use U-weighting.")
            tmpvec      = self.first_order_U_importance(vector, delta_t)
        else:
            tmpvec      = self.use_module.abs(vector)
        sorted_indices  = tmpvec.argsort()
        sorted_vector   = tmpvec[sorted_indices]
        del tmpvec

        # Shuffle equal-weighted basis states if shuffle_seed is given:
        if (self.shuffle_seed is None
                or len(vector) < self.maxstates
                or sorted_vector[-self.maxstates-1] < sorted_vector[-self.maxstates]):
            if self.debug_verb > 4 and self.shuffle_seed is not None:
                print("    No state shuffling to be performed.")
            select_whoami       = self.whoami[sorted_indices][-self.maxstates:]
            if garbage_tol >= 0:
                select_whoami       = select_whoami[sorted_vector[-self.maxstates:] > garbage_tol]
        else:
            decision_val        = sorted_vector[-self.maxstates]
            firstind            = cupy.searchsorted(sorted_vector, decision_val, "left")
            lastind             = cupy.searchsorted(sorted_vector, decision_val, "right")
            if self.debug_verb > 4:
                print("    Number of states to be shuffled: %i" % (lastind - firstind))
            indfromback         = len(vector) - lastind
            if indfromback > self.maxstates:
                raise ValueError("A catastrophic error occurred while shuffling the equal-valued basis states.")

            sorted_whoami                   = self.whoami[sorted_indices]

            select_whoami                   = self.use_module.zeros((self.maxstates, self.whoami.shape[1]), dtype=self.whoami.dtype)
            select_whoami[-indfromback:]    = sorted_whoami[-indfromback:]
#            assert cupy.all(sorted_vector[firstind:lastind] == sorted_vector[firstind])
            select_whoami[:-indfromback]    = cupy.random.permutation(sorted_whoami[firstind:lastind])[:self.maxstates-indfromback]
            if garbage_tol >= 0:
                raise NotImplementedError("garbage_tol combined with equal-value shuffling has not yet been implemented.")
            del sorted_whoami
        del sorted_indices, sorted_vector
        select_whoami       = self.use_module.array(select_whoami[self.use_module.lexsort(select_whoami.T[::-1])])
        mempool.free_all_blocks()

#        max_bath_pre = select_whoami[:,1:].max()
        return select_whoami, select_whoami.shape[0] #, max_bath_pre


    ###################################
    # determine weight of basis states using a forward-looking method
    ###################################
    def first_order_U_importance_coherence(self, vector, delta_t):
        """
        Determine, more or less, the contribution of each basis state of |psi> to coherence it provides in the future.
        """
        psi_squared = self.use_module.abs(vector)**2
    # the following is the original version, which, however, doesn't work, so use the other one as long as M is hermitian and real:
#        return psi_squared + (delta_t**2) * psi_squared * (self.use_module.ones(self.numstates).dot(self.sparse_coupling) + self.use_modules.ones(self.numstates).dot(self.sparse_hopping) + self.diag_vals)**2
        return psi_squared + (delta_t**2) * psi_squared * self.use_module.power(self.total_H("ones"), 2)

    ###################################
    # determine weight of basis states using a forward-looking method
    ###################################
    def first_order_U_importance_norm(self, vector, delta_t):
        """
        Determine the contribution of each basis state of |psi> to the norm_squared of (1 - i δt H)|psi>.
        """
        return self.use_module.abs(vector - 1j*delta_t * self.total_H(vector))

###################################################################################################################################################################################


