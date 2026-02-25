"""
time_evolution: Hosts the class TimeEvolutionFramework whose subclasses provide propagation funcs.
"""

import time
import inspect
import sys
import pprint
import abc

from typing import Optional
from dataclasses import dataclass

import numpy
import cupy     # pylint: disable=import-error
import cupyx    # pylint: disable=import-error

from ..aux import expm_multiply_simple
from ..aux.helpers import write_params, flatten_dbg_dict
from ..config import *

if numpy.uintc != numpy.uint32:
    raise TypeError("This code will not work on a system whose integer size is not 32 bits.")


####################################################################################################

@dataclass
class CoeffSaveParams:
    """
    Helper dataclass used to determine when to save explicit wavefunction coefficients.

    Args:
        save_every (int): Save the wavefunction and basis set at every n-th timestep.
            Default is 0, which disables saving (but is overridden by save_first and save_last).
        save_first (bool): If True, save the initial wavefunction and basis set. Default: False.
        save_last (bool): If True, save the final wavefunction and basis set. Default: False.
        max_i (int): Maximum number of iterations that can be reached. Default: 0.
            This may be left unset at instantiation, as it will be set by the timeline later.
    """
    save_every: int = 0
    save_first: bool = False
    save_last: bool = False
    max_i: int = 0

    def save_necessary(self, i: int):
        """Determine whether it is necessary to save at timestep with index i."""
        if self.max_i == 0:
            raise ValueError("max_i should be set before calling this method.")
        necessary   = (self.save_first and i == 0)
        necessary   |= (self.save_last and i == self.max_i)
        if self.save_every != 0:
            necessary   |= (self.save_every is not None and i % self.save_every == 0)
        return necessary

####################################################################################################

@dataclass
class ExpmParams:
    """
    Dataclass containg parameters for the computation of the matrix exponential.

    Args:
        m_star (int): Maximum order to use in the series expansion of U(δt). Default: 100.
        explosion_cutoff (float): If the l2 norm of the state ever exceeds this value
            after applying U(δt), abort the computation. Default: 2.
        use_scaling (bool): If False, then a simple Taylor series is used to compute U(δt),
            otherwise the more costly scaling-and-squaring method is used. Default: False.
    """
    m_star: int = 100
    explosion_cutoff: float = 2.0
    use_scaling: bool = False


@dataclass
class TimeEvoParams:
    """
    Dataclass containg basic, generic parameters for the time evolution.

    Args:
        maxstates (uint): Nominal truncation number when truncating basis states.
        enlarge_steps (uint): The number of additional matrix elements to incorporate when
            determining the next Hilbert subspace. Note that enlarge_steps = neighbor_degree - 1.
        diagnostics (bool): If True, compute and save extra diagnostic data. Default: True.
        shuffle_seed (int or None): When determining the new Hilbert space, equal-weight values
            will be shuffled to avoid bias using this value as the initial seed.
            None will disable shuffling and keep the (biased) lexicographic order. Default: 0.
        weighting_name (str): Weighting method to use when determining the basis states to
            be truncated. Must be one of ["coherence", "norm"]. Default: "coherence".
        tau (float): time tau to use for the forward-looking part of the Hilbert
            subspace determination, where 0 disables forward-looking. Default: 0.
        garbage_tol (float): The value to use as a garbage cutoff when determining the Hilbert
            subspace evolution. Negative values disable the garbage function and keep all
            basis states up to maxstates. Experience has shown this probably shouldn't be enabled.
            Default: -1 (disable garbage_tol).
        use_two_streams (bool): Whether to spread the computation over two GPUs.
            This doesn't seem to work properly, do not enable. Default: False.
    """
    maxstates: int
    enlarge_steps: int
    diagnostics: bool = True
    shuffle_seed: Optional[int] = 0
    weighting_name: str = "coherence"
    tau: float = 0.0
    garbage_tol: float = -1
    use_two_streams: bool = False

    if use_two_streams:
        raise NotImplementedError("Using two streams does not appear to confer a benefit.")


####################################################################################################

class TimeEvolutionFramework:
    """
    Abstract base class of time-evolution that takes a concretized Hamiltonian/Observables as input.

    This class is used to actually run the time evolution based on the specific model that has
    been constructed in the subclasses of HamiltonianFramework and ObservablesFramework.

    The user should only need to add model-specific initial-state
    and initial basis-set generation functions to subclasses of this class.
    See the ../../paces/models folder for examples of existing models.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, hamobj, ObsObj, obs_list, dirname, te_params, expm_params, params_file=None):
        """
        Framework to perform time evolution based on a given Hamiltonian Object.

        Args:
            hamobj: The instance of a subclass of HamiltonianFramework to be worked on.
            ObsObj: The (non-instantiated) subclass of ObservablesFramework to use for observables.
            obs_list (list of str): List of observables to computed. Will be passed on to ObsObj.
            dirname (str): Path of directory where the calculation files shall be stored.
            expm_params: ExpmParams instance containing parameters for the matrix exponential.
            te_params: TimeEvoParams instance containing basic time evolution parameters.
            params_file (str or None): The parameters will be saved under this file name.
                If None, the name is automatically constructed and identified with a timestamp.
                Default: None.
        """
        # basic configuration:
        self.use_module = hamobj.use_module
        self.debug_verb = hamobj.debug_verb

        self.start_time = time.time()
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Initializing TimeEvolution object...")

        # First of the two core objects, the Hamiltonian:
        self.hamobj     = hamobj

        # parameters:
        self.dirname    = dirname
        self.te_params  = te_params
        self.expm_params= expm_params
        self.weighting_func = getattr(self, te_params.weighting_name + "_weighting_func")
        if te_params.shuffle_seed is not None:
            self.use_module.random.seed(te_params.shuffle_seed)

        # Second of the two core objects, the observables object, which we instantiate now:
        # (this has to refer to the attributes set above, which is why it has to be down here)
        self.obsobj     = ObsObj(self, obs_list)

        # to be constructed later:
        self.ham_mat        = None  # total Hamiltonian matrix, constructed in create_ham_mat

        # the following will all be constructed in generate_timeline:
        self.whoami         = None  # array of indices mapping the position to specific basis states
        self.numstates      = None  # total number of states in the current vector
        self.vector         = None  # current state vector
        self.cso            = None  # CoeffSaveParams instance
        self.diag_vals      = None  # diagonal values of Hamiltonian
        self.wf_name_list   = None  # names of wavefunction-coefficient files
        self.log_ind        = 0     # used for printing debug info to stdout
        self.sparse_mats_dict = None # dict of sparse matrices of Hamiltonian

        real_valued = all(all(value.imag == 0 for value in paramdict.values())
                                    for paramdict in self.hamobj.use_terms.values())
        if not real_valued:
            # The following applies specifically to the weighting_func methods
            raise NotImplementedError("Not all functions have been adapted to complex-valued"
                                            " off-diagonal Hamiltonian matrix elements.")

        if params_file is None:
            self.params_file    = self.dirname + "/paces_params_run_" + str(time.time()) + ".log"
        else:
            self.params_file    = params_file

        self.initialize_params_file()

        if vector_device != whoami_device and self.te_params.use_two_streams:
            with whoami_device:
                self.whoami_stream = cupy.cuda.Stream()

        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Finished initialization of TimeEvolution object!\n")

    ############################################################################################
    # params file functions
    ############################################################################################

    def initialize_params_file(self):
        """Write the header and main parameters to the central parameter file."""
        with open(self.params_file, "w", encoding="utf-8") as pf:
            pf.write("### Hilbert space parameters:\n")
            for itemstr in [i for i in vars(self.hamobj) if i != "use_terms"]:
                write_params(pf, self.hamobj, itemstr)

            pf.write("\n### Hamiltonian terms and parameters:\n")
            pf.write(pprint.pformat(self.hamobj.use_terms, width=1) + "\n")

            pf.write("\n### TimeEvolution parameters:\n")
            pf.write(repr(self.te_params))
            pf.write(repr(self.expm_params))

            pf.write("\n### End of parameter list\n"
                "#################################################################\n")


    def write_current_params_to_file(self, frame, start_time=None):
        """Helper function to write function parameters to the central parameter file."""
        if start_time is None:
            start_time  = time.time()
        localtime   = time.asctime(time.localtime(start_time))
        args, _, _, values  = inspect.getargvalues(frame)
        arg_list            = [(str(i) + "=" + str(values[i])) for i in args if str(i) != "self"]
        fname               = frame.f_code.co_name
        header = f"\nFunction call at {start_time} ({localtime} local):\n   {fname}("
        header += ", ".join(arg_list) + ")\n"
        with open(self.params_file, "a", encoding="utf-8") as params_file:
            params_file.write(header)


    ############################################################################################
    # initial basis creation and manipulation methods
    ############################################################################################

    def load_basis_from_file(self, loadfile):
        """Load a basis set from file and set as the current basis set."""
        function_start_time = time.time()
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Loading basis set from file...", end=" ")
            sys.stdout.flush()
        self.write_current_params_to_file(inspect.currentframe(), function_start_time)

        self.whoami     = self.use_module.load(loadfile)
        self.numstates  = self.whoami.shape[0]
        if self.whoami.shape[1] != self.hamobj.totwordsize:
            raise ValueError("Loaded basis set doesn't match total number of bits given by hamobj.")
        if self.whoami.dtype != self.hamobj.dtype:
            raise ValueError("Loaded basis set does not match wordsize given by hamobj.")

        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Done!")


    def grow_optimal_basis(self, n_max=numpy.inf, fillfac=1.0):
        """
        Enlarge a small initial basis set by repeatedly applying the Hamiltonian.

        Args:
            n_max (int): The maximum number of enlargement iterations to perform. Default: infinity.
            fillfac (float): The proportion of maxstates to use up. Default: 1.
        """
        function_start_time = time.time()
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Growing initial basis set...", end=" ")
            sys.stdout.flush()
        self.write_current_params_to_file(inspect.currentframe(), function_start_time)

        initsize            = len(self.whoami)
        whoami              = self.whoami
        n                   = 0
        while len(whoami) <= self.te_params.maxstates * fillfac:
            previous_whoami = whoami
            n               += 1
            if n > n_max:
                break
            whoami          = self.hamobj.enlarge_basis_set(previous_whoami)

        # use previous_whoami, since the last was the one that triggered the break condition
        self.whoami         = previous_whoami
        self.numstates      = self.whoami.shape[0]
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print(f"Done! Performed {n} enlargements (grew from {initsize} to"
                                        f" {len(self.whoami)} states).")

    ############################################################################################
    # initial state creation
    ############################################################################################

    def create_initial_vector(self, vector_coeffs, vector_coo, auto_normalize=True):
        """
        Take coeffs and coos of vector, look up the positions in whoami and create the state vector.
        """
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Creating the initial state vector...", end=" ")
            sys.stdout.flush()

        try:
            vector_coo  = cupy.asnumpy(vector_coo)
        except NameError:
            vector_coo  = numpy.asarray(vector_coo)
        if not isinstance(vector_coo, numpy.ndarray):
            raise TypeError("Conversion of initial vector_coo to numpy.ndarray failed.")

        if vector_coo.ndim != 2:
            raise ValueError("The coo list must be 2D"
                                " (even if it contains only the entry for one basis state).")
        if numpy.unique(vector_coo, axis=0).shape != vector_coo.shape:
            raise ValueError("There are duplicates in the specified vector coordinates.")
        if vector_coo.shape[0] != len(vector_coeffs):
            raise ValueError("The number of coefficients does not match the number of coordinates.")
        if auto_normalize:
            vector_coeffs /= numpy.linalg.norm(vector_coeffs)

        vector_coo      = self.hamobj.compress_ind_arr(cupy.asarray(vector_coo))

        self.vector     = self.use_module.zeros(len(self.whoami), dtype=self.hamobj.complex_type)
        for i, coo in enumerate(vector_coo):
            index               = self.use_module.all(self.whoami == self.use_module.asarray(coo),
                                                                            axis=1).nonzero()[0][0]
            self.vector[index]  = vector_coeffs[i]

        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Done!")

    ############################################################################################
    # matrix creation
    ############################################################################################

    def create_matrices(self, diag_params, coo_dict):
        """Generate sparse matrices from diagonal values and coo_dict"""
        self.sparse_mats_dict   = {term: self.use_module.sparse.coo_matrix(coo,
                                            shape=(self.numstates, self.numstates)).tocsr()
                                            for term, coo in coo_dict.items()}
        self.diag_vals          = diag_params


    def create_ham_mat(self):
        """Construct the sparse total Hamiltonian matrix and set to self.ham_mat."""
        self.ham_mat    = cupyx.scipy.sparse.diags(self.diag_vals)
        for mat in self.sparse_mats_dict.values():
            self.ham_mat   += mat


    def total_ham(self, vector, use_ham_mat=False):
        """
        Apply the total Hamiltonian to a vector.

        Args:
            vector (ndarray): vector that the Hamiltonian will be applied to.
            use_ham_mat (bool): If True, use the explicit total Hamiltonian matrix,
                else use only the constituent matrices and diagonal values. Default: False.

        Returns:
            res (ndarray): The result of applying H to the vector.
        """
        if use_ham_mat:
            return self.ham_mat @ vector

        res = self.diag_vals * vector
        for mat in self.sparse_mats_dict.values():
            res += mat.dot(vector)
        return res


    def total_ham_ones(self, use_ham_mat=False):
        """
        Apply the total Hamiltonian to a vector of all ones.

        Args:
            use_ham_mat (bool): If True, use the explicit total Hamiltonian matrix,
                else use only the constituent matrices and diagonal values. Default: False.

        Returns:
            B: The result of applying H to the vector of all ones.
        """
        if use_ham_mat:
            return self.ham_mat.sum(axis=1).flatten()

        res = self.diag_vals
        for mat in self.sparse_mats_dict.values():
            res += mat.sum(axis=1).flatten()
        return res

    ############################################################################################
    # entire timeline evolution method
    ############################################################################################

    def generate_timeline(self, t_array, coeff_save_obj):
        """
        Evolve the states along the time-points given in t_array and compute and save observables.

        In order for this to work, the hamobj object must be initialized
        (which is a required argument for the initialization of the TimeEvolution object),
        and there must be an initial basis (see, e.g., the initial basis set generation functions)
        and an initial vector (via te.create_initial_vector(args) or by loading from file).

        Args:
        t_array (ndarray of floats): The points in time. The initial state is assumed to correspond
            to the first time value in t_array. Then evolve step-by-step until the last value in
            t_array is reached.
        coeff_save_obj: Instance of CoeffSaveParams (contains data on when to save coefficients).
        """
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Setting up generate_timeline function...")
        self.write_current_params_to_file(inspect.currentframe(), time.time())

        self._compute_wf_name_list(t_array)
        self.cso                = coeff_save_obj
        self.cso.max_i          = len(t_array) - 1

        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Finished generate_timeline setup! Beginning time evolution...")

        for i, t_curr in enumerate(t_array):
            self.log_ind = 0 # reset at the start of each timestep
            if i == 0:  # the first step is special because we compute observables etc. w/o evolving
                diag_coo_debug = self._perform_first_step(t_array[0])
            else:
                delta_t = t_curr - t_array[i-1]
                diag_coo_debug = self._evolve_single_step(i, t_curr, delta_t, diag_coo_debug)

    ############################################################################################
    # non-trivial weighting functions (which don't appear to provide any benefit)
    ############################################################################################

    def coherence_weighting_func(self, vector, tau):
        """
        Determine the contribution of each basis state to the coherence it provides after time tau.
        """
        psi_sq = self.use_module.abs(vector)**2
        return psi_sq + (tau**2) * psi_sq * self.use_module.power(self.total_ham_ones(), 2)


    def norm_weighting_func(self, vector, tau):
        """
        Determine the contribution of each basis state to the squared norm of (1 - i δt H)|psi>.
        """
        return self.use_module.abs(vector - 1j*tau * self.total_ham(vector))

    ############################################################################################
    # single-step time evolution
    ############################################################################################

    def _simple_taylor(self, delta_t):
        """Perform time evolution step to current vector without scaling and squaring."""
        tol = 2**-53
        converged   = False
        term        = self.vector
        res         = self.vector.copy()
        c1          = cupy.linalg.norm(res, cupy.inf)
        if self.debug_verb > EXPM_EVO_LEVEL:
            print(f"         l2 norm prior to evolution: {self.use_module.linalg.norm(res, 2)}.")
        for j_ind in range(self.expm_params.m_star):
            coeff       = -1j * delta_t / float(j_ind+1)
            term        = self.total_ham(term)
            term        *= coeff
            c2          = cupy.linalg.norm(term, cupy.inf)
            res         += term
            c1_plus_c2  = c1 + c2
            term_ratio  = c1_plus_c2/cupy.linalg.norm(res, cupy.inf)
            if term_ratio <= tol:
                converged = True    # added
                break
            c1 = c2
            if self.debug_verb > EXPM_EVO_LEVEL:
                print(f"           l2 norm of vector at iteration {j_ind+1:3}:"
                                f" {self.use_module.linalg.norm(res, 2)}.")

        if self.debug_verb > EXPM_EVO_LEVEL:
            print(f"         l2 norm after evolution: {self.use_module.linalg.norm(res, 2)}.")
        del term
        if self.te_params.diagnostics:                 # added
            return res, [converged, j_ind, c1_plus_c2, term_ratio]
        return res


    def _evolve_single_step(self, i, t_curr, delta_t, diag_coo_debug):
        """Perform a single time evolution step including computation of expectation values."""
        if self.debug_verb > BASE_STEP_LEVEL:
            print(f"\nCheck {self.log_ind}: Initiated timestep from {t_curr-delta_t} to {t_curr}.\n"
                    f"Check {self.log_ind+1}: Determining next Hilbert subspace.")
            self.log_ind += 1
        if vector_device != whoami_device and self.te_params.use_two_streams:
            self.whoami_stream.synchronize()
        use_weight_function = (self.te_params.tau != 0 and i != 0)
        # At this point:
        # both vector and self.vector: state of t - delta_t, basis of t - delta_t
        dbg_dict, post_adapt_norm_en = self._generate_new_hilbert_space_cupy(
                                                enlarge_steps = self.te_params.enlarge_steps,
                                                use_weight_function = use_weight_function,
                                                tau = self.te_params.tau,
                                                diag_coo_debug = diag_coo_debug)
        if self.debug_verb > BASE_STEP_LEVEL:
            print("         Finished determining next Hilbert subspace.")
            self.log_ind += 1
        # Now:
        # self.vector:  state of t - delta_t, basis of t

        #
        # Perform the actual time evolution step:
        #
        if self.expm_params.use_scaling:  # with scaling and squaring
            self.create_ham_mat()
            self.vector, expm_dbg = expm_multiply_simple(self.ham_mat, self.vector,
                                                                t=1j*delta_t, return_dbg=True)
        else:                               # without scaling and squaring
            self.vector, expm_dbg = self._simple_taylor(delta_t)

        if self.use_module.linalg.norm(self.vector) > self.expm_params.explosion_cutoff:
            raise RuntimeError("Norm has exceeded preset explosion cutoff value"
                                        f" ({self.expm_params.explosion_cutoff})!")
        if self.debug_verb > BASE_STEP_LEVEL:
            print(f"Check {self.log_ind}: Applied time-evolution operator to vector.")
            self.log_ind += 1
        # Now:
        # self.vector:   state of t,           basis of t

        #
        # Begin the concurrent evaluation of the next Hilbert space, if applicable:
        #
        if (vector_device != whoami_device
                and self.te_params.use_two_streams
                and i < self.cso.max_i):
            diag_coo_debug = self._concurrent_whoami(use_weight_function)

        norm = self.use_module.linalg.norm(self.vector).item()
        if self.debug_verb > BASE_STEP_LEVEL:
            print(f"Check {self.log_ind}: Calculated norm.")
            self.log_ind += 1

        self._postprocess(t_curr, norm, i)

        # save and print diagnostic data:
        expm_dbg = expm_dbg[:2] + [x.item() for x in expm_dbg[2:]]
        self._save_diagnostic_data(t_curr, norm, post_adapt_norm_en, expm_dbg, dbg_dict)
        self._print_memory_info()
        return diag_coo_debug


    def _perform_first_step(self, t_curr):
        """Perform the initial adaptation and computation of expectation values."""
        if self.debug_verb > BASE_STEP_LEVEL:
            print(f"\nCheck {self.log_ind}: Began initial timestep (no evolution in this step).\n"
                    f"Check {self.log_ind+1}: Determining next Hilbert subspace.")
            self.log_ind += 1
        if vector_device != whoami_device and self.te_params.use_two_streams:
            self.whoami_stream.synchronize()
        dbg_dict, post_adapt_norm_en = self._generate_new_hilbert_space_cupy(
                                                enlarge_steps = self.te_params.enlarge_steps,
                                                use_weight_function = False,
                                                tau = self.te_params.tau,
                                                diag_coo_debug = None)
        if self.debug_verb > BASE_STEP_LEVEL:
            print("         Finished determining next Hilbert subspace.")
            self.log_ind += 1

        # Begin the concurrent evaluation of the next Hilbert space, if applicable:
        if (vector_device != whoami_device
                and self.te_params.use_two_streams
                and 0 < self.cso.max_i):
            diag_coo_debug = self._concurrent_whoami(False)
        else:
            diag_coo_debug = None

        norm = self.use_module.linalg.norm(self.vector).item()
        if self.debug_verb > BASE_STEP_LEVEL:
            print(f"Check {self.log_ind}: Calculated norm.")
            self.log_ind += 1

        self._postprocess(t_curr, norm, 0)

        # save and print diagnostic data:
        expm_dbg = [0,] * 4
        self._save_diagnostic_data(t_curr, norm, post_adapt_norm_en, expm_dbg, dbg_dict)
        self._print_memory_info()
        return diag_coo_debug


    def _concurrent_whoami(self, use_weight_function):
        """Run a concurrent computation of the adapted Hilbert space, if two streams are used."""
        select_whoami = self._determine_select_whoami(
                                        self.vector,
                                        garbage_tol=self.te_params.garbage_tol,
                                        use_weight_function=use_weight_function,
                                        tau=self.te_params.tau)
        if self.debug_verb > BASE_STEP_LEVEL:
            print("Check {self.log_ind}: Beginning concurrent calculation of next Hilbert space.")
        with whoami_device:
            with self.whoami_stream:
                new_select      = cupy.asarray(select_whoami)
                diag_coo_debug  = self.hamobj.generate_mel(
                                            new_select,
                                            enlarge_steps=self.te_params.enlarge_steps,
                                            log_ind=self.log_ind)
        del select_whoami
        return diag_coo_debug # this is (diag_vals, new_inds), coo_dict, dbg_dict


    def _postprocess(self, t_curr, norm, i):
        """Compute observables and save wavefunction data for a single timestep."""
        # Compute and save observables:
        if self.debug_verb > BASE_STEP_LEVEL:
            print(f"Check {self.log_ind}: Beginning calculation of observable expectation values.")
        self.obsobj.primer_hook()
        self.obsobj.compute_all_and_write(self.vector, t_curr, norm, self.log_ind)

        # save wavefunction data if necessary:
        if self.cso.save_necessary(i):
            self.use_module.save(self.wf_name_list[i], self.vector)
            # the following works even if self.whoami is not on current_device:
            self.use_module.save(self.wf_name_list[i].replace("wf_file_", "whoami_"), self.whoami)


    def _print_memory_info(self):
        if self.debug_verb > MEM_INFO_LEVEL:
            print(f"Used {cupy.get_default_memory_pool().used_bytes()/1024**2:1.1f} or"
                    f" {numpy.diff(cupy.cuda.Device(vector_device).mem_info)[0]/1024**2:1.1f}"
                    " MiB on vector_device.")
            with cupy.cuda.Device(whoami_device):
                print(f"Used {cupy.get_default_memory_pool().used_bytes()/1024**2:1.1f} or"
                    f" {numpy.diff(cupy.cuda.Device(whoami_device).mem_info)[0]/1024**2:1.1f}"
                    " MiB on whoami_device.")


    def _compute_wf_name_list(self, t_array):
        """Helper function to determine the filenames for the wavefunctions."""
        smallest_val    = numpy.min(numpy.abs(t_array))
        smallest_diff   = numpy.diff(numpy.abs(t_array)).min()
        if smallest_val == 0:
            decnum  = numpy.abs(numpy.floor(numpy.log10(smallest_diff))) + 1
        else:
            decnum  = numpy.abs(numpy.floor(numpy.log10(min(smallest_val, smallest_diff)))) + 1
        largest_val     = numpy.max(numpy.abs(t_array))
        intnum          = numpy.abs(numpy.ceil(numpy.log10(largest_val)))

        f_str           = "wf_file_{0:0%i.%if}" % (intnum + decnum + 1, decnum)
        self.wf_name_list = [self.dirname + "/wf_coeffs/" + f_str.format(i) for i in t_array]


    def _save_diagnostic_data(self, t_curr: float, norm: float, norm_energy: list,
                                        expm_dbg: list, dbg_dict: dict):
        """Helper function to save diagnostic data during the timeline."""
        if self.te_params.diagnostics:
            l = [t_curr, norm, time.time() - self.start_time,] + norm_energy + expm_dbg
            l += [self.numstates]
            l += [x.item() for x in flatten_dbg_dict(dbg_dict)]

            with open(self.obsobj.fname_dict["diagnostics"], "a", encoding="utf-8") as diag_f:
                numpy.savetxt(diag_f, l, newline=" ")
                diag_f.write("\n")

    ###################################
    # generate new Hilbert subspace from evolved vector using a cupy alternative to numpy.isin
    # (the built-in cupy.isin method is very memory-inefficient)
    # XXX add docstring!
    ###################################
    def _generate_new_hilbert_space_cupy(self, use_weight_function, enlarge_steps, tau,
                                                diag_coo_debug=None):
        if diag_coo_debug is None:  # if there was no previous concurrent computation of whoami
            select_whoami = self._determine_select_whoami(self.vector,
                                                        use_weight_function=use_weight_function,
                                                        tau=tau,
                                                        garbage_tol=self.te_params.garbage_tol)
            try:
                del self.sparse_mats_dict
            except NameError:   # if the dict doesn't exist yet
                pass
            # create the new matrices, switching to whoami_device if applicable:
            with whoami_device:
                if self.debug_verb > HILBERT_SPACE_LEVEL:
                    print(f"      {self.log_ind}.2: Beginning creation of matrix elements.")
                diag_coo_debug = self.hamobj.generate_mel(cupy.asarray(select_whoami),
                                                            enlarge_steps=enlarge_steps,
                                                            log_ind=self.log_ind+0.2)
            del select_whoami

        # Now transfer to current_device, if necessary:
        if whoami_device != vector_device:
            t0 = time.time()
            diag_vals, new_inds = [cupy.asarray(i) for i in diag_coo_debug[0]]
            coo_dict = {term: (cupy.asarray(vals), (cupy.asarray(inds_to), cupy.asarray(inds_from)))
                            for term, (vals, (inds_to, inds_from)) in diag_coo_debug[1].items()}
            debug_dict = {term: cupy.asarray(vals) for term, vals in diag_coo_debug[2].items()}
            if self.debug_verb > MOVE_DATA_TIMING_LEVEL:
                t1 = time.time()
                print(f"Moving data from device {whoami_device} to device {vector_device} took"
                            f" {(t1-t0)*1000} ms.")
        else:
            (diag_vals, new_inds), coo_dict, debug_dict = diag_coo_debug

        if self.debug_verb > HILBERT_SPACE_LEVEL:
            print(f"           Created all matrix elements.")

        # self.numstates now represents the new numstates:
        self.numstates      = new_inds.shape[0]
        self.create_matrices(diag_vals, coo_dict) # this needs the new numstates to work
        del coo_dict

        # Insert old vector coefficients, i.e., post-adaptation detruncation;
        # this assumes that all whoami's are sorted:
        if self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL:
            print(f"             Norm prior to reassignment: {cupy.linalg.norm(self.vector)}.")
        new_vector          = self.use_module.zeros(self.numstates, dtype=self.hamobj.complex_type)
        t0 = time.time()
        ind_array           = self.hamobj.searchsorted(self.whoami, new_inds, allow_escapes=True)
        # The following is True only where an old value can be copied, i.e. new is in old:
        mask_array          = cupy.all(self.whoami[ind_array] == new_inds, axis=1)
        if self.debug_verb > SEARCHSORTED_TIMING_LEVEL:
            t1 = time.time()
            print(f"This application of searchsorted (and masking) took {(t1-t0)*1000} ms"
                                    f" (arg sizes {self.whoami.shape[0]}, {new_inds.shape[0]})")
            t0 = time.time()

        # assign common values to positions:
        new_vector[mask_array]  = self.vector[ind_array][mask_array]
        del mask_array, ind_array

        self.whoami         = new_inds
        self.vector         = new_vector

        if self.debug_verb > HILBERT_SPACE_LEVEL:
            print(f"      {self.log_ind}.3: Inserted previous vector coefficients.")

        return debug_dict, self._post_assignment_diagnostics()


    def _post_assignment_diagnostics(self):
        """
        Helper function that prints and/or returns information about norm and energy as required.
        """
        pre_evolve_norm     = -1
        pre_evolve_energy   = numpy.nan

        if self.te_params.diagnostics:
            pre_evolve_norm     =   self.use_module.linalg.norm(self.vector).item()
            pre_evolve_energy   =   self.obsobj.calculate_total_energy(self.vector)
            if pre_evolve_energy.imag > 1e-12:
                raise RuntimeError(f"Energy has non-zero imaginary part: {pre_evolve_energy.imag}.")
            pre_evolve_energy    = pre_evolve_energy.real

        if self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL:
            if pre_evolve_norm == -1:
                print(f"             Norm after reassignment: {cupy.linalg.norm(self.vector)}")
            else:
                print(f"             Norm after reassignment: {pre_evolve_norm}")
        return [pre_evolve_norm, pre_evolve_energy]



    def _determine_select_whoami(self, vector, use_weight_function, tau, garbage_tol):
        """
        Determine the most relevant states that we will truncate to in the following.

        If we have two GPUs at our disposal, we can theoretically call this concurrently to the
        observable calculations; however, in practice, it doesn't seem to be worth doing so.

        Args:
            vector (1D ndarray): vector to base the truncation on
            use_weight_function (bool): whether or not to use a non-trivial weighting function.
                If True, then self.te_params.weighting_func will be used.
            tau (float): "lookahead" time to use for the weighting function.
            garbage_tol (float): The value to use as a garbage cutoff when determining the Hilbert
                subspace evolution. Negative values disable the garbage function and keep all basis
                states up to maxstates. Experience has shown this probably shouldn't be enabled.
                Default: -1 (disable garbage_tol).

        Returns:
            select_whoami (1D ndarray): the new, lexicographically sorted, lookup array after
                (potentially shuffling and) truncating to the `maxstates` most important states.
        """
        if use_weight_function:
            if tau == 0:
                raise ValueError("Specify a non-zero tau for this function to use U-weighting.")
            tmpvec      = self.weighting_func(vector, tau)
        else:
            tmpvec      = self.use_module.abs(vector)
        sorted_indices  = tmpvec.argsort()
        sorted_vector   = tmpvec[sorted_indices]
        del tmpvec

        ms  = self.te_params.maxstates
        if (self.te_params.shuffle_seed is None
                or len(vector) < ms
                or sorted_vector[-ms-1] < sorted_vector[-ms]):
            if (self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL
                        and self.te_params.shuffle_seed is not None):
                print("           No state shuffling to be performed.")
            select_whoami = self.whoami[sorted_indices][-ms:]
            if garbage_tol >= 0:
                select_whoami = select_whoami[sorted_vector[ms:] > garbage_tol]
        else:
            decision_val    = sorted_vector[-ms]
            firstind        = cupy.searchsorted(sorted_vector, decision_val, "left")
            lastind         = cupy.searchsorted(sorted_vector, decision_val, "right")
            if self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL:
                print(f"           Number of states to be shuffled: {lastind - firstind}")
            indfromback         = len(vector) - lastind
            if indfromback > ms:
                raise ValueError("A catastrophic error occurred while"
                                    " shuffling the equal-valued basis states.")

            sorted_whoami = self.whoami[sorted_indices]

            select_whoami = self.use_module.zeros((ms, self.whoami.shape[1]),
                                                    dtype=self.whoami.dtype)
            select_whoami[-indfromback:] = sorted_whoami[-indfromback:]
            select_whoami[:-indfromback] = cupy.random.permutation(
                                                            sorted_whoami[firstind:lastind]
                                                                    )[:ms-indfromback]
            if garbage_tol >= 0:
                raise NotImplementedError("garbage_tol combined with equal-value shuffling"
                                            " has not yet been implemented.")
            del sorted_whoami
        del sorted_indices, sorted_vector
        select_whoami = self.use_module.array(
                                    select_whoami[self.use_module.lexsort(select_whoami.T[::-1])])
        mempool.free_all_blocks()

        if self.debug_verb > HILBERT_SPACE_LEVEL:
            print(f"      {self.log_ind}.1: Determined important states"
                                        f" ({select_whoami.shape[0]} in total)")

        return select_whoami
