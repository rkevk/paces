"""holstein: Concretizations of the core classes for a 1D single-exciton Holstein model."""

import time
import inspect
import sys

import numpy
import cupy     # pylint: disable=import-error
import cupyx    # pylint: disable=import-error

from . import _holstein_kernels as holke
from ..aux import cupy_search
from ..aux.helpers import cartesian_product, obs_attrs, debug_lister
from ..config import INIT_VERBOSITY_LEVEL, SEARCHSORTED_TIMING_LEVEL, devices

from ..core.hamiltonian import HamiltonianFramework, MelTriple
from ..core.observables import ObservablesFramework
from ..core.time_evolution import TimeEvolutionFramework

####################################################################################################

class Hamiltonian(HamiltonianFramework):
    """Concretized Hamiltonian class for the 1D single-exciton Holstein model."""
    def __init__(self, nchain, max_ho_dims, **kwargs):
        """
        Initialize the Hamiltonian.

        Args:
            nchain (uint): Number of sites in the chain. This is distinct from n_sites,
                as the first "site" is actually used to store the position of the exciton
                and all subsequent "sites" contain the Fock state of the QHOs.
            max_ho_dims (iterable of uints): maximum dimension of QHO at each site.

        See HamiltonianFramework for other args (with the exception of max_dims).
        """
        self.nchain = nchain
        if nchain != len(max_ho_dims):
            raise ValueError("nchain should match the number of QHOs.")
        max_dims    = numpy.empty(len(max_ho_dims)+1)
        max_dims[0] = self.nchain
        max_dims[1:]= max_ho_dims
        super().__init__(max_dims=max_dims, **kwargs)

        # The following two are used for logging and must be run after super().__init__:
        self.input_args["nchain"]       = nchain
        self.input_args["max_ho_dims"]  = max_ho_dims

        self.posbitwidth    = int(self.use_module.ceil(self.use_module.log2(self.nchain)))
        with devices.vector_dev:
            self.qhobitwidth_v  = self.bitwidths_v[1:]
        with devices.whoami_dev:
            self.qhobitwidth_w  = self.bitwidths_w[1:]

        self.periodic = False

    def get_phonon_occ(self, arr, site):
        """Get the phonon occupation number at a specific QHO site from a compressed 1D array"""
        return self.get_n_at_site(arr, site+1)


    def get_pos(self, arr):
        """Get the position of the exciton from a compressed 1D array"""
        return self.get_n_at_site(arr, 0)


    def get_phonon_at_exc(self, arr):
        """Get the number of phonons at the position of the exciton from a compressed 1D array"""
        if arr.device == devices.vector_dev:
            return holke.get_phonon_at_exc(arr, self.posbitwidth, self.qhobitwidth_v, self.wordsize)
        if arr.device == devices.whoami_dev:
            return holke.get_phonon_at_exc(arr, self.posbitwidth, self.qhobitwidth_w, self.wordsize)
        raise ValueError(f"Array is stored on an unknown device {arr.device}.")

    def sum_all_phonons(self, arr, omega=None):
        """Get the total number of phonons (per row) from a compressed 1D array"""
        if type(omega) in (int, float):
            omega   = self.use_module.full(self.nchain, omega, dtype="float64")
        elif isinstance(omega, (list, tuple, numpy.ndarray)) or isinstance(omega, cupy.ndarray):
            omega   = self.use_module.asarray(omega, dtype="float64")
        elif omega is None:
            omega   = cupy.ones(self.nchain, dtype="float64")
        else:
            raise TypeError("Unrecognized type of omega.")
        if arr.device == devices.vector_dev:
            return holke.sum_all_phonons(
                                omega, arr, self.posbitwidth, self.qhobitwidth_v, self.wordsize)
        if arr.device == devices.whoami_dev:
            return holke.sum_all_phonons(
                                omega, arr, self.posbitwidth, self.qhobitwidth_w, self.wordsize)
        raise ValueError(f"Array is stored on an unknown device {arr.device}.")


    def add_n_to_qho(self, arr, n, site):
        """
        Takes an array arr and adds (or subtracts) n phonons to site n (in-place).

        This method takes care of rollovers between words,
        but is unsafe with regard to over-/underflows due to floor/ceiling hits.

        Args:
            arr (ndarray): The compressed array that will be modified in-place.
            n (int): The number of phonons to add.
            site (uint): The index of the QHO to which the phonons shall be added.
        """
        self.add_n_at_site(arr, n, site+1)


    def add_n_to_pos(self, arr, n):
        """
        Takes an array arr and moves the exciton/electron by n sites (in-place).

        This method takes care of rollovers between bytes,
        but is unsafe with regard to over-/underflows due to system boundary excursions.

        Args:
            arr (ndarray): The compressed array that will be modified in-place.
            n (int): The number by which the exciton position will be shifted.
        """
        self.add_n_at_site(arr, n, 0)


    def add_phonon_at_exc(self, arr):
        """Add one phonon at the position of the exciton to arr (in-place)."""
        if arr.device == devices.vector_dev:
            holke.add_phonon_at_exc(arr, self.posbitwidth, self.qhobitwidth_v, self.wordsize)
        elif arr.device == devices.whoami_dev:
            holke.add_phonon_at_exc(arr, self.posbitwidth, self.qhobitwidth_w, self.wordsize)


    def rem_phonon_at_exc(self, arr):
        """Remove one phonon at the position of the exciton to arr (in-place)."""
        if arr.device == devices.vector_dev:
            holke.rem_phonon_at_exc(arr, self.posbitwidth, self.qhobitwidth_v, self.wordsize)
        elif arr.device == devices.whoami_dev:
            holke.rem_phonon_at_exc(arr, self.posbitwidth, self.qhobitwidth_w, self.wordsize)

    ############################################################################################

    def _generate_mel_hopping(self, basis_states, t_sys, order=1, raw_map_to=False):
        """
        Generate n-th order hopping matrix elements.

        basis_states does not have to be sorted for this to work.

        Args:
            basis_states (compressed 1D arr): array of basis states whose mel's will be calculated.
            t_sys (complex): hopping coupling constant.
            order (uint): order of hopping. 1 corresponds to nearest-neighbor,
                2 is next-to-nearest-neighbor, etc. Default: 1.
            raw_map_to: If True, do not mask duplicate indices from the complex conjugate.
                True should be used when enlarging the Hilbert space, but not when computing
                the matrix elements themselves. This changes the return signature. Default: False.

        Returns:
            If `raw_map_to`, then a tuple `(plus_inds, minus_inds)`,
            which are both compressed arrays representing the basis states that are mapped to.

            If not `raw_map_to`, then a tuple `((plus_meltriple, minus_meltriple), debug_info)`,
            where the first two are both `MelTriple`s representing right and left hopping.
            Here, terms that would cause duplication from the hermitian conj. have been removed.
            `debug_info` is always `None`.
        """
        if order < 1 or order > self.nchain:
            raise ValueError("Invalid value for order of hopping operator.")
        excpos  = self.get_pos(basis_states)

        # generate the plus side of the hopping (= to the right):
        plus_inds = basis_states.copy()
        self.add_n_to_pos(plus_inds, order)
        if self.periodic:
            raise NotImplementedError("Periodic hopping not yet implemented.")
        else:         ### for OBC, remove boundary excursions:
            plus_mask = excpos < self.nchain - order

        # generate the minus side of the coupling (= to the left):
        minus_inds = basis_states.copy()
        self.add_n_to_pos(minus_inds, -order)
        if self.periodic:
            raise NotImplementedError("Periodic hopping not yet implemented.")
        else:         ### for OBC, remove boundary excursions:
            minus_mask = excpos > order - 1

        if raw_map_to:
            return plus_inds[plus_mask], minus_inds[minus_mask]

        # add a mask to remove values that occur in both the input and plus_inds:
        minus_mask  &= self._generate_o_star_mask(basis_states, plus_inds)

        # generate the values of the matrix elements:
        minus_meltriple = MelTriple(minus_inds[minus_mask], t_sys.conjugate(), minus_mask)
        plus_meltriple  = MelTriple(plus_inds[plus_mask], t_sys, plus_mask)
        return (plus_meltriple, minus_meltriple), None


    def generate_mel_hopping(self, basis_states, raw_map_to=False):
        """Generate first-order (nearest-neighbor) hopping matrix elements."""
        return self._generate_mel_hopping(
                                    basis_states,
                                    self.use_terms["hopping"]["J"],
                                    order=1,
                                    raw_map_to=raw_map_to)


    def generate_mel_hopping2(self, basis_states, raw_map_to=False):
        """Generate second-order (next-to-nearest-neighbor) hopping matrix elements."""
        return self._generate_mel_hopping(
                                    basis_states,
                                    self.use_terms["hopping2"]["J2"],
                                    order=2,
                                    raw_map_to=raw_map_to)

    @debug_lister(["ceiling_hits"])
    def generate_mel_vib_coupling(self, basis_states, raw_map_to=False):
        """Generate vibronic-coupling matrix elements."""
        if basis_states.device == devices.vector_dev:
            max_ho_dims = self.max_dims_v[1:]
        else:
            max_ho_dims = self.max_dims_w[1:]

        phonon_occs     = self.get_phonon_at_exc(basis_states)
        excpos          = self.get_pos(basis_states)

        # generate the plus side of the coupling:
        plus_inds       = basis_states.copy()
        self.add_phonon_at_exc(plus_inds)
        # add mask for ceiling hits:
        plus_mask       = phonon_occs < max_ho_dims[excpos] - 1

        # generate the minus side of the coupling:
        minus_inds      = basis_states.copy()
        self.rem_phonon_at_exc(minus_inds)
        minus_mask      = phonon_occs > 0

        if raw_map_to:
            return plus_inds[plus_mask], minus_inds[minus_mask]

        ceiling_hits    = (len(basis_states) - plus_mask.sum())
        # add a mask to remove values that occur in both the input and plus_inds:
        minus_mask      &= self._generate_o_star_mask(basis_states, plus_inds)

        # generate the values of the matrix elements:
        plus_vals   = self.use_module.sqrt(phonon_occs[plus_mask]+1, dtype=cupy.float64)
        plus_vals   *= self.use_terms["vib_coupling"]["g"]
        minus_vals  = self.use_module.sqrt(phonon_occs[minus_mask], dtype=cupy.float64)
        minus_vals  *= self.use_terms["vib_coupling"]["g"].conjugate()

        return (MelTriple(plus_inds[plus_mask], plus_vals, plus_mask),
                MelTriple(minus_inds[minus_mask], minus_vals, minus_mask)), [ceiling_hits]


    def generate_mel_diag(self, basis_states):
        """Generate diagonal values of basis states."""
        param_dict  = self.use_terms["diag"]
        diag_vals   = param_dict["eps_sys"]
        diag_vals   += param_dict["hbar_omega"] * self.sum_all_phonons(basis_states)
        if param_dict["delta_eps"] != 0:
            diag_vals   += param_dict["delta_eps"] * self.get_pos(basis_states)
        return diag_vals

####################################################################################################

class Observables(ObservablesFramework):
    """Concretized Observables class for the 1D single-exciton Holstein model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hop_board  = None # this will be lazily constructed if calculate_hopping is called
        self.partition_lens = None # this will be constructed by primer_hook


    def primer_hook(self):
        """
        An initialization function to be run at the start of each calculation.

        For the Holstein model, this calculates the indices of the components belonging to
        each position of the exciton, which is used in the computation of some of the observables.
        """
        self.partition_lens = cupy_search.calc_partition_lens(
                                    self.hamobj.get_pos(self.teobj.whoami),
                                    self.use_module.arange(self.hamobj.nchain))


    @obs_attrs(fname="n_exc", header=" ".join(f"n^{{exc}}_{i}" for i in range(3)) + " etc.")
    def calculate_n_exc(self, vector):
        """Compute the expected number of excitons at each site of a given vector."""
        return self._calculate_partitioned_sum(abs(vector)**2)


    @obs_attrs(fname="n_pho", header=" ".join(f"n^{{pho}}_{i}" for i in range(3)) + " etc.")
    def calculate_n_pho(self, vector):
        """Compute the expected number of phonons at each site of a given vector."""
        weights = cupy.abs(vector)**2
        return holke.calculate_bath_n_b(weights, cupy.asarray(self.teobj.whoami),
                        self.hamobj.posbitwidth, self.hamobj.qhobitwidth_v, self.hamobj.wordsize)


    @obs_attrs(fname="V_hop", header=" ".join(f"V_{i}" for i in range(3)) + " etc.")
    def calculate_hopping(self, vector):
        """
        Compute the expectation value of excitonic hopping coupling interaction per site.

        As long as we don't have an even number of lattice sites with periodic boundary conditions,
        there is a nice trick to calculate the hopping energies using the position projector
        and the total hopping operator.
        The trick is:
        hop_i = (... + V_{i-2} - V_{i-1}^+) + V_i
                + (V_{i+1} - V_{i+1}^+ + V_{i+3} - V_{i+4}^+ + ...),
        where V_i = <P_i T>, with T being the total hopping operator.
        """
        if self.hamobj.nchain % 2 == 0 and self.hamobj.periodic:
            raise NotImplementedError("This doesn't work for even chain lengths with"
                                        " periodic boundary conditions.")
        allvals = vector.conj() * self.teobj.sparse_mats_dict["hopping"].dot(vector)

        hop_int_i = self._calculate_partitioned_sum(allvals)

        if self.hop_board is None:
            nc              = self.hamobj.nchain
            checkerboard    = self.use_module.array([[1,0] * ((nc+1)//2),
                                                     [0,1] * ((nc+1)//2)] * ((nc+1)//2))[:nc,:nc]
            self.hop_board  = self.use_module.triu(checkerboard, 1)
            self.hop_board  += self.use_module.triu(1 - checkerboard).T

        result  = ((1 - self.hop_board) * hop_int_i - self.hop_board * hop_int_i.conj()).sum(axis=1)

        if self.hamobj.periodic:
            return result
        assert abs(result[-1]) < 1e-12
        return result[:-1]


    @obs_attrs(fname="V_vib_coupl", header=" ".join(f"V_{i}" for i in range(3)) + " etc.")
    def calculate_coupling(self, vector):
        """Compute the expectation value of exciton-phonon coupling interactions per site."""
        allvals     = vector.conj() * self.teobj.sparse_mats_dict["coupling"].dot(vector)
        return self._calculate_partitioned_sum(allvals)


    def _calculate_partitioned_sum(self, sumvals, part_constant=5):
        """
        Calculate a partitioned sum, taking into consideration the length of the partitions

        This is not an observable per se, it is only used to compute other observables.
        """
        # to conserve RAM, only do this if the max partition length is
        #   less than part_constant times an equal partitioning:
        if self.partition_lens.max() < part_constant * self.teobj.numstates/self.hamobj.nchain:
            return cupy_search.jagged_to_regular(sumvals, self.partition_lens).sum(axis=1)

        c = 0
        result  = self.use_module.empty(self.hamobj.nchain, dtype=sumvals.dtype)
        for i in range(self.hamobj.nchain):
            result[i]   = sumvals[c:c+self.partition_lens[i]].sum()
            c += self.partition_lens[i]
        return result


    def calculate_reduced_dm(self, vector, mindiff=32):
        """Compute excitonic reduced density matrix from a given vector"""
        raise NotImplementedError("The reduced density matrix calculation has not yet been"
                                    " converted to the generalized data segmentation.")
        upper_tri   = self.use_module.zeros((self.hamobj.nchain, self.hamobj.nchain),
                                                dtype=self.hamobj.complex_type)
        # use the fact that whoami is lexicographically sorted
        #   to determine the break points between the different indices:
        indli       = cupy_search.find_changes_local_single(self.teobj.whoami[:,0]).tolist()
        indli       += [len(self.teobj.whoami),]
        lci         = 0
        for left_i in range(self.hamobj.nchain):
            if lci >= len(indli) - 1:
                break
            # if there is no basis state corresponding to the requested index, skip the iteration:
            if self.teobj.whoami[indli[lci]][0] != left_i:
                continue
            left_vec    = vector[indli[lci]:indli[lci+1]]
            left_side   = self.teobj.whoami[:,1:][indli[lci]:indli[lci+1]]
            lci         += 1
            rci         = lci   # start testing the position left_index + 1 as the first right_index
            for right_i in range(left_i, self.hamobj.nchain):
                if left_i == right_i:
                    upper_tri[left_i, right_i] = cupy.vdot(left_vec, left_vec)
                elif rci >= len(indli) - 1:
                    break
                else:
                    # if no basis state corresponds to the requested index, skip the iteration
                    if self.teobj.whoami[indli[rci]][0] != right_i:
                        print(f"({left_i}, {right_i}) not found. rci,"
                                    f" location: {rci}, {self.teobj.whoami[indli[rci]][0]}.")
                        continue
                    right_vec   = vector[indli[rci]:indli[rci+1]]
                    right_side  = self.teobj.whoami[:,1:][indli[rci]:indli[rci+1]]
                    rci         += 1
                    # left_vec are the coefficients matching the left_i.
                    # searchsorted(right_side, left_side) returns the indexes of the left_i whoami
                    #   as they occur in the right_i whoami if they occur,
                    #   else some value between 0 and 2**32 - 1 depending on certain criteria.
                    # right_side[searchsorted(right_side, left_side)] yields a matching
                    #   (relative to left_side) value from right_side or some arbitrary value from
                    #   right_side if there is no matching value.
                    # all(right_side[searchsorted(...)] == left_side, axis=1) is a mask that is true
                    #   only where the values of left_side occur exactly in right_side.
                    t0 = time.time()
                    left_search = cupy_search.searchsorted_multidim_list_8bits(
                                            left_side, right_side,
                                            allow_escapes=True, linear_only=False,
                                            mindiff=mindiff)
                    if self.teobj.debug_verb > SEARCHSORTED_TIMING_LEVEL:
                        print("This application of searchsorted_multidim_list took"
                                f" {(time.time() - t0)*1000} ms (arg sizes {left_side.shape[0]},"
                                f" {right_side.shape[0]}).")
                    bool_arr    = cupy.all(left_side[left_search] == right_side, axis=1)
                    upper_tri[left_i, right_i] = ((left_vec[left_search]
                                                * self.use_module.conj(right_vec))[bool_arr]).sum()
                    del left_search, right_side, bool_arr
                    devices.mempool.free_all_blocks()

        reduced_dm  = upper_tri + self.use_module.conj(self.use_module.triu(upper_tri, 1).T)
        return reduced_dm

####################################################################################################

class TimeEvolution(TimeEvolutionFramework):
    """Concretized TimeEvolution class for the 1D single-exciton Holstein model."""
    ###################################
    # generate initial basis set
    ###################################
    def create_uniform_truncated_basis(self, truncate_d):
        """Create a basis with truncate_d phonon levels at each site."""
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Creating initial basis set...", end=" ")
            sys.stdout.flush()
        if truncate_d > self.hamobj.max_dims_v[1:].min():
            raise ValueError("Specified basis truncation value exceeds at least one HO dimension.")
        if self.hamobj.nchain * truncate_d**self.hamobj.nchain > self.te_params.maxstates:
            raise ValueError("Number of states that would result"
                                " from this value of truncate_d exceeds maxstates!")
        # print current basis creation to file
        self.write_current_params_to_file(inspect.currentframe())
        raw_whoami      = self.use_module.asarray(cartesian_product(
                                numpy.arange(self.hamobj.nchain, dtype=numpy.uint8),
                                    *(numpy.arange(truncate_d, dtype=numpy.uint8)
                                    * numpy.ones(self.hamobj.nchain, dtype=numpy.uint8)[None].T)))
        self.whoami     = self.hamobj.compress_ind_arr(raw_whoami)
        self.numstates  = self.whoami.shape[0]
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Done!")

    def create_nonuniform_basis(self, truncate_d_list, lowest_d_list=None, minpos=0, maxpos=None):
        """
        Create a non-uniform basis with a varying number of phonon states per site.

        Args:
            truncate_d_list (list of uints): Number of basis states at given phonon site.
            lowest_d_list (None or list of uints): Lowest basis state to construct.
                The highest n at each site is then lowest_d + truncate_d.
                None corresponds to 0 everywhere. Default: None.
            minpos (uint): The leftmost exciton position.
            maxpos: The rightmost exciton position + 1 (maxpos==nchain is the largest possible val).
        """
        function_start_time = time.time()
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Creating initial basis set...", end=" ")
            sys.stdout.flush()
        if maxpos is None:
            maxpos = self.hamobj.nchain
        if lowest_d_list is None:
            lowest_d_list = numpy.zeros_like(truncate_d_list)
        if len(truncate_d_list) != self.hamobj.nchain or len(lowest_d_list) != self.hamobj.nchain:
            raise ValueError("Incorrect chain length.")
        if any(lowest_d_list[i] + truncate_d_list[i] > self.hamobj.max_dims_v[i+1]
                                                            for i in range(self.hamobj.nchain)):
            raise ValueError("Specified truncation value exceeds at least one of the max_ho_dims.")
        if numpy.prod(truncate_d_list) * (maxpos-minpos) > self.te_params.maxstates:
            raise ValueError("Number of states that would result from this value of truncate_d"
                                " exceeds maxstates!")
        if minpos < 0 or maxpos > self.hamobj.nchain or minpos >= maxpos:
            raise ValueError("Invalid minpos or maxpos.")

        # print current basis creation to file
        self.write_current_params_to_file(inspect.currentframe(), function_start_time)

        nontrivialdim       = numpy.array(truncate_d_list) > 1
        dimlist             = [numpy.arange(lowest_d_list[i],
                                            lowest_d_list[i] + truncate_d_list[i],
                                            dtype=numpy.uint8)
                                for i in range(self.hamobj.nchain) if nontrivialdim[i]]
        nopad_whoami        = self.use_module.asarray(cartesian_product(
                                        numpy.arange(minpos, maxpos, dtype=numpy.uint8), *dimlist))

        raw_whoami          = self.use_module.zeros((nopad_whoami.shape[0], self.hamobj.nchain+1),
                                                                                dtype=numpy.uint8)
        raw_whoami[:,numpy.nonzero(nontrivialdim)[0]+1] = nopad_whoami[:,1:]
        raw_whoami[:,0]     = nopad_whoami[:,0]
        self.whoami         = self.hamobj.compress_ind_arr(raw_whoami)
        self.numstates      = self.whoami.shape[0]
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Done!")


    def create_moving_gaussian_obc_basis(self, super_mu, super_sigma, sigma, maxval):
        """
        Create an initial basis of Gaussian phonon occupations around a specific exciton position.

        Args:
            super_mu (float): selects the initial exciton position
            super_sigma (float): sets how much bias is given to the initial position
                (where numpy.inf corresponds to zero bias and 0 to infinite bias).
            sigma (float): sets how sharp the individual distributions are
            maxval (int): sets the highest phonon occupation in total
                (note that the true max occupation will be slightly higher than maxval, however).
        """
        function_start_time = time.time()
        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Creating initial basis set...", end=" ")
            sys.stdout.flush()
        if maxval >= self.hamobj.max_dims_v[1:].max():
            raise ValueError("Specified basis truncation value exceeds the maximal max_ho_dims.")
        # print current basis creation to file
        self.write_current_params_to_file(inspect.currentframe(), function_start_time)
        def gauss(x, mu, sigma):
            return numpy.exp(-0.5 * ((x-mu)/sigma)**2)
        narr                = numpy.arange(self.hamobj.nchain)
        super_gauss         = 1 + maxval * gauss(narr, super_mu, super_sigma)
        basis_list          = []
        self.numstates      = 0
        for i in range(self.hamobj.nchain):
            this_d_gauss    = gauss(narr, i, sigma)
            basis_list      += [cartesian_product(*[numpy.arange(super_gauss[i] * this_d_gauss[j],
                                        dtype=numpy.uint8) for j in range(self.hamobj.nchain)])]
            self.numstates  += len(basis_list[-1])
            if self.numstates > self.te_params.maxstates:
                raise ValueError("The parameters specified for the initial basis set generate"
                                    " a basis set whose size exceeds maxstates.")
        whoami              = self.use_module.empty((self.numstates, self.hamobj.nchain + 1),
                                                                                dtype=cupy.uint8)
        c = 0
        for i, basis_part in enumerate(basis_list):
            whoami[c:c+len(basis_part),1:]      = self.use_module.asarray(basis_part)
            whoami[c:c+len(basis_part),0]       = i
            c += len(basis_part)

        if self.use_module.any(whoami.max(axis=0) >= self.hamobj.max_dims_v[1:]):
            raise ValueError("max_ho_dims exceeded!")
        self.whoami = self.hamobj.compress_ind_arr(whoami)

        if self.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Done!")

    def create_tensor_init_state(self, bstates, coeffs, excsite, sites="all"):
        r"""
        Generate a localized-exciton/tensor-product phonon state (requires an existing basis set).

        In other words, the initial vector of the TimeEvolution is set to a state
        |k> \otimes [\bigotimes_{l \in L} (\sum_j c_j |j_l>)],
        where k is the position of the exciton,
        l is the index of the phonon modes with L the set of phonon modes that are occupied,
        and j is the index of the Fock state with coefficients c_j.
        All phonon modes that are not in the set L will simply get the local vacuum state.

        Args:
            bstates (ndarray of uint): The Fock-state indices j which the coefficients refer to.
            coeffs (ndarray of complex): The Fock-state coefficients c_j.
            excsite (uint): The position of the exciton.
            sites (iterable of uints or "all"): The set L over which the tensor product is taken.
                The special value "all" takes the tensor product over all sites. Default: "all".
        """
        bstates = numpy.asarray(bstates)
        coeffs  = numpy.asarray(coeffs, dtype=complex)
        if sites != "all" and len(sites) > self.hamobj.nchain:
            raise ValueError("Invalid number of sites.")
        if len(bstates) != len(coeffs):
            raise ValueError("Number of coefficients does not match number of states.")

        if sites == "all":
            sitemask    = numpy.ones(self.hamobj.nchain, dtype=bool)
        else:
            sitemask    = numpy.array([i in sites for i in range(self.hamobj.nchain)])
        coeffs  = [coeffs] * sitemask.sum()
        mat     = cartesian_product(*coeffs)
        vector  = mat.prod(axis=1)
        if abs(1 - numpy.linalg.norm(vector)) > 1e-12:
            print(numpy.linalg.norm(vector))
            raise ValueError("State is not normalized, aborting.")

        bmask       = numpy.append([False], sitemask)
        basis       = numpy.zeros((len(vector), self.hamobj.nchain+1))
        basis[:,bmask]  = cartesian_product(*numpy.tile(bstates, (sum(sitemask), 1)))
        basis[:,0]      = excsite

        self.create_initial_vector(vector, basis, auto_normalize=True)

####################################################################################################

# XXX This is untested with the current version, possibly broken.
class PhononRedDensityMatrix:
    """Class to compute reduced density matrices of a single phononic site."""
    def __init__(self, hamobj):
        self.searchsorted   = hamobj.searchsorted
        self.get_phonon_occ = hamobj.get_phonon_occ
        self.add_n_to_qho   = hamobj.add_n_to_qho
        self.qho_dims       = hamobj.max_dims_v[1:]

    def calculate_rdm(self, site, whoami, vector):
        """Calculate the phononic reduced density matrix at a given site from a given vector."""
        occs        = self.get_phonon_occ(whoami, site)
        uniques     = cupy.unique(occs)
        adagger     = self.offdiag1_csr(whoami, site)
        dim         = self.qho_dims[site]
        upper_tri   = cupy.zeros((dim, dim), dtype=complex)
        rightvec    = vector.copy()
        for diarow in range(dim):
            vecprod     = numpy.conj(vector) * rightvec
            this_res    = cupy.zeros(dim - diarow, dtype=complex)
            for val in uniques:         # this creates entry rho_{val-diarow,val}
                if val < diarow:
                    continue
                this_res[val-diarow]    = cupy.sum(vecprod[occs == val])
            upper_tri   += cupy.diag(this_res, diarow)
            rightvec    = adagger.dot(rightvec)

        reduced_dm  = upper_tri + cupy.conj(cupy.triu(upper_tri, 1).T)
        return reduced_dm

    def offdiag1_csr(self, basis_states, site):
        """Generate a CSR first off-diagonal matrix (like a^+ without the numeric factor)."""
        plus_inds               = basis_states.copy()
        self.add_n_to_qho(plus_inds, 1, site)   # now plus_inds have one more phonon at the site
        inds_to         = self.searchsorted(basis_states, plus_inds, allow_escapes=True)
        mask            = cupy.all(basis_states[inds_to] == plus_inds, axis=1)
        numbs           = len(basis_states)
        return  cupyx.scipy.sparse.coo_matrix(
                      (cupy.ones(mask.sum().item()),
                          (inds_to[mask], cupy.arange(numbs, dtype=inds_to.dtype)[mask])),
                    shape=(numbs, numbs)).tocsr()
