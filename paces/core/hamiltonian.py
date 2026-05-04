"""
hamiltonian: Hosts the class HamiltonianFramework whose subclasses provide the base Hamiltonian.
"""

import time
import sys
import abc

from typing import NamedTuple

import numpy
import cupy     # pylint: disable=import-error

from ..aux import cupy_search
from ..aux.helpers import print_searchsorted_timing, flatten_dbg_dict, cupy_unique
from ..config import (SEARCHSORTED_TIMING_LEVEL, MEM_INFO_LEVEL, HILBERT_SPACE_DETAILED_LEVEL,
                            devices)

py_version = sys.version_info
if not (py_version.major > 3 or (py_version.major == 3 and py_version.minor >= 10)):
    # for the order-preserving dicts and strict kwarg in zip:
    print("Python >= 3.10 is required to run this.")
    sys.exit(1)

####################################################################################################

class MelTriple(NamedTuple):
    """
    Format used for the data produced by matrix-element generation functions.

    Args:
        inds: An array of (compressed) indices that the term maps each basis state to.
        vals: An array of matrix elements between `inds` and the input basis states.
            `vals` may also be a single number if all matrix elements are identical.
        mask: The mask to apply to the input basis states, i.e., a mask specifying if
            the basis states mapped to zero. The len of `inds` and `vals` must match
            the number of True in `mask`.
            This mask must also ensure that the hermitian conjugate will not contain any duplicates.
    """
    inds: cupy.ndarray
    vals: cupy.ndarray | complex
    mask: cupy.ndarray

####################################################################################################

class HamiltonianFramework:
    """
    Abstract base class of the object that provides structure to the Hilbert space.

    At the beginning of any calculation, a single instance of a child of this class is generated,
    which is then repeatedly called to perform all basic actions within the Hilbert space.

    The user must add model-specific features via children of this object.
    See the `paces.models` submodule for examples of existing models.

    In particular, each concretized subclass must contain all Hamiltonian-term-generating methods
    that can be combined to yield the total Hamiltonian.
    The names of these functions must begin with generate_mel_, e.g., `generate_mel_hopping`,
    where the part after the prefix is used to identify and call said term (here: hopping).
    Each such `generate_mel_xyz` function must take exactly two inputs,
        `(basis_states, raw_map_to)`,
    and the return signature depends on the bool `raw_map_to`:
        1. If it is `False`, then the return signature must be:
            `((meltriple0, meltriple1, meltriple2, ...), debug_info)`.
        The first return argument is an iterable of `MelTriple` instances,
        and the second, i.e., `debug_info`, may be `None` (see below for comments on `debug_info`).
        The `meltriple`s must conform to the standard given in `MelTriple`, i.e.,
        care must be taken to avoid duplication even when the hermitian conjugate is formed later.

        2. If `raw_map_to` is `True`, then the return signature must be:
            `(ind0, ind1, ind2, ...)`,
        which is a tuple of (compressed) index arrays indicating which basis states
        form the image of said Hamiltonian term when applied to the given `basis_states`.
        These may be non-unique, overlapping, etc., which will be taken care of later.
    If `debug_info` is not None, then it must be an iterable of debugging info that will be saved
    to file if the diagnostics option is used in the `TimeEvolution`.
    The header that is used for this debugging info must be provided
    by applying the decorator `@debug_lister` to the `generate_mel_xyz` function.

    Every subclass must also override the diagonal-term function `generate_mel_diag`:
    This must return only a single array of floats and nothing else.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_dims, use_terms,
                        use_complex_type=numpy.complex128, use_module=cupy, wordsize=32,
                        debug_verb=0, search_mindiff=32):
        """
        Initialize the abstract HamiltonianFramework object.

        Subclasses may want to add features to the initialization,
        but should still call this using super().

        Args:
            max_dims (iterable of uints): The maximum possible dimension per each site.
                This will determine the fundamental structure of many arrays,
                and the bound set here is hard and can never be exceeded.
                The number of sites is inferred from the len of max_dims.
            use_terms (dict of dict): The terms and paramters to be used in a given calculation.
                For every key term_x of use_terms, a corresponding method called generate_mel_term_x
                must be implemented in the concretized subclass that is calling this.
                Conversely, not every generate_mel_xyz will be called:
                Only those in use_terms are actually used in the calculation.
                The val belonging to the key term_x must be a dict specifiying
                the parameters used in the corresponding Hamiltonian term.
                Note that use_terms should almost certainly include the
                diagonal terms with the special key "diag".
            use_complex_type (type): The type to use for complex numbers. Default: numpy.complex128.
            use_module (module): The module to use for the computation.
                Must be either numpy or cupy. In practice, only cupy will work! Default: cupy.
            wordsize (uint): The size of the word to use.
                In particular, unsigned integers will be represented using this number of bits,
                and this value should be the one that functions optimally for your hardware/GPU.
                You almost certainly want this to be 32. Default: 32.
            debug_verb (int): The level of verbosity to use (prints debugging info to stdout).
                Higher values increase the verbosity. Default: 0 (prints nothing).
            search_mindiff (uint): A threshold for switching from binary to linear searches.
                This is irrelevant if `wordsize != 8`. Default: 32.
        """
        if not devices.configure_called:
            raise NameError("The CUDA devices must be configured before starting the calculation"
                            " (by calling paces.config.devices.configure with appropriate args).")
        if cupy.cuda.Device() != devices.vector_dev:
            raise ValueError("This class should be instantiated while vector_dev is current.")
        # The following is only used for logging (use_terms is handled separately):
        self.input_args     = {"max_dims": max_dims, "use_complex_type": use_complex_type,
                                "use_module": use_module, "wordsize": wordsize,
                                "debug_verb": debug_verb, "search_mindiff": search_mindiff}
        self.use_module     = use_module
        self.complex_type   = use_complex_type
        self.debug_verb     = debug_verb
        self.search_mindiff = search_mindiff

        with devices.vector_dev:
            self.max_dims_v = self.use_module.asarray(max_dims)
        with devices.whoami_dev:
            self.max_dims_w = self.use_module.asarray(max_dims)
        self.n_sites        = len(max_dims)

        self.wordsize       = wordsize  # (best should be 32, resulting in uint32)
        if self.wordsize not in (8, 16, 32, 64):
            raise ValueError(f"The specified wordsize ({self.wordsize}) is illegal.")
        # this determines the int type used to store the compressed basis states:

        self.max_word_dim   = 2**self.wordsize
        if self.max_dims_v.max() > self.max_word_dim:
            raise ValueError("At least one of the max dimensions exceeds the specified wordsize of"
                                f" {self.wordsize} bits (= {self.max_word_dim} states).")

        # get the uint type that we will use for the index arrays:
        self.dtype          = getattr(self.use_module, f"uint{self.wordsize}")

        with devices.vector_dev:
            # cupy array on vector_dev describing the bitwidths per site:
            self.bitwidths_v  = cupy.ceil(cupy.log2(self.max_dims_v)).astype(self.dtype)
        with devices.whoami_dev:
            # cupy array on whoami_dev describing the bitwidths per site:
            self.bitwidths_w  = cupy.ceil(cupy.log2(self.max_dims_w)).astype(self.dtype)
        # total number of words per line in the index arrays:
        self.totwordwidth   = int(self.use_module.ceil(self.bitwidths_v.sum()/self.wordsize))

        # This is a dict of dicts: the set of major keys specifies which terms to include,
        # and each sub-dict specifies that term's parameter(s).
        self.use_terms      = use_terms
        # for convenience, define a dict with the diag terms removed:
        self.offdiag_terms  = {k: self.use_terms[k] for k in self.use_terms if k != "diag"}
        # list containing the names of the debug terms belonging to each Hamiltonian component:
        self.extra_dbg_list = []
        for term in self.offdiag_terms:
            func = getattr(self, "generate_mel_" + term)
            if hasattr(func, "dbg_list"):
                self.extra_dbg_list += func.dbg_list


    def searchsorted(self, phonebook, findme, allow_escapes=False):
        """
        Find the index of the rows of findme in the sorted array phonebook.

        Args:
            phonebook (cupy.ndarray): The "reference" uint array of shape (M, N).
                This must be lexicographically sorted.
            findme (cupy.ndarray): The uint array of shape (L, N) whose rows shall be looked up.
                This doesn't need to be sorted. The second dimension must match that of phonebook.
            allow_escapes (bool): If False, then any rows occuring in findme but not in phonebook
                will throw an error. If True, then no error will be thrown (but the result given
                for those rows will be nonsense). Default: False.

        Returns:
            cupy.ndarray: Shape-(L,) array of indices giving the location of each row
            of `findme` in `phonebook`, i.e., `phonebook[indices]` is equal to `findme`.
        """
        if self.dtype == cupy.uint8:
            return cupy_search.searchsorted_multidim_list_vanilla(
                            phonebook,
                            findme,
                            mindiff=self.search_mindiff,
                            allow_escapes=allow_escapes,
                            linear_only=False,
                            shutup=True)
        if self.dtype == cupy.uint32:
            return cupy_search.searchsorted_full_binary(phonebook, findme, allow_escapes)
        raise TypeError(f"No searchsorted algorithm implemented for dtype {self.dtype}.")


    ############################################################################################
    # data segmentation functions,
    # basic compression/decompression
    ############################################################################################

    def decompress_ind_arr(self, arr):
        """Take a compressed array arr and return a decompressed arr with one col per site."""
        result      = self.use_module.zeros((len(arr), self.n_sites), dtype=self.dtype)
        for i in range(self.n_sites):
            result[:,i] = self.get_n_at_site(arr, i)
        return result


    def compress_ind_arr(self, raw_states):
        """Take an array with one col per site and return a compressed array without extra zeros."""
        if raw_states.device == devices.vector_dev:
            bitwidths = self.bitwidths_v
        else:
            bitwidths = self.bitwidths_w

        # add one column to allow for shape matching when assigning values to the last real column:
        result  = self.use_module.zeros((len(raw_states), self.totwordwidth+1), dtype=self.dtype)
        # index of first bit after the i-th block has ended:
        lastbit = bitwidths.cumsum()
        # index of first word containing the i-th block:
        word_i  = (lastbit - bitwidths) // self.wordsize

        if self.totwordwidth > 1:
            ud2             = f"uint{2*self.wordsize}"
            for i in range(self.n_sites):
                conv    = raw_states[:,i].astype(ud2) << ((word_i[i]+2)*self.wordsize - lastbit[i])
                conv    = conv[None].T
                result[:,word_i[i]:word_i[i]+2] += conv.view(self.dtype)[:,::-1]

        else:
            for i in range(self.n_sites):
                result[:,0] += raw_states[:,i].astype(self.dtype) << (self.wordsize - lastbit[i])

        # remove the last column:
        if not self.use_module.all(result[:,-1] == 0):
            raise ValueError("Fatal internal error: In the process of compressing the states,"
                                " an accessory column was not left unchanged.")
        result = self.use_module.ascontiguousarray(result[:,:-1])
        return result


    ############################################################################################
    # data segmentation functions,
    # intepret or modify compressed arrays
    ############################################################################################

    def _get_compression_inds(self, arr, index):
        """
        Helper function that computes word and bit indices of compressed arrays.

        Args:
            arr (1D compressed array of self.dtype): array of compressed basis states.
            index (uint): index of site in question.

        Returns:
            A tuple `(word_i, num2bits, numlead)` of positive integers, where:

            `word_i` is the index of first word containing the block of the site in question.
            `num2bits` is (`wordsize`) + (# of relevant bits in the word following the first word).
            `numlead` is the number of leading bits in first word that must be trimmed.

            Note that `num2bits == wordsize` is "zero", i.e., the next word has no relevant bits
            if num2bits == wordsize; and if num2bits < wordsize, then there are extraneous bits
            at the end of the first word.
        """
        bitwidths = self.bitwidths_v if arr.device == devices.vector_dev else self.bitwidths_w
        # number of bits preceding the block in question (= index of first relevant bit):
        bit_offset  = bitwidths[:index].sum()
        # index of first bit after the block has ended:
        lastbit     = bit_offset + bitwidths[index]
        # index of first word containing the block:
        word_i      = bit_offset // self.wordsize
        # (wordsize) + (number of relevant bits contained in the word following the first word):
        num2bits    = lastbit - self.wordsize*word_i    # (note: num2bits == wordsize is "zero")
        # number of leading bits in first word that must be trimmed:
        numlead     = bit_offset % self.wordsize

        # explicitly return CPU ints because the output of this function is not directly used on GPU
        return int(word_i), int(num2bits), int(numlead)


    def get_n_at_site(self, arr, index):
        """
        Take a compressed 1D index array `arr` and return an array of the basis numbers at a site.

        Args:
            arr (1D compressed array of `self.dtype`): array of compressed basis states.
            index (uint): index of site at which to extract the basis state numbers.

        Returns:
            A 1D array (of `self.dtype`) of basis state numbers at site `index`.
        """

        word_i, num2bits, numlead = self._get_compression_inds(arr, index)
        result      = arr[:,word_i].copy()

        # mask leading bits away, if necessary:
        if numlead != 0:
            result &= self.dtype((1 << (self.wordsize - numlead)) - 1)

        # if we only need the i-th word and need to shift to the right:
        if num2bits < self.wordsize:
            result  >>= self.wordsize - num2bits
        # if we need the i-th and (i+1)-th word, shift the current part to the left ...
        elif num2bits > self.wordsize:
            result  <<= num2bits - self.wordsize
            # ... and then add the contribution of the second word:
            result  += arr[:,word_i+1] >> (2*self.wordsize-num2bits)
        return result


    def add_n_at_site(self, arr, n, index):
        """
        Takes an array `arr` and raises or lowers the number at site `index` by `n` (in-place).

        Note: This method takes care of rollovers between words,
        but is unsafe with regard to over-/underflows due to floor/ceiling hits.

        Args:
            arr (1D compressed array of self.dtype): array of compressed basis states.
            n (int): number by which the basis state at the given site should be modified.
            index (uint): index of site at which to modify the basis state numbers.
        """
        if n == 0:
            return

        word_i, num2bits, _ = self._get_compression_inds(arr, index)

        if num2bits == self.wordsize:   # end of logical unit coincides with end of word
            if n > 0:
                arr[:,word_i] += self.dtype(n)
            else:
                arr[:,word_i] -= self.dtype(-n)
        elif num2bits < self.wordsize:  # end of logical unit is before end of word
            if n > 0:
                arr[:,word_i] += self.dtype(n << (self.wordsize-num2bits))
            else:
                arr[:,word_i] -= self.dtype(-n << (self.wordsize-num2bits))
        elif num2bits > self.wordsize:
            if self.wordsize != 8:
                raise NotImplementedError("This method has not been implemented for wordsizes != 8")
            # If the data spans two bytes, add the number to a 16-bit view spanning both bytes.
            # To do that, we must create a contiguous array from arr and also reverse the order
            # due to the CPU/GPU-level little-endian byte ordering.
            # Unfortunately, this requires creating a copy of the two columns in question.
            v16 = self.use_module.ascontiguousarray(arr[:,word_i:word_i+2][:,::-1]).view("uint16")
            if n > 0:
                res = v16 + self.dtype(n << (16-num2bits))
            else:
                res = v16 - self.dtype(-n << (16-num2bits))
            arr[:,word_i:word_i+2] = res.view("uint8")[:,::-1]


    def _generate_o_star_mask(self, basis_states, plus_inds):
        """Return a mask that removes values occuring in both the input inds and plus_inds."""
        t0 = time.time()
        sortplus    = plus_inds[cupy.lexsort(plus_inds.T[::-1])]
        ind_arr     = self.searchsorted(sortplus, basis_states, allow_escapes=True)
        o_star_mask = cupy.any(sortplus[ind_arr] != basis_states, axis=1)
        del ind_arr
        if self.debug_verb > SEARCHSORTED_TIMING_LEVEL:
            t1 = time.time()
            print("This application of (sorting +) searchsorted (+ masking) took"
                    f" {(t1-t0)*1000} ms (arg sizes {sortplus.shape[0]}, {basis_states.shape[0]}).")
        return o_star_mask


    @abc.abstractmethod
    def generate_mel_diag(self, basis_states):
        """Placeholder for a diagonal-matrix-element function that a subclass must override."""
        return


    def generate_mel(self, basis_states, enlarge_steps=0, log_ind=1.2):
        """
        Generate all Hamiltonian matrix elements required for the given basis_states.

        Args:
            basis_states (ndarray): The (compressed) array of basis states to
                generate the matrix elements from.
            enlarge_steps (uint): The number of additional matrix elements to incorporate when
                determining the next Hilbert subspace. Note: enlarge_steps = neighbor_degree - 1.
            log_ind (float): The number to use for debug printing.

        Returns:
            A tuple `((diag_vals, new_inds), coo_dict, dbg_dict)`,
            where:

            `diag_vals` is a 1D array of diagonal matrix elements after enlarging the Hilbert space.

            `new_inds` is the compressed 2D array of the new, compressed basis state indices.

            `coo_dict` is a dictionary whose keys corresponds to those of self.use_terms
            and whose vals can be fed into a sparse COO matrix generation routine.

            `dbg_dict` is a dictionary containing extra diagnostic data provided by the terms.
        """
        if basis_states.shape[1] != self.totwordwidth:
            raise ValueError("Shape of basis states does not match the specified number of bits.")
        for _ in range(enlarge_steps):
            # The following does a simplified version of what generate_mel does:
            basis_states = self.enlarge_basis_set(basis_states)

        # Generate the various non-diagonal matrix elements:
        melpack_dict    = {}    # this will hold the sets of meltriples
        dbg_dict        = {}    # this will hold the extra debug info returned by generate_mel funcs
        len_dict        = {}    # this will hold the number of basis states mapped to
        totallen        = basis_states.shape[0] # the diagonal values will contribute this much
        sub_index = 1 # used for debug printing to stdout
        for term in self.offdiag_terms:
            func                = getattr(self, "generate_mel_" + term)
            melpack, dbg_terms  = func(basis_states)
            if dbg_terms is not None:
                dbg_dict[term]  = dbg_terms
            len_dict[term]      = [int(meltriple.mask.sum()) for meltriple in melpack]
            totallen            += sum(len_dict[term])
            melpack_dict[term]  = melpack
            if self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL:
                print(f"      {log_ind}.{sub_index}: Determined {term} indices.")
            sub_index += 1
        dbg_list    = flatten_dbg_dict(dbg_dict)
        if len(dbg_list) != len(self.extra_dbg_list):
            raise IndexError("Number of debug terms does not match number of headers.")

        # Now create a new array to hold all of the new indices:
        all_inds    = self.use_module.empty((totallen, basis_states.shape[1]), dtype=self.dtype)

        # Start filling in all the indices, starting with the diagonal terms:
        all_inds[:basis_states.shape[0]]    = basis_states
        start                               = basis_states.shape[0]

        for term, melpack in melpack_dict.items():
            for thislen, meltriple in zip(len_dict[term], melpack, strict=True):
                end                 = start + thislen
                all_inds[start:end] = meltriple.inds
                start               = end
        if end != totallen:
            raise ValueError("Unknown indexing error encountered while generating matrix elements.")

        # all_inds now contains all indices that will be mapped to under the new Hamiltonian.
        # Note that all_inds will almost always be a larger set of states than basis_states,
        # since applying the Hamiltonian maps to other states. Now remove duplicates:
        new_inds = cupy_unique(all_inds)
        # new_inds is "final" now, it will become the whoami of time_evolution.

        statenum    = len(new_inds)
        if statenum >= 2147483648: # cupy uses 32-bit *signed* integers to store sparse arrays
            raise ValueError("The number of states is too large for signed 32-bit-integer indices.")
        if statenum >= self.max_word_dim:
            raise ValueError("The indexing array is too long to be stored in a"
                                                    f" {self.wordsize}-bit format.")
        if self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL:
            print(f"      {log_ind}.{sub_index}: Determined new whoami array.")
            sub_index += 1

        diag_vals   = self.generate_mel_diag(new_inds)
        if self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL:
            print(f"      {log_ind}.{sub_index}: Number of diag vals: {diag_vals.shape[0]}")
            sub_index += 1

        # Find the positions of the old basis states in the new set of indices:
        t0 = time.time()
        basis_lookup = self.searchsorted(new_inds, basis_states)
        dt = time.time() - t0
        print_searchsorted_timing(self.debug_verb, dt, new_inds.shape[0], basis_states.shape[0])

        # Based on the newly determined unique set of indices,
        # convert the existing melpacks into the format
        # that can be fed into the sparse matrix routines:
        coo_dict    = {}
        for term, melpack in melpack_dict.items():
            coo_dict[term]  = self._add_hc_and_generate_coo(new_inds, basis_lookup, len_dict[term],
                                                            melpack)
            if self.debug_verb > MEM_INFO_LEVEL:
                print("          Estimated near-maximal memory usage on whoami_dev:"
                            f" {int(cupy.get_default_memory_pool().used_bytes()/1024**2)} MiB")
            del melpack

        if self.debug_verb > HILBERT_SPACE_DETAILED_LEVEL:
            print(f"      {log_ind}.{sub_index}: Converted Hamiltonian matrix elements"
                                                        " into dense coo format.")
            sub_index += 1
        return (diag_vals, new_inds), coo_dict, dbg_dict


    def _add_hc_and_generate_coo(self, new_inds, basis_lookup, len_list, melpack):
        """
        Add the hermitian conjugate to the output of some generate_mel_xyz and return COO data.

        Args:
            new_inds (ndarray): Lexicographically sorted, unique set of new basis states.
            basis_lookup (1D ndarray): positions of old basis states in new_inds.
            len_list (iterable of ints): Number of basis states that do not map to zero
                under the data given in melpack. The len of len_list must match the len of melpack.
            melpack (iterable of `MelTriple`): Set of `MelTriple`s to be processed.
                Note that the dtype of all vals must be identical.

        Returns:
            A tuple `(vals, (inds_to, inds_from))`,
            where:

            `vals` is the 1D array of values of the matrix elements.

            `inds_to` and `inds_from` are two 1D arrays of indices for constructing COO matrices.
        """
        ind_pairs = self.use_module.empty((2 * sum(len_list), 2), dtype=self.dtype)

        try: # this assumes the first dtype is representative of the rest:
            valdtype    = melpack[0].vals.dtype # if the vals *are* stored as arrays
        except AttributeError:
            valdtype    = type(melpack[0].vals)      # if the vals are *not* stored as arrays
        vals        = self.use_module.empty(ind_pairs.shape[0], dtype=valdtype)

        start       = 0
        for thislen, meltriple in zip(len_list, melpack, strict=True):
            # compute indices for filling:
            midp    = start + thislen
            end     = midp + thislen

            # Start with the indices being mapped from:
            # Do "normal"/"outward"-mapping terms first ...
            ind_pairs[start:midp,0] = basis_lookup[meltriple.mask]
            # ... and then the hermitian conjugate:
            t0 = time.time()
            ind_pairs[midp:end,0]   = self.searchsorted(new_inds, meltriple.inds)
            print_searchsorted_timing(self.debug_verb, time.time() - t0,
                                            new_inds.shape[0], meltriple.inds.shape[0])

            # Now the indices being mapped to:
            ind_pairs[start:midp,1] = ind_pairs[midp:end,0]
            ind_pairs[midp:end,1]   = ind_pairs[start:midp,0]

            # Finally, fill in the actual values of the matrix elements:
            vals[start:midp]        = meltriple.vals
            vals[midp:end]          = meltriple.vals.conjugate()

            start = end
        return vals, (ind_pairs[:,1], ind_pairs[:,0])


    def enlarge_basis_set(self, basis_states):
        """Enlarge a given set of basis_states by adding 1st-degree neighboring basis states."""
        # Generate the various non-diagonal matrix elements:
        ind_dict    = {}
        len_dict    = {}
        totallen    = basis_states.shape[0]
        for term in self.offdiag_terms:
            func            = getattr(self, "generate_mel_" + term)
            # We don't need to filter for duplicates here (and we don't need the vals either):
            ind_dict[term]  = func(basis_states, raw_map_to=True)
            len_dict[term]  = [len(inds) for inds in ind_dict[term]]
            totallen        += sum(len_dict[term])

        # Now create a new array to hold all of the new indices:
        all_inds    = cupy.empty((totallen, basis_states.shape[1]), dtype=self.dtype)

        all_inds[:basis_states.shape[0]]    = basis_states
        start                               = basis_states.shape[0]
        for term, ind_list in ind_dict.items():
            for thislen, inds in zip(len_dict[term], ind_list, strict=True):
                end                 = start + thislen
                all_inds[start:end] = inds
                start               = end
        if end != totallen:
            raise ValueError("Unkown error encountered while expanding effective Hilbert space.")

        # Remove duplicates within the new indices:
        return cupy_unique(all_inds)
