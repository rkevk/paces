paces.core.hamiltonian.HamiltonianFramework
===========================================

.. py:class:: paces.core.hamiltonian.HamiltonianFramework(max_dims, use_terms, use_complex_type=numpy.complex128, use_module=cupy, wordsize=32, debug_verb=0, search_mindiff=32)

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

   Initialize the abstract HamiltonianFramework object.

   Subclasses may want to add features to the initialization,
   but should still call this using super().

   :param max_dims: The maximum possible dimension per each site.
                    This will determine the fundamental structure of many arrays,
                    and the bound set here is hard and can never be exceeded.
                    The number of sites is inferred from the len of max_dims.
   :type max_dims: iterable of uints
   :param use_terms: The terms and paramters to be used in a given calculation.
                     For every key term_x of use_terms, a corresponding method called generate_mel_term_x
                     must be implemented in the concretized subclass that is calling this.
                     Conversely, not every generate_mel_xyz will be called:
                     Only those in use_terms are actually used in the calculation.
                     The val belonging to the key term_x must be a dict specifiying
                     the parameters used in the corresponding Hamiltonian term.
                     Note that use_terms should almost certainly include the
                     diagonal terms with the special key "diag".
   :type use_terms: dict of dict
   :param use_complex_type: The type to use for complex numbers. Default: numpy.complex128.
   :type use_complex_type: type
   :param use_module: The module to use for the computation.
                      Must be either numpy or cupy. In practice, only cupy will work! Default: cupy.
   :type use_module: module
   :param wordsize: The size of the word to use.
                    In particular, unsigned integers will be represented using this number of bits,
                    and this value should be the one that functions optimally for your hardware/GPU.
                    You almost certainly want this to be 32. Default: 32.
   :type wordsize: uint
   :param debug_verb: The level of verbosity to use (prints debugging info to stdout).
                      Higher values increase the verbosity. Default: 0 (prints nothing).
   :type debug_verb: int
   :param search_mindiff: A threshold for switching from binary to linear searches.
                          This is irrelevant if `wordsize != 8`. Default: 32.
   :type search_mindiff: uint


   .. py:method:: searchsorted(phonebook, findme, allow_escapes=False)

      Find the index of the rows of findme in the sorted array phonebook.

      :param phonebook: The "reference" uint array of shape (M, N).
                        This must be lexicographically sorted.
      :type phonebook: cupy.ndarray
      :param findme: The uint array of shape (L, N) whose rows shall be looked up.
                     This doesn't need to be sorted. The second dimension must match that of phonebook.
      :type findme: cupy.ndarray
      :param allow_escapes: If False, then any rows occuring in findme but not in phonebook
                            will throw an error. If True, then no error will be thrown (but the result given
                            for those rows will be nonsense). Default: False.
      :type allow_escapes: bool

      :returns: Shape-(L,) array of indices giving the location of each row
                of `findme` in `phonebook`, i.e., `phonebook[indices]` is equal to `findme`.
      :rtype: cupy.ndarray



   .. py:method:: decompress_ind_arr(arr)

      Take a compressed array arr and return a decompressed arr with one col per site.



   .. py:method:: compress_ind_arr(raw_states)

      Take an array with one col per site and return a compressed array without extra zeros.



   .. py:method:: get_n_at_site(arr, index)

      Take a compressed 1D index array `arr` and return an array of the basis numbers at a site.

      :param arr: array of compressed basis states.
      :type arr: 1D compressed array of `self.dtype`
      :param index: index of site at which to extract the basis state numbers.
      :type index: uint

      :returns: A 1D array (of `self.dtype`) of basis state numbers at site `index`.



   .. py:method:: add_n_at_site(arr, n, index)

      Takes an array `arr` and raises or lowers the number at site `index` by `n` (in-place).

      Note: This method takes care of rollovers between words,
      but is unsafe with regard to over-/underflows due to floor/ceiling hits.

      :param arr: array of compressed basis states.
      :type arr: 1D compressed array of self.dtype
      :param n: number by which the basis state at the given site should be modified.
      :type n: int
      :param index: index of site at which to modify the basis state numbers.
      :type index: uint



   .. py:method:: generate_mel_diag(basis_states)
      :abstractmethod:


      Placeholder for a diagonal-matrix-element function that a subclass must override.



   .. py:method:: generate_mel(basis_states, enlarge_steps=0, log_ind=1.2)

      Generate all Hamiltonian matrix elements required for the given basis_states.

      :param basis_states: The (compressed) array of basis states to
                           generate the matrix elements from.
      :type basis_states: ndarray
      :param enlarge_steps: The number of additional matrix elements to incorporate when
                            determining the next Hilbert subspace. Note: enlarge_steps = neighbor_degree - 1.
      :type enlarge_steps: uint
      :param log_ind: The number to use for debug printing.
      :type log_ind: float

      :returns: A tuple `((diag_vals, new_inds), coo_dict, dbg_dict)`,
                where:

                `diag_vals` is a 1D array of diagonal matrix elements after enlarging the Hilbert space.

                `new_inds` is the compressed 2D array of the new, compressed basis state indices.

                `coo_dict` is a dictionary whose keys corresponds to those of self.use_terms
                and whose vals can be fed into a sparse COO matrix generation routine.

                `dbg_dict` is a dictionary containing extra diagnostic data provided by the terms.



   .. py:method:: enlarge_basis_set(basis_states)

      Enlarge a given set of basis_states by adding 1st-degree neighboring basis states.


