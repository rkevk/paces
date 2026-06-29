paces.core.hamiltonian.MelTriple
================================

.. py:class:: paces.core.hamiltonian.MelTriple

   Bases: :py:obj:`NamedTuple`


   Format used for the data produced by matrix-element generation functions.

   :param inds: An array of (compressed) indices that the term maps each basis state to.
   :param vals: An array of matrix elements between `inds` and the input basis states.
                `vals` may also be a single number if all matrix elements are identical.
   :param mask: The mask to apply to the input basis states, i.e., a mask specifying if
                the basis states mapped to zero. The len of `inds` and `vals` must match
                the number of True in `mask`.
                This mask must also ensure that the hermitian conjugate will not contain any duplicates.

