paces.models.holstein.Hamiltonian
=================================

.. py:class:: paces.models.holstein.Hamiltonian(nchain, max_ho_dims, geo_dim=1, **kwargs)

   Bases: :py:obj:`paces.core.hamiltonian.HamiltonianFramework`


   Concretized Hamiltonian class for the single-exciton Holstein model.

   Initialize the Hamiltonian.

   :param nchain: Number of sites in the entire lattice. This is distinct from n_sites,
                  as the first "site" is actually used to store the position of the exciton
                  and all subsequent "sites" contain the Fock state of the QHOs.
   :type nchain: uint
   :param max_ho_dims: maximum dimension of QHO at each site.
                       This must be commensurate with nchain.
   :type max_ho_dims: iterable of uints
   :param geo_dim: Geometric dimension of the Holstein lattice
                   (1 is a chain, 2 is a square, 3 is a cube, etc.).
                   This determines the neighbors for the excitonic nearest-neigbor coupling.
                   Note that nchain must be compatible with geo_dim (e.g., if geo_dim is 2,
                   then nchain must be a square number). Default: 1 (a 1D chain).
   :type geo_dim: uint

   See HamiltonianFramework for other args (with the exception of max_dims).


   .. py:method:: get_phonon_occ(arr, site)

      Get the phonon occupation number at a specific QHO site from a compressed 1D array



   .. py:method:: get_pos(arr)

      Get the position of the exciton from a compressed 1D array



   .. py:method:: get_phonon_at_exc(arr)

      Get the number of phonons at the position of the exciton from a compressed 1D array



   .. py:method:: sum_all_phonons(arr, omega=None)

      Get the total number of phonons (per row) from a compressed 1D array



   .. py:method:: add_n_to_qho(arr, n, site)

      Takes an array arr and adds (or subtracts) n phonons to site n (in-place).

      This method takes care of rollovers between words,
      but is unsafe with regard to over-/underflows due to floor/ceiling hits.

      :param arr: The compressed array that will be modified in-place.
      :type arr: ndarray
      :param n: The number of phonons to add.
      :type n: int
      :param site: The index of the QHO to which the phonons shall be added.
      :type site: uint



   .. py:method:: add_n_to_pos(arr, n)

      Takes an array arr and moves the exciton/electron by n sites (in-place).

      This method takes care of rollovers between bytes,
      but is unsafe with regard to over-/underflows due to system boundary excursions.

      :param arr: The compressed array that will be modified in-place.
      :type arr: ndarray
      :param n: The number by which the exciton position will be shifted.
      :type n: int



   .. py:method:: add_phonon_at_exc(arr)

      Add one phonon at the position of the exciton to arr (in-place).



   .. py:method:: rem_phonon_at_exc(arr)

      Remove one phonon at the position of the exciton to arr (in-place).



   .. py:method:: generate_mel_hopping(basis_states, raw_map_to=False)

      Generate first-order (nearest-neighbor) hopping matrix elements.



   .. py:method:: generate_mel_hopping2(basis_states, raw_map_to=False)

      Generate second-order (next-to-nearest-neighbor) hopping matrix elements.



   .. py:method:: generate_mel_vib_coupling(basis_states, raw_map_to=False)

      Generate vibronic-coupling matrix elements.



   .. py:method:: generate_mel_diag(basis_states)

      Generate diagonal values of basis states.


