paces.models.holstein.TimeEvolution
===================================

.. py:class:: paces.models.holstein.TimeEvolution(hamobj, ObsObj, obs_list, dirname, te_params, expm_params, params_file=None)

   Bases: :py:obj:`paces.core.time_evolution.TimeEvolutionFramework`


   Concretized TimeEvolution class for the 1D single-exciton Holstein model.

   Framework to perform time evolution based on a given Hamiltonian Object.

   :param hamobj: The instance of a subclass of HamiltonianFramework to be worked on.
   :param ObsObj: The (non-instantiated) subclass of ObservablesFramework to use for observables.
   :param obs_list: List of observables to computed. Will be passed on to ObsObj.
   :type obs_list: list of str
   :param dirname: Path of directory where the calculation files shall be stored.
   :type dirname: str
   :param expm_params: ExpmParams instance containing parameters for the matrix exponential.
   :param te_params: TimeEvoParams instance containing basic time evolution parameters.
   :param params_file: The parameters will be saved under this file name.
                       If None, the name is automatically constructed and identified with a timestamp.
                       Default: None.
   :type params_file: str or None


   .. py:method:: create_uniform_truncated_basis(truncate_d)

      Create a basis with truncate_d phonon levels at each site.



   .. py:method:: create_nonuniform_basis(truncate_d_list, lowest_d_list=None, minpos=0, maxpos=None)

      Create a non-uniform basis with a varying number of phonon states per site.

      :param truncate_d_list: Phonon dimension per site.
      :type truncate_d_list: list of uints
      :param lowest_d_list: Lowest basis state to construct.
                            The highest n at each site is then lowest_d + truncate_d.
                            None corresponds to 0 everywhere. Default: None.
      :type lowest_d_list: None or list of uints
      :param minpos: The leftmost exciton position.
      :type minpos: uint
      :param maxpos: The rightmost exciton position + 1 (maxpos==nchain is the largest possible val).



   .. py:method:: create_moving_gaussian_obc_basis(super_mu, super_sigma, sigma, maxval)

      Create an initial basis of Gaussian phonon occupations around a specific exciton position.

      :param super_mu: selects the initial exciton position
      :type super_mu: float
      :param super_sigma: sets how much bias is given to the initial position
                          (where numpy.inf corresponds to zero bias and 0 to infinite bias).
      :type super_sigma: float
      :param sigma: sets how sharp the individual distributions are
      :type sigma: float
      :param maxval: sets the highest phonon occupation in total
                     (note that the true max occupation will be slightly higher than maxval, however).
      :type maxval: int



   .. py:method:: create_tensor_init_state(bstates, coeffs, excsite, sites='all')

      Generate a localized-exciton/tensor-product phonon state (requires an existing basis set).

      In other words, the initial vector of the TimeEvolution is set to a state
      |k> \otimes [\bigotimes_{l \in L} (\sum_j c_j |j_l>)],
      where k is the position of the exciton,
      l is the index of the phonon modes with L the set of phonon modes that are occupied,
      and j is the index of the Fock state with coefficients c_j.
      All phonon modes that are not in the set L will simply get the local vacuum state.

      :param bstates: The Fock-state indices j which the coefficients refer to.
      :type bstates: ndarray of uint
      :param coeffs: The Fock-state coefficients c_j.
      :type coeffs: ndarray of complex
      :param excsite: The position of the exciton.
      :type excsite: uint
      :param sites: The set L over which the tensor product is taken.
                    The special value "all" takes the tensor product over all sites. Default: "all".
      :type sites: iterable of uints or "all"


