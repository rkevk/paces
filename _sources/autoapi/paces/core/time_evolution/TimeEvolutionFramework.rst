paces.core.time_evolution.TimeEvolutionFramework
================================================

.. py:class:: paces.core.time_evolution.TimeEvolutionFramework(hamobj, ObsObj, obs_list, dirname, te_params, expm_params, params_file=None)

   Abstract base class of time-evolution that takes a concretized Hamiltonian/Observables as input.

   This class is used to actually run the time evolution based on the specific model that has
   been constructed in the subclasses of HamiltonianFramework and ObservablesFramework.

   The user should only need to add model-specific initial-state
   and initial basis-set generation functions to subclasses of this class.
   See the ../../paces/models folder for examples of existing models.

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


   .. py:method:: initialize_params_file()

      Write the header and main parameters to the central parameter file.



   .. py:method:: write_current_params_to_file(frame, start_time=None)

      Helper function to write function parameters to the central parameter file.



   .. py:method:: load_basis_from_file(loadfile)

      Load a basis set from file and set as the current basis set.



   .. py:method:: grow_optimal_basis(n_max=numpy.inf, fillfac=1.0)

      Enlarge a small initial basis set by repeatedly applying the Hamiltonian.

      :param n_max: The maximum number of enlargement iterations to perform. Default: infinity.
      :type n_max: int
      :param fillfac: The proportion of maxstates to use up. Default: 1.
      :type fillfac: float



   .. py:method:: create_initial_vector(vector_coeffs, vector_coo, auto_normalize=True)

      Take coeffs and coos of vector, look up the positions in whoami and create the state vector.



   .. py:method:: create_matrices(diag_params, coo_dict)

      Generate sparse matrices from diagonal values and coo_dict



   .. py:method:: create_ham_mat()

      Construct the sparse total Hamiltonian matrix and set to self.ham_mat.



   .. py:method:: total_ham(vector, use_ham_mat=False)

      Apply the total Hamiltonian to a vector.

      :param vector: vector that the Hamiltonian will be applied to.
      :type vector: ndarray
      :param use_ham_mat: If True, use the explicit total Hamiltonian matrix,
                          else use only the constituent matrices and diagonal values. Default: False.
      :type use_ham_mat: bool

      :returns: The result of applying H to the vector.
      :rtype: 1D ndarray



   .. py:method:: total_ham_ones(use_ham_mat=False)

      Apply the total Hamiltonian to a vector of all ones.

      :param use_ham_mat: If True, use the explicit total Hamiltonian matrix,
                          else use only the constituent matrices and diagonal values. Default: False.
      :type use_ham_mat: bool

      :returns: The result of applying H to the vector of all ones.
      :rtype: 1D ndarray



   .. py:method:: generate_timeline(t_array, coeff_save_obj)

      Evolve the states along the time-points given in t_array and compute and save observables.

      In order for this to work, the hamobj object must be initialized
      (which is a required argument for the initialization of the TimeEvolution object),
      and there must be an initial basis (see, e.g., the initial basis set generation functions)
      and an initial vector (via te.create_initial_vector(args) or by loading from file).

      :param t_array: The points in time.
                      The initial state is assumed to correspond to the first time value in t_array.
                      Then this function evolves step-by-step until the last value in t_array is reached.
      :type t_array: ndarray of floats
      :param coeff_save_obj: Instance of CoeffSaveParams (information on when to save coefficients).



   .. py:method:: coherence_weighting_func(vector, tau)

      Determine the contribution of each basis state to the coherence it provides after time tau.



   .. py:method:: norm_weighting_func(vector, tau)

      Determine the contribution of each basis state to the squared norm of (1 - i δt H)|psi>.


