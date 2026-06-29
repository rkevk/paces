paces.models.holstein.Observables
=================================

.. py:class:: paces.models.holstein.Observables(*args, **kwargs)

   Bases: :py:obj:`paces.core.observables.ObservablesFramework`


   Concretized Observables class for the 1D single-exciton Holstein model.

   Initialize the Observables object and write the headers.

   :param teobj: The instantiated subclass of TimeEvolution that this is attached to.
   :param obs_list: List of observables to compute.
   :type obs_list: list


   .. py:method:: primer_hook()

      An initialization function to be run at the start of each calculation.

      For the Holstein model, this calculates the indices of the components belonging to
      each position of the exciton, which is used in the computation of some of the observables.



   .. py:method:: calculate_n_exc(vector)

      Compute the expected number of excitons at each site of a given vector.



   .. py:method:: calculate_n_pho(vector)

      Compute the expected number of phonons at each site of a given vector.



   .. py:method:: calculate_hopping(vector)

      Compute the expectation value of excitonic hopping coupling interaction per site.

      As long as we don't have an even number of lattice sites with periodic boundary conditions,
      there is a nice trick to calculate the hopping energies using the position projector
      and the total hopping operator.
      The trick is:
      hop_i = (... + V_{i-2} - V_{i-1}^+) + V_i
              + (V_{i+1} - V_{i+1}^+ + V_{i+3} - V_{i+4}^+ + ...),
      where V_i = <P_i T>, with T being the total hopping operator.



   .. py:method:: calculate_coupling(vector)

      Compute the expectation value of exciton-phonon coupling interactions per site.



   .. py:method:: calculate_mu_to_0(vector)

      Compute <mu(t) mu(0)> assuming the initial state was the global vacuum state.


