paces.core.observables.ObservablesFramework
===========================================

.. py:class:: paces.core.observables.ObservablesFramework(teobj, obs_list)

   Abstract base class for the calculation of any observables over the course of a timeline.

   The user must define a subclass of this class for use with a specific model.
   Note that the TimeEvolution and Hamiltonian are accessible as the attributes
   self.teobj and self.hamobj and, for convenience, use_module is inherited as self.use_module.

   When designing a subclass, the following things must be taken into account:
   1.  If necessary, preprocessing can take place by defining a primer_hook method.
       This method may not have any arguments beyond self.
   2.  The heart of this class will be the methods that compute the observables.
       These must have the following signature/structure:
           @obs_attr(fname="fname for the observable", header="header for the file")
           def calculate_coupling(self, vector):
               return observable_list
       The decorator should contain the relative filename and the header line *without* time/norm.
       Note! The observables output should not be renormalized to account for lost norm.

   Initialize the Observables object and write the headers.

   :param teobj: The instantiated subclass of TimeEvolution that this is attached to.
   :param obs_list: List of observables to compute.
   :type obs_list: list


   .. py:method:: primer_hook()

      An initialization function to be run at the start of each set of observables calculation.

      This function is just a placeholder, children should override it if necessary.
      However, if the children do not require any preprocessing, this method can be left as-is.
      This function should not have any arguments beyond self.



   .. py:method:: calculate_total_energy(vector)

      Compute the expected total energy of a vector.



   .. py:method:: compute_all_and_write(vector, t, norm, log_index)

      Call all necessary observables functions and write the results to file.

      :param vector: The vector whose expectation values shall be computed.
      :type vector: ndarray
      :param t: The current simulation time.
      :type t: float
      :param norm: The norm of the vector
      :type norm: float
      :param log_index: The number to use for debug printing.
                        This should be one larger than the last number used in the code that calls this.
      :type log_index: int



   .. py:method:: calculate_rdm(site, vector)

      Calculate and return a reduced density matrix at a given site from a given vector.


