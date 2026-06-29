paces.core.time_evolution.CoeffSaveParams
=========================================

.. py:class:: paces.core.time_evolution.CoeffSaveParams

   Helper dataclass used to determine when to save explicit wavefunction coefficients.

   :param save_every: Save the wavefunction and basis set at every n-th timestep.
                      Default is 0, which disables saving (but is overridden by save_first and save_last).
   :type save_every: int
   :param save_first: If True, save the initial wavefunction and basis set. Default: False.
   :type save_first: bool
   :param save_last: If True, save the final wavefunction and basis set. Default: False.
   :type save_last: bool
   :param max_i: Maximum number of iterations that can be reached. Default: 0.
                 This may be left unset at instantiation, as it will be set by the timeline later.
   :type max_i: int


   .. py:method:: save_necessary(i: int)

      Determine whether it is necessary to save at timestep with index i.


