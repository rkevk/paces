paces.core.time_evolution.ExpmParams
====================================

.. py:class:: paces.core.time_evolution.ExpmParams

   Dataclass containing parameters for the computation of the matrix exponential.

   :param m_star: Maximum order to use in the series expansion of U(δt). Default: 100.
   :type m_star: int
   :param explosion_cutoff: If the l2 norm of the state ever exceeds this value
                            after applying U(δt), abort the computation. Default: 2.
   :type explosion_cutoff: float
   :param use_scaling: If False, then a simple Taylor series is used to compute U(δt),
                       otherwise the more costly scaling-and-squaring method is used. Default: False.
   :type use_scaling: bool
   :param renormalize: If True, then the state is renormalized after each application of U(δt).
                       As the norm constitutes useful diagnostic data, this should only ever be used in
                       special testing or debugging cases. Default: False.
   :type renormalize: bool
   :param lanczos_order: The Krylov-space dimension + 1 that is used to compute the Lanczos
                         representation of U(δt). If this is 0, then the default series-based exponentiation
                         is used instead. Default: 0.
   :type lanczos_order: int

