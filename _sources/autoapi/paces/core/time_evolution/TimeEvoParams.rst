paces.core.time_evolution.TimeEvoParams
=======================================

.. py:class:: paces.core.time_evolution.TimeEvoParams

   Dataclass containg basic, generic parameters for the time evolution.

   :param maxstates: Nominal truncation number when truncating basis states.
   :type maxstates: uint
   :param enlarge_steps: The number of additional matrix elements to incorporate when
                         determining the next Hilbert subspace. Note that enlarge_steps = neighbor_degree - 1.
   :type enlarge_steps: uint
   :param diagnostics: If True, compute and save extra diagnostic data. Default: True.
   :type diagnostics: bool
   :param shuffle_seed: When determining the new Hilbert space, equal-weight values
                        will be shuffled to avoid bias using this value as the initial seed.
                        None will disable shuffling and keep the (biased) lexicographic order. Default: 0.
   :type shuffle_seed: int or None
   :param weighting_name: Weighting method to use when determining the basis states to
                          be truncated. Must be one of ["coherence", "norm"]. Default: "coherence".
   :type weighting_name: str
   :param tau: time tau to use for the forward-looking part of the Hilbert
               subspace determination, where 0 disables forward-looking. Default: 0.
   :type tau: float
   :param garbage_tol: The value to use as a garbage cutoff when determining the Hilbert
                       subspace evolution. Negative values disable the garbage function and keep all
                       basis states up to maxstates. Experience has shown this probably shouldn't be enabled.
                       Default: -1 (disable garbage_tol).
   :type garbage_tol: float
   :param use_two_streams: Whether to spread the computation over two GPUs.
                           This doesn't seem to work properly, do not enable. Default: False.
   :type use_two_streams: bool

