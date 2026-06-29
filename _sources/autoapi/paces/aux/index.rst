paces.aux
=========

.. py:module:: paces.aux

.. autoapi-nested-parse::

   aux: Auxiliary functions that may be required across different situations.

   The _scipy submodule is adapted from SciPy. It contains matrix-exponential functions and is subject
   to the BSD-3-clause license as mentioned in the root directory of this module.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/paces/aux/cupy_search/index
   /autoapi/paces/aux/helpers/index


Functions
---------

.. autoapisummary::

   paces.aux.expm_multiply
   paces.aux.expm_multiply_simple


Package Contents
----------------

.. py:function:: expm_multiply(A, B, start=None, stop=None, num=None, endpoint=None, return_dbg=False)

   Compute the action of the matrix exponential of A on B. See scipy.sparse.linalg.expm_multiply.

   :param A: The operator whose exponential is of interest.
   :type A: transposable linear operator
   :param B: The matrix or vector to be multiplied by the matrix exponential of A.
   :type B: ndarray
   :param start: The starting time point of the sequence.
   :type start: scalar, optional
   :param stop: The end time point of the sequence, unless `endpoint` is set to False.
                In that case, the sequence consists of all but the last of ``num + 1``
                evenly spaced time points, so that `stop` is excluded.
                Note that the step size changes when `endpoint` is False.
   :type stop: scalar, optional
   :param num: Number of time points to use.
   :type num: int, optional
   :param endpoint: If True, `stop` is the last time point.  Otherwise, it is not included.
   :type endpoint: bool, optional

   :returns: **expm_A_B** -- The result of the action :math:`e^{t_k A} B`.
   :rtype: ndarray

   .. rubric:: References

   .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
          "Computing the Action of the Matrix Exponential,
          with an Application to Exponential Integrators."
          SIAM Journal on Scientific Computing,
          33 (2). pp. 488-511. ISSN 1064-8275
          http://eprints.ma.man.ac.uk/1591/

   .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
          "Computing Matrix Functions."
          Acta Numerica,
          19. 159-208. ISSN 0962-4929
          http://eprints.ma.man.ac.uk/1451/


.. py:function:: expm_multiply_simple(A, B, t=1.0, balance=False, return_dbg=False)

   Compute the action of the matrix exponential at a single time point.

   :param A: The operator whose exponential is of interest.
   :type A: transposable linear operator
   :param B: The matrix to be multiplied by the matrix exponential of A.
   :type B: ndarray
   :param t: A time point.
   :type t: float
   :param balance: Indicates whether or not to apply balancing.
   :type balance: bool
   :param return_dbg: If True, additional debugging/diagnostic information is returned.
   :type return_dbg: bool

   :returns: * **F** (*ndarray*) -- :math:`e^{t A} B`
             * *If return_dbg, then the following are also returned *as a separate list**
             * **converged** (*bool*) -- If True, the algorithm converged; if False, then it exhausted the maximal order.
             * **final_m** (*int*) -- The final order of the expansion.
             * **c1_plus_c2** (*complex*) -- The sum of the last two terms in the series.
             * **term_ratio** (*complex*) -- The relative contribution of the last two terms to the result, measured by the inf norm.

   .. rubric:: Notes

   This is algorithm (3.2) in Al-Mohy and Higham (2011).


