paces.aux.helpers
=================

.. py:module:: paces.aux.helpers

.. autoapi-nested-parse::

   Helper functions that may be of use across different situations.



Functions
---------

.. autoapisummary::

   paces.aux.helpers.cartesian_product
   paces.aux.helpers.print_searchsorted_timing
   paces.aux.helpers.obs_attrs
   paces.aux.helpers.debug_lister
   paces.aux.helpers.flatten_dbg_dict
   paces.aux.helpers.cupy_unique


Module Contents
---------------

.. py:function:: cartesian_product(*arrays)

   Return the Cartesian product of a list of arrays.


.. py:function:: print_searchsorted_timing(verb, delta_t, size1, size2)

   Helper function to print the time it took to apply searchsorted.


.. py:function:: obs_attrs(**kwargs)

   Decorator to add header and fname to observable functions.

   This should be called with at least the following kwargs:
       fname (str): Base name of the file where the corresponding observable expectation values
           should be saved. The entire path will be derived from this fname as
           [main_calc_dir]/observables/[fname].real and [main_calc_dir]/observables/[fname].imag.
       header (str): The header line to be printed at the top of the file, but without the
           preceding time and norm (i.e., just a description of the observables themselves).


.. py:function:: debug_lister(dbg_list)

   Decorator to add debug description to generate_mel_xxx

   :param dbg_list: list of strings providing names for the debug output.
   :type dbg_list: list


.. py:function:: flatten_dbg_dict(d)

   Flatten the dbg dicts (header or vals) into a list


.. py:function:: cupy_unique(array)

   Find and return the unique, lexicographically sorted rows in an array.

   Replacement for numpy.unique with option axis=0.


