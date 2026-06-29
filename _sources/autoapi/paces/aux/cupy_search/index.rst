paces.aux.cupy_search
=====================

.. py:module:: paces.aux.cupy_search

.. autoapi-nested-parse::

   Low-level algorithms and CUDA kernels relating to searching in cupy and CUDA.



Functions
---------

.. autoapisummary::

   paces.aux.cupy_search.calc_partition_lens
   paces.aux.cupy_search.find_changes_local_single


Module Contents
---------------

.. py:function:: calc_partition_lens(partition_by, uniquevals)

   Calculates the length of the blocks of repeating elements in an array.

   :param partition_by: A sorted array whose partition length are to be computed.
   :type partition_by: 1D cupy.ndarray
   :param uniquevals: The array of unique values that should occur in partition_by.
                      uniquevals must be sorted and unique.
                      Not every value in uniquevals must necessarily occur in partition_by
                      (i.e., some partition lengths may be zero),
                      but every value in partition_by must indeed occur in uniquevals.
   :type uniquevals: 1D cupy.ndarray

   :returns: The length of each repeating block in partition_by.
   :rtype: 1D cupy.ndarray


.. py:function:: find_changes_local_single(arr, mindiff=0)

   Takes a (sorted) 1D array and returns the segmentation indices where the value changes.


