paces.config.dbg_verbosity_levels
=================================

.. py:module:: paces.config.dbg_verbosity_levels

.. autoapi-nested-parse::

   Debug verbosity levels used throughout the program.

   (Note that the auto-generated documentation is incomplete: See inline comments in the source file.)

   These values may be modified to customize the default debugging verbosity levels,
   but the main parameter that governs the verbosity of a single calculation is the
   debug_verb argument passed to a subclass of HamiltonianFramework upon instantiation.

   During a calculation, the value of debug_verb is compared to the levels given in this file:
   Therefore, setting a level in this file to a larger number will cause it to *stop* being printed.



