paces
=====

.. py:module:: paces

.. autoapi-nested-parse::

   paces: Parallelized Application of Co-Evolving Subspaces, a method for computing quantum dynamics.

   There are four submodules:
       - aux: Auxiliary functions that may be required across different situations.
       - core: Core classes and functions.
       - models: The extensible part of the module.
               Each submodule within models should inherit from the classes defined in core
               and extend their functionality to a specific Hamiltonian model.
       - config: Settings that will be applied globally.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/paces/aux/index
   /autoapi/paces/config/index
   /autoapi/paces/core/index
   /autoapi/paces/models/index


