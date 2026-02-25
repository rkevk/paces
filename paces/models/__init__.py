"""
models: The extensible part of the module, representing specific Hamiltonians.

Each submodule within models should inherit from the classes defined in core
and extend their functionality to a specific Hamiltonian model.
"""

__all__ = ["holstein"]

from . import holstein
