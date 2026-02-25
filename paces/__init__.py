"""
paces: Parallelized Application of Co-Evolving Subspaces, a method for computing quantum dynamics.

There are four submodules:
    - aux: Auxiliary functions that may be required across different situations.
    - core: Core classes and functions.
    - models: The extensible part of the module.
            Each submodule within models should inherit from the classes defined in core
            and extend their functionality to a specific Hamiltonian model.
    - config: Settings that will be applied globally.
            The end-user must copy the file device_config_template.py within config to
            device_config.py and change it according to the available hardware.
"""

__all__ = ["aux", "core", "models", "config"]
