"""
config: Global configuration settings.
"""

from .dbg_verbosity_levels import *
from ._dev_conf import DeviceConfig

# Initialize without setting anything; the user must call device_config.configure to set values:
devices = DeviceConfig()
