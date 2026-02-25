"""
config: Global configuration settings.

The end-user must copy the file device_config_template.py to device_config.py
and change it according to the available hardware.
"""

from .dbg_verbosity_levels import *
from .device_config import whoami_device, vector_device, mempool
