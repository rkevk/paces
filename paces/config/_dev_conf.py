"""Calculation-wide global settings that must be set according to the system the code is run on."""

import cupy # pylint: disable=import-error

class DeviceConfig:
    """
    Object that holds the device and memory settings used by everything else.

    The initialized version of this is a placeholder.
    A usable version must be created by calling the `configure` method after instantiating.
    """
    def __init__(self):
        """Placeholder instantiation; call `configure` to actually set variables."""
        # These are set by configure:
        self.whoami_dev = None  # will be a cupy/CUDA device object
        self.vector_dev = None  # will be a cupy/CUDA device object
        self.mempool    = None  # will be a cupy/CUDA memory pool object
        self.split      = False # convenience flag equivalent to self.whoami_dev != self.vector_dev

        # Flag to check if configuration was ever set:
        self.configure_called = False

    def configure(self, whoami_device_id=0, vector_device_id=0, unified_memory_bytes=None):
        """
        Configure the memory and device settings for the paces calculation.

        Args:
            whoami_device_id (uint): The device for storing and manipulating the lookup tables etc.
                Default: 0 (selects the first GPU that is available to the calculation).
            vector_device_id (uint): The device for storing and manipulating vectors and matrices.
                If this is chosen to be different from whoami_device_id,
                then the calculation is spread over both GPUs, which increases available memory,
                but isn't very time-efficient due to inter-host memory transfer.
                Default: 0 (selects the first GPU that is available to the calculation).
            unified_memory_bytes (uint or none): The unified memory in bytes to use.
                Unified memory spreads the data over both the GPU (device) and CPU (host) RAM.
                This allows for much larger arrays, but comes at the cost of speed due to slow
                memory transfer between host and device.
                Setting this to None disables unified memory entirely. Default: None.
        """
        print("######### CUDA device setup: #########")

        self.whoami_dev = cupy.cuda.Device(whoami_device_id)
        self.vector_dev = cupy.cuda.Device(vector_device_id)

        if self.vector_dev == self.whoami_dev:
            print("    Using one GPU.")
        else:
            print("    Using two GPUs.")

        if unified_memory_bytes is not None:
            self.mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed) # get unified pool
            cupy.cuda.set_allocator(self.mempool.malloc) # set unified pool as default allocator
            self.mempool.set_limit(size=unified_memory_bytes)
            self.split = True
            print("    Using unified (hybrid CPU/GPU) memory pool with limit set to"
                        f" {unified_memory_bytes/1024**3} GiB.")
        else:
            self.mempool = cupy.get_default_memory_pool()
            print("    Using default (GPU-based) memory pool.")

        self.configure_called = True
        print("#########    end of setup    #########\n")
