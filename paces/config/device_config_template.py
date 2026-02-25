"""Calculation-wide global settings that must be set according to the system the code is run on."""

import cupy # pylint: disable=import-error

__all__ = ["whoami_device", "vector_device", "mempool"]

print("######### CUDA device setup: #########")

################################################################
# As an end-user, perform configuration in this section:
################################################################

# To change current device: DO NOT USE cupy.cuda.Device.use(); use "with xxxxxx_device:" instead!
# the default device is given by: cupy.cuda.Device()
whoami_device   = cupy.cuda.Device(0)
vector_device   = cupy.cuda.Device(0)
MEMSIZE_BYTES   = 63*1024**3

USE_UNIFIED_MEM = False

################################################################
# Do not change anything in the remainder of the file:
################################################################

if vector_device == whoami_device:
    print("    Using one GPU.")
else:
    print("    Using two GPUs.")

if USE_UNIFIED_MEM:
    mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed) # get unified pool
    cupy.cuda.set_allocator(mempool.malloc) # set unified pool as default allocator
    mempool.set_limit(size=MEMSIZE_BYTES)
    print("    Using unified (hybrid CPU/GPU) memory pool with limit set to"
                f" {MEMSIZE_BYTES/1024**3} GiB.")
else:
    mempool = cupy.get_default_memory_pool()
    print("    Using default (GPU-based) memory pool.")


print("#########    end of setup    #########\n")
