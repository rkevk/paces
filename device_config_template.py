#!/usr/bin/env python3
# coding: utf-8
 
import cupy

print("######### CUDA device setup: #########")

################################################################
# As an end-user, perform configuration in this section:
################################################################

whoami_device   = cupy.cuda.Device(0)       # change current device: DO NOT USE cupy.cuda.Device.use(); use "with xxxxxx_device:" instead!
                                            # the default device is given by: cupy.cuda.Device()
vector_device   = cupy.cuda.Device(0)
memsize_bytes   = 63*1024**3

use_unified_mem = False

################################################################
# Do not change anything in the remainder of the file:
################################################################

if vector_device == whoami_device:
    print("    Using one GPU.")
else:
    print("    Using two GPUs.")

if use_unified_mem:
    mempool = cupy.get_default_memory_pool()
    print("    Using default (GPU-based) memory pool.")
else:
    mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed) # get unified pool
    cupy.cuda.set_allocator(mempool.malloc) # set unified pool as default allocator
    mempool.set_limit(size=memsize_bytes)
    print("    Using unified (hybrid CPU/GPU) memory pool with limit set to %1.1f GiB." % (memsize_bytes/1024**3))

print("#########    end of setup    #########\n")
