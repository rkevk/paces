paces.config.DeviceConfig
=========================

.. py:class:: paces.config.DeviceConfig

   Object that holds the device and memory settings used by everything else.

   The initialized version of this is a placeholder.
   A usable version must be created by calling the `configure` method after instantiating.

   Placeholder instantiation; call `configure` to actually set variables.


   .. py:method:: configure(whoami_device_id=0, vector_device_id=0, unified_memory_bytes=None)

      Configure the memory and device settings for the paces calculation.

      :param whoami_device_id: The device for storing and manipulating the lookup tables etc.
                               Default: 0 (selects the first GPU that is available to the calculation).
      :type whoami_device_id: uint
      :param vector_device_id: The device for storing and manipulating vectors and matrices.
                               If this is chosen to be different from whoami_device_id,
                               then the calculation is spread over both GPUs, which increases available memory,
                               but isn't very time-efficient due to inter-host memory transfer.
                               Default: 0 (selects the first GPU that is available to the calculation).
      :type vector_device_id: uint
      :param unified_memory_bytes: The unified memory in bytes to use.
                                   Unified memory spreads the data over both the GPU (device) and CPU (host) RAM.
                                   This allows for much larger arrays, but comes at the cost of speed due to slow
                                   memory transfer between host and device.
                                   Setting this to None disables unified memory entirely. Default: None.
      :type unified_memory_bytes: uint or none


