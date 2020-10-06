import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv

# So far, we have been importing pycuda.autoinit at the beginning of all of our PyCUDA programs;
# this effectively creates a context at the beginning of our program and has it destroyed at the end.

# a small program that just copies a small array to the GPU, copies it back to the host, prints the array, and exits.
# First, we initialize CUDA with the pycuda.driver.init function, which is here aliased as drv
drv.init()
# we choose which GPU we wish to work with; this is necessary for the cases where one has more than one GPU.
# We can select a specific GPU with  pycuda.driver.Device;
# if you only have one GPU, you can access it with pycuda.driver.Device(0)
dev = drv.Device(0)
# We can now create a new context on this device with make_context
ctx = dev.make_context()
# Now that we have a new context, this will automatically become the default context
x = gpuarray.to_gpu(np.float32([1, 2, 3]))
print x.get()
# We can destroy the context by calling the pop function
ctx.pop()
