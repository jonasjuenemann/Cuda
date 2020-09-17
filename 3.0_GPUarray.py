from __future__ import print_function
from __future__ import print_function
from time import time

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.scan import InclusiveScanKernel

"""
Fortunately, PyCUDA covers all of the overhead of memory allocation, deallocation, and data transfers with the gpuarray class. 
this class acts similarly to NumPy arrays, using vector/ matrix/tensor shape structure information for the data. 
gpuarray objects even perform automatic cleanup based on the lifetime, so no worry about freeing any GPU memory stored in a gpuarray object when done 
"""

# a_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))
# a_doubled = (2 * a_gpu).get()
# print(a_gpu)
# print(a_doubled)

# First, we must contain our host data in some form of NumPy array, wichtig: 32bit DT,
# wird spaeter beim C Code auch noch wichtig, da da floats(die 32 bits haben), diese auch brauchen, da statische Programmiersprache
host_data = np.array([1, 2, 3, 4], dtype=np.int32)
# transfer this over to the GPU and create a new GPU array.
device_data = gpuarray.to_gpu(host_data)
# a pointwise operation is intrinsically parallelizable, and so when we use this operation on a gpuarray object
# PyCUDA is able to offload each multiplication operation onto a single thread,
# rather than computing each multiplication in serial
device_datax2 = device_data * 2
# retrieve the GPU data into a new with the gpuarray.get
host_datax2 = device_datax2.get()
print(host_datax2)


x_host = np.array([1, 2, 3, 4], dtype=np.float32)
y_host = np.array([5, 6, 7, 8], dtype=np.float32)
x_device = gpuarray.to_gpu(x_host)
y_device = gpuarray.to_gpu(y_host)
print(x_host + y_host)
print((x_device + y_device).get())
print((y_device/2).get())
#s. TimeCalc fuer Zeitreduktion bei diesem vorgehen