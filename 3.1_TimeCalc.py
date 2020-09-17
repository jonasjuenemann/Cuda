# coding=utf-8
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time


host_data = np.float32(np.random.random(50000000))
# 50Mio großes Array, ~48MB

t1 = time()
host_data_2x =  host_data * np.float32(2)
t2 = time()

print 'total time to compute on CPU: %f' % (t2 - t1)

device_data = gpuarray.to_gpu(host_data)

t1 = time()
device_data_2x =  device_data * np.float32(2)
t2 = time()

from_device = device_data_2x.get()

print 'total time to compute on GPU: %f' % (t2 - t1)
print 'Is the host computation the same as the GPU computation? : {}'.format(np.allclose(from_device, host_data_2x))

"""
Ist beim ersten Mal langsamer sein als CPU:
total time to compute on CPU: 0.122000
total time to compute on GPU: 1.601000
Warum?:
there are a number of suspicious calls to a Python module file, compiler.py; 
these take roughly one second total, a little less than the time it takes to do the GPU computation 
By the nature of the PyCUDA library, GPU code is often compiled and linked with NVIDIA's nvcc compiler the first time it is run in a given Python session;
 
it is then cached and, if the code is called again, then it doesn't have to be recompiled.
total time to compute on CPU: 0.114000
total time to compute on GPU: 0.007000
Is the host computation the same as the GPU computation? : True
"""