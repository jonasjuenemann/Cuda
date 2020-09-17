"""
how to program our own point-wise (or equivalently, element-wise) operations directly onto our GPU
with the help of PyCUDA's ElementWiseKernel function
We will use several functions from PyCUDA that generate templates and design patterns for different types of kernels,
easing our transition into GPU programming.
"""

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel

host_data = np.float32(np.random.random(50000000))
# We first set the input and output variables in the first line ( "float *in, float *out" ),
# which will generally be in the form of C pointers to allocated memory on the GPU.
# In the second line, we define our element-wise operation with "out[i] = 2*in[i];",
# which will multiply each point in in by two and place this in the corresponding index of out
# we give our piece of code its internal CUDA C kernel name ( "gpu_2x_ker" ).
# this refers to CUDA C's namespace and not Python's, it's fine and convenient to give this the same name as in Python.
gpu_2x_ker = ElementwiseKernel(
    "float *in, float *out",
    "out[i] = 2*in[i];",
    "gpu_2x_ker")


# Note that PyCUDA automatically sets up the integer index i for us.
# When we use i as our index, ElementwiseKernel will automatically parallelize our
# calculation over i among the many cores in our GPU

def speedcomparison():
    t1 = time()
    host_data_2x = host_data * np.float32(2)
    t2 = time()
    print 'total time to compute on CPU: %f' % (t2 - t1)
    # automatically allocates data onto the GPU and copies it over from the CPU space
    device_data = gpuarray.to_gpu(host_data)
    # The little kernel function we defined operates on C float pointers; -> allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)
    t1 = time()
    gpu_2x_ker(device_data, device_data_2x) # function from line21 is called
    t2 = time()
    from_device = device_data_2x.get()
    print 'total time to compute on GPU: %f' % (t2 - t1)
    print 'Is the host computation the same as the GPU computation? : {}'.format(np.allclose(from_device, host_data_2x))


if __name__ == '__main__':
    speedcomparison()

"""
Pretty fucking slow tho:
total time to compute on CPU: 0.121000
total time to compute on GPU: 5.506000
Is the host computation the same as the GPU computation? : True

total time to compute on CPU: 0.121000
total time to compute on GPU: 2.178000
Is the host computation the same as the GPU computation? : True

total time to compute on CPU: 0.122000
total time to compute on GPU: 1.332000
Is the host computation the same as the GPU computation? : True
"""

