import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

num_arrays = 200
array_len = 1024 ** 2

ker = SourceModule("""       
__global__ void mult_ker(float * array, int array_len)
{
     int thd = blockIdx.x*blockDim.x + threadIdx.x;
     int num_iters = array_len / blockDim.x;

     for(int j=0; j < num_iters; j++)
     {
         int i = j * blockDim.x + thd;

         for(int k = 0; k < 50; k++)
         {
              array[i] *= 2.0;
              array[i] /= 2.0;
         }
     }

}
""")

mult_ker = ker.get_function('mult_ker')

# Bis hierhin alles gleich wie in 9.0

data = []
data_gpu = []
gpu_out = []
# we note that for each kernel launch we have a separate array of data that it processes, and these
# are stored in Python lists. We will have to create a separate stream object for each individual
# array/kernel launch pair, so let's first add an empty list
streams = []

# We can now generate a series of streams that we will use to organize the kernel launches.
# We can get a stream object from the pycuda.driver submodule with the Stream class
for _ in range(num_arrays):
    streams.append(drv.Stream())

# generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

t_start = time()

# copy arrays to GPU.
# switch to the asynchronous and stream-friendly version of this function,
# gpu_array.to_gpu_async, instead. (We must now also specify which stream
# each memory operation should use with the stream parameter)
for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

# process arrays.
# This is exactly as before, only we must specify what stream to use by using the stream parameter
for k in range(num_arrays):
    mult_ker(data_gpu[k], np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1), stream=streams[k])

# copy arrays from GPU.
# We can do this by switching the gpuarray get function to get_async, and again using the stream parameter
for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

t_end = time()

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

print 'Total time: %f' % (t_end - t_start)

"""
Wesentlich Schneller: Total time: 0.901000
Warum: Paralleles Ablaufen der Kernels auf den 200 Arrays (Auch memory allocation/disallocation parallel->async)
"""
