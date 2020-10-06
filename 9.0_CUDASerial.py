"""
We know that within a single kernel, there is one level of concurrency among its many threads;
however, there is another level of concurrency over multiple kernels and GPU memory operations
that is also available to us. This means that we can launch multiple memory and kernel operations at
once, without waiting for each operation to finish. However, on the other hand, we will have to be
somewhat organized to ensure that all inter-dependent operations are synchronized; this means that we
shouldn't launch a particular kernel until its input data is fully copied to the device memory, or we
shouldn't copy the output data of a launched kernel to the host until the kernel has finished execution.
"""
# - CUDAstreams - a stream is a sequence of operations that are run in order on the GPU
# - events, which are a feature of streams that are used to precisely time kernels and
# indicate to the host as to what operations have been completed within a given stream.
# - context can be thought of as analogous to a process in your operating system, in that the GPU
# keeps each context's data and kernel code walled off and encapsulated away from the other contexts
# currently existing on the GPU

# Cuda device synchronisation:
# This is an operation where the host blocks any further execution until all operations issued to the GPU
# (memory transfers and kernel executions) have completed
# This is required to ensure that operations dependent on prior operations are not executed outoforder, for example,
# to ensure that a CUDA kernel launch is completed before the host tries to read its output.
# In CUDA C sieht das ganze so aus:
"""
// Copy an array of floats from the host to the device.
cudaMemcpy(device_array, host_array, size_of_array*sizeof(float), cudaMemcpyHostToDevice);
// Block execution until memory transfer to device is complete.
cudaDeviceSynchronize();
// Launch CUDA kernel.
Some_CUDA_Kernel <<< block_size, grid_size >>> (device_array, size_of_array);
// Block execution until GPU kernel function returns.
cudaDeviceSynchronize();
// Copy output of kernel to host.
cudaMemcpy(host_array,  device_array, size_of_array*sizeof(float), cudaMemcpyDeviceToHost);
// Block execution until memory transfer to host is complete.
cudaDeviceSynchronize();
"""
# we haven't seen this yet, because PyCUDA has been invisibly calling this for us automatically as needed.

# But if we want to concurrently launch multiple independent kernels and memory operations operating on different arrays
# of data,  it would be inefficient to synchronize across the entire device.
# In this case, we should synchronize across multiple streams. We'll see how to do this right now.

# generate a series of random GPU arrays, process each array with a simple kernel, and copy the arrays back to the host
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

# We now will specify how many arrays we wish to process,each array will be processed by a different kernel launch
# We also specify the length of the random arrays we will generate, as follows:
num_arrays = 200
array_len = 1024 ** 2

# all this will do is iterate over each point in the array, and multiply and divide it by 2 for 50 times
# We want to restrict the number of threads that each kernel launch will use, which will help us gain concurrency
# among many kernel launches on the GPU so that we will have each thread
# iterate over different parts of the array with a for loop
ker = SourceModule("""       
__global__ void mult_ker(float * array, int array_len)
{
     int thd = blockIdx.x*blockDim.x + threadIdx.x; // blockIdx.x*blockDim.x hier noch nicht relevant
     int num_iters = array_len / blockDim.x; //Anzahl Bloecke, bei uns 1048576/64 = 16384 (fuer die 64, s. blockDim unten bei blocksize)

     for(int j=0; j < num_iters; j++) //jeder der 64 threads bearbeitet nicht einen, sondern 16384 Punkte manuell
     {
         int i = j * blockDim.x + thd; //thread 0 bearbeitet also 0, 64, 128, ..., 1048512
        
        //letztlich irrelevant, es wird 50mal multipliziert und geteilt, Endergebnis ist gleich
         for(int k = 0; k < 50; k++)
         {
              array[i] *= 2.0;
              array[i] /= 2.0;
         }
     }
}
""")
# was ist relevant an dem Kernel? > wir bearbeiten ein 1048576 grosses Array mit "nur" 64 threads, dauert aber auch entsprechend,
# positiv: durch unsere Schleifenkonstruktion wird trotzdem jeder Index des Arrays bearbeitet

mult_ker = ker.get_function('mult_ker')

# we will generate some random data array, copy these arrays to the GPU, iteratively launch our kernel over each array
# across 64 threads, and then copy the output data back to the host and assert that the same with NumPy's allclose function.
# We will time the duration of all operations from start to finish by using Python's time function, as follows:

data = []  # 200 mal ein ~1mio grosses Array
data_gpu = []  # Kopie von data
gpu_out = []

# generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

t_start = time()

# copy arrays to GPU.
for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu(data[k]))

# iteratively launch our kernel over each array across 64 threads,
# and then copy the output data back to the host and assert that the same with NumPy's allclose function.
# process arrays.
for k in range(num_arrays):
    mult_ker(data_gpu[k], np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1))

# copy arrays from GPU.
for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get())

t_end = time()

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

print "Single Kernel Ausfuehrung (braucht ~3sec.)"
print 'Total time: %f' % (t_end - t_start)

