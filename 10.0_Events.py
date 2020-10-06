import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

"""
Events are objects that exist on the GPU, whose purpose is to act as milestones or progress markers for a stream of operations.
Events are generally used to provide measure time duration on the device side to precisely time operations.
Additionally, events they can also be used to provide a status update for the host as to the state of a stream and what 
operations it has already completed, as well as for explicit stream-based synchronization.
"""

# start with an example that uses no explicit streams and uses events to measure only one single kernel launch.
# (If we don't explicitly use streams in our code, CUDA actually invisibly defines a default stream that
# all operations will be placed into)
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

array_len = 100*1024**2
data = np.random.randn(array_len).astype('float32')
data_gpu = gpuarray.to_gpu(data)

start_event = drv.Event()
end_event = drv.Event()
# we have to mark the start_event instance's place in the stream of execution with the event record function.
start_event.record()
mult_ker(data_gpu, np.int32(array_len), block=(64,1,1), grid=(1,1,1))
end_event.record()

# Ohne diese Line ist der Kernel zum Zeitpunkt des prints nocht nicht mal gelaunched.
# Kernels in PyCUDA have launched asynchronously (whether they exist in a specific stream or not),
# so we have to have to ensure that our host code is properly synchronized with the GPU.
end_event.synchronize()


# Events have a binary value that indicates whether they were reached or not yet, which is given by the function query.
# Let's print a status update for both events, immediately after the kernel launch:
print 'Has the kernel started yet? {}'.format(start_event.query())
print 'Has the kernel ended yet? {}'.format(end_event.query())

print 'Kernel execution time in milliseconds: %f ' % start_event.time_till(end_event)


# Events und Streams:
# how to use event objects with respect to streams
#  highly intricate level of control over the flow of our various GPU operations, allowing us to know
#  exactly how far each individual stream has progressed via the query function, and even allowing us
#  to synchronize particular streams with the host while ignoring the other streams.

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

# each stream has to have its own dedicated collection of event objects;
# multiple streams cannot share an event object

data = []
data_gpu = []
gpu_out = []
streams = []
start_events = []
end_events = []

for _ in range(num_arrays):
    streams.append(drv.Stream())
    start_events.append(drv.Event())
    end_events.append(drv.Event())

# generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

t_start = time()

# copy arrays to GPU.
for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

# process arrays.
# we can time each kernel launch individually by modifying the second loop to use the record of the
# event at the beginning and end of the launch
# since there are multiple streams, we have to input the appropriate stream as a parameter to each
# event object's record function
for k in range(num_arrays):
    start_events[k].record(streams[k])
    mult_ker(data_gpu[k], np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1), stream=streams[k])
for k in range(num_arrays):
    end_events[k].record(streams[k])

# copy arrays from GPU.
for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

t_end = time()

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

# to extract the duration of each individual kernel launch. Let's add a new empty list
# after the iterative assert check, and fill it with the duration by way of the time_till function

kernel_times = []

for k in range(num_arrays):
    kernel_times.append(start_events[k].time_till(end_events[k]))


print 'Total time: %f' % (t_end - t_start)
print 'Mean kernel duration (milliseconds): %f' % np.mean(kernel_times)
print 'Mean kernel standard deviation (milliseconds): %f' % np.std(kernel_times)