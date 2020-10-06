import pycuda
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import threading

"""
Of course, we may seek to gain concurrency on the host side by using multiple processes or threads on the host's CPU. 
Let's make the distinction right now between a host-side operating system process and thread with a quick overview.

Every host-side program that exists outside the operating system kernel is executed as a process, and can also exist 
in multiple processes. A process has its own address space, as it runs concurrently with, and independently of, all 
other processes. A process is, generally speaking, blind to the actions of other processes, although multiple processes 
can communicate through sockets or pipes. In Linux and Unix, new processes are spawned with the fork system call.

In contrast, a host-side thread exists within a single process, and multiple threads can also exist within a single process. 
Multiple threads in a single process run concurrently. All threads in the same process share the same address space within 
the process and have access to the same shared variables and data. Generally, resource locks are used for accessing data 
among multiple threads, so as to avoid race conditions. In compiled languages such as C, C++, or Fortran, multiple process 
threads are usually managed with the Pthreads or OpenMP APIs.

Threads are much more lightweight than processes, and it is far faster for an operating system kernel to switch tasks between 
multiple threads in a single process, than to switch tasks between multiple processes. Normally, an operating system kernel 
will automatically execute different threads and processes on different CPU cores to establish true concurrency.

A peculiarity of Python is that while it supports multi-threading through the threading module, all threads will execute on the same CPU core.
(Unfortunately, the multiprocessing module is currently not fully functional under Windows, due to how Windows handles 
processes. Windows users will sadly have to stick to single-core multithreading here if they want to have any form of 
host-side concurrency.)
"""


# Multiple contexts for host-side concurrency

# how to create a single host thread in Python that can return a value to the host
class PointlessExampleThread(threading.Thread):
    # We call the parent class's constructor and set up an empty variable within the object that
    # will be the return value from the thread
    def __init__(self):
        threading.Thread.__init__(self)
        self.return_value = None

    # set up the run function within our thread class, which is what will be executed
    # when the thread is launched. We'll just have it print a line and set the return value
    def run(self):
        print 'Hello from the thread you just spawned!'
        self.return_value = 123

    # set up the join function. This will allow us to receive a return value from the thread
    def join(self):
        threading.Thread.join(self)
        return self.return_value


NewThread = PointlessExampleThread()
NewThread.start()
thread_output = NewThread.join()
print 'The thread completed and returned this value: %s' % thread_output


# expand this idea among multiple concurrent threads on the host to launch concurrent CUDA operations by way of multiple contexts and threading.

# Generally, we don't want to spawn more than 20 or so threads on the host, so we will only go for 10 arrays
num_arrays = 10
array_len = 1024 ** 2

# store our old kernel as a string object; since this can only be compiled within a context, we will have to compile this in each thread individually
kernel_code = """       
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
"""

# make another subclass of threading.Thread as before, and set up the constructor to take one parameter as
# the input array. We will initialize an output variable with None, as we did before.
class KernelLauncherThread(threading.Thread):
    def __init__(self, input_array):
        threading.Thread.__init__(self)
        self.input_array = input_array
        self.output_array = None

    # We choose our device, create a context on that device, compile our kernel, and extract the kernel function reference
    def run(self):
        self.dev = drv.Device(0)
        self.context = self.dev.make_context()
        self.ker = SourceModule(kernel_code)
        self.mult_ker = self.ker.get_function('mult_ker')

        #copy the array to the GPU, launch the kernel, and copy the output back to the host. We then destroy the context
        self.array_gpu = gpuarray.to_gpu(self.input_array)
        self.mult_ker(self.array_gpu, np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1))
        self.output_array = self.array_gpu.get()
        self.context.pop()

    # This will return output_array to the host
    def join(self):
        threading.Thread.join(self)
        return self.output_array


drv.init()

# set up some empty lists to hold our random test data, thread objects, and thread output values,
data = []
gpu_out = []
threads = []
# generate random arrays and thread objects.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))
for k in range(num_arrays):
    # create a thread that uses data we just generated
    threads.append(KernelLauncherThread(data[k]))

# launch each thread object, and extract its output into the gpu_out list by using join
# launch threads to process arrays.
for k in range(num_arrays):
    threads[k].start()
# get data from launched threads.
for k in range(num_arrays):
    gpu_out.append(threads[k].join())

# we just to a simple assert on the output arrays to ensure they are the same as the input
for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))
