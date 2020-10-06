"""
A CUDA context is usually described as being analogous to a process in an operating system.
a process is an instance of a single program running on a computer; all programs outside of the operating system kernel run in a process.
Each process has its own set of instructions, variables, and allocated memory, and is, generally speaking, blind to the actions and memory of other processes.
When a process ends, the operating system kernel performs a cleanup, ensuring that all memory that the process allocated
has been de-allocated, and closing any files, network connections, or other resources the process has made use of.

Similar to a process, a context is associated with a single host program that is using the GPU.
A context holds in memory all CUDA kernels and allocated memory that is making use of and is blind to the kernels and memory of other currently existing contexts.
When a context is destroyed (at the end of a GPU based program, for example), the GPU performs a cleanup of all code and allocated memory within the context,
freeing resources up for other current and future contexts. The programs that we have been writing so far have all existed within a single
context, so these operations and concepts have been invisible to us.

a single CUDA host program can generate and use multiple CUDA contexts on the GPU.
Usually, we will create a new context when we want to gain host-side concurrency when we fork new processes or threads of a host process.
"""

# Synchronising current context
# We're going to see how to explicitly synchronize our device within a context from within Python as in CUDA C;
# this is actually one of the most fundamental skills to know in CUDA C, and is covered in the first or second
# chapters in most other books on the topic. So far, we have been able to avoid this topic, since PyCUDA has
# performed most synchronizations for us automatically with pycuda.gpuarray functions such as to_gpu or get

# Example with the Mandelbrot Example:
from time import time
import matplotlib
# this will prevent the figure from popping up
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

mandel_ker = ElementwiseKernel(
    "pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
    """
    mandelbrot_graph[i] = 1;
    
    pycuda::complex<float> c = lattice[i]; 
    pycuda::complex<float> z(0,0);
    
    for (int j = 0; j < max_iters; j++)
        {
    
         z = z*z + c;
    
         if(abs(z) > upper_bound)
             {
              mandelbrot_graph[i] = 0;
              break;
             }
    
        }
    
    """,
    "mandel_ker")


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    # we set up our complex lattice as such
    real_vals = np.matrix(np.linspace(real_low, real_high, width), dtype=np.complex64)
    imag_vals = np.matrix(np.linspace(imag_high, imag_low, height), dtype=np.complex64) * 1j
    mandelbrot_lattice = np.array(real_vals + imag_vals.transpose(), dtype=np.complex64)

    # copy complex lattice to the GPU
    # changed this to be explicitly synchronized. (to: to_gpu_async)
    # We can copy to the GPU asynchronously with to_gpu_async, and then synchronize as follows:
    mandelbrot_lattice_gpu = gpuarray.to_gpu_async(mandelbrot_lattice)
    # We can access the current context object with pycuda.autoinit.context,
    # and we can synchronize in our current context by calling the pycuda.autoinit.context.synchronize() function.
    # synchronize in current context
    pycuda.autoinit.context.synchronize()

    # allocate an empty array on the GPU
    # allocates memory on the GPU with the gpuarray.empty function. Memory allocation in CUDA is,
    # by the nature of the GPU architecture, automatically synchronized; there is no asynchronous memory
    # allocation equivalent here.
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)

    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))

    pycuda.autoinit.context.synchronize()

    mandelbrot_graph = mandelbrot_graph_gpu.get_async()

    pycuda.autoinit.context.synchronize()

    return mandelbrot_graph


t1 = time()
mandel = gpu_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2)
t2 = time()

mandel_time = t2 - t1

t1 = time()
fig = plt.figure(1)
plt.imshow(mandel, extent=(-2, 2, -2, 2))
plt.savefig('mandelbrot.png', dpi=fig.dpi)
t2 = time()

dump_time = t2 - t1

print 'It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time)
print 'It took {} seconds to dump the image.'.format(dump_time)
