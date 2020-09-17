# coding=utf-8
from time import time
import matplotlib

# this will prevent the figure from popping up
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

# internal PyCUDA class template is used for complex types—here PyCUDA ::complex<float> corresponds to Numpy complex64

mandel_ker = ElementwiseKernel(
    "pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",  # inputs
    # Actual Code In """ to use multiple lines within the string
    # the arrays we have passed in are two-dimensional arrays in Python,
    # CUDA will only see these as being one-dimensional and indexed by i.
    # Again, ElementwiseKernel indexes i across multiple cores and threads for us automatically
    # will run serial over j though
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
    "mandel_ker")  # internal CUDA C kernel name


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    real_vals = np.matrix(np.linspace(real_low, real_high, width), dtype=np.complex64)
    imag_vals = np.matrix(np.linspace(imag_high, imag_low, height),
                          dtype=np.complex64) * 1j  # imaginary number for complex literals
    mandelbrot_lattice = np.array(real_vals + imag_vals.transpose(), dtype=np.complex64)

    # copy complex lattice to the GPU
    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)
    # allocate an empty array on the GPU
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape,
                                          dtype=np.float32)  # empty -> specifying the size/shape of the array and the type
    # we won't have to deallocate or free this memory later, due to the gpuarray object destructor taking care
    # of memory clean-up automatically when the end of the scope is reached.
    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))
    # the first input will be the complex lattice of points (NumPy complex64 type) we generated
    # second will be a pointer to a two-dimensional floating point array (NumPy float32 type) that will indicate which elements are members of the Mandelbrot set
    # third will be an integer indicating the maximum number of iterations for each point
    # final input will be the upper bound for each point used for determining membership in the Mandelbrot class
    # important: careful typecasting
    mandelbrot_graph = mandelbrot_graph_gpu.get()
    # retrieves the Mandelbrot set we generated from the GPU back into CPU spac
    return mandelbrot_graph


if __name__ == '__main__':
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
