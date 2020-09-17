from __future__ import print_function
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

a = numpy.random.randn(4, 4)
# as nVidia devices tend to only support "single precision" (no doubles)
a = a.astype(numpy.float32)
# we need somewhere to transfer data to, so we need to allocate memory:
a_gpu = drv.mem_alloc(a.nbytes) # numpy .nbytes -> Total bytes consumed by the elements of the array.
# transfer the data to the GPU (wir uebergeben a und sagen, dass a auf a_gpu (allocated memory) gespeichert werden soll
drv.memcpy_htod(a_gpu, a)
print(a)
# simple: code to double each entry in a_gpu
SourceMod = SourceModule("""
  __global__ void doublify(float *a) //__global__ is device Code (kernels), while code on CPU is called host code
  {
    int idx = threadIdx.x + threadIdx.y*blockDim.x; //Verhalten von threadIdx.y*4, ist groesse der 2D der Matrix (threadIdx.y=3, es werden so nur die ersten 3 werte aus y=1 verarbeitet), daher Mult. *4
    a[idx] *= 2;
  }
  """)
"""
Von Interesse: In C++ gibt es die <<<1, 1>>> Syntax, die CUDA sagt, wie viele parallele Threads benutzt werden sollen. 
Der erste Parameter bestimmt Anzahl an thread blocks
Der zweite bestimmt dabei die Anzahl an Threads in einem ThreadBlock (multiples of 32 in Size).
threadIdx.x contains the index of the current thread within its block
blockDim.x contains the number of threads in the block
blockIdx.x contains index of the current block
gridDim.x number of blocks in the grid
each thread gets its index by blockIdx.x * blockDim.x (+ threadIdx.x)
total number of threads in the grid: blockDim.x * gridDim.x
"""

# code is now compiled and loaded onto the device
# find a reference to our pycuda.driver.Function and call it, specifying a_gpu as the argument, and a block size of 4x4
func = SourceMod.get_function("doublify")
func(a_gpu, block=(4, 4, 1))
"""
instead of creating a_gpu, if replacing a is fine:
func(drv.InOut(a), block=(4, 4, 1))
"""
# fetch the data back from the GPU and display it
a_doubled = numpy.empty_like(a)  # Return a new array with the same shape and type as a given array.
# print(a_doubled)
drv.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
