# implement the parallel prefix algorithm, also known as the scan design pattern.
# We have already seen simple examples of this in the form of PyCUDA's
# InclusiveScanKernel and ReductionKernel functions
# we have a binary operator (a function that acts on two input values and gives one output value)
# and a collection of elements, and from these we wish to compute efficiently
# (assumption that our binary is associative)
# The aim of the parallel prefix algorithm is to produce this collection of n sums efficiently.
# It normally takes O(n) time to produce these n sums in a serial operation, and we wish to reduce the time complexity.

# (When the terms "parallel prefix" or "scan" are used, it usually means an algorithm that produces all of these n results, while "reduce"/"reduction" usually means an algorithm that only yields the single final result)

"""naive parallel prefix algorithm"""

# original version of this algorithm; this algorithm is "naive" because it makes an assumption that given n input elements,
# with the further assumption that n is dyadic (n=2^k) for some postive integer k
# and we can run the algorithm in parallel over n processors (or n threads)
# given these conditions are satisfied, we have a nice result in that its computational time complexity is only O(log n)

from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time

naive_ker = SourceModule("""
__global__ void naive_prefix(double *vec, double *out)
{
    //We declare a shared memory sum_bufarray that we'll use for the calculation of our output
     __shared__ double sum_buf[1024];     
     int tid = threadIdx.x;     
     sum_buf[tid] = vec[tid];
     
     // begin parallel prefix sum algorithm
     
     int iter = 1;
     for (int i=0; i < 10; i++)
     {
         __syncthreads();
         if (tid >= iter ) //iter = 1,2,4,8,16,32,64,128,256,512,1024
         {
             sum_buf[tid] = sum_buf[tid] + sum_buf[tid - iter];
             // Im 1.Schritt: 1=1+0, 2=2+1 ...
             // Im 2.Schritt: 2=(1+2)+(0), 3=(2+3)+(0+1), 4=(4+3)+(2+1) ...
             // Im 3.Schritt  4=((4+3)+(2+1))+(0), 8 = ((8+7)+(6+5))+((4+3)+(2+1))
             // jeweils an den 2^k Stellen bilden sich entsprechende aussummierungen
             // Laufzeit -> log2(n) (log2(1024)=10, -> 10 Schritte            
         }
         
         iter *= 2;
     }
         
    __syncthreads();
    out[tid] = sum_buf[tid];
    __syncthreads();
        
}
""")
naive_gpu = naive_ker.get_function("naive_prefix")

if __name__ == '__main__':
    testvec = np.random.randn(1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)

    outvec_gpu = gpuarray.empty_like(testvec_gpu)

    naive_gpu(testvec_gpu, outvec_gpu, block=(1024, 1, 1), grid=(1, 1, 1))

    total_sum = sum(testvec)
    total_sum_gpu = outvec_gpu[-1].get()

    print "Does our kernel work correctly? : {}".format(np.allclose(total_sum_gpu, total_sum))

# By its nature, the parallel prefix algorithm has to run over n threads, corresponding to a size-n array, where n is dyadic
# however, we can extend this algorithm to an arbitrary non-dyadic size assuming that our operator has an identity element
# that is to say, that there is some value e so that for any x value

"""Inclusive versus exclusive prefix"""

# Prefix algorithms that produce output as we had just now are called inclusive
# in the case of an inclusive prefix algorithm,
# the corresponding element at each index is included in the summation in the same index of the output array.

# An exclusive prefix algorithm differs in that it similarly takes n input values
# and produces the length-n output array: e, x0, x0+x1,...,x0+...+xn-2 (instead of x0, x0+x1,...,x0+...+xn-1)
# Note that the exclusive algorithm yields nearly the same output as the inclusive algorithm,
# only it is right-shifted and omits the final value. We can therefore trivially obtain the equivalent output
# from either algorithm, provided we keep a copy of x0,....,xn-1

"""A work-efficient parallel prefix algorithm"""

#  In an ideal case, the computational time complexity is O(log n), but this is only when we have
#  a sufficient number of processors for our data set; when the cardinality (number of elements) of our
#  dataset, n, is much larger than the number of processors, we have that this becomes an O(n log n) time algorithm.
#  <<k left shift, multiplying by 2^k, >>k right shift, dividing by 2^k,
#  at the edges of the bits, these shift will have different results

# the work performed by a parallel algorithm here is the number of invocations of this operator across all threads
# or the duration of the execution
# span is the number of invocations a thread makes in the duration of execution of the kernel;
# span of the whole algorithm is the same as the longest span among each individual thread which will tell us
# the total execution time.

# We seek to specifically reduce the amount of work performed by the algorithm across all threads
# In the case of the naive prefix, the additional work that is required costs a more time when the
# number of available processors falls short
# a new algorithm that is work efficient, and hence more suitable for a limited number of processors.
# This consists of two separate two distinct parts, the up-sweep (or reduce) phase and the down-sweep phase.
# We should also note the algorithm we'll see is an exclusive prefix algorithm.

# this algorithm can operate on arrays of arbitrarily large size over 1,024.
# This will mean that this will operate over grids as well as blocks; we'll have to use the host for synchronization

# The up-sweep phase is similar to a single reduce operation to produce the value that is given by the reduce algorithm
# x0+x1,x2+x3, ...
# kernel for up-sweep phase
# we'll be iteratively re-launching this kernel from the host, we'll need a parameter that indicates current iteration k
# We'll use two arrays for the computation to avoid race conditions: x (for the current iteration) and x_old
# tid, da nur eine Dimension, recht trivial
# int _2k use C bit-wise shift operators to generate 2k and 2k+1 directly from k//*2^k (also bei k = 3, -> 1*8)
up_ker = SourceModule("""
__global__ void up_ker(double *x, double *x_old, int k )
{

     int tid =  blockIdx.x*blockDim.x + threadIdx.x;


     int _2k = 1 << k;

     int _2k1 = 1 << (k+1);

     int j = tid* _2k1;

     x[j + _2k1 - 1] = x_old[j + _2k -1 ] + x_old[j + _2k1 - 1];

}
""")

up_gpu = up_ker.get_function("up_ker")


# implementation of up-sweep phase
def up_sweep(x):
    # let's typecast to be safe.
    x = np.float64(x)
    x_gpu = gpuarray.to_gpu(np.float64(x))
    x_old_gpu = x_gpu.copy()
    for k in range(int(np.log2(x.size))):
        num_threads = int(np.ceil(x.size / 2 ** (k + 1)))
        # returns the ceil of the elements of array. The ceil of the scalar x is the is the smallest integer i
        grid_size = int(np.ceil(num_threads / 32))

        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads
        # Also note how we set our block and grid sizes depending on how many threads we have to launch,
        # we try to keep our block sizes as multiples of size 32, up to 1024 would be possible here (no 2D)
        up_gpu(x_gpu, x_old_gpu, np.int32(k), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
        # updating x_old_gpu by copying from x_gpu using [:], which will preserve the memory allocation and merely copy the new data over rather than re-allocate.
        x_old_gpu[:] = x_gpu[:]

    x_out = x_gpu.get()
    return (x_out)

# The down-sweep phase will then operate on these partial sums and give us the final result
# kernel for down-sweep phase
down_ker = SourceModule("""
__global__ void down_ker(double *y, double *y_old,  int k)
{
     int tid =  blockIdx.x*blockDim.x + threadIdx.x;

     int _2k = 1 << k;
     int _2k1 = 1 << (k+1);

     int j = tid*_2k1;

     y[j + _2k - 1 ] = y_old[j + _2k1 - 1];
     y[j + _2k1 - 1] = y_old[j + _2k1 - 1] + y_old[j + _2k - 1];
}
""")

down_gpu = down_ker.get_function("down_ker")


# implementation of down-sweep phase
def down_sweep(y):
    y = np.float64(y)
    y[-1] = 0
    y_gpu = gpuarray.to_gpu(y)
    y_old_gpu = y_gpu.copy()
    # important distinction: we have to iterate from the largest value in the outer for loop to the smallest
    for k in reversed(range(int(np.log2(y.size)))):
        num_threads = int(np.ceil(y.size / 2 ** (k + 1)))
        grid_size = int(np.ceil(num_threads / 32))

        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads

        down_gpu(y_gpu, y_old_gpu, np.int32(k), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
        y_old_gpu[:] = y_gpu[:]
    y_out = y_gpu.get()
    return (y_out)


# full implementation of work-efficient parallel prefix sum
def efficient_prefix(x):
    return (down_sweep(up_sweep(x)))


if __name__ == '__main__':
    testvec = np.random.randn(32 * 1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)

    outvec_gpu = gpuarray.empty_like(testvec_gpu)

    prefix_sum = np.roll(np.cumsum(testvec), 1)
    prefix_sum[0] = 0

    prefix_sum_gpu = efficient_prefix(testvec)

    print "Does our work-efficient prefix work? {}".format(np.allclose(prefix_sum_gpu, prefix_sum))
