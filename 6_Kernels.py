import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

# SourceModule actually compiles code into a CUDA module
# We'll always just declare this as a void function
# because we get our output values by passing a pointer to some empty chunk of memory that we pass in as a parameter
# If we wish to pass simple singleton input values to our kernel (scalar), we can always do so without using pointers
ker = SourceModule("""
__global__ void scalar_multiply_kernel(float *outvec, float scalar, float *vec)
{
     int i = threadIdx.x + blockDim.x*blockIdx.x;
     outvec[i] = scalar*vec[i];
}
""")
# ElementwiseKernel automatically parallelized over multiple GPU threads by a value, i, which was set for us by PyCUDA;
# the identification of each individual thread is given by the threadIdx value,
# which we retrieve as follows: int i = threadIdx.x;


# This means we'll have to "pull out" a reference to the kernel we want to use with PyCUDA's get_function
scalar_multiply_gpu = ker.get_function("scalar_multiply_kernel")

testvec = np.random.randn(10000).astype(np.float32)
testvec_gpu = gpuarray.to_gpu(testvec)
outvec_gpu = gpuarray.empty_like(testvec_gpu)

# since the scalar is a singleton, we don't have to copy this value to the GPU,
# we should be careful that we typecast it properly
# we'll have to specifically set the number of threads to 512 with the block and grid parameters.
scalar_multiply_gpu(outvec_gpu, np.float32(2), testvec_gpu, block=(100, 1, 1), grid=(100, 1, 1))

# print(testvec)
# print(outvec_gpu.get())
print "Does our kernel work correctly? : {}".format(np.allclose(outvec_gpu.get(), 2 * testvec))
