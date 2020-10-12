# coding=utf-8
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas
from time import time

a = np.float32(10)
x = np.float32([1,2,3])
y = np.float32([-.345,8.15,-15.867])
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

# create a cuBLAS context. This is similar in nature to CUDA contexts
cublas_context_h = cublas.cublasCreate()

"""Level-1 AXPY (vector-vector)"""

# this is a direct wrapper to a low-level C function, so the input may seem more like a C function than a true Python function.
# In short, this performed an "AXPY" operation, ultimately putting the output data into the y_gpu array
# first input is always the CUDA context handle. We then have to specify the size of the vectors, since this function will be ultimately
# operating on C pointers; we can do this by using the size parameter of a gpuarray. Having typecasted our scalar already to a NumPy float32 variable,
# we can pass the a variable right over as the scalar parameter. We then hand the underlying C pointer of the x_gpu array to this function using the gpudata
# parameter. Then we specify the stride of the first array as 1: the stride specifies how many steps we should take between each input value.
# (In contrast, if you were using a vector from a column in a row-wise matrix, you would set the stride to the width of the matrix.)
# We then put in the pointer to the y_gpu array, and set its stride to 1 as well

#We can now use the cublasSaxpy function. The S stands for single precision, which is what we will need since we are working with 32-bit floating point arrays:
cublas.cublasSaxpy(cublas_context_h, x_gpu.size, a, x_gpu.gpudata, 1, y_gpu.gpudata, 1)

print(y_gpu.get())
print 'This is close to the NumPy approximation: %s' % np.allclose(a*x + y , y_gpu.get())

w_gpu = gpuarray.to_gpu(x)
v_gpu = gpuarray.to_gpu(y)

#perform a dot product
dot_output = cublas.cublasSdot(cublas_context_h, v_gpu.size, v_gpu.gpudata, 1, w_gpu.gpudata, 1)

print(dot_output)

l2_output = cublas.cublasSnrm2(cublas_context_h, v_gpu.size, v_gpu.gpudata, 1)

print(l2_output)

cublas.cublasDestroy(cublas_context_h)

#(f we want to operate on arrays of 64-bit real floating point values, (float64 in NumPy and PyCUDA), then we would use the cublasDaxpy)

"""Level-2 GEMV (general matrix-vector)"""

# m and n are the number of rows and columns
m = 10
n = 100
# is the floating-point value for α
alpha = 1
# s the floating-point value for β
beta = 0
# set alpha to 1 and beta to 0 to get a direct matrix multiplication with no scaling
A = np.random.rand(m,n).astype('float32')
x = np.random.rand(n).astype('float32')
y = np.zeros(m).astype('float32')


# We will now have to get A into column-major (or column-wise) format
# transposed matrix
A_columnwise = A.T.copy()
A_gpu = gpuarray.to_gpu(A_columnwise)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

# refers to the structure of the matrix,we can specify whether we want to use the original matrix, a direct transpose, or a conjugate transpose (for complex matrices)
# Since we now have the column-wise matrix stored properly on the GPU, we can set the trans variable to not take the transpose by using the _CUBLAS_OP dictionary
trans = cublas._CUBLAS_OP['N']

# indicates the leading dimension of the matrix, where the total size of the matrix is actually lda x n (if lda > m -> Problems)
lda = m

# x and its stride, incx; x is the underlying C pointer of the vector being multiplied by A. Remember, x will have to be of size n;
incx = 1
# y and its stride incy as the last parameters. We should remember that y should be of size m, or the number of rows
incy = 1
handle = cublas.cublasCreate() # refers to the cuBLAS context.

#

cublas.cublasSgemv(handle, trans, m, n, alpha, A_gpu.gpudata, lda, x_gpu.gpudata, incx, beta, y_gpu.gpudata, incy)

cublas.cublasDestroy(handle)
print 'cuBLAS returned the correct value: %s' % np.allclose(np.dot(A,x), y_gpu.get())


"""Level-3 GEMM (general matrix-matrix)"""

# performance metric for our GPU to determine the number of Floating Point Operations Per Second (FLOPS) it can perform,
# which will be two separate values: the case of single precision, and that of double precision

# m, n, and k variables for our matrix sizes
m = 5000
n = 10000
k = 10000


def compute_gflops(precision='S'):
    if precision == 'S':
        float_type = 'float32'
    elif precision == 'D':
        float_type = 'float64'
    else:
        return -1

    # some random matrices that are of the appropriate precision that we will use for timing
    A = np.random.randn(m, k).astype(float_type)
    B = np.random.randn(k, n).astype(float_type)
    C = np.random.randn(m, n).astype(float_type)

    # wie gehabt
    A_cm = A.T.copy()
    B_cm = B.T.copy()
    C_cm = C.T.copy()
    A_gpu = gpuarray.to_gpu(A_cm)
    B_gpu = gpuarray.to_gpu(B_cm)
    C_gpu = gpuarray.to_gpu(C_cm)
    alpha = np.random.randn()
    beta = np.random.randn()
    transa = cublas._CUBLAS_OP['N']
    transb = cublas._CUBLAS_OP['N']
    lda = m
    ldb = k
    ldc = m

    t = time()
    handle = cublas.cublasCreate()

    # two different (relevant) precision modes, D and S
    exec ('cublas.cublas%sgemm(handle, transa, transb, m, n, k, alpha, A_gpu.gpudata, lda, \
						B_gpu.gpudata, ldb, beta, C_gpu.gpudata, ldc)' % precision)

    cublas.cublasDestroy(handle)
    t = time() - t

    # a total of 2kmn - mn + 3mn = 2kmn + 2mn = 2mn(k+1) floating point operations in a given GEMM operation
    gflops = 2 * m * n * (k + 1) * (10 ** -9) / t

    return gflops

print 'Single-precision performance: %s GFLOPS' % compute_gflops('S')
print 'Double-precision performance: %s GFLOPS' % compute_gflops('D')