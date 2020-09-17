# coding=utf-8
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.scan import InclusiveScanKernel

# review of two functions available in Python for functional programming — map and reduce
# both considered to be functional because they both act on functions for their operation
pow2 = lambda x: x ** 2
print(pow2(2))
# map acts on two input values: a function and a list of objects that the given function can act on
print(map(lambda x: x ** 2, [2, 3, 4]))
# map acts as ElementwiseKernel
print(reduce(lambda x, y: x + y, [1, 2, 3, 4]))
# final returned result (10) -> aus 6 + 4 am Ende wird returned

# PyCUDA handles programming patterns akin to reduce — with parallel scan and reduction kernels.

seq = np.array([1, 2, 3, 4], dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
sum_gpu = InclusiveScanKernel(np.int32, "a+b")
# analogous to lambda a,b: a + b
print sum_gpu(seq_gpu).get()
print np.cumsum(seq)

seq = np.array([1, 100, -3, -10000, 4, 10000, 66, 14, 21], dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
max_gpu = InclusiveScanKernel(np.int32, "a > b ? a : b")
print max_gpu(seq_gpu).get()
print max_gpu(seq_gpu).get()[-1]
# print np.max(seq) auch 10000

# Skalarprodukt einer Matrix in paralleler Ausfueugrung auf der gpu,
# kann erstmal nur einfache Vektoren, keine 2D-Matrizen

dot_prod = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b", map_expr="vec1[i]*vec2[i]",
                           arguments="float *vec1, float *vec2")

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([6, 7, 8]).astype(np.float32)
device_x = gpuarray.to_gpu(x)
device_y = gpuarray.to_gpu(y)
product = dot_prod(device_x, device_y)
print(product.get())
