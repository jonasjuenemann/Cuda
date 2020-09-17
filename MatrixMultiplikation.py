from __future__ import print_function
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import cProfile


# OwnTime bei Cprofile -> Zeit die die Funktion verbraucht hat, abgezogen die unterfunktionen die sie aufgerufen hat.

# SourceModule compiles C code for CUDA
def matrix_multi_GPU():
    mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x + threadIdx.y*4; //Verhalten von threadIdx.y*4, ist groesse der 2D der Matrix
  dest[i] = a[i] * b[i];
}
""")

    multiply_them = mod.get_function("multiply_them")

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).astype(np.float32)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).astype(np.float32)
    # print(a)
    # print(b)
    dest = np.zeros_like(a)
    # print(dest)
    multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(4, 4, 1), grid=(1, 1))
    return dest


# pr = cProfile.Profile()
# pr.enable()
print(matrix_multi_GPU())
# pr.disable()
# pr.print_stats(sort="calls")

"""Klassische Matrix Multiplikation:"""
x = np.array([[1, 2, 3], [7, 8, 9]])
y = np.array([[1, 2], [3, 4], [5, 6]])
# print(y)
"""[[1. 2.]
    [3. 4.]
    [5. 6.]]"""
print(x[0][0])
x_0 = [1, 2, 3]


# y = [[1, 2], [3, 4], [5, 6]]


def inner_mult(list, matrix):
    new_list = []
    for z in range(len(matrix[0])):
        x = 0
        for i in range(len(list)):
            x += list[i] * matrix[i][z]
        new_list.append(x)
    return new_list


# print(inner_mult(x_0, y))


def matrix_mult(a_matrix, b_matrix):
    if len(a_matrix[0]) != len(b_matrix):
        raise BaseException("Matrizen koennen von den Dimensionen her nicht multipliziert werden")
    new_matrix = []
    for i in range(len(a_matrix)):
        new_matrix.append(inner_mult(a_matrix[i], b_matrix))
    return new_matrix


print(matrix_mult(x, y))
# x = np.array([[1, 2, 3, 4], [7, 8, 9]])
# print(matrix_mult(x, y))
