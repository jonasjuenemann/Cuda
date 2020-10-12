import numpy as np
from collections import Counter

x = np.array([[1, 2, 3], [7, 8, 9]])
y = np.array([[1, 2], [3, 4], [5, 6]])
print(y)
"""[[1. 2.]
    [3. 4.]
    [5. 6.]]"""
print(x[0][0])
x_0 = [1, 2, 3]
y = [[1, 2], [3, 4], [5, 6]]


def inner_mult(list, matrix):
    new_list = []
    for z in range(len(matrix[0])):
        x = 0
        for i in range(len(list)):
            x += list[i] * matrix[i][z]
        new_list.append(x)
    return new_list


print(inner_mult(x_0, y))


def matrix_mult(a_matrix, b_matrix):
    if len(a_matrix[0]) != len(b_matrix):
        raise BaseException("Matrizen koennen von den Dimensionen her nicht multipliziert werden")
    new_matrix = []
    for i in range(len(a_matrix)):
        new_matrix.append(inner_mult(a_matrix[i], b_matrix))
    return new_matrix


print(matrix_mult(x, y))
# x = np.array([[1, 2, 3, 4], [7, 8, 9]])
#print(matrix_mult(x, y))

#print(np.random.randn(1024))
# matmul von numpy bildet matrizenprodukte
print(np.matmul(x, y))

#print(Counter(Counter([7,8,6,4,4,3,54,6]))) aendert nix